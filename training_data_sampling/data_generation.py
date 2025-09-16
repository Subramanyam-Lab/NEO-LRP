"""
Training Data Generation for Neural Embedded Optimization

This module provides functionality to generate training data for neural networks
used in Location-Routing Problems (LRP) optimization.
"""

import os
import argparse
import shutil
import random
import h5py
import numpy as np
from scipy.spatial import distance_matrix
import math
import logging
from datetime import datetime

###############################################################################
# Utility Functions: Distance Calculation, Shift & Scale
###############################################################################

def euclid_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1 (tuple): First point coordinates (x, y)
        p2 (tuple): Second point coordinates (x, y)
        
    Returns:
        float: Euclidean distance between the points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def shift_and_scale_instance(instance):
    """
    Shift depot to (0,0), scale by fi=max(range_x, range_y).
    Then recalculate demands, cost, and distance matrix.
    
    Args:
        instance (dict): Instance data with keys:
            - x_coordinates: List of x coordinates
            - y_coordinates: List of y coordinates  
            - demands: List of customer demands
            - vehicle_capacity: Vehicle capacity
            - cost: Cost matrix
            - is_depot: Boolean list indicating depot locations (optional)
            - mask: Boolean mask for valid locations (optional)
            
    Returns:
        dict: Normalized instance data
    """
    x_coords = np.array(instance['x_coordinates'], dtype=float)
    y_coords = np.array(instance['y_coordinates'], dtype=float)

    # Index 0 is depot
    depot_x = x_coords[0]
    depot_y = y_coords[0]

    # shift
    x_shifted = x_coords - depot_x
    y_shifted = y_coords - depot_y

    # fi = max(range_x, range_y)
    max_x, min_x = np.max(x_shifted), np.min(x_shifted)
    max_y, min_y = np.max(y_shifted), np.min(y_shifted)
    range_x = max_x - min_x
    range_y = max_y - min_y
    fi = max(range_x, range_y)
    if fi < 1e-9:
        fi = 1.0

    # scale
    x_normed = x_shifted / fi
    y_normed = y_shifted / fi

    # demands / capacity
    demands = np.array(instance['demands'], dtype=float)
    cap     = float(instance['vehicle_capacity'])
    demands_normed = demands / cap if cap > 1e-9 else demands
    # cap_normed     = 1.0

    # cost scaled
    old_cost = instance.get('cost', 0.0)
    cost_scaled = old_cost / (1000.0 * fi)

    # dist_matrix
    coords_2d = np.column_stack((x_normed, y_normed))
    dist_mtx = distance_matrix(coords_2d, coords_2d).astype(np.float32)

    # store back
    instance['x_coordinates']    = x_normed
    instance['y_coordinates']    = y_normed
    instance['demands']          = demands_normed
    instance['fi']            = fi
    # instance['vehicle_capacity'] = cap_normed
    instance['used_vehicles'] = instance['used_vehicles']
    instance['cost']             = cost_scaled
    instance['dist_matrix']      = dist_mtx

    return instance


###############################################################################
# (1) Function to solve CVRP using VROOM to get cost and route count
###############################################################################
import math
import time
import vroom  # pip install pyvroom (or GitHub version)

def solve_cvrp_with_vroom_fixed_cost(
    depot_coord,
    cust_coords,
    cust_demands,
    vehicle_cap,
    fixed_vehicle_cost=1000,
    rc_cal_index=0,
    exploration_level=5,
    nb_threads=4
):
    """
    Function to solve CVRP using VROOM that minimizes
    'distance multiplied by 100 + fixed cost per vehicle (fixed_vehicle_cost)'.
    """
    start_t = time.time()
    
    n_cust = len(cust_coords)
    all_points = [depot_coord] + cust_coords
    matrix_size = n_cust + 1

    # (1) Build distance matrix
    dist_matrix = []
    for i in range(matrix_size):
        row = []
        for j in range(matrix_size):
            if i == j:
                row.append(0)
            else:
                dx = all_points[i][0] - all_points[j][0]
                dy = all_points[i][1] - all_points[j][1]
                dist_ij = math.hypot(dx, dy)

                if rc_cal_index == 0:
                    # If rc_cal_index=0, use Euclidean distance * 100
                    row.append(int(100 * dist_ij))
                else:
                    row.append(int(dist_ij))
        dist_matrix.append(row)

    # (2) Create VROOM Input and set durations_matrix
    problem = vroom.Input()
    problem.set_durations_matrix(
        profile="car",
        matrix_input=dist_matrix
    )

    # (3) Define Vehicle (including fixed cost per vehicle)
    #     When creating VehicleCosts in Python binding,
    #     pass fixed, per_hour, per_km etc. as constructor arguments
    for v_id in range(n_cust):
        vc = vroom.VehicleCosts(
            fixed=fixed_vehicle_cost,  # Fixed cost per vehicle
            per_hour=3600,             # Default 3600
            # per_km=0                   # Default 0
        )

        veh = vroom.Vehicle(
            id=v_id + 1,
            start=0,
            end=0,
            capacity=vroom.Amount([vehicle_cap]),
            profile="car",
            costs=vc
        )
        problem.add_vehicle(veh)

    # (4) Add customer Jobs
    for c_idx in range(1, matrix_size):
        job = vroom.Job(
            id=1000 + c_idx,
            location=c_idx,
            delivery=[cust_demands[c_idx - 1]]
        )
        problem.add_job(job)

    # (5) Solve
    solution = problem.solve(
        exploration_level=exploration_level,
        nb_threads=nb_threads
    )
    end_t = time.time()
    solve_time = end_t - start_t

    # (6) Parse results
    total_cost = solution.summary.cost
    used_vehicles = solution.routes["vehicle_id"].nunique()
    # solution.summary.cost internally includes
    #   (distance_cost + fixed_vehicle_cost × number_of_used_vehicles)

    return total_cost, used_vehicles, solve_time

###############################################################################
# (2) Instance file writing utilities
###############################################################################
def pick_depot_caps(C, D, vehicle_cap=70):
    """Randomly generate capacity per depot (C: number of customers, D: number of depots, vehicle_cap: capacity of one vehicle)"""
    if C == 20:
        factor_low, factor_high = 1, 3
    elif C == 50:
        factor_low, factor_high = 4, 6
    elif C == 100:
        if D == 5:
            factor_low, factor_high = 9, 11
        else:
            factor_low, factor_high = 6, 8
    else:  # C == 200
        factor_low, factor_high = 13, 18

    caps = []
    for _ in range(D):
        factor = random.randint(factor_low, factor_high)
        caps.append(factor * vehicle_cap)
    return caps

def write_instance_file(filepath, C, D,
                        depot_coords, cust_coords,
                        vehicle_cap, depot_caps,
                        cust_demands, open_costs,
                        route_cost, rc_cal_index):
    """
    Save parameters like (C,D), coordinates, demands, etc. to file
    """
    with open(filepath, 'w') as f:
        f.write(f"{C}\n")
        f.write(f"{D}\n\n")

        for (dx, dy) in depot_coords:
            f.write(f"{dx}\t{dy}\n")
        f.write("\n")

        for (cx, cy) in cust_coords:
            f.write(f"{cx}\t{cy}\n")
        f.write("\n")

        f.write(f"{vehicle_cap}\n\n")
        for cap in depot_caps:
            f.write(f"{cap}\n")
        f.write("\n")
        for dem in cust_demands:
            f.write(f"{dem}\n")
        f.write("\n")
        for oc in open_costs:
            f.write(f"{oc}\n")
        f.write("\n")
        f.write(f"{route_cost}\n\n")
        f.write(f"{rc_cal_index}\n\n")

def generate_instance(C, D, instance_index, out_dir):
    """
    Generate random depots, customers, demands, etc. based on (C,D) and save to files
    - Create two versions with vehicle capacities 70 and 150
    """
    depot_coords = [(random.randint(1, 50), random.randint(1, 50)) for _ in range(D)]
    cust_coords  = [(random.randint(1, 50), random.randint(1, 50)) for _ in range(C)]
    cust_demands = [random.randint(11, 20) for _ in range(C)]

    if C in [20, 50]:
        open_costs = [random.randint(5000, 15000) for _ in range(D)]
    elif C == 100:
        open_costs = [random.randint(40000, 60000) for _ in range(D)]
    else:
        open_costs = [random.randint(70000, 130000) for _ in range(D)]

    route_cost   = 1000
    rc_cal_index = 0

    # Generate depot capacity (based on vehicle capacity 70)
    depot_caps_70 = pick_depot_caps(C, D, 70)

    # 70 capacity version
    fname_70 = f"set{C}-{D}-{instance_index}_vc70.txt"
    path_70 = os.path.join(out_dir, fname_70)
    write_instance_file(path_70, C, D,
                        depot_coords, cust_coords,
                        70, depot_caps_70,
                        cust_demands, open_costs,
                        route_cost, rc_cal_index)

    # 150 capacity version
    fname_150 = f"set{C}-{D}-{instance_index}_vc150.txt"
    path_150 = os.path.join(out_dir, fname_150)
    write_instance_file(path_150, C, D,
                        depot_coords, cust_coords,
                        150, depot_caps_70,  # Keep depot_caps as is (example)
                        cust_demands, open_costs,
                        route_cost, rc_cal_index)

    return path_70, path_150

###############################################################################
# (3) Parse instance text -> return in (C, D, ...) format
###############################################################################
def create_data(txt_file):
    """
    Open txt_file and parse in order:
     C, D
     depot coords
     cust coords
     vehicle_cap
     depot_caps
     cust_demands
     open_costs
     route_cost
     rc_cal_index
    Return as tuple
    """
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    idx = 0
    C = int(lines[idx]); idx+=1
    D = int(lines[idx]); idx+=1

    depot_coords = []
    for _ in range(D):
        x_str, y_str = lines[idx].split()
        idx += 1
        depot_coords.append((float(x_str), float(y_str)))

    cust_coords = []
    for _ in range(C):
        x_str, y_str = lines[idx].split()
        idx += 1
        cust_coords.append((float(x_str), float(y_str)))

    vehicle_cap = int(lines[idx]); idx+=1

    depot_caps = []
    for _ in range(D):
        depot_caps.append(int(lines[idx]))
        idx += 1

    cust_demands = []
    for _ in range(C):
        cust_demands.append(int(lines[idx]))
        idx += 1

    open_costs = []
    for _ in range(D):
        open_costs.append(int(lines[idx]))
        idx += 1

    route_cost = int(lines[idx]); idx+=1
    rc_cal_index = int(lines[idx]); idx+=1

    return (C, D, depot_coords, cust_coords,
            vehicle_cap, depot_caps, cust_demands,
            open_costs, route_cost, rc_cal_index)

###############################################################################
# (4) For each depot, select random k(1..C) customer subset, route with VROOM, and save to HDF5
###############################################################################
def process_random_subset_for_depot_and_save(
    h5_filename,
    solution_id,
    depot_id,
    subset_customers,
    depot_coords,
    customer_coords,
    customer_demands,
    vehicle_cap,
    rc_cal_index=0
):
    """
    depot_id: Which depot to assign to
    subset_customers: List of customer indices assigned to depot (e.g. [3,7,9])
    """
    if not subset_customers:
        return  # Skip if no customers

    # Depot coordinates
    depot_coord = depot_coords[depot_id]
    # Subset customer coordinates/demands
    assigned_cust_coords = [customer_coords[c_i] for c_i in subset_customers]
    assigned_cust_dem    = [customer_demands[c_i] for c_i in subset_customers]

    # ====== (A) Calculate CVRP cost & routing using VROOM ======
    cost_vrp, route_count, solve_t = solve_cvrp_with_vroom_fixed_cost(
        depot_coord,
        assigned_cust_coords,
        assigned_cust_dem,
        vehicle_cap,
        exploration_level=5,
        nb_threads=4
    )
    # cost_vrp = (sum of distances) + 1000×(number_of_used_vehicles),
    #            if rc_cal_index=0, multiply distance by *100
    # ============================================================

    # Use cost_vrp as training label
    total_cost_ = cost_vrp

    # Prepare data for shift&scale_instance()
    # (mask: depot=0, only subset included items=1)
    C_ = len(customer_coords)
    x_full = [depot_coord[0]] + [customer_coords[i][0] for i in range(C_)]
    y_full = [depot_coord[1]] + [customer_coords[i][1] for i in range(C_)]
    dem_full= [0] + list(customer_demands)

    mask = [0]*(1 + C_)
    mask[0] = 1  # depot
    for c_i in subset_customers:
        mask[c_i+1] = 1

    is_depot = [0]*(1 + C_)
    is_depot[0] = 1  # Index 0 is depot

    instance_data = {
        "x_coordinates": x_full,
        "y_coordinates": y_full,
        "demands":       dem_full,
        "vehicle_capacity": float(vehicle_cap),
        "used_vehicles": route_count,
        "cost": float(total_cost_),
        "mask": mask,
        "is_depot": is_depot
    }

    scaled_data = shift_and_scale_instance(instance_data)

    # Save to HDF5
    group_name = f"instance_{solution_id}_depot_{depot_id}"
    with h5py.File(h5_filename, 'a') as hf:
        if group_name in hf:
            del hf[group_name]
        grp = hf.create_group(group_name)
        for k, v in scaled_data.items():
            if isinstance(v, np.ndarray):
                # float64 -> float32
                if v.dtype == np.float64:
                    v = v.astype(np.float32)
                grp.create_dataset(k, data=v)
            elif isinstance(v, list):
                arr = np.array(v)
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                grp.create_dataset(k, data=arr)
            else:
                grp.create_dataset(k, data=v)

    print(f"[RandomSubset] group='{group_name}', VROOM_cost={total_cost_}, fi={instance_data['fi']}, cost ={total_cost_ / (1000.0 * instance_data['fi'])}, routes={route_count}, solve_time={solve_t:.2f}s")


###############################################################################
# (5) Main: Each (C,D) instance -> load (VC70, VC150) -> random subset of 1..C customers per depot
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_num_data", type=int, default=128000,
                        help="Target total number of HDF5 groups")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    instance_dir = "train_set"
    if os.path.exists(instance_dir):
        shutil.rmtree(instance_dir)
    os.makedirs(instance_dir, exist_ok=True)

    h5_filename = "train_data/train_data_VROOM_128k.h5"
    os.makedirs("train_data", exist_ok=True)
    if os.path.exists(h5_filename):
        os.remove(h5_filename)

    target_num_data = args.target_num_data


    types = [(20,5),(50,5),(100,5),(100,10),(200,10)]
    idx = 0
    instance_count = 0
    total_group_count = 0

    while True:
        # Number of groups in HDF5
        if os.path.exists(h5_filename):
            with h5py.File(h5_filename, 'r') as hf:
                total_group_count = len(hf.keys())

        print(f"[###] Current data count: {total_group_count}")
        if total_group_count >= target_num_data:
            print(f"[Main] Enough data: {total_group_count} >= {target_num_data}")
            break

        (C, D) = types[idx % len(types)]
        idx += 1
        instance_count += 1

        txt70, txt150 = generate_instance(C, D, instance_count, instance_dir)
        print(f"[Info] Generated: {txt70}, {txt150}")

        for txt_file in [txt70, txt150]:
            (C_, D_, depot_coords, cust_coords,
             vehicle_cap, depot_caps, cust_demands,
             open_costs, route_cost, rc_cal_index) = create_data(txt_file)

            for depot_id in range(D_):
                # Iterate k=1..C_ for each depot_id
                for k in range(1, C_+1):
                    ### Select only subsets that satisfy capacity ###
                    attempt_count = 0
                    max_attempts = 10
                    found_feasible = False

                    while attempt_count < max_attempts:
                        attempt_count += 1
                        subset_customers = random.sample(range(C_), k)
                        total_dem = sum(cust_demands[i] for i in subset_customers)

                        # depot_caps[depot_id] is the capacity
                        if total_dem <= depot_caps[depot_id]:
                            # feasible
                            found_feasible = True
                            break
                        # Otherwise repeat

                    if not found_feasible:
                        # Could not find feasible solution after max_attempts tries -> just skip
                        continue

                    solution_id = f"{instance_count}_{os.path.basename(txt_file)}_dep{depot_id}_size{k}"

                    process_random_subset_for_depot_and_save(
                        h5_filename,
                        solution_id,
                        depot_id,
                        subset_customers,
                        depot_coords,
                        cust_coords,
                        cust_demands,
                        vehicle_cap,
                        rc_cal_index
                    )

                    with h5py.File(h5_filename, 'r') as hf:
                        total_group_count = len(hf.keys())
                    if total_group_count >= target_num_data:
                        print(f"[Main] Reached target {target_num_data}. Stop.")
                        return

    print("[Done] All completed.")

if __name__ == "__main__":
    main()