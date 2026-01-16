"""
CVRP label generator using VROOM solver.
Solves CVRP instances and appends routing costs, number of routes and route details
to instance files. Supports both scaled and unscaled appraoches.
"""

import math
import os
import argparse
import time
import vroom
import threading


class TimeoutError(Exception):
    pass


def timeout_wrapper(func, timeout, *args, **kwargs):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Function {func.__name__} exceeded {timeout} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


def solve_cvrp_vroom(depot_coord, cust_coords, cust_demands, vehicle_cap,
                     scale_factor=1, fixed_vehicle_cost=0,
                     exploration_level=5, nb_threads=4, timeout_sec=5):
    n = len(cust_coords)
    points = [depot_coord] + cust_coords
    dist_matrix = [
        [0 if i == j else int(scale_factor * math.hypot(
            points[i][0] - points[j][0],
            points[i][1] - points[j][1]))
         for j in range(n + 1)]
        for i in range(n + 1)
    ]

    problem = vroom.Input()
    problem.set_durations_matrix(profile="car", matrix_input=dist_matrix)

    for i in range(n):
        vc = vroom.VehicleCosts(fixed=fixed_vehicle_cost, per_hour=3600)
        veh = vroom.Vehicle(id=i + 1, start=0, end=0,
                            capacity=vroom.Amount([vehicle_cap]),
                            profile="car", costs=vc)
        problem.add_vehicle(veh)

    for i in range(1, n + 1):
        job = vroom.Job(id=1000 + i, location=i, delivery=[cust_demands[i - 1]])
        problem.add_job(job)

    try:
        sol = timeout_wrapper(problem.solve, timeout_sec,
                              exploration_level=exploration_level, nb_threads=nb_threads)
    except TimeoutError:
        print(f"VROOM exceeded {timeout_sec}s, skipping solution.")
        return float("inf"), 0, timeout_sec, []

    used_vehicles = sol.routes["vehicle_id"].nunique()

    routes_info = []
    for vid, group in sol.routes.groupby("vehicle_id"):
        route_jobs = (
            group.loc[group["type"] == "job", "id"]
            .dropna().astype(int).tolist()
        )
        route_jobs = [jid - 1000 for jid in route_jobs]

        routes_info.append({
            "vehicle_id": int(vid),
            "route": route_jobs,
            "route_load": None,
            "route_distance": group["arrival"].iloc[-1]
        })

    return sol.summary.cost, used_vehicles, 0.0, routes_info


def write_solution_to_file(instance_path, instance_file, mode,
                           vroom_cost, vroom_routes, vroom_time, vroom_routes_info):
    suffix = "_scaled" if mode == "scaled" else f"_{mode}"
    
    with open(instance_path, "a") as file:
        file.write(f"\n#cost_vroom{suffix} {vroom_cost}\n")
        file.write(f"#num_routes_vroom{suffix} {vroom_routes}\n")
        file.write(f"#solve_time_vroom{suffix} {vroom_time:.2f}s\n")
        file.write(f"#actual_routes_vroom{suffix} {vroom_routes_info}\n")
        file.write("#EOF\n")


class DataCvrp:
    def __init__(self, vehicle_capacity, nb_customers, cust_demands, cust_coordinates, depot_coordinates):
        self.vehicle_capacity = vehicle_capacity
        self.nb_customers = nb_customers
        self.cust_demands = cust_demands
        self.cust_coordinates = cust_coordinates
        self.depot_coordinates = depot_coordinates


def read_instance(name: str):
    with open(os.path.normpath(name), "r", encoding="UTF-8") as file:
        return [str(e) for e in file.read().split()]


def read_cvrp_instances(instance_full_path):
    instance_iter = iter(read_instance(instance_full_path))
    dimension_input, capacity_input = -1, -1

    while True:
        element = next(instance_iter)
        if element == "DIMENSION":
            next(instance_iter)
            dimension_input = int(next(instance_iter))
        elif element == "CAPACITY":
            next(instance_iter)
            capacity_input = int(next(instance_iter))
        elif element == "EDGE_WEIGHT_TYPE":
            next(instance_iter)
            if next(instance_iter) != "EUC_2D":
                raise Exception("EDGE_WEIGHT_TYPE not supported")
        elif element == "NODE_COORD_SECTION":
            break

    vehicle_capacity = capacity_input
    cust_coords, depot_coord = [], []

    for current_id in range(dimension_input):
        point_id = int(next(instance_iter))
        x_coord = float(next(instance_iter))
        y_coord = float(next(instance_iter))
        if current_id == 0:
            depot_coord = [x_coord, y_coord]
        else:
            cust_coords.append([x_coord, y_coord])

    if next(instance_iter) != "DEMAND_SECTION":
        raise Exception("Expected DEMAND_SECTION")

    cust_demands = []
    for current_id in range(dimension_input):
        point_id = int(next(instance_iter))
        demand = int(next(instance_iter))
        if current_id > 0:
            cust_demands.append(demand)

    if next(instance_iter) != "DEPOT_SECTION":
        raise Exception("Expected DEPOT_SECTION")
    next(instance_iter)
    if int(next(instance_iter)) != -1:
        raise Exception("Expected only one depot.")

    return DataCvrp(vehicle_capacity, dimension_input - 1, cust_demands, cust_coords, depot_coord)


def solve_file(instance_path, mode, scale_factor, fixed_cost):
    instance_file = os.path.basename(instance_path)

    with open(instance_path, "r") as file:
        content = file.read()
        if "#EOF" in content:
            print(f"Skipping {instance_file}, already solved.")
            return

    cvrp_data = read_cvrp_instances(instance_path)

    print(f"Solving {instance_file} with VROOM (mode={mode})...")
    start = time.time()
    vroom_cost, vroom_routes, _, vroom_routes_info = solve_cvrp_vroom(
        cvrp_data.depot_coordinates,
        cvrp_data.cust_coordinates,
        cvrp_data.cust_demands,
        cvrp_data.vehicle_capacity,
        scale_factor=scale_factor,
        fixed_vehicle_cost=fixed_cost
    )
    vroom_time = time.time() - start

    write_solution_to_file(instance_path, instance_file, mode,
                           vroom_cost, vroom_routes, vroom_time, vroom_routes_info)

    print(f"Results appended to {instance_file}")


def solve_files_in_directory(input_path, mode, scale_factor, fixed_cost):
    if os.path.isfile(input_path):
        solve_file(input_path, mode, scale_factor, fixed_cost)
    else:
        for instance_file in sorted(os.listdir(input_path)):
            full_path = os.path.join(input_path, instance_file)
            if os.path.isfile(full_path):
                solve_file(full_path, mode, scale_factor, fixed_cost)


def parse_arguments():
    parser = argparse.ArgumentParser(description="CVRP Label Generator with VROOM")
    parser.add_argument("input_path", help="Path to input directory or single file")
    parser.add_argument("--mode", choices=["scaled", "unscaled"], default="scaled",
                        help="scaled (Prodhon, *100) or unscaled (Barreto/Tuzun)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode == "scaled":
        scale_factor = 100
        fixed_cost = 1000
    else:  # unscaled
        scale_factor = 1
        fixed_cost = 0

    solve_files_in_directory(args.input_path, args.mode, scale_factor, fixed_cost)
