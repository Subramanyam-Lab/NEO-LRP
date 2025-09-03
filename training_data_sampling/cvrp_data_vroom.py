import os, argparse, shutil, random, h5py, time, math
import numpy as np
from scipy.spatial import distance_matrix
import vroom

def euclid_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def shift_and_scale_instance(inst):
    x = np.array(inst['x_coordinates'], dtype=float)
    y = np.array(inst['y_coordinates'], dtype=float)
    depot_x, depot_y = x[0], y[0]
    x, y = x-depot_x, y-depot_y
    fi = max(np.ptp(x), np.ptp(y)) or 1.0
    inst['x_coordinates'] = x / fi
    inst['y_coordinates'] = y / fi
    inst['demands'] = np.array(inst['demands'], dtype=float) / inst['vehicle_capacity']
    inst['cost'] /= 1000.0*fi
    coords_2d = np.column_stack((inst['x_coordinates'], inst['y_coordinates']))
    inst['dist_matrix'] = distance_matrix(coords_2d, coords_2d).astype(np.float32)
    inst['fi'] = fi
    return inst

def solve_cvrp_with_vroom_fixed_cost(depot_coord, cust_coords, cust_demands, vehicle_cap,
                                     fixed_vehicle_cost=1000, rc_cal_index=0, exploration_level=5, nb_threads=4):
    n = len(cust_coords)
    points = [depot_coord]+cust_coords
    dist_matrix = [[0 if i==j else int(100*math.hypot(points[i][0]-points[j][0],
                                                     points[i][1]-points[j][1])) for j in range(n+1)] for i in range(n+1)]
    problem = vroom.Input()
    problem.set_durations_matrix(profile="car", matrix_input=dist_matrix)
    for i in range(n):
        vc = vroom.VehicleCosts(fixed=fixed_vehicle_cost, per_hour=3600)
        veh = vroom.Vehicle(id=i+1, start=0, end=0, capacity=vroom.Amount([vehicle_cap]),
                             profile="car", costs=vc)
        problem.add_vehicle(veh)
    for i in range(1,n+1):
        job = vroom.Job(id=1000+i, location=i, delivery=[cust_demands[i-1]])
        problem.add_job(job)
    sol = problem.solve(exploration_level=exploration_level, nb_threads=nb_threads)
    used_vehicles = sol.routes["vehicle_id"].nunique()
    return sol.summary.cost, used_vehicles, sol.summary.time

def pick_depot_caps(C,D,vehicle_cap=70):
    factors = {20:(1,3),50:(4,6),100:(9,11) if D==5 else (6,8), 200:(13,18)}
    return [random.randint(*factors[C])*vehicle_cap for _ in range(D)]

def write_instance_file(path,C,D,depot_coords,cust_coords,vehicle_cap,depot_caps,cust_demands,open_costs,route_cost,rc_cal_index):
    with open(path,'w') as f:
        f.write(f"{C}\n{D}\n\n")
        for dx,dy in depot_coords: f.write(f"{dx}\t{dy}\n")
        f.write("\n")
        for cx,cy in cust_coords: f.write(f"{cx}\t{cy}\n")
        f.write("\n")
        f.write(f"{vehicle_cap}\n\n")
        for cap in depot_caps: f.write(f"{cap}\n")
        f.write("\n")
        for dem in cust_demands: f.write(f"{dem}\n")
        f.write("\n")
        for oc in open_costs: f.write(f"{oc}\n")
        f.write("\n")
        f.write(f"{route_cost}\n\n{rc_cal_index}\n\n")

def generate_instance(C,D,idx,out_dir):
    depot_coords = [(random.randint(1,50),random.randint(1,50)) for _ in range(D)]
    cust_coords = [(random.randint(1,50),random.randint(1,50)) for _ in range(C)]
    cust_demands = [random.randint(11,20) for _ in range(C)]
    open_costs = [random.randint(5000,15000) if C<=50 else random.randint(40000,60000) if C==100 else random.randint(70000,130000) for _ in range(D)]
    depot_caps = pick_depot_caps(C,D)
    path70 = os.path.join(out_dir,f"set{C}-{D}-{idx}_vc70.txt")
    path150 = os.path.join(out_dir,f"set{C}-{D}-{idx}_vc150.txt")
    write_instance_file(path70,C,D,depot_coords,cust_coords,70,depot_caps,cust_demands,open_costs,1000,0)
    write_instance_file(path150,C,D,depot_coords,cust_coords,150,depot_caps,cust_demands,open_costs,1000,0)
    return path70,path150

def create_data(txt_file):
    with open(txt_file,'r') as f: lines=[l.strip() for l in f if l.strip()]
    idx=0; C=int(lines[idx]); idx+=1; D=int(lines[idx]); idx+=1
    depot_coords = [tuple(map(float,lines[idx+i].split())) for i in range(D)]; idx+=D
    cust_coords = [tuple(map(float,lines[idx+i].split())) for i in range(C)]; idx+=C
    vehicle_cap=int(lines[idx]); idx+=1
    depot_caps=[int(lines[idx+i]) for i in range(D)]; idx+=D
    cust_demands=[int(lines[idx+i]) for i in range(C)]; idx+=C
    open_costs=[int(lines[idx+i]) for i in range(D)]; idx+=D
    route_cost=int(lines[idx]); idx+=1
    rc_cal_index=int(lines[idx]); idx+=1
    return (C,D,depot_coords,cust_coords,vehicle_cap,depot_caps,cust_demands,open_costs,route_cost,rc_cal_index)
