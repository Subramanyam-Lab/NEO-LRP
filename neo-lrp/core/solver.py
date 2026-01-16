"""
Unified CVRP Solver with ORTools, VROOM, and VRPSolverEasy

Distance and cost handling derived from data file:
- rc_cal_index = 0 so int(100 * distance), use VROOM cost directly
- rc_cal_index = 1 so float(distance), recompute cost from routes
- fixed_route_cost from data file ans[8]
"""

import math
import time
import threading
from datetime import datetime
import numpy as np

import vroom
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from VRPSolverEasy.src import solver
import pandas as pd

def _to_int_or_none(val):
    """Safely convert value to int or return None."""
    if val is None or (hasattr(pd, "isna") and pd.isna(val)):
        return None
    try:
        return int(val)
    except Exception:
        return None


def _extract_routes_robust(sol, assigned_ids, depot_label="Depot_0"):
    """
    Extract routes from a VROOM solution object.
    
    Returns list of dicts: [{'vehicle_id': int, 'Ids': ['Depot_k', 'cust1', ..., 'Depot_k']}, ...]
    Only routes with actual customers are included.
    """
    routes_out = []

    raw = None
    if hasattr(sol, "to_dict"):
        try:
            raw = sol.to_dict()
        except Exception:
            raw = None
    if raw is None and hasattr(sol, "raw"):
        raw = sol.raw

    if isinstance(raw, dict) and isinstance(raw.get("routes"), list):
        for idx, r in enumerate(raw["routes"]):
            steps = r.get("steps") or []
            vid = r.get("vehicle") or r.get("vehicle_id") or (idx + 1)
            seq, has_job = [depot_label], False
            for st in steps:
                if st.get("type") == "job":
                    j_int = _to_int_or_none(st.get("job") or st.get("id") or st.get("job_id"))
                    if j_int is None:
                        continue
                    k = j_int - 1000  # jobs created with ids 1001..1000+N
                    if 1 <= k <= len(assigned_ids):
                        seq.append(str(assigned_ids[k - 1]))  # Map to ORIGINAL IDs
                        has_job = True
            seq.append(depot_label)
            if has_job:
                routes_out.append({"vehicle_id": int(vid), "Ids": seq})
        if routes_out:
            return routes_out

    try:
        df = sol.routes
    except Exception:
        df = None

    if df is not None and "steps" in df.columns:
        for _, row in df.iterrows():
            steps = row["steps"] or []
            vid = _to_int_or_none(row.get("vehicle_id")) or _to_int_or_none(row.get("vehicle")) or 1
            seq, has_job = [depot_label], False
            for st in steps:
                if st.get("type") == "job":
                    j_int = _to_int_or_none(st.get("job") or st.get("id") or st.get("job_id"))
                    if j_int is None:
                        continue
                    k = j_int - 1000
                    if 1 <= k <= len(assigned_ids):
                        seq.append(str(assigned_ids[k - 1]))
                        has_job = True
            seq.append(depot_label)
            if has_job:
                routes_out.append({"vehicle_id": int(vid), "Ids": seq})

    return routes_out


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

class DataCvrp:
    """
    Container for CVRP instance data.
    
    Key attributes from data file:
        rc_cal_index: 0 = integer costs (x100), 1 = real costs (float)
        fixed_route_cost: cost per vehicle/route (ans[8] from data file)
    """
    def __init__(
        self,
        vehicle_capacity,
        nb_customers,
        cust_demands,
        cust_coordinates,
        depot_coordinates,
        original_ids=None,
        depot_label=None,
        rc_cal_index=0,
        fixed_route_cost=0
    ):
        self.vehicle_capacity = vehicle_capacity
        self.nb_customers = nb_customers
        self.cust_demands = cust_demands
        self.cust_coordinates = cust_coordinates
        self.depot_coordinates = depot_coordinates
        self.original_ids = original_ids
        self.depot_label = depot_label
        self.rc_cal_index = rc_cal_index
        self.fixed_route_cost = fixed_route_cost

    @property
    def use_integer_distances(self):
        """rc_cal_index=0 means integer distances with x100 scaling."""
        return self.rc_cal_index == 0

def compute_distance(x_i, y_i, x_j, y_j, use_integer):
    """
    Compute Euclidean distance with proper handling.
    
    Args:
        use_integer: If True (rc_cal_index=0), return int(100 * d)
                     If False (rc_cal_index=1), return raw float d
    
        - P_prodhon/S_schneider: int(100 * math.hypot(...))
        - T_tuzun/B_barreto: math.hypot(...) as float
    """
    d = math.hypot(x_i - x_j, y_i - y_j)
    if use_integer:
        return int(100 * d)  # Truncate after multiplication (matches VROOM behavior)
    return d


def compute_distance_matrix_ortools(all_coords, fixed_cost, use_integer):
    """
    Build distance matrix for OR-Tools with depot penalty.
    
    OR-Tools requires INTEGER distances.
    - rc_cal_index=0: int(100 * d) - already integer
    - rc_cal_index=1: scale by 1000 to preserve precision
    """
    num_nodes = len(all_coords)
    dist_matrix = []
    
    # Determine scale factor
    # use_integer=True (rc_cal_index=0): distances are int(100*d), scale=1
    # use_integer=False (rc_cal_index=1): distances are float, scale=1000
    scale_factor = 1 if use_integer else 1000
    
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i == j:
                row.append(0)
            else:
                d = math.hypot(
                    all_coords[i][0] - all_coords[j][0],
                    all_coords[i][1] - all_coords[j][1]
                )
                
                if use_integer:
                    # P_prodhon/S_schneider: int(100 * d)
                    d = int(100 * d)
                    if i == 0 or j == 0:
                        d = d + int(fixed_cost / 2)
                else:
                    # T_tuzun/B_barreto: scale float by 1000
                    d = int(d * scale_factor)
                    if i == 0 or j == 0:
                        d = d + int(fixed_cost * scale_factor / 2)
                
                row.append(d)
        dist_matrix.append(row)
    
    return dist_matrix, scale_factor


def extract_solution_details_ortools(data, manager, routing, solution):
    total_distance = 0
    routes_info = []
    scale_factor = data.get("scale_factor", 1)
    
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        if route_distance > 0:
            routes_info.append({
                "vehicle_id": vehicle_id,
                "route": route,
                "route_distance": route_distance / scale_factor,
                "route_load": sum(data["demands"][node] for node in route)
            })
            total_distance += route_distance
    
    num_non_zero_routes = len(routes_info)
    objective = solution.ObjectiveValue() / scale_factor
    return objective, total_distance / scale_factor, num_non_zero_routes, routes_info

def solve_cvrp_ortools(cvrp_data, ortools_timeout=5):
    """
    Solve CVRP with OR-Tools.
    
    Distances scaled to integers:
    - rc_cal_index=0: int(100 * d)
    - rc_cal_index=1: int(1000 * d), results scaled back
    """
    all_coords = [cvrp_data.depot_coordinates] + cvrp_data.cust_coordinates
    
    distance_matrix, scale_factor = compute_distance_matrix_ortools(
        all_coords,
        cvrp_data.fixed_route_cost,
        cvrp_data.use_integer_distances
    )
    
    demands = [0] + cvrp_data.cust_demands

    if isinstance(cvrp_data.vehicle_capacity, list):
        cap = int(cvrp_data.vehicle_capacity[0])
    else:
        cap = int(cvrp_data.vehicle_capacity)

    vehicle_capacities = [cap for _ in range(cvrp_data.nb_customers)]

    data = {
        "distance_matrix": distance_matrix,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "num_vehicles": cvrp_data.nb_customers,
        "depot": 0,
        "scale_factor": scale_factor,
    }

    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix), 
        data["num_vehicles"], 
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index):
        return data["demands"][manager.IndexToNode(from_index)]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index, 0, data["vehicle_capacities"], True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = ortools_timeout

    solution = routing.SolveWithParameters(search_params)
    return data, manager, routing, solution

def solve_cvrp_vroom(cvrp_data, vroom_timeout=5):
    """
    Solve CVRP with VROOM.
    
    Behavior based on rc_cal_index:
        rc_cal_index=0 (P_prodhon, S_schneider):
            - Distance matrix: int(100 * hypot(...))
            - Return: sol.summary.cost directly
        
        rc_cal_index=1 (T_tuzun, B_barreto):
            - Distance matrix: hypot(...) as float
            - Return: recomputed true_cost + fixed_cost * used_vehicles
    """
    depot_coord = cvrp_data.depot_coordinates
    cust_coords = cvrp_data.cust_coordinates
    cust_demands = cvrp_data.cust_demands
    vehicle_cap = cvrp_data.vehicle_capacity
    fixed_cost = cvrp_data.fixed_route_cost
    use_integer = cvrp_data.use_integer_distances

    # Normalize vehicle capacity
    if isinstance(vehicle_cap, (list, tuple)):
        vehicle_cap = vehicle_cap[0]
    vehicle_cap = int(vehicle_cap)

    n = len(cust_coords)
    points = [depot_coord] + cust_coords

    if use_integer:
        dist_matrix = [
            [
                0 if i == j else int(100 * math.hypot(
                    points[i][0] - points[j][0],
                    points[i][1] - points[j][1]
                ))
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]
    else:
        dist_matrix = [
            [
                0.0 if i == j else math.hypot(
                    points[i][0] - points[j][0],
                    points[i][1] - points[j][1]
                )
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]

    problem = vroom.Input()
    problem.set_durations_matrix(profile="car", matrix_input=dist_matrix)

    for i in range(n):
        vc = vroom.VehicleCosts(fixed=fixed_cost)
        veh = vroom.Vehicle(
            id=i + 1,
            start=0,
            end=0,
            capacity=vroom.Amount([vehicle_cap]),
            profile="car",
            costs=vc
        )
        problem.add_vehicle(veh)

    for i in range(1, n + 1):
        job = vroom.Job(id=1000 + i, location=i, delivery=[cust_demands[i - 1]])
        problem.add_job(job)

    sol = timeout_wrapper(
        problem.solve, 
        vroom_timeout, 
        exploration_level=5, 
        nb_threads=4
    )

    assigned_ids = list(cvrp_data.original_ids) if cvrp_data.original_ids else list(range(1, n + 1))
    depot_label = cvrp_data.depot_label or "Depot_0"

    routes_info = _extract_routes_robust(sol, assigned_ids, depot_label=depot_label)

    if not routes_info:
        routes_info = []
        df = getattr(sol, "routes", None)
        if df is not None:
            for vid in range(1, n + 1):
                group = df[df["vehicle_id"] == vid]
                if len(group) == 0:
                    routes_info.append({"vehicle_id": vid, "Ids": []})
                    continue
                jobs = group.loc[group["type"] == "job", "id"].dropna().astype(int).tolist()
                local_idx = [jid - 1000 for jid in jobs]  # 1..N
                orig = [
                    (assigned_ids[k - 1] if 1 <= k <= len(assigned_ids) else k)
                    for k in local_idx
                ]
                seq = [depot_label] + [str(x) for x in orig] + [depot_label]
                routes_info.append({"vehicle_id": vid, "Ids": seq})
        else:
            for vid in range(1, n + 1):
                routes_info.append({"vehicle_id": vid, "Ids": []})

    present = {r["vehicle_id"] for r in routes_info}
    for vid in range(1, n + 1):
        if vid not in present:
            routes_info.append({"vehicle_id": vid, "Ids": []})

    used_vehicles = sum(
        1 for r in routes_info 
        if any("Depot" not in str(s) for s in r["Ids"])
    )

    if use_integer:
        return sol.summary.cost, used_vehicles, routes_info
    else:
        id_to_coord = {cid: coord for cid, coord in zip(assigned_ids, cust_coords)}
        
        true_cost = 0.0
        for r in routes_info:
            # Extract customer IDs (exclude depot labels)
            route_ids = [int(i) for i in r["Ids"] if "Depot" not in str(i)]
            if not route_ids:
                continue
            
            seq = [depot_coord] + [id_to_coord[c] for c in route_ids] + [depot_coord]
            for a, b in zip(seq[:-1], seq[1:]):
                true_cost += math.hypot(a[0] - b[0], a[1] - b[1])
        
        return true_cost + fixed_cost * used_vehicles, used_vehicles, routes_info


def solve_cvrp_vrpeasy(cvrp_data, vrpeasy_timeout=5):
    """
    Solve CVRP with VRPSolverEasy.
    
    VRPSolverEasy supports float distances (unlike OR-Tools).
    - rc_cal_index=0: int(100 * d) 
    - rc_cal_index=1: raw float (no scaling needed)
    """
    model = solver.Model()

    fixed_cost = cvrp_data.fixed_route_cost
    use_integer = cvrp_data.use_integer_distances

    if isinstance(cvrp_data.vehicle_capacity, (list, np.ndarray)):
        cap = int(cvrp_data.vehicle_capacity[0])
    else:
        cap = int(cvrp_data.vehicle_capacity)

    model.add_vehicle_type(
        id=1,
        start_point_id=0,
        end_point_id=0,
        max_number=cvrp_data.nb_customers,
        capacity=cap,
        fixed_cost=fixed_cost,  # No scaling needed
        var_cost_dist=1
    )

    model.add_depot(id=0)
    
    for i, demand in enumerate(cvrp_data.cust_demands):
        model.add_customer(id=i + 1, demand=demand)

    all_coords = [cvrp_data.depot_coordinates] + cvrp_data.cust_coordinates

    for i in range(1, len(all_coords)):
        d = math.hypot(
            all_coords[i][0] - all_coords[0][0],
            all_coords[i][1] - all_coords[0][1]
        )
        dist = int(100 * d) if use_integer else round(d, 3)
        model.add_link(start_point_id=0, end_point_id=i, distance=dist)

    for i in range(1, len(all_coords)):
        for j in range(i + 1, len(all_coords)):
            d = math.hypot(
                all_coords[i][0] - all_coords[j][0],
                all_coords[i][1] - all_coords[j][1]
            )
            dist = int(100 * d) if use_integer else round(d, 3)
            model.add_link(start_point_id=i, end_point_id=j, distance=dist)

    model.set_parameters(time_limit=vrpeasy_timeout, solver_name="CLP")
    model.solve()

    if not model.solution.is_defined:
        print(f"[warn] vrpsolvereasy found no solution: {model.message}")
        return 0, 0, []

    cost = model.solution.value  # No scaling back needed
    num_routes = len(model.solution.routes)

    depot_label = cvrp_data.depot_label or "Depot_0"
    original_ids = cvrp_data.original_ids or list(range(cvrp_data.nb_customers))

    routes_info = []
    for idx, r in enumerate(model.solution.routes):
        mapped_ids = []
        for pid in r.point_ids:
            if pid == 0:
                mapped_ids.append(depot_label)
            else:
                mapped_ids.append(str(original_ids[pid - 1]))
        routes_info.append({"vehicle_id": idx + 1, "Ids": mapped_ids})

    return cost, num_routes, routes_info

def solve_instance(cvrp_data, solver_type="vroom", config=None):
    """
    Unified interface for all solvers.
    
    Args:
        cvrp_data: DataCvrp instance with problem data
                   (rc_cal_index and fixed_route_cost come from data file)
        solver_type: "ortools", "vroom", or "vrpeasy"
        config: Optional config dict with timeout settings
    
    Returns:
        (cost, num_routes, message, routes_info, solve_time)
    """
    vroom_timeout = config.get("vroom_timeout", 5) if config else 5
    ortools_timeout = config.get("ortools_timeout", 5) if config else 5
    vrpeasy_timeout = config.get("vrpeasy_timeout", 5) if config else 5

    if solver_type == "ortools":
        start = time.time()
        data, manager, routing, solution = solve_cvrp_ortools(cvrp_data, ortools_timeout)
        solve_time = time.time() - start

        if solution:
            obj, _, num_routes, routes_info = extract_solution_details_ortools(
                data, manager, routing, solution
            )
            return obj, num_routes, "Solved with ORTools", routes_info, solve_time
        else:
            return None, 0, "No solution (ORTools)", [], solve_time

    elif solver_type == "vroom":
        start = time.time()
        cost, num_routes, routes_info = solve_cvrp_vroom(cvrp_data, vroom_timeout)
        solve_time = time.time() - start
        return cost, num_routes, "Solved with VROOM", routes_info, solve_time

    elif solver_type == "vrpeasy":
        start = time.time()
        cost, num_routes, routes_info = solve_cvrp_vrpeasy(cvrp_data, vrpeasy_timeout)
        solve_time = time.time() - start
        return cost, num_routes, "Solved with VRPSolverEasy", routes_info, solve_time

    else:
        raise ValueError(f"Unknown solver: {solver_type}")