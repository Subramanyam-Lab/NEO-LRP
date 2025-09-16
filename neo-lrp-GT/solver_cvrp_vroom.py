"""
VROOM-based CVRP solver for Location-Routing Problems
Provides exact VRP solution using VROOM optimization engine
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import vroom  # pyvroom
import logging

# Helper Functions


def _euclid_cost(x_i: float, y_i: float,
                 x_j: float, y_j: float,
                 rc_cal_index: int) -> int:
    """
    Calculate Euclidean distance between two points and convert to VROOM cost units.
    
    Args:
        x_i, y_i: Coordinates of first point
        x_j, y_j: Coordinates of second point
        rc_cal_index: Route cost calculation index (0 for scaled, 1 for direct)
        
    Returns:
        int: Distance in VROOM cost units
    """
    dist = math.hypot(x_i - x_j, y_i - y_j)
    return int(100 * dist) if rc_cal_index == 0 else int(dist)


def _build_distance_matrix(coords: List[Tuple[float, float]],
                           rc_cal_index: int) -> List[List[int]]:
    """
    Build symmetric distance matrix from coordinate list.
    
    Args:
        coords: List of (x, y) coordinate tuples
        rc_cal_index: Route cost calculation index
        
    Returns:
        List[List[int]]: Symmetric distance matrix
    """
    n = len(coords)
    mtx: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cij = _euclid_cost(coords[i][0], coords[i][1],
                               coords[j][0], coords[j][1],
                               rc_cal_index)
            mtx[i][j] = mtx[j][i] = cij
    return mtx


def _extract_routes(solution: vroom.Solution) -> List[Dict[str, List[str]]]:
    """
    Extract routes from VROOM solution in the expected format.
    
    Args:
        solution: VROOM solution object
        
    Returns:
        List[Dict[str, List[str]]]: Routes in format [{'Ids': ['0','3','5',...]}, ...]
    """
    routes_out: List[Dict[str, List[str]]] = []
    routes_df = solution.routes
    for vid, df in routes_df.groupby("vehicle_id"):
        # Sort by arrival time to get depot-customer-...-depot sequence
        locs = df.sort_values("arrival")["location_index"].to_list()
        routes_out.append({"Ids": [str(i) for i in locs]})
    return routes_out


# Main Solver Class


class dataCvrp:
    """
    VROOM-based CVRP solver for Location-Routing Problems.
    
    This class handles the exact VRP solution using VROOM optimization engine
    for each depot-customer assignment from the FLP solution.
    
    Args:
        ans: Problem data from create_data() function
        flp_ass: FLP assignment results (facility_obj, flp_dict, rout_dist, fac_cust_dem, cust_dem_fac)
    """

    def __init__(self, ans, flp_ass):
        # Store problem data
        self.flp_ass = flp_ass
        self.nb_customers = ans[0]
        self.nb_open_depot = list(flp_ass[1].keys())
        self.coord = flp_ass[2]              # {dep_id: [dep,(x,y),cust1,…]}
        self.vehicle_capacity = ans[4]
        self.rc_cal_index = ans[9]

        self.flp_cost = flp_ass[0]
        
        # Solution tracking variables
        self.vrp_cost = 0
        self.tot_routes = 0
        self.variable_cost: List[int] = []
        self.num_routes: List[int] = []
        self.actual_routes: List[List[Dict]] = []
        self.message: List[str] = []
        self.solve_time: List[float] = []
        self.model_exec_time: float = 0.0

    # VROOM solver

    def _solve_single_depot(self,
                            depot_coord: Tuple[float, float],
                            cust_coords: List[Tuple[float, float]],
                            cust_demands: List[int],
                            fixed_cost: int = 1000,
                            exploration_level: int = 5,
                            nb_threads: int = 4) -> Tuple[int, int,
                                                          List[Dict], float]:
        """
        Solve single depot CVRP using VROOM.
        
        Args:
            depot_coord: Depot coordinates (x, y)
            cust_coords: List of customer coordinates
            cust_demands: List of customer demands
            fixed_cost: Fixed cost per vehicle
            exploration_level: VROOM exploration level
            nb_threads: Number of threads for VROOM
            
        Returns:
            Tuple containing:
                - cost: Total cost (distance + fixed_cost * vehicles)
                - used_vehicles: Number of routes/vehicles used
                - routes_list: Routes in [{'Ids':[...]}] format
                - exec_time: Solver execution time in seconds
        """

        # Build distance/cost matrix
        coords_all = [depot_coord] + cust_coords
        matrix = _build_distance_matrix(coords_all, self.rc_cal_index)

        # Create VROOM Input
        problem = vroom.Input()
        problem.set_durations_matrix("car", matrix)

        # Add vehicles
        # Maximum number of vehicles = #customers (generous allocation)
        cap = vroom.Amount([self.vehicle_capacity[0]])
        for v_id in range(len(cust_coords)):
            vc = vroom.VehicleCosts(fixed=fixed_cost)
                                    # time=0,    # durations unit → 1:1 cost
                                    # distance=1)
            veh = vroom.Vehicle(id=v_id + 1,
                                start=0,
                                end=0,
                                profile="car",
                                capacity=cap,
                                costs=vc,
                                description=f"veh_{v_id+1}")
            problem.add_vehicle(veh)

        # Add jobs
        for idx, demand in enumerate(cust_demands, start=1):
            jb = vroom.Job(id=1000 + idx,
                           location=idx,
                           delivery=[demand])
            problem.add_job(jb)

        # Solve
        t0 = time.time()
        sol = problem.solve(exploration_level=exploration_level,
                            nb_threads=nb_threads)
        exec_t = time.time() - t0

        cost = sol.summary.cost
        used_veh = sol.routes["vehicle_id"].nunique()
        routes_list = _extract_routes(sol)

        return cost, used_veh, routes_list, exec_t

    # Public API

    def runVRPeasy(self):
        """
        (Name maintained) – After calling VROOM for all open depots,
        returns the same tuple as the existing logic in main.py.
        """
        for j in self.nb_open_depot:
            depot_coord = self.coord[j][0]           # depot (x,y)
            cust_coords = self.coord[j][1:]          # Customer coordinates
            cust_demands = self.flp_ass[4][j][1:]       # Customer demands

            logging.info(f"[VROOM] depot {j} – |cust|={len(cust_coords)}")

            cost, m, routes, exec_t = self._solve_single_depot(
                depot_coord,
                cust_coords,
                cust_demands,
                fixed_cost=1000,
                exploration_level=5,
                nb_threads=4
            )

            # Accumulation
            self.vrp_cost += cost
            self.tot_routes += m
            self.variable_cost.append(cost)
            self.num_routes.append(m)
            self.actual_routes.append(routes)
            self.solve_time.append(exec_t)
            self.model_exec_time += exec_t
            self.message.append(f"VROOM ok – routes:{m}, cost:{cost}")

            print(f"[VROOM] depot {j} → cost={cost}, routes={m}")

        print("Final VRP cost (variable+fixed):", self.vrp_cost)
        print("Number of routes:", self.tot_routes)

        total_lrp_cost = self.flp_cost + self.vrp_cost
        total_vrp_cost = self.vrp_cost

        return (total_lrp_cost,
                total_vrp_cost,
                self.variable_cost,
                self.num_routes,
                self.message,
                self.solve_time,
                self.actual_routes,
                self.model_exec_time)