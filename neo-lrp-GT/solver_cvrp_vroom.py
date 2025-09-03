from __future__ import annotations
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import vroom  # pyvroom


def _euclid_cost(x_i: float, y_i: float,
                 x_j: float, y_j: float,
                 rc_cal_index: int) -> int:
    """
    Compute Euclidean distance between (xi,yi) and (xj,yj)
    and convert to integer cost for VROOM.
    """
    dist = math.hypot(x_i - x_j, y_i - y_j)
    return int(100 * dist) if rc_cal_index == 0 else int(dist)


def _build_distance_matrix(coords: List[Tuple[float, float]],
                           rc_cal_index: int) -> List[List[int]]:
    """
    Build a square distance/cost matrix including depot.
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
    Convert VROOM solution.routes (DataFrame) into list of dicts:
    [{'Ids': ['0','3','5',…]}, …]
    """
    routes_out: List[Dict[str, List[str]]] = []
    routes_df = solution.routes
    for vid, df in routes_df.groupby("vehicle_id"):
        # Sort jobs by arrival order to preserve depot->customer->depot sequence
        locs = df.sort_values("arrival")["location_index"].to_list()
        routes_out.append({"Ids": [str(i) for i in locs]})
    return routes_out


# ------------------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------------------

class dataCvrp:
    """
    Wrapper to solve CVRP using VROOM for all open depots.

    Parameters
    ----------
    ans     : Output from create_data()
    flp_ass : Tuple (facility_obj, flp_dict, rout_dist, fac_cust_dem, cust_dem_fac)
              Passed directly from main.py
    """

    def __init__(self, ans, flp_ass):
        self.flp_ass = flp_ass
        self.nb_customers = ans[0]
        self.nb_open_depot = list(flp_ass[1].keys())
        self.coord = flp_ass[2]               # {dep_id: [dep,(x,y),cust1,…]}
        self.vehicle_capacity = ans[4]
        self.rc_cal_index = ans[9]

        self.flp_cost = flp_ass[0]
        # Accumulated and return variables
        self.vrp_cost = 0
        self.tot_routes = 0
        self.variable_cost: List[int] = []
        self.num_routes: List[int] = []
        self.actual_routes: List[List[Dict]] = []
        self.message: List[str] = []
        self.solve_time: List[float] = []
        self.model_exec_time: float = 0.0

    # ------------------------------------------------------------------ VROOM --

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

        Returns
        -------
        cost           : total cost (distance + fixed vehicle costs)
        used_vehicles  : number of vehicles used
        routes_list    : list of routes [{'Ids':[...]}]
        exec_time      : solver execution time in seconds
        """
        # Build distance/cost matrix including depot
        coords_all = [depot_coord] + cust_coords
        matrix = _build_distance_matrix(coords_all, self.rc_cal_index)

        # ----------------- VROOM Input -----------------------------
        problem = vroom.Input()
        problem.set_durations_matrix("car", matrix)

        # ----------------- Vehicles -------------------------------
        cap = vroom.Amount([self.vehicle_capacity[0]])
        for v_id in range(len(cust_coords)):
            vc = vroom.VehicleCosts(fixed=fixed_cost)
            veh = vroom.Vehicle(id=v_id + 1,
                                start=0,
                                end=0,
                                profile="car",
                                capacity=cap,
                                costs=vc,
                                description=f"veh_{v_id+1}")
            problem.add_vehicle(veh)

        # ----------------- Jobs ------------------------------------
        for idx, demand in enumerate(cust_demands, start=1):
            jb = vroom.Job(id=1000 + idx,
                           location=idx,
                           delivery=[demand])
            problem.add_job(jb)

        # ----------------- Solve -----------------------------------
        t0 = time.time()
        sol = problem.solve(exploration_level=exploration_level,
                            nb_threads=nb_threads)
        exec_t = time.time() - t0

        cost = sol.summary.cost
        used_veh = sol.routes["vehicle_id"].nunique()
        routes_list = _extract_routes(sol)

        return cost, used_veh, routes_list, exec_t


    def runVRPeasy(self):
        """
        Solve VRP for all open depots using VROOM.

        Returns
        -------
        Tuple containing:
        (total_lrp_cost, total_vrp_cost, variable_cost, num_routes,
         message, solve_time, actual_routes, model_exec_time)
        """
        for j in self.nb_open_depot:
            depot_coord = self.coord[j][0]          # depot coordinate (x,y)
            cust_coords = self.coord[j][1:]         # customer coordinates
            cust_demands = self.flp_ass[4][j][1:]  # customer demands

            logging.info(f"[VROOM] depot {j} – |cust|={len(cust_coords)}")

            cost, m, routes, exec_t = self._solve_single_depot(
                depot_coord,
                cust_coords,
                cust_demands,
                fixed_cost=1000,
                exploration_level=5,
                nb_threads=4
            )

            # Accumulate results
            self.vrp_cost += cost
            self.tot_routes += m
            self.variable_cost.append(cost)
            self.num_routes.append(m)
            self.actual_routes.append(routes)
            self.solve_time.append(exec_t)
            self.model_exec_time += exec_t
            self.message.append(f"VROOM ok – routes:{m}, cost:{cost}")

            print(f"[VROOM] depot {j} → cost={cost}, routes={m}")

        print("Final VRP cost (variable + fixed):", self.vrp_cost)
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
