import math
import time
import logging
from datetime import datetime

import vroom
import pandas as pd

def _to_int_or_none(val):
    if val is None or (hasattr(pd, "isna") and pd.isna(val)):
        return None
    try:
        return int(val)
    except Exception:
        return None

def _extract_routes_robust(sol, assigned_ids):
    """
    Return [{'Ids': ['D0', <orig ids...>, 'D0']}, ...] where ids are original
    customer indices (assigned_ids).
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
        for r in raw["routes"]:
            steps = r.get("steps") or []
            seq, has_job = ["D0"], False
            for st in steps:
                if st.get("type") == "job":
                    j_int = _to_int_or_none(st.get("job") or st.get("id") or st.get("job_id"))
                    if j_int is None:
                        continue
                    k = j_int - 1000   
                    if 1 <= k <= len(assigned_ids):
                        seq.append(str(assigned_ids[k - 1]))
                        has_job = True
            seq.append("D0")
            if has_job:
                routes_out.append({"Ids": seq})
        if routes_out:
            return routes_out

    try:
        df = sol.routes
    except Exception:
        df = None

    if df is not None:
        if "steps" in df.columns:
            for _, row in df.iterrows():
                steps = row["steps"] or []
                seq, has_job = ["D0"], False
                for st in steps:
                    if st.get("type") == "job":
                        j_int = _to_int_or_none(st.get("job") or st.get("id") or st.get("job_id"))
                        if j_int is None:
                            continue
                        k = j_int - 1000
                        if 1 <= k <= len(assigned_ids):
                            seq.append(str(assigned_ids[k - 1]))
                            has_job = True
                seq.append("D0")
                if has_job:
                    routes_out.append({"Ids": seq})
            if routes_out:
                return routes_out

        cols = set(df.columns)
        if {"vehicle_id", "arrival", "location_index"} <= cols:
            for _, grp in df.groupby("vehicle_id"):
                locs = grp.sort_values("arrival")["location_index"].tolist()
                seq = ["D0"]
                has_job = False
                for loc in locs:
                    if loc and int(loc) >= 1:
                        k = int(loc)   
                        if 1 <= k <= len(assigned_ids):
                            seq.append(str(assigned_ids[k - 1]))
                            has_job = True
                seq.append("D0")
                if has_job:
                    routes_out.append({"Ids": seq})
            if routes_out:
                return routes_out

        job_col = "id" if "id" in df.columns else ("job" if "job" in df.columns else ("job_id" if "job_id" in df.columns else None))
        if job_col:
            for _, grp in df.groupby("vehicle_id"):
                seq, has_job = ["D0"], False
                for __, r in grp.iterrows():
                    j_int = _to_int_or_none(r.get(job_col))
                    if j_int is None:
                        continue
                    k = j_int - 1000
                    if 1 <= k <= len(assigned_ids):
                        seq.append(str(assigned_ids[k - 1]))
                        has_job = True
                seq.append("D0")
                if has_job:
                    routes_out.append({"Ids": seq})

    return routes_out



class dataCvrp:
    """Contains all data for CVRP problem"""

    def __init__(self, ans, flp_ass):
        self.nb_customers = ans[0]
        self.nb_open_depot = list(flp_ass[1].keys())
        self.coord = flp_ass[2]
        self.vehicle_capacity = ans[4]
        self.rc_cal_index = ans[9]

        self.flp_cost = flp_ass[0]
        self.vrp_cost = 0
        self.tot_routes = 0

        self.flp_ass = flp_ass

        self.message = []
        self.variable_cost = []
        self.num_routes = []
        self.actual_routes = []
        self.solve_time = []

    def runVROOM(self):
        """Runs VROOM once per open depot assignment aggregating totals."""
        for j in self.nb_open_depot:
            ass_cord = self.coord[j]                  
            cust_demands = self.flp_ass[4][j]         
            cust = self.flp_ass[1][j]                 

            vrp_data = self.create_data_model(
                ass_cord, self.nb_open_depot, cust_demands,
                self.vehicle_capacity, self.nb_customers, cust, self.rc_cal_index
            )

            cost, m, message, routes, exec_time = self.solve_files_in_directory(vrp_data, cust)

            self.actual_routes.append(routes)
            self.variable_cost.append(cost)
            self.num_routes.append(m)
            self.solve_time.append(exec_time)
            self.message.append(message)

            self.vrp_cost += cost
            print(cost, m)
            self.tot_routes += m
            print("sequence of routes:", routes)
            logging.info(f"Sequence of routes is {routes}")

        total_lrp_cost = self.flp_cost + self.vrp_cost + 1000 * self.tot_routes
        total_vrp_cost = self.vrp_cost + 1000 * self.tot_routes
        print("Total LRP cost", total_lrp_cost)
        logging.info(f"Final VRP cost is {total_lrp_cost}")


        self.last_solution_vroom = {
            "method": "VROOM",
            "objective": {
                "routing_total": float(self.vrp_cost),                
                "lrp_total": float(self.flp_cost + self.vrp_cost),
            },
            "timing_sec": float(sum(self.solve_time)) if self.solve_time else None,
            "per_depot": {
                str(dep): {
                    "routing_total": float(varc),                      
                    "num_routes": int(m),
                    "routes": routes,                                  
                }
                for dep, varc, m, routes in zip(
                    self.nb_open_depot, self.variable_cost, self.num_routes, self.actual_routes
                )
            },
        }


        return (
            total_lrp_cost,
            total_vrp_cost,
            self.variable_cost,
            self.num_routes,
            self.message,
            self.solve_time,
            self.actual_routes,
        )

    def compute_euclidean_distance(self, x_i, y_i, x_j, y_j, rc_cal_index):
        """Compute euclidean distance"""
        if rc_cal_index == 0:
            return int(100 * (math.hypot((x_i - x_j), (y_i - y_j))))
        else:
            return int(math.hypot((x_i - x_j), (y_i - y_j)))

    def create_data_model(self, all_cord, op_dep, cust_dem, veh_cap, customer_no, cust, rc_cal_index):
        """Stores the data for the problem"""
        data = {}
        data["locations"] = all_cord          
        data["open_depot"] = op_dep
        data["nb_cust"] = customer_no
        data["demands"] = cust_dem            
        data["vehicle_capacities"] = veh_cap
        data["depot"] = 0
        data["cust_cor"] = all_cord[1:]
        data["depot_cor"] = all_cord[0]
        data["assigned_cust"] = cust          
        data["rc_cal_index"] = rc_cal_index
        return data

    def _build_vroom_matrix(self, locations, rc_cal_index):
        """Build a symmetric 'durations' matrix for VROOM using distance metric."""
        n = len(locations)
        mat = [[0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = float(locations[i][0]), float(locations[i][1])
            for j in range(i + 1, n):
                xj, yj = float(locations[j][0]), float(locations[j][1])
                d = self.compute_euclidean_distance(xi, yi, xj, yj, rc_cal_index)
                mat[i][j] = d
                mat[j][i] = d
        return mat

    def solve_demo(
        self,
        data,
        cust,
        time_resolution=5,          
        solver_name_input="VROOM",
        solver_path="",                
        exploration_level=5,
        nb_threads=1,                  
    ):
        dist_matrix = self._build_vroom_matrix(data["locations"], data["rc_cal_index"])

        problem = vroom.Input()
        problem.set_durations_matrix(profile="car", matrix_input=dist_matrix)

        cap = (
            int(data["vehicle_capacities"][0])
            if isinstance(data["vehicle_capacities"], (list, tuple))
            else int(data["vehicle_capacities"])
        )

        veh_count = max(1, len(data["locations"]) - 1)

        vc = vroom.VehicleCosts()
        for vid in range(1, veh_count + 1):
            veh = vroom.Vehicle(
                id=vid,
                start=0,
                end=0,
                capacity=vroom.Amount([cap]),
                profile="car",
                costs=vc,
            )
            problem.add_vehicle(veh)

        for i in range(1, len(data["locations"])):
            demand_i = int(data["demands"][i -1])
            job = vroom.Job(id=1000 + i, location=i, delivery=[demand_i])
            problem.add_job(job)

        st = datetime.now()
        sol = problem.solve(exploration_level=exploration_level, nb_threads=nb_threads)
        ed = datetime.now()
        ex_time = (ed - st).total_seconds()

        routing_total = float(getattr(sol.summary, "cost", 0.0))

        try:
            df = sol.routes  
        except Exception:
            df = None

        routes = _extract_routes_robust(sol, data["assigned_cust"])
        used_vehicles = len(routes)

        message = f"VROOM status={getattr(sol, 'status', 'ok')} routing_total={routing_total} vehicles={used_vehicles}"

        if routes:
            print("[vroom-debug] route example:", routes[0])

        return routing_total, used_vehicles, message, routes, ex_time

    def solve_files_in_directory(self, data, cust):
        start_time = time.time()
        cost, m, message, routes, ex_time = self.solve_demo(data, cust)
        return cost, m, message, routes, ex_time
