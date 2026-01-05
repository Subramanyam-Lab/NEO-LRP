from ortools.linear_solver import pywraplp
import logging
from dataparse import  dist_calc
from datetime import datetime
import os
import sys
import math
import numpy as np
import pandas as pd

class createFlp():
    def __init__(self,ans):
        self.cust_no=ans[0]
        self.depot_no=ans[1]
        self.dep_cord=ans[2]
        self.cus_cord=ans[3]
        self.depot_capacity=ans[5]
        self.cust_demand=ans[6]
        self.facility_cost=ans[7]
        self.rc=ans[8]
        self.rc_cal_index=ans[9]
        #self.data_input_file=x
    def flp(self):
        dist=dist_calc(self.dep_cord,self.cus_cord,self.rc_cal_index)
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Variables
        # x[i, j] = 1 if customer i served by facility j.
        x = {}
        for i in range(self.cust_no):
            for j in range(self.depot_no):
                x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

        # y[j] = 1 if facility j is opened.
        y = {}
        for j in range(self.depot_no):
            y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

        # Constraints
        # Each customer must be served by exactly one facility.
        for i in range(self.cust_no):
            solver.Add(sum(x[i, j] for j in range(self.depot_no)) == 1)

        # The customers served should not exceed the facility capacity
        for j in range(self.depot_no):
            solver.Add(
                sum(x[(i, j)] * self.cust_demand[i] for i in range(self.cust_no)) <= (y[j] * self.depot_capacity[j] ))

        # Objective: minimize the total cost
        fac_obj=[y[j]*self.facility_cost[j] for j in range(self.depot_no)]
        dist_factor=[x[(i,j)]*dist[j][i] for i in range(self.cust_no) for j in range(self.depot_no)]
        obj_exp=(fac_obj+dist_factor)

        solver.Minimize(solver.Sum(obj_exp))

        st1=datetime.now()
        status = solver.Solve()
        ed1=datetime.now()
        exec_time=(ed1-st1).total_seconds()

        if status == pywraplp.Solver.OPTIMAL:
            print('Solution:')
            print('Objective value =', solver.Objective().Value())
            
            flp_cost=sum([var.solution_value() for var in fac_obj])

            fac_cost = float(sum(var.solution_value() for var in fac_obj))
            sur_cost = float(sum(var.solution_value() for var in dist_factor))
            total_obj = fac_cost + sur_cost

            opened_vector = [int(y[j].solution_value()) for j in range(self.depot_no)]
            opened_indices = [j for j, v in enumerate(opened_vector) if v == 1]
            closed_indices = [j for j, v in enumerate(opened_vector) if v == 0]

            # customer to depot assignment
            assign_depot = []
            for i in range(self.cust_no):
                j_star = None
                for j in range(self.depot_no):
                    if x[i, j].solution_value() > 0.5:
                        j_star = j
                        break
                assign_depot.append(j_star if j_star is not None else -1)

            # depot to [customers]
            depot_to_customers = {j: [] for j in opened_indices}
            for i, j in enumerate(assign_depot):
                if j in depot_to_customers:
                    depot_to_customers[j].append(i)

            logging.info(f"Objective value of FLP {flp_cost}")
            flp_dict={}
            for j in range(self.depot_no):
                print('facility {} is {}'.format(j,y[j].solution_value()))
                logging.info(f"Facilities {j} is {y[j].solution_value()}")
                if y[j].solution_value()==1:
                    ls=[]
                    for i in range(self.cust_no):          
                        if x[i,j].solution_value()==1:
                            ls.append(i)
                    
                    flp_dict[j]=ls

        else:
            print('The problem does not have an optimal solution.')

        rout_dist={}
        fac_cust_dem={}
        cust_dem_fac={}
        for f in flp_dict:
            ls1=[]
            ls2=[]
            dem_sum=0
            for c in (flp_dict[f]):
                ls1.append(self.cus_cord[c])
                dem_sum=dem_sum+self.cust_demand[c]
                ls2.append(self.cust_demand[c])
            ls1.insert(0,self.dep_cord[f])
            rout_dist[f]=ls1
            fac_cust_dem[f]=dem_sum
            cust_dem_fac[f]=ls2

        logging.info(f"Objective value of FLP {flp_cost}")
        logging.info(f"Customer Assigment for open facilities {flp_dict}")
        logging.info(f"Coordinates for facility and customer for each set {rout_dist}")
        logging.info(f"Aggregated demands for each facility {fac_cust_dem}")
        logging.info(f"Assigned customer demand for open facilities {cust_dem_fac}")

        self.last_solution = {
            "method": "FLPVRP",
            "objective": {
                "facility_opening": fac_cost,
                "surrogate_assignment": sur_cost,
                "total": total_obj
            },
            "timing_sec": exec_time,
            "sets": {
                "opened_vector": opened_vector,
                "opened_indices": opened_indices,
                "closed_indices": closed_indices,
                "assign_depot_per_customer": assign_depot,
                "depot_to_customers": depot_to_customers
            },
            "data": {
                "facility_cost": list(map(float, self.facility_cost)),
                "depot_capacity": list(map(int, self.depot_capacity)),
                "cust_demand": list(map(int, self.cust_demand)),
                "dep_coords": [list(map(float, xy)) for xy in self.dep_cord],
                "cus_coords": [list(map(float, xy)) for xy in self.cus_cord],
                "distance_surrogate": "out-and-back"
            }
        }

        return flp_cost,flp_dict,rout_dist,fac_cust_dem,cust_dem_fac,exec_time, sur_cost

