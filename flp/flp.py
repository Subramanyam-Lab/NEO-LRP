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

        


        return flp_cost,flp_dict,rout_dist,fac_cust_dem,cust_dem_fac,exec_time

