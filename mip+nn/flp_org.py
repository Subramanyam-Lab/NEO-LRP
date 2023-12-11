from ortools.linear_solver import pywraplp
from datetime import datetime
from dataparse import create_data, dist_calc
import math
import numpy as np
import pandas as pd


def flp(cust_no,depot_no,dep_cord,cus_cord,depot_capacity,cust_demand,facility_cost,rc,rc_cal_index):
    dist=dist_calc(dep_cord,cus_cord,rc_cal_index)
# Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    # x[i, j] = 1 if customer i served by facility j.
    x = {}
    for i in range(cust_no):
        for j in range(depot_no):
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if facility j is opened.
    y = {}
    for j in range(depot_no):
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

    # Constraints
    # Each customer must be served by exactly one facility.
    for i in range(cust_no):
        solver.Add(sum(x[i, j] for j in range(depot_no)) == 1)

    # The customers served should not exceed the facility capacity
    for j in range(depot_no):
        solver.Add(
            sum(x[(i, j)] * cust_demand[i] for i in range(cust_no)) <= (y[j] * depot_capacity[j] ))

    # Objective: minimize the total cost
    fac_obj=[y[j]*facility_cost[j] for j in range(depot_no)]
    dist_factor=[x[(i,j)]*dist[j][i] for i in range(cust_no) for j in range(depot_no)]
    obj_exp=(fac_obj+dist_factor)

    solver.Minimize(solver.Sum(obj_exp))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        flp_cost=sum([var.solution_value() for var in fac_obj])
        flp_dict={}
        for j in range(depot_no):
            print('facility {} is {}'.format(j,y[j].solution_value()))
            if y[j].solution_value()==1:
                ls=[]
                for i in range(cust_no):          
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
            ls1.append(cus_cord[c])
            dem_sum=dem_sum+cust_demand[c]
            ls2.append(cust_demand[c])
        ls1.insert(0,dep_cord[f])
        rout_dist[f]=ls1
        fac_cust_dem[f]=dem_sum
        cust_dem_fac[f]=ls2

    assignment_df=pd.DataFrame(columns=['Facility','cust_x','cust_y','cust_dem'])
    l1=[]
    l2=[]
    l3=[]
    l4=[]
    for f in flp_dict:
        for c in (flp_dict[f]):
                l1.append(f)
                l2.append(float(cus_cord[c][0]))
                l3.append(float(cus_cord[c][1]))
                l4.append(cust_demand[c])
    assignment_df['Facility']=l1
    assignment_df['cust_x']=l2
    assignment_df['cust_y']=l3
    assignment_df['cust_dem']=l4
    assignment_df.to_csv("coord200-10-2.csv")


    return flp_cost,flp_dict,rout_dist,fac_cust_dem,cust_dem_fac

if __name__=='__main__':
    St_time=datetime.now()
    print("Start time for FLP:",St_time)
    ans=create_data("F:\Benchmark\B1\coord20-5-1b.dat")
    customer_no=ans[0]
    depotno=ans[1]
    depot_cord=ans[2]
    customer_cord=ans[3]
    vehicle_capacity=ans[4]
    depot_capacity=ans[5]
    customer_demand=ans[6]
    facilitycost=ans[7]
    route_cost=ans[8]
    rc_cal_index=ans[9]
    a,b,c,d,e=flp(customer_no,depotno,depot_cord,customer_cord,depot_capacity,customer_demand,facilitycost,route_cost,rc_cal_index)
    print("Facility solution value:",a,'\n')
    print("Facility Customer assignment:",b,'\n')
    print("Facility Customer distances:",c,'\n')
    print("Total Demands on each facility",d,'\n')
    print("Demands on each facility per assigned customers",e,'\n')
    Ed_time=datetime.now()
    print("Lrp Script Execution time:", (Ed_time-St_time).total_seconds())
    
