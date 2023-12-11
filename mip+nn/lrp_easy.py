import torch.nn as nn
import torch
import onnx
import numpy as np
from onnx2torch import convert
from dataparse import *
from network import *
from flp_org import *
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from gurobi_ml import *
import gurobi_ml.torch as gt           
from datetime import datetime
import logging
import os
import sys
import openpyxl

name_pattern = os.environ.get('name_pattern')
class createLRP ():
    def __init__(self,ans):
        self.customer_no=ans[0]
        self.depotno=ans[1]
        self.depot_cord=ans[2]
        self.customer_cord=ans[3]
        self.vehicle_capacity=ans[4]
        self.depot_capacity=ans[5]
        self.customer_demand=ans[6]
        self.facilitycost=ans[7]
        self.init_route_cost=ans[8]
        self.rc_cal_index=ans[9]
        
        
    def dataprocess(self,data_input_file):
        #input data file location
        phi_loc='onnx_files/model_phi_modifed_loss_no_route_abs.onnx'
        rho_loc='onnx_files/model_rho_modifed_loss_no_route_abs.onnx'
        logging.info(f"The phi file name:{phi_loc}\n")
        logging.info(f"The rho file name:{rho_loc}\n")
        # #preparing for log file
        # input_file_base_name = os.path.basename(data_input_file)

        # # Create a directory for log files (if it doesn't exist)
        # log_dir = "log_files/lrp2"
        # os.makedirs(log_dir, exist_ok=True)
        # # Create the log file name based on the input file's name
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        # file=os.path.splitext(input_file_base_name)[0] 
        # log_filename = f"{file}_nn_lrp_{current_time}.log"

        # logging.basicConfig(
        #     level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
        #     format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
        #     handlers=[
        #         logging.FileHandler(os.path.join(log_dir, log_filename)),  # Log to a file
        #         logging.StreamHandler(sys.stdout)   # Log to stdout
        #     ]
        # )

        # parse the dat file
        #self.customer_no,self.customer_demand,self.depotno,self.depot_cord,self.customer_cord,self.vehicle_capacity,self.depot_capacity,self.facilitycost,self.init_route_cost,self.rc_cal_index=createdata(data_input_file)

        #normalize data wrt depot
        facility_dict,big_m,rc_norm=norm_data(self.depot_cord,self.customer_cord,self.vehicle_capacity,self.customer_demand,self.rc_cal_index)


        #initial facility customer assignments
        initial_flp_assignment=flp(self.customer_no,self.depotno,self.depot_cord,self.customer_cord,self.depot_capacity,self.customer_demand,self.facilitycost,self.init_route_cost,self.rc_cal_index)
        logging.info(f"Initial FLP Assignments{initial_flp_assignment}")

        return facility_dict,big_m,rc_norm, initial_flp_assignment,phi_loc,rho_loc

    def warmstart(self,flp_assignment,init_route_cost,customer_cord,customer_demand,rc_cal_index,phi_net,rho_net):
        ws_dt_st=datetime.now()
        y_open=[]
        for p in flp_assignment[1]:
            y_open.append(p)
        
        x_start=[[0]*self.customer_no for j in range(self.depotno)]

            
        for j in y_open:
            i=0
            for i in range(self.customer_no):
                    if i in flp_assignment[1][j]:
                        #print("For cust {} and facility {}".format(i,j))
                        x_start[j][i]=1
                    else:
                        i+=1

        y_start=[0]*self.depotno
        for k in range(self.depotno):
            if k in y_open:
                y_start[k]=1
            else:
                k+=1

        #normalize customers only to the facilities they are assigned
        phi_start={}
        for i in flp_assignment[2]:
            print(flp_assignment[2][i])
            d_cord=[flp_assignment[2][i][0]]
            c_cord=self.customer_cord
            c_dem=self.customer_demand
            v_cap=[self.vehicle_capacity[i]]
            print("\n\n")
            print(d_cord,c_cord)
            phi_start[i]=norm_data(d_cord,c_cord,v_cap,c_dem,self.rc_cal_index)
        
        print(phi_start)
        fac_dict_initial={}
        rc_norm_factor= {}
        for k in phi_start:
            fac_dict_initial[k]=phi_start[k][0][0][phi_start[k][0][0].index.isin(flp_assignment[1][k])]
            rc_norm_factor[k]=phi_start[k][2]

        print(fac_dict_initial)  
        print(rc_norm_factor) 


        phi_outputs={}
        for j in y_open:
            phi_outputs[j]=extract_onnx(fac_dict_initial[j].values,phi_net)
        
        
        
        print(phi_outputs[y_open[0]].size())
        sz=phi_outputs[y_open[0]].size()
        ls=sz[1]
        #logging.info(f"\nWarmstart phi output:{phi_outputs}\n")

        ws_phi_outputs=dict()
        for i in flp_assignment[1]:
            ws_phi_outputs[i]={}
            for j in range (len(flp_assignment[1][i])):
                ws_phi_outputs[i][flp_assignment[1][i][j]]=phi_outputs[i][j]
        #print(ws_phi_outputs)
        #logging.info(f"\nWarmstart original indexing of customers:{ws_phi_outputs}\n")
        
        
        z_start={}
        for j in range(self.depotno):
            z_start[j]=[0]*ls 
        for j in y_open:
            l=0
            for l in range(ls):
                    for i in flp_assignment[1][j]:
                        if x_start[j][i]==1:
                            z_start[j][l]+= (x_start[j][i])* (ws_phi_outputs[j][i][l])
                

                

        # initial routes cost and number
        route_cost_start=[0]*self.depotno
        num_route_start=[0]*self.depotno
        for j in y_open:
            rho_output=extract_onnx(z_start[j],rho_net)
            route_cost_start[j]=rho_output[0].item()
            num_route_start[j]=rho_output[1].item()

        print("Initial Route cost",route_cost_start)
        logging.info(f"Initial individual Route cost {route_cost_start}")
        print("Initial Number of Routes",num_route_start)
        logging.info(f"Initial Number of routes {num_route_start}")
        logging.info(f"Normalization factor for route cost {rc_norm_factor}")
        initial_flp_cost= sum(self.facilitycost[j]*y_start[j] for j in range(self.depotno))
        logging.info(f"Initial Facility Objective value is {initial_flp_cost}")
        if self.rc_cal_index==0:
            initial_vrp_cost_variable=sum(rc_norm_factor[j][0]*route_cost_start[j]*100  for j in y_open)
        else:
            initial_vrp_cost_variable=sum(rc_norm_factor[j][0]*route_cost_start[j] for j in y_open)
        initial_vrp_cost_fixed=sum(num_route_start[j]*init_route_cost for j in y_open)
        print("Fixed cost {} and Variable cost {}".format(initial_vrp_cost_fixed,initial_vrp_cost_variable))
        
        initial_vrp_cost=initial_vrp_cost_variable+initial_vrp_cost_fixed
        logging.info(f"Initial VRP Objective value is {initial_vrp_cost}")
        initial_obj= initial_flp_cost+initial_vrp_cost
        logging.info(f"Initial Total Objective value is {initial_obj}")
        ws_dt_ed=datetime.now()
        ws_exec=(ws_dt_ed-ws_dt_st).total_seconds()
        return initial_obj,x_start,y_start,route_cost_start,num_route_start,z_start,ws_exec


    def model(self,loc,log_filename):
        
        facility_dict,big_m,rc_norm, initial_flp_assignment,phi_loc,rho_loc=self.dataprocess(loc)

        #initial Feasible Solution for gurobi model
        initial_objective_value,xst,yst,routecost_st,numroute_st,z_st,ws_time=self.warmstart(initial_flp_assignment,self.init_route_cost,self.customer_cord,self.customer_demand,self.rc_cal_index,phi_loc,rho_loc)
        print("Initial Feasible solution:",initial_objective_value)
        

        #passing data through phi network
        phi_final_outputs={}
        for j in range(self.depotno):
            phi_final_outputs[j]=extract_onnx(facility_dict[j].values,phi_loc)

        print(phi_final_outputs[0].size())
        sz=phi_final_outputs[0].size()
        latent_space=sz[1]


        #LRP Model
        

        m = gp.Model('facility_location')

        # Decision variables
        cartesian_prod= list(product(range(self.depotno),range(self.customer_no)))
        cp2=list(product(range(self.depotno),range(latent_space)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY,lb=0,ub=1,name='Facility')
        for j in range(self.depotno):
            y[j].Start = yst[j]

        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        for j in range(self.depotno):
            for i in range(self.customer_no):
                x[j,i].Start = xst[j][i]

        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=0 ,name="z-plus")
        for j in range(self.depotno):
            for l in range(latent_space):
                z[j,l].Start=z_st[j][l]

        route_cost=m.addVars(self.depotno,vtype=GRB.CONTINUOUS,lb=0,name='route cost ')
        for j in range(self.depotno):
            route_cost[j].Start = routecost_st[j]

        num_routes=m.addVars(self.depotno,vtype=GRB.CONTINUOUS, lb=0,name='Number of routes')

        for j in range(self.depotno):
            num_routes[j].Start = numroute_st[j]
            
        u=m.addVars(self.depotno,vtype=GRB.CONTINUOUS,lb=0, name= "dummy route cost")

        v=m.addVars(self.depotno,vtype=GRB.CONTINUOUS,lb=0, name= "dummy number of routes")

        for j in range(self.depotno):
            for l in range(latent_space):
                m.addConstr(z[j, l] == gp.quicksum(x[j, i] * phi_final_outputs[j][i,l] for i in range(self.customer_no)),name=f'Z-plus[{j}][{l}]')
                

        #Constraint
        m.addConstrs((gp.quicksum(x[(j,i)] for j in range(self.depotno)) == 1 for i in range(self.customer_no)), name='Demand')

        m.addConstrs((gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) <= self.depot_capacity[j] * y[j] for j in range(self.depotno)), name="facility_capacity_constraint")


        St_time=datetime.now()
        print("Start time for MIP part:",St_time)

        #Neural Network Constraints
        onnx_model = onnx.load(rho_loc)
        pytorch_rho_mdl = convert(onnx_model).double()
        layers = []
        # Get layers of the GraphModule
        for name, layer in pytorch_rho_mdl.named_children():
            layers.append(layer)
        sequential_model = nn.Sequential(*layers)

        z_values_per_depot = {}
        route_per_depot={}

        # Extract the values of z for each depot and store them in the dictionary
        for j in range(self.depotno):
            
            z_values_per_depot[j] = [z[j, l] for l in range(latent_space)]
            route_per_depot[j]=[route_cost[j],num_routes[j]]  


        for j in range(self.depotno):
            t_const=gt.add_sequential_constr(m, sequential_model,z_values_per_depot[j],route_per_depot[j])
            t_const.print_stats()

        # Indicator Constraint to stop cost calculation for closed depot
        for j in range(self.depotno):
            m.addConstr((y[j]==0)>>(u[j]==0))
            m.addConstr((y[j]==1)>>(u[j]==route_per_depot[j][0]))
            m.addConstr((y[j]==0)>>(v[j]==0))
            m.addConstr((y[j]==1)>>(v[j]==route_per_depot[j][1]))
            
        #Objective
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        if self.rc_cal_index==0:
            route_obj = gp.quicksum((rc_norm[j]*u[j]*100 + self.init_route_cost * v[j])  for j in range(self.depotno))
        else:
            route_obj = gp.quicksum((rc_norm[j]*u[j] + self.init_route_cost * v[j])  for j in range(self.depotno))

        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        sys.stdout = open(log_filename, "a")

        #Soluton terminate at 5%gap
        m.setParam('MIPGAP', 0)

        #Optimize model
        St_time1=datetime.now()
        m.optimize()
        Ed_time=datetime.now()
        print("Objective value is ", end='')
        print(m.objVal,'\n')

        lrp_obj=m.objVal
        logging.info(f"Objective value is {lrp_obj}")

        print('Facility objective value:',facility_obj.getValue())
        f_obj=facility_obj.getValue()
        logging.info(f'Facility objective value: {f_obj}')

        print('Route Objective value:',route_obj.getValue())
        r_obj=route_obj.getValue()
        logging.info(f'Route Objective value: {r_obj}')

        
        execution_time = (Ed_time - St_time).total_seconds()
        print("Lrp NN Script Execution time:", execution_time)
        logging.info(f"Lrp NN Script Execution time: {execution_time}")
        execution_time1 = (Ed_time - St_time1).total_seconds()
        print("Lrp model Execution time:", execution_time1)
        logging.info(f"Lrp model Execution time: {execution_time1}")

        #execution time per depot
        cou=0
        y_val=[]
        for j in range(self.depotno):
            y_val.append(y[j].x)
            if y[j].x!=0:
                cou+=1
            print(cou)
        
        x_val=[]
        for j in range(self.depotno):
            ls1=[]
            i=0
            for i in range(self.customer_no):
                ls1.append(x[j,i].x)
            x_val.append(ls1)

        etpd= execution_time1/cou

        for v in m.getVars():
            logging.info(v.varName)
            logging.info(str(v.x))
            logging.info("\n")
        


        #print(m.display())
        logging.info(m.display())

        #self.writeexcel(file,f_obj,r_obj,lrp_obj,etpd)

        sys.stdout.close()
        sys.stdout = sys.__stdout__

        # Close the log file
        logging.shutdown()

        return y_val, x_val, f_obj,r_obj, ws_time,execution_time1

       

    def writeexcel(self,data_input_file,f_obj,r_obj,lrp_obj,etpd):

        existing_excel_file="F:\MLEV\lrp_output.xlsx"
        workbook = openpyxl.load_workbook(existing_excel_file)

        # Select the worksheet where you want to append the new row
        worksheet = workbook.active  # Or specify a particular worksheet by name if needed

        # Create a new row
        new_row = [data_input_file, f_obj, r_obj, lrp_obj,etpd]  # Replace with your data
        
        # Append the new row to the worksheet
        worksheet.append(new_row)

        # Save the modified Excel file
        workbook.save(existing_excel_file)

            


