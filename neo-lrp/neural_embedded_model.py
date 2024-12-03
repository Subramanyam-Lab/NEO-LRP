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

def write_to_txt_cvrplib_format(depot_id, depot_customers, depot_coords, customer_demands, filename, vehicle_capacity):
    with open(filename, 'w') as file:
        file.write(f"NAME : {os.path.basename(filename)}\n")
        file.write("COMMENT : decision informed instance\n")
        file.write("TYPE : CVRP\n")
        file.write(f"DIMENSION : {len(depot_customers) + 1}\n")  # +1 for the depot
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write(f"CAPACITY : {vehicle_capacity}\n")
        file.write("NODE_COORD_SECTION\n")
        
        # Write depot coordinates (node 1)
        file.write(f"1 {depot_coords[0][0]} {depot_coords[0][1]}\n")
        
        # Write customer coordinates starting from node 2
        for i, coords in enumerate(depot_customers, start=2):
            file.write(f"{i} {coords[0]} {coords[1]}\n")

        file.write("DEMAND_SECTION\n")
        
        # Write depot demand (node 1)
        file.write(f"1 0\n")  # Depot demand is zero
        
        # Write customer demands starting from node 2
        for i, demand in enumerate(customer_demands, start=2):
            file.write(f"{i} {demand}\n")

        file.write("DEPOT_SECTION\n")
        file.write("1\n")  
        file.write("-1\n")  
        file.write("EOF\n")

class createLRP():
    def __init__(self, ans):
        self.customer_no = ans[0]
        self.depotno = ans[1]
        self.depot_cord = ans[2]
        self.customer_cord = ans[3]
        self.vehicle_capacity = ans[4]
        self.depot_capacity = ans[5]
        self.customer_demand = ans[6]
        self.facilitycost = ans[7]
        self.init_route_cost = ans[8]
        self.rc_cal_index = ans[9]
            
    def dataprocess(self, data_input_file, fi_mode='dynamic', fixed_fi_value=1000.0):
        # Input data file location
        # Normalize data wrt depot
        facility_dict, big_m, rc_norm = norm_data(self.depot_cord, self.customer_cord, self.vehicle_capacity, self.customer_demand, self.rc_cal_index, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)

        file_base_name = os.path.basename(data_input_file)
        file_name_without_ext = os.path.splitext(file_base_name)[0]
        output_dir = 'output'  # Specify your output directory
        os.makedirs(output_dir, exist_ok=True)
        # output_file_name = os.path.join(output_dir, f"{file_name_without_ext}_rc_norm.txt")

        # with open(output_file_name, 'w') as f:
        #     f.write("Normalization factor for route cost (rc_norm):\n")
        #     for idx, value in enumerate(rc_norm):
        #         f.write(f"Depot {idx}: {value}\n")

        print(f"Normalization factor for route cost {rc_norm}")

        # Initial facility customer assignments
        initial_flp_assignment = flp(self.customer_no, self.depotno, self.depot_cord, self.customer_cord, self.depot_capacity, self.customer_demand, self.facilitycost, self.init_route_cost, self.rc_cal_index)
        print(f"Initial FLP Assignments {initial_flp_assignment}")

        return facility_dict, big_m, rc_norm, initial_flp_assignment

    # Generating initial solution through simple flp model for customer assignments and using a forward pass through phi and rho networks
    def warmstart(self, flp_assignment, init_route_cost, customer_cord, customer_demand, rc_cal_index, phi_net, rho_net, fi_mode='dynamic', fixed_fi_value=1000.0):
        ws_dt_st = datetime.now()
        y_open = []
        for p in flp_assignment[1]:
            y_open.append(p)
        
        x_start = [[0]*self.customer_no for j in range(self.depotno)]

        for j in y_open:
            for i in range(self.customer_no):
                if i in flp_assignment[1][j]:
                    x_start[j][i] = 1

        y_start = [0]*self.depotno
        for k in range(self.depotno):
            if k in y_open:
                y_start[k] = 1

        # Normalize customers only to the facilities they are assigned
        phi_start = {}
        for i in flp_assignment[2]:
            d_cord = [flp_assignment[2][i][0]]
            c_cord = self.customer_cord
            c_dem = self.customer_demand
            v_cap = [self.vehicle_capacity[i]]
            phi_start[i] = norm_data(d_cord, c_cord, v_cap, c_dem, self.rc_cal_index, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)
        
        fac_dict_initial = {}
        rc_norm_factor = {}
        for k in phi_start:
            fac_dict_initial[k] = phi_start[k][0][0][phi_start[k][0][0].index.isin(flp_assignment[1][k])]
            rc_norm_factor[k] = phi_start[k][2]

        phi_outputs = {}
        for j in y_open:
            phi_outputs[j] = extract_onnx(fac_dict_initial[j].values, phi_net)
        
        sz = phi_outputs[y_open[0]].size()
        ls = sz[1]

        ws_phi_outputs = dict()
        for i in flp_assignment[1]:
            ws_phi_outputs[i] = {}
            for j in range(len(flp_assignment[1][i])):
                ws_phi_outputs[i][flp_assignment[1][i][j]] = phi_outputs[i][j]

        z_start = {}
        for j in range(self.depotno):
            z_start[j] = [0]*ls 
        for j in y_open:
            for l in range(ls):
                for i in flp_assignment[1][j]:
                    if x_start[j][i] == 1:
                        z_start[j][l] += x_start[j][i] * ws_phi_outputs[j][i][l]
                    
        # Initial routes cost
        route_cost_start = [0]*self.depotno
        
        for j in y_open:
            rho_output = extract_onnx(z_start[j], rho_net)
            route_cost_start[j] = rho_output[0].item()

        print(f"Initial individual Route cost {route_cost_start}")
        print(f"Normalization factor for route cost {rc_norm_factor}")

        initial_flp_cost = sum(self.facilitycost[j]*y_start[j] for j in range(self.depotno))
        print(f"Initial Facility Objective value is {initial_flp_cost}")
        if self.rc_cal_index == 0:
            initial_vrp_cost_variable = sum(rc_norm_factor[j][0]*route_cost_start[j] for j in y_open)
        else:
            initial_vrp_cost_variable = sum(rc_norm_factor[j][0]*route_cost_start[j] for j in y_open)
        
        initial_vrp_cost = initial_vrp_cost_variable

        print(f"Initial VRP Objective value is {initial_vrp_cost}")
        initial_obj = initial_flp_cost + initial_vrp_cost
        print(f"Initial Total Objective value is {initial_obj}")
        ws_dt_ed = datetime.now()
        ws_exec = (ws_dt_ed - ws_dt_st).total_seconds()
        return initial_obj, x_start, y_start, route_cost_start, z_start, ws_exec

    def model(self, loc, log_filename, DIL_instances, phi_loc, rho_loc, fi_mode='dynamic', fixed_fi_value=1000.0):
        facility_dict, big_m, rc_norm, initial_flp_assignment= self.dataprocess(loc, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)

        # Initial Feasible Solution for Gurobi model
        initial_objective_value, xst, yst, routecost_st, z_st, ws_time = self.warmstart(initial_flp_assignment, self.init_route_cost, self.customer_cord, self.customer_demand, self.rc_cal_index, phi_loc, rho_loc, fi_mode='dynamic', fixed_fi_value=1000.0)
        print("Initial Feasible solution:", initial_objective_value)
        
        # Passing data through phi network
        phi_final_outputs = {}
        for j in range(self.depotno):
            phi_final_outputs[j] = extract_onnx(facility_dict[j].values, phi_loc)

        # print(phi_final_outputs)
        sz = phi_final_outputs[0].size()
        latent_space = sz[1]

        # LRP Model
        m = gp.Model('facility_location')

        # Decision variables
        cartesian_prod = list(product(range(self.depotno), range(self.customer_no)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')
        for j in range(self.depotno):
            y[j].Start = yst[j]

        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        for j in range(self.depotno):
            for i in range(self.customer_no):
                x[j, i].Start = xst[j][i]

        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        for j in range(self.depotno):
            for l in range(latent_space):
                z[j, l].Start = z_st[j][l]

        route_cost = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route_cost')
        for j in range(self.depotno):
            route_cost[j].Start = routecost_st[j]

        num_routes = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='Number_of_routes')
                
        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_route_cost")

        v = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_number_of_routes")

        for j in range(self.depotno):
            for l in range(latent_space):
                m.addConstr(z[j, l] == gp.quicksum(x[j, i] * phi_final_outputs[j][i, l] for i in range(self.customer_no)), name=f'Z-plus[{j}][{l}]')
                    
        # Constraints
        m.addConstrs((gp.quicksum(x[(j, i)] for j in range(self.depotno)) == 1 for i in range(self.customer_no)), name='Demand')

        m.addConstrs((gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) <= self.depot_capacity[j] * y[j] for j in range(self.depotno)), name="facility_capacity_constraint")

        m.addConstrs((x[j, i] <= y[j] for j in range(self.depotno) for i in range(self.customer_no)), name='Assignment_to_open_facility')

        St_time = datetime.now()
        print("Start time for MIP part:", St_time)

        # Neural Network Constraints
        onnx_model = onnx.load(rho_loc)
        pytorch_rho_mdl = convert(onnx_model).double()
        layers = []
        # Get layers of the GraphModule
        for name, layer in pytorch_rho_mdl.named_children():
            layers.append(layer)
        sequential_model = nn.Sequential(*layers)

        z_values_per_depot = {}
        route_per_depot = {}

        # Extract the values of z for each depot and store them in the dictionary
        for j in range(self.depotno):
            z_values_per_depot[j] = [z[j, l] for l in range(latent_space)]
            route_per_depot[j] = [route_cost[j]]   

        for j in range(self.depotno):
            t_const = gt.add_sequential_constr(m, sequential_model, z_values_per_depot[j], route_per_depot[j])
            t_const.print_stats()

        # Indicator Constraint to stop cost calculation for closed depot
        for j in range(self.depotno):
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            m.addConstr((y[j] == 1) >> (u[j] == route_per_depot[j][0]))
                
        # Objective
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        if self.rc_cal_index == 0:
            route_obj = gp.quicksum((rc_norm[j] * u[j])  for j in range(self.depotno))
        else:
            route_obj = gp.quicksum((rc_norm[j] * u[j])  for j in range(self.depotno))

        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        # Save variables and data needed in the callback
        m._x = x
        m._y = y
        m._customer_no = self.customer_no
        m._depotno = self.depotno
        m._depot_cord = self.depot_cord
        m._customer_cord = self.customer_cord
        m._customer_demand = self.customer_demand
        m._vehicle_capacity = self.vehicle_capacity
        m._loc = loc
        m._DIL_instances = DIL_instances
        m._feasible_solution_count = 0  # Initialize feasible solution counter

        # Define callback function
        def mycallback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # A new integer feasible solution has been found
                # Increment feasible solution counter
                model._feasible_solution_count += 1

                # Create a subfolder for this feasible solution
                solution_folder = os.path.join(model._DIL_instances, f"feasible_solution_{model._feasible_solution_count}")
                os.makedirs(solution_folder, exist_ok=True)

                # Get the solution values
                x_vals = model.cbGetSolution(model._x)
                y_vals = model.cbGetSolution(model._y)

                # Process the variable values to extract the assignment
                open_depots = [j for j in range(model._depotno) if y_vals[j] > 0.5]

                # For each depot, get the customers assigned
                depot_customers = {}
                for j in open_depots:
                    assigned_customers = [i for i in range(model._customer_no) if x_vals[j, i] > 0.5]
                    depot_customers[j] = assigned_customers

                # For each open depot, write the DIL instance into the subfolder
                for depot_id in open_depots:
                    customers = depot_customers[depot_id]

                    # Get depot coordinates
                    depot_coords = [model._depot_cord[depot_id]]

                    # Get customer coordinates
                    customer_coords = [model._customer_cord[i] for i in customers]

                    # Get customer demands
                    customer_demands = [model._customer_demand[i] for i in customers]

                    # Vehicle capacity
                    vehicle_capacity = model._vehicle_capacity[0]  # Assuming same capacity for all depots

                    # Construct filename
                    filename = f"cvrp_instance_{os.path.basename(model._loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
                    output_file_path = os.path.join(solution_folder, filename)

                    # Write to file using the existing function
                    write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords, customer_demands, output_file_path, vehicle_capacity)

        # Solution terminate at 1% gap
        m.setParam('MIPGAP', 0.01)
        m.setParam('TimeLimit', 3600)
        m.setParam('MIPFocus', 1)

        # Optimize model with callback
        St_time1 = datetime.now()
        # m.write('model_feasible.lp')
        m.optimize(mycallback)  

        if m.Status == GRB.INFEASIBLE:
            print("Model is infeasible; computing IIS...")
            m.computeIIS()
            m.write("model.ilp")
            print("IIS written to model.ilp")
        else:
            # Optimization successful
            # Extract final solution values
            x_vals = m.getAttr('X', x)
            y_vals = m.getAttr('X', y)

            # Process the variable values to extract the assignment
            open_depots = [j for j in range(self.depotno) if y_vals[j] > 0.5]

            # For each depot, get the customers assigned
            depot_customers = {}
            for j in open_depots:
                assigned_customers = [i for i in range(self.customer_no) if x_vals[j, i] > 0.5]
                depot_customers[j] = assigned_customers

            # Create a subfolder for the final solution
            final_solution_folder = os.path.join(DIL_instances, 'final_solution')
            os.makedirs(final_solution_folder, exist_ok=True)

            # For each open depot, write the DIL instance into the final_solution subfolder
            for depot_id in open_depots:
                customers = depot_customers[depot_id]

                # Get depot coordinates
                depot_coords = [self.depot_cord[depot_id]]

                # Get customer coordinates
                customer_coords = [self.customer_cord[i] for i in customers]

                # Get customer demands
                customer_demands = [self.customer_demand[i] for i in customers]

                # Vehicle capacity
                vehicle_capacity = self.vehicle_capacity[0]  # Assuming same capacity for all depots

                # Construct filename
                filename = f"cvrp_instance_{os.path.basename(loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
                output_file_path = os.path.join(final_solution_folder, filename)

                write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords, customer_demands, output_file_path, vehicle_capacity)

        Ed_time = datetime.now()
        print("Objective value is ", end='')
        print(m.objVal, '\n')

        lrp_obj = m.objVal
        print(f"Objective value is {lrp_obj}")

        print('Facility objective value:', facility_obj.getValue())
        f_obj = facility_obj.getValue()
        print(f'Facility objective value: {f_obj}')

        print('Route Objective value:', route_obj.getValue())
        r_obj = route_obj.getValue()
        print(f'Route Objective value: {r_obj}')

        execution_time = (Ed_time - St_time).total_seconds()
        print("Lrp NN Script Execution time:", execution_time)
        print(f"Lrp NN Script Execution time: {execution_time}")
        execution_time1 = (Ed_time - St_time1).total_seconds()
        print("Lrp model Execution time:", execution_time1)
        print(f"Lrp model Execution time: {execution_time1}")

        # Execution time per depot
        cou = 0
        y_val = []
        for j in range(self.depotno):
            y_val.append(y[j].x)
            if y[j].x != 0:
                cou += 1
                print(cou)
        
        x_val = []
        for j in range(self.depotno):
            ls1 = []
            for i in range(self.customer_no):
                ls1.append(x[j, i].x)
            x_val.append(ls1)

        etpd = execution_time1 / cou if cou != 0 else 0

        return y_val, x_val, f_obj, r_obj, ws_time, execution_time1