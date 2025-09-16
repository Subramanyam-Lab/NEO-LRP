"""
Neural Embedded Optimization Model for Location-Routing Problems - MLP Implementation

This module implements the neural embedded optimization approach using Multi-Layer Perceptron (MLP)
networks for routing cost prediction in Location-Routing Problems.
"""

import torch.nn as nn
import torch
import onnx
import numpy as np
from onnx2torch import convert
from dataparse import *
from network import *
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

def write_to_txt_cvrplib_format(depot_id, depot_customers, depot_coords, customer_demands, filename, vehicle_capacity, depot_route_cost):
    """
    Write VRP instance to CVRPLIB format file.
    
    Args:
        depot_id (int): ID of the depot
        depot_customers (list): List of customer coordinates assigned to this depot
        depot_coords (list): Depot coordinates
        customer_demands (list): Customer demands
        filename (str): Output filename
        vehicle_capacity (int): Vehicle capacity
        depot_route_cost (float): Route cost for this depot
    """
    with open(filename, 'w') as file:
        file.write(f"NAME : {os.path.basename(filename)}\n")
        file.write("COMMENT : decision informed instance\n")
        file.write("TYPE : CVRP\n")
        file.write(f"DIMENSION : {len(depot_customers) + 1}\n")  # +1 for the depot
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write(f"CAPACITY : {vehicle_capacity}\n")
        file.write("NODE_COORD_SECTION\n")
        
        # Write depot coordinates
        file.write(f"1 {depot_coords[0][0]} {depot_coords[0][1]}\n")
        
        # Write customer coordinates
        for i, coords in enumerate(depot_customers, start=2):
            file.write(f"{i} {coords[0]} {coords[1]}\n")

        file.write("DEMAND_SECTION\n")
        
        # Write depot demand
        file.write(f"1 0\n")  # Depot demand is zero
        
        # Write customer demands
        for i, demand in enumerate(customer_demands, start=2):
            file.write(f"{i} {demand}\n")

        file.write("DEPOT_SECTION\n")
        file.write("1\n")  
        file.write("-1\n")  
        file.write("EOF\n")

        file.write(f"\n# ROUTE_COST: {depot_route_cost}\n")

class createLRP():
    def __init__(self, ans):
        self.customer_no = ans[0] # number of customers
        self.depotno = ans[1] # number of depots
        self.depot_cord = ans[2] # depot coordinates
        self.customer_cord = ans[3] # customer coordinates
        self.vehicle_capacity = ans[4]
        self.depot_capacity = ans[5]
        self.customer_demand = ans[6]
        self.facilitycost = ans[7] # Opening cost of depots
        self.init_route_cost = ans[8] # Not used
        self.rc_cal_index = ans[9] # 0 or not
            
    def dataprocess(self, data_input_file, fi_mode='dynamic', fixed_fi_value=1000.0):
        # Normalize data wrt depot
        facility_dict, big_m, rc_norm = norm_data(self.depot_cord, self.customer_cord, self.vehicle_capacity, self.customer_demand, self.rc_cal_index, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)
        # facility_dict: normalized customer coordinates per depot, len(facility_dict) = depotno
        # rc_norm : cost_norm_factor (max_x_range or max_y_range)
        file_base_name = os.path.basename(data_input_file)
        file_name_without_ext = os.path.splitext(file_base_name)[0]
        output_dir = 'output'  # Specify your output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Normalization factor for route cost {rc_norm}")


        return facility_dict, big_m, rc_norm

    def model(self, loc, log_filename, DIL_instances, phi_loc, rho_loc, fi_mode='dynamic', fixed_fi_value=1000.0):
        facility_dict, big_m, rc_norm = self.dataprocess(loc, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)

        phi_final_outputs = {}
        for j in range(self.depotno):
            phi_final_outputs[j] = extract_onnx(facility_dict[j].values, phi_loc)

        sz = phi_final_outputs[0].size()
        latent_space = sz[1]

        # LRP Model
        m = gp.Model('facility_location')

        # Decision variables
        cartesian_prod = list(product(range(self.depotno), range(self.customer_no)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')
        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        route_cost = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route_cost')
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
            # t_const.print_stats()

        # Indicator Constraint to stop cost calculation for closed depot
        for j in range(self.depotno):
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            m.addConstr((y[j] == 1) >> (u[j] == route_per_depot[j][0]))
                
        # Objective
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        if self.rc_cal_index == 0:
            route_obj = gp.quicksum((1000.0 * rc_norm[j] * u[j])  for j in range(self.depotno))
        else:
            route_obj = gp.quicksum((1000.0 * rc_norm[j] * u[j])  for j in range(self.depotno))

        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)






        # Solution terminate at 1% gap
        m.setParam('MIPGAP', 0.01)
        m.setParam('TimeLimit', 3600)
        m.setParam('MIPFocus', 1)

        # Optimize model with callback
        St_time1 = datetime.now()
        # m.write('model_feasible.lp')
        m.optimize()  




        Ed_time = datetime.now()

        lrp_obj = m.objVal
        print(f"Objective value is {lrp_obj}")

        f_obj = facility_obj.getValue()
        print(f'Facility objective value: {f_obj}')

        r_obj = route_obj.getValue()
        print(f'Route Objective value: {r_obj}')


        execution_time1 = (Ed_time - St_time1).total_seconds()
        print("Lrp model Execution time:", execution_time1)

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

            
        # Collect predicted route cost for every open depot
        u_vals = m.getAttr("X", u)          # u[j] = NN predicted dummy_route_cost
        pred_costs = {}                    # depot_id -> actual currency unit route cost
        for j in range(self.depotno):
            if y[j].x > 0.5:               # depot is open
                pred_costs[j] = 1000.0 * rc_norm[j] * u_vals[j]


        return y_val, x_val, f_obj, r_obj, 0, execution_time1, pred_costs
