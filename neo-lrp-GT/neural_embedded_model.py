"""
Neural Embedded Mixed-Integer Optimization for Location-Routing Problems
Core implementation using Graph Transformer networks
"""

import torch.nn as nn
import torch
import numpy as np
from dataparse import *
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from datetime import datetime
import logging
import os
import sys
import openpyxl
from net import *

def write_to_txt_cvrplib_format(depot_id, depot_customers, depot_coords, customer_demands, filename, vehicle_capacity, depot_route_cost):
    """
    Write VRP instance data to CVRPLIB format text file.
    
    Args:
        depot_id (int): Depot identifier
        depot_customers (list): List of customer coordinates assigned to this depot
        depot_coords (list): Depot coordinates
        customer_demands (list): Customer demand values
        filename (str): Output file path
        vehicle_capacity (int): Vehicle capacity constraint
        depot_route_cost (float): Predicted route cost for this depot
    """
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

        file.write(f"\n# ROUTE_COST: {depot_route_cost}\n")

class createLRP():
    """
    Neural Embedded Location-Routing Problem solver using Graph Transformer.
    
    This class implements the main LRP optimization model with neural network
    embedded routing cost prediction.
    """
    
    def __init__(self, ans):
        """
        Initialize LRP solver with problem data.
        
        Args:
            ans (list): Problem data containing:
                - customer_no: Number of customers
                - depotno: Number of depots
                - depot_cord: Depot coordinates
                - customer_cord: Customer coordinates
                - vehicle_capacity: Vehicle capacity
                - depot_capacity: Depot capacity
                - customer_demand: Customer demand values
                - facilitycost: Facility opening costs
                - init_route_cost: Initial route costs
                - rc_cal_index: Route cost calculation index
        """
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

    def model(self, loc, log_filename, DIL_instances, phi_loc, fi_mode='dynamic', fixed_fi_value=1000.0):
        """
        Solve the LRP using neural embedded optimization.
        
        Args:
            loc (str): Instance file path
            log_filename (str): Log file name
            DIL_instances (str): Directory for storing intermediate instances
            phi_loc (str): Path to trained Graph Transformer model
            fi_mode (str): Facility index mode ('dynamic' or 'fixed')
            fixed_fi_value (float): Fixed facility index value
            
        Returns:
            tuple: (y_values, x_values, facility_obj, vrp_cost, warmstart_time, pred_costs_dict)
        """
        # Normalize data for neural network processing
        facility_dict, rc_norm = norm_data(self.depot_cord, self.customer_cord, self.vehicle_capacity, self.customer_demand)

        # Neural Network Embedding using trained Graph Transformer model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model configuration for Graph Transformer
        model_config = {
            "activation": "elu",
            "beta": False,
            "decode_method": "pool",
            "dropout": 0.4,
            "encoding_dim": 64,
            "heads": 8,
            "normalization": "layer_norm",
            "num_gat_layers": 4
        }

        # Initialize and load trained Graph Transformer model
        trained_model = GraphTransformerNetwork(
            in_channels=model_config["encoding_dim"],
            hidden_channels=model_config["encoding_dim"],
            out_channels=model_config["encoding_dim"],
            heads=model_config["heads"],
            beta=model_config["beta"],
            dropout=model_config["dropout"],
            normalization=model_config["normalization"],
            num_gat_layers=model_config["num_gat_layers"],
            activation=model_config["activation"],
            decode_method=model_config["decode_method"]
        ).to(device)
        
        # Load pre-trained model weights
        trained_model.load_state_dict(torch.load(phi_loc, map_location=device))
        trained_model.eval()


        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        # Generate neural embeddings for each depot-customer subgraph
        phi_final_outputs = {}
        for j in range(self.depotno):
            df = facility_dict[j]['df']  # DataFrame with columns: 'x', 'y', 'dem'
            # Set depot indicator: 1 for depot, 0 for customers
            is_depot = np.zeros((df.shape[0], 1), dtype=np.float32)
            is_depot[0, 0] = 1  # First node is depot
            # Feature order: [x, y, is_depot, demand]
            features = np.hstack((df[['x', 'y']].to_numpy(), is_depot, df[['dem']].to_numpy()))
            x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            dist_mat = facility_dict[j]['dist']
            edge_index, edge_attr = dense_to_sparse(torch.tensor(dist_mat, dtype=torch.float32, device=device))
            data_obj = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
            
            # Generate embeddings using trained model
            with torch.no_grad():
                phi_emb, _ = trained_model.encode(data_obj)
            # phi_emb shape: (num_customers_in_depot, latent_space)
            phi_final_outputs[j] = phi_emb.detach().cpu().numpy()

        # Initialize Gurobi MIP model
        latent_space = model_config["encoding_dim"]
        m = gp.Model('facility_location')

        # Decision variables
        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')  # Facility opening
        x = m.addVars(self.depotno, self.customer_no, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')  # Customer assignment
        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")  # Neural embeddings
        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_route_cost")  # Predicted route costs
        
        # Constraints
        # Each customer must be assigned to exactly one depot
        m.addConstrs((gp.quicksum(x[j, i] for j in range(self.depotno)) == 1 for i in range(self.customer_no)), name='Demand')

        # Depot capacity constraint
        m.addConstrs((gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) <= self.depot_capacity[j] * y[j] for j in range(self.depotno)), name="facility_capacity_constraint")

        # Assignment constraint: customers can only be assigned to open facilities
        m.addConstrs((x[j, i] <= y[j] for j in range(self.depotno) for i in range(self.customer_no)), name='Assignment_to_open_facility')

        # Add neural embedding constraints
        print("Adding neural embedding constraints...")
        for j in range(self.depotno):
            for l in range(latent_space):
                # Embedding constraint: z[j,l] = depot_embedding + sum of assigned customer embeddings
                m.addConstr(
                    z[j, l] ==
                    phi_final_outputs[j][0, l]  # Depot embedding (row 0)
                    + gp.quicksum(x[j, i] * phi_final_outputs[j][i+1, l]  # Customer embeddings (row i+1)
                                   for i in range(self.customer_no)),
                    name=f"Zplus_{j}_{l}"
                )

        # Extract trained model weights for cost prediction
        W_out = trained_model.output_layer.weight.detach().cpu().numpy()[0]  # shape: (latent_space,)
        b_out = float(trained_model.output_layer.bias.detach().cpu().numpy()[0])

        # Add indicator constraints for route cost calculation
        for j in range(self.depotno):
            route_cost_expr = gp.quicksum(W_out[l] * z[j, l] for l in range(latent_space)) + b_out
            m.addConstr((y[j] == 0) >> (u[j] == 0))  # If depot closed, route cost = 0
            m.addConstr((y[j] == 1) >> (u[j] == route_cost_expr))  # If depot open, route cost = prediction
                
        # Define objective function
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        route_obj = gp.quicksum((1000.0 * rc_norm[j] * u[j]) for j in range(self.depotno))
        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        # Save variables and data needed in the callback
        m._rc_norm = rc_norm
        m._u = u
    
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

                rc_norm = model._rc_norm
                u_vals = model.cbGetSolution(model._u)

                # For each open depot, write the DIL instance into the subfolder
                for depot_id in open_depots:
                    customers = depot_customers[depot_id]
                    depot_coords = [model._depot_cord[depot_id]]
                    customer_coords = [model._customer_cord[i] for i in customers]
                    customer_demands = [model._customer_demand[i] for i in customers]
                    vehicle_capacity = model._vehicle_capacity[0]  # Assuming same capacity for all depots

                    cost_j = rc_norm[depot_id] * u_vals[depot_id] * 1000.0

                    filename = f"cvrp_instance_{os.path.basename(model._loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
                    output_file_path = os.path.join(solution_folder, filename)

                    write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords, customer_demands, output_file_path, vehicle_capacity, depot_route_cost=cost_j)

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

            rc_norm_vals = m._rc_norm            # rc_norm stored in model
            u_vals = m.getAttr("X", m._u)        # dummy_route_cost (dictionary)

            # For each open depot, write the DIL instance into the final_solution subfolder
            for depot_id in open_depots:
                customers = depot_customers[depot_id]
                depot_coords = [self.depot_cord[depot_id]]
                customer_coords = [self.customer_cord[i] for i in customers]
                customer_demands = [self.customer_demand[i] for i in customers]
                vehicle_capacity = self.vehicle_capacity[0]  # Assuming same capacity for all depots

                cost_j = rc_norm_vals[depot_id] * u_vals[depot_id] * 1000.0

                filename = f"cvrp_instance_{os.path.basename(loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
                output_file_path = os.path.join(final_solution_folder, filename)

                write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords, customer_demands, output_file_path, vehicle_capacity, depot_route_cost=cost_j)

        Ed_time = datetime.now()

        lrp_obj = m.objVal

        f_obj = facility_obj.getValue()
        r_obj = route_obj.getValue()


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



        return y_val, x_val, f_obj, r_obj, 0, pred_costs
