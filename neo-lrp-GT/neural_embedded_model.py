import torch
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from datetime import datetime
import os
import logging
import openpyxl

from dataparse import *
from net import *


def write_to_txt_cvrplib_format(depot_id, depot_customers, depot_coords, customer_demands,
                                filename, vehicle_capacity, depot_route_cost):
    """
    Write a CVRP instance in standard CVRPLIB format.
    """
    with open(filename, 'w') as file:
        file.write(f"NAME : {os.path.basename(filename)}\n")
        file.write("COMMENT : decision informed instance\n")
        file.write("TYPE : CVRP\n")
        file.write(f"DIMENSION : {len(depot_customers) + 1}\n")  # +1 for depot
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write(f"CAPACITY : {vehicle_capacity}\n")
        file.write("NODE_COORD_SECTION\n")
        
        # Depot coordinates (node 1)
        file.write(f"1 {depot_coords[0][0]} {depot_coords[0][1]}\n")
        
        # Customer coordinates (nodes 2..n)
        for i, coords in enumerate(depot_customers, start=2):
            file.write(f"{i} {coords[0]} {coords[1]}\n")

        file.write("DEMAND_SECTION\n")
        
        # Depot demand
        file.write("1 0\n")
        
        # Customer demands
        for i, demand in enumerate(customer_demands, start=2):
            file.write(f"{i} {demand}\n")

        file.write("DEPOT_SECTION\n1\n-1\nEOF\n")
        file.write(f"\n# ROUTE_COST: {depot_route_cost}\n")


class createLRP:
    """
    LRP model class integrating MIP optimization with Graph Transformer-based
    cost predictions for CVRP routing.
    """

    def __init__(self, ans):
        # Extract input data
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

    def model(self, loc, log_filename, DIL_instances, phi_loc,
              fi_mode='dynamic', fixed_fi_value=1000.0):
        """
        Solve LRP + CVRP problem using Gurobi MIP and Graph Transformer
        latent-space route cost predictions.
        """

        # Normalize depot/customer data for the neural network
        facility_dict, rc_norm = norm_data(
            self.depot_cord, self.customer_cord, self.vehicle_capacity, self.customer_demand
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Transformer configuration
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

        # Load pre-trained Graph Transformer
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
        trained_model.load_state_dict(torch.load(phi_loc, map_location=device))
        trained_model.eval()

        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        # Encode latent-space embeddings for each depot
        phi_final_outputs = {}
        for j in range(self.depotno):
            df = facility_dict[j]['df']  # columns: 'x', 'y', 'dem'
            is_depot = np.zeros((df.shape[0], 1), dtype=np.float32)
            is_depot[0, 0] = 1  # mark depot
            features = np.hstack((df[['x', 'y']].to_numpy(), is_depot, df[['dem']].to_numpy()))
            x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            dist_mat = facility_dict[j]['dist']
            edge_index, edge_attr = dense_to_sparse(torch.tensor(dist_mat, dtype=torch.float32, device=device))
            data_obj = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
            with torch.no_grad():
                phi_emb, _ = trained_model.encode(data_obj)
            phi_final_outputs[j] = phi_emb.detach().cpu().numpy()

        # Initialize Gurobi model
        latent_space = model_config["encoding_dim"]
        m = gp.Model('facility_location')

        # Decision variables
        y = m.addVars(self.depotno, vtype=GRB.BINARY, name='Facility')
        x = m.addVars(self.depotno, self.customer_no, vtype=GRB.BINARY, name='Assign')
        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS,
                      lb=-GRB.INFINITY, name="z")
        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_route_cost")

        # ---------------- Constraints ----------------
        # Each customer assigned to exactly one depot
        m.addConstrs(
            (gp.quicksum(x[j, i] for j in range(self.depotno)) == 1
             for i in range(self.customer_no)), name='Demand'
        )

        # Depot capacity constraints
        m.addConstrs(
            (gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no))
             <= self.depot_capacity[j] * y[j] for j in range(self.depotno)), name="facility_capacity_constraint"
        )

        # Assignment only to open depots
        m.addConstrs((x[j, i] <= y[j] for j in range(self.depotno) for i in range(self.customer_no)),
                     name='Assignment_to_open_facility')

        # ---------------- Latent-space embedding constraints ----------------
        for j in range(self.depotno):
            for l in range(latent_space):
                m.addConstr(
                    z[j, l] == phi_final_outputs[j][0, l]
                    + gp.quicksum(x[j, i] * phi_final_outputs[j][i+1, l]
                                  for i in range(self.customer_no)),
                    name=f"Zplus_{j}_{l}"
                )

        # ---------------- Route cost calculation using trained model ----------------
        W_out = trained_model.output_layer.weight.detach().cpu().numpy()[0]
        b_out = float(trained_model.output_layer.bias.detach().cpu().numpy()[0])

        for j in range(self.depotno):
            route_expr = gp.quicksum(W_out[l] * z[j, l] for l in range(latent_space)) + b_out
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            m.addConstr((y[j] == 1) >> (u[j] == route_expr))

        # ---------------- Objective ----------------
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        route_obj = gp.quicksum(1000.0 * rc_norm[j] * u[j] for j in range(self.depotno))
        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        # Save additional info for callback
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
        m._feasible_solution_count = 0

        # ---------------- Callback function ----------------
        def mycallback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                model._feasible_solution_count += 1
                solution_folder = os.path.join(model._DIL_instances,
                                               f"feasible_solution_{model._feasible_solution_count}")
                os.makedirs(solution_folder, exist_ok=True)

                x_vals = model.cbGetSolution(model._x)
                y_vals = model.cbGetSolution(model._y)
                u_vals = model.cbGetSolution(model._u)

                # Identify open depots and their assigned customers
                open_depots = [j for j in range(model._depotno) if y_vals[j] > 0.5]
                depot_customers = {
                    j: [i for i in range(model._customer_no) if x_vals[j, i] > 0.5]
                    for j in open_depots
                }

                # Write CVRP instances per depot
                for depot_id in open_depots:
                    customers = depot_customers[depot_id]
                    depot_coords = [model._depot_cord[depot_id]]
                    customer_coords = [model._customer_cord[i] for i in customers]
                    customer_demands = [model._customer_demand[i] for i in customers]
                    vehicle_capacity = model._vehicle_capacity[0]
                    cost_j = model._rc_norm[depot_id] * u_vals[depot_id] * 1000.0
                    filename = f"cvrp_instance_{os.path.basename(model._loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
                    write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords,
                                                customer_demands, os.path.join(solution_folder, filename),
                                                vehicle_capacity, depot_route_cost=cost_j)

        # ---------------- Gurobi Parameters ----------------
        m.setParam('MIPGAP', 0.01)
        m.setParam('TimeLimit', 3600)
        m.setParam('MIPFocus', 1)

        # Optimize model with callback
        St_time1 = datetime.now()
        m.optimize(mycallback)

        # Handle infeasible model
        if m.Status == GRB.INFEASIBLE:
            print("Model is infeasible; computing IIS...")
            m.computeIIS()
            m.write("model.ilp")
            print("IIS written to model.ilp")

        # ---------------- Extract final solution ----------------
        x_vals = m.getAttr('X', x)
        y_vals = m.getAttr('X', y)
        u_vals = m.getAttr('X', u)

        open_depots = [j for j in range(self.depotno) if y_vals[j] > 0.5]
        depot_customers = {
            j: [i for i in range(self.customer_no) if x_vals[j, i] > 0.5]
            for j in open_depots
        }

        # Create final solution folder
        final_solution_folder = os.path.join(DIL_instances, 'final_solution')
        os.makedirs(final_solution_folder, exist_ok=True)

        for depot_id in open_depots:
            customers = depot_customers[depot_id]
            depot_coords = [self.depot_cord[depot_id]]
            customer_coords = [self.customer_cord[i] for i in customers]
            customer_demands = [self.customer_demand[i] for i in customers]
            vehicle_capacity = self.vehicle_capacity[0]
            cost_j = rc_norm[depot_id] * u_vals[depot_id] * 1000.0
            filename = f"cvrp_instance_{os.path.basename(loc).split('.')[0]}_depot_{depot_id}_customers_{len(customers)}.txt"
            write_to_txt_cvrplib_format(depot_id, customer_coords, depot_coords,
                                        customer_demands, os.path.join(final_solution_folder, filename),
                                        vehicle_capacity, depot_route_cost=cost_j)

        # Objective values
        lrp_obj = m.objVal
        f_obj = facility_obj.getValue() if 'facility_obj' in locals() else 0
        r_obj = route_obj.getValue() if 'route_obj' in locals() else 0

        # Collect solution values
        y_val = [y[j].x for j in range(self.depotno)]
        x_val = [[x[j, i].x for i in range(self.customer_no)] for j in range(self.depotno)]

        # Predicted route costs per open depot
        pred_costs = {j: 1000.0 * rc_norm[j] * u_vals[j] for j in open_depots}

        return y_val, x_val, f_obj, r_obj, 0, pred_costs
