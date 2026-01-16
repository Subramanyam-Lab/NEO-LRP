"""
Neural Embedded LRP Model - Supports DeepSets and Graph Transformer.
Uses Gurobi-ML to embed neural network in MIP optimization.
"""

import torch.nn as nn
import torch
import onnx
import numpy as np
from onnx2torch import convert
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import gurobi_ml.torch as gt
from datetime import datetime
import os
import json

from core.dataparse import norm_data
from core.network_ds import extract_onnx
from core.network_gt import load_gt_model
from scipy.spatial import distance_matrix as scipy_distance_matrix

def denormalize_cost(y_hat, fi, mode, config):
    """
    De-normalizes model output based on the normalization mode used during training.
    """
    norm_file = config.get("normalization_file")
    if norm_file and os.path.exists(norm_file):
        with open(norm_file, "r") as f:
            norm_data_dict = json.load(f)
    else:
        norm_data_dict = {}

    # Get custom keys from config, or use defaults
    keys = config.get("normalization_keys", {})

    if mode == "raw":
        return y_hat

    elif mode == "minmax":
        key_min = keys.get("minmax_min", "minmax_min_vroom")
        key_max = keys.get("minmax_max", "minmax_max_vroom")
        cmin = norm_data_dict.get(key_min, 0)
        cmax = norm_data_dict.get(key_max, 1)
        return y_hat * (cmax - cmin) + cmin

    elif mode == "cost_over_fi":
        return y_hat * fi

    elif mode == "cost_over_fi_minmax":
        key_min = keys.get("cost_over_fi_minmax_min", "cost_over_fi_minmax_min_vroom")
        key_max = keys.get("cost_over_fi_minmax_max", "cost_over_fi_minmax_max_vroom")
        cmin = norm_data_dict.get(key_min, 0)
        cmax = norm_data_dict.get(key_max, 1)
        return (y_hat * (cmax - cmin) + cmin) * fi

    else:
        raise ValueError(f"Unknown normalization mode '{mode}'")

class createLRP():
    def __init__(self, ans, cost_normalization="cost_over_fi", config=None):
        if cost_normalization is None:
            raise ValueError("You must provide a cost_normalization mode.")

        self.customer_no = ans[0]
        self.depotno = ans[1]
        self.depot_cord = ans[2]
        self.customer_cord = ans[3]
        self.vehicle_capacity = ans[4]
        self.depot_capacity = ans[5]
        self.customer_demand = ans[6]
        self.facilitycost = ans[7]
        self.init_route_cost = ans[8]  # From .dat file!
        self.rc_cal_index = ans[9]     # From .dat file!
        self.cost_normalization = cost_normalization
        self.config = config if config is not None else {}
        
        self.model_type = self.config.get("model_type", "deepsets")

    def model(self, loc, log_filename, phi_loc, rho_loc, cost_normalization=None):
        """
        Run LRP with NN embedded via GurobiML package
        Handles both DeepSets and Graph Transformer architectures
        """
        if cost_normalization:
            self.cost_normalization = cost_normalization
        
        # Data prep: it is different for DeepSets vs GT
        if self.model_type == "deepsets":
            # DeepSets: norm_data() returns DataFrame with N rows (customers only)
            facility_dict, big_m, rc_norm = norm_data(
                self.depot_cord, self.customer_cord, 
                self.vehicle_capacity, self.customer_demand,
                self.rc_cal_index, self.config
            )
        elif self.model_type == "graph_transformer":
            # GT: norm_data_gt() returns dict with 'df' (N+1 rows) and 'dist' matrix
            from core.dataparse import norm_data_gt # we import here to avoid circular imports
            facility_dict, rc_norm = norm_data_gt(
                self.depot_cord, self.customer_cord,
                self.vehicle_capacity, self.customer_demand,
                self.rc_cal_index, self.config
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        print(f"normalization factors (fi): {rc_norm}")

        # phi embeddings
        if self.model_type == "deepsets":
            phi_final_outputs = {}
            for j in range(self.depotno):
                phi_final_outputs[j] = extract_onnx(facility_dict[j].values, phi_loc)
            
            sz = phi_final_outputs[0].size()
            latent_space = sz[1]
            
            onnx_model = onnx.load(rho_loc)
            pytorch_rho_mdl = convert(onnx_model).double()
            layers = [layer for name, layer in pytorch_rho_mdl.named_children()]
            sequential_model = nn.Sequential(*layers)
            
        elif self.model_type == "graph_transformer":
            gt_config = self.config.get("gt_config", None)
            gt_predictor = load_gt_model(phi_loc, gt_config)
            
            phi_final_outputs = {}
            for j in range(self.depotno):
                # facility_dict[j] already has 'df' and 'dist' from norm_data_gt()
                embeddings = gt_predictor.get_embeddings(facility_dict[j])
                phi_final_outputs[j] = torch.tensor(embeddings, dtype=torch.float64)
            
            # Shape: should be (N+1, latent_dim) for GT
            expected_rows = self.customer_no + 1
            actual_rows = phi_final_outputs[0].size(0)
            print(f"[GT] Embedding shape: {phi_final_outputs[0].shape}")
            print(f"[GT] Expected rows: {expected_rows} (depot + {self.customer_no} customers)")
            assert actual_rows == expected_rows, \
                f"GT embedding should have {expected_rows} rows, got {actual_rows}"
            
            latent_space = phi_final_outputs[0].size(1)
            sequential_model = gt_predictor.model.rho.double().cpu()

        m = gp.Model('lrp')

        cartesian_prod = list(product(range(self.depotno), range(self.customer_no)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')
        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        route_cost = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route_cost')
        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_route_cost")

        if self.model_type == "deepsets":
            # DeepSets: z[j,l] = sum_i(x[j,i] * phi[j][i,l])
            # phi[j] has N rows, row i = customer i
            for j in range(self.depotno):
                for l in range(latent_space):
                    m.addConstr(
                        z[j, l] == gp.quicksum(
                            x[j, i] * phi_final_outputs[j][i, l].item() 
                            for i in range(self.customer_no)
                        ),
                        name=f'Z-plus[{j}][{l}]'
                    )
        
        elif self.model_type == "graph_transformer":
            # GT: z[j,l] = depot_embedding + sum_i(x[j,i] * customer_embedding[i])
            # phi[j] has N+1 rows: row 0 = depot, row i+1 = customer i
            for j in range(self.depotno):
                for l in range(latent_space):
                    m.addConstr(
                        z[j, l] == 
                        phi_final_outputs[j][0, l].item()  # depot at row 0
                        + gp.quicksum(
                            x[j, i] * phi_final_outputs[j][i + 1, l].item()  # customer i at row i+1
                            for i in range(self.customer_no)
                        ),
                        name=f'Z-plus[{j}][{l}]'
                    )

        # Each customer assigned to exactly one depot
        m.addConstrs(
            (gp.quicksum(x[(j, i)] for j in range(self.depotno)) == 1 
            for i in range(self.customer_no)),
            name='Demand'
        )
        
        # Depot capacity constraints
        m.addConstrs(
            (gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) 
            <= self.depot_capacity[j] * y[j]
            for j in range(self.depotno)),
            name="facility_capacity_constraint"
        )
        
        # Can only assign to open depots
        m.addConstrs(
            (x[j, i] <= y[j] for j in range(self.depotno) for i in range(self.customer_no)),
            name='Assignment_to_open_facility'
        )

        # Rho constraints
        z_values_per_depot = {}
        route_per_depot = {}
        for j in range(self.depotno):
            z_values_per_depot[j] = [z[j, l] for l in range(latent_space)]
            route_per_depot[j] = [route_cost[j]]

        print("Adding neural network constraints via GurobiML...")
        for j in range(self.depotno):
            t_const = gt.add_sequential_constr(m, sequential_model, z_values_per_depot[j], route_per_depot[j])
            t_const.print_stats()

        for j in range(self.depotno):
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            y_hat = route_per_depot[j][0]
            m.addConstr((y[j] == 1) >> (u[j] == denormalize_cost(y_hat, rc_norm[j], self.cost_normalization, self.config)))

        # Objective
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        route_obj = gp.quicksum(u[j] for j in range(self.depotno))
        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        m.setParam('MIPGAP', 0.01)
        m.setParam('TimeLimit', self.config.get("solver_timeout"))

        St_time = datetime.now()
        m.optimize()
        Ed_time = datetime.now()

        f_obj = facility_obj.getValue()
        r_obj = route_obj.getValue()
        execution_time = (Ed_time - St_time).total_seconds()

        print(f"objective value: {m.objVal}")
        print(f"facility cost: {f_obj}")
        print(f"route cost (nn): {r_obj}")
        print(f"solve time: {execution_time:.2f}s")

        y_val = [y[j].x for j in range(self.depotno)]
        x_val = [[x[j, i].x for i in range(self.customer_no)] for j in range(self.depotno)]

        return y_val, x_val, f_obj, r_obj, 0, execution_time