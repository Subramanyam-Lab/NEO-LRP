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
from core.network import extract_onnx


def denormalize_cost(y_hat, fi, mode, config):
    """
    De-normalizes model output based on the normalization mode used during training.
    """
    norm_file = config.get("normalization_file")
    with open(norm_file, "r") as f:
        norm_data_dict = json.load(f)

    keys = config.get("normalization_keys", {})
    extra_modes = config.get("extra_normalization_modes", [])

    if mode == "raw":
        return y_hat

    elif mode == "minmax":
        cmin = norm_data_dict[keys["minmax_min"]]
        cmax = norm_data_dict[keys["minmax_max"]]
        return y_hat * (cmax - cmin) + cmin

    elif mode == "cost_over_fi" or mode in extra_modes:
        return y_hat * fi

    elif mode == "cost_over_fi_minmax":
        cmin = norm_data_dict[keys["cost_over_fi_minmax_min"]]
        cmax = norm_data_dict[keys["cost_over_fi_minmax_max"]]
        return (y_hat * (cmax - cmin) + cmin) * fi

    else:
        raise ValueError(f"Unknown normalization mode '{mode}'")


class createLRP():
    def __init__(self, ans, cost_normalization, config=None):
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
        self.init_route_cost = ans[8]
        self.rc_cal_index = ans[9]
        self.USE_WARMSTART = False
        self.cost_normalization = cost_normalization
        self.config = config if config is not None else {}

    def dataprocess(self, data_input_file):
        facility_dict, big_m, rc_norm = norm_data(
            self.depot_cord, self.customer_cord, self.vehicle_capacity,
            self.customer_demand, self.rc_cal_index, self.config
        )
        print(f"normalization factor for route cost {rc_norm}")
        initial_flp_assignment = (None, {}, {})
        print("flp skipped (warmstart disabled).")
        return facility_dict, big_m, rc_norm, initial_flp_assignment

    def model(self, loc, log_filename, phi_loc, rho_loc, cost_normalization):
        facility_dict, big_m, rc_norm, initial_flp_assignment = self.dataprocess(loc)

        phi_final_outputs = {}
        for j in range(self.depotno):
            phi_final_outputs[j] = extract_onnx(facility_dict[j].values, phi_loc)

        sz = phi_final_outputs[0].size()
        latent_space = sz[1]

        ws_time = 0.0
        print("running without warm start.")

        m = gp.Model('clrp')

        cartesian_prod = list(product(range(self.depotno), range(self.customer_no)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')
        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        route_cost = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route_cost')
        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy_route_cost")

        for j in range(self.depotno):
            for l in range(latent_space):
                m.addConstr(
                    z[j, l] == gp.quicksum(x[j, i] * phi_final_outputs[j][i, l] for i in range(self.customer_no)),
                    name=f'Z-plus[{j}][{l}]'
                )

        m.addConstrs(
            (gp.quicksum(x[(j, i)] for j in range(self.depotno)) == 1 for i in range(self.customer_no)),
            name='Demand'
        )
        m.addConstrs(
            (gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) <= self.depot_capacity[j] * y[j]
             for j in range(self.depotno)),
            name="facility_capacity_constraint"
        )
        m.addConstrs(
            (x[j, i] <= y[j] for j in range(self.depotno) for i in range(self.customer_no)),
            name='Assignment_to_open_facility'
        )

        St_time = datetime.now()

        onnx_model = onnx.load(rho_loc)
        pytorch_rho_mdl = convert(onnx_model).double()
        layers = [layer for name, layer in pytorch_rho_mdl.named_children()]
        sequential_model = nn.Sequential(*layers)

        z_values_per_depot = {}
        route_per_depot = {}
        for j in range(self.depotno):
            z_values_per_depot[j] = [z[j, l] for l in range(latent_space)]
            route_per_depot[j] = [route_cost[j]]

        for j in range(self.depotno):
            t_const = gt.add_sequential_constr(m, sequential_model, z_values_per_depot[j], route_per_depot[j])
            t_const.print_stats()

        for j in range(self.depotno):
            print(f"depot {j} rc_norm[j] = {rc_norm[j]} (type={type(rc_norm[j])})")
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            y_hat = route_per_depot[j][0]
            m.addConstr((y[j] == 1) >> (u[j] == denormalize_cost(y_hat, rc_norm[j], self.cost_normalization, self.config)))

        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        route_obj = gp.quicksum(u[j] for j in range(self.depotno))
        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        m.setParam('MIPGAP', 0.01)
        m.setParam('TimeLimit', self.config.get("solver_timeout"))

        St_time1 = datetime.now()
        m.optimize()
        Ed_time = datetime.now()

        f_obj = facility_obj.getValue()
        r_obj = route_obj.getValue()
        execution_time1 = (Ed_time - St_time1).total_seconds()

        print(f"objective value is {m.objVal}")
        print(f'facility objective value: {f_obj}')
        print(f'route objective value: {r_obj}')
        print(f"clrp model execution time: {execution_time1}")

        y_val = [y[j].x for j in range(self.depotno)]
        x_val = [[x[j, i].x for i in range(self.customer_no)] for j in range(self.depotno)]

        return y_val, x_val, f_obj, r_obj, ws_time, execution_time1