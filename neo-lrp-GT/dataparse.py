import math
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def create_data(file_loc: str):
    """
    Reads and parses the input file.
    Returns:
        List containing number of customers, depots, coordinates, capacities, demands, costs, etc.
    """
    with open(file_loc, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = [list(map(float, line.split())) for line in lines]
    idx = 0

    # Number of customers and depots
    no_cust = int(data[idx][0])
    idx += 1
    no_depot = int(data[idx][0])
    idx += 1

    # Depot and customer coordinates
    depot_coords = [tuple(data[i]) for i in range(idx, idx + no_depot)]
    idx += no_depot
    cust_coords = [tuple(data[i]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    # Vehicle and depot capacities
    vehicle_cap = [int(data[idx][0])] * no_depot
    idx += 1
    depot_cap = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    # Customer demands
    cust_dem = [int(data[i][0]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    # Depot opening costs
    open_dep_cost = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    # Route cost and rc_cal_index
    route_cost = int(data[idx][0])
    idx += 1
    rc_cal_index = int(data[idx][0])

    return [
        no_cust, no_depot, depot_coords, cust_coords, vehicle_cap,
        depot_cap, cust_dem, open_dep_cost, route_cost, rc_cal_index
    ]


def dist_calc(cord_set1, cord_set2, rc_cal_index):
    """
    Computes pairwise Euclidean distances between depots and customers.
    """
    cord_set1 = np.array(cord_set1, dtype=float)
    cord_set2 = np.array(cord_set2, dtype=float)

    diff = cord_set1[:, np.newaxis, :] - cord_set2[np.newaxis, :, :]
    distances = np.hypot(diff[..., 0], diff[..., 1])

    if rc_cal_index == 0:
        distances = (100 * distances).astype(int)
    else:
        distances = distances.astype(int)

    return distances


def normalize_coord(cord_set1, cord_set2, fi_mode='dynamic', fixed_fi_value=1000.0):
    """
    Normalize customer coordinates relative to each depot.
    Returns:
        mod_coord: dict mapping depot index to normalized customer coordinates
        fi_list: list of normalization factors for each depot
    """
    cord_set1 = np.array(cord_set1, dtype=float)
    cord_set2 = np.array(cord_set2, dtype=float)

    mod_coord = {}
    fi_list = []

    for j, depot in enumerate(cord_set1):
        shifted_coords = cord_set2 - depot

        # Include depot's own coordinates (0,0)
        x_vals = np.append(shifted_coords[:, 0], 0.0)
        y_vals = np.append(shifted_coords[:, 1], 0.0)

        if fi_mode == 'dynamic':
            fi = max(np.ptp(x_vals), np.ptp(y_vals))
        elif fi_mode == 'fixed':
            fi = fixed_fi_value
        else:
            raise ValueError("Invalid fi_mode. Choose 'dynamic' or 'fixed'.")

        fi_list.append(fi)
        mod_coord[j] = shifted_coords / fi

    return mod_coord, fi_list


def norm_data(cord_set1, cord_set2, veh_cap, cust_dem):
    """
    Generates normalized dataframes and distance matrices for each depot.
    """
    mod_coord, cost_norm_factor = normalize_coord(cord_set1, cord_set2)
    cust_dem = np.array(cust_dem, dtype=float)
    facility_dict = {}

    for j in range(len(cord_set1)):
        norm_cust_dem = cust_dem / veh_cap[j]
        coords = mod_coord[j]

        x_vals = np.insert(coords[:, 0], 0, 0)
        y_vals = np.insert(coords[:, 1], 0, 0)
        dem_vals = np.insert(norm_cust_dem, 0, 0)

        norm_df = pd.DataFrame({'x': x_vals, 'y': y_vals, 'dem': dem_vals})

        new_coords = np.column_stack((x_vals, y_vals))
        dist_mtx = distance_matrix(new_coords, new_coords).astype(np.float32)

        facility_dict[j] = {'df': norm_df, 'dist': dist_mtx}

    return facility_dict, cost_norm_factor
