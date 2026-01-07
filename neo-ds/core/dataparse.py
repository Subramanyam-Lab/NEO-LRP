import json
import math
import numpy as np
import pandas as pd


def load_config(config_path):
    """Load dataset configuration from JSON file."""
    import os
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config_dir = os.path.dirname(os.path.abspath(config_path))
    norm_file = config.get("normalization_file")
    if norm_file and not os.path.isabs(norm_file):
        config["normalization_file"] = os.path.join(config_dir, norm_file)
    
    print(f"[debug] config loaded from: {config_path}")
    print(f"[debug] normalization file resolved to: {config.get('normalization_file')}")
    
    return config


def get_distance_scaling(rc_cal_index):
    """rc_cal_index = 0 means integer costs (x100), rc_cal_index = 1 means float costs."""
    return 100 if rc_cal_index == 0 else 1


def create_data(file_loc, config=None):
    """
    Load instance data from file. Supports both .dat and .json formats.

    Returns:
        List: [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap,
               cust_dem, open_dep_cost, route_cost, rc_cal_index]
    """
    if config is not None:
        data_format = config.get("data_format", "dat")
    else:
        data_format = "json" if file_loc.endswith(".json") else "dat"

    if data_format == "json":
        return _create_data_json(file_loc)
    else:
        return _create_data_dat(file_loc)


def _create_data_dat(file_loc):
    """Parse .dat format files (B_barreto, T_tuzun, P_prodhon style)."""
    with open(file_loc, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = [list(map(float, line.split())) for line in lines]
    idx = 0

    no_cust = int(data[idx][0])
    idx += 1

    no_depot = int(data[idx][0])
    idx += 1

    depot_cord = [tuple(data[i]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    cust_cord = [tuple(data[i]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    vehicle_cap = int(data[idx][0])
    vehicle_cap = [vehicle_cap] * no_depot
    idx += 1

    depot_cap = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    cust_dem = [int(data[i][0]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    open_dep_cost = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    route_cost = int(data[idx][0])
    idx += 1

    rc_cal_index = int(data[idx][0])

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap,
            cust_dem, open_dep_cost, route_cost, rc_cal_index]


def _create_data_json(file_loc):
    """
    Parse JSON format files (S_schneider style).
    
    Schneider instances follow Prodhon structure:
    - Distance = int(100 * euclidean_distance)
    - rc_cal_index = 0 (integer costs)
    """
    with open(file_loc, "r") as f:
        data = json.load(f)

    no_cust = len(data["customers"])
    no_depot = len(data["depots"])

    depot_cord = [(d["x"], d["y"]) for d in data["depots"]]
    cust_cord = [(c["x"], c["y"]) for c in data["customers"]]

    vehicle_cap = [int(data["vehicle_capacity"])] * no_depot
    depot_cap = [int(d["capacity"]) for d in data["depots"]]
    cust_dem = [int(c["demand"]) for c in data["customers"]]
    open_dep_cost = [int(d["costs"]) for d in data["depots"]]

    # Fixed cost per route from JSON
    route_cost = int(data.get("vehicle_costs", 0))
    
    rc_cal_index = 0

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap,
            cust_dem, open_dep_cost, route_cost, rc_cal_index]


def normalize_coord(cord_set1, cord_set2):
    """
    Normalize customer coordinates with respect to each depot.
    
    This is purely geometric normalization for NN input features.
    Distance scaling (x100 vs x1) is not applied here.
    """
    cord_set1 = np.array(cord_set1, dtype=float)
    cord_set2 = np.array(cord_set2, dtype=float)
    no_depot = cord_set1.shape[0]

    mod_coord = {}
    fi_list = []

    for j in range(no_depot):
        shifted_coords = cord_set2 - cord_set1[j]

        x_list = np.concatenate((shifted_coords[:, 0], [0.0]))
        y_list = np.concatenate((shifted_coords[:, 1], [0.0]))

        fi = max(np.ptp(x_list), np.ptp(y_list))
        fi_list.append(fi)

        normalized_coords = shifted_coords / fi
        mod_coord[j] = normalized_coords

    return mod_coord, fi_list


def norm_data(cord_set1, cord_set2, veh_cap, cust_dem, rc_cal_index, config=None):
    """
    Return normalized dataframes and scaling factors for each depot.

    Returns:
        Tuple: (facility_dict, big_m, fi_list)
        - facility_dict: dict of DataFrames with normalized x, y, demand
        - big_m: 0 (unused, kept for compatibility)
        - fi_list: scaling factors for each depot (needed for denormalization)
    """
    mod_coord, fi_list = normalize_coord(cord_set1, cord_set2)
    cust_dem = np.array(cust_dem, dtype=float)

    facility_dict = {}
    for j in range(len(cord_set1)):
        norm_cust_dem = cust_dem / veh_cap[j]
        coords = mod_coord[j]
        norm_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'dem': norm_cust_dem
        })
        facility_dict[j] = norm_df

    return facility_dict, 0.0, fi_list