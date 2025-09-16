import math
import numpy as np
import pandas as pd

def create_data(file_loc):
    # Read all non-empty lines from the file
    with open(file_loc, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Convert lines into list of floats
    data = [list(map(float, line.split())) for line in lines]
    idx = 0

    # Number of Customers
    no_cust = int(data[idx][0])
    idx += 1

    # Number of Depots
    no_depot = int(data[idx][0])
    idx += 1

    # Depot coordinates
    depot_cord = [tuple(data[i]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    # Customer coordinates
    cust_cord = [tuple(data[i]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    # Vehicle Capacity
    vehicle_cap = int(data[idx][0])
    vehicle_cap = [vehicle_cap] * no_depot  # Assuming same capacity for all depots
    idx += 1

    # Depot capacities
    depot_cap = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    # Customer Demands
    cust_dem = [int(data[i][0]) for i in range(idx, idx + no_cust)]
    idx += no_cust

    # Opening cost of depots
    open_dep_cost = [int(data[i][0]) for i in range(idx, idx + no_depot)]
    idx += no_depot

    # Route cost
    route_cost = int(data[idx][0])
    idx += 1

    # rc_cal_index
    rc_cal_index = int(data[idx][0])

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost, rc_cal_index]

def dist_calc(cord_set1, cord_set2, rc_cal_index):
    # Convert coordinates to NumPy arrays
    cord_set1 = np.array(cord_set1, dtype=float)  # Shape: (no_depot, 2)
    cord_set2 = np.array(cord_set2, dtype=float)  # Shape: (no_cust, 2)

    # Compute pairwise differences and distances using broadcasting
    diff = cord_set1[:, np.newaxis, :] - cord_set2[np.newaxis, :, :]  # Shape: (no_depot, no_cust, 2)
    distances = np.hypot(diff[..., 0], diff[..., 1])  # Euclidean distances

    if rc_cal_index == 0:
        distances = (100 * distances).astype(int)
    else:
        distances = distances.astype(int)

    return distances  # Returns a 2D NumPy array of distances

# cord_set1 : depot_cord, cord_set2 : customer_cord
# 
def normalize_coord(cord_set1, cord_set2, rc_cal_index, fi_mode='dynamic', fixed_fi_value=1000.0):
    cord_set1 = np.array(cord_set1, dtype=float)  # Depots: Shape (no_depot, 2)
    cord_set2 = np.array(cord_set2, dtype=float)  # Customers: Shape (no_cust, 2)
    no_depot = cord_set1.shape[0]
    no_cust = cord_set2.shape[0]

    mod_coord = {}
    fi_list = []
    big_m = 0

    for j in range(no_depot):
        # Shifted coordinates for depot j
        shifted_coords = cord_set2 - cord_set1[j]  # Shape: (no_cust, 2)

        # Include depot's shifted coordinates (0, 0)
        x_list = np.concatenate((shifted_coords[:, 0], [0.0]))
        y_list = np.concatenate((shifted_coords[:, 1], [0.0]))


        if fi_mode == 'dynamic':
            # Calculate max and min values
            max_x = np.max(x_list)
            min_x = np.min(x_list)
            max_y = np.max(y_list)
            min_y = np.min(y_list)
            fi = max((max_x - min_x), (max_y - min_y))
        elif fi_mode == 'fixed':
            fi = fixed_fi_value
        else:
            raise ValueError("Invalid fi_mode. Choose 'dynamic' or 'fixed'.")

        fi_list.append(fi)

        # Normalize coordinates
        normalized_coords = shifted_coords / fi
        mod_coord[j] = normalized_coords

        # Compute distances for big_m
        dist = np.hypot(normalized_coords[:, 0], normalized_coords[:, 1])
        if rc_cal_index == 0:
            dist *= 100
        big_m += dist.sum()


    return mod_coord, big_m, fi_list

# cord_set1 : depot_cord, cord_set2 : customer_cord
def norm_data(cord_set1, cord_set2, veh_cap, cust_dem, rc_cal_index, fi_mode='dynamic', fixed_fi_value=1000.0):
    # Normalize coordinates and compute scaling factors
    mod_coord, big_m, cost_norm_factor = normalize_coord(cord_set1, cord_set2, rc_cal_index, fi_mode=fi_mode, fixed_fi_value=fixed_fi_value)
    cust_dem = np.array(cust_dem, dtype=float)

    facility_dict = {}

    for j in range(len(cord_set1)):
        # Normalized customer demands
        norm_cust_dem = cust_dem / veh_cap[j]

        # Get normalized coordinates for depot j
        coords = mod_coord[j]

        is_depot = np.zeros(coords.shape[0], dtype=np.float32)
        is_depot[0] = 1.0

        norm_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'is_depot' : is_depot,
            'dem': norm_cust_dem
        })
        facility_dict[j] = norm_df

    return facility_dict, big_m, cost_norm_factor