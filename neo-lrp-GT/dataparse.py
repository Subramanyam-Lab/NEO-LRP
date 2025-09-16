"""
Data parsing utilities for Location-Routing Problems
Handles reading and processing of LRP instance files
"""

import math
import numpy as np
import pandas as pd

def create_data(file_loc):
    """
    Parse LRP instance file and extract problem data.
    
    Args:
        file_loc (str): Path to the LRP instance file
        
    Returns:
        list: Problem data containing:
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

    # Vehicle Capacity (assume same capacity for all depots)
    vehicle_cap = int(data[idx][0])
    vehicle_cap = [vehicle_cap] * no_depot
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
    """
    Calculate pairwise distances between depot and customer coordinates.
    
    Args:
        cord_set1: Depot coordinates
        cord_set2: Customer coordinates
        rc_cal_index: Route cost calculation index
        
    Returns:
        np.ndarray: 2D array of distances
    """
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

    return distances

def normalize_coord(cord_set1, cord_set2, fi_mode='dynamic', fixed_fi_value=1000.0):
    """
    Normalize coordinates for neural network processing.
    
    Args:
        cord_set1: Depot coordinates
        cord_set2: Customer coordinates
        fi_mode: Normalization mode ('dynamic' or 'fixed')
        fixed_fi_value: Fixed normalization value when fi_mode='fixed'
        
    Returns:
        tuple: (mod_coord, fi_list) where mod_coord contains normalized coordinates
               and fi_list contains normalization factors
    """
    cord_set1 = np.array(cord_set1, dtype=float)  # Depots: Shape (no_depot, 2)
    cord_set2 = np.array(cord_set2, dtype=float)  # Customers: Shape (no_cust, 2)
    no_depot = cord_set1.shape[0]
    no_cust = cord_set2.shape[0]

    mod_coord = {}
    fi_list = []

    for j in range(no_depot):
        # Shift coordinates relative to depot j
        shifted_coords = cord_set2 - cord_set1[j]  # Shape: (no_cust, 2)

        # Include depot's shifted coordinates (0, 0)
        x_list = np.concatenate((shifted_coords[:, 0], [0.0]))
        y_list = np.concatenate((shifted_coords[:, 1], [0.0]))

        if fi_mode == 'dynamic':
            # Calculate normalization factor based on coordinate range
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


    return mod_coord, fi_list


from scipy.spatial import distance_matrix

def norm_data(cord_set1, cord_set2, veh_cap, cust_dem):
    """
    Normalize data for neural network processing.
    
    Args:
        cord_set1: Depot coordinates
        cord_set2: Customer coordinates
        veh_cap: Vehicle capacity
        cust_dem: Customer demands
        
    Returns:
        tuple: (facility_dict, cost_norm_factor) where facility_dict contains
               normalized data for each depot and cost_norm_factor contains
               normalization factors
    """
    mod_coord, cost_norm_factor = normalize_coord(cord_set1, cord_set2)
    cust_dem = np.array(cust_dem, dtype=float)
    facility_dict = {}

    for j in range(len(cord_set1)):
        # Normalize customer demands by vehicle capacity
        norm_cust_dem = cust_dem / veh_cap[j]
        coords = mod_coord[j]

        # Add depot coordinates (0, 0) at the beginning
        x_vals = np.insert(coords[:, 0], 0, 0)  # Add 0 for depot x-coordinate
        y_vals = np.insert(coords[:, 1], 0, 0)  # Add 0 for depot y-coordinate
        dem_vals = np.insert(norm_cust_dem, 0, 0)  # Add 0 for depot demand

        # Create DataFrame for neural network input
        norm_df = pd.DataFrame({
            'x': x_vals,
            'y': y_vals,
            'dem': dem_vals
        })

        # Calculate distance matrix
        new_coords = np.column_stack((x_vals, y_vals))
        dist_mtx = distance_matrix(new_coords, new_coords).astype(np.float32)

        facility_dict[j] = {
            'df': norm_df,      # Customer data (x, y, demand)
            'dist': dist_mtx    # Distance matrix
        }

    return facility_dict, cost_norm_factor
