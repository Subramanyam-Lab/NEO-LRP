"""
Utility functions for Graph Transformer data loading and instance parsing.
Handles HDF5 data loading for PyTorch Geometric, LRP instance parsing,
coordinate normalization and BKS loading for evaluation.
"""

import h5py
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
import torch
from tqdm.auto import tqdm
import openpyxl
from openpyxl import Workbook
import os
import time
import math
import pandas as pd
from scipy.spatial import distance_matrix

class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def prepare_pretrain_data(file_path, split_ratios, num_entries=None, device=None, k=None):
    with h5py.File(file_path, 'r') as f:
        data_list = []
        keys = list(f.keys())

        if num_entries is not None:
            keys = keys[:num_entries]

        for key in tqdm(keys, desc="Loading pretrain instances"):
            group = f[key]

            # Load basic features
            x_coordinates = torch.tensor(group['x_coordinates'][:], dtype=torch.float32, device=device)
            y_coordinates = torch.tensor(group['y_coordinates'][:], dtype=torch.float32, device=device)
            demands = torch.tensor(group['demands'][:], dtype=torch.float32, device=device)
            is_depot = torch.tensor(group['is_depot'][:], dtype=torch.float32, device=device)
            # Concatenate features: [x, y, is_depot, demand]
            x = torch.stack([x_coordinates, y_coordinates, is_depot, demands], dim=1)

            # Load dist_matrix and convert to sparse edge_index and edge_attr
            dist_matrix = torch.tensor(group['dist_matrix'][:], dtype=torch.float32, device=device)
            edge_index, edge_attr = dense_to_sparse(dist_matrix)
            edge_attr = edge_attr.to(torch.float32)

            # Modified: Load masked_cost as target
            if 'masked_cost' in group:
                label = torch.tensor(group['masked_cost'][()], dtype=torch.float32, device=device)
            else:
                label = torch.tensor(group['cost'][()], dtype=torch.float32, device=device)
            # Modified: Load mask if available, otherwise use all ones
            if 'mask' in group:
                mask = torch.tensor(group['mask'][:], dtype=torch.float32, device=device).unsqueeze(1)
            else:
                mask = torch.ones(x.size(0), 1, device=device)

            if label.item() == 0:
                continue

            # Create Data object with extra attribute 'mask'
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, mask=mask)
            data_list.append(data)

    total_size = len(data_list)
    indices = torch.randperm(total_size)
    train_idx = indices[:int(total_size * split_ratios[0])]
    val_idx = indices[int(total_size * split_ratios[0]):int(total_size * (split_ratios[0] + split_ratios[1]))]
    test_idx = indices[int(total_size * (split_ratios[0] + split_ratios[1])):]

    train_data = CustomGraphDataset([data_list[idx] for idx in train_idx])
    val_data = CustomGraphDataset([data_list[idx] for idx in val_idx])
    test_data = CustomGraphDataset([data_list[idx] for idx in test_idx])

    return train_data, val_data, test_data

def parse_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.startswith("!")]

    trucks = list(map(int, lines[0].split(',')))
    n_trucks = trucks[0]
    truck_capacity = trucks[1]

    city_freighters = list(map(int, lines[1].split(',')))
    max_cf_per_satellite = city_freighters[0]
    total_city_freighters = city_freighters[1]
    cf_capacity = city_freighters[2]

    stores = list(map(str.strip, lines[2].split()))
    depot = tuple(map(float, stores[0].split(',')))
    satellites = [tuple(map(float, coord.split(','))) for coord in stores[1:]]

    customers = []
    customer_demands = []
    for customer_data in lines[3:]:
        for coord in customer_data.split():
            x, y, demand = map(float, coord.split(','))
            customers.append((x, y))
            customer_demands.append(demand)

    return {
        "total_trucks": n_trucks,
        "truck_capacity": truck_capacity,
        "max_cf_per_satellite": max_cf_per_satellite,
        "total_city_freighters": total_city_freighters,
        "cf_capacity": cf_capacity,
        "depot": depot,
        "satellites": satellites,
        "customers": customers,
        "customer_demands": customer_demands,
    }


def load_bks_info(bks_filename):
    """
    bks_filename: Path to text file containing BKS information.
      File format: Each line contains "folder,instance,BKS" (comma-separated, ignore lines starting with '#')
    Returns: bks_all: { folder1: { instance_name: bks_value, ... }, folder2: {...}, ... }
    """
    bks_all = {}
    with open(bks_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            folder, instance, bks_str = parts[0], parts[1], parts[2]
            try:
                bks_val = float(bks_str)
            except ValueError:
                continue
            if folder not in bks_all:
                bks_all[folder] = {}
            bks_all[folder][instance] = bks_val
    return bks_all


def euclid_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def normalize_coord(cord_set1, cord_set2, fi_mode='dynamic', fixed_fi_value=1000.0):
    cord_set1 = np.array(cord_set1, dtype=float)
    cord_set2 = np.array(cord_set2, dtype=float)
    no_depot = cord_set1.shape[0]

    mod_coord = {}
    fi_list = []

    for j in range(no_depot):
        shifted_coords = cord_set2 - cord_set1[j]
        x_list = np.concatenate((shifted_coords[:, 0], [0.0]))
        y_list = np.concatenate((shifted_coords[:, 1], [0.0]))

        if fi_mode == 'dynamic':
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
        normalized_coords = shifted_coords / fi
        mod_coord[j] = normalized_coords

    return mod_coord, fi_list

def norm_data(cord_set1, cord_set2, veh_cap, cust_dem):
    mod_coord, cost_norm_factor = normalize_coord(cord_set1, cord_set2)
    cust_dem = np.array(cust_dem, dtype=float)
    facility_dict = {}

    for j in range(len(cord_set1)):
        norm_cust_dem = cust_dem / veh_cap
        coords = mod_coord[j]

        # Add 0 at the beginning of each column
        x_vals = np.insert(coords[:, 0], 0, 0)  # Add 0 at the beginning of x coordinates
        y_vals = np.insert(coords[:, 1], 0, 0)  # Add 0 at the beginning of y coordinates
        dem_vals = np.insert(norm_cust_dem, 0, 0)  # Add 0 at the beginning of demands

        # Create DataFrame
        norm_df = pd.DataFrame({
            'x': x_vals,
            'y': y_vals,
            'dem': dem_vals
        })

        new_coords = np.column_stack((x_vals, y_vals))
        dist_mtx = distance_matrix(new_coords, new_coords).astype(np.float32)

        facility_dict[j] = {
            'df': norm_df,      # Per customer (x, y, dem)
            'dist': dist_mtx    # Distance matrix between customers
        }

    return facility_dict, cost_norm_factor
