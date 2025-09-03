import h5py
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import math

class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def prepare_pretrain_data(file_path, split_ratios, num_entries=None, device=None):
    with h5py.File(file_path, 'r') as f:
        data_list = []
        keys = list(f.keys())[:num_entries] if num_entries else list(f.keys())

        for key in keys:
            group = f[key]
            x_coords = torch.tensor(group['x_coordinates'][:], dtype=torch.float32, device=device)
            y_coords = torch.tensor(group['y_coordinates'][:], dtype=torch.float32, device=device)
            demands = torch.tensor(group['demands'][:], dtype=torch.float32, device=device)
            is_depot = torch.tensor(group['is_depot'][:], dtype=torch.float32, device=device)
            x = torch.stack([x_coords, y_coords, is_depot, demands], dim=1)

            dist_matrix = torch.tensor(group['dist_matrix'][:], dtype=torch.float32, device=device)
            edge_index, edge_attr = dense_to_sparse(dist_matrix)
            edge_attr = edge_attr.to(torch.float32)

            label = torch.tensor(group.get('masked_cost', group.get('cost', 0.0))[()], dtype=torch.float32, device=device)
            if label.item() == 0: 
                continue

            mask = torch.tensor(group.get('mask', np.ones(x.size(0)))[:], dtype=torch.float32, device=device).unsqueeze(1)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, mask=mask))

    total_size = len(data_list)
    indices = torch.randperm(total_size)
    train_idx = indices[:int(total_size * split_ratios[0])]
    val_idx = indices[int(total_size * split_ratios[0]):int(total_size * sum(split_ratios[:2]))]
    test_idx = indices[int(total_size * sum(split_ratios[:2])):]

    return (
        CustomGraphDataset([data_list[i] for i in train_idx]),
        CustomGraphDataset([data_list[i] for i in val_idx]),
        CustomGraphDataset([data_list[i] for i in test_idx])
    )

def parse_instance(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("!")]

    trucks = list(map(int, lines[0].split(',')))
    city_freighters = list(map(int, lines[1].split(',')))
    stores = [tuple(map(float, s.split(','))) for s in lines[2].split()]
    depot, satellites = stores[0], stores[1:]

    customers, demands = [], []
    for line in lines[3:]:
        for c in line.split():
            x, y, d = map(float, c.split(','))
            customers.append((x, y))
            demands.append(d)

    return {
        "total_trucks": trucks[0],
        "truck_capacity": trucks[1],
        "max_cf_per_satellite": city_freighters[0],
        "total_city_freighters": city_freighters[1],
        "cf_capacity": city_freighters[2],
        "depot": depot,
        "satellites": satellites,
        "customers": customers,
        "customer_demands": demands
    }

def load_bks_info(bks_filename):
    bks_all = {}
    with open(bks_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            folder, instance, val = map(str.strip, line.split(","))
            try:
                val = float(val)
            except ValueError:
                continue
            bks_all.setdefault(folder, {})[instance] = val
    return bks_all

def euclid_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def normalize_coord(depot_coords, cust_coords, fi_mode='dynamic', fixed_fi_value=1000.0):
    depot_coords, cust_coords = np.array(depot_coords), np.array(cust_coords)
    fi_list, mod_coord = [], {}
    for j, dep in enumerate(depot_coords):
        shifted = cust_coords - dep
        x_list, y_list = np.concatenate((shifted[:,0],[0])), np.concatenate((shifted[:,1],[0]))
        fi = max(max(x_list)-min(x_list), max(y_list)-min(y_list)) if fi_mode=='dynamic' else fixed_fi_value
        fi_list.append(fi)
        mod_coord[j] = shifted / fi
    return mod_coord, fi_list

def norm_data(depot_coords, cust_coords, veh_cap, cust_dem):
    mod_coord, cost_norm = normalize_coord(depot_coords, cust_coords)
    cust_dem = np.array(cust_dem)
    facility_dict = {}
    for j in range(len(depot_coords)):
        coords = mod_coord[j]
        norm_dem = cust_dem / veh_cap
        x_vals = np.insert(coords[:,0],0,0)
        y_vals = np.insert(coords[:,1],0,0)
        dem_vals = np.insert(norm_dem,0,0)
        df = pd.DataFrame({'x':x_vals, 'y':y_vals, 'dem':dem_vals})
        dist_mtx = distance_matrix(np.column_stack((x_vals,y_vals)), np.column_stack((x_vals,y_vals))).astype(np.float32)
        facility_dict[j] = {'df': df, 'dist': dist_mtx}
    return facility_dict, cost_norm
