"""
Data preprocessing for DeepSets training.
Parses CVRP instance files, normalizes coordinates and costs, handles padding,
and exports to NumPy arrays (used in training DS). Supports multiple normalization modes.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import json
import h5py

DEPOT_INDICATOR = False
HB5_EXPORT      = False
PADDING         = True

print("Using updated data preprocessing updated...")
def parse_numeric_value(s):
    s = s.strip()
    match = re.match(r'^([0-9.+-eE]+)', s)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Cannot parse numeric value from '{s}'")

class InstanceNew:
    def __init__(self, instance_str, max_len, solver_tag,
                 depot_indicator=True, hb5_export=True, padding=True):

        problem_str, metadata_str = instance_str.split("EOF\n", 1)
        problem_lines = problem_str.strip().split("\n")
        metadata_lines = metadata_str.strip().split("\n")
        name_line = next(line for line in instance_str.split('\n') if line.startswith('NAME :'))
        full_name = name_line.split(':')[1].strip()

        match = re.search(r'(XML\d+_\d+_\d+)', full_name)
        if match:
            self.name = match.group(1)
        else:
            print(f"Warning: unusual NAME format: '{full_name}', using full string.")
            self.name = full_name

        self.capacity = float(next(line.split(":")[1].strip() for line in problem_lines if line.startswith("CAPACITY :")))


        node_coord_start = next(i for i, line in enumerate(problem_lines) if line.startswith("NODE_COORD_SECTION")) + 1
        demand_start = next(i for i, line in enumerate(problem_lines) if line.startswith("DEMAND_SECTION"))

        self.customers, first_line = [], True
        for line in problem_lines[node_coord_start:demand_start]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            idx, x, y = parts
            x, y = float(x), float(y)
            if first_line:
                self.depot = {'x': x, 'y': y, 'idx': int(idx)}
                first_line = False
            else:
                self.customers.append({'x': x, 'y': y, 'idx': int(idx)})

        for line in problem_lines[demand_start + 1:]:
            if line.startswith("DEPOT_SECTION"):
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            index, demand = int(parts[0]), float(parts[1])
            if index == self.depot['idx']:
                continue
            self.customers[index - 2]['demand'] = demand / self.capacity

        self.metadata = {}
        for line in metadata_lines:
            if line.startswith("#"):
                parts = line[1:].strip().split(" ", 1)
                if len(parts) == 2:
                    self.metadata[parts[0]] = parts[1]

        cost_key = f"cost_{solver_tag}"
        num_routes_key = f"num_routes_{solver_tag}"
        solve_time_key = f"solve_time_{solver_tag}"
        actual_routes_key = f"actual_routes_{solver_tag}"

        if cost_key not in self.metadata:
            raise ValueError(f"Missing {cost_key} in metadata for {self.name}")

        self.cost = parse_numeric_value(self.metadata.get(cost_key))
        self.num_routes = int(self.metadata.get(num_routes_key))

        solve_time_val = self.metadata.get(solve_time_key).rstrip("s")
        self.solve_time = parse_numeric_value(solve_time_val)
        self.actual_routes = self.metadata.get(actual_routes_key, "[]")

        x_all = [self.depot['x']] + [c['x'] for c in self.customers]
        y_all = [self.depot['y']] + [c['y'] for c in self.customers]
        x_shifted = np.array(x_all) - self.depot['x']
        y_shifted = np.array(y_all) - self.depot['y']
        fi = max(np.max(x_shifted) - np.min(x_shifted),
                 np.max(y_shifted) - np.min(y_shifted))
        if fi < 1e-9:
            fi = 1.0

        for c in self.customers:
            c['x'] = (c['x'] - self.depot['x']) / fi
            c['y'] = (c['y'] - self.depot['y']) / fi

        if hb5_export:
            x_coords = [0.0] + [c['x'] for c in self.customers]
            y_coords = [0.0] + [c['y'] for c in self.customers]
            demands  = [0.0] + [c['demand'] for c in self.customers]
            mask = [1] * len(x_coords)
            is_depot = [1] + [0 for _ in self.customers] if depot_indicator else []

            if padding:
                pad_len = (max_len + 1) - len(x_coords)
                for _ in range(pad_len):
                    x_coords.append(0.0)
                    y_coords.append(0.0)
                    demands.append(0.0)
                    mask.append(0)
                    if depot_indicator: is_depot.append(0)

            from scipy.spatial import distance_matrix
            coords_2d = np.column_stack((x_coords, y_coords))
            dist_mtx = distance_matrix(coords_2d, coords_2d).astype(np.float32)

            self.training_data = {
                "x_coordinates": np.array(x_coords, dtype=np.float32),
                "y_coordinates": np.array(y_coords, dtype=np.float32),
                "demands": np.array(demands, dtype=np.float32),
                "mask": np.array(mask, dtype=np.float32),
                "cost": float(self.cost),
                "fi": float(fi),
                "used_vehicles": self.num_routes,
                "dist_matrix": dist_mtx,
            }
            if depot_indicator:
                self.training_data["is_depot"] = np.array(is_depot, dtype=np.float32)

        else:
            training_customers = self.customers.copy()
            if padding:
                pad_len = max_len - len(training_customers)
                if pad_len > 0:
                    special_pad = {'x': -1.0e+04, 'y': -1.0e+04, 'demand': -1.0e+04}
                    zero_pad = {'x': 0.0, 'y': 0.0, 'demand': 0.0}
                    training_customers.append(special_pad)
                    for _ in range(pad_len - 1):
                        training_customers.append(zero_pad)

            self.training_data = {
                'x': [c['x'] for c in training_customers],
                'y': [c['y'] for c in training_customers],
                'demand': [c.get('demand') for c in training_customers],
                'fi': fi,
                'cost': self.cost
            }


def normalize_costs(data_list, normalization_mode, existing_stats=None):

    costs = np.array([d["cost"] for d in data_list])
    fis = np.array([d["fi"] for d in data_list])  
    stats = {}

    if normalization_mode == "cost_over_fi":
        for d in data_list:
            d["cost"] = d["cost"] / d["fi"]
            d.pop("fi", None)
        return None

    elif normalization_mode == "cost_over_fi_minmax":
        if existing_stats is None:
            scaled = costs / fis
            stats["min"], stats["max"] = scaled.min(), scaled.max()
        else:
            stats = existing_stats

        if stats["max"] == stats["min"]:
            raise ValueError(f"{normalization_mode} normalization has zero range (all values identical).")

        for d in data_list:
            c_scaled = d["cost"] / d["fi"]
            d["cost"] = (c_scaled - stats["min"]) / (stats["max"] - stats["min"])
            d.pop("fi", None)

    elif normalization_mode == "minmax":
        if existing_stats is None:
            stats["min"], stats["max"] = costs.min(), costs.max()
        else:
            stats = existing_stats

        if stats["max"] == stats["min"]:
            raise ValueError(f"{normalization_mode} normalization has zero range (all values identical).")

        for d in data_list:
            d["cost"] = (d["cost"] - stats["min"]) / (stats["max"] - stats["min"])
            d.pop("fi", None)

    elif normalization_mode == "raw":
        for d in data_list:
            d.pop("fi", None)
        return None

    else:
        raise ValueError(f"Unknown normalization mode '{normalization_mode}'")

    return stats


def preprocess_data(file_path_train_val, file_path_test, solver_tag,
                    num_instances=None, seed=42,
                    normalization_mode="cost_over_fi_minmax",
                    norm_json_path=None):

    random.seed(seed)

    def read_instances(file_path):
        instances = []
        instance_lines = []
        with open(file_path, 'r') as file:
            for line in file:
                instance_lines.append(line)
                if line.strip() == '#EOF':
                    instance_str = ''.join(instance_lines)
                    instances.append(instance_str)
                    instance_lines = []
        return instances

    train_val_instances = read_instances(file_path_train_val)
    print(f"Total instances read from training/validation file: {len(train_val_instances)}")

    if num_instances is not None:
        train_val_instances = random.sample(train_val_instances, min(num_instances, len(train_val_instances)))

    test_instances = read_instances(file_path_test)
    print(f"Total instances read from test file: {len(test_instances)}")

    # Combine all instances to compute the maximum length
    all_instances = train_val_instances + test_instances

    num_discarded_maxlen = 0
    num_discarded_train_val = 0
    num_discarded_test = 0

    # Compute max_len across all instances to ensure consistent padding
    max_len = 0
    for instance_str in all_instances:
        try:

            instance = InstanceNew(
                instance_str,
                max_len=0,
                solver_tag=solver_tag,
                depot_indicator=DEPOT_INDICATOR,
                hb5_export=HB5_EXPORT,
                padding=PADDING
            )

            max_len = max(max_len, len(instance.customers))
        except ValueError as e:
            num_discarded_maxlen += 1
            print(f"Discarding instance during max_len computation: {e}")
            continue

    print(f"Number of discarded instances during max_len computation: {num_discarded_maxlen}")
    print(f"Computed max_len: {max_len}")

    training_data = []
    for instance_str in train_val_instances:
        try:
            instance = InstanceNew(
                instance_str,
                max_len=max_len,
                solver_tag=solver_tag,
                depot_indicator=DEPOT_INDICATOR,
                hb5_export=HB5_EXPORT,
                padding=PADDING
            )
            training_data.append(instance.training_data)
        except ValueError as e:
            num_discarded_train_val += 1
            print(f"Discarding instance during training data collection: {e}")
            continue

    print(f"Data preprocessing completed for training/validation data. Total instances selected: {len(training_data)}")
    print(f"Number of discarded training/validation instances: {num_discarded_train_val}")

    if len(training_data) == 0:
        raise ValueError("No valid training/validation instances were processed.")

    if norm_json_path is not None:
        norm_filename = norm_json_path
    else:
        norm_filename = f"label_range_{solver_tag.replace('vroom_', '')}.json"
    
    print(f"Applying cost normalization mode: {normalization_mode}")
    print(f"Looking for normalization file: {norm_filename}")

    if os.path.exists(norm_filename):
        with open(norm_filename, "r") as f:
            all_stats = json.load(f)
        print(f"Loaded normalization file: {norm_filename}")
    else:
        all_stats = {}
        print(f"Normalization file not found will compute new stats")

    # keys match JSON format: eg "minmax_min_vroom_scaled"
    key_min = f"{normalization_mode}_min_{solver_tag}"
    key_max = f"{normalization_mode}_max_{solver_tag}"

    if key_min in all_stats and key_max in all_stats:
        norm_stats = {"min": all_stats[key_min], "max": all_stats[key_max]}
        print(f"Loaded normalization constants from {norm_filename}: {norm_stats}")
    else:
        norm_stats = None
        print(f"No existing normalization constants found for mode '{normalization_mode}', will compute new ones.")

    computed_stats = normalize_costs(training_data, normalization_mode, existing_stats=norm_stats)

    if computed_stats is not None and norm_stats is None:
            all_stats[key_min] = computed_stats["min"]
            all_stats[key_max] = computed_stats["max"]
            if norm_json_path is None:
                with open(norm_filename, "w") as f:
                    json.dump(all_stats, f, indent=2)
                print(f"Saved normalization constants: {computed_stats} -> {norm_filename}")
            else:
                print(f"Computed normalization constants: {computed_stats} (not saving using provided JSON)")

    if HB5_EXPORT:
        test_data = []
        for instance_str in test_instances:
            try:
                instance = InstanceNew(
                    instance_str,
                    max_len=max_len,
                    solver_tag=solver_tag,
                    depot_indicator=DEPOT_INDICATOR,
                    hb5_export=HB5_EXPORT,
                    padding=PADDING
                )
                test_data.append(instance.training_data)
            except ValueError as e:
                num_discarded_test += 1
                print(f"Discarding test instance: {e}")

        print(f"Number of discarded test instances: {num_discarded_test}")

        if computed_stats is not None:
            normalize_costs(test_data, normalization_mode, existing_stats=computed_stats)
        else:
            normalize_costs(test_data, normalization_mode)

        all_data = training_data + test_data
        h5_filename = f"exported_{solver_tag}_{normalization_mode}.h5"
        with h5py.File(h5_filename, 'w') as hf:
            for i, data in enumerate(all_data):
                group_name = f"instance_{i}"
                grp = hf.create_group(group_name)
                for k, v in data.items():
                    arr = np.array(v)

                    if np.issubdtype(arr.dtype, np.floating):
                        arr = arr.astype(np.float32)  
                    elif np.issubdtype(arr.dtype, np.integer):
                        arr = arr.astype(np.int32)     

                    grp.create_dataset(k, data=arr)
        print(f"HB5 export: saved {len(all_data)} instances to {h5_filename}")
        return h5_filename
    else:
        test_data = []
        for instance_str in test_instances:
            try:
                instance = InstanceNew(
                    instance_str,
                    max_len=max_len,
                    solver_tag=solver_tag,
                    depot_indicator=DEPOT_INDICATOR,
                    hb5_export=HB5_EXPORT,
                    padding=PADDING
                )
                test_data.append(instance.training_data)
            except ValueError as e:
                num_discarded_test += 1
                print(f"Discarding test instance: {e}")

        print(f"Number of discarded test instances: {num_discarded_test}")

        training_data = pd.DataFrame(training_data, columns=["x", "y", "demand", "cost"])
        training_data_np = np.vstack(training_data.apply(lambda x: np.array(list(x), dtype=object), axis=1))
        X = training_data_np[:, :-1]
        Y = training_data_np[:, -1:]

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

        X_train = np.array([np.column_stack(instance) for instance in X_train])
        X_val = np.array([np.column_stack(instance) for instance in X_val])

        y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
        y_val = np.array(y_val, dtype=np.float32).reshape(-1, 1)

        if computed_stats is not None:
            normalize_costs(test_data, normalization_mode, existing_stats=computed_stats)
        else:
            normalize_costs(test_data, normalization_mode)


        test_data = pd.DataFrame(test_data, columns=["x", "y", "demand", "cost"])
        
        test_data_np = np.vstack(test_data.apply(lambda x: np.array(list(x), dtype=object), axis=1))
        X_test = test_data_np[:, :-1]
        y_test = test_data_np[:, -1:]
        X_test = np.array([np.column_stack(instance) for instance in X_test])
        y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

        print("Shape of X_train:", X_train.shape)
        print("Shape of X_val:", X_val.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_val:", y_val.shape)
        print("Shape of y_test:", y_test.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test