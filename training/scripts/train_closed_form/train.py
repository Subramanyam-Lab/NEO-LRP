"""
Fit Daganzo and Robuste approximations for CVRP routing costs.
Uses raw CVRP features and scaled VROOM labels without normalization,
exports regression equations, coefficients and prediction errors to JSON.
"""

import argparse
import hashlib
import json
import os
import random

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


def extract_raw_cvrp(instance_str):
    """Parse raw coordinates, capacity and demands, including the depot zero."""
    lines = instance_str.strip().split("\n")
    capacity = float(next(line for line in lines if line.startswith("CAPACITY")).split(":")[1])
    start = lines.index("NODE_COORD_SECTION") + 1
    end = lines.index("DEMAND_SECTION")
    coordinates = np.array([list(map(float, line.split()[1:3])) for line in lines[start:end]])

    demands = []
    for line in lines[end + 1:]:
        if line.startswith("DEPOT_SECTION"):
            break
        demands.append(float(line.split()[1]))

    return coordinates[1:], np.array(demands), capacity, coordinates[0]


def gen_CA_features_CVRP(customers, demands, capacity, depot):
    """Compute [n, r, Q_cust, sqrt_AN, N_over_C2, interaction] from raw features."""
    n = customers.shape[0]
    avg_demand = demands.mean() if n > 0 else 1.0
    Q_cust = capacity / avg_demand if avg_demand > 0 else 0.0
    r = np.linalg.norm(customers - depot[None, :], axis=1).mean() if n > 0 else 0.0

    A = 0.0
    if n > 2:
        try:
            A = ConvexHull(customers).volume
        except QhullError:
            pass

    sqrt_AN = np.sqrt(n * A)
    N_over_C2 = n / (Q_cust ** 2) if Q_cust > 0 else 0.0
    interaction = N_over_C2 * sqrt_AN
    return np.array([n, r, Q_cust, sqrt_AN, N_over_C2, interaction], dtype=float)


def read_data(file_path):
    """Read CVRP features and scaled VROOM cost labels from the existing dataset."""
    features, labels, names = [], [], []
    instance_lines = []
    with open(file_path, "r") as file:
        for line in file:
            instance_lines.append(line)
            if line.strip() != "#EOF":
                continue

            instance_str = "".join(instance_lines)
            features.append(gen_CA_features_CVRP(*extract_raw_cvrp(instance_str)))
            cost_line = next(line for line in instance_lines
                             if line.startswith("#cost_vroom_scaled "))
            labels.append(float(cost_line.split()[1]))
            names.append(next(line.split(":", 1)[1].strip() for line in instance_lines
                              if line.startswith("NAME")))
            instance_lines = []

    if "".join(instance_lines).strip():
        raise ValueError(f"Incomplete final instance in {file_path}")
    if not features:
        raise ValueError(f"No CVRP instances found in {file_path}")
    print(f"Read {len(features)} instances from {file_path}", flush=True)
    return np.array(features), np.array(labels, dtype=np.float32), np.array(names)


def compute_error_stats(values):
    stats = {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
    }
    iqr = stats["q75"] - stats["q25"]
    stats["whisker_lo"] = stats["q25"] - 1.5 * iqr
    stats["whisker_hi"] = stats["q75"] + 1.5 * iqr
    return {key: float(value) for key, value in stats.items()}


def regression_features(features, model_name):
    if model_name == "Daganzo":
        return np.column_stack((2 * features[:, 1] * (features[:, 0] / features[:, 2]),
                                features[:, 3]))
    return features[:, [3, 5]]


def fit_models(datasets):
    """Fit two coefficients without an intercept and evaluate each data split."""
    model_features = {
        "Daganzo": ["2*r*n/Q_cust", "sqrt(n*A)"],
        "Robuste": ["sqrt(n*A)", "(n/Q_cust**2)*sqrt(n*A)"],
    }
    results = {}
    for name, terms in model_features.items():
        X_train, y_train = datasets["train"]
        model = LinearRegression(fit_intercept=False)
        model.fit(regression_features(X_train, name), y_train)
        coefficients = model.coef_.ravel().tolist()
        result = {
            "Selected Features": terms,
            "Coefficients": coefficients,
            "Intercept": float(model.intercept_),
            "Equation": f"R = {coefficients[0]:.17g} * ({terms[0]}) + "
                        f"{coefficients[1]:.17g} * ({terms[1]})",
        }
        for split, (X, y) in datasets.items():
            prediction = model.predict(regression_features(X, name))
            result[f"{split}_R2"] = float(r2_score(y, prediction))
            percentage_error = (y - prediction) / (y + 1e-8) * 100
            for metric, values in (("mpe", percentage_error), ("mape", np.abs(percentage_error))):
                for key, value in compute_error_stats(values).items():
                    result[f"{split}_{metric}_{key}"] = value
        results[name] = result
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fit closed-form CVRP cost approximations")
    parser.add_argument("--train-val", default=os.path.join(BASE_DIR, "training/data/train_val.txt"))
    parser.add_argument("--test", default=os.path.join(BASE_DIR, "training/data/test.txt"))
    parser.add_argument("--num-instances", type=int, default=110000)
    parser.add_argument("--seed", type=int, default=42, help="Instance sampling seed")
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "trained_models/closed_form"))
    args = parser.parse_args()
    if args.num_instances < 20:
        parser.error("--num-instances must be at least 20")
    return args


def main():
    args = parse_arguments()
    X, labels, names = read_data(args.train_val)
    X_test, test_labels, test_names = read_data(args.test)
    if set(names) & set(test_names):
        raise ValueError("Training/validation and test instance names overlap")
    if not np.isfinite(X).all() or not np.isfinite(X_test).all():
        raise ValueError("Nonfinite CVRP features")

    selected = random.Random(args.seed).sample(range(len(X)), min(args.num_instances, len(X)))
    os.makedirs(args.output_dir, exist_ok=True)

    y = labels[selected]
    valid = np.isfinite(y) & (y > 0)
    indices = np.array(selected)[valid]
    valid_test = np.isfinite(test_labels) & (test_labels > 0)
    test = np.flatnonzero(valid_test)
    if len(indices) < 20 or len(test) < 2:
        raise ValueError("Insufficient finite, positive scaled labels for fitting and evaluation")
    train, validation = train_test_split(indices, test_size=0.1, random_state=42)
    datasets = {
        "train": (X[train], labels[train]),
        "val": (X[validation], labels[validation]),
        "test": (X_test[test], test_labels[test]),
    }
    result = {
        "data": {
            "train_val": os.path.relpath(args.train_val, BASE_DIR),
            "test": os.path.relpath(args.test, BASE_DIR),
            "label": "cost_vroom_scaled",
            "normalization": "raw",
            "seed": args.seed,
            "validation_seed": 42,
            "selected_instances": len(selected),
            "train_instances": len(train),
            "validation_instances": len(validation),
            "test_instances": len(test),
            "discarded_train_val": int((~valid).sum()),
            "discarded_test": int((~valid_test).sum()),
            "train_names_sha256": hashlib.sha256("\n".join(names[train]).encode()).hexdigest(),
            "validation_names_sha256": hashlib.sha256("\n".join(names[validation]).encode()).hexdigest(),
            "test_names_sha256": hashlib.sha256("\n".join(test_names[test]).encode()).hexdigest(),
        },
        "feature_definitions": {
            "n": "number of customers",
            "r": "mean Euclidean customer-depot distance",
            "A": "customer convex-hull area, zero for degenerate point sets",
            "Q_cust": "vehicle capacity / mean demand including the depot zero",
        },
        "cost_convention": {
            "distance_multiplier": 100,
            "distance_rounding": "floor",
            "fixed_vehicle_cost": 1000,
        }
    }
    result.update(fit_models(datasets))
    output_path = os.path.join(args.output_dir, "scaled.json")
    with open(output_path, "w") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Saved {output_path}", flush=True)
    print(f"Train/validation/test: {len(train)}/{len(validation)}/{len(test)}", flush=True)
    for name in ("Daganzo", "Robuste"):
        print(f"{name}: {result[name]['Equation']}", flush=True)
        print(f"Test MAPE: {result[name]['test_mape_mean']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
