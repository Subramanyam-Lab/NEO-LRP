"""
Verify solution JSONs by recomputing costs from routes.
"""

import os
import json
import math
import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.dataparse import create_data


def dist_calc(depot_coords, cust_coords, rc_cal_index):
    """Calculate depot to customer distances."""
    depot_coords = np.array(depot_coords, dtype=float)
    cust_coords = np.array(cust_coords, dtype=float)
    
    diff = depot_coords[:, np.newaxis, :] - cust_coords[np.newaxis, :, :]
    distances = np.hypot(diff[..., 0], diff[..., 1])
    
    if rc_cal_index == 0:
        distances = (100 * distances).astype(int)
    # else: keep as float for Tuzun/Barreto
    
    return distances


def verify_instance(dat_path, json_path):
    """Verify a single instance."""
    instance_name = os.path.basename(dat_path).replace('.dat', '')
    
    if not os.path.exists(json_path):
        print(f"  [skip] Missing JSON: {json_path}")
        return None
    
    # Load instance data
    ans = create_data(dat_path)
    no_cust, no_depot = ans[0], ans[1]
    depot_coords, cust_coords = ans[2], ans[3]
    veh_cap, cust_dem = ans[4], ans[6]
    open_dep_cost, route_cost_param, rc_cal_index = ans[7], ans[8], ans[9]
    
    # Load solution
    with open(json_path, "r") as f:
        sol = json.load(f)
    
    routes_info = sol.get("routes_info", [])
    
    # Handle different JSON structures
    if "sets" in sol:
        depots_open = sol["sets"].get("opened_indices", [])
    else:
        depots_open = sol.get("depots_open", [])
    
    if "objective" in sol:
        reported_flp = sol["objective"].get("facility_opening", 0)
        reported_vrp = sol["objective"].get("vrp_actual", 0)
        reported_lrp = sol["objective"].get("lrp_total_actual", 0)
    else:
        reported_flp = sol.get("fixed_cost", 0)
        reported_vrp = sol.get("vrp_cost_actual", 0)
        reported_lrp = sol.get("lrp_total_actual", 0)
    
    # Compute distances
    depot_to_customer_dist = dist_calc(depot_coords, cust_coords, rc_cal_index)
    
    cust_coords = np.array(cust_coords)
    n_cust = len(cust_coords)
    cust_to_cust_dist = np.zeros((n_cust, n_cust))
    for i in range(n_cust):
        for j in range(n_cust):
            if i != j:
                d = math.hypot(cust_coords[i, 0] - cust_coords[j, 0],
                               cust_coords[i, 1] - cust_coords[j, 1])
                if rc_cal_index == 0:
                    d = int(100 * d)
                # else: keep as float for Tuzun/Barreto
                cust_to_cust_dist[i, j] = d
    
    # Recompute VRP cost from routes
    recomputed_vrp_cost = 0.0
    nonempty_routes = 0
    
    for r in routes_info:
        route_ids = [int(i) for i in r["Ids"] if "Depot" not in str(i)]
        if not route_ids:
            continue
        nonempty_routes += 1
        depot_idx = r.get("depot_index", 0)
        
        # Depot -> first customer
        cost = depot_to_customer_dist[depot_idx, route_ids[0]]
        # Customer to customer
        for i in range(len(route_ids) - 1):
            c1, c2 = route_ids[i], route_ids[i + 1]
            cost += cust_to_cust_dist[c1, c2]
        # Last customer -> depot
        cost += depot_to_customer_dist[depot_idx, route_ids[-1]]
        recomputed_vrp_cost += cost
    
    # Recompute FLP and fixed route cost
    recomputed_flp_cost = sum(open_dep_cost[j] for j in depots_open)
    recomputed_fixed_vrp_cost = route_cost_param * nonempty_routes
    recomputed_lrp_total = recomputed_flp_cost + recomputed_fixed_vrp_cost + recomputed_vrp_cost
    
    diff = recomputed_lrp_total - reported_lrp
    
    # Use tolerance for float comparisons (Tuzun/Barreto)
    tolerance = 0.01 if rc_cal_index == 1 else 1e-6
    
    return {
        "instance": instance_name,
        "rc_cal_index": rc_cal_index,
        "recomputed_flp": recomputed_flp_cost,
        "recomputed_fixed_vrp": recomputed_fixed_vrp_cost,
        "recomputed_var_vrp": recomputed_vrp_cost,
        "recomputed_total": recomputed_lrp_total,
        "reported_flp": reported_flp,
        "reported_vrp": reported_vrp,
        "reported_total": reported_lrp,
        "diff": diff,
        "match": abs(diff) < tolerance
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify solution JSONs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['P_prodhon', 'S_schneider', 'T_tuzun', 'B_barreto'])
    parser.add_argument('--model_type', type=str, default='deepsets',
                        choices=['deepsets', 'graph_transformer'])
    parser.add_argument('--solver', type=str, default='vroom')
    parser.add_argument('--normalization', type=str, default='cost_over_fi')
    parser.add_argument('--N', type=int, default=110000)
    args = parser.parse_args()
    
    # Paths
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    INSTANCES_DIR = os.path.join(BASE_DIR, "benchmark_instances", args.dataset)
    SOLUTIONS_DIR = os.path.join(
        SCRIPT_DIR, "output", args.dataset,
        f"{args.model_type}_{args.solver}_{args.normalization}_{args.N}"
    )
    
    print(f"\n{'~'*60}")
    print(f"Verifying Solutions")
    print(f"{'~'*60}")
    print(f"Dataset:       {args.dataset}")
    print(f"Model type:    {args.model_type}")
    print(f"Solver:        {args.solver}")
    print(f"Normalization: {args.normalization}")
    print(f"N:             {args.N}")
    print(f"Instances dir: {INSTANCES_DIR}")
    print(f"Solutions dir: {SOLUTIONS_DIR}")
    print(f"{'~'*60}")
    
    if not os.path.exists(SOLUTIONS_DIR):
        print(f"ERROR: Solutions directory not found: {SOLUTIONS_DIR}")
        return
    
    # Get file extension
    ext = ".json" if args.dataset == "S_schneider" else ".dat"
    
    results = []
    mismatches = []
    
    for filename in sorted(os.listdir(INSTANCES_DIR)):
        if not filename.endswith(ext):
            continue
        
        instance_stem = os.path.splitext(filename)[0]
        dat_path = os.path.join(INSTANCES_DIR, filename)
        json_path = os.path.join(SOLUTIONS_DIR, instance_stem, "run_1.json")
        
        print(f"\n--- {instance_stem} ---")
        
        result = verify_instance(dat_path, json_path)
        if result is None:
            continue
        
        results.append(result)
        
        print(f"  rc_cal_index:      {result['rc_cal_index']}")
        print(f"  FLP cost:          {result['recomputed_flp']:10.2f}")
        print(f"  Fixed VRP cost:    {result['recomputed_fixed_vrp']:10.2f}")
        print(f"  Variable VRP cost: {result['recomputed_var_vrp']:10.2f}")
        print(f"  ----------------------------------")
        print(f"  Recomputed total:  {result['recomputed_total']:10.2f}")
        print(f"  Reported total:    {result['reported_total']:10.2f}")
        print(f"  Difference:        {result['diff']:10.2f}")
        
        if result['match']:
            print(f"  MATCH")
        else:
            print(f"  MISMATCH")
            mismatches.append(result)
    
    # Summary
    print(f"\n{'~'*60}")
    print(f"SUMMARY")
    print(f"{'~'*60}")
    print(f"Total instances checked: {len(results)}")
    print(f"Matches:                 {len(results) - len(mismatches)}")
    print(f"Mismatches:              {len(mismatches)}")
    
    if mismatches:
        print(f"\nMismatched instances:")
        for m in mismatches:
            print(f"  {m['instance']}: diff = {m['diff']:.2f}")


if __name__ == "__main__":
    main()