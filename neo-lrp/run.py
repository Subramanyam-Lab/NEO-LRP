"""
Unified NEO DS Runner

Usage:
    # DeepSets (default)
    python run.py --dataset P_prodhon --normalization cost_over_fi --N 110000 --solver vroom --num_runs 5
    
    # Graph Transformer
    python run.py --dataset P_prodhon --normalization cost_over_fi --N 110000 --solver vroom --num_runs 5 --model_type graph_transformer
    
    # Single instance (array job)
    python run.py --dataset S_schneider --instance 100-5-1a.json --normalization cost_over_fi --N 110000 --solver vroom --num_runs 1
"""

import os
import sys
import argparse
import json
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.dataparse import create_data, load_config
from core.lrp_model import createLRP
from core.solver import DataCvrp, solve_instance

BASE_DIR = os.path.dirname(SCRIPT_DIR)
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
BKS_DIR = os.path.join(CONFIGS_DIR, "BKS")
INSTANCES_DIR = os.path.join(BASE_DIR, "benchmark_instances")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")  # No "deepsets" here

SCALED_DATASETS = ["P_prodhon", "S_schneider"]
UNSCALED_DATASETS = ["T_tuzun", "B_barreto"]


def get_model_paths(dataset, normalization, N, model_type):
    """Get model paths based on dataset, normalization, N, and model type."""
    model_subdir = "scaled" if dataset in SCALED_DATASETS else "unscaled"
    
    if model_type == "deepsets":
        phi = os.path.join(MODELS_DIR, "deepsets", model_subdir, "phi", normalization, f"{N}.onnx")
        rho = os.path.join(MODELS_DIR, "deepsets", model_subdir, "rho", normalization, f"{N}.onnx")
    else:  # graph_transformer
        phi = os.path.join(MODELS_DIR, "graph_transformer", model_subdir, normalization, f"{N}.pth")
        rho = None
    
    return phi, rho


def write_solution_json(output_path, result, ans, solver_choice, run_index):
    """Write solution JSON for analysis scripts."""
    num_depots = ans[1]
    opened_vector = [0] * num_depots
    for j in result["open_depots"]:
        if 0 <= j < num_depots:
            opened_vector[j] = 1
    
    opened_indices = result["open_depots"]
    closed_indices = [j for j in range(num_depots) if j not in opened_indices]
    
    num_customers = ans[0]
    assign_depot_per_customer = [-1] * num_customers
    depot_to_customers = {j: [] for j in opened_indices}
    
    for depot_idx, customers in result.get("assignments", {}).items():
        d = int(depot_idx)
        for c in customers:
            if 0 <= c < num_customers:
                assign_depot_per_customer[c] = d
                if d in depot_to_customers:
                    depot_to_customers[d].append(c)
    
    solution_dict = {
        "instance": result["instance"],
        "run_index": run_index,
        "method": "NEO_LRP",
        "model_type": result.get("model_type", "deepsets"),
        "timestamp": datetime.now().isoformat(),
        "solver_choice": solver_choice,
        "bks": result.get("bks"),
        "objective": {
            "facility_opening": result.get("flp_cost"),
            "vrp_nn": result.get("vrp_cost_nn"),
            "vrp_actual": result.get("vrp_cost_actual"),
            "lrp_total_nn": result.get("lrp_total_nn"),
            "lrp_total_actual": result.get("lrp_total_actual"),
        },
        "timing_sec": {
            "lrp_solve_time": result.get("lrp_solve_time"),
            "vrp_solve_time": result.get("vrp_solve_time"),
            "total_exec_time": result.get("exec_time"),
        },
        "sets": {
            "opened_vector": opened_vector,
            "opened_indices": opened_indices,
            "closed_indices": closed_indices,
            "assign_depot_per_customer": assign_depot_per_customer,
            "depot_to_customers": {str(k): v for k, v in depot_to_customers.items()},
        },
        "data": {
            "facility_cost": [float(x) for x in ans[7]],
            "depot_capacity": [int(x) for x in ans[5]],
            "cust_demand": [int(x) for x in ans[6]],
            "dep_coords": [[float(x), float(y)] for x, y in ans[2]],
            "cus_coords": [[float(x), float(y)] for x, y in ans[3]],
        },
        "routes_info": result.get("routes_info", []),
        "num_routes": result.get("num_routes"),
        "gap_signed": result.get("gap"),
        "gap_abs": abs(result["gap"]) if result.get("gap") is not None else None,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(solution_dict, f, indent=2)


def write_plot_json(output_path, result, ans):
    """Write plot JSON for visualization scripts."""
    num_customers = ans[0]
    assign_depot_per_customer = [-1] * num_customers
    for depot_idx, customers in result.get("assignments", {}).items():
        d = int(depot_idx)
        for c in customers:
            if 0 <= c < num_customers:
                assign_depot_per_customer[c] = d
    
    per_depot = {}
    for rt in result.get("routes_info", []):
        d_idx = rt.get("depot_index")
        if d_idx is None:
            continue
        dkey = str(d_idx)
        if dkey not in per_depot:
            per_depot[dkey] = {"routes": []}
        per_depot[dkey]["routes"].append(rt)
    
    plot_payload = {
        "data": {
            "dep_coords": [[float(x), float(y)] for x, y in ans[2]],
            "cus_coords": [[float(x), float(y)] for x, y in ans[3]],
        },
        "sets": {
            "opened_indices": result["open_depots"],
            "assign_depot_per_customer": assign_depot_per_customer,
        },
        "per_depot": per_depot,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(plot_payload, f, indent=2)


def result_to_excel_row(result, solver_name):
    """Convert result to Excel row dict."""
    if result.get("status") == "SOLVER_FAILED":
        return {
            "Instance": result["instance"],
            "Model_Type": result.get("model_type", "deepsets"),
            "BKS": result.get("bks"),
            "Status": "SOLVER_FAILED",
            "Error": result.get("error"),
            "Optimization_gap_signed": None,
            "Prediction_gap_abs": None,
            "NN model execution time": result.get("lrp_solve_time"),
            "LRP_total_actual": None,
            "LRP_total_nn": None,
            "VRP_cost_actual": None,
            "VRP_cost_nn": result.get("vrp_cost_nn"),
            "FLP_cost": result.get("flp_cost"),
            "Num_routes": None,
            "Open_depots": str(result.get("open_depots")),
            "vroom model solve time": None,
            "ortools model solve time": None,
            "vrpeasy model solve time": None,
        }
    
    vrp_actual = result.get("vrp_cost_actual")
    vrp_nn = result.get("vrp_cost_nn")
    prediction_gap_abs = None
    if vrp_actual and vrp_actual > 0 and vrp_nn is not None:
        prediction_gap_abs = abs(vrp_nn - vrp_actual) / vrp_actual * 100
    
    return {
        "Instance": result["instance"],
        "Model_Type": result.get("model_type", "deepsets"),
        "BKS": result.get("bks"),
        "Status": "SUCCESS",
        "Error": None,
        "Optimization_gap_signed": result.get("gap"),
        "Prediction_gap_abs": prediction_gap_abs,
        "NN model execution time": result.get("lrp_solve_time"),
        "LRP_total_actual": result.get("lrp_total_actual"),
        "LRP_total_nn": result.get("lrp_total_nn"),
        "VRP_cost_actual": result.get("vrp_cost_actual"),
        "VRP_cost_nn": result.get("vrp_cost_nn"),
        "FLP_cost": result.get("flp_cost"),
        "Num_routes": result.get("num_routes"),
        "Open_depots": str(result.get("open_depots", [])),
        "vroom model solve time": result.get("vrp_solve_time") if solver_name == "vroom" else None,
        "ortools model solve time": result.get("vrp_solve_time") if solver_name == "ortools" else None,
        "vrpeasy model solve time": result.get("vrp_solve_time") if solver_name == "vrpeasy" else None,
    }


def write_excel(output_path, rows, solver_name):
    """Write Excel file with one or more rows."""
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='results', index=False)
    print(f"  [saved] excel: {output_path}")


def run_single_instance(config, file_path, phi_loc, rho_loc, normalization, solver_choice, bks_dict, model_type):
    """Run LRP on a single instance."""
    filename = os.path.basename(file_path)
    print(f"\n{'~'*60}")
    print(f"processing: {filename}")
    print(f"{'~'*60}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"instance file not found: {file_path}")
    
    ans = create_data(file_path, config)
    fixed_route_cost = ans[8]
    rc_cal_index = ans[9]
    
    if ans[0] is None or ans[0] <= 0:
        raise ValueError(f"invalid number of customers: {ans[0]}")
    if ans[1] is None or ans[1] <= 0:
        raise ValueError(f"invalid number of depots: {ans[1]}")
    
    print(f"  customers: {ans[0]}, depots: {ans[1]}")
    print(f"  rc_cal_index: {rc_cal_index}, fixed_route_cost: {fixed_route_cost}")
    print(f"  model_type: {model_type}")
    
    # Add model_type to config
    config["model_type"] = model_type
    
    lrp_st = datetime.now()
    lrp_solver = createLRP(ans, cost_normalization=normalization, config=config)
    lrp_result = lrp_solver.model(
        file_path, "log.txt", phi_loc, rho_loc,
        cost_normalization=normalization
    )
    lrp_ed = datetime.now()
    lrp_solve_time = (lrp_ed - lrp_st).total_seconds()
    
    if lrp_result is None:
        raise RuntimeError(f"lrp model returned none for instance {filename}")
    
    flp_dict = {}
    for j in range(len(lrp_result[0])):
        if lrp_result[0][j] > 0.5:
            flp_dict[j] = [i for i in range(len(lrp_result[1][j])) if lrp_result[1][j][i] > 0.5]
    
    if not flp_dict:
        raise RuntimeError(f"no depots opened for instance {filename}")
    
    print(f"\n  open depots: {list(flp_dict.keys())}")
    
    total_vrp_cost = 0
    total_routes = 0
    all_routes_info = []
    solver_failures = []
    total_vrp_solve_time = 0
    
    for f, cust_list in flp_dict.items():
        if not cust_list:
            print(f"  [warn] depot {f}: no customers assigned, skipping")
            continue
        
        demands = [ans[6][c] for c in cust_list]
        coords = [ans[3][c] for c in cust_list]
        depot = ans[2][f]
        
        cvrp_data = DataCvrp(
            vehicle_capacity=ans[4],
            nb_customers=len(cust_list),
            cust_demands=demands,
            cust_coordinates=coords,
            depot_coordinates=depot,
            original_ids=cust_list,
            depot_label=f"Depot_{f}",
            rc_cal_index=rc_cal_index,
            fixed_route_cost=fixed_route_cost
        )
        
        var_cost, num_routes, msg, routes, t_solve = solve_instance(
            cvrp_data, solver_type=solver_choice, config=config
        )
        
        total_vrp_solve_time += t_solve
        
        if var_cost is None or var_cost == 0 or num_routes == 0:
            solver_failures.append({
                "depot": f,
                "customers": len(cust_list),
                "message": msg,
                "cost": var_cost,
                "routes": num_routes
            })
            print(f"  [warn] depot {f}: solver failed")
        else:
            for r in routes:
                r["depot_index"] = f
            all_routes_info.extend(routes)
            total_vrp_cost += var_cost
            total_routes += num_routes
            print(f"  depot {f}: {len(cust_list)} customers, {num_routes} routes, cost={var_cost:.2f}")
    
    exec_time = (datetime.now() - lrp_st).total_seconds()
    
    if solver_failures:
        print(f"\n  [error] solver failed on {len(solver_failures)}/{len(flp_dict)} depots!")
        return {
            "instance": filename,
            "model_type": model_type,
            "status": "SOLVER_FAILED",
            "failed_depots": solver_failures,
            "open_depots": list(flp_dict.keys()),
            "assignments": {j: cust_list for j, cust_list in flp_dict.items()},
            "flp_cost": lrp_result[2],
            "vrp_cost_nn": lrp_result[3],
            "vrp_cost_actual": None,
            "lrp_total_actual": None,
            "lrp_total_nn": None,
            "num_routes": None,
            "bks": bks_dict.get(filename),
            "gap": None,
            "exec_time": exec_time,
            "lrp_solve_time": lrp_solve_time,
            "vrp_solve_time": total_vrp_solve_time,
            "routes_info": all_routes_info,
            "error": f"solver failed on {len(solver_failures)} depot(s)"
        }, ans
    
    flp_cost = lrp_result[2]
    vrp_cost_nn = lrp_result[3]
    lrp_cost_nn = flp_cost + vrp_cost_nn
    actual_lrp_cost = flp_cost + total_vrp_cost
    bks = bks_dict.get(filename)
    
    gap = None
    if bks is not None and bks > 0:
        gap = ((actual_lrp_cost - bks) / bks) * 100
    
    print(f"\n  results:")
    print(f"    flp cost:       {flp_cost:.2f}")
    print(f"    vrp cost (nn):  {vrp_cost_nn:.2f}")
    print(f"    vrp cost (act): {total_vrp_cost:.2f}")
    print(f"    lrp total:      {actual_lrp_cost:.2f}")
    print(f"    bks:            {bks}")
    print(f"    gap:            {gap:.2f}%" if gap else "    gap:            n/a")
    
    return {
        "instance": filename,
        "model_type": model_type,
        "status": "SUCCESS",
        "open_depots": list(flp_dict.keys()),
        "assignments": {j: cust_list for j, cust_list in flp_dict.items()},
        "flp_cost": flp_cost,
        "vrp_cost_nn": vrp_cost_nn,
        "vrp_cost_actual": total_vrp_cost,
        "lrp_total_nn": lrp_cost_nn,
        "lrp_total_actual": actual_lrp_cost,
        "num_routes": total_routes,
        "bks": bks,
        "gap": gap,
        "exec_time": exec_time,
        "lrp_solve_time": lrp_solve_time,
        "vrp_solve_time": total_vrp_solve_time,
        "routes_info": all_routes_info
    }, ans


def average_results(results):
    """Average numeric fields across multiple runs."""
    if len(results) == 1:
        return results[0]
    
    failed = [r for r in results if r.get("status") == "SOLVER_FAILED"]
    if len(failed) == len(results):
        return results[0]
    
    successful = [r for r in results if r.get("status") != "SOLVER_FAILED"]
    if not successful:
        return results[0]

    avg = successful[0].copy()
    numeric_fields = [
        "flp_cost", "vrp_cost_nn", "vrp_cost_actual", 
        "lrp_total_nn", "lrp_total_actual", "num_routes",
        "gap", "exec_time", "lrp_solve_time", "vrp_solve_time"
    ]
    
    for field in numeric_fields:
        values = [r[field] for r in successful if r.get(field) is not None]
        if values:
            avg[field] = sum(values) / len(values)
    
    return avg


def main():
    parser = argparse.ArgumentParser(description='NEO-LRP Runner')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['P_prodhon', 'S_schneider', 'T_tuzun', 'B_barreto'])
    parser.add_argument('--normalization', type=str, required=True,
                        choices=['raw', 'minmax', 'cost_over_fi', 'cost_over_fi_minmax'])
    parser.add_argument('--N', type=int, required=True,
                        choices=[110, 1100, 11000, 110000])
    parser.add_argument('--solver', type=str, required=True,
                        choices=['vroom', 'ortools', 'vrpeasy'])
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--instance', type=str, default=None,
                        help='Single instance (array job mode)')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--model_type', type=str, default='deepsets',
                        choices=['deepsets', 'graph_transformer'],
                        help='Neural network architecture')
    
    args = parser.parse_args()
    
    if args.num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    
    # Load config
    config_path = os.path.join(CONFIGS_DIR, f"{args.dataset}.json")
    config = load_config(config_path)
    
    # Get model paths (now with model_type)
    phi_loc, rho_loc = get_model_paths(args.dataset, args.normalization, args.N, args.model_type)
    
    # Validate model paths
    if not os.path.exists(phi_loc):
        raise FileNotFoundError(f"model not found: {phi_loc}")
    if args.model_type == "deepsets" and not os.path.exists(rho_loc):
        raise FileNotFoundError(f"rho model not found: {rho_loc}")
    
    # Load BKS
    bks_path = os.path.join(BKS_DIR, f"{args.dataset}.json")
    bks_dict = {}
    if os.path.exists(bks_path):
        with open(bks_path, 'r') as f:
            bks_dict = json.load(f)
    
    # Get instances
    instance_dir = os.path.join(INSTANCES_DIR, args.dataset)
    is_array_job = args.instance is not None
    
    if is_array_job:
        instances = [args.instance]
    else:
        instances = list(bks_dict.keys())
    
    print(f"\n{'~'*60}")
    print(f"neo-lrp runner")
    print(f"{'~'*60}")
    print(f"dataset:       {args.dataset}")
    print(f"model_type:    {args.model_type}")
    print(f"normalization: {args.normalization}")
    print(f"n:             {args.N}")
    print(f"solver:        {args.solver}")
    print(f"num runs:      {args.num_runs}")
    print(f"mode:          {'array job (single instance)' if is_array_job else 'all instances'}")
    print(f"instances:     {len(instances)}")
    print(f"phi model:     {phi_loc}")
    print(f"rho model:     {rho_loc}")
    print(f"{'~'*60}")
    
    # Output directories - include model_type
    base_output_dir = os.path.join(args.output_dir, args.dataset)
    json_subdir = f"{args.model_type}_{args.solver}_{args.normalization}_{args.N}"
    plot_subdir = f"{args.model_type}_{args.solver}_{args.normalization}_{args.N}_plots"
    
    json_output_dir = os.path.join(base_output_dir, json_subdir)
    plot_output_dir = os.path.join(base_output_dir, plot_subdir)
    
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)
    
    if is_array_job:
        excel_temp_dir = os.path.join(base_output_dir, f"{args.model_type}_{args.solver}_{args.normalization}_{args.N}_excel_temp")
        os.makedirs(excel_temp_dir, exist_ok=True)
    
    all_results = []
    all_ans = {}
    
    for filename in instances:
        file_path = os.path.join(instance_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"instance not found: {file_path}")
        
        instance_stem = os.path.splitext(filename)[0]
        
        for run in range(args.num_runs):
            run_index = run + 1
            if args.num_runs > 1:
                print(f"\n--- run {run_index}/{args.num_runs} ---")
            
            result, ans = run_single_instance(
                config, file_path, phi_loc, rho_loc,
                args.normalization, args.solver, bks_dict, args.model_type
            )
            result["run"] = run_index
            result["solver"] = args.solver
            result["normalization"] = args.normalization
            result["N"] = args.N
            
            all_ans[filename] = ans
            
            # Write solution JSON
            solution_json_path = os.path.join(
                json_output_dir, instance_stem, f"run_{run_index}.json"
            )
            write_solution_json(solution_json_path, result, ans, args.solver, run_index)
            print(f"  [saved] solution json: {solution_json_path}")
            
            # Write plot JSON
            plot_json_path = os.path.join(
                plot_output_dir, instance_stem, f"run_{run_index}.json"
            )
            write_plot_json(plot_json_path, result, ans)
            print(f"  [saved] plot json: {plot_json_path}")
            
            all_results.append(result)
    
    # Write Excel
    if is_array_job:
        instance_stem = os.path.splitext(args.instance)[0]
        if args.num_runs > 1:
            avg_result = average_results(all_results)
        else:
            avg_result = all_results[0]
        
        row = result_to_excel_row(avg_result, args.solver)
        excel_path = os.path.join(excel_temp_dir, f"{instance_stem}.xlsx")
        write_excel(excel_path, [row], args.solver)
    else:
        instance_results = defaultdict(list)
        for r in all_results:
            instance_results[r["instance"]].append(r)
        
        rows = []
        for instance_name in instances:
            results_for_instance = instance_results[instance_name]
            if args.num_runs > 1 and len(results_for_instance) > 1:
                avg_result = average_results(results_for_instance)
            else:
                avg_result = results_for_instance[0]
            rows.append(result_to_excel_row(avg_result, args.solver))
        
        # Excel filename includes model_type
        excel_path = os.path.join(
            base_output_dir,
            f"{args.dataset}_{args.model_type}_{args.N}_{args.normalization}_{args.solver}.xlsx"
        )
        write_excel(excel_path, rows, args.solver)
    
    # Summary
    print(f"\n{'~'*60}")
    print("summary")
    print(f"{'~'*60}")

    successful = [r for r in all_results if r.get("status") != "SOLVER_FAILED"]
    failed = [r for r in all_results if r.get("status") == "SOLVER_FAILED"]

    print(f"total runs:      {len(all_results)}")
    print(f"successful:      {len(successful)}")
    print(f"failed:          {len(failed)}")

    if successful:
        gaps = [r["gap"] for r in successful if r["gap"] is not None]
        if gaps:
            print(f"\ngap statistics:")
            print(f"  average: {sum(gaps)/len(gaps):.2f}%")
            print(f"  min:     {min(gaps):.2f}%")
            print(f"  max:     {max(gaps):.2f}%")


if __name__ == "__main__":
    main()