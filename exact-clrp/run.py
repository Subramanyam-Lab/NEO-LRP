"""
Three-index commodity-flow CLRP formulation using Gurobi.
Solves depot openings, customer assignments and vehicle routes jointly.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import gurobipy as gp
from gurobipy import GRB


def distance_cost(first, second, rounding):
    """Compute the scaled Euclidean arc cost with explicit rounding."""
    distance = 100 * math.hypot(first[0] - second[0], first[1] - second[1])
    if rounding == "truncate":
        return int(distance)
    if rounding == "ceil":
        return math.ceil(distance)
    if rounding == "nearest":
        return math.floor(distance + 0.5)
    raise ValueError(f"Unknown rounding convention: {rounding}")


def build_model(data, time_limit=3600, log_file=None,
                rounding="ceil", mip_gap=0.01):
    """Build the formulation on each depot's admissible arc set."""
    n, m, depots, customers, capacity, depot_capacity, demand, opening, fixed, index = data
    if index != 0:
        raise ValueError("This implementation expects scaled Prodhon instances.")
    if n < 1 or m < 1 or len(set(capacity)) != 1 or capacity[0] <= 0:
        raise ValueError("A positive homogeneous vehicle capacity is required.")
    if any(q <= 0 for q in demand):
        raise ValueError("Customer demands must be positive.")
    if any(cost < 0 for cost in opening) or fixed < 0:
        raise ValueError("Opening and fixed vehicle costs must be nonnegative.")
    coordinates = customers + depots
    arcs = [(i, j, d) for d in range(m)
            for i in list(range(n)) + [n + d]
            for j in list(range(n)) + [n + d] if i != j]
    costs = {(i, j): distance_cost(coordinates[i], coordinates[j], rounding)
             for i, j in {(i, j) for i, j, d in arcs}}

    model = gp.Model("CLRP")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    if log_file is not None:
        model.Params.LogFile = str(log_file)
    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    w = model.addVars(n, m, vtype=GRB.BINARY, name="w")
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    v = model.addVars(arcs, lb=0, name="v")
    model.setObjective(
        gp.quicksum(opening[d] * y[d] for d in range(m))
        + gp.quicksum(costs[i, j] * x[i, j, d] for i, j, d in arcs)
        + fixed * gp.quicksum(x[n + d, j, d] for d in range(m) for j in range(n)),
        GRB.MINIMIZE)
    model.addConstrs((w[i, d] <= y[d] for i in range(n) for d in range(m)),
                     name="open")
    model.addConstrs((w.sum(i, "*") == 1 for i in range(n)), name="visit")
    model.addConstrs((gp.quicksum(demand[i] * w[i, d] for i in range(n))
                      <= depot_capacity[d] * y[d] for d in range(m)),
                     name="depot_capacity")
    model.addConstrs((x.sum("*", i, d) == w[i, d]
                      for i in range(n) for d in range(m)), name="incoming")
    model.addConstrs((x.sum(i, "*", d) == w[i, d]
                      for i in range(n) for d in range(m)), name="outgoing")
    model.addConstrs((v.sum(i, "*", "*") == v.sum("*", i, "*") - demand[i]
                      for i in range(n)), name="load_balance")
    model.addConstrs((v[i, j, d] <= capacity[d] * x[i, j, d]
                      for i, j, d in arcs), name="vehicle_capacity")
    model.update()
    return model, {"y": y, "w": w, "x": x, "v": v, "costs": costs}


def extract_routes(n, m, selected):
    """Recover depot-to-depot routes and detect disconnected selected arcs."""
    successors = {}
    for i, j, d in selected:
        successors.setdefault((d, i), []).append(j)
    routes, traversed, errors = [], set(), []
    for d in range(m):
        depot = n + d
        for first in sorted(successors.get((d, depot), [])):
            path, current, previous = [], first, depot
            while current != depot:
                arc = (previous, current, d)
                if current < 0 or current >= n or arc in traversed or current in path:
                    errors.append(f"Invalid route from depot {d}.")
                    break
                traversed.add(arc)
                path.append(current)
                following = successors.get((d, current), [])
                if len(following) != 1:
                    errors.append(f"Customer {current} has {len(following)} successors.")
                    break
                previous, current = current, following[0]
            else:
                traversed.add((previous, depot, d))
                routes.append({"depot": d, "customers": path})
    if traversed != selected:
        errors.append("Selected arcs include disconnected or untraversed arcs.")
    return routes, errors


def validate_solution(data, opened, assignments, routes, objective, rounding):
    """Check route feasibility and recompute the objective from coordinates."""
    n, m, depots, customers, capacity, depot_capacity, demand, opening, fixed, _ = data
    errors, visits = [], []
    depot_loads = [0] * m
    travel_cost = 0
    if len(set(opened)) != len(opened) or any(d < 0 or d >= m for d in opened):
        errors.append("Invalid opened depot list.")
    if len(assignments) != n or any(d not in opened for d in assignments):
        errors.append("Every customer must be assigned to one open depot.")
    for route in routes:
        d, sequence = route["depot"], route["customers"]
        if d < 0 or d >= m or d not in opened or not sequence or any(i < 0 or i >= n for i in sequence):
            errors.append("A route has an invalid depot or customer sequence.")
            continue
        visits.extend(sequence)
        route_load = sum(demand[i] for i in sequence)
        depot_loads[d] += route_load
        if route_load > capacity[d]:
            errors.append(f"Vehicle capacity exceeded at depot {d}.")
        if any(i >= len(assignments) or assignments[i] != d for i in sequence):
            errors.append(f"Route and assignment disagree at depot {d}.")
        coordinates = [depots[d]] + [customers[i] for i in sequence] + [depots[d]]
        travel_cost += sum(distance_cost(a, b, rounding)
                           for a, b in zip(coordinates[:-1], coordinates[1:]))
    if Counter(visits) != Counter(range(n)):
        errors.append("Customers are missing or visited more than once.")
    if any(load > depot_capacity[d] for d, load in enumerate(depot_loads)):
        errors.append("Depot capacity exceeded.")
    opening_cost = sum(opening[d] for d in opened if 0 <= d < m)
    vehicle_cost = fixed * len(routes)
    total = opening_cost + travel_cost + vehicle_cost
    if not math.isclose(total, objective, rel_tol=1e-9, abs_tol=1e-5):
        errors.append("Recomputed route objective differs from the MIP objective.")
    return {"feasible": not errors, "errors": errors,
            "recomputed_objective": total, "opening_cost": opening_cost,
            "travel_cost": travel_cost, "vehicle_cost": vehicle_cost,
            "depot_loads": depot_loads}


def solve_instance(data, time_limit=3600, log_file=None,
                   rounding="ceil", output_dir=None, mip_gap=0.01):
    """Solve the CLRP and return bounds, decisions and a feasibility check."""
    start = time.perf_counter()
    model, variables = build_model(data, time_limit, log_file, rounding, mip_gap)
    build_time = time.perf_counter() - start
    if output_dir is not None:
        model.write(str(Path(output_dir) / "model.lp"))
    model.optimize()
    result = {"status": model.Status, "has_solution": model.SolCount > 0,
              "objective": None, "bound": None, "mip_gap": None,
              "runtime": model.Runtime, "build_time": build_time,
              "nodes": model.NodeCount, "variables": model.NumVars,
              "binaries": model.NumBinVars, "constraints": model.NumConstrs,
              "rounding": rounding, "time_limit": time_limit,
              "threads": model.Params.Threads, "mip_gap_target": model.Params.MIPGap,
              "gurobi_version": list(gp.gurobi.version()), "validation": None}
    if model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        bound = model.ObjBound
        result["bound"] = bound if math.isfinite(bound) else None
    if model.SolCount:
        n, m = data[:2]
        opened = [d for d in range(m) if variables["y"][d].X > 0.5]
        assignments = []
        for i in range(n):
            chosen = [d for d in range(m) if variables["w"][i, d].X > 0.5]
            assignments.append(chosen[0] if len(chosen) == 1 else -1)
        selected = {arc for arc, variable in variables["x"].items() if variable.X > 0.5}
        routes, errors = extract_routes(n, m, selected)
        validation = validate_solution(data, opened, assignments, routes, model.ObjVal, rounding)
        validation["errors"].extend(errors)
        validation["gurobi_constraint_violation"] = model.ConstrVio
        validation["gurobi_bound_violation"] = model.BoundVio
        validation["gurobi_integrality_violation"] = model.IntVio
        if max(model.ConstrVio, model.BoundVio, model.IntVio) > 1e-5:
            validation["errors"].append("Gurobi solution violations exceed 1e-5.")
        validation["feasible"] = not validation["errors"]
        result.update(objective=model.ObjVal, mip_gap=model.MIPGap,
                      open_depots=opened, assignments=assignments, routes=routes,
                      selected_arcs=[list(arc) for arc in sorted(selected)],
                      arc_loads=[list(arc) + [variable.X]
                                 for arc, variable in variables["v"].items()
                                 if variable.X > 1e-8], validation=validation)
        if output_dir is not None:
            model.write(str(Path(output_dir) / "solution.sol"))
    result["total_time"] = time.perf_counter() - start
    model.dispose()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, help="Prodhon .dat instance")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rounding", default="ceil", choices=["truncate", "ceil", "nearest"])
    parser.add_argument("--time_limit", type=float, default=3600)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this experiment inside a Slurm compute-node job.")
    base_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(base_dir / "neo-lrp"))
    from core.dataparse import create_data

    instance = Path(args.instance).resolve()
    data = create_data(str(instance))
    if data[0] not in (20, 50):
        parser.error("This experiment is restricted to 20- and 50-customer instances.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = solve_instance(data, args.time_limit,
                            output / "gurobi.log", args.rounding, output)
    result.update(instance=instance.name, instance_path=str(instance),
                  instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"))
    with (output / "result.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in
                      ("instance", "status", "objective", "bound", "mip_gap", "runtime", "validation")},
                     indent=2, allow_nan=False))
    if result["validation"] is not None and not result["validation"]["feasible"]:
        raise SystemExit("The returned solution failed validation.")


if __name__ == "__main__":
    main()
