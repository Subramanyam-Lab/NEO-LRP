"""
Generates instance-wise comparison tables (CSV and LaTeX) between FLP-VRP and NEO-DS solutions.
Analyzes opened depots, facility costs, routing costs, and customer allocations
across benchmark instances.
"""

import json, os, re, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR_FLP = Path("NEO-LRP/flpvrp/results/solutions_flpvrp")
DIR_VROOM = DIR_FLP

DIR_NEO = Path("NEO-LRP/neo-lrp/output/P_prodhon/deepsets_vroom_cost_over_fi_110000")
DIR_NEO_ACTUAL = DIR_NEO

RUN_INDEX = 1
INSTANCE_FILTER = r"^coord(20|50|100|200)-"

OUT_CSV = "surrogate_analysis.csv"
OUT_LATEX_INSTANCES_TXT = "table_value_loc_alloc_instances.tex"


def safe_read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_file(base: Path, instance_stem: str, method_dir: str, run_index: int):
    p = base / instance_stem / method_dir / f"run_{run_index}.json"
    return p if p.exists() else None


def list_instances(base: Path):
    return [child.name for child in base.iterdir() if child.is_dir()] if base.exists() else []


def binvec_from_opened_indices(opened_indices, n_candidates):
    vec = [0] * n_candidates
    for j in opened_indices:
        if 0 <= j < n_candidates:
            vec[j] = 1
    return vec


def sum_fixed_cost(vec, facility_cost):
    return sum(facility_cost[j] for j, v in enumerate(vec) if v == 1)


def norm_diff_count(vec_a, vec_b, norm="opened"):
    raw = sum(1 for a, b in zip(vec_a, vec_b) if a != b)
    if norm == "opened":
        denom = max(1, sum(vec_a) + sum(vec_b))
        return raw, raw / denom
    denom = len(vec_a)
    return raw, raw / max(1, denom)


def flatten_opened_indices(payload):
    sets = payload.get("sets", {})
    data = payload.get("data", {})
    opened_indices = sets.get("opened_indices", [])
    facility_cost = data.get("facility_cost")
    return opened_indices, facility_cost


def get_assignment_vector(payload):
    sets = payload.get("sets", {})
    assign = sets.get("assign_depot_per_customer")
    return list(assign) if isinstance(assign, (list, tuple)) else None


def extract_size_from_stem(stem: str):
    m = re.match(r"^coord(\d+)-", stem)
    return int(m.group(1)) if m else None

rows = []

all_stems = sorted(
    set(list_instances(DIR_FLP))
    | set(list_instances(DIR_NEO))
    | set(list_instances(DIR_NEO_ACTUAL))
)

if INSTANCE_FILTER:
    rx = re.compile(INSTANCE_FILTER)
    all_stems = [s for s in all_stems if rx.search(s)]

for stem in all_stems:

    flp_json_path = find_file(DIR_FLP, stem, "FLPVRP", RUN_INDEX)
    vroom_json_path = find_file(DIR_VROOM, stem, "VROOM", RUN_INDEX)

    neo_json_path = DIR_NEO / stem / f"run_{RUN_INDEX}.json"
    if not neo_json_path.exists():
        neo_json_path = None

    neo_actual_path = DIR_NEO_ACTUAL / stem / f"run_{RUN_INDEX}.json"
    if not neo_actual_path.exists():
        neo_actual_path = None

    flp = safe_read_json(flp_json_path) if flp_json_path else None
    vroom = safe_read_json(vroom_json_path) if vroom_json_path else None
    neo = safe_read_json(neo_json_path) if neo_json_path else None
    neo_actual = safe_read_json(neo_actual_path) if neo_actual_path else None

    if not (flp and vroom and neo and neo_actual):
        continue

    flp_opened_idx, flp_fac_cost = flatten_opened_indices(flp)
    neo_opened_idx, neo_fac_cost = flatten_opened_indices(neo)

    n_cand = len(flp_fac_cost) if flp_fac_cost else (
        max([*(flp_opened_idx or [0]), *(neo_opened_idx or [0])]) + 1
    )

    flp_vec = binvec_from_opened_indices(flp_opened_idx, n_cand)
    neo_vec = binvec_from_opened_indices(neo_opened_idx, n_cand)

    # opened depots
    dep_flp = sum(flp_vec)
    dep_neo = sum(neo_vec)
    dep_diff = dep_neo - dep_flp

    # S-norm difference
    s_raw, s_norm = norm_diff_count(flp_vec, neo_vec, norm="opened")

    # facility costs
    fac_cost = flp_fac_cost or neo_fac_cost or []
    if fac_cost:
        f_flp = sum_fixed_cost(flp_vec, fac_cost)
        f_neo = sum_fixed_cost(neo_vec, fac_cost)
        f_diff = f_neo - f_flp
    else:
        f_flp = f_neo = f_diff = np.nan

    identical_locations = (flp_vec == neo_vec)

    flp_assign = get_assignment_vector(flp)
    neo_assign = get_assignment_vector(neo)

    alloc_norm_sameS = np.nan
    ident_alloc_sameS = None

    if flp_assign is not None and neo_assign is not None and len(flp_assign) == len(neo_assign):
        n_c = len(flp_assign)
        alloc_count = sum(1 for i in range(n_c) if flp_assign[i] != neo_assign[i])
        alloc_norm = alloc_count / max(1, n_c)
        if identical_locations:
            alloc_norm_sameS = alloc_norm
            ident_alloc_sameS = (alloc_count == 0)
    else:
        alloc_norm = np.nan

    r_flp_travel = vroom.get("objective", {}).get("routing_total")
    num_routes_flp = sum(
        info.get("num_routes") 
        for info in vroom.get("per_depot", {}).values()
    )
    r_flp = r_flp_travel

    r_neo = neo_actual.get("objective", {}).get("vrp_actual")
    r_diff = r_neo - r_flp if not (np.isnan(r_flp) or np.isnan(r_neo)) else np.nan

    rows.append({
        "instance": f"{stem}.dat",
        "size": extract_size_from_stem(stem),

        "OpenDepots_FLPVRP": dep_flp,
        "OpenDepots_NEOLRP": dep_neo,
        "OpenDepots_Diff_NEOLRP_minus_FLPVRP": dep_diff,

        "OpenedSet_DiffNorm": s_norm,

        "FixedFacilityCost_FLPVRP": f_flp,
        "FixedFacilityCost_NEOLRP": f_neo,
        "FixedFacilityCost_Diff_NEOLRP_minus_FLPVRP": f_diff,

        "Routing_FLPVRP": r_flp,
        "Routing_NEOLRP": r_neo,
        "Routing_Diff_NEOLRP_minus_FLPVRP": r_diff,

        "Alloc_DiffNorm_ifSameOpen": alloc_norm_sameS,
        "Identical_Allocation_ifSameOpen": ("" if ident_alloc_sameS is None else bool(ident_alloc_sameS)),
    })

df = pd.DataFrame(rows).sort_values(["size", "instance"])
print(df)

df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")

def yesno(b):
    if b == "": 
        return "—"
    return "Yes" if bool(b) else "No"


def _display_instance_name(name: str):
    base = name
    if base.endswith(".dat"):
        base = base[:-4]
    if base.startswith("coord"):
        base = base[len("coord"):]
        if base.startswith("-"):
            base = base[1:]
    return base


def write_instance_table(df: pd.DataFrame, path: str):
    df2 = df[df["size"].isin([20, 50, 100, 200])].copy().sort_values(["size", "instance"])

    def fmt_i(x): return f"{int(x)}" if pd.notna(x) else ""
    def fmt_f0(x): return f"{int(round(x))}" if pd.notna(x) else ""
    def fmt_f3(x): return f"{x:.3f}" if pd.notna(x) else "—"

    L = []
    L.append(r"\begin{table}")
    L.append(r"\centering")
    L.append(r"\caption{Instance-wise comparison of FLP-VRP and NEO-DS on the $\mathbb{P}$ benchmark set of " + "\n" +
             r"\citet{prins2004nouveaux}. " + "\n" +
             r"For each instance: " + "\n" +
             r"$n^{\text{open}}$ is the number of opened depots; " + "\n" +
             r"$F$ is the total fixed facility cost; " + "\n" +
             r"$R$ is the total routing cost. " + "\n" +
             r"Differences are defined as NEO minus FLP: " + "\n" +
             r"$\Delta n^{\text{open}} = n^{\text{open}}_{\text{NEO}} - n^{\text{open}}_{\text{FLP}}$, " + "\n" +
             r"$\Delta F = F_{\text{NEO}} - F_{\text{FLP}}$, " + "\n" +
             r"$\Delta R = R_{\text{NEO}} - R_{\text{FLP}}$. " + "\n" +
             r"$\|\Delta \mathcal{D}\|_{\text{norm}}$ measures the normalized symmetric difference between opened-depot sets. " + "\n" +
             r"$\|\Delta A\|_{\text{norm}}$ measures the normalized Hamming distance between customer assignments, computed only when both methods open the same depot set.}")
    L.append(r"\label{tab:flp_vrp}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{2pt}")

    L.append(r"\begin{tabular}{l r r r  r r r  r r r r  r c}")
    L.append(r"\toprule")

    L.append(
        r"\textbf{Instance} & "
        r"\multicolumn{3}{c}{\textbf{FLP--VRP}} & "
        r"\multicolumn{3}{c}{\textbf{NEO--DS}} & "
        r"\multicolumn{4}{c}{\textbf{Diff.\ (NEO$-$FLP)}} & "
        r"\multicolumn{2}{c}{\textbf{Alloc (same $\mathcal{D}$)}} \\"
    )

    L.append(r"\cmidrule(lr){2-4}")
    L.append(r"\cmidrule(lr){5-7}")
    L.append(r"\cmidrule(lr){8-11}")
    L.append(r"\cmidrule(lr){12-13}")

    L.append(
        r"& $n^{open}$ & $F$ & $R$ & "
        r"$n^{open}$ & $F$ & $R$ & "
        r"$\Delta n^{open}$ & $\Delta F$ & $\Delta R$ & $\|\Delta \mathcal{D}\|_{\text{norm}}$ & "
        r"$\|\Delta A\|_{\text{norm}}$ & Identical \\"
    )
    L.append(r"\midrule")

    for _, r in df2.iterrows():
        inst_clean = _display_instance_name(str(r["instance"]))

        L.append(
            r"%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\"
            % (
                inst_clean,

                fmt_i(r["OpenDepots_FLPVRP"]),
                fmt_f0(r["FixedFacilityCost_FLPVRP"]),
                fmt_f0(r["Routing_FLPVRP"]),

                fmt_i(r["OpenDepots_NEOLRP"]),
                fmt_f0(r["FixedFacilityCost_NEOLRP"]),
                fmt_f0(r["Routing_NEOLRP"]),

                fmt_i(r["OpenDepots_Diff_NEOLRP_minus_FLPVRP"]),
                fmt_f0(r["FixedFacilityCost_Diff_NEOLRP_minus_FLPVRP"]),
                fmt_f0(r["Routing_Diff_NEOLRP_minus_FLPVRP"]),
                fmt_f3(r["OpenedSet_DiffNorm"]),

                fmt_f3(r["Alloc_DiffNorm_ifSameOpen"]),
                yesno(r["Identical_Allocation_ifSameOpen"]),
            )
        )

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote instance-wise table to %s" % path)


write_instance_table(df, OUT_LATEX_INSTANCES_TXT)
