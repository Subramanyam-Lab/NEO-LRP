import json, os, re, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR_FLP = Path("/storage/group/azs7266/default/wzk5140/MLforVRP/Codes/vpr_model/NEOS-LRP-Codes_WK/flp/results/solutions_flpvrp")
DIR_NEO = Path("/storage/group/azs7266/default/wzk5140/MLforVRP/Codes/vpr_model/NEOS-LRP-Codes_WK/neos/prodhon/prodhon_solutions_for_plots_cost_over_fi")

RUN_INDEX = 1
INSTANCE_FILTER = r"^coord(20|50|100|200)-"

OUT_CSV = "analysis.csv"
OUT_LATEX_TEX = "table_value_loc_alloc.tex"
OUT_LATEX_INSTANCES_TEX = "table_value_loc_alloc_instances.tex"

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
    stems = []
    if not base.exists():
        return stems
    for child in base.iterdir():
        if child.is_dir():
            stems.append(child.name)
    return stems

def binvec_from_opened_indices(opened_indices, n_candidates):
    vec = [0]*n_candidates
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
    elif norm == "candidates":
        denom = len(vec_a)
        return raw, raw / max(1, denom)
    else:
        return raw, None

def flatten_opened_indices(payload):
    sets = payload.get("sets", {})
    data = payload.get("data", {})
    opened_indices = sets.get("opened_indices", [])
    facility_cost = data.get("facility_cost")
    return opened_indices, facility_cost

def get_assignment_vector(payload):
    """Returns list of depot indices per customer (length = |V_C|)."""
    sets = payload.get("sets", {})
    assign = sets.get("assign_depot_per_customer")
    return list(assign) if isinstance(assign, (list, tuple)) else None

def extract_size_from_stem(stem: str):
    m = re.match(r"^coord(\d+)-", stem)
    return int(m.group(1)) if m else None

rows = []
all_stems = sorted(set(list_instances(DIR_FLP)) | set(list_instances(DIR_NEO)))
if INSTANCE_FILTER:
    rx = re.compile(INSTANCE_FILTER)
    all_stems = [s for s in all_stems if rx.search(s)]

for stem in all_stems:
    flp_json_path = find_file(DIR_FLP, stem, "FLPVRP", RUN_INDEX)
    # neo_json_path = find_file(DIR_NEO, stem, "NEO_LRP", RUN_INDEX)

    neo_json_path = DIR_NEO / stem / f"run_{RUN_INDEX}.json"
    if not neo_json_path.exists():
        neo_json_path = None


    flp = safe_read_json(flp_json_path) if flp_json_path else None
    neo = safe_read_json(neo_json_path) if neo_json_path else None
    if not (flp and neo):
        continue

    # opened sets and facility cost
    flp_opened_idx, flp_fac_cost = flatten_opened_indices(flp)
    neo_opened_idx, neo_fac_cost = flatten_opened_indices(neo)

    n_cand = len(flp_fac_cost) if flp_fac_cost else max([*(flp_opened_idx or [0]), *(neo_opened_idx or [0])]) + 1
    flp_vec = binvec_from_opened_indices(flp_opened_idx, n_cand)
    neo_vec = binvec_from_opened_indices(neo_opened_idx, n_cand)

    a1_flp = sum(flp_vec)
    a1_neo = sum(neo_vec)
    a1_diff = a1_neo - a1_flp

    a2_raw, a2_norm = norm_diff_count(flp_vec, neo_vec, norm="opened")

    fac_cost = flp_fac_cost or neo_fac_cost or []
    if fac_cost:
        a3_flp_fixed = sum_fixed_cost(flp_vec, fac_cost)
        a3_neo_fixed = sum_fixed_cost(neo_vec, fac_cost)
        a3_diff = a3_neo_fixed - a3_flp_fixed
    else:
        a3_flp_fixed = np.nan
        a3_neo_fixed = np.nan
        a3_diff = np.nan

    identical_locations = (flp_vec == neo_vec)

    # allocation differences (customer to depot assignments)
    flp_assign = get_assignment_vector(flp)
    neo_assign = get_assignment_vector(neo)
    alloc_count = np.nan
    alloc_norm = np.nan
    alloc_norm_sameS = np.nan
    ident_alloc_sameS = None

    if flp_assign is not None and neo_assign is not None and len(flp_assign) == len(neo_assign):
        n_cust = len(flp_assign)
        alloc_count = sum(1 for i in range(n_cust) if flp_assign[i] != neo_assign[i])
        alloc_norm = alloc_count / max(1, n_cust)
        if identical_locations:
            alloc_norm_sameS = alloc_norm
            ident_alloc_sameS = (alloc_count == 0)

    rows.append({
        "instance": f"{stem}.dat",
        "size": extract_size_from_stem(stem),
        "OpenDepots_FLPVRP": a1_flp,
        "OpenDepots_NEOLRP": a1_neo,
        "OpenDepots_Diff_NEOLRP_minus_FLPVRP": a1_diff,
        "OpenedSet_DiffCount": a2_raw,
        "OpenedSet_DiffNorm": a2_norm,
        "FixedFacilityCost_FLPVRP": a3_flp_fixed,
        "FixedFacilityCost_NEOLRP": a3_neo_fixed,
        "FixedFacilityCost_Diff_NEOLRP_minus_FLPVRP": a3_diff,
        "Identical_Location_Decisions": bool(identical_locations),
        # allocation metrics
        "Alloc_DiffCount": alloc_count,
        "Alloc_DiffNorm": alloc_norm,
        "Alloc_DiffNorm_ifSameOpen": alloc_norm_sameS,
        "Identical_Allocation_ifSameOpen": ("" if ident_alloc_sameS is None else bool(ident_alloc_sameS)),
    })

df = pd.DataFrame(rows).sort_values(["size", "instance"])
print(df)
if OUT_CSV:
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

def write_aggregated_table(df: pd.DataFrame, path: str):
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Location--allocation comparison of FLP--VRP vs.\ NEO--DS for $|\mathcal{V}_C|\in\{20,50\}$. "
                       r"Open depots and fixed facility costs are averaged over instances in each size class.}")
    latex_lines.append(r"\label{tab:value_loc_alloc_20_50}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lrrrrrrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"$|\mathcal{V}_C|$ & \textbf{OpenDepots (FLP--VRP)} & \textbf{OpenDepots (NEO--DS)} & $\Delta$ & "
                       r"$\|\Delta \text{opened}\|_{\mathrm{norm}}$ & \textbf{Fixed} (FLP--VRP) & \textbf{Fixed} (NEO--DS) & $\Delta$ & "
                       r"\textbf{Identical loc.\ (\%)} \\")
    latex_lines.append(r"\midrule")

    for sz in [20, 50, 100, 200]:
        g = df[df["size"] == sz]
        if g.empty:
            continue
        mean_open_flp = g["OpenDepots_FLPVRP"].mean()
        mean_open_neo = g["OpenDepots_NEOLRP"].mean()
        mean_open_diff = g["OpenDepots_Diff_NEOLRP_minus_FLPVRP"].mean()
        mean_norm_diff = g["OpenedSet_DiffNorm"].mean()
        mean_fix_flp = g["FixedFacilityCost_FLPVRP"].mean()
        mean_fix_neo = g["FixedFacilityCost_NEOLRP"].mean()
        mean_fix_diff = g["FixedFacilityCost_Diff_NEOLRP_minus_FLPVRP"].mean()
        ident_pct = 100.0 * g["Identical_Location_Decisions"].mean()

        latex_lines.append(
            (r"%d & %.2f & %.2f & %.2f & %.3f & %.0f & %.0f & %.0f & %.1f \\") % (
                sz, mean_open_flp, mean_open_neo, mean_open_diff, mean_norm_diff,
                mean_fix_flp, mean_fix_neo, mean_fix_diff, ident_pct
            )
        )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"Wrote LaTeX aggregated table to {path}")

write_aggregated_table(df, OUT_LATEX_TEX)

def yesno(b) -> str:
    if b == "":
        return "—"
    return "Yes" if bool(b) else "No"

def _display_instance_name(name: str) -> str:
    base = name
    if base.endswith(".dat"):
        base = base[:-4]
    if base.startswith("coord"):
        base = base[len("coord"):]
        if base.startswith("-"):
            base = base[1:]
    return base

def write_instance_table(df: pd.DataFrame, path: str):
    df2 = df[df["size"].isin([20, 50, 100, 200])].copy()
    df2 = df2.sort_values(["size", "instance"])

    def fmt_i(x):  return f"{int(x)}" if pd.notna(x) else ""
    def fmt_f0(x): return f"{int(round(x))}" if pd.notna(x) else ""
    def fmt_f3(x): return f"{x:.3f}" if pd.notna(x) else "—"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Instance-wise comparison of FLP--VRP and NEO--DS on $|\mathcal{V}_C|\in\{20,50\}$ instances. "
        r"Notation: $n^{\text{open}}$ = \# opened depots; $F$ = fixed facility cost; "
        r"$\Delta n^{\text{open}}$ and $\Delta F$ are NEO$-$FLP differences; "
        r"$\|\Delta S\|_{\text{norm}}$ is the normalized opened-set difference; "
        r"$\|\Delta A\|_{\text{norm}}$ is the normalized Hamming distance between customer-to-depot assignments, "
        r"reported only when $S^{\text{NEO}}=S^{\text{FLP}}$.}"
    )
    lines.append(r"\label{tab:value_loc_alloc_instances}")
    lines.append(r"\small")

    # Instance | FLP–VRP (nopen,F) | NEO–LRP (nopen,F) | Diff (delta nopen, delta F, ||delta S||_norm) | Allocation (same S): ||delta A||_norm, Identical
    lines.append(r"\begin{tabular}{l  r r  r r  r r r  r c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Instance} &"
                 r" \multicolumn{2}{c}{\textbf{FLP--VRP}} &"
                 r" \multicolumn{2}{c}{\textbf{NEO--DS}} &"
                 r" \multicolumn{3}{c}{\textbf{Diff.\ (NEO$-$FLP)}} &"
                 r" \multicolumn{2}{c}{\textbf{Allocation (same $S$)}} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}")
    lines.append(r"& $n^{\text{open}}$ & $F$ & $n^{\text{open}}$ & $F$ & $\Delta n^{\text{open}}$ & $\Delta F$ & $\|\Delta S\|_{\text{norm}}$ & $\|\Delta A\|_{\text{norm}}$ & Identical \\")
    lines.append(r"\midrule")

    for _, r in df2.iterrows():
        inst_clean = _display_instance_name(str(r["instance"]))
        lines.append(
            (r"%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\") % (
                inst_clean,
                fmt_i(r["OpenDepots_FLPVRP"]),
                fmt_f0(r["FixedFacilityCost_FLPVRP"]),
                fmt_i(r["OpenDepots_NEOLRP"]),
                fmt_f0(r["FixedFacilityCost_NEOLRP"]),
                fmt_i(r["OpenDepots_Diff_NEOLRP_minus_FLPVRP"]),
                fmt_f0(r["FixedFacilityCost_Diff_NEOLRP_minus_FLPVRP"]),
                fmt_f3(r["OpenedSet_DiffNorm"]),
                fmt_f3(r["Alloc_DiffNorm_ifSameOpen"]),
                yesno(r["Identical_Allocation_ifSameOpen"])
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX instance-wise table to %s" % path)

write_instance_table(df, OUT_LATEX_INSTANCES_TEX)
