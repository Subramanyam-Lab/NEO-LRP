"""
Ablation study comparing different normalization schemes (N1-N4) for NEO-DS
on the Prodhon benchmark. N1=raw, N2=cost_over_fi, N3=minmax, N4=cost_over_fi_minmax.
Generates a latex table showing optimization gap and runtime for each scheme.
"""
import pandas as pd
import numpy as np
import os


def normalize_instance(name: str) -> str:
    name = name.replace("coord", "").replace(".dat", "").replace("_", "-")
    parts = name.split("-")
    if parts[-1].isdigit():
        parts[-1] = parts[-1] + "a"
    return "-".join(parts).replace("bis", "Bis").replace("BIS", "Bis")


def load_excel(path):
    df = pd.read_excel(path, sheet_name="results")
    df.columns = df.columns.astype(str).str.strip()
    df["gap"] = df["Optimization_gap_signed"]
    df["T_total"] = df["NN model execution time"] + df["vroom model solve time"]
    df["Instance"] = df["Instance"].apply(normalize_instance)
    return df[["Instance", "BKS", "gap", "T_total"]]


base_ds = "NEO-LRP/neo-lrp/output/P_prodhon"

ds_suffix = {
    "N1": "P_prodhon_deepsets_110000_raw_vroom.xlsx",
    "N2": "P_prodhon_deepsets_110000_cost_over_fi_vroom.xlsx",
    "N3": "P_prodhon_deepsets_110000_minmax_vroom.xlsx",
    "N4": "P_prodhon_deepsets_110000_cost_over_fi_minmax_vroom.xlsx"
}

ds_paths = {key: os.path.join(base_ds, fname) for key, fname in ds_suffix.items()}

print("NEO-DS paths:")
for k, v in ds_paths.items():
    print(f"  {k}: {v}")

data = {}
for tag, path in ds_paths.items():
    data[tag] = load_excel(path)

block_groups = {
    "20-5": "20-5",
    "50-5": "50-5",
    "100-5": "100",
    "100-10": "100",
    "200-10": "200-10"
}
block_order = ["20-5", "50-5", "100", "200-10"]


def get_block_prefix(instance):
    for key, val in block_groups.items():
        if instance.startswith(key):
            return val
    return "OTHER"


def sort_instances(instances):
    def parse_key(s):
        prefix = get_block_prefix(s)
        idx = block_order.index(prefix) if prefix in block_order else len(block_order)
        rest = s[len(prefix):]
        return (idx, rest)
    return sorted(instances, key=parse_key)


all_instances = sort_instances(set().union(*[df["Instance"] for df in data.values()]))

lines = []
current_block = None
block_gaps = {N: [] for N in ["N1","N2","N3","N4"]}
block_times = {N: [] for N in ["N1","N2","N3","N4"]}
overall_gaps = {N: [] for N in ["N1","N2","N3","N4"]}
overall_times = {N: [] for N in ["N1","N2","N3","N4"]}

for inst in all_instances:
    block_prefix = get_block_prefix(inst)
    if current_block and block_prefix != current_block:
        avg_row = [r"\textbf{Average}", ""]
        for N in ["N1", "N2", "N3", "N4"]:
            gvals = block_gaps[N]
            tvals = block_times[N]
            if gvals:
                gmean, tmean = np.mean(gvals), np.mean(tvals)
                avg_row.extend([f"\\textbf{{{gmean:.2f}}}", f"\\textbf{{{tmean:.2f}}}"])
            else:
                avg_row.extend(["", ""])
            block_gaps[N].clear()
            block_times[N].clear()
        lines.append("\\midrule\n" + " & ".join(avg_row) + " \\\\")
        lines.append("\\midrule")
    current_block = block_prefix
    bks = np.nan
    for df in data.values():
        sub = df[df["Instance"] == inst]
        if not sub.empty:
            bks = int(sub["BKS"].iloc[0])
            break
    row = [inst, f"{int(bks)}" if not np.isnan(bks) else ""]
    for N in ["N1", "N2", "N3", "N4"]:
        df = data.get(N)
        sub = df[df["Instance"] == inst]
        if not sub.empty:
            gap = sub["gap"].iloc[0]
            t = sub["T_total"].iloc[0]
            row.extend([f"{gap:.2f}", f"{t:.2f}"])
            block_gaps[N].append(gap)
            block_times[N].append(t)
            overall_gaps[N].append(gap)
            overall_times[N].append(t)
        else:
            row.extend(["", ""])
    lines.append(" & ".join(row) + " \\\\")

if any(block_gaps.values()):
    avg_row = [r"\textbf{Average}", ""]
    for N in ["N1", "N2", "N3", "N4"]:
        gvals = block_gaps[N]
        tvals = block_times[N]
        if gvals:
            gmean, tmean = np.mean(gvals), np.mean(tvals)
            avg_row.extend([f"\\textbf{{{gmean:.2f}}}", f"\\textbf{{{tmean:.2f}}}"])
        else:
            avg_row.extend(["", ""])
    lines.append("\\midrule\n" + " & ".join(avg_row) + " \\\\")
    lines.append("\\midrule")

overall_row = [r"\textbf{Overall Avg.}", ""]
for N in ["N1", "N2", "N3", "N4"]:
    gvals = overall_gaps[N]
    tvals = overall_times[N]
    if gvals:
        gmean, tmean = np.mean(gvals), np.mean(tvals)
        overall_row.extend([f"\\textbf{{{gmean:.2f}}}", f"\\textbf{{{tmean:.2f}}}"])
    else:
        overall_row.extend(["", ""])
lines.append("\\midrule\n" + " & ".join(overall_row) + " \\\\")
lines.append("\\midrule")

header = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of normalization schemes ($N_1$–$N_4$) for NEO-DS on Prodhon benchmark.}
\label{tab:neos_norm_comparison}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l r cc cc cc cc}
\toprule
 & & \multicolumn{8}{c}{\textbf{NEO-DS}} \\
\cmidrule(lr){3-10}
Instance & BKS
& \multicolumn{2}{c}{$N_1$} & \multicolumn{2}{c}{$N_2$} & \multicolumn{2}{c}{$N_3$} & \multicolumn{2}{c}{$N_4$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
 &  & $E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{total}}$
 & $E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{total}}$
 & $E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{total}}$
 & $E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{total}}$ \\
\midrule
"""
footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

latex_text = header + "\n".join(lines) + footer
with open("neos_norm_comparison_table.tex", "w") as f:
    f.write(latex_text)

print("latex table saved as neos_norm_comparison_table.tex")
