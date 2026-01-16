"""
Ablation study comparing NEO-DS (Deep Sets architecture) against NEO-GT
(Graph Transformer architecture) on the Prodhon benchmark. Generates a
latex table showing optimization gap, LA time, and total time for each
architecture across all Prodhon instances.
"""
import pandas as pd
import re

excel_gt = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_graph_transformer_110000_cost_over_fi_vroom.xlsx"
excel_ds = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_deepsets_110000_cost_over_fi_vroom.xlsx"
df_ds = pd.read_excel(excel_ds, sheet_name="results", header=0)
df_ds.columns = df_ds.columns.astype(str).str.strip()


def normalize_instance(name: str) -> str:
    name = name.replace("coord", "").replace(".dat", "")
    parts = name.split("-")
    if parts[-1].isdigit():
        parts[-1] = parts[-1] + "a"
    return "-".join(parts)


df_ds["Instance"] = df_ds["Instance"].apply(normalize_instance)
df_ds["gap"] = df_ds["Optimization_gap_signed"]
df_ds["T_LA"] = df_ds["NN model execution time"]
df_ds["T_total"] = df_ds["NN model execution time"] + df_ds["vroom model solve time"]

neo_ds_dict = {row["Instance"]: (row["gap"], row["T_LA"], row["T_total"]) for _, row in df_ds.iterrows()}

block_groups_ds = {
    "20-5": ["20-5"],
    "50-5": ["50-5"],
    "100": ["100-5", "100-10"],
    "200-10": ["200-10"]
}

block_avgs_ds = {}
for block_name, prefixes in block_groups_ds.items():
    sub = pd.concat([df_ds[df_ds["Instance"].str.startswith(p)] for p in prefixes])
    if not sub.empty:
        block_avgs_ds[block_name] = (
            sub["gap"].mean(),
            sub["T_LA"].mean(),
            sub["T_total"].mean()
        )

df_gt = pd.read_excel(excel_gt, sheet_name="results", header=0)
df_gt.columns = df_gt.columns.astype(str).str.strip()
df_gt["Instance"] = df_gt["Instance"].apply(normalize_instance)

df_gt["gap"] = df_gt["Optimization_gap_signed"]
df_gt["T_LA"] = df_gt["NN model execution time"]
df_gt["T_total"] = df_gt["NN model execution time"] + df_gt["vroom model solve time"]

neo_gt_dict = {row["Instance"]: (row["gap"], row["T_LA"], row["T_total"]) for _, row in df_gt.iterrows()}

block_groups_gt = block_groups_ds
block_avgs_gt = {}

for block_name, prefixes in block_groups_gt.items():
    sub = pd.concat([df_gt[df_gt["Instance"].str.startswith(p)] for p in prefixes])
    if not sub.empty:
        block_avgs_gt[block_name] = (
            sub["gap"].mean(),
            sub["T_LA"].mean(),
            sub["T_total"].mean()
        )

latex_text = r"""
\begin{sidewaystable}
\caption{Comparison of NEO-DS and NEO-GT on Prodhon Benchmark}
\label{tab:optimization_neods_neogt}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{NEO-DS}
   & \multicolumn{3}{c}{NEO-GT} \\
\cmidrule(r){3-5}\cmidrule(l){6-8}
Instance & BKS
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s) \\
\midrule
20-5-1a & 54793 &  &  &  &  &  & \\
20-5-1b & 39104 &  &  &  &  &  & \\
20-5-2a & 48908 &  &  &  &  &  & \\
20-5-2b & 37542 &  &  &  &  &  & \\
\midrule
Average &  &  &  &  &  &  & \\
\midrule
50-5-1a & 90111 &  &  &  &  &  & \\
50-5-1b & 63242 &  &  &  &  &  & \\
50-5-2a & 88293 &  &  &  &  &  & \\
50-5-2b & 67308 &  &  &  &  &  & \\
50-5-2bBIS & 51822 &  &  &  &  &  & \\
50-5-2BIS & 84055 &  &  &  &  &  & \\
50-5-3a & 86203 &  &  &  &  &  & \\
50-5-3b & 61830 &  &  &  &  &  & \\
\midrule
Average &  &  &  &  &  &  & \\
\midrule
100-5-1a & 274814 &  &  &  &  &  & \\
100-5-1b & 213568 &  &  &  &  &  & \\
100-5-2a & 193671 &  &  &  &  &  & \\
100-5-2b & 157095 &  &  &  &  &  & \\
100-5-3a & 200079 &  &  &  &  &  & \\
100-5-3b & 152441 &  &  &  &  &  & \\
100-10-1a & 287661 &  &  &  &  &  & \\
100-10-1b & 230989 &  &  &  &  &  & \\
100-10-2a & 243590 &  &  &  &  &  & \\
100-10-2b & 203988 &  &  &  &  &  & \\
100-10-3a & 250882 &  &  &  &  &  & \\
100-10-3b & 203114 &  &  &  &  &  & \\
\midrule
Average &  &  &  &  &  &  & \\
\midrule
200-10-1a & 474850 &  &  &  &  &  & \\
200-10-1b & 375177 &  &  &  &  &  & \\
200-10-2a & 448077 &  &  &  &  &  & \\
200-10-2b & 373696 &  &  &  &  &  & \\
200-10-3a & 469433 &  &  &  &  &  & \\
200-10-3b & 362320 &  &  &  &  &  & \\
\midrule
Average &  &  &  &  &  &  & \\
\midrule
\multicolumn{2}{l}{Processor}
  & \multicolumn{3}{r}{Xeon Gold 6248R}
  & \multicolumn{3}{r}{Xeon Gold 6248R} \\
\multicolumn{2}{l}{GHz}
  & \multicolumn{3}{r}{3.00}
  & \multicolumn{3}{r}{3.00}\\
\bottomrule
\end{tabular}
\end{sidewaystable}
"""


def update_line(line, current_block):
    match = re.match(r"(\d{2,3}-\d{1,2}-\S+)", line.strip())
    if match:
        instance = match.group(1)
        parts = [p.strip() for p in line.split("&")]
        if instance in neo_ds_dict:
            gap, t_la, t_total = neo_ds_dict[instance]
            parts[2] = f" {gap:.2f} "
            parts[3] = f" {t_la:.2f} "
            parts[4] = f" {t_total:.2f} "
        if instance in neo_gt_dict:
            gap, t_la, t_total = neo_gt_dict[instance]
            parts[5] = f" {gap:.2f} "
            parts[6] = f" {t_la:.2f} "
            parts[7] = f" {t_total:.2f} "
        return " & ".join(parts) + r" \\"
    if line.strip().startswith("Average"):
        block_map = {
            "20-5": "20-5",
            "50-5": "50-5",
            "100-5": "100",
            "100-10": "100",
            "200-10": "200-10"
        }
        merged = block_map[current_block]
        parts = [p.strip() for p in line.split("&")]
        if merged in block_avgs_ds:
            gap, t_la, t_total = block_avgs_ds[merged]
            parts[2] = f" {gap:.2f} "
            parts[3] = f" {t_la:.2f} "
            parts[4] = f" {t_total:.2f} "
        if merged in block_avgs_gt:
            gap, t_la, t_total = block_avgs_gt[merged]
            parts[5] = f" {gap:.2f} "
            parts[6] = f" {t_la:.2f} "
            parts[7] = f" {t_total:.2f} "
        parts = [f"\\textbf{{{p}}}" if p else "" for p in parts]
        return " & ".join(parts) + r" \\"
    return line


new_lines = []
current_block = None

for line in latex_text.splitlines():
    block_match = re.match(r"(\d{2,3}-\d{1,2})", line.strip())
    if block_match:
        current_block = block_match.group(1)
    if "&" in line and not line.strip().startswith("\\"):
        new_lines.append(update_line(line, current_block))
    else:
        new_lines.append(line)

new_latex = "\n".join(new_lines)

with open("prodhon_neods_neogt_table.tex", "w") as f:
    f.write(new_latex)

print("NEO-DS and NEO-GT latex table written.")
