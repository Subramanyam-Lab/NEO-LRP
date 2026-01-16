"""
Generate latex table comparing NEO-DS against state-of-the-art heuristics
(HCC-500K, TSBA_speed, GRASP/VNS) on the Prodhon benchmark. Reads NEO-DS
results from Excel and merges them into a pre-defined latex template.
"""
import pandas as pd
import re

excel_ds = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_deepsets_110000_cost_over_fi_vroom.xlsx"


def normalize_instance(name: str) -> str:
    name = name.replace("coord", "").replace(".dat", "")
    parts = name.split("-")
    if parts[-1].isdigit():
        parts[-1] = parts[-1] + "a"
    return "-".join(parts)


df_ds = pd.read_excel(excel_ds, sheet_name="results", header=0)
df_ds.columns = df_ds.columns.astype(str).str.strip()
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

latex_text = r"""
\begin{sidewaystable}
\caption{Comparison with State-of-the-Art Heuristics on Prodhon Benchmark}
\label{tab:optimization_main_full_extended}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{HCC-500K}
   & \multicolumn{2}{c}{TSBA$_\text{speed}$}
   & \multicolumn{2}{c}{GRASP/VNS}
   & \multicolumn{3}{c}{NEO-DS} \\
\cmidrule(r){3-4}\cmidrule(r){5-6}\cmidrule(r){7-8}\cmidrule(r){9-11}
Instance & BKS
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s) \\
\midrule
20-5-1a & 54793 & 0.00 & 39 & 0.00 & 0.80 & 0.08 & 0.78 &  &  &  \\
20-5-1b & 39104 & 0.00 & 54 & 0.00 & 0.53 & 0.00 & 0.67 &  &  &  \\
20-5-2a & 48908 & 0.00 & 38 & 0.00 & 0.74 & 0.00 & 0.76 &  &  &  \\
20-5-2b & 37542 & 0.00 & 67 & 0.00 & 0.51 & 0.00 & 0.65 &  &  &  \\
\midrule
Average &  & 0.00 & 49.50 & 0.00 & 0.65 & 0.02 & 0.71 &  &  &  \\
\midrule
50-5-1a & 90111 & 0.00 & 101 & 0.00 & 2.48 & 0.00 & 7.95 &  &  &  \\
50-5-1b & 63242 & 0.00 & 65 & 0.00 & 2.35 & 0.00 & 8.59 &  &  &  \\
50-5-2a & 88293 & 0.32 & 99 & 0.06 & 3.32 & 0.35 & 8.52 &  &  &  \\
50-5-2b & 67308 & 0.21 & 200 & 0.14 & 3.07 & 0.54 & 9.18 &  &  &  \\
50-5-2bBIS & 51822 & 0.03 & 98 & 0.08 & 2.70 & 0.02 & 8.98 &  &  &  \\
50-5-2BIS & 84055 & 0.08 & 107 & 0.00 & 3.40 & 0.00 & 7.90 &  &  &  \\
50-5-3a & 86203 & 0.07 & 101 & 0.19 & 3.34 & 0.19 & 7.78 &  &  &  \\
50-5-3b & 61830 & 0.00 & 137 & 0.01 & 2.35 & 0.00 & 7.59 &  &  &  \\
\midrule
Average &  & 0.09 & 113.50 & 0.06 & 2.88 & 0.14 & 8.31 &  &  &  \\
\midrule
100-5-1a & 274814 & 0.56 & 520 & 0.37 & 15.14 & 0.44 & 70.15 &  &  &  \\
100-5-1b & 213568 & 0.69 & 1190 & 0.50 & 11.68 & 0.38 & 70.81 &  &  &  \\
100-5-2a & 193671 & 0.12 & 463 & 0.07 & 11.86 & 0.23 & 82.00 &  &  &  \\
100-5-2b & 157095 & 0.04 & 859 & 0.05 & 8.11 & 0.07 & 61.93 &  &  &  \\
100-5-3a & 200079 & 0.21 & 454 & 0.21 & 14.05 & 0.24 & 64.37 &  &  &  \\
100-5-3b & 152441 & 0.30 & 684 & 0.03 & 8.39 & 1.03 & 57.29 &  &  &  \\
100-10-1a & 287661 & 4.28 & 210 & 0.24 & 25.54 & 0.61 & 78.81 &  &  &  \\
100-10-1b & 230989 & 4.03 & 188 & 0.47 & 16.57 & 1.19 & 87.95 &  &  &  \\
100-10-2a & 243590 & 0.80 & 136 & 0.05 & 21.16 & 2.06 & 75.65 &  &  &  \\
100-10-2b & 203988 & 0.25 & 261 & 0.00 & 10.93 & 1.23 & 67.50 &  &  &  \\
100-10-3a & 250882 & 1.59 & 202 & 0.93 & 22.60 & 3.83 & 71.87 &  &  &  \\
100-10-3b & 203114 & 1.51 & 224 & 0.29 & 14.88 & 5.53 & 79.76 &  &  &  \\
\midrule
Average &  & 1.20 & 449.25 & 0.27 & 15.08 & 1.40 & 72.34 &  &  &  \\
\midrule
200-10-1a & 474702 & 1.79 & 752 & 0.65 & 179.62 & 3.19 & 752.03 &  &  &  \\
200-10-1b & 375177 & 1.43 & 1346 & 0.42 & 115.72 & 2.74 & 735.75 &  &  &  \\
200-10-2a & 448077 & 0.82 & 1201 & 0.35 & 147.04 & 0.38 & 642.16 &  &  &  \\
200-10-2b & 373696 & 0.65 & 1349 & 0.14 & 69.52 & 0.23 & 683.19 &  &  &  \\
200-10-3a & 469433 & 2.12 & 1251 & 0.49 & 176.25 & 0.48 & 661.82 &  &  &  \\
200-10-3b & 362320 & 2.01 & 1137 & 0.14 & 67.58 & 0.45 & 818.25 &  &  &  \\
\midrule
Average &  & 1.47 & 1172.67 & 0.37 & 125.96 & 1.25 & 715.53 &  &  &  \\
\midrule
\multicolumn{2}{l}{Processor}
  & \multicolumn{2}{r}{Opteron 275}
  & \multicolumn{2}{r}{Xeon E5-2670}
  & \multicolumn{2}{r}{Xeon E5-2430v2}
  & \multicolumn{3}{r}{Xeon Gold 6248R} \\
\multicolumn{2}{l}{GHz}
  & \multicolumn{2}{r}{2.2}
  & \multicolumn{2}{r}{2.6}
  & \multicolumn{2}{r}{2.5}
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
            parts[8] = f" {gap:.2f} "
            parts[9] = f" {t_la:.2f} "
            parts[10] = f" {t_total:.2f} "
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
            parts[8] = f" {gap:.2f} "
            parts[9] = f" {t_la:.2f} "
            parts[10] = f" {t_total:.2f} "
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

with open("prodhon_sota_table_updated.tex", "w") as f:
    f.write(new_latex)

print("Merged and updated latex table written.")
