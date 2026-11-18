"""
Generate latex table comparing NEO-DS against state-of-the-art heuristics
(HCC-500K, TSBA_speed, GRASP/VNS, HALNS, GANCP) on the Tuzun-Burke benchmark. Reads NEO-DS
results from Excel or CSV and merges them into a pre-defined latex template with
computed averages for all methods.
"""
import pandas as pd
import re
import numpy as np
import argparse
import json
from pathlib import Path

excel_ds = (Path(__file__).resolve().parents[3] /
            "neo-lrp/output/T_tuzun/T_tuzun_deepsets_110000_cost_over_fi_vroom.xlsx")


def normalize_instance(name: str) -> str:
    return name.replace("coordP", "").replace(".dat", "")


parser = argparse.ArgumentParser(description="Generate baseline comparison table")
parser.add_argument("--results", type=str, help="NEO-DS results CSV")
args = parser.parse_args()

if args.results:
    df_ds = pd.read_csv(args.results, dtype={"Instance": str})
else:
    df_ds = pd.read_excel(excel_ds, sheet_name="results", header=0)
    df_ds.columns = df_ds.columns.astype(str).str.strip()
    df_ds["gap"] = df_ds["Optimization_gap_signed"]
    df_ds["T_LA"] = df_ds["NN model execution time"]
    df_ds["T_total"] = df_ds["NN model execution time"] + df_ds["vroom model solve time"]

df_ds["Instance"] = df_ds["Instance"].apply(normalize_instance)
neo_ds_dict = {row["Instance"]: (row["gap"], row["T_LA"], row["T_total"]) for _, row in df_ds.iterrows()}
neo_ds_std = {
    row["Instance"]: (row.get("gap_std"), row.get("T_LA_std"), row.get("T_total_std"))
    for _, row in df_ds.iterrows()
}

with (Path(__file__).resolve().parent.parent / "sobhanan_gancp_tuzun.json").open() as f:
    gancp_results = json.load(f)["instances"]
gancp_values = {}


def format_ds(instance, value, column):
    sd = neo_ds_std.get(instance, (None, None, None))[column]
    if sd is None or pd.isna(sd):
        return f"{value:.2f}"
    return f"{value:.2f}" + rf"\(\pm {sd:.2f}\)"


latex_text = r"""
\begin{sidewaystable}
\caption{Comparison with State-of-the-Art Heuristics on Tuzun-Burke Benchmark}\label{tab:optimization_tuzun} %
\scriptsize
\setlength{\tabcolsep}{1pt}
\begin{tabular}{@{}l r rr rr rr rr rr rrr@{}}
\toprule
& & \multicolumn{2}{c}{HCC-500K}
  & \multicolumn{2}{c}{TSBA$_\text{speed}$}
  & \multicolumn{2}{c}{GRASP/VNS}
  & \multicolumn{2}{c}{HALNS}
  & \multicolumn{2}{c}{GANCP}
  & \multicolumn{3}{c}{NEO-DS} \\
\cmidrule(r){3-4}\cmidrule(r){5-6}\cmidrule(r){7-8}\cmidrule(r){9-10}\cmidrule(r){11-12}\cmidrule(l){13-15}
Instance & BKS
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s) \\
\midrule
111112 & 1467.68 & 0.54 & 275 & 0.07 & 23 & 2.29 & 65 & 0.00 & 163.22 & - & - & - \\
111122 & 1448.37 & 1.13 & 321 & 0.44 & 24 & 2.05 & 81 & 0.01 & 239.21 & - & - & - \\
111212 & 1394.80 & 0.41 & 244 & 0.12 & 22 & 0.64 & 59 & 0.00 & 162.94 & - & - & - \\
111222 & 1432.29 & 0.62 & 376 & 0.03 & 20 & 2.54 & 77 & 0.02 & 234.86 & - & - & - \\
112112 & 1167.16 & 0.50 & 489 & 0.07 & 21 & 0.44 & 69 & 0.00 & 184.35 & - & - & - \\
112122 & 1102.24 & 0.01 & 373 & 0.03 & 16 & 1.84 & 100 & 0.00 & 151.51 & - & - & - \\
112212 & 791.66  & 0.02 & 739 & 0.34 & 16 & 0.39 & 97 & 0.00 & 168.88 & - & - & - \\
112222 & 728.30  & 0.00 & 384 & 0.00 & 13 & 0.18 & 91 & 0.00 & 137.87 & - & - & - \\
113112 & 1238.24 & 0.17 & 357 & 0.10 & 20 & 1.14 & 74 & 0.02 & 197.30 & - & - & - \\
113122 & 1245.30 & 0.23 & 445 & 0.11 & 19 & 0.64 & 86 & 0.01 & 150.63 & - & - & - \\
113212 & 902.26  & 0.00 & 321 & 0.06 & 16 & 0.10 & 70 & 0.00 & 141.88 & - & - & - \\
113222 & 1018.29 & 0.03 & 386 & 0.47 & 19 & 0.13 & 89 & 0.18 & 154.53 & - & - & - \\
121112 & 2237.73 & 1.81 & 944 & 1.66 & 125 & 1.73 & 574 & 0.33 & 856.79 & - & - & - \\
121122 & 2137.45 & 2.58 & 847 & 1.74 & 107 & 5.35 & 681 & 0.13 & 1011.30 & - & - & - \\
121212 & 2195.17 & 2.40 & 907 & 1.25 & 145 & 1.04 & 558 & 0.20 & 858.64 & - & - & - \\
121222 & 2214.86 & 2.18 & 860 & 1.08 & 131 & 3.19 & 662 & 0.06 & 1014.78 & - & - & - \\
122112 & 2070.43 & 1.13 & 1606 & 0.65 & 117 & 0.62 & 704 & 0.08 & 861.55 & - & - & - \\
122122 & 1685.52 & 2.76 & 941 & 0.60 & 128 & 7.56 & 927 & 0.01 & 1029.48 & - & - & - \\
122212 & 1449.62 & 0.86 & 1861 & 0.45 & 128 & 1.58 & 697 & 0.01 & 1067.59 & - & - & - \\
122222 & 1082.46 & 0.33 & 812 & 0.33 & 101 & 0.52 & 656 & 0.01 & 842.09 & - & - & - \\
123112 & 1942.23 & 1.48 & 968 & 1.24 & 120 & 5.25 & 595 & 0.55 & 1047.41 & - & - & - \\
123122 & 1910.08 & 2.21 & 740 & 1.08 & 157 & 3.26 & 770 & 0.16 & 1005.32 & - & - & - \\
123212 & 1760.20 & 0.23 & 2055 & 0.38 & 103 & 0.78 & 690 & 0.05 & 1118.70 & - & - & - \\
123222 & 1390.74 & 0.33 & 1038 & 0.16 & 91 & 4.01 & 704 & 0.00 & 663.32 & - & - & - \\
131112 & 1866.75 & 3.90 & 504 & 2.83 & 56 & 3.87 & 230 & 1.45 & 481.69 & - & - & - \\
131122 & 1819.68 & 2.07 & 635 & 0.59 & 61 & 2.51 & 259 & 0.17 & 473.25 & - & - & - \\
131212 & 1960.02 & 2.52 & 664 & 0.55 & 49 & 0.58 & 233 & 0.27 & 406.93 & - & - & - \\
131222 & 1792.77 & 2.55 & 485 & 0.76 & 63 & 0.93 & 273 & 0.14 & 377.68 & - & - & - \\
132112 & 1443.32 & 0.40 & 1049 & 0.12 & 72 & 1.15 & 260 & 0.00 & 368.65 & - & - & - \\
132122 & 1429.30 & 1.23 & 805 & 1.31 & 72 & 1.16 & 344 & 0.68 & 441.85 & - & - & - \\
132212 & 1204.42 & 0.12 & 2197 & 0.26 & 51 & 1.14 & 256 & 0.00 & 427.53 & - & - & - \\
132222 & 924.68  & 0.91 & 982 & 0.26 & 55 & 1.01 & 318 & 0.00 & 484.31 & - & - & - \\
133112 & 1694.18 & 0.37 & 1046 & 0.83 & 50 & 1.05 & 281 & 0.26 & 530.37 & - & - & - \\
133122 & 1392.00 & 0.83 & 925 & 1.34 & 56 & 1.85 & 309 & 0.13 & 473.67 & - & - & - \\
133212 & 1197.95 & 0.11 & 1375 & 0.14 & 42 & 0.37 & 276 & 0.00 & 478.82 & - & - & - \\
133222 & 1151.37 & 0.26 & 911 & 0.34 & 58 & 0.57 & 343 & 0.04 & 385.31 & - & - & - \\
\midrule
\multicolumn{2}{l}{Processor}
  & \multicolumn{2}{r}{Opteron 275}
  & \multicolumn{2}{r}{Xeon E5-2670}
  & \multicolumn{2}{r}{Xeon E5-2430v2}
  & \multicolumn{2}{r}{Ryzen 9 3900X}
  & \multicolumn{2}{r}{Core i9-11900H}
  & \multicolumn{3}{r}{Xeon Gold 6248R} \\
\multicolumn{2}{l}{GHz}
  & \multicolumn{2}{r}{2.2}
  & \multicolumn{2}{r}{2.6}
  & \multicolumn{2}{r}{2.5}
  & \multicolumn{2}{r}{3.8}
  & \multicolumn{2}{r}{2.50}
  & \multicolumn{3}{r}{3.00}\\
\bottomrule
\end{tabular}
\end{sidewaystable}
"""


def update_line(line):
    match = re.match(r"(\d{6})", line.strip())
    if match:
        instance = match.group(1)
        parts = [p.strip() for p in line.split("&")]
        parts[9] = f" {float(parts[9]):.0f} "
        if instance in neo_ds_dict:
            gap, t_la, t_total = neo_ds_dict[instance]
            parts[10] = format_ds(instance, gap, 0)
            parts[11] = format_ds(instance, t_la, 1)
            parts[12] = format_ds(instance, t_total, 2)
        line = line.rstrip("\\").strip()
        result = gancp_results[instance]
        bks = float(parts[1])
        gap = 100 * abs(result["best_cost"] - bks) / bks
        runtime = result["runtime_s"]
        gancp_values[instance] = (gap, runtime)
        parts[10:10] = [f"{gap:.2f}", f"{runtime:.2f}"]
        return " & ".join(parts) + " \\\\"
    return line


new_lines = []
rows_data = []

for line in latex_text.splitlines():
    if "&" in line and not line.strip().startswith("\\"):
        updated_line = update_line(line)
        new_lines.append(updated_line)
        parts = [p.strip().replace("\\\\", "") for p in updated_line.split("&")]
        if re.match(r"^\d{6}$", parts[0]):
            instance = parts[0]

            def num(x):
                try:
                    return float(x)
                except Exception:
                    return None

            hcc_gap, hcc_t = num(parts[2]), num(parts[3])
            tsba_gap, tsba_t = num(parts[4]), num(parts[5])
            grasp_gap, grasp_t = num(parts[6]), num(parts[7])
            halns_gap, halns_t = num(parts[8]), num(line.split("&")[9])
            ds_gap, ds_tla, ds_ttot = neo_ds_dict.get(instance, (None, None, None))
            rows_data.append((hcc_gap, hcc_t, tsba_gap, tsba_t, grasp_gap, grasp_t, halns_gap, halns_t, ds_gap, ds_tla, ds_ttot))
    else:
        new_lines.append(line)


def safe_avg(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return sum(vals) / len(vals) if vals else None


avg_hcc_gap = safe_avg([r[0] for r in rows_data])
avg_hcc_t = safe_avg([r[1] for r in rows_data])
avg_tsba_gap = safe_avg([r[2] for r in rows_data])
avg_tsba_t = safe_avg([r[3] for r in rows_data])
avg_grasp_gap = safe_avg([r[4] for r in rows_data])
avg_grasp_t = safe_avg([r[5] for r in rows_data])
avg_halns_gap = safe_avg([r[6] for r in rows_data])
avg_halns_t = safe_avg([r[7] for r in rows_data])
avg_ds_gap, avg_ds_tla, avg_ds_ttot = neo_ds_dict.get("Average", (
    safe_avg([r[8] for r in rows_data]),
    safe_avg([r[9] for r in rows_data]),
    safe_avg([r[10] for r in rows_data]),
))
avg_gancp_gap = safe_avg([r[0] for r in gancp_values.values()])
avg_gancp_t = safe_avg([r[1] for r in gancp_values.values()])


def fmt(val, dec=2):
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.{dec}f}"


def fmt_int(val):
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.0f}"


avg_row = (
    "\\textbf{Average} & "
    f"& \\textbf{{{fmt(avg_hcc_gap)}}} & \\textbf{{{fmt_int(avg_hcc_t)}}} "
    f"& \\textbf{{{fmt(avg_tsba_gap)}}} & \\textbf{{{fmt_int(avg_tsba_t)}}} "
    f"& \\textbf{{{fmt(avg_grasp_gap)}}} & \\textbf{{{fmt_int(avg_grasp_t)}}} "
    f"& \\textbf{{{fmt(avg_halns_gap)}}} & \\textbf{{{fmt_int(avg_halns_t)}}} "
    f"& \\textbf{{{fmt(avg_gancp_gap)}}} & \\textbf{{{fmt(avg_gancp_t)}}} "
    f"& \\textbf{{{format_ds('Average', avg_ds_gap, 0)}}} "
    f"& \\textbf{{{format_ds('Average', avg_ds_tla, 1)}}} "
    f"& \\textbf{{{format_ds('Average', avg_ds_ttot, 2)}}} \\\\"
)

insert_index = None
for i, line in enumerate(new_lines):
    if "Processor" in line:
        insert_index = i
        break

if insert_index is not None:
    new_lines.insert(insert_index, "\\midrule")
    new_lines.insert(insert_index, avg_row)
else:
    new_lines.append(avg_row)
    new_lines.append("\\midrule")

new_latex = "\n".join(new_lines)
with open("tuzun_sota_table_updated.tex", "w") as f:
    f.write(new_latex)

print("Updated latex table written with averages for HCC, TSBA, GRASP/VNS, HALNS, GANCP, and NEO-DS.")
