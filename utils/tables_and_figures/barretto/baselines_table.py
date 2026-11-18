"""
Generate latex table comparing NEO-DS against baseline methods (hcc-500k, tsba_speed, grasp/vns, halns, GANCP) on the barreto benchmark. reads neo-ds
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
            "neo-lrp/output/B_barreto/B_barreto_deepsets_110000_cost_over_fi_vroom.xlsx")

mapping = {
    "coordChrist50.dat": "Christofides69-50x5",
    "coordChrist75.dat": "Christofides69-75x10",
    "coordChrist100.dat": "Christofides69-100x10",
    "coordDas88.dat": "Daskin95-88x8",
    "coordDas150.dat": "Daskin95-150x10",
    "coordGaspelle.dat": "Gaskell67-21x5",
    "coordGaspelle2.dat": "Gaskell67-22x5",
    "coordGaspelle3.dat": "Gaskell67-29x5",
    "coordGaspelle4.dat": "Gaskell67-32x5-1",
    "coordGaspelle5.dat": "Gaskell67-32x5-2",
    "coordGaspelle6.dat": "Gaskell67-36x5",
    "coordMin27.dat": "Min92-27x5",
    "coordMin134.dat": "Min92-134x8",
}


def normalize_instance(name: str) -> str:
    return mapping.get(name, name)


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

with (Path(__file__).resolve().parent.parent / "sobhanan_gancp_barreto.json").open() as f:
    gancp_results = json.load(f)["instances"]
gancp_values = {}


def format_ds(instance, value, column):
    sd = neo_ds_std.get(instance, (None, None, None))[column]
    if sd is None or pd.isna(sd):
        return f"{value:.2f}"
    return f"{value:.2f}" + rf"\(\pm {sd:.2f}\)"


latex_text = r"""
\begin{sidewaystable}
\caption{Comparison with State-of-the-Art Heuristics on Barreto Benchmark}\label{tab:optimization_barreto} %
\scriptsize
\setlength{\tabcolsep}{1pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l r rr rr rr rr rr rrr@{}}
\toprule
& & \multicolumn{2}{c}{HCC-500K}
  & \multicolumn{2}{c}{TSBA$_\text{speed}$}
  & \multicolumn{2}{c}{GRASP/VNS}
  & \multicolumn{2}{c}{HALNS}
  & \multicolumn{2}{c}{GANCP}
  & \multicolumn{3}{c}{NEO-DS} \\
\cmidrule(r){3-4}\cmidrule(r){5-6}\cmidrule(r){7-8}\cmidrule(r){9-10}\cmidrule(r){11-12}\cmidrule(l){13-15}
Instance & BKS & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
  & $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s) \\
\midrule
Christofides69-50x5 & 565.6  & 0.00 & 73  & 0.00 & 2  & 2.55 & 7  & 0.00 & 54.96 & - & - & - \\
Christofides69-75x10 & 844.4  & 1.24 & 207  & 0.72 & 12  & 0.92 & 27  & 0.53 & 120.92 & - & - & - \\
Christofides69-100x10 & 833.4  & 0.24 & 403  & 0.14 & 26  & 1.31 & 64  & 0.07 & 196.76 & - & - & - \\
Daskin95-88x8 & 355.8  & 0.01 & 250  & 0.03 & 6  & 0.90 & 46  & 0.01 & 91.74 & - & - & - \\
Daskin95-150x10 & 43919.9  & 1.31 & 613  & 0.38 & 52  & 0.82 & 345  & 0.00 & 410.93 & - & - & - \\
Gaskell67-21x5 & 424.9  & 0.00 & 25  & 0.00 & 1  & 0.00 & 1  & 0.00 & 24.90 & - & - & - \\
Gaskell67-22x5 & 585.1  & 0.00 & 21  & 0.00 & 0  & 0.00 & 1  & 0.00 & 23.67 & - & - & - \\
Gaskell67-29x5 & 512.1  & 0.00 & 40  & 0.00 & 1  & 0.00 & 2  & 0.00 & 29.05 & - & - & - \\
Gaskell67-32x5-1 & 562.2  & 0.00 & 58  & 0.00 & 1  & 0.00 & 2  & 0.00 & 34.49 & - & - & - \\
Gaskell67-32x5-2 & 504.3  & 0.01 & 55  & 0.00 & 1  & 0.00 & 2  & 0.01 & 32.52 & - & - & - \\
Gaskell67-36x5 & 460.4  & 0.01 & 61  & 0.00 & 1  & 0.00 & 2  & 0.01 & 35.88 & - & - & - \\
Min92-27x5 & 3062.0  & 0.00 & 38  & 0.00 & 1  & 0.00 & 2  & 0.00 & 28.55 & - & - & - \\
Min92-134x8 & 5709.0  & 0.41 & 460  & 0.23 & 29  & 1.11 & 166  & 0.00 & 240.03 & - & - & - \\
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
  & \multicolumn{3}{r}{3.00} \\
\bottomrule
\end{tabular}%
}
\end{sidewaystable}
"""


def update_line(line):
    match = re.match(r"([A-Za-z0-9-]+)", line.strip())
    if match:
        instance = match.group(1)
        if instance not in mapping.values():
            return line
        parts = [p.strip() for p in line.split("&")]
        parts[9] = f" {float(parts[9]):.0f} "
        if instance in neo_ds_dict:
            gap, t_la, t_total = neo_ds_dict[instance]
            parts[10] = format_ds(instance, gap, 0)
            parts[11] = format_ds(instance, t_la, 1)
            parts[12] = format_ds(instance, t_total, 2)
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
        if len(parts) >= 13 and re.match(r"^[A-Za-z0-9-]+$", parts[0]):
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


def fmt(val, dec=2):
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.{dec}f}"


def fmt_int(val):
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.0f}"


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

avg_row = (
    "\\textbf{average} & "
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
with open("barreto_sota_table_updated.tex", "w") as f:
    f.write(new_latex)

print("updated barreto latex table written with averages, bold formatting, and midrule.")
