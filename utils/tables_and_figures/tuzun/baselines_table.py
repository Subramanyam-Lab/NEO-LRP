"""
Generate latex table comparing NEO-DS against state-of-the-art heuristics
(HCC-500K, TSBA_speed, GRASP/VNS) on the Tuzun-Burke benchmark. Reads NEO-DS
results from Excel and merges them into a pre-defined latex template with
computed averages for all methods.
"""
import pandas as pd
import re
import numpy as np

excel_ds = "NEO-LRP/neo-lrp/output/T_tuzun/T_tuzun_deepsets_110000_cost_over_fi_vroom.xlsx"

df_ds = pd.read_excel(excel_ds, sheet_name="results", header=0)
df_ds.columns = df_ds.columns.astype(str).str.strip()


def normalize_instance(name: str) -> str:
    return name.replace("coordP", "").replace(".dat", "")


df_ds["Instance"] = df_ds["Instance"].apply(normalize_instance)
df_ds["gap"] = df_ds["Optimization_gap_signed"]
df_ds["T_LA"] = df_ds["NN model execution time"]
df_ds["T_total"] = df_ds["NN model execution time"] + df_ds["vroom model solve time"]
neo_ds_dict = {row["Instance"]: (row["gap"], row["T_LA"], row["T_total"]) for _, row in df_ds.iterrows()}

latex_text = r"""
\begin{sidewaystable}
\caption{Comparison with State-of-the-Art Heuristics on Tuzun-Burke Benchmark}\label{tab:optimization_tuzun} %
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l r rr rr rr rrr@{}}
\toprule
& & \multicolumn{2}{c}{HCC-500K}
  & \multicolumn{2}{c}{TSBA$_\text{speed}$}
  & \multicolumn{2}{c}{GRASP/VNS}
  & \multicolumn{3}{c}{NEO-DS} \\
\cmidrule(r){3-4}\cmidrule(r){5-6}\cmidrule(r){7-8}\cmidrule(l){9-11}
Instance & BKS
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T_{\text{total}}$ (s)
& $E^{\text{gap}}_{\text{BKS}}$ (\%) & $T^{\text{LA}}$ (s) & $T_{\text{total}}$ (s) \\
\midrule
111112 & 1467.68 & 0.54 & 275 & 0.07 & 23 & 2.29 & 65 & - & - & - \\
111122 & 1448.37 & 1.13 & 321 & 3.58 & 22 & 2.05 & 81 & - & - & - \\
111212 & 1394.80 & 0.41 & 244 & 16.26 & 21 & 0.64 & 59 & - & - & - \\
111222 & 1432.29 & 0.62 & 376 & 44.54 & 16 & 2.54 & 77 & - & - & - \\
112112 & 1167.16 & 0.50 & 489 & 6.20 & 20 & 0.44 & 69 & - & - & - \\
112122 & 1102.24 & 0.01 & 373 & 18.09 & 16 & 1.84 & 100 & - & - & - \\
112212 & 791.66  & 0.02 & 739 & 83.76 & 24 & 0.39 & 97 & - & - & - \\
112222 & 728.30  & 0.00 & 384 & 96.72 & 20 & 0.18 & 91 & - & - & - \\
113112 & 1238.24 & 0.17 & 357 & 10.96 & 16 & 1.14 & 74 & - & - & - \\
113122 & 1245.30 & 0.23 & 445 & 41.52 & 13 & 0.64 & 86 & - & - & - \\
113212 & 902.26  & 0.00 & 321 & 38.17 & 19 & 0.10 & 70 & - & - & - \\
113222 & 1018.29 & 0.03 & 386 & 0.47 & 19 & 0.13 & 89 & - & - & - \\
121112 & 2237.73 & 1.81 & 944 & 1.66 & 125 & 1.73 & 574 & - & - & - \\
121122 & 2137.45 & 2.58 & 847 & 3.98 & 145 & 5.35 & 681 & - & - & - \\
121212 & 2195.17 & 2.40 & 907 & 5.07 & 117 & 1.04 & 558 & - & - & - \\
121222 & 2214.86 & 2.18 & 860 & 34.26 & 128 & 3.19 & 662 & - & - & - \\
122112 & 2070.43 & 1.13 & 1606 & 5.03 & 120 & 0.62 & 704 & - & - & - \\
122122 & 1685.52 & 2.76 & 941 & 4.82 & 103 & 7.56 & 927 & - & - & - \\
122212 & 1449.62 & 0.86 & 1861 & 50.01 & 107 & 1.58 & 697 & - & - & - \\
122222 & 1082.46 & 0.33 & 812 & 106.82 & 131 & 0.52 & 656 & - & - & - \\
123112 & 1942.23 & 1.48 & 968 & 12.70 & 128 & 5.25 & 595 & - & - & - \\
123122 & 1910.08 & 2.21 & 740 & 43.14 & 101 & 3.26 & 770 & - & - & - \\
123212 & 1760.20 & 0.23 & 2055 & 9.69 & 157 & 0.78 & 690 & - & - & - \\
123222 & 1390.74 & 0.33 & 1038 & 0.16 & 91 & 4.01 & 704 & - & - & - \\
131112 & 1866.75 & 3.90 & 504 & 2.83 & 56 & 3.87 & 230 & - & - & - \\
131122 & 1819.68 & 2.07 & 635 & 8.31 & 49 & 2.51 & 259 & - & - & - \\
131212 & 1960.02 & 2.52 & 664 & 26.27 & 72 & 0.58 & 233 & - & - & - \\
131222 & 1792.77 & 2.55 & 485 & 32.64 & 51 & 0.93 & 273 & - & - & - \\
132112 & 1443.32 & 0.40 & 1049 & 18.36 & 50 & 1.15 & 260 & - & - & - \\
132122 & 1429.30 & 1.23 & 805 & 16.07 & 42 & 1.16 & 344 & - & - & - \\
132212 & 1204.42 & 0.12 & 2197 & 51.98 & 61 & 1.14 & 256 & - & - & - \\
132222 & 924.68  & 0.91 & 982 & 95.36 & 63 & 1.01 & 318 & - & - & - \\
133112 & 1694.18 & 0.37 & 1046 & 14.53 & 72 & 1.05 & 281 & - & - & - \\
133122 & 1392.00 & 0.83 & 925 & 33.40 & 55 & 1.85 & 309 & - & - & - \\
133212 & 1197.95 & 0.11 & 1375 & 17.76 & 56 & 0.37 & 276 & - & - & - \\
133222 & 1151.37 & 0.26 & 911 & 0.34 & 58 & 0.57 & 343 & - & - & - \\
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


def update_line(line):
    match = re.match(r"(\d{6})", line.strip())
    if match:
        instance = match.group(1)
        parts = [p.strip() for p in line.split("&")]
        if instance in neo_ds_dict:
            gap, t_la, t_total = neo_ds_dict[instance]
            parts[8] = f" {gap:.2f} "
            parts[9] = f" {t_la:.2f} "
            parts[10] = f" {t_total:.2f} "
        line = line.rstrip("\\").strip()
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
            ds_gap, ds_tla, ds_ttot = neo_ds_dict.get(instance, (None, None, None))
            rows_data.append((hcc_gap, hcc_t, tsba_gap, tsba_t, grasp_gap, grasp_t, ds_gap, ds_tla, ds_ttot))
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
avg_ds_gap = safe_avg([r[6] for r in rows_data])
avg_ds_tla = safe_avg([r[7] for r in rows_data])
avg_ds_ttot = safe_avg([r[8] for r in rows_data])


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
    f"& \\textbf{{{fmt(avg_ds_gap)}}} & \\textbf{{{fmt(avg_ds_tla)}}} & \\textbf{{{fmt(avg_ds_ttot)}}} \\\\"
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

print("Updated latex table written with averages for HCC, TSBA, GRASP/VNS, and NEO-DS.")
