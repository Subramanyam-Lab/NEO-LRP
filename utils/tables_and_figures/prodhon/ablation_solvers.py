"""
Ablation study comparing different routing solvers (VROOM, OR-Tools, VRPSolverEasy)
with NEO-DS on the Prodhon benchmark. Generates a latex table showing optimization
gap and runtime for each solver across different instance sizes.
"""
import pandas as pd


def normalize_instance(name: str) -> str:
    name = name.replace("coord", "").replace(".dat", "")
    parts = name.split("-")
    if parts[-1].isdigit():
        parts[-1] = parts[-1] + "a"
    return "-".join(parts)


def load_neods(excel_path, solver_name):
    df = pd.read_excel(excel_path, sheet_name="results")
    df.columns = df.columns.astype(str).str.strip()
    df["Instance"] = df["Instance"].apply(normalize_instance)
    df["gap"] = df["Optimization_gap_signed"]
    df["TLA"] = df["NN model execution time"]

    if solver_name == "vroom":
        df["Troute"] = df["vroom model solve time"]
    elif solver_name == "ortools":
        df["Troute"] = df["ortools model solve time"]
    elif solver_name == "vrpeasy":
        df["Troute"] = df["vrpeasy model solve time"]

    df["TOTAL"] = df["TLA"] + df["Troute"]
    df = df[["Instance", "BKS", "gap", "TLA", "Troute", "TOTAL"]]
    return df


excel_vroom = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_deepsets_110000_cost_over_fi_vroom.xlsx"
excel_ortools = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_deepsets_110000_cost_over_fi_ortools.xlsx"
excel_vrpeasy = "NEO-LRP/neo-lrp/output/P_prodhon/P_prodhon_deepsets_110000_cost_over_fi_vrpeasy.xlsx"

df_vroom = load_neods(excel_vroom, "vroom")
df_ortools = load_neods(excel_ortools, "ortools")
df_vrpeasy = load_neods(excel_vrpeasy, "vrpeasy")

df_all = (
    df_vroom
    .merge(df_ortools, on=["Instance", "BKS"], suffixes=("_vroom", "_ortools"))
    .merge(df_vrpeasy, on=["Instance", "BKS"])
)

df_all = df_all.rename(columns={
    "gap": "gap_vrpeasy",
    "TLA": "TLA_vrpeasy",
    "Troute": "Troute_vrpeasy",
    "TOTAL": "total_vrpeasy",
    "TOTAL_vroom": "total_vroom",
    "TOTAL_ortools": "total_ortools",
    "TLA_vroom": "TLA_vroom",
    "TLA_ortools": "TLA_ortools",
    "Troute_vroom": "Troute_vroom",
    "Troute_ortools": "Troute_ortools",
})

blocks = {
    "20": ["20-5"],
    "50": ["50-5"],
    "100": ["100-5", "100-10"],
    "200": ["200-10"]
}

avg_rows = []

for block_name, patterns in blocks.items():
    sub = pd.concat([
        df_all[df_all["Instance"].str.startswith(p)]
        for p in patterns
    ])
    if sub.empty:
        continue

    row = {"Instance": f"{block_name} AVERAGE", "BKS": ""}
    for method in ["vroom", "ortools", "vrpeasy"]:
        row[f"gap_{method}"] = sub[f"gap_{method}"].mean()
        row[f"TLA_{method}"] = sub[f"TLA_{method}"].mean()
        row[f"Troute_{method}"] = sub[f"Troute_{method}"].mean()
        row[f"total_{method}"] = sub[f"total_{method}"].mean()
    avg_rows.append(row)

df_avg = pd.DataFrame(avg_rows)
df_final = pd.concat([df_all, df_avg], ignore_index=True)

latex = r"""
\begin{table*}[ht!]
\centering
\scriptsize
\caption{Effect of Routing Solver on NEO-DS Performance ($\mathbb{P}$ Benchmark)}
\label{tab:routing_solver_ablation}
\begin{tabular}{l r  ccc  ccc  ccc}
\toprule
Instance & BKS &
\multicolumn{3}{c}{VROOM (5s)} &
\multicolumn{3}{c}{OR-Tools (5s)} &
\multicolumn{3}{c}{VRPSolverEasy (3600s)} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
 & &
$E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{LA}}$(s) & $T_{\text{total}}$(s) &
$E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{LA}}$(s) & $T_{\text{total}}$(s) &
$E^{\text{gap}}_{\text{BKS}}$ & $T_{\text{LA}}$(s) & $T_{\text{total}}$(s) \\
\midrule
"""

for block_name, patterns in blocks.items():
    block_rows = df_final[
        df_final["Instance"].str.startswith(tuple(patterns))
    ]
    block_rows = block_rows[~block_rows["Instance"].str.contains("AVERAGE")]

    for _, row in block_rows.iterrows():
        latex += (
            f"{row['Instance']} & {row['BKS']} & "
            f"{row['gap_vroom']:.2f} & {row['TLA_vroom']:.2f} & {row['total_vroom']:.2f} & "
            f"{row['gap_ortools']:.2f} & {row['TLA_ortools']:.2f} & {row['total_ortools']:.2f} & "
            f"{row['gap_vrpeasy']:.2f} & {row['TLA_vrpeasy']:.2f} & {row['total_vrpeasy']:.2f} \\\\ \n"
        )

    avg = df_final[df_final["Instance"] == f"{block_name} AVERAGE"].iloc[0]
    latex += "\\midrule\n"
    latex += (
        "\\textbf{Average} &  & "
        f"\\textbf{{{avg['gap_vroom']:.2f}}} & "
        f"\\textbf{{{avg['TLA_vroom']:.2f}}} & "
        f"\\textbf{{{avg['total_vroom']:.2f}}} & "
        f"\\textbf{{{avg['gap_ortools']:.2f}}} & "
        f"\\textbf{{{avg['TLA_ortools']:.2f}}} & "
        f"\\textbf{{{avg['total_ortools']:.2f}}} & "
        f"\\textbf{{{avg['gap_vrpeasy']:.2f}}} & "
        f"\\textbf{{{avg['TLA_vrpeasy']:.2f}}} & "
        f"\\textbf{{{avg['total_vrpeasy']:.2f}}} \\\\ \n"
    )
    latex += "\\midrule\n"

latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

with open("routing_solver_comparison.tex", "w") as f:
    f.write(latex)

print("latex table written to routing_solver_comparison.tex")
