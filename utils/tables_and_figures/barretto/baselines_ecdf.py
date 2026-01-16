"""
Generate ECDF plots comparing NEO-DS against baseline methods (HCC-500k, TSBA_speed, grasp/vns) on the
barreto benchmark. produces plots for both optimization gap and runtime,
along with summary statistics for paper discussion.
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse

line_width_curves = 3
marker_size = 0.2
marker_edge_width = 3
axis_spine_width = 0.5
tick_width = 0.4
grid_width = 1
font_axis_label = 22
font_tick_label = 22
font_legend = 22

plt.rcParams.update({
    "axes.linewidth": axis_spine_width,
    "xtick.major.width": tick_width,
    "ytick.major.width": tick_width,
    "grid.linewidth": grid_width,
    "axes.labelsize": font_axis_label,
    "xtick.labelsize": font_tick_label,
    "ytick.labelsize": font_tick_label,
    "legend.fontsize": font_legend,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

parser = argparse.ArgumentParser(description="generate ecdf plots")
parser.add_argument("--neods", type=str, default="true", help="include neo-ds (true/false)")
parser.add_argument("--hcc", type=str, default="true", help="include hcc-500k (true/false)")
parser.add_argument("--tsba", type=str, default="true", help="include tsba_speed (true/false)")
parser.add_argument("--grasp", type=str, default="true", help="include grasp/vns (true/false)")
args = parser.parse_args()

include_methods = {
    "NEO-DS": args.neods.lower() == "true",
    "HCC-500K": args.hcc.lower() == "true",
    "TSBA_speed": args.tsba.lower() == "true",
    "GRASP/VNS": args.grasp.lower() == "true",
}

latex_file = "barreto_sota_table_updated.tex"

with open(latex_file, "r") as f:
    lines = f.readlines()

instance_pattern = re.compile(
    r"^\s*(?P<instance>[A-Za-z0-9\-x]+)\s*&\s*"
    r"(?P<BKS>[\d\.]+)\s*&\s*"
    r"(?P<hcc_gap>[-\d\.]+)\s*&\s*(?P<hcc_time>[-\d\.]+)\s*&\s*"
    r"(?P<tsb_gap>[-\d\.]+)\s*&\s*(?P<tsb_time>[-\d\.]+)\s*&\s*"
    r"(?P<grasp_gap>[-\d\.]+)\s*&\s*(?P<grasp_time>[-\d\.]+)\s*&\s*"
    r"(?P<neos_gap>[-\d\.]+)\s*&\s*(?P<neos_la>[-\d\.]+)\s*&\s*(?P<neos_time>[-\d\.]+)\s*\\\\",
    re.MULTILINE
)

records = []
for line in lines:
    match = instance_pattern.search(line)
    if match:
        d = match.groupdict()
        d["BKS"] = d["BKS"].replace(",", "")
        records.append(d)

for line in lines[:10]:
    print(instance_pattern.search(line))

if not records:
    raise RuntimeError("no rows matched. check latex format or regex.")

df_raw = pd.DataFrame(records).astype({
    "hcc_gap": float, "hcc_time": float,
    "tsb_gap": float, "tsb_time": float,
    "grasp_gap": float, "grasp_time": float,
    "neos_gap": float, "neos_time": float,
})

print("\nparsed rows:", len(df_raw))
print(df_raw.head())

methods = {
    "HCC-500K": ("hcc_gap", "hcc_time"),
    "TSBA_speed": ("tsb_gap", "tsb_time"),
    "GRASP/VNS": ("grasp_gap", "grasp_time"),
    "NEO-DS": ("neos_gap", "neos_time"),
}

rows = []
for _, row in df_raw.iterrows():
    for m, (gap_col, time_col) in methods.items():
        rows.append({
            "Instance": row["instance"],
            "Method": m,
            "Gap": float(row[gap_col]),
            "Time": float(row[time_col]),
        })

df_ecdf = pd.DataFrame(rows)
print("\nlong-format ecdf dataframe created.")
print(df_ecdf.head())


def ecdf(values):
    x = np.sort(values)
    y = 100.0 * np.arange(1, len(x) + 1) / len(x)
    return x, y


ecdf_styles = {
    "purple_solid_circle": dict(
        color="purple", linestyle="-",
        marker="o", markersize=marker_size,
        markerfacecolor="none", markeredgecolor="purple",
        markeredgewidth=marker_edge_width,
        linewidth=line_width_curves
    ),
    "orange_dash_circle": dict(
        color="orange", linestyle="--",
        marker="o", markersize=marker_size,
        markerfacecolor="none", markeredgecolor="orange",
        markeredgewidth=marker_edge_width,
        linewidth=line_width_curves
    ),
    "red_dashdot": dict(
        color="red", linestyle="dashdot",
        marker=None,
        linewidth=line_width_curves
    ),
    "green_dash_circle": dict(
        color="green", linestyle="--",
        marker="o", markersize=marker_size,
        markerfacecolor="none", markeredgecolor="green",
        markeredgewidth=marker_edge_width,
        linewidth=line_width_curves
    ),
}

method_styles = {
    "HCC-500K": ecdf_styles["purple_solid_circle"],
    "TSBA_speed": ecdf_styles["orange_dash_circle"],
    "GRASP/VNS": ecdf_styles["green_dash_circle"],
    "NEO-DS": ecdf_styles["red_dashdot"],
}


def plot_ecdf(df, value_col, filename, xlabel, ylabel, legend_bool):
    plt.figure(figsize=(8, 7), dpi=600)

    for method in methods:
        if not include_methods.get(method, True):
            continue
        vals = df[df["Method"] == method][value_col].values
        x, y = ecdf(vals)
        label = method.replace("_", r"\_")
        plt.step(x, y, where='post', label=label, **method_styles[method])

    ax = plt.gca()

    if value_col == "Time":
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x/100:.0f}")
        )

    ax.set_xlabel(xlabel, labelpad=14)

    if value_col != "Time":
        ax.set_ylabel(ylabel)

    if filename.endswith("gap_barretto_new"):
        ax.text(0.5, -0.28, r"(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=22)

    if filename.endswith("runtime_barretto_new"):
        ax.text(0.5, -0.28, r"(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=22)

    ax.grid(True, linestyle="--", alpha=0.5)

    if legend_bool:
        plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=600)
    plt.savefig(filename + ".pdf")
    plt.close()

    print("saved:", filename + ".png and .pdf")


plot_ecdf(
    df=df_ecdf,
    value_col="Gap",
    filename="ecdf_gap_barretto_new",
    xlabel=r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)",
    ylabel="instances (\%)",
    legend_bool=False
)

plot_ecdf(
    df=df_ecdf,
    value_col="Time",
    filename="ecdf_runtime_barretto_new",
    xlabel=r"$T_{\mathrm{total}}$ ($\times 10^{2}$ s)",
    ylabel="instances (\%)",
    legend_bool=True
)

print("\nall ecdf plots generated successfully!")

print("\n" + "~"*70)
print("statistics for paper discussion")
print("~"*70)

for method in methods:
    if not include_methods.get(method, True):
        continue
    method_data = df_ecdf[df_ecdf["Method"] == method]
    gaps = method_data["Gap"].values
    times = method_data["Time"].values

    print(f"\n{method}:")
    print(f"gap statistics:")
    print(f"- median gap: {np.median(gaps):.2f}%")
    print(f"- mean gap: {np.mean(gaps):.2f}%")
    print(f"- % instances with gap <= 1%: {100 * np.sum(gaps <= 1) / len(gaps):.1f}%")
    print(f"- % instances with gap <= 2%: {100 * np.sum(gaps <= 2) / len(gaps):.1f}%")
    print(f"- % instances with gap <= 5%: {100 * np.sum(gaps <= 5) / len(gaps):.1f}%")
    print(f"- max gap: {np.max(gaps):.2f}%")

    print(f"time statistics:")
    print(f"- median time: {np.median(times):.1f}s")
    print(f"- mean time: {np.mean(times):.1f}s")
    print(f"- % instances solved within 100s: {100 * np.sum(times <= 100) / len(times):.1f}%")
    print(f"- % instances solved within 200s: {100 * np.sum(times <= 200) / len(times):.1f}%")
    print(f"- % instances solved within 500s: {100 * np.sum(times <= 500) / len(times):.1f}%")
    print(f"- max time: {np.max(times):.1f}s")

print("\n" + "~"*70)
