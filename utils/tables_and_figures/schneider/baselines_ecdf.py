"""
Generate ECDF plots comparing NEO-DS against baseline methods (TSBA_basic, HGAMP) on the Schneider
benchmark. Also produces scalability analysis plots showing performance trends
across different problem sizes (100-600 customers).
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse

parser = argparse.ArgumentParser(description="Generate ECDF plots")
parser.add_argument("--neods", type=str, default="true",
                    help="Include NEO-DS (true/false)")
parser.add_argument("--tsba", type=str, default="true",
                    help="Include TSBA_basic (true/false)")
parser.add_argument("--hgamp", type=str, default="true",
                    help="Include HGAMP (true/false)")
args = parser.parse_args()

include_methods = {
    "NEO-DS": args.neods.lower() == "true",
    "TSBA_basic": args.tsba.lower() == "true",
    "HGAMP": args.hgamp.lower() == "true",
}

LINE_WIDTH_CURVES = 3
MARKER_SIZE = 0.2
MARKER_EDGE_WIDTH = 3
AXIS_SPINE_WIDTH = 0.5
TICK_WIDTH = 0.4
GRID_WIDTH = 1
FONT_AXIS_LABEL = 22
FONT_TICK_LABEL = 22
FONT_LEGEND = 22

plt.rcParams.update({
    "axes.linewidth": AXIS_SPINE_WIDTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "grid.linewidth": GRID_WIDTH,
    "axes.labelsize": FONT_AXIS_LABEL,
    "xtick.labelsize": FONT_TICK_LABEL,
    "ytick.labelsize": FONT_TICK_LABEL,
    "legend.fontsize": FONT_LEGEND,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

latex_file = "sota_schneider_all.tex"
lines = open(latex_file, "r").readlines()
instance_pattern = re.compile(
    r"(?P<instance>\d{2,3}-\d{1,2}-\S+)\s*&\s*"
    r"(?P<BKS>\d+)\s*&\s*"
    r"(?P<tsba_gap>[-\d\.]+)\s*&\s*(?P<tsba_time>[-\d\.]+)\s*&\s*"
    r"(?P<hgamp_gap>[-\d\.]+)\s*&\s*(?P<hgamp_time>[-\d\.]+)\s*&\s*"
    r"(?P<neos_gap>[-\d\.]+)\s*&\s*(?P<neos_la>[-\d\.]+)\s*&\s*(?P<neos_time>[-\d\.]+)"
)
records = []
for line in lines:
    m = instance_pattern.search(line)
    if m:
        d = m.groupdict()
        records.append(d)
if not records:
    raise RuntimeError("ERROR: Regex did not match any latex rows.")
df = pd.DataFrame(records).astype({
    "tsba_gap": float, "tsba_time": float,
    "hgamp_gap": float, "hgamp_time": float,
    "neos_gap": float, "neos_time": float,
})
df = df.drop(columns=["neos_la"])
print("\nParsed rows:", len(df))
print(df.head())

methods = {
    "TSBA_basic": ("tsba_gap", "tsba_time"),
    "HGAMP": ("hgamp_gap", "hgamp_time"),
    "NEO-DS": ("neos_gap", "neos_time"),
}
rows = []
for _, row in df.iterrows():
    for name, (gap_col, time_col) in methods.items():
        rows.append({
            "Instance": row["instance"],
            "Method": name,
            "Gap": float(row[gap_col]),
            "Time": float(row[time_col]),
        })
df_ecdf = pd.DataFrame(rows)
print("\nECDF long format:")
print(df_ecdf.head())


def ecdf(values):
    x = np.sort(values)
    y = 100 * np.arange(1, len(values) + 1) / len(values)
    return x, y


ecdf_styles = {
    "red_dashdot": dict(color="red", linestyle="dashdot", linewidth=LINE_WIDTH_CURVES),
    "blue_solid": dict(color="blue", linestyle="-", linewidth=LINE_WIDTH_CURVES),
    "black_dash": dict(color="black", linestyle="--", linewidth=LINE_WIDTH_CURVES),
    "magenta_dotted": dict(color="magenta", linestyle=":", linewidth=LINE_WIDTH_CURVES),
}
method_styles = {
    "TSBA_basic": ecdf_styles["black_dash"],
    "HGAMP": ecdf_styles["magenta_dotted"],
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
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/100:.0f}"))
    ax.set_xlabel(xlabel, labelpad=14)
    if value_col != "Time":
        ax.set_ylabel(ylabel)
    if filename.endswith("gap_schneider_new"):
        ax.text(
            0.5, -0.28, r"(a)", transform=ax.transAxes,
            ha="center", va="center",
            fontsize=22
        )
    if filename.endswith("runtime_schneider_new"):
        ax.text(
            0.5, -0.28, r"(b)", transform=ax.transAxes,
            ha="center", va="center",
            fontsize=22
        )
    ax.grid(True, linestyle="--", alpha=0.5)
    if legend_bool:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=600)
    plt.savefig(filename + ".pdf")
    plt.close()
    print("Saved:", filename)


plot_ecdf(
    df=df_ecdf,
    value_col="Gap",
    filename="ecdf_gap_schneider_new",
    xlabel=r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)",
    ylabel="Instances (\%)",
    legend_bool=False
)
plot_ecdf(
    df=df_ecdf,
    value_col="Time",
    filename="ecdf_runtime_schneider_new",
    xlabel=r"$T_{\mathrm{total}}$ ($\times 10^{2}$ s)",
    ylabel="Instances (\%)",
    legend_bool=True
)
print("\nAll ECDF plots generated successfully!")
print("\n" + "~"*70)
print("STATISTICS FOR PAPER DISCUSSION")
print("~"*70)

for method in methods:
    if not include_methods.get(method, True):
        continue
    method_data = df_ecdf[df_ecdf["Method"] == method]
    gaps = method_data["Gap"].values
    times = method_data["Time"].values
    print(f"\n{method}:")
    print(f"  Gap Statistics:")
    print(f"    - Median gap: {np.median(gaps):.2f}%")
    print(f"    - Mean gap: {np.mean(gaps):.2f}%")
    print(f"    - % instances with gap <= 1%: {100 * np.sum(gaps <= 1) / len(gaps):.1f}%")
    print(f"    - % instances with gap <= 2%: {100 * np.sum(gaps <= 2) / len(gaps):.1f}%")
    print(f"    - % instances with gap <= 5%: {100 * np.sum(gaps <= 5) / len(gaps):.1f}%")
    print(f"    - Max gap: {np.max(gaps):.2f}%")
    print(f"  Time Statistics:")
    print(f"    - Median time: {np.median(times):.1f}s")
    print(f"    - Mean time: {np.mean(times):.1f}s")
    print(f"    - % instances solved within 100s: {100 * np.sum(times <= 100) / len(times):.1f}%")
    print(f"    - % instances solved within 200s: {100 * np.sum(times <= 200) / len(times):.1f}%")
    print(f"    - % instances solved within 500s: {100 * np.sum(times <= 500) / len(times):.1f}%")
    print(f"    - Max time: {np.max(times):.1f}s")
print("\n" + "~"*70)
print("\n" + "~"*70)
print("SCALABILITY ANALYSIS BY PROBLEM SIZE")
print("~"*70)

df['customers'] = df['instance'].str.extract(r'^(\d+)-')[0].astype(int)
size_bins = [100, 200, 300, 400, 500, 600]
size_stats = []
for size in size_bins:
    df_size = df[df['customers'] == size]
    if len(df_size) == 0:
        continue
    print(f"\n{'='*50}")
    print(f"Problem Size: {size} customers")
    print(f"Number of instances: {len(df_size)}")
    print(f"{'='*50}")
    for method_name, (gap_col, time_col) in methods.items():
        med_gap = df_size[gap_col].median()
        lower_q_gap = df_size[gap_col].quantile(0.25)
        upper_q_gap = df_size[gap_col].quantile(0.75)
        med_time = df_size[time_col].median()
        lower_q_time = df_size[time_col].quantile(0.25)
        upper_q_time = df_size[time_col].quantile(0.75)
        size_stats.append({
            'Size': size,
            'Method': method_name,
            'Med_Gap': med_gap,
            'Lower_Q_Gap': lower_q_gap,
            'Upper_Q_Gap': upper_q_gap,
            'Med_Time': med_time,
            'Lower_Q_Time': lower_q_time,
            'Upper_Q_Time': upper_q_time,
            'Count': len(df_size)
        })
        print(f"{method_name:15} | Med Gap: {med_gap:6.2f}% [Q1: {lower_q_gap:5.2f}%, Q3: {upper_q_gap:5.2f}%] | Med Time: {med_time:8.1f}s [Q1: {lower_q_time:7.1f}s, Q3: {upper_q_time:7.1f}s]")
df_scalability = pd.DataFrame(size_stats)
fig1, ax1 = plt.subplots(figsize=(8, 7), dpi=600)

for method in methods:
    data = df_scalability[df_scalability['Method'] == method]
    label = method.replace("_", r"\_")
    ax1.plot(data['Size'], data['Med_Gap'],
             label=label, **method_styles[method], marker='o', markersize=8)
    lower_bound = np.maximum(0, data['Lower_Q_Gap'])
    upper_bound = data['Upper_Q_Gap']
    ax1.fill_between(data['Size'], lower_bound, upper_bound,
                      alpha=0.08, color=method_styles[method]['color'])
ax1.set_ylabel(r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)", labelpad=14)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.text(0.5, -0.22, r"(a)", transform=ax1.transAxes,
         ha="center", va="center", fontsize=22)
plt.tight_layout()
plt.savefig("scalability_gap_schneider.png", dpi=600)
plt.savefig("scalability_gap_schneider.pdf")
plt.close()
print("\nScalability gap plot saved!")

fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=600)
for method in methods:
    data = df_scalability[df_scalability['Method'] == method]
    label = method.replace("_", r"\_")
    ax2.plot(data['Size'], data['Med_Time'] / 100,
             label=label, **method_styles[method], marker='o', markersize=8)
    lower_bound = np.maximum(0, data['Lower_Q_Time'] / 100)
    upper_bound = data['Upper_Q_Time'] / 100
    ax2.fill_between(data['Size'], lower_bound, upper_bound,
                      alpha=0.08, color=method_styles[method]['color'])
ax2.set_ylabel(r"$T_{\mathrm{total}}$ ($\times 10^{2}$ s)", labelpad=14)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(frameon=False, loc='best')
ax2.text(0.5, -0.22, r"(b)", transform=ax2.transAxes,
         ha="center", va="center", fontsize=22)
plt.tight_layout()
plt.savefig("scalability_time_schneider.png", dpi=600)
plt.savefig("scalability_time_schneider.pdf")
plt.close()
print("Scalability time plot saved!")
print("~"*70)
print("\n" + "~"*70)
print("KEY SCALABILITY INSIGHTS FOR PAPER")
print("~"*70)

neo_ds_data = df_scalability[df_scalability['Method'] == 'NEO-DS']
tsba_data = df_scalability[df_scalability['Method'] == 'TSBA_basic']
hgamp_data = df_scalability[df_scalability['Method'] == 'HGAMP']

print("\n~~~ SOLUTION QUALITY TRENDS (MEDIAN [Q1, Q3]) ~~~")
print(f"NEO-DS median gap at 100 customers: {neo_ds_data[neo_ds_data['Size']==100]['Med_Gap'].values[0]:.2f}% [Q1: {neo_ds_data[neo_ds_data['Size']==100]['Lower_Q_Gap'].values[0]:.2f}%, Q3: {neo_ds_data[neo_ds_data['Size']==100]['Upper_Q_Gap'].values[0]:.2f}%]")
print(f"NEO-DS median gap at 600 customers: {neo_ds_data[neo_ds_data['Size']==600]['Med_Gap'].values[0]:.2f}% [Q1: {neo_ds_data[neo_ds_data['Size']==600]['Lower_Q_Gap'].values[0]:.2f}%, Q3: {neo_ds_data[neo_ds_data['Size']==600]['Upper_Q_Gap'].values[0]:.2f}%]")
print(f"NEO-DS trend: {'IMPROVING' if neo_ds_data[neo_ds_data['Size']==600]['Med_Gap'].values[0] < neo_ds_data[neo_ds_data['Size']==100]['Med_Gap'].values[0] else 'WORSENING'}")
print(f"\nTSBA-basic median gap at 100 customers: {tsba_data[tsba_data['Size']==100]['Med_Gap'].values[0]:.2f}% [Q1: {tsba_data[tsba_data['Size']==100]['Lower_Q_Gap'].values[0]:.2f}%, Q3: {tsba_data[tsba_data['Size']==100]['Upper_Q_Gap'].values[0]:.2f}%]")
print(f"TSBA-basic median gap at 500 customers: {tsba_data[tsba_data['Size']==500]['Med_Gap'].values[0]:.2f}% [Q1: {tsba_data[tsba_data['Size']==500]['Lower_Q_Gap'].values[0]:.2f}%, Q3: {tsba_data[tsba_data['Size']==500]['Upper_Q_Gap'].values[0]:.2f}%]")
print(f"TSBA-basic median gap at 600 customers: {tsba_data[tsba_data['Size']==600]['Med_Gap'].values[0]:.2f}% [Q1: {tsba_data[tsba_data['Size']==600]['Lower_Q_Gap'].values[0]:.2f}%, Q3: {tsba_data[tsba_data['Size']==600]['Upper_Q_Gap'].values[0]:.2f}%]")
print(f"\nHGAMP median gap range: {hgamp_data['Med_Gap'].min():.2f}% to {hgamp_data['Med_Gap'].max():.2f}%")

print("\n~~~ COMPUTATIONAL TIME TRENDS (MEDIAN [Q1, Q3]) ~~~")
print(f"NEO-DS median time at 100 customers: {neo_ds_data[neo_ds_data['Size']==100]['Med_Time'].values[0]:.1f}s [Q1: {neo_ds_data[neo_ds_data['Size']==100]['Lower_Q_Time'].values[0]:.1f}s, Q3: {neo_ds_data[neo_ds_data['Size']==100]['Upper_Q_Time'].values[0]:.1f}s]")
print(f"NEO-DS median time at 600 customers: {neo_ds_data[neo_ds_data['Size']==600]['Med_Time'].values[0]:.1f}s [Q1: {neo_ds_data[neo_ds_data['Size']==600]['Lower_Q_Time'].values[0]:.1f}s, Q3: {neo_ds_data[neo_ds_data['Size']==600]['Upper_Q_Time'].values[0]:.1f}s]")
print(f"NEO-DS max median time across all sizes: {neo_ds_data['Med_Time'].max():.1f}s")
print(f"\nTSBA-basic median time at 600 customers: {tsba_data[tsba_data['Size']==600]['Med_Time'].values[0]:.1f}s [Q1: {tsba_data[tsba_data['Size']==600]['Lower_Q_Time'].values[0]:.1f}s, Q3: {tsba_data[tsba_data['Size']==600]['Upper_Q_Time'].values[0]:.1f}s]")
print(f"HGAMP median time at 600 customers: {hgamp_data[hgamp_data['Size']==600]['Med_Time'].values[0]:.1f}s [Q1: {hgamp_data[hgamp_data['Size']==600]['Lower_Q_Time'].values[0]:.1f}s, Q3: {hgamp_data[hgamp_data['Size']==600]['Upper_Q_Time'].values[0]:.1f}s]")

print("\n~~~ SPEEDUP AT 600 CUSTOMERS (MEDIAN) ~~~")
speedup_tsba_med = tsba_data[tsba_data['Size']==600]['Med_Time'].values[0] / neo_ds_data[neo_ds_data['Size']==600]['Med_Time'].values[0]
speedup_hgamp_med = hgamp_data[hgamp_data['Size']==600]['Med_Time'].values[0] / neo_ds_data[neo_ds_data['Size']==600]['Med_Time'].values[0]
print(f"NEO-DS vs TSBA-basic: {speedup_tsba_med:.1f}x")
print(f"NEO-DS vs HGAMP: {speedup_hgamp_med:.1f}x")
print("~"*70)
