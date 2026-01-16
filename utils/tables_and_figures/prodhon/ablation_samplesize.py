"""
Ablation study analyzing the effect of training sample size on NEO-DS performance.
Compares models trained with 110, 1100, 11000, and 110000 instances. Generates a 2x2 panel plot showing training error, test error,
prediction error and optimization gap trends across sample sizes.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import traceback

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

ecdf_styles = {
    "red_dashdot": dict(color="red", linestyle="dashdot", marker=None, linewidth=LINE_WIDTH_CURVES),
}
method_style = ecdf_styles["red_dashdot"]


def normalize_instance(name: str) -> str:
    name = name.replace("coord", "").replace(".dat", "").replace("_", "-")
    parts = name.split("-")
    if parts[-1].isdigit():
        parts[-1] = parts[-1] + "a"
    return "-".join(parts).replace("bis", "Bis").replace("BIS", "Bis")


def load_excel(path, model_type):
    df = pd.read_excel(path, sheet_name="results")
    df.columns = df.columns.astype(str).str.strip()
    if model_type == "DS":
        gap_col = "Optimization_gap_signed"
        df["gap"] = df[gap_col]
        df["T_total"] = df["NN model execution time"] + df["vroom model solve time"]
        df["E_pre"] = df["Prediction_gap_abs"]
    df["Instance"] = df["Instance"].apply(normalize_instance)
    return df[["Instance", "BKS", "gap", "T_total", "E_pre"]]


def load_txt_metrics(path):
    metrics = {}
    try:
        with open(path, 'r') as f:
            content = f.read()
        print(f"\n  [DEBUG] Reading {path}...")
        if "median_train_abs_gap:" in content:
            val = float(content.split("median_train_abs_gap:")[1].split("\n")[0].strip())
            metrics['E_train'] = val
        if "lower_quartile_train_abs_gap:" in content:
            val = float(content.split("lower_quartile_train_abs_gap:")[1].split("\n")[0].strip())
            metrics['lower_quartile_train_abs_gap'] = val
        if "upper_quartile_train_abs_gap:" in content:
            val = float(content.split("upper_quartile_train_abs_gap:")[1].split("\n")[0].strip())
            metrics['upper_quartile_train_abs_gap'] = val
        if "median_test_abs_gap:" in content:
            val = float(content.split("median_test_abs_gap:")[1].split("\n")[0].strip())
            metrics['E_test'] = val
        if "lower_quartile_test_abs_gap:" in content:
            val = float(content.split("lower_quartile_test_abs_gap:")[1].split("\n")[0].strip())
            metrics['lower_quartile_test_abs_gap'] = val
        if "upper_quartile_test_abs_gap:" in content:
            val = float(content.split("upper_quartile_test_abs_gap:")[1].split("\n")[0].strip())
            metrics['upper_quartile_test_abs_gap'] = val
    except Exception as e:
        print(f"  [ERROR] Exception in load_txt_metrics({path}): {e}")
        traceback.print_exc()
    return metrics


base_ds = "NEO-LRP/neo-lrp/output/P_prodhon"
base_txt = "NEO-LRP/utils/tables_and_figures/prodhon"

ds_suffix = {
    "110": "P_prodhon_deepsets_110_cost_over_fi_vroom.xlsx",
    "1100": "P_prodhon_deepsets_1100_cost_over_fi_vroom.xlsx",
    "11000": "P_prodhon_deepsets_11000_cost_over_fi_vroom.xlsx",
    "110000": "P_prodhon_deepsets_110000_cost_over_fi_vroom.xlsx"
}

txt_suffix = {
    "110": "prodhon_110.txt",
    "1100": "prodhon_1100.txt",
    "11000": "prodhon_11000.txt",
    "110000": "prodhon_110000.txt"
}

ds_paths = {key: os.path.join(base_ds, fname) for key, fname in ds_suffix.items()}
txt_paths = {key: os.path.join(base_txt, fname) for key, fname in txt_suffix.items()}

print("NEO-DS paths:")
for k, v in ds_paths.items():
    print(f"  {k}: {v}")

print("\nMetrics .txt paths:")
for k, v in txt_paths.items():
    print(f"  {k}: {v}")

all_data = []
for size_label, path in ds_paths.items():
    print(f"\nLoading {size_label}...")
    df_size = load_excel(path, "DS")
    df_size["Size"] = int(size_label)
    all_data.append(df_size)
    print(f"  Loaded {len(df_size)} instances")

df_combined = pd.concat(all_data, ignore_index=True)
print(f"\nTotal instances loaded: {len(df_combined)}")
print(df_combined.head())

txt_metrics = {}
print("\nLoading .txt metrics files...")
for size_label, path in txt_paths.items():
    print(f"  Loading {size_label}...")
    metrics = load_txt_metrics(path)
    txt_metrics[int(size_label)] = metrics
    print(f"    E_test: {metrics.get('E_test', 'N/A')}")

print("\n" + "="*70)
print("SCALABILITY ANALYSIS BY PROBLEM SIZE")
print("="*70)

size_stats = []
for size in [110, 1100, 11000, 110000]:
    df_size = df_combined[df_combined['Size'] == size]
    if len(df_size) == 0:
        continue
    print(f"\n{'='*50}")
    print(f"Problem Size: {size} customers")
    print(f"Number of instances: {len(df_size)}")
    print(f"{'='*50}")
    avg_gap = df_size['gap'].mean()
    med_gap = df_size['gap'].median()
    avg_epre = df_size['E_pre'].mean()
    med_epre = df_size['E_pre'].median()
    avg_time = df_size['T_total'].mean()
    std_time = df_size['T_total'].std()
    med_time = df_size['T_total'].median()
    lower_q_pred = df_size['E_pre'].quantile(0.25)
    upper_q_pred = df_size['E_pre'].quantile(0.75)
    lower_q_gap = df_size['gap'].quantile(0.25)
    upper_q_gap = df_size['gap'].quantile(0.75)
    etrain = txt_metrics.get(size, {}).get('E_train', np.nan)
    etest = txt_metrics.get(size, {}).get('E_test', np.nan)
    lower_q_train = txt_metrics.get(size, {}).get('lower_quartile_train_abs_gap', np.nan)
    upper_q_train = txt_metrics.get(size, {}).get('upper_quartile_train_abs_gap', np.nan)
    lower_q_test = txt_metrics.get(size, {}).get('lower_quartile_test_abs_gap', np.nan)
    upper_q_test = txt_metrics.get(size, {}).get('upper_quartile_test_abs_gap', np.nan)
    size_stats.append({
        'Size': size,
        'Med_Gap': med_gap,
        'Lower_Q_Gap': lower_q_gap,
        'Upper_Q_Gap': upper_q_gap,
        'Med_EPre': med_epre,
        'Lower_Q_EPre': lower_q_pred,
        'Upper_Q_EPre': upper_q_pred,
        'E_Train': etrain,
        'Lower_Q_Train': lower_q_train,
        'Upper_Q_Train': upper_q_train,
        'E_Test': etest,
        'Lower_Q_Test': lower_q_test,
        'Upper_Q_Test': upper_q_test,
        'Med_Time': med_time,
        'Count': len(df_size)
    })
    print(f"Gap:      Median: {med_gap:6.2f}% [Q1: {lower_q_gap:5.2f}%, Q3: {upper_q_gap:5.2f}%]")
    print(f"E_pred:   Median: {med_epre:6.2f}% [Q1: {lower_q_pred:5.2f}%, Q3: {upper_q_pred:5.2f}%]")
    print(f"E_train:  Median: {etrain:6.2f}% [Q1: {lower_q_train:5.2f}%, Q3: {upper_q_train:5.2f}%]")
    print(f"E_test:   Median: {etest:6.2f}% [Q1: {lower_q_test:5.2f}%, Q3: {upper_q_test:5.2f}%]")
    print(f"Time:     Median: {med_time:8.1f}s")

df_scalability = pd.DataFrame(size_stats)
print("\n" + "="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)

style_with_marker = method_style.copy()
style_with_marker['marker'] = 'o'
style_with_marker['markersize'] = 8

ax = axes[0, 0]
ax.plot(df_scalability['Size'], df_scalability['E_Train'], label=r"NEO-DS", **style_with_marker)
lower_bound = np.maximum(0, df_scalability['Lower_Q_Train'])
upper_bound = df_scalability['Upper_Q_Train']
ax.fill_between(df_scalability['Size'], lower_bound, upper_bound, alpha=0.08, color=method_style['color'])
ax.set_ylabel(r"$E^{\mathrm{train}}$ (\%)", labelpad=14)
ax.set_xscale('log')
ax.set_ylim(0, 45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(labelbottom=False)
ax.text(0.5, -0.09, r"(a)", transform=ax.transAxes, ha="center", va="center", fontsize=22)

ax = axes[0, 1]
ax.plot(df_scalability['Size'], df_scalability['E_Test'], label=r"NEO-DS", **style_with_marker)
lower_bound = np.maximum(0, df_scalability['Lower_Q_Test'])
upper_bound = df_scalability['Upper_Q_Test']
ax.fill_between(df_scalability['Size'], lower_bound, upper_bound, alpha=0.08, color=method_style['color'])
ax.set_ylabel(r"$E^{\mathrm{test}}$ (\%)", labelpad=14)
ax.set_xscale('log')
ax.set_ylim(0, 45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(labelbottom=False)
ax.legend(frameon=False, loc='upper right')
ax.text(0.5, -0.09, r"(b)", transform=ax.transAxes, ha="center", va="center", fontsize=22)

ax = axes[1, 1]
ax.plot(df_scalability['Size'], df_scalability['Med_Gap'], label=r"NEO-DS", **style_with_marker)
lower_bound = np.maximum(0, df_scalability['Lower_Q_Gap'])
upper_bound = df_scalability['Upper_Q_Gap']
ax.fill_between(df_scalability['Size'], lower_bound, upper_bound, alpha=0.08, color=method_style['color'])
ax.set_xlabel(r"", labelpad=14)
ax.set_ylabel(r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)", labelpad=14)
ax.set_xscale('log')
ax.set_ylim(0, 45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.text(0.5, -0.15, r"(d)", transform=ax.transAxes, ha="center", va="center", fontsize=22)

ax = axes[1, 0]
ax.plot(df_scalability['Size'], df_scalability['Med_EPre'], label=r"NEO-DS", **style_with_marker)
lower_bound = np.maximum(0, df_scalability['Lower_Q_EPre'])
upper_bound = df_scalability['Upper_Q_EPre']
ax.fill_between(df_scalability['Size'], lower_bound, upper_bound, alpha=0.08, color=method_style['color'])
ax.set_xlabel(r"", labelpad=14)
ax.set_ylabel(r"$E^{\mathrm{pred}}$ (\%)", labelpad=14)
ax.set_xscale('log')
ax.set_ylim(0, 45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.text(0.5, -0.15, r"(c)", transform=ax.transAxes, ha="center", va="center", fontsize=22)

plt.tight_layout()
plt.savefig("samplesize_4panel_prodhon.png", dpi=600)
plt.savefig("samplesize_4panel_prodhon.pdf")
plt.close()

print("Saved: samplesize_4panel_prodhon.png and .pdf (2x2 clockwise grid)")

print("\n" + "="*70)
print("KEY SCALABILITY INSIGHTS FOR PAPER")
print("="*70)

print("\nPREDICTION ERROR TRENDS (MEDIAN [Q1, Q3])")
for size in [110, 1100, 11000, 110000]:
    row = df_scalability[df_scalability['Size'] == size]
    if len(row) > 0:
        print(f"NEO-DS E_pred at {size:>6} samples: Median {row['Med_EPre'].values[0]:5.2f}% [Q1: {row['Lower_Q_EPre'].values[0]:5.2f}%, Q3: {row['Upper_Q_EPre'].values[0]:5.2f}%]")

print("\nTRAIN ERROR TRENDS (MEDIAN [Q1, Q3])")
for size in [110, 1100, 11000, 110000]:
    row = df_scalability[df_scalability['Size'] == size]
    if len(row) > 0:
        print(f"NEO-DS E_train at {size:>6} samples: Median {row['E_Train'].values[0]:5.2f}% [Q1: {row['Lower_Q_Train'].values[0]:5.2f}%, Q3: {row['Upper_Q_Train'].values[0]:5.2f}%]")

print("\nTEST ERROR TRENDS (MEDIAN [Q1, Q3])")
for size in [110, 1100, 11000, 110000]:
    row = df_scalability[df_scalability['Size'] == size]
    if len(row) > 0:
        print(f"NEO-DS E_test at {size:>6} samples: Median {row['E_Test'].values[0]:5.2f}% [Q1: {row['Lower_Q_Test'].values[0]:5.2f}%, Q3: {row['Upper_Q_Test'].values[0]:5.2f}%]")

print("\nSOLUTION QUALITY TRENDS (MEDIAN [Q1, Q3])")
for size in [110, 1100, 11000, 110000]:
    row = df_scalability[df_scalability['Size'] == size]
    if len(row) > 0:
        print(f"NEO-DS Gap at {size:>6} samples: Median {row['Med_Gap'].values[0]:5.2f}% [Q1: {row['Lower_Q_Gap'].values[0]:5.2f}%, Q3: {row['Upper_Q_Gap'].values[0]:5.2f}%]")

print("\nCOMPUTATIONAL TIME TRENDS (MEDIAN)")
for size in [110, 1100, 11000, 110000]:
    row = df_scalability[df_scalability['Size'] == size]
    if len(row) > 0:
        print(f"NEO-DS time at {size:>6} samples: Median {row['Med_Time'].values[0]:8.1f}s")

print("="*70)
