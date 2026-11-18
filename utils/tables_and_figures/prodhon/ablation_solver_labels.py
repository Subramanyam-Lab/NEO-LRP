"""Compare saved VROOM and VRPSolverEasy labels and solve times."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_labels(directory):
    """Read both solver labels for each complete instance in the two data files."""
    rows = []
    statuses, sources = Counter(), Counter()
    for name in ("train_val.txt", "test.txt"):
        metadata = {}
        with (directory / name).open() as stream:
            for line in stream:
                if line.strip() == "#EOF":
                    if metadata["label_source_vrpsolvereasy_scaled"] != "vrpsolvereasy":
                        metadata = {}
                        continue
                    values = [float(metadata[key].rstrip("s")) for key in (
                        "cost_vroom_scaled", "cost_vrpsolvereasy_scaled",
                        "solve_time_vroom_scaled", "solve_time_vrpsolvereasy_scaled")]
                    if not np.isfinite(values).all() or min(values[:2]) <= 0 or min(values[2:]) < 0:
                        raise ValueError("Invalid saved objective or time")
                    rows.append(values)
                    statuses[metadata["status_vrpsolvereasy_scaled"]] += 1
                    sources[metadata["label_source_vrpsolvereasy_scaled"]] += 1
                    metadata = {}
                elif line.startswith("#") and " " in line:
                    key, value = line[1:].strip().split(" ", 1)
                    metadata[key] = value
        if metadata:
            raise ValueError(f"Incomplete final record in {name}")
    if not rows:
        raise ValueError("No paired labels found")
    return np.asarray(rows), statuses, sources


def read_results(path):
    """Read matched Prodhon gaps for models trained with each solver's labels."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or len({row["instance"] for row in rows}) != 30:
        raise ValueError("Expected one result for each of the 30 Prodhon instances")
    gaps = np.array([[float(row["baseline_gap_bks_pct"]), float(row["gap_bks_pct"])]
                     for row in rows])
    expected = np.array([100 * abs(float(row["objective"]) - float(row["bks"])) /
                         float(row["bks"]) for row in rows])
    if not np.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError("Invalid BKS gaps")
    np.testing.assert_allclose(gaps[:, 1], expected, rtol=1e-10, atol=1e-10)
    return gaps


def plot(directory, results, output):
    """Draw solver times, CLRP gaps, and the label-gap histogram."""
    values, statuses, sources = read_labels(directory)
    gaps = 100 * np.abs(values[:, 0] - values[:, 1]) / values[:, 1]
    clrp_gaps = read_results(results)
    style = {"text.usetex": True, "font.family": "serif",
             "font.serif": ["Computer Modern Roman"], "axes.labelsize": 22,
             "xtick.labelsize": 22, "ytick.labelsize": 22, "axes.linewidth": 0.5,
             "xtick.major.width": 0.4, "ytick.major.width": 0.4}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, data in zip(axes[:2], [values[:, 2:], clrp_gaps]):
            boxes = ax.boxplot(data, tick_labels=["Heuristic", "Exact"],
                               patch_artist=True, showfliers=False, widths=0.5,
                               medianprops={"color": "black", "linewidth": 1.5})
            for box, color in zip(boxes["boxes"], ["red", "#4477AA"]):
                box.set_facecolor(color)
                box.set_alpha(0.35)
        axes[0].set_yscale("symlog", linthresh=0.01)
        axes[0].set_ylabel("Solver time (s)")
        axes[2].hist(gaps, bins=np.linspace(0, max(5, np.ceil(gaps.max())), 31),
                     color="#8267A0", alpha=0.45, edgecolor="black", linewidth=0.5)
        axes[2].set_yscale("log")
        axes[2].set_xlabel(r"$E^{\mathrm{label}}$ (\%)")
        axes[2].set_ylabel("Frequency")
        axes[1].set_ylabel(r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)")
        axes[1].set_ylim(bottom=0)
        for letter, ax in zip("abc", axes):
            ax.grid(True, linestyle="--", alpha=0.5, linewidth=1)
            ax.set_axisbelow(True)
            ax.text(0.5, -0.23, f"({letter})", transform=ax.transAxes,
                    ha="center", va="center", fontsize=22)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output)
        fig.savefig(output.with_suffix(".png"), dpi=180)
        plt.close(fig)
    summary = {"instances": len(values), "statuses": dict(statuses), "sources": dict(sources),
               "clrp_instances": len(clrp_gaps),
               "clrp_gap_mean_pct": dict(zip(("heuristic", "exact"), clrp_gaps.mean(axis=0))),
               "clrp_gap_median_pct": dict(zip(("heuristic", "exact"), np.median(clrp_gaps, axis=0))),
               "clrp_gap_mean_difference_pp": float(np.diff(clrp_gaps.mean(axis=0))[0]),
               "clrp_gap_median_difference_pp": float(np.diff(np.median(clrp_gaps, axis=0))[0]),
               "clrp_results_file": str(results),
               "gap_formula": "100 * abs(vroom - vrpsolvereasy) / vrpsolvereasy",
               "mean_label_gap_pct": float(gaps.mean()), "median_label_gap_pct": float(np.median(gaps)),
               "max_label_gap_pct": float(gaps.max()), "label_gaps_above_5_pct": int((gaps > 5).sum()),
               "time_axis": "symlog with linear threshold 0.01 seconds to retain recorded zero times",
               "box_whiskers": "1.5 IQR; outlier markers hidden",
               "time_mean_seconds": dict(zip(("vroom", "vrpsolvereasy"), values[:, 2:].mean(axis=0))),
               "time_median_seconds": dict(zip(("vroom", "vrpsolvereasy"), np.median(values[:, 2:], axis=0)))}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[3] /
                        "training/data/vrpsolvereasy_scaled")
    parser.add_argument("--output", type=Path, default=Path("solver_labels.pdf"))
    parser.add_argument("--results", type=Path, required=True,
                        help="CSV containing matched heuristic-label and exact-solver-label CLRP results")
    args = parser.parse_args()
    plot(args.data, args.results, args.output)
