"""Plot the effect of latent dimension and hidden units in rho on NEO-DS performance."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
import pandas as pd


CONTROL = 6
LEVELS = [2, 4, 6, 8, 16]
DISPLAY_LEVELS = [2, 6, 8, 16]
PARAMETERS = ["latent_dimension", "rho_width"]
METRICS = ["e_pred_pct_mean", "gap_bks_pct_mean"]
STYLES = {
    2: ("#4477AA", "D", "-"), 4: ("#AF781E", "s", "--"),
    6: ("red", "o", "-."), 8: ("#368472", "^", ":"),
    16: ("#8267A0", "v", (0, (6, 2.2))),
}


def summarize_errors(path):
    """Compute means and quartiles across five-run instance means of each size."""
    results = pd.read_csv(path)
    baseline = results[PARAMETERS].eq(CONTROL).all(axis=1)
    if not results["source"].eq("Published Prodhon benchmark").equals(baseline):
        raise ValueError("The (6, 6) control must come from the published Prodhon benchmark.")
    if not results["n_runs"].eq(5).all():
        raise ValueError("Each instance mean must include five runs.")
    values = results.set_index([*PARAMETERS, "instance"])[METRICS].unstack("instance").sort_index()
    sweeps = sorted({(CONTROL, level) for level in LEVELS} | {(level, CONTROL) for level in LEVELS})
    if values.index.to_list() != sweeps or not np.isfinite(values).all().all():
        raise ValueError("Expected both errors for every instance in each parameter sweep.")
    customers = results["instance"].str.extract(r"^coord(\d+)-", expand=False).astype(int)
    counts = customers.groupby(results["instance"]).first().value_counts().to_dict()
    if counts != {20: 4, 50: 8, 100: 12, 200: 6}:
        raise ValueError("Expected the 30 Prodhon instances with 20, 50, 100 and 200 customers.")
    groups = results.assign(customers=customers).groupby([*PARAMETERS, "customers"])[METRICS]
    summaries = {"mean": groups.mean(), "q1": groups.quantile(0.25), "q3": groups.quantile(0.75)}
    return pd.concat(summaries, axis=1).swaplevel(axis=1)


def plot(errors, output, levels):
    """Compare each parameter sweep with the published six-by-six NEO-DS model."""
    visible = (errors.index.get_level_values("latent_dimension").isin(levels)
               & errors.index.get_level_values("rho_width").isin(levels))
    errors = errors.loc[visible]
    style = {"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
             "axes.labelsize": 22, "xtick.labelsize": 22, "ytick.labelsize": 22,
             "legend.fontsize": 22, "axes.linewidth": 0.5, "xtick.major.width": 0.4,
             "ytick.major.width": 0.4}
    ylabels = [r"$E^{\mathrm{pred}}$ (\%)", r"$E^{\mathrm{gap}}_{\mathrm{BKS}}$ (\%)"]
    legend_titles = [r"$n_1$", r"$N_\ell$"]
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey="col")
        for row, fixed in enumerate(PARAMETERS):
            sweep = errors.xs(CONTROL, level=fixed)
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                for level in levels:
                    curve = sweep[metric].loc[level]
                    color, marker, linestyle = STYLES[level]
                    baseline = level == CONTROL
                    ax.fill_between(curve.index, curve["q1"], curve["q3"], color=color, alpha=0.08)
                    ax.plot(curve.index, curve["mean"], color=color, linestyle=linestyle,
                            marker=marker, markersize=8, linewidth=3, zorder=4 if baseline else 2,
                            label=str(level))
                ax.set_ylabel(ylabels[col], labelpad=14)
                ax.set_ylim(0, 1.05 * errors[metric].max().max())
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                ax.grid(True, linestyle="--", alpha=0.5, linewidth=1)
                ax.set_axisbelow(True)
                letter = "abcd"[2 * row + col]
                ax.text(0.5, -0.09 if row == 0 else -0.15, f"({letter})",
                        transform=ax.transAxes, ha="center", va="center", fontsize=22)
            axes[row, 1].legend(title=legend_titles[row], title_fontsize=22, loc="upper right",
                                frameon=False, borderaxespad=0.15, handlelength=2.8,
                                handletextpad=0.55, labelspacing=0.15)
        axes[1, 0].set_xscale("log")
        ticks = [20, 50, 100, 200]
        axes[1, 0].set_xticks(ticks, labels=[str(tick) for tick in ticks])
        axes[1, 0].xaxis.set_minor_formatter(NullFormatter())
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.text(0.53, 0.007, "Number of customers", ha="center", va="bottom", fontsize=22)
        fig.savefig(output)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path,
                        help="CSV of five-run instance means including the published baseline")
    parser.add_argument("--output", type=Path, default=Path("neods_model_size.pdf"))
    parser.add_argument("--all-levels", action="store_true", help="Display all five parameter values.")
    args = parser.parse_args()
    plot(summarize_errors(args.results), args.output, LEVELS if args.all_levels else DISPLAY_LEVELS)
