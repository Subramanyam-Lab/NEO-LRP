# Instructions

This part of the repo contains code to generate all plots and figures presented in our paper.

## Prerequisites

- Ensure you have run NEO-DS on all benchmarks first along with NEO-GT on prodhon benchmark, as these scripts require the Excel result files.
- A LaTeX distribution (e.g., TeX Live) is required for generating figures with LaTeX-rendered text.

## Comparison with Baselines

For each benchmark folder (`barretto`, `prodhon`, `schneider`, `tuzun`), run:

```bash
python baselines_table.py
python baselines_ecdf.py
```

For T and B, a CSV of instance averages can replace the Excel results:

```bash
python baselines_table.py --results path/to/results.csv
python baselines_ecdf.py
```

The CSV columns are `Instance`, `gap`, `T_LA`, and `T_total`, with optional `gap_std`, `T_LA_std`, and `T_total_std`. Gaps are percentages and times are seconds. An optional `Average` row provides the benchmark average.

## Ablation Studies

| Ablation | Folder | Script |
|----------|--------|--------|
| Effect of problem size | `schneider` | `baselines_ecdf.py` |
| Effect of sample size | `prodhon` | `ablation_samplesize.py` |
| Effect of routing solver | `prodhon` | `ablation_solvers.py` |
| Effect of neural network architecture | `prodhon` | `ablation_neods_neogt.py` |
| Effect of target normalization | `prodhon` | `ablation_norm_compare.py` |
| Effect of latent dimension and number of hidden units | `prodhon` | `ablation_model_size.py` |
| Effect of the solver used to generate training labels | `prodhon` | `ablation_solver_labels.py` |

Run the following from `prodhon` with the saved ablation results:

```bash
python ablation_model_size.py path/to/model_size.csv
python ablation_solver_labels.py --results path/to/solver_labels.csv
```

For `ablation_model_size.py`, the CSV contains `latent_dimension`, `rho_width`, `instance`, `n_runs`, `e_pred_pct_mean`, `gap_bks_pct_mean`, and `source`. Include the 30 Prodhon instances for both sweeps over 2, 4, 6, 8, and 16, fixing the other hyperparameter at 6. Each row contains an instance average over five runs. The (6, 6) rows use the published baseline and `source` is `Published Prodhon benchmark`. The figure shows levels 2, 6, 8, and 16 by default; `--all-levels` includes 4.

For `ablation_solver_labels.py`, the CSV contains `instance`, `baseline_gap_bks_pct`, `gap_bks_pct`, `objective`, and `bks` for the 30 Prodhon instances. The baseline uses heuristic training labels. The comparison uses VRPSolverEasy training labels. Both gaps are percentages. Paired label costs and solve times are read from `training/data/vrpsolvereasy_scaled` by default, or from `--data`. Only labels produced by VRPSolverEasy are included.

Result files and generated figures and tables are kept locally and are not tracked in Git.

The saved model trained with VRPSolverEasy labels is in `trained_models/deepsets/scaled/solver_labels/vrpsolvereasy`. From the repository root, run:

```bash
python neo-lrp/run.py --dataset P_prodhon --normalization cost_over_fi --N 105315 --solver vroom --num_runs 1 --model_dir trained_models/deepsets/scaled/solver_labels/vrpsolvereasy --output_dir output/solver_labels
```

This selects the model trained with VRPSolverEasy labels and keeps VROOM for final routing. The saved model uses 94,802 training instances and 10,513 validation instances with VRPSolverEasy labels. Its configuration is stored alongside the model.
