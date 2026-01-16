# Instructions

This directory contains code to generate all plots and figures presented in our paper.

## Prerequisites

- Ensure you have run NEO-DS on all benchmarks first along with NEO-GT on prodhon benchmark, as these scripts require the Excel result files.
- A LaTeX distribution (e.g., TeX Live) is required for generating figures with LaTeX-rendered text.

## Comparison with Baselines

For each benchmark folder (`barretto`, `prodhon`, `schneider`, `tuzun`), run:

```bash
python baselines_table.py
python baselines_ecdf.py
```

## Ablation Studies

| Ablation | Folder | Script |
|----------|--------|--------|
| Effect of problem size | `schneider` | `baselines_ecdf.py` |
| Effect of sample size | `prodhon` | `ablation_samplesize.py` |
| Effect of routing solver | `prodhon` | `ablation_solvers.py` |
| Effect of neural network architecture | `prodhon` | `ablation_neods_neogt.py` |
| Effect of target normalization | `prodhon` | `ablation_norm_compare.py` |
