# Neural Embedded Optimization for Location-Routing Problems

This repository contains the implementation, datasets, pre-trained models, and training data for the Neural Embedded Optimization approach for solving Location-Routing Problems. We provide everything needed to reproduce our results: benchmark instances, pre-trained neural networks, data sampling and label generation scripts, and training code.

**Naming Convention:** We refer to our approach as **NEO-LRP**. In the paper and codebase:
- **NEO-DS** = NEO-LRP using DeepSets surrogate
- **NEO-GT** = NEO-LRP using Graph Transformer surrogate

## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running NEO-LRP](#running-neo-lrp)
- [Training Neural Networks](#training-neural-networks)
- [Ablation Studies](#ablation-studies)
- [Generating Figures and Tables](#generating-figures-and-tables)
- [Citation](#citation)

---

## Repository Structure

```
NEO-LRP/
├── benchmark_instances/            # Benchmark datasets
│   ├── B_barreto/                  # Barreto benchmark (12 instances)
│   ├── P_prodhon/                  # Prodhon benchmark (30 instances)
│   ├── S_schneider/                # Schneider benchmark (203 instances)
│   └── T_tuzun/                    # Tuzun benchmark (36 instances)
│
├── trained_models/                 # Pre-trained neural networks (requires git lfs pull)
│   ├── deepsets/
│   │   ├── scaled/                 # For P_prodhon, S_schneider
│   │   │   ├── phi/
│   │   │   └── rho/
│   │   └── unscaled/               # For T_tuzun, B_barreto
│   │       ├── phi/
│   │       └── rho/
│   └── graph_transformer/
│       └── scaled/
│
├── neo-lrp/                        # Main NEO-LRP implementation
│   ├── run.py                      # Main runner script
│   ├── combine_results.py          # Combine array job Excel files
│   ├── verify_solutions.py         # Verify solution correctness
│   ├── submit.sh                   # SLURM/bash submission script
│   ├── configs/                    # Dataset configs and BKS values
│   │   ├── {P_prodhon,S_schneider,T_tuzun,B_barreto}.json
│   │   ├── schneider_instances.txt
│   │   └── BKS/                    # Best Known Solutions (from the literature)
│   └── core/                       # Core modules
│       ├── dataparse.py            # Data parsing utilities
│       ├── lrp_model.py            # Neural embedded LRP model
│       ├── network_ds.py           # DeepSets network wrapper
│       ├── network_gt.py           # Graph Transformer network wrapper
│       └── solver.py               # CVRP solvers (VROOM, ORTools, VRPSolverEasy)
│
├── training/                       # Training pipeline (see training/README.md)
│   ├── data/                       # Pre-sampled training data (requires git lfs pull)
│   ├── sampling/                   # CVRP instance generation (GVS sampling)
│   ├── labeling/                   # Label generation using VROOM
│   └── scripts/                    # Training scripts
│       ├── train_deepsets/
│       └── train_graph_transformer/
│
├── flpvrp/                         # FLP-VRP ablation study (see flpvrp/README.md)
│
└── utils/
    └── tables_and_figures/         # Generate paper figures (see its README.md)
```

---

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Subramanyam-Lab/NEO-LRP.git
cd NEO-LRP
```

2. **Pull Large Files (trained models and training data)**
```bash
git lfs install
git lfs pull
```

3. **Create Conda Environment**
```bash
conda create --name neo_lrp python=3.9
conda activate neo_lrp
pip install -r requirements.txt
```

---

## Running NEO-LRP

> [!TIP]
> **Using Pre-trained Models on Your Own Benchmarks**
>
> You can directly apply our pre-trained models to new LRP instances without retraining:
>
> 1. **Determine your problem type:**
>    - **Scaled**: Distances are `int(100 * euclidean_distance)` and fixed route cost = 1000 → use `scaled` models
>    - **Unscaled**: Distances are floats (no scaling) and no fixed route cost → use `unscaled` models
>
> 2. **Add your benchmark:**
>    - Place instance files in `benchmark_instances/H_hypothetical/`
>    - Create config `neo-lrp/configs/H_hypothetical.json` (copy from existing config)
>    - Add BKS values to `neo-lrp/configs/BKS/H_hypothetical.json` (if available)
>
> 3. **Run:**
>    ```bash
>    # For scaled instances (like P_prodhon, S_schneider)
>    python run.py --dataset H_hypothetical --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --num_runs 1
>
>    # For unscaled instances (like T_tuzun, B_barreto)
>    python run.py --dataset H_hypothetical --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --num_runs 1
>    ```
>    The code automatically selects scaled/unscaled models based on your config.

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--dataset` | `P_prodhon`, `S_schneider`, `T_tuzun`, `B_barreto` | Benchmark dataset |
| `--normalization` | `raw`, `minmax`, `cost_over_fi`, `cost_over_fi_minmax` | Use model trained with this target normalization |
| `--N` | `110`, `1100`, `11000`, `110000` | Use model trained on N instances |
| `--solver` | `vroom`, `ortools`, `vrpeasy` | VRP solver for route computation |
| `--model_type` | `deepsets`, `graph_transformer` | Neural network architecture |
| `--num_runs` | Integer >= 1 | Number of runs per instance |
| `--instance` | (optional) | Single instance filename |

### Using Python

```bash
cd neo-lrp

# All instances in a dataset
python run.py --dataset P_prodhon --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --num_runs 5

# Single instance
python run.py --dataset S_schneider --instance 100-5-1a.json --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --num_runs 1
```

### Using Bash

Edit `submit.sh` to set `BASE_DIR` and conda environment, then remove or comment out `#SBATCH` lines:

```bash
cd neo-lrp
bash submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5
```

### Using SLURM

```bash
cd neo-lrp

# Edit submit.sh: set BASE_DIR and conda environment

# DeepSets on Prodhon
sbatch --job-name=prodhon_ds submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5

# Graph Transformer on Prodhon
sbatch --job-name=prodhon_gt submit.sh P_prodhon cost_over_fi 110000 vroom graph_transformer 5

# Array jobs for large datasets (e.g., Schneider with 203 instances)
sbatch --job-name=schneider_ds --array=1-203 submit.sh S_schneider cost_over_fi 110000 vroom deepsets 1 configs/schneider_instances.txt
```

### Output

Results are saved to `neo-lrp/output/<DATASET>/`:
- **Solution JSONs**: `<model_type>_<solver>_<normalization>_<N>/<instance>/run_<i>.json`
- **Excel summary**: `<DATASET>_<model_type>_<N>_<normalization>_<solver>.xlsx`

### Utilities

**Combine array job results:**
```bash
python combine_results.py --dataset S_schneider --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --instances_file configs/schneider_instances.txt --delete_temp
```

**Verify solutions:**
```bash
python verify_solutions.py --dataset P_prodhon --model_type deepsets --solver vroom --normalization cost_over_fi --N 110000
```

---

## Training Neural Networks

For training new models from scratch, see [`training/README.md`](training/README.md).

The training pipeline consists of:
1. **Data Sampling** - Generate synthetic CVRP instances (GVS sampling)
2. **Label Generation** - Solve instances using VROOM to obtain routing cost labels
3. **Model Training** - Train DeepSets or Graph Transformer architectures

Pre-trained models and pre-sampled data are provided (requires `git lfs pull`).

---

## Ablation Studies

### Value of Location Allocation Decisions (FLP-VRP Comparison)

See [`flpvrp/README.md`](flpvrp/README.md) for instructions on running the FLP-VRP ablation study.

### Other Ablation Studies

After running NEO-LRP on all benchmarks, ablation study results can be generated using scripts in `utils/tables_and_figures/prodhon/`:

| Ablation | Script |
|----------|--------|
| Effect of sample size | `ablation_samplesize.py` |
| Effect of routing solver | `ablation_solvers.py` |
| Effect of neural network architecture | `ablation_neods_neogt.py` |
| Effect of target normalization | `ablation_norm_compare.py` |

---

## Generating Figures and Tables

After running NEO-LRP on all benchmarks, see [`utils/tables_and_figures/README.md`](utils/tables_and_figures/README.md) for instructions on generating all figures and tables presented in the paper.

---

## Citation

If you use our pre-trained models or code, please cite:

```bibtex
@inproceedings{kaleem2024neural,
  title={Neural Embedded Optimization for Integrated Location and Routing Problems},
  author={Kaleem, Waquar and Ayala, Harshita and Subramanyam, Anirudh},
  booktitle={IISE Annual Conference. Proceedings},
  pages={1--6},
  year={2024},
  organization={Institute of Industrial and Systems Engineers (IISE)}
}

@article{kaleem2024neural,
  title={Neural Embedded Mixed-Integer Optimization for Location-Routing Problems},
  author={Kaleem, Waquar and Subramanyam, Anirudh},
  journal={arXiv preprint arXiv:2412.05665},
  year={2024}
}
```

For questions and support, please open an issue in the repository.
