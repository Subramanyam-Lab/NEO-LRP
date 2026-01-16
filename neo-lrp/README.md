# NEO-LRP: Neural Embedded Optimization for Location-Routing Problems

This module runs the NEO-LRP approach on benchmark instances using trained neural network models (DeepSets or Graph Transformer).

---

## Setup

1. **Edit `submit.sh`**: Set `BASE_DIR` to your NEO-LRP root path and `conda activate <your_conda_env>` to your environment.

2. **Required dependencies**: torch, gurobipy, gurobi-ml, onnx, onnx2torch, vroom, ortools, VRPSolverEasy, pandas, numpy, torch_geometric

3. **Trained models**: Ensure models are in `trained_models/` (requires `git lfs pull`).

---

## Running NEO-LRP

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| DATASET | `P_prodhon`, `S_schneider`, `T_tuzun`, `B_barreto` | Benchmark dataset |
| NORMALIZATION | `raw`, `minmax`, `cost_over_fi`, `cost_over_fi_minmax` | Cost normalization mode |
| N | `110`, `1100`, `11000`, `110000` | Training instances count (model variant) |
| SOLVER | `vroom`, `ortools`, `vrpeasy` | VRP solver for route computation |
| MODEL_TYPE | `deepsets`, `graph_transformer` | Neural network architecture |
| NUM_RUNS | Integer >= 1 | Number of runs per instance (for averaging) |
| INSTANCES_FILE | (optional) | Path to file listing instances (for array jobs) |

### Dataset Notes

- **Scaled datasets** (`P_prodhon`, `S_schneider`): Use `int(100 * distance)` and fixed route cost of 1000
- **Unscaled datasets** (`T_tuzun`, `B_barreto`): Use float distances and no fixed route cost

---

## Using SLURM

### Run all instances in a dataset

```bash
# DeepSets on Prodhon
sbatch --job-name=prodhon_ds submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5

# Graph Transformer on Prodhon
sbatch --job-name=prodhon_gt submit.sh P_prodhon cost_over_fi 110000 vroom graph_transformer 5

# DeepSets on Tuzun (unscaled)
sbatch --job-name=tuzun_ds submit.sh T_tuzun cost_over_fi 110000 vroom deepsets 5
```

### Array jobs (for large datasets like Schneider with 203 instances)

```bash
# DeepSets on Schneider (array job)
sbatch --job-name=schneider_ds --array=1-203 submit.sh S_schneider cost_over_fi 110000 vroom deepsets 1 configs/schneider_instances.txt

# Graph Transformer on Schneider (array job)
sbatch --job-name=schneider_gt --array=1-203 submit.sh S_schneider cost_over_fi 110000 vroom graph_transformer 1 configs/schneider_instances.txt
```

Array jobs process one instance per task. The `INSTANCES_FILE` should contain one instance filename per line.

---

## Using Bash (without SLURM)

Remove or comment out the `#SBATCH` lines in `submit.sh`, then run directly:

```bash
# Run on all Prodhon instances with DeepSets
bash submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5

# Run on all Tuzun instances with Graph Transformer
bash submit.sh T_tuzun cost_over_fi 110000 vroom graph_transformer 5

# Run on all Barreto instances
bash submit.sh B_barreto cost_over_fi 110000 vroom deepsets 5
```

### Running directly with Python

You can also call `run.py` directly:

```bash
# All instances in a dataset
python run.py --dataset P_prodhon --normalization cost_over_fi --N 110000 --solver vroom --model_type deepsets --num_runs 5

# Single instance
python run.py --dataset S_schneider --instance 100-5-1a.json --normalization cost_over_fi --N 110000 --solver vroom --model_type graph_transformer --num_runs 1
```

---

## Output

Results are saved to `output/<DATASET>/`:

- **Solution JSONs**: `<model_type>_<solver>_<normalization>_<N>/<instance>/run_<i>.json`
- **Plot JSONs**: `<model_type>_<solver>_<normalization>_<N>_plots/<instance>/run_<i>.json`
- **Excel summary**: `<DATASET>_<model_type>_<N>_<normalization>_<solver>.xlsx`

For array jobs, individual Excel files are saved to `*_excel_temp/` and can be combined using `combine_results.py`.

---

## Combining Array Job Results

After running array jobs, combine the individual Excel files:

```bash
python combine_results.py \
    --dataset S_schneider \
    --normalization cost_over_fi \
    --N 110000 \
    --solver vroom \
    --model_type deepsets \
    --instances_file configs/schneider_instances.txt \
    --delete_temp
```

The `--delete_temp` flag removes individual Excel files after combining.

---

## Verifying Solutions

After running NEO-LRP, you can verify the solutions by recomputing costs from routes:

```bash
python verify_solutions.py \
    --dataset P_prodhon \
    --model_type deepsets \
    --solver vroom \
    --normalization cost_over_fi \
    --N 110000
```

This script:
- Loads each solution JSON
- Recomputes FLP cost, fixed VRP cost, and variable VRP cost from the routes
- Compares against reported values
- Reports any mismatches

---

## File Structure

```
neo-lrp/
├── run.py                  # Main runner script
├── combine_results.py      # Combine array job Excel files
├── verify_solutions.py     # Verify solution correctness
├── submit.sh               # SLURM/bash submission script
├── configs/                # Dataset configs and BKS values
│   ├── P_prodhon.json
│   ├── S_schneider.json
│   ├── T_tuzun.json
│   ├── B_barreto.json
│   ├── schneider_instances.txt
│   └── BKS/                # Best Known Solutions
├── core/                   # Core modules
│   ├── dataparse.py        # Data parsing utilities
│   ├── lrp_model.py        # Neural embedded LRP model (Gurobi-ML)
│   ├── network_ds.py       # DeepSets network wrapper
│   ├── network_gt.py       # Graph Transformer network
│   ├── solver.py           # CVRP solvers (VROOM, ORTools, VRPSolverEasy)
│   └── utils_train.py      # Shared utilities
└── output/                 # Results output directory
```
