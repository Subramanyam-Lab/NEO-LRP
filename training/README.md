# Instructions

This part of the repo provides a three step pipeline for training neural network models to predict CVRP routing costs:

1. **Data Sampling** - Generate synthetic CVRP instances
2. **Label Generation** - Solve instances to obtain routing costs labels
3. **Model Training**
   - a: DeepSets architecture
   - b: Graph Transformer architecture

---

**Note:** We provide pre-sampled and labeled data in `training/data/`:
- `training/data/train_val.txt` - Training and validation instances
- `training/data/test.txt` - Test instances

To use them directly, set up Git LFS and pull the files:
```bash
git lfs install
git lfs pull
```
Then skip to **Step 3** for model training, but you would like to sample and generate the label yourself please procced Step 1 -> 2 -> 3

---

## Step 1: Data Sampling

We adopt the GVS sampling procedure from [Uchoa et al. (2017)](https://www.sciencedirect.com/science/article/pii/S0377221716306270) and [Queiroga et al.](https://openreview.net/pdf?id=yHiMXKN6nTl). The generator code (`sampling/generator.py`) is taken directly from [CVRPLib webpage](http://vrp.galgos.inf.puc-rio.br/index.php/en/updates) and we make very small modifications to it (please see the data sammpling section of our paper). The script `sampling/gvs_sampling.py` generates the bash file `generation.sh` with all instance configurations.

**Run:**
```bash
# 1. Set output_directory in sampling/generator.py to your desired path
# 2. Generate the batch script
cd sampling
python gvs_sampling.py
# 3. Run generation
bash generation.sh
```

---

## Step 2: Label Generation

The label generator uses [VROOM](https://github.com/VROOM-Project/vroom) solver to compute routing costs for each CVRP instance from Step 1. Solutions are appended directly to the instance files.

**Run:**
```bash
cd labeling
python label_generator.py <input_path> --mode <scaled|unscaled>
```
Where `<input_path>` is a single `.vrp` file or a directory containing multiple `.vrp` files from Step 1.

- **scaled**: Euclidean distance scaled by 100, fixed cost of 1000 per route
- **unscaled**: No scaling, no fixed cost (both zero)

For more details on scaled and unscaled, please refer to Section 5 (Experimental Results and Discussion) in our paper.

The solver timeout is set to 5 seconds per instance. To modify, change the `timeout_sec` parameter in the `solve_cvrp_vroom()` function call in `labeling/label_generator.py`.

After labeling, concatenate files into training and test sets. We use 110K instances for `train_val.txt` and 10K for `test.txt`. The pre-generated data files we provide in `training/data/` follow this split.

---

## Step 3a: Training DeepSets

Training scripts are in `scripts/train_deepsets/`. The training uses [DeepHyper](https://github.com/deephyper/deephyper) for hyperparameter optimization.

**Run:**
```bash
cd scripts/train_deepsets
# 1. Edit submit.sh: set BASE_DIR, conda environment and SLURM parameters (if using SLURM)
# 2. Run training
sbatch submit.sh        # if using SLURM
bash submit.sh          # if running locally (remove #SBATCH lines first)
```

HPO parameters can be modified in `hpo.py`. Training parameters (num_instances, max_evals, normalization_modes) can be adjusted in `submit.sh`.

Trained models are saved to `trained_models/`. We provide our pre-trained DeepSets models in this folder (requires `git lfs pull`).

---

## Step 3b: Training Graph Transformer

Training scripts are in `scripts/train_graph_transformer/`. The majority of this code was written by [Doyoung Lee](https://github.com/2dozero).

**HPO (uses [Weights & Biases](https://wandb.ai/)):**
```bash
cd scripts/train_graph_transformer
# Edit submit_pretrain.sh: set BASE_DIR, conda env, SLURM params, and WANDB_API_KEY
sbatch submit_pretrain.sh   # if using SLURM
bash submit_pretrain.sh     # if running locally (remove #SBATCH lines first)
```

**Training (after HPO):**
```bash
# Update the best architecture found from HPO in train.py
# Edit submit_train.sh: set BASE_DIR, conda env, SLURM params
sbatch submit_train.sh      # if using SLURM
bash submit_train.sh        # if running locally (remove #SBATCH lines first)
```

Trained models are saved to `trained_models/`. We provide our pre-trained Graph Transformer models in this folder (requires `git lfs pull`).
