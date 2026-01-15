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

For more details, please refer to Section 5 (Experimental Results and Discussion) in our paper.

**Configuration:** The solver timeout is set to 5 seconds per instance. To modify, change the `timeout_sec` parameter in the `solve_cvrp_vroom()` function call in `labeling/label_generator.py`.

**Concatenate:** After labeling, concatenate files into training and test sets. We use 110K instances for `train_val.txt` and 10K for `test.txt`:

The pre-generated data files we provide in `training/data/` follow this split.

---

## Step 3A: Training DeepSets

*(Documentation to be added)*

---

## Step 3B: Training Graph Transformer

*(Documentation to be added)*
