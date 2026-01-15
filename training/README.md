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

We adopt the GVS sampling procedure from [Uchoa et al. (2017)](https://www.sciencedirect.com/science/article/pii/S0377221716306270) and [Queiroga et al.](https://openreview.net/pdf?id=yHiMXKN6nTl). The generator code (`sampling/generator.py`) is taken directly from [CVRPLib webpage](http://vrp.galgos.inf.puc-rio.br/index.php/en/updates) and we make very small modifcations to it (please see the data sammpling section of our paper). The script `sampling/gvs_sampling.py` generates the batch file `generation.sh` with all instance configurations.

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

### Overview

The label generator uses the [VROOM](https://github.com/VROOM-Project/vroom) solver to compute routing costs for each CVRP instance. Solutions are appended directly to the instance files.

### Files

| File | Description |
|------|-------------|
| `labeling/label_generator.py` | VROOM-based CVRP solver and label appender |

### Usage

```bash
cd labeling
python label_generator.py <input_path> --mode <scaled|unscaled>
```

| Argument | Description |
|----------|-------------|
| `input_path` | Path to a single instance file OR a directory containing multiple `.vrp` files |
| `--mode` | `scaled` (Prodhon-style, coordinates multiplied by 100) or `unscaled` (Barreto/Tuzun-style) |

### Examples

```bash
# Solve a single instance (scaled mode)
python label_generator.py /path/to/instance.vrp --mode scaled

# Solve all instances in a directory (unscaled mode)
python label_generator.py /path/to/sampled_data/ --mode unscaled
```

### Configuration

- **Time limit**: The solver timeout is set to 5 seconds per instance by default. To modify, change the `timeout_sec` parameter in the `solve_cvrp_vroom()` function call within `label_generator.py`.

- **Scaled vs Unscaled modes**:
  | Mode | Scale Factor | Fixed Vehicle Cost | Use Case |
  |------|--------------|-------------------|----------|
  | `scaled` | 100 | 1000 | Prodhon benchmark style |
  | `unscaled` | 1 | 0 | Barreto/Tuzun benchmark style |

### Output Format

After solving, the following metadata is appended to each instance file:
```
#cost_vroom_<mode> <total_cost>
#num_routes_vroom_<mode> <number_of_vehicles_used>
#solve_time_vroom_<mode> <solve_time>s
#actual_routes_vroom_<mode> <route_details>
#EOF
```

### Concatenating Instances

After all instances are solved, concatenate them into a single file for training:

```bash
# Concatenate all solved instances into training file
cat /path/to/instances/*.vrp > train_val.txt

# Or split into train/val and test sets as needed
# (ensure proper shuffling and splitting)
```

---

## Step 3A: Training DeepSets

*(Documentation to be added)*

---

## Step 3B: Training Graph Transformer

*(Documentation to be added)*
