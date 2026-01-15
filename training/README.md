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
Then skip to **Step 3** for model training.

---

## Step 1: Data Sampling

### Overview

The instance generator follows the GVS (Uchoa et al. 2017) methodology for creating diverse CVRP benchmark instances. The generation code is adapted from the [CVRPLIB XML100 dataset](http://vrp.galgos.inf.puc-rio.br/index.php/en/).

**Reference:**
> Uchoa et al. (2017). *New benchmark instances for the Capacitated Vehicle Routing Problem.* European Journal of Operational Research.

### Files

| File | Description |
|------|-------------|
| `sampling/generator.py` | Core instance generator (GVS methodology) |
| `sampling/gvs_sampling.py` | Script to generate `generation.sh` with all configurations |
| `sampling/generation.sh` | Generated bash script containing all instance generation commands |

### Configuration Options

The generator accepts the following arguments:

```
python generator.py n depotPos custPos demandType avgRouteSize instanceID randSeed
```

| Argument | Options | Description |
|----------|---------|-------------|
| `n` | Integer | Number of customers |
| `depotPos` | 1=Random, 2=Centered, 3=Cornered | Depot positioning |
| `custPos` | 1=Random, 2=Clustered, 3=Random-clustered | Customer positioning |
| `demandType` | 1-7 | Demand distribution pattern |
| `avgRouteSize` | 1=Very short, 2=Short, 3=Medium, 4=Long, 5=Very long, 6=Ultra long | Average route size |
| `instanceID` | Integer | Instance identifier |
| `randSeed` | Integer | Random seed for reproducibility |

### Usage

1. **Set the output directory** in `sampling/generator.py`:
   ```python
   output_directory = 'Specify folder where you would like to save the sampled data'
   ```
   Change this to your desired path, e.g.:
   ```python
   output_directory = '/path/to/where/you/want/to/store'
   ```

2. **Generate the batch script** (creates `generation.sh` instance configurations):
   ```bash
   cd sampling
   python gvs_sampling.py
   ```

3. **Run the generation script**:
   ```bash
   bash generation.sh
   ```
   This will generate CVRP instances in `.vrp` format in the specified output directory.

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
