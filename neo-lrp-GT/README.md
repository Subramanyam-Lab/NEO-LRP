
# Neural Embedded Location-Routing Problem - Graph Transformer Implementation

This directory contains the Graph Transformer implementation of the neural embedded framework for solving Location-Routing Problems.

## Contents

- `neo_lrp_execute_vroom.py`: Main execution script using VROOM solver
- `neural_embedded_model.py`: Neural embedded LRP solver implementation
- `net.py`: Graph Transformer network architecture
- `dataparse.py`: Data parsing and normalization utilities
- `solver_cvrp_vroom.py`: VROOM-based CVRP solver integration
- `utils.py`, `utils_train.py`: Utility functions for training and processing

## Usage

### Running with Pre-trained Models

To run the neural embedded framework using pre-trained Graph Transformer:

```bash
cd neo-lrp-GT
python neo_lrp_execute_vroom.py
```

The script will automatically:
- Use the pre-trained Graph Transformer model from `../pre_trained_models/graph_transformer.pth`
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_gt_results.xlsx`

### Training from Scratch with New Dataset

To train a new Graph Transformer model with your own dataset, follow these steps:

#### 1. Data Generation

First, generate training data using the VROOM solver:

```bash
cd ../training_data_sampling
python data_generation.py --target_num_data 128000 --seed 0
```

This will create training data in HDF5 format. See `data_generation.py` for available parameters.

#### 2. Hyperparameter Tuning

Use the pre-training script to find optimal hyperparameters:

```bash
cd ../neo-lrp-GT
python pre_train.py
```

This script uses Weights & Biases (wandb) to perform Bayesian optimization over various hyperparameters including network architecture, learning rate, and training settings.

**Note:** Make sure to update the data path in `pre_train.py` to point to your generated training data.

#### 3. Model Training

Once you have optimal hyperparameters, train the final model:

```bash
python train.py
```

**Configuration:**
- Update the data path in `train.py` to use your training data
- Modify hyperparameters based on results from step 2
- The script will save the best model to the `model_state/` directory

#### 4. Using Your Trained Model

After training, update the model path in `neo_lrp_execute_vroom.py`:

```python
phi_loc = "path/to/your/trained_model.pth"  # Update this path
```

Then run the execution script:

```bash
python neo_lrp_execute_vroom.py
```

## Configuration

The main parameters can be modified in `neo_lrp_execute_vroom.py`:

```python
# Configuration parameters
BFS = "solutions"  # Directory for storing intermediate solutions
phi_loc = "../pre_trained_models/graph_transformer.pth"  # Model path
existing_excel_file = "results/neo_lrp_gt_results.xlsx"  # Results file
sheet_name = "results"  # Excel sheet name
fi_mode_input = "dynamic"  # Normalization mode
directory_path = "../prodhon_dataset"  # Dataset directory
```

## Requirements

This implementation requires:
- PyTorch with CUDA support (recommended)
- Gurobi optimizer (for MIP solving)
- VROOM solver (for exact VRP solutions)
- PyTorch Geometric (for graph neural networks)

## Results

The framework generates results in the `results/` directory:

### `neo_lrp_gt_results.xlsx`
Contains comprehensive solution metrics including:
- **Instance details**: Instance name and problem parameters
- **Solution costs**: FLP cost, VRP cost, and total LRP cost
- **Route information**: Number of routes in optimal solution
- **Execution times**: LRP solver time, VROOM solver time
- **Cost comparisons**: VROOM computed VRP cost vs. actual LRP cost
- **Best known solutions (BKS)**: Benchmark comparison
- **Gap analysis**: Optimization gap and prediction gap metrics

## Example Execution

```bash
cd neo-lrp-GT
source ../venv/bin/activate  # or activate your Python 3.11 environment
python neo_lrp_execute_vroom.py
```

**Sample Output:**
```
Working on: ../prodhon_dataset/coord100-10-1.dat
Run 1 for instance coord100-10-1.dat

Adding neural embedding constraints...
Gurobi Optimizer version 12.0.3 build v12.0.3rc0

Optimal solution found (tolerance 1.00e-02)
Best objective 2.232088769495e+05, best bound 2.223082800906e+05, gap 0.4035%

[VROOM] depot 2 → cost=25396, routes=4
[VROOM] depot 4 → cost=18747, routes=4
[VROOM] depot 5 → cost=21126, routes=4
Final VRP cost (variable+fixed): 65269
Number of routes: 12
```

## CVRPLIB Training Pipeline

In addition to the standard training workflow, this implementation provides a specialized pipeline for training with CVRPLIB instances. This allows you to use benchmark VRP instances from the CVRPLIB repository as training data.

### Overview

The CVRPLIB pipeline processes standard .vrp format files and converts them to HDF5 format compatible with the Graph Transformer training. The pipeline automatically:

1. Reads CVRPLIB instances (.vrp format) using the vrplib library
2. Generates optimal solutions using VROOM solver (if .sol files don't exist)
3. Converts the data to HDF5 format for neural network training
4. Preserves solution timing information for analysis

### Requirements

Install additional dependencies for CVRPLIB processing:

```bash
pip install vrplib
```

### Usage

#### 1. Prepare CVRPLIB Instances

Organize your .vrp files in a single directory:

```
cvrplib_instances/
├── A-n32-k5.vrp
├── A-n33-k5.vrp
├── A-n34-k5.vrp
└── ...
```

Optional: Include corresponding .sol files if available:

```
cvrplib_instances/
├── A-n32-k5.vrp
├── A-n32-k5.sol    # Optional: will be generated if missing
├── A-n33-k5.vrp
└── ...
```

#### 2. Process Instances

Convert CVRPLIB instances to HDF5 training format using the processor in the training_data_sampling directory:

```bash
cd training_data_sampling
python cvrplib_processor.py /path/to/cvrplib_instances cvrplib_train_data.h5
```

**Command Options:**
- `input_folder`: Directory containing .vrp files
- `output_file`: Output HDF5 file path
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--force-regenerate`: Force regenerate solutions even if .sol files exist (original files will be backed up)
- `--time-limit`: Time limit for VROOM solver in seconds (default: 300)

**Example:**
```bash
python cvrplib_processor.py ../cvrplib_instances cvrplib_train_data.h5 --log-level INFO
```

**Force Regeneration Example:**
```bash
# Force regenerate all solutions with backup
python cvrplib_processor.py ../cvrplib_instances cvrplib_train_data.h5 --force-regenerate
```

**Time Limit Examples:**
```bash
# Set custom time limit (10 minutes)
python cvrplib_processor.py ../cvrplib_instances cvrplib_train_data.h5 --time-limit 600

# Combine options: force regenerate with 1-minute time limit
python cvrplib_processor.py ../cvrplib_instances cvrplib_train_data.h5 --force-regenerate --time-limit 60
```

#### 3. Train with CVRPLIB Data

Update `neo-lrp-GT/train.py` to use the processed CVRPLIB data:

```python
# In train.py, modify the data loading section:
train_data, test_data, _ = prepare_pretrain_data(
    "../training_data_sampling/cvrplib_train_data.h5",  # Use CVRPLIB data
    split_ratios=[0.8, 0.2, 0.0],
)
```

Then run training as usual:

```bash
cd neo-lrp-GT
python train.py
```

### Pipeline Features

#### Automatic Solution Generation

If a .sol file doesn't exist for a .vrp instance, the pipeline automatically:
- Uses VROOM solver to generate optimal solutions
- Records solution time for performance analysis
- Saves the solution as a .sol file for future use

#### Force Regeneration and Backup

When using the `--force-regenerate` option, the pipeline will:
- Regenerate solutions even if .sol files already exist
- Automatically backup original .sol files with timestamp (e.g., `A-n32-k5_backup_20240917_143052.sol`)
- Generate fresh solutions using VROOM
- Preserve all original solutions for comparison

#### VROOM Solver Time Limits

The pipeline includes time limit controls for VROOM solver performance:
- **Default time limit**: 300 seconds (5 minutes) per instance
- **Customizable**: Use `--time-limit` to set custom limits
- **Timeout handling**: Instances that exceed time limits are marked as failed with informative logging
- **Performance monitoring**: Actual solve times are recorded for analysis

**Time Limit Guidelines:**
- Small instances (< 50 nodes): 30-60 seconds usually sufficient
- Medium instances (50-200 nodes): 300-600 seconds recommended
- Large instances (> 200 nodes): 600+ seconds may be needed
- For batch processing: Consider shorter limits to avoid hanging on difficult instances

#### Data Format Compatibility

The processor ensures full compatibility with the existing training pipeline by:
- Converting coordinates to the expected HDF5 format
- Generating distance matrices automatically
- Creating proper node features (x, y, is_depot, demand)
- Maintaining cost and mask information

#### Logging and Monitoring

The processor provides detailed logging:
- Instance processing progress
- Solution generation status
- Conversion success/failure tracking
- Performance timing information

**Sample Output:**
```
INFO:cvrplib_processor:Found 50 .vrp files in ../cvrplib_instances
INFO:cvrplib_processor:Processing A-n32-k5.vrp
INFO:cvrplib_processor:No solution found for A-n32-k5.vrp, generating with VROOM
INFO:cvrplib_processor:Successfully processed A-n32-k5.vrp
...
INFO:cvrplib_processor:Processing complete!
INFO:cvrplib_processor:Successfully processed: 48
INFO:cvrplib_processor:Failed: 2
INFO:cvrplib_processor:Total time: 124.56 seconds
```

### Integration with Existing Workflow

The CVRPLIB pipeline is designed as an augmenting component that doesn't interfere with existing workflows:

- **Standalone Operation**: Can be run independently to generate training data
- **Format Compatibility**: Outputs standard HDF5 format used by train.py
- **Flexible Input**: Works with any .vrp files following CVRPLIB format
- **Solution Preservation**: Maintains generated .sol files for future reference

### Testing and Demo

To test the CVRPLIB pipeline or see a demonstration:

```bash
# Run comprehensive test suite
cd training_data_sampling
python test_cvrplib_pipeline.py

# Run demo with sample data
python demo_cvrplib_pipeline.py
```

The test suite validates the complete pipeline including vrplib reading, VROOM solution generation, HDF5 conversion, and compatibility with the training system.

### Best Practices

1. **Instance Selection**: Choose diverse instances with varying sizes and characteristics
2. **Solution Verification**: Verify generated solutions match expected costs when available
3. **Data Split**: Use appropriate train/validation splits for generalization
4. **Performance Monitoring**: Monitor VROOM solution times for computational budget planning

### Troubleshooting

**Common Issues:**

- **Missing vrplib**: Install with `pip install vrplib`
- **VROOM failures**: Ensure instances have valid coordinate and demand data
- **Memory issues**: Process large datasets in batches if needed
- **Permission errors**: Ensure write access to output directory

**Note**: The system has been successfully tested and verified to work with:
- Python 3.11
- VROOM solver (pyvroom 1.14.0)
- Gurobi optimizer
- PyTorch with Graph Transformer networks
- vrplib library for CVRPLIB processing
