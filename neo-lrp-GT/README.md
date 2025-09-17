
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
python pre_train.py --data-file /path/to/your/training_data.h5 --count 100
```

This script uses Weights & Biases (wandb) to perform Bayesian optimization over various hyperparameters including network architecture, learning rate, and training settings.

**Arguments:**
- `--data-file`: **Required**. Path to HDF5 training data file
- `--count`: Number of sweep runs to execute (default: 100)

#### 3. Model Training

Once you have optimal hyperparameters, train the final model:

```bash
python train.py --data-file /path/to/your/training_data.h5
```

**Arguments:**
- `--data-file`: **Required**. Path to HDF5 training data file

**Configuration:**
- Modify hyperparameters in the script based on results from step 2
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

## Test Data

For testing and demonstration purposes, a small test dataset `test_data.h5` is included in this directory. This file was generated using sample CVRPLIB instances and can be used to quickly test the training scripts:

```bash
# Test pre-training with the included test data
python pre_train.py --data-file test_data.h5 --count 5

# Test training with the included test data
python train.py --data-file test_data.h5
```

**Note:** The test dataset is small and intended only for testing functionality. For actual training, generate larger datasets using the methods described above.

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

Train the Graph Transformer using CVRPLIB benchmark instances.

### Quick Start

```bash
# 1. Process CVRPLIB data (.vrp files → .h5 training data)
cd training_data_sampling
python cvrplib_processor.py /path/to/vrp/files output_data.h5

# 2. Pre-train (optional)
cd ../neo-lrp-GT
python pre_train_simple.py --data-file ../training_data_sampling/output_data.h5 --epochs 50

# 3. Train model
python train.py --data-file ../training_data_sampling/output_data.h5
```

**Note:** Place your `.vrp` files in the `/path/to/vrp/files` directory. The processor will automatically generate solutions using VROOM if `.sol` files are missing.

### Test with Included Data

```bash
# Test the pipeline with existing test data
cd neo-lrp-GT/tests
./test_pipeline_quick.sh

# Train with test data
cd ../
python train.py --data-file test_data.h5
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
