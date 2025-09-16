# Neural Embedded Location-Routing Problem - MLP Implementation

This directory contains the Multi-Layer Perceptron (MLP) implementation of the neural embedded framework for solving Location-Routing Problems.

## Contents

- `neo_lrp_execute.py`: Main execution script for MLP implementation
- `neural_embedded_model.py`: Neural embedded LRP solver using MLP networks
- `network.py`: MLP network utilities for ONNX model loading
- `solver_cvrp.py`: VRPSolverEasy-based CVRP solver integration
- `dataparse.py`: Data parsing and normalization utilities
- `utils.py`: Utility functions for processing
- `train.py`: Training utilities for MLP models
- `pre_train.py`: Pre-training script for MLP models

## Usage

### Prerequisites

#### BaPCod Library Installation (Required for VRPSolverEasy)

The MLP implementation uses VRPSolverEasy which requires the BaPCod library for exact VRP solutions.

**For Intel Macs:**
```bash
# Download BaPCod from: https://bapcod.math.u-bordeaux.fr
# Extract and copy the library to VRPSolverEasy
cp /path/to/bapcod/libbapcod-shared.dylib ./venv/lib/python3.11/site-packages/VRPSolverEasy/lib/Darwin/
```

**For Apple Silicon Macs (M1/M2/M3):**
```bash
# Install Rosetta 2
softwareupdate --install-rosetta --agree-to-license

# Create x86_64 virtual environment
arch -x86_64 /usr/bin/python3 -m venv venv_x86
arch -x86_64 ./venv_x86/bin/pip install -r requirements.txt

# Copy BaPCod library and remove quarantine
cp /path/to/bapcod/libbapcod-shared.dylib ./venv_x86/lib/python3.9/site-packages/VRPSolverEasy/lib/Darwin/
xattr -d com.apple.quarantine ./venv_x86/lib/python3.9/site-packages/VRPSolverEasy/lib/Darwin/libbapcod-shared.dylib

# Run with x86_64 Python
arch -x86_64 ./venv_x86/bin/python neo_lrp_execute.py
```

### Running with Pre-trained Models

To run the neural embedded framework using pre-trained MLP networks:

**Intel Macs:**
```bash
cd neo-lrp-MLP
python neo_lrp_execute.py
```

**Apple Silicon Macs:**
```bash
cd neo-lrp-MLP
arch -x86_64 ../venv_x86/bin/python neo_lrp_execute.py
```

The script will automatically:
- Use pre-trained ONNX models from `../pre_trained_models/`
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_mlp_results.xlsx`

### Training from Scratch with New Dataset

To train new MLP models (PhiNet and RhoNet) with your own dataset, follow these steps:

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
cd ../neo-lrp-MLP
python pre_train.py
```

This script uses Weights & Biases (wandb) to perform Bayesian optimization over various hyperparameters including network architecture, learning rate, and training settings.

**Note:** Make sure to update the data path in `pre_train.py` to point to your generated training data.

#### 3. Model Training

Once you have optimal hyperparameters, train the final models:

```bash
python train.py
```

**Configuration:**
- Update the data path in `train.py` to use your training data
- Modify hyperparameters based on results from step 2
- The script will automatically export trained models to ONNX format

#### 4. Using Your Trained Models

After training, update the model paths in `neo_lrp_execute.py`:

```python
phi_loc = "path/to/your/phi_net.onnx"  # Update this path
rho_loc = "path/to/your/rho_net.onnx"  # Update this path
```

Then run the execution script:

```bash
python neo_lrp_execute.py
```

## Configuration

The main parameters can be modified in `neo_lrp_execute.py`:

```python
# Configuration parameters
BFS = "solutions"  # Directory for storing intermediate solutions
phi_loc = "../pre_trained_models/mlp_phi.onnx"  # Phi model path
rho_loc = "../pre_trained_models/mlp_rho.onnx"  # Rho model path
existing_excel_file = "results/neo_lrp_mlp_results.xlsx"  # Results file
sheet_name = "results"  # Excel sheet name
fi_mode_input = "dynamic"  # Normalization mode
directory_path = "../prodhon_dataset"  # Dataset directory
```

## Requirements

This implementation requires:
- PyTorch with CUDA support (recommended)
- Gurobi optimizer (for MIP solving)
- VRPSolverEasy with BaPCod library (for exact VRP solutions)
- ONNX and onnx2torch (for MLP model loading)

## Pre-trained Models

The MLP implementation uses ONNX models located in `../pre_trained_models/`:
- `feed_forward/`: Main MLP models directory
  - `phi_net.onnx`: Pre-trained PhiNet model for node embedding
  - `rho_net.onnx`: Pre-trained RhoNet model for cost prediction
- `mlp_phi.onnx`, `mlp_rho.onnx`: Alternative MLP models

## Results

The framework generates results in the `results/` directory:

### `neo_lrp_mlp_results.xlsx`
Contains comprehensive solution metrics including:
- **Instance details**: Instance name and problem parameters
- **Solution costs**: FLP cost, VRP cost, and total LRP cost
- **Route information**: Number of routes in optimal solution
- **Execution times**: LRP solver time, VRPSolverEasy solver time
- **Cost comparisons**: VRPSolverEasy computed VRP cost vs. actual LRP cost
- **Best known solutions (BKS)**: Benchmark comparison
- **Gap analysis**: Optimization gap and prediction gap metrics

## Example Execution

```bash
cd neo-lrp-MLP
source ../venv/bin/activate  # or activate your Python 3.11 environment
python neo_lrp_execute.py
```

**Sample Output:**
```
Working on: ../prodhon_dataset/coord100-10-3.dat
Run 1 for instance coord100-10-3.dat

Adding neural embedding constraints...
Gurobi Optimizer version 12.0.3 build v12.0.3rc0

Optimal solution found (tolerance 1.00e-02)
Best objective 2.709928196581e+05, best bound 2.706830699383e+05, gap 0.1143%

[VRPSolverEasy] depot 2 → cost=29375, routes=8
[VRPSolverEasy] depot 4 → cost=42732, routes=9
[VRPSolverEasy] depot 5 → cost=33570, routes=8
Final VRP cost (variable+fixed): 105677
Number of routes: 25
```

**Note**: The system has been successfully tested and verified to work with:
- Python 3.11
- VRPSolverEasy with BaPCod library
- Gurobi optimizer
- ONNX models for MLP implementation