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

To run the neural embedded framework using Multi-Layer Perceptron networks:

```bash
cd neo-lrp-MLP
python neo_lrp_execute.py
```

The script will automatically:
- Use pre-trained ONNX models from `../pre_trained_models/` (GVS, PSCC, RSCC)
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_mlp_results.xlsx`

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
- `GVS/`: ONNX models for GVS sampling method
- `PSCC/`: ONNX models for PSCC sampling method  
- `RSCC/`: ONNX models for RSCC sampling method

Each sampling method directory contains:
- `model_phi_*.onnx`: Pre-trained phi models for cost prediction
- `model_rho_*.onnx`: Pre-trained rho models for cost prediction

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