# Neural Embedded Mixed-Integer Optimization for Location-Routing Problems

This repository contains the implementation and datasets for the Neural Embedded Mixed-Integer Optimization approach to solving Location-Routing Problems.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

This project implements a neural embedded framework for solving Location-Routing Problems, combining traditional optimization techniques with neural networks for improved solution quality and computational efficiency.

## Repository Structure

### 1. `prodhon_dataset/`
Contains benchmark datasets used in numerical experiments.

### 2. `neo-lrp/`
Core implementation of the neural embedded framework using Graph Transformer:
- `neo_lrp_execute_vroom.py`: Main execution script using VROOM solver
- `neural_embedded_model.py`: Neural embedded LRP solver implementation
- `net.py`: Graph Transformer network architecture
- `dataparse.py`: Data parsing and normalization utilities
- `solver_cvrp_vroom.py`: VROOM-based CVRP solver integration
- `utils.py`, `utils_train.py`: Utility functions for training and processing

### 3. `neo-lrp-MLP/`
Alternative implementation using Multi-Layer Perceptron (MLP) networks:
- `neo_lrp_execute.py`: Main execution script for MLP implementation
- `neural_embedded_model.py`: Neural embedded LRP solver using MLP networks
- `network.py`: MLP network utilities for ONNX model loading
- `solver_cvrp.py`: VRPSolverEasy-based CVRP solver integration

### 4. `pre_trained_models/`
Pre-trained neural networks for routing cost prediction:
- `graph_transformer.pth`: Pre-trained Graph Transformer model (for GT implementation)
- `GVS/`, `PSCC/`, `RSCC/`: ONNX models for different sampling methods (for MLP implementation)
  - `model_phi_*.onnx`: Pre-trained phi models for cost prediction
  - `model_rho_*.onnx`: Pre-trained rho models for cost prediction

### 5. `training_data_sampling/`
Training data generation utilities:
- `data_generation.py`: Script for generating training data using VROOM solver

### 6. `flp/`
FLP model implementation:
- `dataparse.py`: Data parsing utilities
- `flp_execute.py`: Main execution script
- `flp.py`: Facility assignment logic
- `vrp.py`: Vehicle routing implementation
- `solver_cvrp.py`: Exact routing solver integration

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Subramanyam-Lab/NEO-LRP.git
cd neo-lrp
```

2. **Create and Activate Python Virtual Environment (Python 3.11 recommended)**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Note**: This project requires Python 3.11 for VROOM compatibility. If you don't have Python 3.11, install it first:
```bash
# On macOS with Homebrew
brew install python@3.11

# On Ubuntu/Debian
sudo apt install python3.11 python3.11-venv
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

This repository provides two implementations of the neural embedded framework:

### 1. Graph Transformer Implementation (`neo-lrp/`)

To run the neural embedded framework using Graph Transformer:

```bash
cd neo-lrp
python neo_lrp_execute_vroom.py
```

The script will automatically:
- Use the pre-trained Graph Transformer model from `../pre_trained_models/graph_transformer.pth`
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_gt_results.xlsx`

### 2. MLP Implementation (`neo-lrp-MLP/`)

To run the neural embedded framework using Multi-Layer Perceptron networks:

```bash
cd neo-lrp-MLP
python neo_lrp_execute.py
```

The script will automatically:
- Use pre-trained ONNX models from `../pre_trained_models/` (GVS, PSCC, RSCC)
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_mlp_results.xlsx`

### Configuration

#### Graph Transformer Implementation (`neo-lrp-GT/`)
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

#### MLP Implementation (`neo-lrp-MLP/`)
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

### Requirements

Make sure to install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: This implementation requires:
- PyTorch with CUDA support (recommended)
- Gurobi optimizer (for MIP solving)
- VROOM solver (for exact VRP solutions in GT implementation)
- VRPSolverEasy with BaPCod library (for exact VRP solutions in MLP implementation)
- PyTorch Geometric (for graph neural networks in GT implementation)
- ONNX and onnx2torch (for MLP implementation)

## Results

The framework generates comprehensive results in the `results/` directory:

### Graph Transformer Implementation Results

#### 1. `neo_lrp_gt_results.xlsx`
Contains comprehensive solution metrics including:
- **Instance details**: Instance name and problem parameters
- **Solution costs**: FLP cost, VRP cost, and total LRP cost
- **Route information**: Number of routes in optimal solution
- **Execution times**: LRP solver time, VROOM solver time
- **Cost comparisons**: VROOM computed VRP cost vs. actual LRP cost
- **Best known solutions (BKS)**: Benchmark comparison
- **Gap analysis**: Optimization gap and prediction gap metrics

### MLP Implementation Results

#### 1. `neo_lrp_mlp_results.xlsx`
Contains comprehensive solution metrics including:
- **Instance details**: Instance name and problem parameters
- **Solution costs**: FLP cost, VRP cost, and total LRP cost
- **Route information**: Number of routes in optimal solution
- **Execution times**: LRP solver time, VRPSolverEasy solver time
- **Cost comparisons**: VRPSolverEasy computed VRP cost vs. actual LRP cost
- **Best known solutions (BKS)**: Benchmark comparison
- **Gap analysis**: Optimization gap and prediction gap metrics

### 2. `solutions/` directory
Contains intermediate VRP instances generated during optimization:
- Individual depot-customer assignments
- CVRPLIB format files for each feasible solution
- Route cost predictions from neural network

### 3. `log_files/` directory
Contains detailed execution logs:
- MIP solver logs
- Neural network processing logs
- Performance metrics and timing information

## Execution Examples

### Graph Transformer Implementation

```bash
cd neo-lrp
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

### MLP Implementation

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
- VROOM solver (pyvroom 1.14.0) for GT implementation
- VRPSolverEasy with BaPCod library for MLP implementation
- Gurobi optimizer
- PyTorch with Graph Transformer networks
- ONNX models for MLP implementation

### Training Data Generation

To generate training data for neural networks:

```bash
cd training_data_sampling
python data_generation.py
```

This script will:
- Process instances from `../prodhon_dataset/`
- Generate VRP solutions using VROOM solver
- Create training datasets in HDF5 format
- Support different sampling methods (GVS, PSCC, RSCC)

## Citation
If you happen to use our pre-trained models or codes please cite the following papers: 
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


For questions and support, please open an issue in the repository or contact the authors.
