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
Core implementation of the neural embedded framework:
- `flp_org/`: Initial MIP model solution generators
- `dataparse/`: Dataset conversion utilities
- `network/`: Neural network transformation tools (ONNX to PyTorch)
- `neural_embedded_model/`: Main neural-embedded implemention
- `solver_cvrp/`: VRPSolverEasy integration for exact route cost calculation

### 3. `pre_trained_model/`
Pre-trained neural networks (ONNX format) for routing cost prediction:
- Includes pre-trained $\hat{\phi}$ and $\hat{\rho}$ models for GVS, PSCC, and RSCC sampling methods

### 4. `flp/`
FLP model implementation:
- `dataparser/`: Data format conversion tools
- `flp_execute/`: Main execution script
- `flp.py`: Facility assignment logic
- `vrp.py`: Vehicle routing implementation
- `solver_cvrp/`: Exact routing solver integration

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/neo-lrp.git
cd neo-lrp
```

2. **Create and Activate Conda Environment**
```bash
conda create --name neos_lrp python=3.9
conda activate neos_lrp
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

To run the neural embedded framework:

```bash
python neo_lrp_execute.py \
    --BFS /path/to/solutions \
    --phi_loc /path/to/phi_model.onnx \
    --rho_loc /path/to/rho_model.onnx \
    --existing_excel_file results.xlsx \
    --sheet_name "Results" \
    --normalization dynamic
```

### Command Line Arguments

- `--BFS`: Directory path for storing integer feasible solutions (will not be used)
- `--phi_loc`: Path to the phi (φ) model file (ONNX format)
- `--rho_loc`: Path to the rho (ρ) model file (ONNX format)
- `--existing_excel_file`: Path to the Excel file for storing results
- `--sheet_name`: Name of the worksheet in the Excel file
- `--normalization`: Normalization strategy (`fixed` for GVS or `dynamic` for RSCC and PSCC)

## Results

The framework generates two Excel files in the `results/` directory:

### 1. `neo_lrp_results.xlsx`
Contains comprehensive solution metrics including:
- Instance details and solution costs (FLP, VRP, LRP)
- Number of routes in optimal solution
- Execution times (MIP+NN, initial solution, NN model)
- VRPSolverEasy computed costs and times
- Best known solutions (BKS)
- Optimization and prediction gaps

### 2. `flp.xlsx`
Contains FLP-specific metrics including:
- Instance details
- FLP and VRP costs
- Execution times for different components
- OR-Tools and VRPSolverEasy performance metrics
- Solution quality gaps

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
```

For questions and support, please open an issue in the repository or contact the authors.