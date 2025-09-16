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

### 2. `neo-lrp-GT/`
Graph Transformer implementation of the neural embedded framework. See [neo-lrp-GT/README.md](neo-lrp-GT/README.md) for detailed usage instructions.

### 3. `neo-lrp-MLP/`
Multi-Layer Perceptron (MLP) implementation of the neural embedded framework. See [neo-lrp-MLP/README.md](neo-lrp-MLP/README.md) for detailed usage instructions.

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

1. **Graph Transformer Implementation** (`neo-lrp-GT/`): Uses Graph Transformer networks for routing cost prediction. See [neo-lrp-GT/README.md](neo-lrp-GT/README.md) for detailed usage instructions.

2. **MLP Implementation** (`neo-lrp-MLP/`): Uses Multi-Layer Perceptron networks with ONNX models. See [neo-lrp-MLP/README.md](neo-lrp-MLP/README.md) for detailed usage instructions.

### Requirements

Make sure to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Results

Both implementations generate comprehensive results including solution metrics, execution times, and gap analysis. See the individual README files for detailed information about results and output formats:

- Graph Transformer results: [neo-lrp-GT/README.md](neo-lrp-GT/README.md#results)
- MLP results: [neo-lrp-MLP/README.md](neo-lrp-MLP/README.md#results)

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
