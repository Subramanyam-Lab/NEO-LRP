
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

To run the neural embedded framework using Graph Transformer:

```bash
cd neo-lrp-GT
python neo_lrp_execute_vroom.py
```

The script will automatically:
- Use the pre-trained Graph Transformer model from `../pre_trained_models/graph_transformer.pth`
- Process all instances in `../prodhon_dataset/`
- Generate results in `results/neo_lrp_gt_results.xlsx`

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

**Note**: The system has been successfully tested and verified to work with:
- Python 3.11
- VROOM solver (pyvroom 1.14.0)
- Gurobi optimizer
- PyTorch with Graph Transformer networks
