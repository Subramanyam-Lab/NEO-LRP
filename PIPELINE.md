# NEO-LRP-GT Training Pipeline

## Quick Start

### 1. Process CVRPLIB Data
```bash
cd training_data_sampling
python cvrplib_processor.py /path/to/vrp/files output_data.h5
```

### 2. Pre-train (Optional)
```bash
cd neo-lrp-GT
python pre_train_simple.py --data-file ../training_data_sampling/output_data.h5 --epochs 50
```

### 3. Train Model
```bash
cd neo-lrp-GT
python train.py --data-file ../training_data_sampling/output_data.h5
```

## Example with Test Data

```bash
# Test the pipeline
cd neo-lrp-GT/tests
./test_pipeline_quick.sh

# Train with existing test data
cd neo-lrp-GT
python train.py --data-file test_data.h5
```

## Input/Output

- **Input**: CVRPLIB `.vrp` files
- **Processing**: Creates `.h5` training data
- **Output**: Trained model in `model_state/` folder

That's it!