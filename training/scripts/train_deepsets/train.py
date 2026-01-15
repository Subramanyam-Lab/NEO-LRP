"""
DeepSets training script with early stopping and model checkpointing.
Trains phi and rho network for routing cost prediction,
exports trained models to ONNX format and also tracks comprehensive metrics.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import time
import os
import random
import argparse
import csv
import json
import logging

from architecture import DeepSetArchitecture
from preprocessing import preprocess_data


num_instances = int(os.getenv('num_instances'))
normalization_mode = os.getenv('normalization_mode')
mode = os.getenv('mode')  # 'scaled' or 'unscaled'
file_path_train_val = os.getenv('file_path_train_val')
file_path_test = os.getenv('file_path_test')
norm_json_path = os.getenv('norm_json_path')
tag = f"vroom_{mode}"  # "vroom_scaled" or "vroom_unscaled"

print(f"~~~~~Training Configuration~~~~")
print(f"Mode: {mode}")
print(f"Tag: {tag}")
print(f"Normalization: {normalization_mode}")
print(f"Num instances: {num_instances}")
print(f"Train/Val file: {file_path_train_val}")
print(f"Test file: {file_path_test}")
print(f"Norm JSON: {norm_json_path}")
print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
    file_path_train_val,
    file_path_test,
    solver_tag=tag,
    num_instances=num_instances,
    seed=42,
    normalization_mode=normalization_mode,
    norm_json_path=norm_json_path
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model_statistics(model):
    total_neurons = sum(layer.out_features for layer in model.modules() if isinstance(layer, nn.Linear))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_weights = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)
    total_biases = sum(p.numel() for p in model.parameters() if len(p.shape) == 1)
    total_non_zero_weights = sum(p.count_nonzero() for p in model.parameters() if len(p.shape) > 1)
    total_non_zero_biases = sum(p.count_nonzero() for p in model.parameters() if len(p.shape) == 1)

    return total_neurons, total_trainable_params, total_weights, total_biases, total_non_zero_weights, total_non_zero_biases

def master(config, metrics=True, exportonnx=True, testing=True, seed=42, N_dim=3, checkpoint_suffix=""):

    set_seed(seed)

    batch_size = int(config["batch_size"])
    num_epochs = int(config["num_epochs"])
    learning_rate = config["learning_rate"]

    checkpoint_dir = f"checkpoints_{checkpoint_suffix}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, f'best_model_{num_instances}.pth')
    last_epoch_model_path = os.path.join(checkpoint_dir, f'last_epoch_model_{num_instances}.pth')

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running on:", device)
    print("Device status:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    model = DeepSetArchitecture(N_dim, config)
    model.to(device)

    criterion_cost = nn.MSELoss(reduction='mean')

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float('inf')
    early_stopping_patience = config["early_stopping_patience"]
    early_stopping_counter = 0

    train_losses_cost = []
    val_losses_cost = []

    train_val_start_time = time.time()
    print(f"Training started with num_instances={num_instances}")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        total_instances_train = 0  # Number of instances in the training set

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs_cost = model(inputs)
            targets_cost = targets[:, 0]

            loss_cost = criterion_cost(outputs_cost, targets_cost)
            loss_cost.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            epoch_train_loss += loss_cost.item() * batch_size
            total_instances_train += batch_size

        epoch_train_loss /= total_instances_train
        train_losses_cost.append(epoch_train_loss)

        # Validation loop
        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0.0
            total_instances_val = 0  # Number of instances in the validation set

            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_cost = model(inputs)
                targets_cost = targets[:, 0]

                val_loss_cost = criterion_cost(outputs_cost, targets_cost)

                batch_size = inputs.shape[0]
                epoch_val_loss += val_loss_cost.item() * batch_size
                total_instances_val += batch_size

            epoch_val_loss /= total_instances_val
            val_losses_cost.append(epoch_val_loss)

            torch.save(model.state_dict(), last_epoch_model_path)
            scheduler.step()

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    train_val_end_time = time.time()
    train_val_time_sec = train_val_end_time - train_val_start_time
    print(f"Training completed in {train_val_time_sec:.2f} seconds")

    if early_stopping_counter < early_stopping_patience:
        print("Training completed without triggering early stopping.")

    if testing:
        if os.path.exists(best_model_path):
            print("Loading the best model from this run.")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        elif os.path.exists(last_epoch_model_path):
            print("Loading the model from the last epoch.")
            model.load_state_dict(torch.load(last_epoch_model_path, map_location=device))
        else:
            raise FileNotFoundError("No model checkpoint found. Ensure that training has been performed.")

        with torch.no_grad():
            train_start_time = time.time()
            model.eval()
            train_loss = 0.0
            total_instances_train = 0
            train_opt_gap_cost_abs = 0.0
            train_opt_gap_cost = 0.0
            total_new_metrics_train = 0.0
            total_new_metrics_abs_train = 0.0

            actual_costs_train = []
            predicted_costs_train = []
            train_abs_gaps = []  

            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_cost = model(inputs)
                targets_cost = targets[:, 0]

                actual_costs_train.extend(targets_cost.cpu().numpy())
                predicted_costs_train.extend(outputs_cost.cpu().detach().numpy())

                loss = criterion_cost(outputs_cost, targets_cost)
                train_loss += loss.item() * inputs.size(0)

                total_instances_train += inputs.size(0)

                train_opt_gap_cost_abs += torch.abs((targets_cost - outputs_cost) / targets_cost).sum().item()
                train_opt_gap_cost += ((targets_cost - outputs_cost) / targets_cost).sum().item()

                new_metrics_batch_train = ((targets_cost) - (outputs_cost)) / (targets_cost)
                total_new_metrics_train += new_metrics_batch_train.sum().item()

                new_metrics_abs_batch_train = torch.abs((targets_cost - outputs_cost) / targets_cost) * 100
                total_new_metrics_abs_train += new_metrics_abs_batch_train.sum().item()

                train_abs_gaps.extend(new_metrics_abs_batch_train.cpu().numpy())

            train_loss /= total_instances_train

            train_opt_gap_cost_abs = (train_opt_gap_cost_abs / total_instances_train) * 100
            train_opt_gap_cost = (train_opt_gap_cost / total_instances_train) * 100

            avg_new_metrics_train = (total_new_metrics_train / total_instances_train) * 100
            avg_new_metrics_abs_train = (total_new_metrics_abs_train / total_instances_train)

            train_end_time = time.time()
            train_time_sec = train_end_time - train_start_time
            print(f"Training evaluation completed in {train_time_sec:.2f} seconds")

        with torch.no_grad():
            test_start_time = time.time()
            model.eval()
            test_loss = 0.0
            test_loss_cost = 0.0
            total_instances_test = 0  # Number of instances in the testing data
            test_opt_gap_cost_abs = 0.0
            test_opt_gap_cost = 0.0
            total_new_metrics = 0.0
            total_new_metrics_abs = 0.0

            actual_costs = []
            predicted_costs = []
            test_abs_gaps = []  

            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_cost = model(inputs)
                targets_cost = targets[:, 0]

                actual_costs.extend(targets_cost.cpu().numpy())
                predicted_costs.extend(outputs_cost.cpu().detach().numpy())

                loss = criterion_cost(outputs_cost, targets_cost)
                test_loss += loss.item() * inputs.size(0)

                total_instances_test += inputs.size(0)

                test_opt_gap_cost_abs += torch.abs((targets_cost - outputs_cost) / targets_cost).sum().item()
                test_opt_gap_cost += ((targets_cost - outputs_cost) / targets_cost).sum().item()

                new_metrics_batch = ((targets_cost) - (outputs_cost)) / (targets_cost)
                total_new_metrics += new_metrics_batch.sum().item()

                new_metrics_abs_batch = torch.abs((targets_cost - outputs_cost) / targets_cost) * 100
                total_new_metrics_abs += new_metrics_abs_batch.sum().item()

                test_abs_gaps.extend(new_metrics_abs_batch.cpu().numpy())

            test_loss /= total_instances_test

            test_opt_gap_cost_abs = (test_opt_gap_cost_abs / total_instances_test) * 100
            test_opt_gap_cost = (test_opt_gap_cost / total_instances_test) * 100

            avg_new_metrics = (total_new_metrics / total_instances_test) * 100
            avg_new_metrics_abs = (total_new_metrics_abs / total_instances_test)

            test_end_time = time.time()
            test_time_sec = test_end_time - test_start_time
            print(f"Testing completed in {test_time_sec:.2f} seconds")

            min_costs = min(min(actual_costs), min(predicted_costs))
            max_costs = max(max(actual_costs), max(predicted_costs))

            actual_vs_predicted_filename = f'actual_vs_predicted_{num_instances}.csv'
            costs_plot_filename = f'costs_{num_instances}.png'

            plt.figure(figsize=(10, 6))
            plt.scatter(actual_costs, predicted_costs, alpha=0.5, color='red', label='Predicted')
            plt.plot([min_costs, max_costs], [min_costs, max_costs], color='blue', label='Perfect prediction')
            plt.xlim([min_costs, max_costs])
            plt.ylim([min_costs, max_costs])
            plt.title('Actual vs Predicted Costs')
            plt.xlabel('Actual Costs')
            plt.ylabel('Predicted Costs')
            plt.legend()
            plt.close()
            print(f"Costs plot saved to {costs_plot_filename}")

            train_abs_gaps_array = np.array(train_abs_gaps)
            test_abs_gaps_array = np.array(test_abs_gaps)

            mean_train_abs_gap = np.mean(train_abs_gaps_array)
            median_train_abs_gap = np.median(train_abs_gaps_array)
            lower_quartile_train_abs_gap = np.percentile(train_abs_gaps_array, 25)
            upper_quartile_train_abs_gap = np.percentile(train_abs_gaps_array, 75)
            whisker_low_train_abs_gap = np.percentile(train_abs_gaps_array, 5)
            whisker_high_train_abs_gap = np.percentile(train_abs_gaps_array, 95)

            mean_test_abs_gap = np.mean(test_abs_gaps_array)
            median_test_abs_gap = np.median(test_abs_gaps_array)
            lower_quartile_test_abs_gap = np.percentile(test_abs_gaps_array, 25)
            upper_quartile_test_abs_gap = np.percentile(test_abs_gaps_array, 75)
            whisker_low_test_abs_gap = np.percentile(test_abs_gaps_array, 5)
            whisker_high_test_abs_gap = np.percentile(test_abs_gaps_array, 95)

    else:
        print("Testing is disabled.")
        test_loss = None
        test_loss_cost = None
        actual_costs = []
        predicted_costs = []

    if exportonnx:
        phi_model = model.phi
        rho_model = model.rho
        trained_models_dir = os.getenv('trained_models_dir', '.')

        phi_dir = os.path.join(trained_models_dir, "deepsets", mode, "phi", normalization_mode)
        rho_dir = os.path.join(trained_models_dir, "deepsets", mode, "rho", normalization_mode)

        os.makedirs(phi_dir, exist_ok=True)
        os.makedirs(rho_dir, exist_ok=True)

        phi_onnx_filename = os.path.join(phi_dir, f"{num_instances}.onnx")
        rho_onnx_filename = os.path.join(rho_dir, f"{num_instances}.onnx")

        dummy_input_phi = torch.randn(1, N_dim).to(device)
        torch.onnx.export(phi_model, dummy_input_phi, phi_onnx_filename)
        print(f"Phi model ONNX exported to {phi_onnx_filename}")

        dummy_input_rho = torch.randn(1, config['latent_space_dimension']).to(device)
        torch.onnx.export(rho_model, dummy_input_rho, rho_onnx_filename)
        print(f"Rho model ONNX exported to {rho_onnx_filename}")

    else:
        print("No ONNX Export")

    if metrics:
        l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        total_neurons, total_trainable_params, total_weights, total_biases, total_non_zero_weights, total_non_zero_biases = get_model_statistics(model)

        metrics = {
            'train_losses_cost': train_losses_cost,
            'val_losses_cost': val_losses_cost,
            'best_val_loss': best_loss,
            'test_cost_loss': test_loss,
            'test_opt_gap_cost_abs_percent': test_opt_gap_cost_abs,
            'test_opt_gap_cost_percent': test_opt_gap_cost,
            'avg_new_metrics': avg_new_metrics,
            'avg_new_metrics_abs': avg_new_metrics_abs,
            'L1 Norm': l1_norm.item(),
            'Total Neurons': total_neurons,
            'Total Trainable Params': total_trainable_params,
            'Total Weights': total_weights,
            'Total Biases': total_biases,
            'Total Non-zero Weights': total_non_zero_weights,
            'Total Non-zero Biases': total_non_zero_biases,
            'Training Time (s)': train_val_time_sec,
            'Testing Time (s)': test_time_sec,
            'train_opt_gap_cost_abs_percent': train_opt_gap_cost_abs,
            'train_opt_gap_cost_percent': train_opt_gap_cost
        }

        metrics.update({
            'mean_train_abs_gap': mean_train_abs_gap,
            'median_train_abs_gap': median_train_abs_gap,
            'lower_quartile_train_abs_gap': lower_quartile_train_abs_gap,
            'upper_quartile_train_abs_gap': upper_quartile_train_abs_gap,
            'whisker_low_train_abs_gap': whisker_low_train_abs_gap,
            'whisker_high_train_abs_gap': whisker_high_train_abs_gap,
            'mean_test_abs_gap': mean_test_abs_gap,
            'median_test_abs_gap': median_test_abs_gap,
            'lower_quartile_test_abs_gap': lower_quartile_test_abs_gap,
            'upper_quartile_test_abs_gap': upper_quartile_test_abs_gap,
            'whisker_low_test_abs_gap': whisker_low_test_abs_gap,
            'whisker_high_test_abs_gap': whisker_high_test_abs_gap
        })

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses_cost) + 1), train_losses_cost, label='Training Loss')
        plt.plot(range(1, len(val_losses_cost) + 1), val_losses_cost, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        loss_curve_filename = f'loss_curve_{num_instances}.png'
        plt.savefig(os.path.join(checkpoint_dir, loss_curve_filename))
        plt.close()
        print(f"Loss curve saved to {os.path.join(checkpoint_dir, loss_curve_filename)}")

        return metrics
    else:
        print('HPO searching stage')
        return metrics
