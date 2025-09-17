#!/usr/bin/env python3
"""
Simple Pre-training Script for GraphTransformer (No wandb required)

This is a simplified version of pre_train.py that works without wandb.
For full hyperparameter sweeps with wandb, use pre_train.py instead.

Usage:
    python pre_train_simple.py --data-file data.h5 [--epochs 50]
"""

import argparse
import os
import torch
import schedulefree
from torch_geometric.loader import DataLoader

from net import GraphTransformerNetwork
from utils import prepare_pretrain_data
from utils_train import get_loss_function


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target))) * 100


def train_simple(data_file, epochs=50):
    """
    Simple training loop without hyperparameter sweep.
    """
    print(f"🚀 Starting simple pre-training with {epochs} epochs")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Load data
    train_data, val_data, _ = prepare_pretrain_data(
        data_file,
        split_ratios=[0.8, 0.2, 0.0]
    )
    print(f"📊 Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Simple default configuration (can be modified here)
    config = {
        'encoding_dim': 64,
        'batch_size': 32,
        'dropout': 0.1,
        'initial_lr': 0.001,
        'heads': 8,
        'normalization': 'graph_norm',
        'activation': 'elu',
        'num_gat_layers': 3,
        'loss_function': 'smooth_l1',
        'beta': True,
        'decode_method': 'pool'
    }

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    model = GraphTransformerNetwork(
        in_channels=config['encoding_dim'],
        hidden_channels=config['encoding_dim'],
        out_channels=config['encoding_dim'],
        heads=config['heads'],
        beta=config['beta'],
        dropout=config['dropout'],
        normalization=config['normalization'],
        num_gat_layers=config['num_gat_layers'],
        activation=config['activation'],
        decode_method=config['decode_method']
    ).to(device)

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config['initial_lr'])
    criterion = get_loss_function(config['loss_function'])

    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Configuration: {config}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.train()
        total_loss = 0
        total_mape = 0

        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data, mask=data.mask)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mape += mape(out, data.y.view(-1, 1)).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape / len(train_loader)

        # Validation
        model.eval()
        optimizer.eval()
        with torch.no_grad():
            val_loss = 0
            val_mape = 0
            for data in val_loader:
                data.to(device)
                out = model(data, mask=data.mask)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_mape = val_mape / len(val_loader)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch

                # Save model
                os.makedirs("model_state", exist_ok=True)
                base_id = os.path.splitext(os.path.basename(data_file))[0]
                model_path = f"model_state/pretrain_{base_id}_epoch{epoch}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved best model: {model_path}")

        print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.2f} | '
              f'Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.2f}')

        # Clear cache periodically
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    print(f"🎉 Pre-training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple pre-training without wandb")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to HDF5 training data file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")

    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {args.data_file}")

    train_simple(args.data_file, args.epochs)