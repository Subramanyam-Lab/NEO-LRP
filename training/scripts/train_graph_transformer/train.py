"""
Graph Transformer Training Script
"""
import argparse
import os
import torch
import schedulefree
from torch_geometric.loader import DataLoader

from architecture import GraphTransformerNetwork
from utils import prepare_pretrain_data
from utils_train import get_loss_function
from preprocessing import preprocess_data

mode = os.getenv('mode')
normalization_mode = os.getenv('normalization_mode')
file_path_train_val = os.getenv('file_path_train_val')
file_path_test = os.getenv('file_path_test')
norm_json_path = os.getenv('norm_json_path')
h5_cache_dir = os.getenv('h5_cache_dir')
trained_models_dir = os.getenv('trained_models_dir', '.')
num_instances = os.getenv('num_instances')

if num_instances is not None:
    num_instances = int(num_instances)

tag = f"vroom_{mode}"

print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Graph Transformer Training Configuration")
print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Mode: {mode}")
print(f"Tag: {tag}")
print(f"Normalization: {normalization_mode}")
print(f"Train or Val: {file_path_train_val}")
print(f"Test: {file_path_test}")
print(f"Norm JSON: {norm_json_path}")
print(f"H5 Cache: {h5_cache_dir}")
print(f"Models dir: {trained_models_dir}")
print(f"Num instances: {num_instances}")
print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def get_or_create_h5():
    """generate h5 file if it doesn't exist, return path"""
    os.makedirs(h5_cache_dir, exist_ok=True)
    h5_filename = f"exported_{tag}_{normalization_mode}.h5"
    h5_path = os.path.join(h5_cache_dir, h5_filename)
    
    if os.path.exists(h5_path):
        print(f"Using existing H5 file: {h5_path}")
        return h5_path
    
    print(f"Generating H5 file: {h5_path}")
    
    # we change to h5_cache_dir so h5 file is saved there
    original_dir = os.getcwd()
    os.chdir(h5_cache_dir)
    
    try:
        preprocess_data(
            file_path_train_val=file_path_train_val,
            file_path_test=file_path_test,
            solver_tag=tag,
            normalization_mode=normalization_mode,
            norm_json_path=norm_json_path
        )
    finally:
        os.chdir(original_dir)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Failed to generate H5 file: {h5_path}")
    
    print(f"Generated H5 file: {h5_path}")
    return h5_path


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target))) * 100


def train():
    epochs = 50

    # get or create h5 file
    data_h5_path = get_or_create_h5()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data, val_data, _ = prepare_pretrain_data(
        data_h5_path,
        split_ratios=[0.9, 0.1, 0.0],
        num_entries=num_instances
    )
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    config = {
        'encoding_dim': 16,
        'batch_size': 8,
        'dropout': 0.3,
        'initial_lr': 0.001,
        'heads': 4,
        'normalization': 'graph_norm',
        'activation': 'elu',
        'num_gat_layers': 5,
        'loss_function': 'huber',
        'beta': False,
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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Configuration: {config}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # training
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

        # validation
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

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch

                model_dir = os.path.join(trained_models_dir, "graph_transformer", mode, normalization_mode)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{num_instances}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model: {model_path} (epoch {epoch}, val_loss={avg_val_loss:.6f})")

        print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f}, MAPE: {avg_train_mape:.2f} | '
              f'Val Loss: {avg_val_loss:.4f}, MAPE: {avg_val_mape:.2f}')

        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    train()