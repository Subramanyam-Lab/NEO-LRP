import argparse
import os
import wandb
import torch
import schedulefree
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net import GraphTransformerNetwork
from utils import prepare_pretrain_data
from utils_train import get_loss_function

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'encoding_dim': {'values': [8, 16, 32, 64, 128]},
        'batch_size': {'values': [8, 16, 32, 64, 128]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'initial_lr': {'values': [0.1, 0.01, 0.001, 0.0001]},
        'heads': {'values': [4, 8]},
        'normalization': {'values': ['graph_norm', 'batch_norm', 'layer_norm']},
        'activation': {'values': ['elu', 'relu', 'leaky_relu']},
        'num_gat_layers': {'values': [2, 3, 4, 5]},
        'loss_function': {'values': ['mse', 'huber', 'smooth_l1']},
        'beta': {'values': [True, False]},
        'decode_method': {'values': ['pool']},
    }
}
sweep_id = wandb.sweep(sweep_config, project="2evrp-ncp-scaled")


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target))) * 100

def train(config = None, data_file = None):
    if data_file is None:
        raise ValueError("HDF5 training data file must be provided")

    run = wandb.init(project = '2evrp-ncp', config = config)
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, _ = prepare_pretrain_data(
        data_file,
        split_ratios=[0.8, 0.2, 0.0]
    )

    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    val_loader = DataLoader(val_data, batch_size = config.batch_size, shuffle = False, pin_memory = True, num_workers = 4)

    model = GraphTransformerNetwork(
        in_channels = config.encoding_dim,
        hidden_channels = config.encoding_dim,
        out_channels = config.encoding_dim,
        heads = config.heads,
        beta = config.beta,
        dropout = config.dropout,
        normalization = config.normalization,
        num_gat_layers = config.num_gat_layers,
        activation = config.activation,
        decode_method = config.decode_method
    ).to(device)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config.initial_lr)
    criterion = get_loss_function(config.loss_function)

    wandb.watch(model, log = 'all', log_freq = 1)

    best_val_loss = float('inf')

    for epoch in range(150):
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
        wandb.log({"epoch":epoch, "train_mape":avg_train_mape, "train_loss":avg_train_loss})

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
            wandb.log({"epoch":epoch, "val_mape":avg_val_mape, "val_loss":avg_val_loss})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
        torch.cuda.empty_cache()
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAPE: {avg_train_mape}, Val Loss: {avg_val_loss}, Val MAPE: {avg_val_mape}')

    run.finish()

def create_sweep_with_data_file(data_file):
    """Create a sweep agent that uses the specified data file."""
    def train_wrapper():
        return train(data_file=data_file)
    return train_wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train GraphTransformer with hyperparameter sweep")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to HDF5 training data file")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of sweep runs to execute")

    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {args.data_file}")

    print(f"🚀 Starting hyperparameter sweep with data file: {args.data_file}")
    print(f"📊 Running {args.count} sweep iterations")

    # Create sweep agent with custom data file
    sweep_function = create_sweep_with_data_file(args.data_file)
    wandb.agent(sweep_id, sweep_function, count=args.count)
