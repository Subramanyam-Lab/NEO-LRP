import os
import argparse
import torch
import torch.nn as nn
import schedulefree
from torch_geometric.loader import DataLoader

# Try to import wandb, but continue without it if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("⚠️  wandb not available. Training will continue without logging.")
    WANDB_AVAILABLE = False
    wandb = None

from net import GraphTransformerNetwork
from utils import prepare_pretrain_data
from utils_train import get_loss_function


# Disable wandb if needed (uncomment to disable)
# os.environ["WANDB_MODE"] = "disabled"


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target))) * 100


def train(config, data_file):
    """
    Training loop for GraphTransformerNetwork.
    """
    global WANDB_AVAILABLE

    # Initialize wandb if available
    run = None
    wandb_enabled = WANDB_AVAILABLE

    if wandb_enabled:
        try:
            run = wandb.init(project="graph-transformer-cvrp", config=config)
            config = wandb.config
            print("📊 W&B logging enabled")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            print("📊 Continuing without W&B logging")
            wandb_enabled = False

    if not wandb_enabled:
        print("📊 Training without W&B logging")
        # Convert config dict to object-like access
        class ConfigObj:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        config = ConfigObj(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if device.type == "cuda":
        print(f"✅ CUDA available! GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU.")

    # Load dataset
    train_data, test_data, _ = prepare_pretrain_data(
        data_file,
        split_ratios=[0.8, 0.2, 0.0],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Model
    model = GraphTransformerNetwork(
        in_channels=config.encoding_dim,
        hidden_channels=config.encoding_dim,
        out_channels=config.encoding_dim,
        heads=config.heads,
        beta=config.beta,
        dropout=config.dropout,
        normalization=config.normalization,
        num_gat_layers=config.num_gat_layers,
        activation=config.activation,
        decode_method=config.decode_method,
    ).to(device)

    # Optimizer & Loss
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config.initial_lr)
    criterion = get_loss_function(config.loss_function)

    if wandb_enabled and run is not None:
        wandb.watch(model, log="all", log_freq=1)

    best_val_mape = float("inf")

    for epoch in range(config.epochs):
        # ---- Training ----
        model.train()
        optimizer.train()
        total_loss, total_mape = 0, 0

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
        if wandb_enabled and run is not None:
            wandb.log(
                {"epoch": epoch, "train_loss": avg_train_loss, "train_mape": avg_train_mape}
            )

        # ---- Validation ----
        model.eval()
        optimizer.eval()
        with torch.no_grad():
            val_loss, val_mape = 0, 0
            for data in test_loader:
                data.to(device)
                out = model(data, mask=data.mask)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()

            avg_val_loss = val_loss / len(test_loader)
            avg_val_mape = val_mape / len(test_loader)
            if wandb_enabled and run is not None:
                wandb.log(
                    {"epoch": epoch, "val_loss": avg_val_loss, "val_mape": avg_val_mape}
                )

            # Save best model
            if avg_val_mape < best_val_mape:
                best_val_mape = avg_val_mape
                os.makedirs("model_state", exist_ok=True)
                # Use a sanitized file identifier (basename without extension)
                base_id = os.path.splitext(os.path.basename(data_file))[0]
                torch.save(
                    model.state_dict(),
                    f"model_state/tf_{base_id}_epoch{epoch}.pth",
                )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.2f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.2f}"
        )

    if wandb_enabled and run is not None:
        run.finish()


if __name__ == "__main__":
    # Minimal CLI to accept a user-supplied HDF5 training file
    parser = argparse.ArgumentParser(description="Train GraphTransformer on CVRP HDF5 data")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to HDF5 training data file",
    )

    # Provide a minimal default config so the script can run out-of-the-box
    default_config = {
        "encoding_dim": 64,
        "batch_size": 64,
        "dropout": 0.1,
        "initial_lr": 1e-3,
        "heads": 8,
        "normalization": "graph_norm",
        "activation": "elu",
        "num_gat_layers": 3,
        "loss_function": "smooth_l1",
        "beta": True,
        "decode_method": "pool",
        "epochs": 100,
    }

    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {args.data_file}")

    # Kick off training with defaults; override via wandb if desired
    train(default_config, args.data_file)
