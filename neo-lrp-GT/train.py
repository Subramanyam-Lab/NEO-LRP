import os
import torch
import wandb
import torch.nn as nn
import schedulefree
from torch_geometric.loader import DataLoader

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

    # Initialize wandb
    run = wandb.init(project="graph-transformer-cvrp", config=config)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if device.type == "cuda":
        print(f"✅ CUDA available! GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU.")

    # Load dataset
    train_data, test_data, _ = prepare_pretrain_data(
        "data/cvrp_data/train_data.h5",
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
            wandb.log(
                {"epoch": epoch, "val_loss": avg_val_loss, "val_mape": avg_val_mape}
            )

            # Save best model
            if avg_val_mape < best_val_mape:
                best_val_mape = avg_val_mape
                os.makedirs("model_state", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"model_state/tf_{data_file}_epoch{epoch}.pth",
                )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.2f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.2f}"
        )

    run.finish()
