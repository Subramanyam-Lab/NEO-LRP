import wandb
import torch
import schedulefree
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import h5py
from torch_scatter import scatter_add
import os

from utils import prepare_pretrain_data
from utils_train import get_loss_function

# Collate function (core of vectorization)
def custom_collate_fn(batch):
    """
    batch: List format. Each element data_i has:
           data_i.x:  shape (N_i, 4)
           data_i.mask: shape (N_i, 1)
           data_i.y: scalar (0D) or shape (1,)

    Concatenates all graph nodes in this batch to generate
    x_cat, mask_cat, y_cat, graph_idx_cat.
    """
    x_list = []
    mask_list = []
    y_list = []
    graph_index_list = []  # Index indicating which graph each node belongs to

    for i, data_i in enumerate(batch):
        x_i = data_i.x
        mask_i = data_i.mask
        y_i = data_i.y

        # If y_i is 0D (scalar), make it (1,) for later concatenation
        if y_i.dim() == 0:
            y_i = y_i.unsqueeze(0)

        x_list.append(x_i)
        mask_list.append(mask_i)
        y_list.append(y_i)

        # Create i indices for the number of nodes per graph and append
        num_nodes = x_i.size(0)
        graph_index_list.append(torch.full((num_nodes,), i, dtype=torch.long))

    # Concatenate
    x_cat = torch.cat(x_list, dim=0)       # (total nodes, 4)
    mask_cat = torch.cat(mask_list, dim=0) # (total nodes, 1)
    y_cat = torch.cat(y_list, dim=0)       # (number of graphs in batch,)
    graph_idx_cat = torch.cat(graph_index_list, dim=0)  # (total nodes,)

    return x_cat, mask_cat, y_cat, graph_idx_cat

# Networks (PhiNet, RhoNet)
class PhiNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(PhiNet, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Last layer -> latent_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (total number of nodes, 4)
        Returns: (total number of nodes, latent_dim)
        """
        return self.model(x)

class RhoNet(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super(RhoNet, self).__init__()
        layers = []
        # First (num_layers - 1) layers: latent_dim -> latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        # Last layer: latent_dim -> 1
        layers.append(nn.Linear(latent_dim, 1))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, z_sum):
        """
        z_sum: (number of graphs in batch, latent_dim)
        Returns: (number of graphs in batch,)  # Prediction for each graph
        """
        out = self.model(z_sum)      # (batch, 1)
        return out

# Loss/Metric
def mape(pred, target):
    """
    MAPE (Mean Absolute Percentage Error) between predicted and actual values
    """
    return torch.mean(torch.abs((target - pred) / target)) * 100

# Config (single configuration)
config = {
    "batch_size": 32,
    "epochs": 100,
    "initial_lr": 0.0001,
    "latent_dim": 8,
    "loss_function": "huber",
    "num_phi_layers": 3,
    "phi_hidden_dim": 1024,
    "rho_num_layers": 1
}

#########################################################
# 4) Train function (vectorized) + ONNX Export
#########################################################
def train_single_config(cfg):
    # wandb setup (use your preferred project name)
    run = wandb.init(
        project='phi-rho-fixed-config',
        config=cfg,
        name='phi-rho-single-run'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    train_data, val_data, _ = prepare_pretrain_data(
        'data/phase1/cvrp_data/pretrain_phase1_OR_128k.h5',
        split_ratios=[0.8, 0.2, 0.0]
    )
    train_loader = DataLoader(train_data,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data,
                            batch_size=cfg["batch_size"],
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4,
                            collate_fn=custom_collate_fn)

    # Model preparation
    phi_net = PhiNet(
        input_dim=4,
        hidden_dim=cfg["phi_hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_phi_layers"]
    ).to(device)

    rho_net = RhoNet(
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["rho_num_layers"]
    ).to(device)

    # schedulefree AdamW
    optimizer = schedulefree.AdamWScheduleFree(
        list(phi_net.parameters()) + list(rho_net.parameters()),
        lr=cfg["initial_lr"]
    )

    # Loss function setup
    criterion = get_loss_function(cfg["loss_function"])

    # wandb watch (optional)
    wandb.watch(phi_net, log='all', log_freq=1)
    wandb.watch(rho_net, log='all', log_freq=1)

    best_val_loss = float('inf')
    best_phi_state = None
    best_rho_state = None

    # ------------------- Training Loop -------------------
    for epoch in range(cfg["epochs"]):
        # --------------------- train phase ---------------------
        phi_net.train()
        rho_net.train()
        optimizer.train()  # train mode for schedulefree

        total_loss = 0.0
        total_mape_val = 0.0

        for x_cat, mask_cat, y_cat, graph_idx_cat in train_loader:
            x_cat = x_cat.to(device)
            mask_cat = mask_cat.to(device)
            y_cat = y_cat.to(device)
            graph_idx_cat = graph_idx_cat.to(device)

            # (1) PhiNet
            out_phi = phi_net(x_cat)          # (total nodes, latent_dim)
            masked_out_phi = out_phi * mask_cat

            # (2) Sum per graph (scatter_add)
            z_sum = scatter_add(masked_out_phi, graph_idx_cat, dim=0)

            # (3) RhoNet
            out_rho = rho_net(z_sum)          # (number of graphs in batch,)
            if y_cat.dim() == 1:
                y_cat = y_cat.unsqueeze(-1)

            # (4) Loss calculation and backpropagation
            loss = criterion(out_rho, y_cat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # (5) MAPE
            batch_mape = mape(out_rho, y_cat).item()
            total_loss += loss.item()
            total_mape_val += batch_mape

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape_val / len(train_loader)

        # --------------------- validation phase ---------------------
        phi_net.eval()
        rho_net.eval()

        val_loss = 0.0
        val_mape_val = 0.0
        with torch.no_grad():
            for x_cat, mask_cat, y_cat, graph_idx_cat in val_loader:
                x_cat = x_cat.to(device)
                mask_cat = mask_cat.to(device)
                y_cat = y_cat.to(device)
                graph_idx_cat = graph_idx_cat.to(device)

                out_phi = phi_net(x_cat)
                masked_out_phi = out_phi * mask_cat
                z_sum = scatter_add(masked_out_phi, graph_idx_cat, dim=0)

                out_rho = rho_net(z_sum)
                if y_cat.dim() == 1:
                    y_cat = y_cat.unsqueeze(-1)

                loss = criterion(out_rho, y_cat)
                batch_mape = mape(out_rho, y_cat).item()

                val_loss += loss.item()
                val_mape_val += batch_mape

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape_val / len(val_loader)

        # wandb logging
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mape": avg_train_mape,
            "val_loss":   avg_val_loss,
            "val_mape":   avg_val_mape
        })

        # Best model update
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_phi_state = phi_net.state_dict()
            best_rho_state = rho_net.state_dict()

        print(
            f"Epoch {epoch}, "
            f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.4f}"
        )

    # ------------------- After training completion, load best model & ONNX Export -------------------
    phi_net.load_state_dict(best_phi_state)
    rho_net.load_state_dict(best_rho_state)
    phi_net.eval()
    rho_net.eval()

    # (A) phi_net.onnx Export
    dummy_phi_input = torch.randn(1, 4, device=device)  # (batch=1, input_dim=4)
    onnx_phi_path   = "phi_net.onnx"
    torch.onnx.export(
        phi_net,
        dummy_phi_input,
        onnx_phi_path,
        input_names=["x"],  
        output_names=["z"],  
        opset_version=11
    )
    print(f"Exported phi_net to {onnx_phi_path}")

    # (B) rho_net.onnx Export
    dummy_rho_input = torch.randn(1, cfg["latent_dim"], device=device)  # (batch=1, latent_dim=8)
    onnx_rho_path   = "rho_net.onnx"
    torch.onnx.export(
        rho_net,
        dummy_rho_input,
        onnx_rho_path,
        input_names=["z_sum"], 
        output_names=["prediction"],
        # opset_version=11
    )
    print(f"Exported rho_net to {onnx_rho_path}")

    run.finish()

############################################
# 5) main
############################################
if __name__ == "__main__":
    # Training with single config & ONNX export
    train_single_config(config)
