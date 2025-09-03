import os
import torch
import torch.nn as nn
import wandb
import schedulefree
from torch.utils.data import DataLoader
from torch_scatter import scatter_add

from utils import prepare_pretrain_data
from utils_train import get_loss_function


#########################################################
# Collate Function
#########################################################
def custom_collate_fn(batch):
    """
    Custom collate function to concatenate batched graph data.
    
    Args:
        batch (list): Each element contains:
            - x: Node features, shape (N_i, 4)
            - mask: Node masks, shape (N_i, 1)
            - y: Graph label, scalar or shape (1,)
    
    Returns:
        x_cat (Tensor): Concatenated node features
        mask_cat (Tensor): Concatenated node masks
        y_cat (Tensor): Concatenated labels
        graph_idx_cat (Tensor): Graph index for each node
    """
    x_list, mask_list, y_list, graph_index_list = [], [], [], []

    for i, data_i in enumerate(batch):
        x_i, mask_i, y_i = data_i.x, data_i.mask, data_i.y
        if y_i.dim() == 0:
            y_i = y_i.unsqueeze(0)

        x_list.append(x_i)
        mask_list.append(mask_i)
        y_list.append(y_i)

        num_nodes = x_i.size(0)
        graph_index_list.append(torch.full((num_nodes,), i, dtype=torch.long))

    x_cat = torch.cat(x_list, dim=0)
    mask_cat = torch.cat(mask_list, dim=0)
    y_cat = torch.cat(y_list, dim=0)
    graph_idx_cat = torch.cat(graph_index_list, dim=0)

    return x_cat, mask_cat, y_cat, graph_idx_cat


#########################################################
# Network (PhiNet, RhoNet)
#########################################################
class PhiNet(nn.Module):
    """Node-level encoder network."""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(PhiNet, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RhoNet(nn.Module):
    """Graph-level aggregator network."""
    def __init__(self, latent_dim, num_layers):
        super(RhoNet, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent_dim, 1))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, z_sum):
        return self.model(z_sum)


#########################################################
# Loss / Metrics
#########################################################
def mape(pred, target):
    """Mean Absolute Percentage Error (MAPE)."""
    return torch.mean(torch.abs((target - pred) / target)) * 100


#########################################################
# Config
#########################################################
config = {
    "batch_size": 32,
    "epochs": 100,
    "initial_lr": 1e-4,
    "latent_dim": 8,
    "loss_function": "huber",
    "num_phi_layers": 3,
    "phi_hidden_dim": 1024,
    "rho_num_layers": 1,
}


#########################################################
# Training Loop + ONNX Export
#########################################################
def train_single_config(cfg):
    run = wandb.init(
        project="phi-rho-training",
        config=cfg,
        name="phi-rho-single-run",
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_data, val_data, _ = prepare_pretrain_data(
        "data/cvrp_data/train_data.h5",
        split_ratios=[0.8, 0.2, 0.0],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    # Model
    phi_net = PhiNet(
        input_dim=4,
        hidden_dim=cfg["phi_hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_phi_layers"],
    ).to(device)

    rho_net = RhoNet(
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["rho_num_layers"],
    ).to(device)

    optimizer = schedulefree.AdamWScheduleFree(
        list(phi_net.parameters()) + list(rho_net.parameters()),
        lr=cfg["initial_lr"],
    )
    criterion = get_loss_function(cfg["loss_function"])

    wandb.watch(phi_net, log="all", log_freq=1)
    wandb.watch(rho_net, log="all", log_freq=1)

    best_val_loss = float("inf")
    best_phi_state, best_rho_state = None, None

    # Training loop
    for epoch in range(cfg["epochs"]):
        # ---- Training ----
        phi_net.train()
        rho_net.train()
        optimizer.train()

        total_loss, total_mape_val = 0.0, 0.0
        for x_cat, mask_cat, y_cat, graph_idx_cat in train_loader:
            x_cat, mask_cat, y_cat, graph_idx_cat = (
                x_cat.to(device),
                mask_cat.to(device),
                y_cat.to(device),
                graph_idx_cat.to(device),
            )

            out_phi = phi_net(x_cat)
            masked_out_phi = out_phi * mask_cat
            z_sum = scatter_add(masked_out_phi, graph_idx_cat, dim=0)

            out_rho = rho_net(z_sum)
            if y_cat.dim() == 1:
                y_cat = y_cat.unsqueeze(-1)

            loss = criterion(out_rho, y_cat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mape_val += mape(out_rho, y_cat).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape_val / len(train_loader)

        # ---- Validation ----
        phi_net.eval()
        rho_net.eval()
        val_loss, val_mape_val = 0.0, 0.0
        with torch.no_grad():
            for x_cat, mask_cat, y_cat, graph_idx_cat in val_loader:
                x_cat, mask_cat, y_cat, graph_idx_cat = (
                    x_cat.to(device),
                    mask_cat.to(device),
                    y_cat.to(device),
                    graph_idx_cat.to(device),
                )

                out_phi = phi_net(x_cat)
                masked_out_phi = out_phi * mask_cat
                z_sum = scatter_add(masked_out_phi, graph_idx_cat, dim=0)

                out_rho = rho_net(z_sum)
                if y_cat.dim() == 1:
                    y_cat = y_cat.unsqueeze(-1)

                loss = criterion(out_rho, y_cat)
                val_loss += loss.item()
                val_mape_val += mape(out_rho, y_cat).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape_val / len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mape": avg_train_mape,
            "val_loss": avg_val_loss,
            "val_mape": avg_val_mape,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_phi_state = phi_net.state_dict()
            best_rho_state = rho_net.state_dict()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.4f}"
        )

    # Save best model & export to ONNX
    phi_net.load_state_dict(best_phi_state)
    rho_net.load_state_dict(best_rho_state)
    phi_net.eval()
    rho_net.eval()

    torch.onnx.export(
        phi_net,
        torch.randn(1, 4, device=device),
        "phi_net.onnx",
        input_names=["x"],
        output_names=["z"],
        opset_version=11,
    )
    print("Exported phi_net to phi_net.onnx")

    torch.onnx.export(
        rho_net,
        torch.randn(1, cfg["latent_dim"], device=device),
        "rho_net.onnx",
        input_names=["z_sum"],
        output_names=["prediction"],
        opset_version=11,
    )
    print("Exported rho_net to rho_net.onnx")

    run.finish()


#########################################################
# Main
#########################################################
if __name__ == "__main__":
    train_single_config(config)
