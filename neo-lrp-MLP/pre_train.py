import wandb
import torch
import schedulefree
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import h5py
from utils import prepare_pretrain_data
from utils_train import get_loss_function

# import os
# os.environ["WANDB_MODE"] = "disabled"

def custom_collate_fn(batch):
    # Return as list format
    return batch

#########################################################
# Networks (PhiNet, RhoNet)
#########################################################
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
        x: (N, 4)
        Returns: (N, latent_dim)
        """
        return self.model(x)  # (N, latent_dim)

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
        z_sum: (latent_dim,)
        Returns: shape (1,)  # Prediction for one graph
        """
        z_sum_2d = z_sum.unsqueeze(0)  # (1, latent_dim)
        out = self.model(z_sum_2d)     # (1, 1)
        return out.squeeze(0)          # Return as (1,)

#########################################################
# 3) Loss/Metric
#########################################################
def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target)) * 100

#########################################################
# 4) Sweep config
#########################################################
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'phi_hidden_dim': {'values': [6,8,32,64,128,256,512,1024]},
        'latent_dim':     {'values': [6,8,16,32]},
        'num_phi_layers': {'values': [2,3,4,5,6]},
        'rho_num_layers': {'values': [1,2,3,4]},      # Added
        'batch_size':     {'values': [8,16,32]},
        'initial_lr':     {'values': [0.1, 0.01, 0.001, 0.0001]},
        'loss_function':  {'values': ['mse','smooth_l1','huber']},
        'epochs':         {'values': [200]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="phi-rho-bayes")

#########################################################
# Train function
#########################################################
def train(config=None):
    run = wandb.init(project='phi-rho-bayes', config=config)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    train_data, val_data, _ = prepare_pretrain_data(
        'data/pretrain_10k.h5',
        split_ratios=[0.8, 0.2, 0.0]
    )
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4,
                            collate_fn=custom_collate_fn)

    # Model preparation
    phi_net = PhiNet(
        input_dim=4,
        hidden_dim=config.phi_hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_phi_layers
    ).to(device)

    # Modified RhoNet to accept num_layers as parameter
    rho_net = RhoNet(
        latent_dim=config.latent_dim,
        num_layers=config.rho_num_layers
    ).to(device)

    optimizer = schedulefree.AdamWScheduleFree(
        list(phi_net.parameters()) + list(rho_net.parameters()),
        lr=config.initial_lr
    )

    criterion = get_loss_function(config.loss_function)

    wandb.watch(phi_net, log='all', log_freq=1)
    wandb.watch(rho_net, log='all', log_freq=1)

    best_val_loss = float('inf')
    max_epochs = config.epochs

    for epoch in range(max_epochs):
        # ---- train phase ----
        phi_net.train()
        rho_net.train()
        optimizer.train()

        total_loss = 0.0
        total_mape_val = 0.0

        for batch in train_loader:
            bsz = len(batch)   # Number of graphs in this batch
            preds = []
            labels_ = []

            for i in range(bsz):
                data_i = batch[i]  # Data(x=[N,4], y=scalar, mask=[N,1])
                x_i = data_i.x.to(device)       # (N, 4)
                mask_i = data_i.mask.to(device) # (N, 1)
                label_i = data_i.y.to(device)   # scalar(0D) or 1D

                # phi_net
                out_phi = phi_net(x_i)                # (N, latent_dim)
                masked_out_phi = out_phi * mask_i     # (N, latent_dim)
                z_sum = masked_out_phi.sum(dim=0)     # (latent_dim,)

                # rho_net
                out_rho = rho_net(z_sum)              # shape (1,)

                preds.append(out_rho)                 # (1,) for each graph
                # If label_i is 0D (scalar), make it 1D for stack/cat
                if label_i.dim() == 0:
                    label_i = label_i.unsqueeze(0)    # (1,)

                labels_.append(label_i)               # shape (1,)

            # Make predictions/labels for this mini-batch as (B,)
            preds_cat = torch.cat(preds, dim=0)       # (B,)
            labels_cat = torch.cat(labels_, dim=0)    # (B,)

            loss = criterion(preds_cat, labels_cat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate MAPE
            mape_val = mape(preds_cat.unsqueeze(-1), labels_cat.unsqueeze(-1)).item()

            total_loss     += loss.item()
            total_mape_val += mape_val

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape_val / len(train_loader)

        # ---- validation phase ----
        val_loss = 0.0
        val_mape_val = 0.0
        phi_net.eval()
        rho_net.eval()

        with torch.no_grad():
            for batch in val_loader:
                bsz = len(batch)
                preds = []
                labels_ = []

                for i in range(bsz):
                    data_i = batch[i]
                    x_i = data_i.x.to(device)
                    mask_i = data_i.mask.to(device)
                    label_i = data_i.y.to(device)

                    out_phi = phi_net(x_i)              # (N, latent_dim)
                    masked_out_phi = out_phi * mask_i   # (N, latent_dim)
                    z_sum = masked_out_phi.sum(dim=0)   # (latent_dim,)

                    out_rho = rho_net(z_sum)            # (1,)

                    preds.append(out_rho)
                    if label_i.dim() == 0:
                        label_i = label_i.unsqueeze(0)
                    labels_.append(label_i)

                preds_cat  = torch.cat(preds, dim=0)      # (B,)
                labels_cat = torch.cat(labels_, dim=0)    # (B,)

                loss = criterion(preds_cat, labels_cat)
                mape_val = mape(preds_cat.unsqueeze(-1), labels_cat.unsqueeze(-1)).item()

                val_loss     += loss.item()
                val_mape_val += mape_val

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape_val / len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mape": avg_train_mape,
            "val_loss":   avg_val_loss,
            "val_mape":   avg_val_mape
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(
            f"Epoch {epoch}, "
            f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.4f}"
        )

    run.finish()

#######################################
# maim
#######################################
if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=100)