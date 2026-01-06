"""
Graph Transformer Pre training with W and B 
"""
import os
import torch
import schedulefree
from torch_geometric.loader import DataLoader
import argparse
import time

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
output_dir = os.getenv('output_dir', '.')
sweep_count = int(os.getenv('sweep_count'))
num_instances = os.getenv('num_instances')

if num_instances is not None:
    num_instances = int(num_instances)

tag = f"vroom_{mode}"

WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")

if not WANDB_DISABLED:
    import wandb
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY, relogin=False)
        except Exception:
            pass

print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Graph Transformer Pre training (HPO)")
print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Mode: {mode}")
print(f"Tag: {tag}")
print(f"Normalization: {normalization_mode}")
print(f"Num instances: {num_instances}")
print(f"Train and Val: {file_path_train_val}")
print(f"Test: {file_path_test}")
print(f"Norm JSON: {norm_json_path}")
print(f"H5 Cache: {h5_cache_dir}")
print(f"Output: {output_dir}")
print(f"Sweep count: {sweep_count}")
print(f"WANDB disabled: {WANDB_DISABLED}")
print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "encoding_dim":   {"values": [8, 16, 32, 64, 128]},
        "batch_size":     {"values": [8, 16, 32, 64, 128]},
        "dropout":        {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "initial_lr":     {"values": [0.1, 0.01, 0.001, 0.0001]},
        "heads":          {"values": [4, 8]},
        "normalization":  {"values": ["graph_norm", "batch_norm", "layer_norm"]},
        "activation":     {"values": ["elu", "relu", "leaky_relu"]},
        "num_gat_layers": {"values": [2, 3, 4, 5]},
        "loss_function":  {"values": ["mse", "huber", "smooth_l1"]},
        "beta":           {"values": [True, False]},
        "decode_method":  {"values": ["pool"]},
    },
}

SWEEP_ID = None
if not WANDB_DISABLED:
    SWEEP_ID = wandb.sweep(sweep_config, project=f"neo_lrp_gt_{mode}_{num_instances}")

def get_or_create_h5():
    """generate h5 file if it doesn't exist and return path"""
    os.makedirs(h5_cache_dir, exist_ok=True)
    h5_filename = f"exported_{tag}_{normalization_mode}.h5"
    h5_path = os.path.join(h5_cache_dir, h5_filename)
    
    if os.path.exists(h5_path):
        print(f"Using existing H5 file: {h5_path}")
        return h5_path
    
    print(f"Generating H5 file: {h5_path}")
    
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


def mape(pred, target, eps=1e-8):
    return torch.mean(torch.abs((target - pred) / (target.abs() + eps))) * 100


def _cfg_to_plain_dict(cfg):
    try:
        d = dict(cfg)
    except Exception:
        keys = [k for k in dir(cfg) if not k.startswith("_")]
        d = {k: getattr(cfg, k) for k in keys}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v and len(v) == 1:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def train(config=None):
    if not WANDB_DISABLED:
        run = wandb.init(config=config)
        cfg = wandb.config
    else:
        class _C(dict):
            __getattr__ = dict.get
        cfg = _C({
            'encoding_dim': 32,
            'batch_size': 32,
            'dropout': 0.4,
            'initial_lr': 0.0001,
            'heads': 4,
            'normalization': 'batch_norm',
            'activation': 'leaky_relu',
            'num_gat_layers': 3,
            'loss_function': 'huber',
            'beta': True,
            'decode_method': 'pool'
        })
        run = None

    data_h5_path = get_or_create_h5()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data, _ = prepare_pretrain_data(
        data_h5_path,
        split_ratios=[0.9, 0.1, 0.0],
        num_entries=num_instances
    )

    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size,
        shuffle=True, pin_memory=True, num_workers=1
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.batch_size,
        shuffle=False, pin_memory=True, num_workers=1
    )

    model = GraphTransformerNetwork(
        in_channels=cfg.encoding_dim,
        hidden_channels=cfg.encoding_dim,
        out_channels=cfg.encoding_dim,
        heads=cfg.heads,
        beta=cfg.beta,
        dropout=cfg.dropout,
        normalization=cfg.normalization,
        num_gat_layers=cfg.num_gat_layers,
        activation=cfg.activation,
        decode_method=cfg.decode_method,
    ).to(device)

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.initial_lr)
    criterion = get_loss_function(cfg.loss_function)

    if not WANDB_DISABLED:
        wandb.watch(model, log="all", log_freq=1)

    best_val_loss = float("inf")
    best_out_file = os.path.join(output_dir, f"best_{mode}_{normalization_mode}_{num_instances}.txt")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(150):
        model.train()
        optimizer.train()

        total_loss = 0.0
        total_mape = 0.0

        for data in train_loader:
            data.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(data, mask=getattr(data, "mask", None))
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mape += mape(out, data.y.view(-1, 1)).item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_train_mape = total_mape / max(1, len(train_loader))

        if not WANDB_DISABLED:
            wandb.log({"epoch": epoch, "train_mape": avg_train_mape, "train_loss": avg_train_loss})

        model.eval()
        optimizer.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_mape = 0.0
            for data in val_loader:
                data.to(device)
                out = model(data, mask=getattr(data, "mask", None))
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_mape = val_mape / max(1, len(val_loader))

            if not WANDB_DISABLED:
                wandb.log({"epoch": epoch, "val_mape": avg_val_mape, "val_loss": avg_val_loss})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                plain = _cfg_to_plain_dict(cfg)
                plain_sorted = {k: plain[k] for k in sorted(plain)}
                lines = []
                lines.append("~~~~~~  NEW BEST ~~~~~~\n")
                lines.append(f"val_loss: {float(best_val_loss):.6f}\n")
                if not WANDB_DISABLED:
                    run_url = f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
                    lines.append(f"wandb_run_id: {wandb.run.id}\n")
                    lines.append(f"wandb_url: {run_url}\n")
                lines.append("config:\n")
                for k, v in plain_sorted.items():
                    lines.append(f"  {k}: {v}\n")
                lines.append("\n")
                with open(best_out_file, "a") as f:
                    f.writelines(lines)
                print("".join(lines).strip())

        torch.cuda.empty_cache()
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {avg_train_loss:.6f} MAPE {avg_train_mape:.3f} | "
            f"Val Loss {avg_val_loss:.6f} MAPE {avg_val_mape:.3f}"
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    start_time = time.time()

    if WANDB_DISABLED:
        train()
    else:
        assert SWEEP_ID is not None, "Sweep was not created (check WANDB login/key)."
        wandb.agent(SWEEP_ID, train, count=sweep_count)

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")