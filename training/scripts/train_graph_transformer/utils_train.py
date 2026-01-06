import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, LayerNorm, InstanceNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, Set2Set
from torch_geometric.utils import scatter

def get_normalization(norm_type, num_features):
    if norm_type == 'batch_norm': return nn.BatchNorm1d(num_features)
    if norm_type == 'layer_norm': return LayerNorm(num_features)
    if norm_type == 'instance_norm': return InstanceNorm(num_features)
    if norm_type == 'graph_norm': return GraphNorm(num_features)
    return nn.Identity()

def get_readout(readout_type, x, batch):
    if readout_type == 'mean': return global_mean_pool(x, batch)
    if readout_type == 'max': return global_max_pool(x, batch)
    if readout_type == 'sum': return global_add_pool(x, batch)
    if readout_type == 'attention':
        node_weights = F.softmax(x, dim=1)
        return scatter(node_weights * x, batch, dim=0, reduce='sum')
    if readout_type == 'set2set':
        set2set = Set2Set(x.size(1), processing_steps=2).to(x.device)
        return set2set(x, batch)
    raise ValueError(f'Unknown readout type: {readout_type}')

def get_loss_function(loss_type):
    if loss_type == 'mse': return nn.MSELoss()
    if loss_type == 'mae': return nn.L1Loss()
    if loss_type == 'huber': return nn.HuberLoss()
    if loss_type == 'smooth_l1': return nn.SmoothL1Loss()
    raise ValueError(f'Unknown loss type: {loss_type}')
