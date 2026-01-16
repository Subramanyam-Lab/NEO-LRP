"""
Shared utilities for neural network training and inference.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm, LayerNorm, InstanceNorm


def get_normalization(norm_type, num_features):
    """Get normalization layer by type."""
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(num_features)
    if norm_type == 'layer_norm':
        return LayerNorm(num_features)
    if norm_type == 'instance_norm':
        return InstanceNorm(num_features)
    if norm_type == 'graph_norm':
        return GraphNorm(num_features)
    return nn.Identity()