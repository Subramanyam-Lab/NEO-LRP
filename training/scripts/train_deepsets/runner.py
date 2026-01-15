"""
DeepHyper HPO trial runner for each architecture.
Executes individual hyperparameter configurations and returns validation loss
for Bayesian optimization during the search process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import time

from architecture import DeepSetArchitecture
from train import master

import uuid

def run(config):

    unique_id = uuid.uuid4().hex
    
    checkpoint_suffix = f"trial_{unique_id}"

    metrics= master(config, metrics=True, exportonnx=False, testing=True, seed=42, N_dim=3, checkpoint_suffix=checkpoint_suffix)
    
    val_loss = metrics.get('best_val_loss')
    
    return -val_loss