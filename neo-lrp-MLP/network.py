"""
Neural Network Utilities for MLP-based Cost Prediction

This module provides utilities for loading and running ONNX models for routing cost prediction.
"""

from datetime import datetime
from dataparse import *
import math
import numpy as np
import torch
import onnx
from onnx2torch import convert

def extract_onnx(input_data, onnx_model_path):
    """
    Extract predictions from an ONNX model.
    
    Args:
        input_data (numpy.ndarray): Input data for the model
        onnx_model_path (str): Path to the ONNX model file
        
    Returns:
        torch.Tensor: Model output predictions
    """
    onnx_model = onnx.load(onnx_model_path)
    pytorch_model = convert(onnx_model).double()
    
    input_tensor = torch.tensor(input_data, dtype=torch.float)
    input_tensor = input_tensor.double()
    output = pytorch_model.forward(input_tensor)
    return output