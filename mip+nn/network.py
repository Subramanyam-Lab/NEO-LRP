from datetime import datetime
from dataparse import *
import math
import numpy as np
import torch
import onnx
from onnx2torch import convert

def extract_onnx(input_data,onnx_model_path):
    #onnx_model_path='F:\MLEV\model_phi.onnx'
    onnx_model = onnx.load(onnx_model_path)
    pytorch_model = convert(onnx_model).double()
    
    #input_tensor = torch.tensor(input_data.values,dtype=torch.float)
    input_tensor = torch.tensor(input_data,dtype=torch.float)
    input_tensor = input_tensor.double()
    output = pytorch_model.forward(input_tensor)
    return output