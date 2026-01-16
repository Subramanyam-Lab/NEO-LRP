"""
DeepSets network wrapper for VRP cost prediction using ONNX models.
"""

import numpy as np
import torch
import onnx
from onnx2torch import convert
import onnxruntime as ort


class DeepSetsPredictor:
    """
    Wrapper for DeepSets ONNX model inference.
    """
    
    def __init__(self, phi_path, rho_path):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        self.phi_session = ort.InferenceSession(phi_path, sess_options)
        self.rho_session = ort.InferenceSession(rho_path, sess_options)
        
        self.phi_input_name = self.phi_session.get_inputs()[0].name
        self.rho_input_name = self.rho_session.get_inputs()[0].name
        
        print(f"[DS] Loaded phi model from {phi_path}")
        print(f"[DS] Loaded rho model from {rho_path}")
    
    def predict(self, facility_data):
        df = facility_data['df']
        features = df[['x', 'y', 'dem']].values[1:].astype(np.float32)
        
        phi_outputs = []
        for i in range(len(features)):
            phi_input = features[i:i+1]
            phi_out = self.phi_session.run(None, {self.phi_input_name: phi_input})[0]
            phi_outputs.append(phi_out)
        
        summed = np.sum(phi_outputs, axis=0)
        rho_out = self.rho_session.run(None, {self.rho_input_name: summed})[0]
        
        return float(rho_out[0, 0])


def load_ds_model(phi_path, rho_path):
    """Factory function to load DeepSets model."""
    return DeepSetsPredictor(phi_path, rho_path)


def extract_onnx(input_data, onnx_model_path):
    """
    Run phi network on input data and return embeddings as torch tensor.
    Used for Gurobi-ML integration.
    """
    onnx_model = onnx.load(onnx_model_path)
    pytorch_model = convert(onnx_model).double()

    input_tensor = torch.tensor(input_data, dtype=torch.float64)
    
    with torch.no_grad():
        output = pytorch_model(input_tensor)
    
    return output