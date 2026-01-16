"""
Graph Transformer network for VRP cost prediction.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from scipy.spatial import distance_matrix

from core.utils_train import get_normalization


class GraphTransformerNetwork(nn.Module):
    """
    Graph Transformer Network for VRP cost prediction.
    
    Args:
        in_channels: Input feature dimension (after input_transform)
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        heads: Number of attention heads
        beta: Beta parameter for TransformerConv
        dropout: Dropout rate
        normalization: Normalization type
        num_gat_layers: Number of transformer layers
        activation: Activation function name
        decode_method: Decoding method ('pool')
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads, beta,
                 dropout, normalization, num_gat_layers, activation, decode_method):
        super(GraphTransformerNetwork, self).__init__()
        
        self.activation = getattr(F, activation)
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.decode_method = decode_method

        # Input: [x, y, is_depot, demand] -> in_channels
        self.input_transform = nn.Linear(4, in_channels)

        # Input layer
        self.transformer_layers.append(
            TransformerConv(in_channels, hidden_channels // heads, heads=heads,
                          concat=True, beta=beta, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Hidden layers
        for _ in range(num_gat_layers - 1):
            self.transformer_layers.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                              concat=True, beta=beta, dropout=dropout, edge_dim=1)
            )
            self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Output layer
        self.transformer_layers.append(
            TransformerConv(hidden_channels, out_channels // heads, heads=heads,
                          concat=True, beta=beta, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(get_normalization(normalization, out_channels))

        # Rho network (matches DS structure)
        rho_hidden_dim = 8
        self.rho = nn.Sequential(
            nn.Linear(out_channels, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, 1),
            nn.ReLU()
        )

    def encode(self, data):
        """Phi network: compute node embeddings."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.reshape(-1, 1)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.input_transform(x)
        
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.norm_layers[i](x)
            x = self.activation(x)
        
        return x, batch

    def decode(self, node_embeddings, batch, mask):
        """Rho network: aggregate and predict."""
        masked_embeddings = node_embeddings * mask
        
        if self.decode_method == 'pool':
            h_G = global_add_pool(masked_embeddings, batch)
            output = self.rho(h_G)
            return output
        else:
            raise ValueError(f"Invalid decode_method={self.decode_method}")

    def forward(self, data, mask=None):
        """Complete forward pass."""
        node_embeddings, batch = self.encode(data)
        return self.decode(node_embeddings, batch, mask=mask)


class GraphTransformerPredictor:
    """
    Wrapper for GT inference - provides same interface as DS predictor.
    """
    
    # DEFAULT_CONFIG = {
    #     'encoding_dim': 64,
    #     'dropout': 0.3,
    #     'heads': 4,
    #     'normalization': 'layer_norm',
    #     'activation': 'elu',
    #     'num_gat_layers': 5,
    #     'beta': True,
    #     'decode_method': 'pool'
    # }

    DEFAULT_CONFIG = {
        'encoding_dim': 16,
        'batch_size': 8,
        'dropout': 0.3,
        'initial_lr': 0.001,
        'heads': 4,
        'normalization': 'graph_norm',
        'activation': 'elu',
        'num_gat_layers': 5,
        'loss_function': 'huber',
        'beta': False,
        'decode_method': 'pool'
    }
    
    def __init__(self, model_path, config=None):
        """
        Args:
            model_path: Path to .pth model file
            config: Model config dict (uses DEFAULT_CONFIG if not provided)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GT model not found: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Build model
        self.model = GraphTransformerNetwork(
            in_channels=self.config['encoding_dim'],
            hidden_channels=self.config['encoding_dim'],
            out_channels=self.config['encoding_dim'],
            heads=self.config['heads'],
            beta=self.config['beta'],
            dropout=self.config['dropout'],
            normalization=self.config['normalization'],
            num_gat_layers=self.config['num_gat_layers'],
            activation=self.config['activation'],
            decode_method=self.config['decode_method']
        ).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"[GT] Loaded model from {model_path}")
        print(f"[GT] Config: {self.config}")
    
    def prepare_graph_data(self, facility_data):
        """
        Prepare PyG Data from facility_dict entry.
        
        EXPECTS data from norm_data_gt() where:
        - Row 0 = depot (coordinates 0,0, demand 0)
        - Rows 1..N = customers
        - facility_data has 'df' and 'dist' keys
        
        Args:
            facility_data: Dict with 'df' (DataFrame) and 'dist' (distance matrix)
        
        Returns:
            PyG Data object
        """
        # Must have 'df' and 'dist' from norm_data_gt()
        if 'df' not in facility_data or 'dist' not in facility_data:
            raise ValueError(
                "GT requires facility_data with 'df' and 'dist' keys from norm_data_gt(). "
                "Got keys: " + str(list(facility_data.keys()))
            )
        
        df = facility_data['df']
        dist_mtx = facility_data['dist']
        
        n = len(df)  # Should be N+1 (depot + customers)
        
        x_coords = df['x'].values.astype(np.float32)
        y_coords = df['y'].values.astype(np.float32)
        demands = df['dem'].values.astype(np.float32)
        
        # Verify depot is at row 0
        if not (x_coords[0] == 0.0 and y_coords[0] == 0.0 and demands[0] == 0.0):
            raise ValueError(
                f"GT expects depot at row 0 with (0,0,0), but got "
                f"({x_coords[0]}, {y_coords[0]}, {demands[0]}). "
                "Make sure to use norm_data_gt() for Graph Transformer!"
            )
        
        # Node features: [x, y, is_depot, demand]
        is_depot = np.zeros(n, dtype=np.float32)
        is_depot[0] = 1.0  # Row 0 is depot
        
        features = np.column_stack([x_coords, y_coords, is_depot, demands])
        
        # Convert distance matrix to edge_index and edge_attr
        dist_tensor = torch.tensor(dist_mtx, dtype=torch.float32)
        edge_index, edge_attr = dense_to_sparse(dist_tensor)
        
        # Mask (all 1s for full graph)
        mask = torch.ones(n, 1, dtype=torch.float32)
        
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr.to(torch.float32),
            mask=mask
        )
        
        return data
    
    def predict(self, facility_data):
        """
        Predict VRP cost for a depot-customer subproblem.
        
        Args:
            facility_data: Dict with normalized data (from norm_data())
        
        Returns:
            float: Predicted normalized cost
        """
        data = self.prepare_graph_data(facility_data)
        data = data.to(self.device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            output = self.model(data, mask=data.mask)
        
        return output.item()
    
    def get_embeddings(self, facility_data):
        """
        Get node embeddings.
        
        Args:
            facility_data: Dict with normalized data
        
        Returns:
            np.ndarray: Node embeddings [num_nodes, encoding_dim]
        """
        data = self.prepare_graph_data(facility_data)
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self.model.encode(data)
        
        return embeddings.cpu().numpy()


def load_gt_model(model_path, config=None):
    """function to load GT model."""
    return GraphTransformerPredictor(model_path, config)