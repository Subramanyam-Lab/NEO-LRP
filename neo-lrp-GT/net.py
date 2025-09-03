import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool
from utils_train import get_normalization

class GraphTransformerNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 beta, dropout, normalization, num_gat_layers,
                 activation, decode_method):
        super().__init__()

        self.activation = getattr(F, activation)
        self.decode_method = decode_method

        # Input transformation: [x, y, is_depot, demand]
        self.input_transform = nn.Linear(4, in_channels)

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

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

        # Readout for scalar prediction (e.g., route cost)
        self.output_layer = nn.Linear(out_channels, 1)

    def encode(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.view(-1, 1)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_transform(x)
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.norm_layers[i](x)
            x = self.activation(x)
        return x, batch

    def decode(self, node_embeddings, batch, mask):
        masked = node_embeddings * mask
        if self.decode_method == 'pool':
            h_G = global_add_pool(masked, batch)
            return self.output_layer(h_G)
        raise ValueError(f"Invalid decode_method={self.decode_method}")

    def forward(self, data, mask=None):
        node_embeddings, batch = self.encode(data)
        return self.decode(node_embeddings, batch, mask=mask)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_channels={self.input_transform.in_features}, "
                f"out_channels={self.output_layer.out_features}, "
                f"heads={self.transformer_layers[0].heads})")
