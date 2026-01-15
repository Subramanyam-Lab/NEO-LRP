import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool
from utils_train import get_normalization, get_readout

class GraphTransformerNetwork(nn.Module):
    """
    Graph Transformer Network
    
    This network processes depot-customer subgraphs to predict routing costs
    using transformer-based graph neural networks.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        heads: Number of attention heads
        beta: Beta parameter for TransformerConv
        dropout: Dropout rate
        normalization: Normalization method ('layer_norm', 'batch_norm', etc.)
        num_gat_layers: Number of transformer layers
        activation: Activation function name
        decode_method: Decoding method for readout
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads, beta, dropout, normalization, num_gat_layers, activation, decode_method):
        super(GraphTransformerNetwork, self).__init__()
        
        self.activation = getattr(F, activation)
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.decode_method = decode_method

        # Input feature transformation: [x, y, is_depot, demand] -> in_channels
        self.input_transform = nn.Linear(4, in_channels)

        # Input layer
        self.transformer_layers.append(
            TransformerConv(in_channels, hidden_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Hidden layers
        for _ in range(num_gat_layers - 1):
            self.transformer_layers.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=1)
            )
            self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Output layer for final embeddings (phi)
        self.transformer_layers.append(
            TransformerConv(hidden_channels, out_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(get_normalization(normalization, out_channels))

        # rho network we are matching DS structure (Linear -> ReLU -> Linear -> ReLU)
        rho_hidden_dim = 8  # Same as neurons_per_layer_rho in DS
        self.rho = nn.Sequential(
            nn.Linear(out_channels, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, 1),
            nn.ReLU()
        )
        self.reset_parameters()
 
    def reset_parameters(self):
        """Initialize weights same as DeepSet architecture"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                            
    def encode(self, data):
        """
        phi network: Computes embeddings for each node through Transformer layers.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.reshape(-1, 1)  # Adjust edge_attr dimensions
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # Input feature transformation
        x = self.input_transform(x)
        # Update embeddings through each Transformer layer
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.norm_layers[i](x)
            x = self.activation(x)
        # x: Final embedding for each node (phi output)
        return x, batch

    def decode(self, node_embeddings, batch, mask):
        masked_embeddings = node_embeddings * mask  # (num_nodes, out_channels)

        if self.decode_method == 'pool':
            # 1) sum pooling
            h_G = global_add_pool(masked_embeddings, batch)  # => (batch_size, out_channels)
            # output = self.output_layer(h_G) + self.bias      # => (batch_size, 1)
            # output = self.output_layer(h_G)
            output = self.rho(h_G)
            return output

        else:
            raise ValueError(f"Invalid decode_method={self.decode_method}, must be 'pool' or 'mlp'.")

    def forward(self, data, mask=None):
        """
        Complete forward: First computes node embeddings in phi network (encode),
        then aggregates only selected nodes (according to mask) through rho network (readout) for final prediction.
        mask: [num_nodes, 1] tensor, 1 means selected.
        """
        node_embeddings, batch = self.encode(data)
        return self.decode(node_embeddings, batch, mask=mask)

    def __repr__(self):
            return (f"{self.__class__.__name__}(in_channels={self.input_transform.in_features}, "
                    f"rho={self.rho}, heads={self.transformer_layers[0].heads})")