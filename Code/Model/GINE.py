import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class GINELayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim, dropout):
        super(GINELayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.conv = GINEConv(self.mlp, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.dropout(x)
        return x


class GINE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_dim, num_layers, dropout):
        super(GINE, self).__init__()
        self.num_layers = num_layers
        
        # First layer
        self.layers = nn.ModuleList()
        self.layers.append(GINELayer(in_features, hidden_features, edge_dim, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GINELayer(hidden_features, hidden_features, edge_dim, dropout))
        
        # Output layer
        self.layers.append(GINELayer(hidden_features, out_features, edge_dim, dropout))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
            
        return x