"""
Graph Attention Network (GAT) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, Any

from .base import BaseGNN


class GAT(BaseGNN):
    """
    Graph Attention Network (GAT) model.
    
    Implementation based on Veličković et al. (2018):
    "Graph Attention Networks"
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GAT model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, hidden_dim, output_dim
                Optional keys: num_layers, dropout, heads, concat, add_self_loops, bias
        """
        super(GAT, self).__init__(config)
        
        # GAT-specific parameters
        self.heads = config.get('heads', 8)  # Number of attention heads
        self.concat = config.get('concat', True)  # Whether to concatenate or average heads
        self.add_self_loops = config.get('add_self_loops', True)
        self.bias = config.get('bias', True)
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        
        # Determine layer dimensions
        if self.num_layers == 1:
            # Single layer: input -> output
            layer_dims = [self.input_dim, self.output_dim]
            layer_heads = [self.heads]
        else:
            # Multi-layer: input -> hidden_dims -> output
            layer_dims = [self.input_dim] + self.hidden_dims[:self.num_layers-1] + [self.output_dim]
            # For multi-head attention, all intermediate layers use multiple heads, final layer uses 1
            layer_heads = [self.heads] * (self.num_layers - 1) + [1]
        
        # Build layers based on computed dimensions
        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            heads = layer_heads[i]
            
            # For intermediate layers, use concat=True, for final layer use concat=False
            concat = self.concat if i < self.num_layers - 1 else False
            
            # Adjust input dimension for subsequent layers when previous layer used concat
            if i > 0 and layer_heads[i-1] > 1 and self.concat:
                in_dim = layer_dims[i] * layer_heads[i-1]
            
            conv = GATConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                concat=concat,
                dropout=self.dropout if self.training else 0.0,
                add_self_loops=self.add_self_loops,
                bias=self.bias
            )
            self.convs.append(conv)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """Forward pass through GAT layers."""
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation and dropout to all layers except the last
            if i < len(self.convs) - 1:
                x = F.elu(x)  # GAT paper uses ELU activation
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=1)

    def get_layer_info(self) -> str:
        """Get information about GAT layers."""
        layer_info = []
        for i, conv in enumerate(self.convs):
            heads = conv.heads
            concat = conv.concat
            in_dim = conv.in_channels
            out_dim = conv.out_channels
            
            if concat and i < len(self.convs) - 1:
                effective_out = out_dim * heads
                layer_info.append(f"GAT({in_dim}→{out_dim}×{heads}={effective_out})")
            else:
                layer_info.append(f"GAT({in_dim}→{out_dim}, heads={heads})")
        
        return " → ".join(layer_info)


class GATWithSkipConnections(GAT):
    """GAT model with skip connections between layers."""
    
    def forward(self, data):
        """Forward pass with skip connections."""
        x, edge_index = data.x, data.edge_index
        
        # Store input for potential skip connection
        x_input = x
        
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            
            # Apply activation and dropout to all layers except the last
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Add skip connection if dimensions match
                if x_prev.size(-1) == x.size(-1):
                    x = x + x_prev
        
        return F.log_softmax(x, dim=1)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gat(input_dim: int, hidden_dim, output_dim: int, **kwargs) -> GAT:
    """
    Convenience function to create a GAT model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension(s) - int or list
        output_dim: Output dimension (number of classes)
        **kwargs: Additional configuration parameters
    
    Returns:
        GAT model instance
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': kwargs.get('num_layers', 2),
        'dropout': kwargs.get('dropout', 0.6),
        'heads': kwargs.get('heads', 8),
        'concat': kwargs.get('concat', True),
        'add_self_loops': kwargs.get('add_self_loops', True),
        'bias': kwargs.get('bias', True),
        'optimizer': kwargs.get('optimizer', 'adam'),
        'learning_rate': kwargs.get('learning_rate', 0.005),  # GAT typically uses lower LR
        'weight_decay': kwargs.get('weight_decay', 5e-4)
    }
    
    return GAT(config)


def create_gat_with_skip_connections(input_dim: int, hidden_dim, output_dim: int, **kwargs) -> GATWithSkipConnections:
    """
    Convenience function to create a GAT model with skip connections.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension(s) - int or list
        output_dim: Output dimension (number of classes)
        **kwargs: Additional configuration parameters
    
    Returns:
        GATWithSkipConnections model instance
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': kwargs.get('num_layers', 3),
        'dropout': kwargs.get('dropout', 0.6),
        'heads': kwargs.get('heads', 8),
        'concat': kwargs.get('concat', True),
        'add_self_loops': kwargs.get('add_self_loops', True),
        'bias': kwargs.get('bias', True),
        'optimizer': kwargs.get('optimizer', 'adam'),
        'learning_rate': kwargs.get('learning_rate', 0.005),
        'weight_decay': kwargs.get('weight_decay', 5e-4)
    }
    
    return GATWithSkipConnections(config) 