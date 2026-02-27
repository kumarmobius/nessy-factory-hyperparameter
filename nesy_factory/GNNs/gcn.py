"""
Graph Convolutional Network (GCN) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Any

from .base import BaseGNN


class GCN(BaseGNN):
    """
    Graph Convolutional Network (GCN) model.
    
    Implementation based on Kipf & Welling (2017):
    "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GCN model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, hidden_dim, output_dim
                Optional keys: num_layers, dropout, normalize, add_self_loops, bias
        """
        super(GCN, self).__init__(config)
        
        # GCN-specific parameters
        self.normalize = config.get('normalize', True)
        self.add_self_loops = config.get('add_self_loops', True)
        self.bias = config.get('bias', True)
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        
        # Determine layer dimensions
        if self.num_layers == 1:
            # Single layer: input -> output
            layer_dims = [self.input_dim, self.output_dim]
        else:
            # Multi-layer: input -> hidden_dims -> output
            layer_dims = [self.input_dim] + self.hidden_dims[:self.num_layers-1] + [self.output_dim]
        
        # Build layers based on computed dimensions
        for i in range(self.num_layers):
            self.convs.append(
                GCNConv(
                    layer_dims[i],
                    layer_dims[i + 1],
                    normalize=self.normalize,
                    add_self_loops=self.add_self_loops,
                    bias=self.bias
                )
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the GCN model.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices tensor of shape [2, num_edges]
            **kwargs: Additional arguments (edge_weight, etc.)
            
        Returns:
            Output tensor of shape [num_nodes, output_dim]
        """
        edge_weight = kwargs.get('edge_weight', None)
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer (no activation or dropout)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index, edge_weight)
        
        return x
    
    def _get_gnn_layers(self):
        """Return the GNN layers for CaFo block creation."""
        return self.convs
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get node embeddings from the last hidden layer.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments
            
        Returns:
            Node embeddings tensor
        """
        edge_weight = kwargs.get('edge_weight', None)
        
        # Forward pass until the last layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        return x
    
    def _get_hidden_features(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from hidden layer for CaFO training.
        
        This method returns features from the last hidden layer (before the final output layer)
        which is appropriate for CaFO matrix multiplication.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments
            
        Returns:
            Hidden features tensor of shape [num_nodes, hidden_dim]
        """
        return self.get_embeddings(x, edge_index, **kwargs)
    
    def _get_gnn_layers(self):
        """
        Get the GNN layers for enhanced CaFO block training.
        
        Returns:
            List of GNN layers (GCNConv layers)
        """
        return list(self.convs)
    
    def enable_layer_wise_cafo(self, cafo_config: dict = None):
        """
        Enable layer-wise CaFO training for this GCN.
        
        This creates CaFO blocks for each GCN layer, allowing progressive
        layer-by-layer training similar to the original CaFO paper.
        
        Args:
            cafo_config: CaFO configuration dictionary
                - loss_fn: 'MSE' or 'CE' (default: 'MSE')
                - lambda: regularization strength (default: 0.001)
                - num_epochs: training epochs per block (default: 100)
                - step: learning rate for gradient descent (default: 0.01)
        
        Returns:
            LayerWiseCaFoGNN instance for this GCN
        """
        from .enhanced_cafo_blocks import create_layer_wise_cafo_gnn
        
        if cafo_config is None:
            cafo_config = {}
        
        # Create configuration for layer-wise CaFO
        layer_wise_config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'hidden_dims': self.hidden_dims,
            'gnn_type': 'gcn',  # GCN-specific
            'gnn_config': {
                'normalize': self.normalize,
                'add_self_loops': self.add_self_loops,
                'bias': self.bias
            },
            'cafo_loss_fn': cafo_config.get('loss_fn', 'MSE'),
            'cafo_lambda': cafo_config.get('lambda', 0.001),
            'cafo_num_epochs': cafo_config.get('num_epochs', 100),
            'cafo_step': cafo_config.get('step', 0.01),
            'cafo_num_batches': cafo_config.get('num_batches', 1),
            'device': str(self.device)
        }
        
        print(f"🔧 Creating layer-wise CaFO GNN with {self.num_layers} blocks")
        layer_wise_model = create_layer_wise_cafo_gnn(layer_wise_config)
        
        print(f"✅ Layer-wise CaFO enabled! Use .train_all_blocks() to train progressively")
        return layer_wise_model


class GCNWithSkipConnections(BaseGNN):
    """
    GCN with skip connections for deeper networks.
    
    Adds residual connections to help with training deeper GCN models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GCN with skip connections.
        
        Args:
            config: Configuration dictionary
        """
        super(GCNWithSkipConnections, self).__init__(config)
        
        # GCN-specific parameters
        self.normalize = config.get('normalize', True)
        self.add_self_loops = config.get('add_self_loops', True)
        self.bias = config.get('bias', True)
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Determine layer dimensions
        if self.num_layers == 1:
            layer_dims = [self.input_dim, self.output_dim]
        else:
            layer_dims = [self.input_dim] + self.hidden_dims[:self.num_layers-1] + [self.output_dim]
        
        # Build conv layers
        for i in range(self.num_layers):
            self.convs.append(
                GCNConv(
                    layer_dims[i],
                    layer_dims[i + 1],
                    normalize=self.normalize,
                    add_self_loops=self.add_self_loops,
                    bias=self.bias
                )
            )
            
            # Add batch normalization for all layers except the last
            if i < self.num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))
        
        # Projection layers for skip connections when dimensions don't match
        self.skip_connections = nn.ModuleList()
        for i in range(self.num_layers - 1):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1]
                
            if input_dim != output_dim:
                self.skip_connections.append(nn.Linear(input_dim, output_dim))
            else:
                self.skip_connections.append(nn.Identity())
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        for skip in self.skip_connections:
            if hasattr(skip, 'reset_parameters'):
                skip.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
        edge_weight = kwargs.get('edge_weight', None)
        
        for i, conv in enumerate(self.convs[:-1]):
            # Store input for skip connection
            identity = x
            
            # GCN layer
            x = conv(x, edge_index, edge_weight)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Skip connection
            identity = self.skip_connections[i](identity)
            x = x + identity
            
            # Activation and dropout
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer (no skip connection, batch norm, or dropout)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index, edge_weight)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get node embeddings from the last hidden layer (before final output).
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments
            
        Returns:
            Node embeddings tensor
        """
        edge_weight = kwargs.get('edge_weight', None)
        
        # Forward pass until the second-to-last layer
        for i, conv in enumerate(self.convs[:-1]):
            # Store input for skip connection
            identity = x
            
            # GCN layer
            x = conv(x, edge_index, edge_weight)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Skip connection
            identity = self.skip_connections[i](identity)
            x = x + identity
            
            # Activation and dropout
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        return x
    
    def _get_hidden_features(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from hidden layer for CaFO training.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments
            
        Returns:
            Hidden features tensor of shape [num_nodes, hidden_dim]
        """
        return self.get_embeddings(x, edge_index, **kwargs)


def create_gcn(input_dim: int, hidden_dim, output_dim: int, **kwargs) -> GCN:
    """
    Convenience function to create a GCN model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension(s). Can be:
                   - int: Single dimension for all hidden layers
                   - list/tuple: Different dimensions for each hidden layer
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GCN model
        
    Examples:
        # Single hidden dimension
        model = create_gcn(10, 64, 3)
        
        # Multiple hidden dimensions  
        model = create_gcn(10, [64, 128, 64], 3, num_layers=4)
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        **kwargs
    }
    return GCN(config)


def create_gcn_with_skip_connections(input_dim: int, hidden_dim, output_dim: int, **kwargs) -> GCNWithSkipConnections:
    """
    Convenience function to create a GCN model with skip connections.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension(s). Can be:
                   - int: Single dimension for all hidden layers
                   - list/tuple: Different dimensions for each hidden layer
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GCN model with skip connections
        
    Examples:
        # Single hidden dimension
        model = create_gcn_with_skip_connections(10, 64, 3)
        
        # Multiple hidden dimensions with skip connections
        model = create_gcn_with_skip_connections(10, [64, 128, 256, 128], 3, num_layers=5)
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        **kwargs
    }
    return GCNWithSkipConnections(config) 