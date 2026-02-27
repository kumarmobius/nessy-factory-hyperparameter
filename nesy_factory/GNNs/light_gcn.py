"""
LightGCN implementation for general graph learning tasks (adapted for general graph learning beyond recommendation systems - which uses learned embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseGNN


class LightGCN(BaseGNN):
    """
        
    LightGCN simplifies GCN by removing feature transformation and nonlinear activation,
    keeping only the neighborhood aggregation component. Originally designed for 
    collaborative filtering, this implementation generalizes it for any graph learning task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGCN model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, output_dim
                Optional keys: num_layers, dropout, layer_combination, normalization_type
        """
        super(LightGCN, self).__init__(config)
        
        # LightGCN-specific parameters
        self.layer_combination = config.get('layer_combination', 'mean')  # 'mean', 'sum', 'concat', 'weighted'
        self.normalization_type = config.get('normalization_type', 'symmetric')  # 'symmetric', 'left', 'right', 'none'
        self.add_self_loops = config.get('add_self_loops', True)
        
        # Initialize node embeddings if input_dim matches embedding usage
        # For general graphs, we use a linear transformation from input features
        if config.get('use_embeddings', False):
            # Use learnable embeddings (like original LightGCN for recommendation)
            num_nodes = config.get('num_nodes', 1000)  # Must be provided if using embeddings
            self.node_embeddings = nn.Embedding(num_nodes, self.input_dim)
            nn.init.xavier_uniform_(self.node_embeddings.weight)
            self.use_embeddings = True
        else:
            # Use input features directly with optional transformation
            self.use_embeddings = False
            self.input_transform = nn.Linear(self.input_dim, self.input_dim) if config.get('transform_input', False) else nn.Identity()
        
        # Output projection layer - adjust for concat strategy
        if self.layer_combination == 'concat':
            final_dim = self.input_dim * (self.num_layers + 1)
        else:
            final_dim = self.input_dim
            
        if final_dim != self.output_dim:
            self.output_projection = nn.Linear(final_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()
        
        # Layer combination weights (for weighted combination)
        if self.layer_combination == 'weighted':
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers + 1))
        
        # Cached normalized adjacency matrix
        self._cached_adj = None
        self._cached_edge_index = None
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        if self.use_embeddings:
            nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        if hasattr(self, 'input_transform') and hasattr(self.input_transform, 'reset_parameters'):
            self.input_transform.reset_parameters()
        
        if hasattr(self.output_projection, 'reset_parameters'):
            self.output_projection.reset_parameters()
        
        if hasattr(self, 'layer_weights'):
            nn.init.ones_(self.layer_weights)
    
    def _normalize_adjacency(self, edge_index: torch.Tensor, num_nodes: int, 
                           edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create normalized adjacency matrix for LightGCN.
        
        Args:
            edge_index: Edge indices tensor of shape [2, num_edges]
            num_nodes: Number of nodes in the graph
            edge_weight: Optional edge weights
            
        Returns:
            Normalized sparse adjacency matrix
        """
        # Add self-loops if specified
        if self.add_self_loops:
            edge_index, edge_weight = self._add_self_loops(edge_index, edge_weight, num_nodes)
        
        # Create sparse adjacency matrix
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=self.device
        ).coalesce()
        
        # Apply normalization
        if self.normalization_type == 'symmetric':
            # D^(-1/2) * A * D^(-1/2)
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            d_inv_sqrt = torch.pow(row_sum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            
            indices = adj.indices()
            values = adj.values()
            normalized_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]
            
        elif self.normalization_type == 'left':
            # D^(-1) * A (row normalization)
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            d_inv = torch.pow(row_sum, -1.0)
            d_inv[torch.isinf(d_inv)] = 0.
            
            indices = adj.indices()
            values = adj.values()
            normalized_values = d_inv[indices[0]] * values
            
        elif self.normalization_type == 'right':
            # A * D^(-1) (column normalization)
            col_sum = torch.sparse.sum(adj, dim=0).to_dense()
            d_inv = torch.pow(col_sum, -1.0)
            d_inv[torch.isinf(d_inv)] = 0.
            
            indices = adj.indices()
            values = adj.values()
            normalized_values = values * d_inv[indices[1]]
            
        else:  # no normalization
            indices = adj.indices()
            normalized_values = adj.values()
        
        normalized_adj = torch.sparse_coo_tensor(
            indices, normalized_values, adj.size(), device=self.device
        ).coalesce()
        
        return normalized_adj
    
    def _add_self_loops(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], 
                       num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add self-loops to the graph."""
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        
        if edge_weight is not None:
            loop_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        else:
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        
        return edge_index, edge_weight
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of LightGCN.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim] 
               (ignored if use_embeddings=True)
            edge_index: Edge indices tensor of shape [2, num_edges]
            **kwargs: Additional arguments including edge_weight
            
        Returns:
            Output tensor of shape [num_nodes, output_dim]
        """
        edge_weight = kwargs.get('edge_weight', None)
        num_nodes = x.shape[0]
        
        # Get initial node representations
        if self.use_embeddings:
            # Use learnable embeddings (original LightGCN style)
            node_ids = torch.arange(num_nodes, device=self.device)
            embeddings = self.node_embeddings(node_ids)
        else:
            # Use input features with optional transformation
            embeddings = self.input_transform(x)
        
        # Cache normalized adjacency matrix if edge_index hasn't changed
        if (self._cached_adj is None or 
            not torch.equal(self._cached_edge_index, edge_index)):
            self._cached_adj = self._normalize_adjacency(edge_index, num_nodes, edge_weight)
            self._cached_edge_index = edge_index.clone()
        
        # LightGCN layers: simple neighborhood aggregation
        layer_embeddings = [embeddings]  # Layer 0 (initial embeddings)
        
        current_emb = embeddings
        for layer in range(self.num_layers):
            # Simple message passing: aggregate neighbors
            current_emb = torch.sparse.mm(self._cached_adj, current_emb)
            layer_embeddings.append(current_emb)
        
        # Combine embeddings from all layers
        final_emb = self._combine_layers(layer_embeddings)
        
        # Apply dropout
        final_emb = self.dropout_layer(final_emb)
        
        # Output projection
        output = self.output_projection(final_emb)
        
        return output
    
    def _combine_layers(self, layer_embeddings):
        """Combine embeddings from different layers."""
        if self.layer_combination == 'mean':
            # Simple mean (original LightGCN)
            stacked = torch.stack(layer_embeddings, dim=1)  # [num_nodes, num_layers+1, embedding_dim]
            return torch.mean(stacked, dim=1)
            
        elif self.layer_combination == 'sum':
            # Sum all layers
            return sum(layer_embeddings)
            
        elif self.layer_combination == 'concat':
            # Concatenate all layers
            return torch.cat(layer_embeddings, dim=1)
            
        elif self.layer_combination == 'weighted':
            # Weighted combination
            weights = F.softmax(self.layer_weights, dim=0)
            weighted_embs = [w * emb for w, emb in zip(weights, layer_embeddings)]
            return sum(weighted_embs)
            
        else:
            # Default to mean
            stacked = torch.stack(layer_embeddings, dim=1)
            return torch.mean(stacked, dim=1)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      layer: int = -1, **kwargs) -> torch.Tensor:
        """
        Get embeddings from a specific layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            layer: Layer index (-1 for final combined embeddings)
            **kwargs: Additional arguments
            
        Returns:
            Node embeddings from the specified layer
        """
        if layer == -1:
            # Return final combined embeddings
            return self.forward(x, edge_index, **kwargs)
        
        edge_weight = kwargs.get('edge_weight', None)
        num_nodes = x.shape[0]
        
        # Get initial representations
        if self.use_embeddings:
            node_ids = torch.arange(num_nodes, device=self.device)
            embeddings = self.node_embeddings(node_ids)
        else:
            embeddings = self.input_transform(x)
        
        if layer == 0:
            return embeddings
        
        # Compute up to the requested layer
        if (self._cached_adj is None or 
            not torch.equal(self._cached_edge_index, edge_index)):
            self._cached_adj = self._normalize_adjacency(edge_index, num_nodes, edge_weight)
            self._cached_edge_index = edge_index.clone()
        
        current_emb = embeddings
        for l in range(min(layer, self.num_layers)):
            current_emb = torch.sparse.mm(self._cached_adj, current_emb)
        
        return current_emb
    
    def compute_similarity(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          node_pairs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute similarity scores between node pairs.
        
        Args:
            x: Node features
            edge_index: Edge indices  
            node_pairs: Tensor of shape [num_pairs, 2] with node pairs
            **kwargs: Additional arguments
            
        Returns:
            Similarity scores for each node pair
        """
        embeddings = self.get_embeddings(x, edge_index, **kwargs)
        
        # Get embeddings for each node in pairs
        node1_emb = embeddings[node_pairs[:, 0]]
        node2_emb = embeddings[node_pairs[:, 1]]
        
        # Compute inner product (like in LightGCN for recommendations)
        similarities = torch.sum(node1_emb * node2_emb, dim=1)
        
        return similarities


class LightGCNWithFeatures(LightGCN):
    """
    LightGCN variant that incorporates input features more explicitly.
    
    This version adds a feature transformation layer and optionally combines
    initial features with the final embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGCN with enhanced feature handling."""
        super(LightGCNWithFeatures, self).__init__(config)
        
        # Feature combination parameters
        self.combine_features = config.get('combine_features', True)
        self.feature_weight = config.get('feature_weight', 0.1)
        
        # Feature transformation layers
        if config.get('feature_transform_layers', 1) > 1:
            layers = []
            current_dim = self.input_dim
            
            for i in range(config['feature_transform_layers'] - 1):
                hidden_dim = config.get('feature_hidden_dim', self.input_dim)
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, self.input_dim))
            self.feature_transform = nn.Sequential(*layers)
        else:
            self.feature_transform = nn.Linear(self.input_dim, self.input_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with enhanced feature handling."""
        # Transform input features
        transformed_features = self.feature_transform(x)
        
        # Get LightGCN embeddings using transformed features
        lightgcn_output = super().forward(transformed_features, edge_index, **kwargs)
        
        # Optionally combine with original features
        if self.combine_features:
            # Weighted combination of features and embeddings
            if self.input_dim == self.output_dim:
                combined = (1 - self.feature_weight) * lightgcn_output + self.feature_weight * transformed_features
                return combined
            else:
                # If dimensions don't match, just return LightGCN output
                return lightgcn_output
        
        return lightgcn_output


def create_lightgcn(input_dim: int, output_dim: int, **kwargs) -> LightGCN:
    """
    Convenience function to create a LightGCN model.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
            - num_layers: Number of LightGCN layers (default: 3)
            - layer_combination: How to combine layers ('mean', 'sum', 'concat', 'weighted')
            - normalization_type: Adjacency normalization ('symmetric', 'left', 'right', 'none')
            - use_embeddings: Whether to use learnable embeddings instead of features
            - num_nodes: Required if use_embeddings=True
            
    Returns:
        Configured LightGCN model
        
    Examples:
        # Basic LightGCN
        model = create_lightgcn(64, 10, num_layers=3)
        
        # LightGCN with embeddings (like original for recommendations)
        model = create_lightgcn(64, 10, use_embeddings=True, num_nodes=1000)
        
        # LightGCN with weighted layer combination
        model = create_lightgcn(64, 10, layer_combination='weighted')
    """
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_layers': kwargs.get('num_layers', 3),
        **kwargs
    }
    return LightGCN(config)


def create_lightgcn_with_features(input_dim: int, output_dim: int, **kwargs) -> LightGCNWithFeatures:
    """
    Convenience function to create a LightGCN model with enhanced feature handling.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
            
    Returns:
        Configured LightGCN model with feature enhancement
        
    Examples:
        # LightGCN with feature transformation
        model = create_lightgcn_with_features(64, 10, feature_transform_layers=2)
        
        # LightGCN with feature combination
        model = create_lightgcn_with_features(64, 10, combine_features=True, feature_weight=0.2)
    """
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_layers': kwargs.get('num_layers', 3),
        **kwargs
    }
    return LightGCNWithFeatures(config)
