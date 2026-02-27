"""
PinSAGE implementation for recommendation systems and node classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import random
from typing import Dict, Any, List, Tuple, Optional

from .base import BaseGNN


class ImportanceBasedSampler:
    """Importance-based neighbor sampler for PinSAGE"""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, walk_length: int = 5, num_walks: int = 50):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.num_walks = num_walks

        # Convert to NetworkX for easier random walk computation
        self.G = to_networkx(
            Data(edge_index=edge_index, num_nodes=num_nodes),
            to_undirected=True
        )

        # Precompute importance scores using random walks
        self.importance_scores = self._compute_importance_scores()

    def _random_walk(self, start_node: int, length: int) -> List[int]:
        """Perform a random walk starting from start_node"""
        if start_node not in self.G:
            return [start_node]

        walk = [start_node]
        current = start_node

        for _ in range(length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def _compute_importance_scores(self) -> Dict[int, Dict[int, float]]:
        """Compute importance scores for all node pairs using random walks"""
        importance = {}

        for node in range(self.num_nodes):
            node_importance = {}
            visit_counts = {}

            # Perform multiple random walks from this node
            for _ in range(self.num_walks):
                walk = self._random_walk(node, self.walk_length)
                for visited_node in walk[1:]:  # Exclude the starting node
                    visit_counts[visited_node] = visit_counts.get(visited_node, 0) + 1

            # Normalize visit counts to get importance scores
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for neighbor, count in visit_counts.items():
                    node_importance[neighbor] = count / total_visits

            importance[node] = node_importance

        return importance

    def sample_neighbors(self, nodes: List[int], num_samples: int = 10) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
        """Sample important neighbors for given nodes"""
        sampled_neighbors = {}
        importance_weights = {}

        for node in nodes:
            node = int(node)
            if node in self.importance_scores:
                # Get neighbors and their importance scores
                neighbors_scores = self.importance_scores[node]

                if neighbors_scores:
                    # Sort neighbors by importance and take top k
                    sorted_neighbors = sorted(
                        neighbors_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Sample up to num_samples neighbors
                    selected = sorted_neighbors[:min(num_samples, len(sorted_neighbors))]

                    if selected:
                        neighbors, weights = zip(*selected)
                        sampled_neighbors[node] = list(neighbors)
                        importance_weights[node] = list(weights)
                    else:
                        sampled_neighbors[node] = []
                        importance_weights[node] = []
                else:
                    sampled_neighbors[node] = []
                    importance_weights[node] = []
            else:
                sampled_neighbors[node] = []
                importance_weights[node] = []

        return sampled_neighbors, importance_weights


class ImportancePooling(nn.Module):
    """Learnable importance-weighted pooling for PinSAGE"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.importance_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, neighbor_indices: Dict[int, List[int]], 
                importance_weights: Dict[int, List[float]]) -> torch.Tensor:
        """
        Args:
            x: node features [num_nodes, in_channels]
            neighbor_indices: dict mapping node_idx -> list of neighbor indices
            importance_weights: dict mapping node_idx -> list of importance weights
        """
        batch_size = len(neighbor_indices)
        out = torch.zeros(batch_size, x.size(1), device=x.device)

        for i, (node_idx, neighbors) in enumerate(neighbor_indices.items()):
            if len(neighbors) == 0:
                # If no neighbors, use self features
                out[i] = x[node_idx] if isinstance(node_idx, int) else x[0]
                continue

            # Get neighbor features
            neighbor_features = x[neighbors]  # [num_neighbors, in_channels]

            # Compute learnable importance scores
            learned_importance = self.importance_mlp(neighbor_features)  # [num_neighbors, 1]
            learned_importance = torch.softmax(learned_importance.squeeze(-1), dim=0)

            # Combine with pre-computed importance weights
            if node_idx in importance_weights and importance_weights[node_idx]:
                precomputed_weights = torch.tensor(
                    importance_weights[node_idx],
                    device=x.device,
                    dtype=torch.float
                )
                # Normalize precomputed weights
                precomputed_weights = torch.softmax(precomputed_weights, dim=0)

                # Combine learned and precomputed importance
                final_weights = 0.5 * learned_importance + 0.5 * precomputed_weights
            else:
                final_weights = learned_importance

            # Weighted aggregation
            out[i] = torch.sum(neighbor_features * final_weights.unsqueeze(-1), dim=0)

        return out


class PinSAGELayer(nn.Module):
    """Single PinSAGE layer with importance-based sampling and pooling"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Self and neighbor transformation
        self.self_transform = nn.Linear(in_channels, out_channels)
        self.neighbor_transform = nn.Linear(in_channels, out_channels)

        # Importance-based pooling
        self.importance_pooling = ImportancePooling(in_channels)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, sampled_neighbors: Dict[int, List[int]], 
                importance_weights: Dict[int, List[float]], batch_nodes: List[int]) -> torch.Tensor:
        """
        Args:
            x: all node features
            sampled_neighbors: dict of sampled neighbors for batch nodes
            importance_weights: importance weights for sampled neighbors
            batch_nodes: nodes in current batch
        """
        # Transform self features
        self_features = self.self_transform(x[batch_nodes])

        # Aggregate neighbor features using importance pooling
        neighbor_features = self.importance_pooling(x, sampled_neighbors, importance_weights)
        neighbor_features = self.neighbor_transform(neighbor_features)

        # Combine self and neighbor features
        out = self_features + neighbor_features
        out = self.layer_norm(out)

        return out


class PinSAGE(BaseGNN):
    """
    PinSAGE model for recommendation systems and node classification.
    
    Implementation based on:
    "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
    by Ying et al. (2018)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PinSAGE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, hidden_dim, output_dim
                Optional keys: num_layers, dropout, num_samples, walk_length, num_walks
        """
        super(PinSAGE, self).__init__(config)
        
        # PinSAGE-specific parameters
        self.num_samples = config.get('num_samples', 10)
        self.walk_length = config.get('walk_length', 5)
        self.num_walks = config.get('num_walks', 50)
        
        # Build PinSAGE layers
        self.layers = nn.ModuleList()
        
        # Determine layer dimensions
        if self.num_layers == 1:
            layer_dims = [self.input_dim, self.output_dim]
        else:
            layer_dims = [self.input_dim] + self.hidden_dims[:self.num_layers-1] + [self.output_dim]
        
        # Build layers
        for i in range(self.num_layers):
            self.layers.append(PinSAGELayer(layer_dims[i], layer_dims[i + 1]))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # Sampler will be initialized when edge_index is provided
        self.sampler = None
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.layers:
            if hasattr(layer.self_transform, 'reset_parameters'):
                layer.self_transform.reset_parameters()
            if hasattr(layer.neighbor_transform, 'reset_parameters'):
                layer.neighbor_transform.reset_parameters()
            if hasattr(layer.layer_norm, 'reset_parameters'):
                layer.layer_norm.reset_parameters()

    def _initialize_sampler(self, edge_index: torch.Tensor, num_nodes: int):
        """Initialize the importance-based sampler."""
        if self.sampler is None:
            self.sampler = ImportanceBasedSampler(
                edge_index=edge_index,
                num_nodes=num_nodes,
                walk_length=self.walk_length,
                num_walks=self.num_walks
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch_nodes: Optional[List[int]] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass of the PinSAGE model.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices tensor of shape [2, num_edges]
            batch_nodes: List of node indices to compute embeddings for.
                        If None, computes for all nodes (memory intensive).
            **kwargs: Additional arguments
            
        Returns:
            Output tensor of shape [len(batch_nodes), output_dim] or [num_nodes, output_dim]
        """
        # Initialize sampler if not already done
        self._initialize_sampler(edge_index, x.size(0))
        
        # If batch_nodes not provided, use all nodes (not recommended for large graphs)
        if batch_nodes is None:
            batch_nodes = list(range(x.size(0)))
        
        current_features = x
        current_nodes = batch_nodes

        for i, layer in enumerate(self.layers):
            # Sample neighbors for current batch
            sampled_neighbors, importance_weights = self.sampler.sample_neighbors(
                current_nodes, self.num_samples
            )

            # Apply PinSAGE layer
            x_batch = layer(current_features, sampled_neighbors, importance_weights, current_nodes)

            # Update features for batch nodes
            if i < len(self.layers) - 1:
                # Apply activation and dropout for intermediate layers
                x_batch = F.relu(x_batch)
                x_batch = self.dropout_layer(x_batch)
                
                # Update features for batch nodes
                new_features = current_features.clone()
                new_features[current_nodes] = x_batch
                current_features = new_features
            else:
                # Final layer - return the output
                final_output = x_batch

        return final_output

    def train_step(self, data, mask=None, batch_nodes=None) -> float:
        """
        Perform a single training step.
        
        Args:
            data: Training data with x (node features), edge_index (edges), and y (labels)
            mask: Optional mask for selecting specific nodes for training
            batch_nodes: Optional list of specific nodes to train on
            
        Returns:
            Loss value for this training step
        """
        # Initialize optimizer and criterion if not already done
        self._init_optimizer_and_criterion()
        
        self.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = self._to_device(data)
        
        # Determine training nodes
        if batch_nodes is not None:
            train_nodes = batch_nodes
        elif mask is not None:
            mask = mask.to(self.device)
            train_nodes = mask.nonzero().squeeze().tolist()
            if isinstance(train_nodes, int):
                train_nodes = [train_nodes]
        else:
            train_nodes = list(range(data.x.size(0)))
        
        # Forward pass
        out = self.forward(data.x, data.edge_index, batch_nodes=train_nodes)
        
        # Compute loss
        if batch_nodes is not None:
            loss = self.criterion(out, data.y[train_nodes])
        elif mask is not None:
            loss = self.criterion(out, data.y[train_nodes])
        else:
            loss = self.criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def eval_step(self, data, mask=None, batch_nodes=None) -> Dict[str, float]:
        """
        Perform evaluation step.
        
        Args:
            data: Evaluation data
            mask: Optional mask for selecting specific nodes for evaluation
            batch_nodes: Optional list of specific nodes to evaluate on
            
        Returns:
            Dictionary containing loss and accuracy
        """
        self._init_optimizer_and_criterion()
        self.eval()
        
        with torch.no_grad():
            # Move data to device
            data = self._to_device(data)
            
            # Determine evaluation nodes
            if batch_nodes is not None:
                eval_nodes = batch_nodes
            elif mask is not None:
                mask = mask.to(self.device)
                eval_nodes = mask.nonzero().squeeze().tolist()
                if isinstance(eval_nodes, int):
                    eval_nodes = [eval_nodes]
            else:
                eval_nodes = list(range(data.x.size(0)))
            
            # Forward pass
            out = self.forward(data.x, data.edge_index, batch_nodes=eval_nodes)
            
            # Compute loss and accuracy
            if batch_nodes is not None:
                loss = self.criterion(out, data.y[eval_nodes])
                pred = out.argmax(dim=1)
                correct = pred.eq(data.y[eval_nodes]).sum().item()
                accuracy = correct / len(eval_nodes)
            elif mask is not None:
                loss = self.criterion(out, data.y[eval_nodes])
                pred = out.argmax(dim=1)
                correct = pred.eq(data.y[eval_nodes]).sum().item()
                accuracy = correct / len(eval_nodes)
            else:
                loss = self.criterion(out, data.y)
                pred = out.argmax(dim=1)
                correct = pred.eq(data.y).sum().item()
                accuracy = correct / data.y.size(0)
                
        return {'loss': loss.item(), 'accuracy': accuracy}

    def predict(self, data, batch_nodes=None) -> torch.Tensor:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            batch_nodes: Optional list of specific nodes to predict for
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            data = self._to_device(data)
            return self.forward(data.x, data.edge_index, batch_nodes=batch_nodes)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      batch_nodes: Optional[List[int]] = None) -> torch.Tensor:
        """
        Get node embeddings from the last hidden layer.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            batch_nodes: Optional list of nodes to get embeddings for
            
        Returns:
            Node embeddings tensor
        """
        # Initialize sampler if not already done
        self._initialize_sampler(edge_index, x.size(0))
        
        if batch_nodes is None:
            batch_nodes = list(range(x.size(0)))
        
        current_features = x
        current_nodes = batch_nodes

        # Forward pass until the last layer
        for i, layer in enumerate(self.layers[:-1]):
            sampled_neighbors, importance_weights = self.sampler.sample_neighbors(
                current_nodes, self.num_samples
            )
            
            x_batch = layer(current_features, sampled_neighbors, importance_weights, current_nodes)
            x_batch = F.relu(x_batch)
            x_batch = self.dropout_layer(x_batch)
            
            # Update features for batch nodes
            new_features = current_features.clone()
            new_features[current_nodes] = x_batch
            current_features = new_features

        return current_features[batch_nodes]


def create_pinsage(input_dim: int, hidden_dim, output_dim: int, **kwargs) -> PinSAGE:
    """
    Convenience function to create a PinSAGE model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension(s). Can be:
                   - int: Single dimension for all hidden layers
                   - list/tuple: Different dimensions for each hidden layer
        output_dim: Output dimension
        **kwargs: Additional configuration parameters including:
                 - num_samples: Number of neighbors to sample (default: 10)
                 - walk_length: Length of random walks (default: 5)
                 - num_walks: Number of random walks per node (default: 50)
        
    Returns:
        Configured PinSAGE model
        
    Examples:
        # Basic PinSAGE model
        model = create_pinsage(128, 64, 10)
        
        # PinSAGE with custom sampling parameters
        model = create_pinsage(128, [64, 128], 10, num_samples=5, walk_length=3)
        
        # PinSAGE for recommendation system
        model = create_pinsage(
            input_dim=128, 
            hidden_dim=[256, 128], 
            output_dim=64,
            num_layers=3,
            num_samples=15,
            walk_length=4,
            num_walks=30
        )
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        **kwargs
    }
    return PinSAGE(config)
