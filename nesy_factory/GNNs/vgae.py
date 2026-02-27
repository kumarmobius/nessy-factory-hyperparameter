"""
Variational Graph Autoencoder (VGAE) implementation for graph generation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling, add_self_loops, remove_self_loops
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .base import BaseGNN


class VariationalGCNEncoder(nn.Module):
    """
    Variational Graph Convolutional Encoder for VGAE.
    
    This encoder uses GCN layers to encode node features into latent representations
    with mean (mu) and log standard deviation (logstd) for variational sampling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: Optional[int] = None,
                 num_layers: int = 2, dropout: float = 0.0, normalize: bool = True, 
                 add_self_loops: bool = True, bias: bool = True):
        """
        Initialize the variational encoder.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output latent dimension
            hidden_channels: Hidden layer dimension (defaults to 2 * out_channels)
            num_layers: Number of encoder layers
            dropout: Dropout probability
            normalize: Whether to normalize in GCN layers
            add_self_loops: Whether to add self loops in GCN layers
            bias: Whether to use bias in GCN layers
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = 2 * out_channels
            
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build encoder layers
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer directly to output
            self.convs.append(GCNConv(in_channels, hidden_channels, normalize, add_self_loops, bias))
        else:
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels, normalize, add_self_loops, bias))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize, add_self_loops, bias))
        
        # Output layers for mean and log standard deviation
        self.conv_mu = GCNConv(hidden_channels, out_channels, normalize, add_self_loops, bias)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, normalize, add_self_loops, bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_logstd.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Tuple of (mu, logstd) tensors for variational sampling
        """
        # Pass through encoder layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Compute mean and log standard deviation
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd


class VariationalGATEncoder(nn.Module):
    """
    Variational Graph Attention Encoder for VGAE.
    
    Uses GAT layers instead of GCN for the encoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: Optional[int] = None,
                 num_layers: int = 2, heads: int = 1, dropout: float = 0.0, 
                 add_self_loops: bool = True, bias: bool = True):
        """
        Initialize the variational GAT encoder.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output latent dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of encoder layers
            heads: Number of attention heads
            dropout: Dropout probability
            add_self_loops: Whether to add self loops
            bias: Whether to use bias
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = 2 * out_channels
            
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build encoder layers
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=dropout, 
                                    add_self_loops=add_self_loops, bias=bias))
        else:
            # First layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=dropout,
                                    add_self_loops=add_self_loops, bias=bias))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads, 
                                        dropout=dropout, add_self_loops=add_self_loops, bias=bias))
        
        # Output layers
        final_hidden = hidden_channels * heads if num_layers > 1 else hidden_channels
        self.conv_mu = GATConv(final_hidden, out_channels, 1, dropout=dropout,
                              add_self_loops=add_self_loops, bias=bias)
        self.conv_logstd = GATConv(final_hidden, out_channels, 1, dropout=dropout,
                                  add_self_loops=add_self_loops, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_logstd.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the GAT encoder."""
        # Pass through encoder layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Compute mean and log standard deviation
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd


class VGAEModel(BaseGNN):
    """
    Variational Graph Autoencoder (VGAE) model for graph generation tasks.
    
    This model learns to encode graphs into a latent space and decode them back,
    enabling graph generation and link prediction tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VGAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, output_dim (latent_dim)
                Optional keys: encoder_type, encoder_layers, encoder_heads, 
                             threshold, add_self_loops_generation, normalize_features
        """
        super(VGAEModel, self).__init__(config)
        
        # VGAE-specific parameters
        self.latent_dim = config['output_dim']  # Use output_dim as latent dimension
        self.encoder_type = config.get('encoder_type', 'gcn').lower()
        self.encoder_layers = config.get('encoder_layers', 2)
        self.encoder_heads = config.get('encoder_heads', 1)  # For GAT encoder
        self.threshold = config.get('threshold', 0.5)
        self.add_self_loops_generation = config.get('add_self_loops_generation', False)
        self.normalize_features = config.get('normalize_features', True)
        
        # Build encoder based on type
        if self.encoder_type == 'gcn':
            encoder = VariationalGCNEncoder(
                in_channels=self.input_dim,
                out_channels=self.latent_dim,
                hidden_channels=self.hidden_dim,
                num_layers=self.encoder_layers,
                dropout=self.dropout,
                normalize=config.get('normalize', True),
                add_self_loops=config.get('add_self_loops', True),
                bias=config.get('bias', True)
            )
        elif self.encoder_type == 'gat':
            encoder = VariationalGATEncoder(
                in_channels=self.input_dim,
                out_channels=self.latent_dim,
                hidden_channels=self.hidden_dim,
                num_layers=self.encoder_layers,
                heads=self.encoder_heads,
                dropout=self.dropout,
                add_self_loops=config.get('add_self_loops', True),
                bias=config.get('bias', True)
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}. "
                           f"Supported types: gcn, gat")
        
        # Create VGAE model
        self.vgae = VGAE(encoder)
        
        # Move to device
        self.vgae = self.vgae.to(self.device)
        
        # Override criterion for VGAE
        if self.criterion is None:
            self.criterion = self._vgae_loss
    
    def _vgae_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, 
                   neg_edge_index: torch.Tensor, num_nodes: int, beta: float = 1.0) -> torch.Tensor:
        """
        Compute VGAE loss (reconstruction + KL divergence).
        
        Args:
            z: Latent representations
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            num_nodes: Number of nodes in the graph
            beta: Weight for KL divergence term
            
        Returns:
            Total VGAE loss
        """
        recon_loss = self.vgae.recon_loss(z, pos_edge_index, neg_edge_index)
        kl_loss = (1 / num_nodes) * self.vgae.kl_loss()
        return recon_loss + beta * kl_loss
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass - encode input to latent space.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            **kwargs: Additional arguments
            
        Returns:
            Latent representations [num_nodes, latent_dim]
        """
        if self.normalize_features:
            x = F.normalize(x.float(), dim=-1)
        
        return self.vgae.encode(x, edge_index)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        if self.normalize_features:
            x = F.normalize(x.float(), dim=-1)
        return self.vgae.encode(x, edge_index)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to adjacency matrix."""
        return self.vgae.decoder.forward_all(z)
    
    def train_step(self, data, mask=None, beta: float = 1.0) -> float:
        """
        Perform a single training step for VGAE.
        
        Args:
            data: Training data with x, edge_index, pos_edge_label_index, neg_edge_label_index
            mask: Not used for VGAE
            beta: Weight for KL divergence term
            
        Returns:
            Loss value for this training step
        """
        # Initialize optimizer if not already done
        self._init_optimizer_and_criterion()
        
        self.train()
        self.vgae.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = self._to_device(data)
        
        # Encode
        z = self.encode(data.x, data.edge_index)
        
        # Compute loss
        loss = self._vgae_loss(z, data.pos_edge_label_index, data.neg_edge_label_index, 
                              data.num_nodes, beta)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, mask=None) -> Dict[str, float]:
        """
        Perform evaluation step for VGAE.
        
        Args:
            data: Evaluation data
            mask: Not used for VGAE
            
        Returns:
            Dictionary containing loss, AUC, and AP scores
        """
        self._init_optimizer_and_criterion()
        self.eval()
        self.vgae.eval()
        
        with torch.no_grad():
            # Move data to device
            data = self._to_device(data)
            
            # Encode
            z = self.encode(data.x, data.edge_index)
            
            # Compute loss
            loss = self._vgae_loss(z, data.pos_edge_label_index, data.neg_edge_label_index, 
                                  data.num_nodes)
            
            # Compute AUC and AP
            auc, ap = self.vgae.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
                
        return {'loss': loss.item(), 'auc': auc, 'ap': ap}
    
    def generate_graphs(self, data_list: List, num_graphs: int = 5) -> List[torch.Tensor]:
        """
        Generate graphs by reconstructing adjacency matrices.
        
        Args:
            data_list: List of data samples to use as templates
            num_graphs: Number of graphs to generate
            
        Returns:
            List of adjacency matrices for generated graphs
        """
        self.eval()
        self.vgae.eval()
        
        generated_adj = []
        
        with torch.no_grad():
            for i, data in enumerate(data_list):
                if i >= num_graphs:
                    break
                    
                # Move data to device
                data = self._to_device(data)
                
                # Encode and decode
                z = self.encode(data.x, data.edge_index)
                adj = self.decode(z)
                generated_adj.append(adj)
        
        return generated_adj
    
    def adjacency_to_networkx(self, adj_matrix: torch.Tensor, threshold: Optional[float] = None) -> nx.Graph:
        """
        Convert adjacency matrix to NetworkX graph.
        
        Args:
            adj_matrix: Adjacency matrix tensor
            threshold: Threshold for edge creation (uses self.threshold if None)
            
        Returns:
            NetworkX graph
        """
        if threshold is None:
            threshold = self.threshold
            
        # Binarize adjacency matrix
        adj_binary = adj_matrix > threshold
        indices = torch.where(adj_binary)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        if not self.add_self_loops_generation:
            edges = [(i.item(), j.item()) for i, j in zip(indices[0], indices[1]) if i != j]
        else:
            edges = [(i.item(), j.item()) for i, j in zip(indices[0], indices[1])]
        
        G.add_edges_from(edges)
        return G
    
    def visualize_generated_graphs(self, data_list: List, num_graphs: int = 5, 
                                 figsize: Tuple[int, int] = (15, 3)) -> None:
        """
        Generate and visualize graphs.
        
        Args:
            data_list: List of data samples
            num_graphs: Number of graphs to generate and visualize
            figsize: Figure size for plotting
        """
        # Generate adjacency matrices
        adj_matrices = self.generate_graphs(data_list, num_graphs)
        
        # Create subplots
        fig, axes = plt.subplots(1, num_graphs, figsize=figsize)
        if num_graphs == 1:
            axes = [axes]
        
        # Plot each graph
        for i, adj in enumerate(adj_matrices):
            G = self.adjacency_to_networkx(adj)
            
            # Plot on subplot
            plt.sca(axes[i])
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_size=50, node_color='lightblue', 
                   edge_color='gray', alpha=0.7)
            plt.title(f'Generated Graph {i+1}')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the VGAE model."""
        info = super().get_model_info()
        info.update({
            'model_type': 'VGAE',
            'latent_dim': self.latent_dim,
            'encoder_type': self.encoder_type,
            'encoder_layers': self.encoder_layers,
            'encoder_heads': self.encoder_heads,
            'threshold': self.threshold,
            'normalize_features': self.normalize_features,
            'add_self_loops_generation': self.add_self_loops_generation
        })
        return info


def create_vgae(input_dim: int, latent_dim: int, **kwargs) -> VGAEModel:
    """
    Convenience function to create a VGAE model.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VGAE model
        
    Examples:
        # Basic VGAE with GCN encoder
        model = create_vgae(input_dim=10, latent_dim=16)
        
        # VGAE with GAT encoder
        model = create_vgae(input_dim=10, latent_dim=16, encoder_type='gat', 
                           encoder_heads=4, encoder_layers=3)
        
        # VGAE with custom parameters
        model = create_vgae(input_dim=10, latent_dim=16, hidden_dim=32,
                           threshold=0.7, normalize_features=True)
    """
    config = {
        'input_dim': input_dim,
        'output_dim': latent_dim,  # Use output_dim for latent dimension
        'hidden_dim': kwargs.get('hidden_dim', 2 * latent_dim),
        **kwargs
    }
    return VGAEModel(config)
