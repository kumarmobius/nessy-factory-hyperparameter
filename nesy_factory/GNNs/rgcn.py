"""
Relational Graph Convolutional Network (R-GCN) for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Dict, Any

from .base import BaseGNN


class RGCN(BaseGNN):
    """
    Relational Graph Convolutional Network (R-GCN) model for link prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RGCN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(RGCN, self).__init__(config)
        
        self.num_nodes = config['num_nodes']
        self.num_relations = config['num_relations']
        self.embed_dim = config.get('embed_dim', 64)
        
        # ComplEx decoder requires embedding dimension to be doubled
        self.complex_embed_dim = self.embed_dim * 2

        self.embedding = nn.Embedding(self.num_nodes, self.complex_embed_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, self.complex_embed_dim)
        
        self.convs = nn.ModuleList()
        
        # Determine layer dimensions
        if self.num_layers == 1:
            layer_dims = [self.complex_embed_dim, self.complex_embed_dim]
        else:
            layer_dims = [self.complex_embed_dim] + [self.hidden_dim] * (self.num_layers - 2) + [self.complex_embed_dim]

        # Build layers
        for i in range(self.num_layers -1):
            self.convs.append(
                RGCNConv(layer_dims[i], layer_dims[i+1], self.num_relations)
            )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.embedding.reset_parameters()
        self.relation_embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RGCN model.
        
        Args:
            edge_index: Edge indices tensor of shape [2, num_edges]
            edge_type: Edge type tensor of shape [num_edges]
            
        Returns:
            Node embeddings tensor of shape [num_nodes, complex_embed_dim]
        """
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        ComplEx decoder for link prediction.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            edge_type: Edge types
            
        Returns:
            Scores for each edge
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        rel = self.relation_embedding(edge_type)
        
        src_real, src_imag = src.chunk(2, dim=-1)
        dst_real, dst_imag = dst.chunk(2, dim=-1)
        rel_real, rel_imag = rel.chunk(2, dim=-1)
        
        score_1 = (src_real * rel_real * dst_real).sum(dim=-1)
        score_2 = (src_real * rel_imag * dst_imag).sum(dim=-1)
        score_3 = (src_imag * rel_real * dst_imag).sum(dim=-1)
        score_4 = (src_imag * rel_imag * dst_real).sum(dim=-1)
        
        return score_1 + score_2 + score_3 - score_4

    def _sample_negatives(self, edge_index: torch.Tensor, num_neg_samples: int) -> torch.Tensor:
        """
        Samples negative edges for training by corrupting the tail entity.
        """
        num_pos = edge_index.size(1)
        neg_src = edge_index[0].repeat_interleave(num_neg_samples)
        neg_dst = torch.randint(0, self.num_nodes, (len(neg_src),), device=self.device)
        return torch.stack([neg_src, neg_dst], dim=0)

    def train_step(self, data, neg_samples: int = 1) -> float:
        """
        Perform a single training step for link prediction.
        """
        self._init_optimizer_and_criterion()
        
        self.train()
        self.optimizer.zero_grad()
        
        data = self._to_device(data)
        
        z = self.forward(data.edge_index, data.edge_type)
        
        # Positive examples
        pos_scores = self.decode(z, data.train_edge_index, data.train_edge_type)
        
        # Negative examples
        neg_edge_index = self._sample_negatives(data.train_edge_index, neg_samples)
        neg_edge_type = data.train_edge_type.repeat_interleave(neg_samples)
        neg_scores = self.decode(z, neg_edge_index, neg_edge_type)
        
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        loss = self.criterion(scores, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def eval_step(self, data) -> Dict[str, float]:
        """
        Perform evaluation step for link prediction.
        """
        self._init_optimizer_and_criterion()
        self.eval()
        
        with torch.no_grad():
            data = self._to_device(data)
            z = self.forward(data.edge_index, data.edge_type)
            
            val_scores = self.decode(z, data.valid_edge_index, data.valid_edge_type)
            hit_rate = (val_scores > 0).float().mean()
            
            loss = self.criterion(val_scores, torch.ones_like(val_scores))

        return {'loss': loss.item(), 'hit_rate': hit_rate.item()}