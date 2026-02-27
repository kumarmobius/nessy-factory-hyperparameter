"""
Temporal Graph Convolutional Network (TGCN) for node classification on temporal graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN as TGCNCell
from typing import Dict, Any

from .base import BaseGNN

class FocalLoss(nn.Module):
    """
    Focal Loss implementation that handles class imbalance using a tensor alpha.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Get log probabilities from the model output
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Gather the log probabilities for the true target classes
        log_probs_targets = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate probabilities (pt)
        probs = log_probs_targets.exp()
        
        # The core focal loss calculation
        focal_term = (1 - probs)**self.gamma
        ce_term = -log_probs_targets
        loss = focal_term * ce_term
        
        # Apply alpha weighting for class balancing
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, device=inputs.device)
            
            # Move alpha to the correct device
            alpha = self.alpha.to(inputs.device)
            
            # Gather the alpha value for each target class
            alpha_t = alpha.gather(0, targets.data.view(-1))
            loss = alpha_t * loss

        # Mask out ignored indices (e.g., 'unknown' class)
        mask = targets != self.ignore_index
        loss = loss[mask]
        
        # Return the mean loss over valid (non-ignored) samples
        if loss.numel() > 0:
            return loss.mean()
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

class TGCN(BaseGNN):
    """
    Temporal Graph Convolutional Network (TGCN) model with improved architecture
    and loss options for imbalanced data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TGCN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(TGCN, self).__init__(config)

        # Improved architecture from the script
        self.tgcn1 = TGCNCell(in_channels=self.input_dim, out_channels=self.hidden_dim)
        self.tgcn2 = TGCNCell(in_channels=self.hidden_dim, out_channels=self.output_dim)
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        # TGCNCell from torch_geometric_temporal doesn't have a public reset_parameters method.
        # Re-initializing the model is the recommended way to reset its parameters.
        pass

    def _init_optimizer_and_criterion(self):
        """
        Initialize optimizer and loss criterion, with options for Focal Loss.
        """
        # Initialize optimizer (from base class)
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Initialize criterion
        if self.criterion is None:
            if self.config.get('use_focal_loss', False):
                print("Using Focal Loss")
                self.criterion = FocalLoss(
                    alpha=self.config.get('focal_loss_alpha'),
                    gamma=self.config.get('focal_loss_gamma', 2.0),
                    ignore_index=self.config.get('ignore_index', 2)
                )
            elif self.config.get('use_class_weights', False) and 'class_weights' in self.config:
                print("Using weighted CrossEntropyLoss")
                class_weights = torch.tensor(self.config['class_weights'], dtype=torch.float).to(self.device)
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights, 
                    ignore_index=self.config.get('ignore_index', 2)
                )
            else:
                print("Using standard CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.get('ignore_index', 2))


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the TGCN model.
        
        Args:
            x: Node features tensor
            edge_index: Edge indices tensor
            **kwargs: Additional arguments (edge_weight, etc.)
            
        Returns:
            Output tensor
        """
        edge_weight = kwargs.get('edge_weight', None)
        
        # First temporal convolution
        h = self.tgcn1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Second temporal convolution (final layer, no activation/dropout)
        h = self.tgcn2(h, edge_index, edge_weight)
        
        return h