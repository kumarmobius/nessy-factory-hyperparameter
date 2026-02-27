"""
Standard Variational Autoencoder (VAE) implementation.

This module implements the standard VAE as described in:
Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseVAE


class StandardVAE(BaseVAE):
    """
    Standard Variational Autoencoder implementation.
    
    This is the basic VAE architecture with fully connected layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Standard VAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, latent_dim
                Optional keys: hidden_dims, learning_rate, beta, kl_weight
        """
        super(StandardVAE, self).__init__(config)
    
    def _build_encoder(self):
        """Build the encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        # Build encoder layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, self.latent_dim)
    
    def _build_decoder(self):
        """Build the decoder network."""
        layers = []
        prev_dim = self.latent_dim
        
        # Build decoder layers (reverse of encoder)
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.input_dim))
        layers.append(nn.Sigmoid())  # For normalized data
        
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (mean, log_var) for the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed output tensor
        """
        return self.decoder(z)


def create_standard_vae(input_dim: int, latent_dim: int, 
                       hidden_dims: list = [512, 256], 
                       learning_rate: float = 1e-3,
                       beta: float = 1.0,
                       **kwargs) -> StandardVAE:
    """
    Create a Standard VAE model with the specified parameters.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        beta: Beta-VAE parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured StandardVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'beta': beta,
        **kwargs
    }
    
    return StandardVAE(config)
