"""
Beta-VAE (Beta Variational Autoencoder) implementation.

This module implements the Beta-VAE as described in:
Higgins, I., et al. (2017). Beta-VAE: Learning basic visual concepts with a constrained variational framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseVAE


class BetaVAE(BaseVAE):
    """
    Beta-VAE implementation with controllable disentanglement.
    
    The beta parameter controls the trade-off between reconstruction quality
    and disentanglement in the latent space.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Beta-VAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, latent_dim
                Optional keys: hidden_dims, learning_rate, beta, kl_weight
        """
        super(BetaVAE, self).__init__(config)
    
    def _build_encoder(self):
        """Build the encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        # Build encoder layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
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
                nn.BatchNorm1d(hidden_dim),
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
    
    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Beta-VAE loss with controllable beta parameter.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss
        if recon_x.shape == x.shape and x.min() >= 0 and x.max() <= 1:
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss with beta weighting
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss with beta parameter
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def create_beta_vae(input_dim: int, latent_dim: int, 
                    hidden_dims: list = [512, 256], 
                    learning_rate: float = 1e-3,
                    beta: float = 4.0,
                    **kwargs) -> BetaVAE:
    """
    Create a Beta-VAE model with the specified parameters.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        beta: Beta parameter for controlling disentanglement
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BetaVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'beta': beta,
        **kwargs
    }
    
    return BetaVAE(config)
