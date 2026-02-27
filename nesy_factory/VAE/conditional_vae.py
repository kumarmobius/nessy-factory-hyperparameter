"""
Conditional VAE (CVAE) implementation.

This module implements the Conditional VAE as described in:
Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseVAE


class ConditionalVAE(BaseVAE):
    """
    Conditional Variational Autoencoder implementation.
    
    This VAE can generate samples conditioned on class labels or other attributes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Conditional VAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, latent_dim, num_classes
                Optional keys: hidden_dims, learning_rate, beta, kl_weight
        """
        # Add num_classes to config if not present
        if 'num_classes' not in config:
            config['num_classes'] = 10  # Default to 10 classes
        
        # Set num_classes before calling parent constructor
        self.num_classes = config['num_classes']
        super(ConditionalVAE, self).__init__(config)
    
    def _build_encoder(self):
        """Build the encoder network."""
        layers = []
        # Input includes both data and class information
        prev_dim = self.input_dim + self.num_classes
        
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
        # Latent includes both z and class information
        prev_dim = self.latent_dim + self.num_classes
        
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
    
    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            y: Class labels tensor of shape [batch_size] or one-hot [batch_size, num_classes]
            
        Returns:
            Tuple of (mean, log_var) for the latent distribution
        """
        # Convert labels to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.num_classes).float()
        else:
            y_onehot = y
        
        # Concatenate input and class information
        xy = torch.cat([x, y_onehot], dim=1)
        
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            y: Class labels tensor of shape [batch_size] or one-hot [batch_size, num_classes]
            
        Returns:
            Reconstructed output tensor
        """
        # Convert labels to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.num_classes).float()
        else:
            y_onehot = y
        
        # Concatenate latent and class information
        zy = torch.cat([z, y_onehot], dim=1)
        
        return self.decoder(zy)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Conditional VAE.
        
        Args:
            x: Input tensor
            y: Class labels tensor
            
        Returns:
            Dictionary containing:
                - 'recon': Reconstructed output
                - 'mu': Mean of latent distribution
                - 'log_var': Log variance of latent distribution
                - 'z': Sampled latent representation
        """
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, y)
        
        return {
            'recon': recon,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def sample(self, num_samples: int = 1, y: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            y: Class labels for conditional generation
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            if y is None:
                # Generate random class labels
                y = torch.randint(0, self.num_classes, (num_samples,)).to(self.device)
            
            samples = self.decode(z, y)
        return samples
    
    def reconstruct(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input without sampling (use mean of latent distribution).
        
        Args:
            x: Input tensor
            y: Class labels tensor
            
        Returns:
            Reconstructed tensor
        """
        with torch.no_grad():
            mu, _ = self.encode(x, y)
            recon = self.decode(mu, y)
        return recon


def create_conditional_vae(input_dim: int, latent_dim: int, num_classes: int,
                          hidden_dims: list = [512, 256], 
                          learning_rate: float = 1e-3,
                          beta: float = 1.0,
                          **kwargs) -> ConditionalVAE:
    """
    Create a Conditional VAE model with the specified parameters.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        num_classes: Number of classes for conditioning
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        beta: Beta parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ConditionalVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'num_classes': num_classes,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'beta': beta,
        **kwargs
    }
    
    return ConditionalVAE(config)
