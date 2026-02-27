"""
Base VAE class that all VAE models should inherit from.

This class provides the common interface and initialization that all VAE models
should implement, including the VAE loss function and common utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import math


class BaseVAE(nn.Module, ABC):
    """
    Abstract base class for all VAE models.
    
    This class provides the common interface and initialization that all VAE models
    should implement, including the VAE loss function and common utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base VAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(BaseVAE, self).__init__()
        
        # Model architecture parameters
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.hidden_dims = config.get('hidden_dims', [512, 256])
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 128)
        
        # VAE-specific parameters
        self.beta = config.get('beta', 1.0)  # Beta-VAE parameter
        self.kl_weight = config.get('kl_weight', 1.0)  # KL divergence weight
        
        # Device configuration
        self.device = config.get('device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model components
        self._build_encoder()
        self._build_decoder()
        
    @abstractmethod
    def _build_encoder(self):
        """Build the encoder network. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _build_decoder(self):
        """Build the decoder network. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log_var) for the latent distribution
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed output tensor
        """
        pass
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent tensor
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing:
                - 'recon': Reconstructed output
                - 'mu': Mean of latent distribution
                - 'log_var': Log variance of latent distribution
                - 'z': Sampled latent representation
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Dictionary containing:
                - 'total_loss': Total VAE loss
                - 'recon_loss': Reconstruction loss
                - 'kl_loss': KL divergence loss
        """
        # Reconstruction loss (MSE or BCE)
        if recon_x.shape == x.shape and x.min() >= 0 and x.max() <= 1:
            # Use BCE for normalized data
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            # Use MSE for general data
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.decode(z)
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input without sampling (use mean of latent distribution).
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            recon = self.decode(mu)
        return recon
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (mean) of input.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation tensor
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'beta': self.beta,
            'kl_weight': self.kl_weight
        }
    
    def save_model(self, filepath: str):
        """
        Save model state dict.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_model_info()
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load model state dict.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
