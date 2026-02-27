"""
Vector Quantized VAE (VQ-VAE) implementation.

This module implements the VQ-VAE as described in:
van den Oord, A., et al. (2017). Neural discrete representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseVAE


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Initialize Vector Quantizer.
        
        Args:
            num_embeddings: Number of codebook vectors
            embedding_dim: Dimension of each codebook vector
            commitment_cost: Commitment loss weight
        """
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through vector quantizer.
        
        Args:
            inputs: Input tensor of shape [batch_size, embedding_dim, ...]
            
        Returns:
            Tuple of (quantized, vq_loss, encoding_indices)
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices


class VQVAE(BaseVAE):
    """
    Vector Quantized Variational Autoencoder implementation.
    
    This VAE uses discrete latent representations through vector quantization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VQ-VAE model.
        
        Args:
            config: Configuration dictionary containing model parameters
                Required keys: input_dim, latent_dim
                Optional keys: hidden_dims, learning_rate, num_embeddings, embedding_dim
        """
        # Add VQ-specific parameters
        self.num_embeddings = config.get('num_embeddings', 512)
        self.embedding_dim = config.get('embedding_dim', config.get('latent_dim', 64))
        self.commitment_cost = config.get('commitment_cost', 0.25)
        
        super(VQVAE, self).__init__(config)
    
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
        
        # Final layer to embedding dimension
        layers.append(nn.Linear(prev_dim, self.embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Vector quantizer
        self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.commitment_cost)
    
    def _build_decoder(self):
        """Build the decoder network."""
        layers = []
        prev_dim = self.embedding_dim
        
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
            Tuple of (quantized, vq_loss) - VQ-VAE doesn't use mean/log_var
        """
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.vq(z)
        return quantized, vq_loss
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Reconstructed output tensor
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing:
                - 'recon': Reconstructed output
                - 'quantized': Quantized latent representation
                - 'vq_loss': Vector quantization loss
                - 'encoding_indices': Codebook indices
        """
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.vq(z)
        recon = self.decode(quantized)
        
        return {
            'recon': recon,
            'quantized': quantized,
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices
        }
    
    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                 quantized: torch.Tensor, vq_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VQ-VAE loss.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            quantized: Quantized latent representation
            vq_loss: Vector quantization loss
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss
        if recon_x.shape == x.shape and x.min() >= 0 and x.max() <= 1:
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # Total loss (no KL divergence in VQ-VAE)
        total_loss = recon_loss + vq_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'kl_loss': torch.tensor(0.0, device=x.device)  # No KL loss in VQ-VAE
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
            # Sample random codebook indices
            encoding_indices = torch.randint(0, self.num_embeddings, (num_samples, 1)).to(self.device)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=self.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Get quantized vectors
            quantized = torch.matmul(encodings, self.vq.embedding.weight)
            
            # Decode
            samples = self.decode(quantized)
        return samples


def create_vqvae(input_dim: int, latent_dim: int, 
                 hidden_dims: list = [512, 256], 
                 learning_rate: float = 1e-3,
                 num_embeddings: int = 512,
                 embedding_dim: int = None,
                 **kwargs) -> VQVAE:
    """
    Create a VQ-VAE model with the specified parameters.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension (used as embedding_dim if not specified)
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        num_embeddings: Number of codebook vectors
        embedding_dim: Dimension of codebook vectors (defaults to latent_dim)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VQVAE model
    """
    if embedding_dim is None:
        embedding_dim = latent_dim
    
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'num_embeddings': num_embeddings,
        'embedding_dim': embedding_dim,
        **kwargs
    }
    
    return VQVAE(config)

