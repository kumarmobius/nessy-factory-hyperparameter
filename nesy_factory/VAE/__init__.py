"""
VAE (Variational Autoencoder) Model Factory Package

This package provides implementations of various Variational Autoencoder models
with a unified interface based on PyTorch. It includes standard VAE, Beta-VAE,
and other VAE variants for different applications.
"""

from .base import BaseVAE
from .standard_vae import StandardVAE, create_standard_vae
from .beta_vae import BetaVAE, create_beta_vae
from .conditional_vae import ConditionalVAE, create_conditional_vae
from .vqvae import VQVAE, create_vqvae

# Registry imports
from .registry import (
    register_vae_model, create_vae_model, create_vae_model_from_config,
    list_available_vae_models, get_available_vae_models, is_vae_model_available, get_vae_model
)

__all__ = [
    # Base class
    'BaseVAE',
    # Standard VAE models
    'StandardVAE',
    'create_standard_vae',
    # Beta VAE models
    'BetaVAE',
    'create_beta_vae',
    # Conditional VAE models
    'ConditionalVAE',
    'create_conditional_vae',
    # VQ-VAE models
    'VQVAE',
    'create_vqvae',
    # Registry functions
    'register_vae_model',
    'create_vae_model',
    'create_vae_model_from_config',
    'list_available_vae_models',
    'get_available_vae_models',
    'is_vae_model_available',
    'get_vae_model'
]

__version__ = '1.0.0'
