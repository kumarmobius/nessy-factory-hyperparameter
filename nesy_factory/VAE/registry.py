"""
VAE Model Registry

This module provides a registry system for VAE models, allowing easy creation
and management of different VAE architectures.
"""

from typing import Dict, Any, Type, Optional
from .base import BaseVAE
from .standard_vae import StandardVAE
from .beta_vae import BetaVAE
from .conditional_vae import ConditionalVAE
from .vqvae import VQVAE


# Registry for VAE models
_VAE_MODELS: Dict[str, Type[BaseVAE]] = {
    'standard_vae': StandardVAE,
    'beta_vae': BetaVAE,
    'conditional_vae': ConditionalVAE,
    'vqvae': VQVAE,
}

# Model descriptions
_VAE_DESCRIPTIONS: Dict[str, str] = {
    'standard_vae': 'Standard Variational Autoencoder',
    'beta_vae': 'Beta-VAE with controllable disentanglement',
    'conditional_vae': 'Conditional VAE for class-conditional generation',
    'vqvae': 'Vector Quantized VAE with discrete latent representations',
}


def register_vae_model(name: str, model_class: Type[BaseVAE], description: str = ""):
    """
    Register a new VAE model.
    
    Args:
        name: Model name
        model_class: Model class that inherits from BaseVAE
        description: Model description
    """
    if not issubclass(model_class, BaseVAE):
        raise ValueError(f"Model class must inherit from BaseVAE, got {model_class}")
    
    _VAE_MODELS[name] = model_class
    _VAE_DESCRIPTIONS[name] = description or f"{name} VAE model"


def create_vae_model(model_name: str, config: Dict[str, Any]) -> BaseVAE:
    """
    Create a VAE model by name.
    
    Args:
        model_name: Name of the model to create
        config: Configuration dictionary
        
    Returns:
        Instantiated VAE model
        
    Raises:
        ValueError: If model_name is not registered
    """
    if model_name not in _VAE_MODELS:
        available = ', '.join(_VAE_MODELS.keys())
        raise ValueError(f"Unknown VAE model '{model_name}'. Available models: {available}")
    
    model_class = _VAE_MODELS[model_name]
    return model_class(config)


def create_vae_model_from_config(config: Dict[str, Any]) -> BaseVAE:
    """
    Create a VAE model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary containing 'model_type' key
        
    Returns:
        Instantiated VAE model
        
    Raises:
        ValueError: If model_type is not specified or not registered
    """
    if 'model_type' not in config:
        raise ValueError("Configuration must contain 'model_type' key")
    
    model_type = config['model_type']
    return create_vae_model(model_type, config)


def list_available_vae_models():
    """
    List all available VAE models.
    """
    print("Available VAE Models:")
    print("=" * 50)
    for name, description in _VAE_DESCRIPTIONS.items():
        print(f"{name:<20} : {description}")


def get_available_vae_models() -> Dict[str, str]:
    """
    Get dictionary of available VAE models and their descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return _VAE_DESCRIPTIONS.copy()


def is_vae_model_available(model_name: str) -> bool:
    """
    Check if a VAE model is available.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    return model_name in _VAE_MODELS


def get_vae_model(model_name: str) -> Optional[Type[BaseVAE]]:
    """
    Get a VAE model class by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class if found, None otherwise
    """
    return _VAE_MODELS.get(model_name)


def get_vae_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a VAE model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing model information
        
    Raises:
        ValueError: If model_name is not registered
    """
    if model_name not in _VAE_MODELS:
        available = ', '.join(_VAE_MODELS.keys())
        raise ValueError(f"Unknown VAE model '{model_name}'. Available models: {available}")
    
    model_class = _VAE_MODELS[model_name]
    description = _VAE_DESCRIPTIONS[model_name]
    
    return {
        'name': model_name,
        'class': model_class,
        'description': description,
        'base_class': BaseVAE,
        'module': model_class.__module__
    }


def create_standard_vae_quick(input_dim: int, latent_dim: int, **kwargs) -> StandardVAE:
    """
    Quick function to create a standard VAE.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        StandardVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        **kwargs
    }
    return create_vae_model('standard_vae', config)


def create_beta_vae_quick(input_dim: int, latent_dim: int, beta: float = 4.0, **kwargs) -> BetaVAE:
    """
    Quick function to create a Beta-VAE.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        beta: Beta parameter for disentanglement
        **kwargs: Additional configuration parameters
        
    Returns:
        BetaVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'beta': beta,
        **kwargs
    }
    return create_vae_model('beta_vae', config)


def create_conditional_vae_quick(input_dim: int, latent_dim: int, num_classes: int, **kwargs) -> ConditionalVAE:
    """
    Quick function to create a conditional VAE.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        num_classes: Number of classes
        **kwargs: Additional configuration parameters
        
    Returns:
        ConditionalVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'num_classes': num_classes,
        **kwargs
    }
    return create_vae_model('conditional_vae', config)


def create_vqvae_quick(input_dim: int, latent_dim: int, num_embeddings: int = 512, **kwargs) -> VQVAE:
    """
    Quick function to create a VQ-VAE.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        num_embeddings: Number of codebook vectors
        **kwargs: Additional configuration parameters
        
    Returns:
        VQVAE model
    """
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'num_embeddings': num_embeddings,
        **kwargs
    }
    return create_vae_model('vqvae', config)
