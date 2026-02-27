"""
Model Registry for GNN Model Factory.
Provides a centralized way to register and instantiate different GNN models.
"""

from typing import Dict, Any, Type, Optional, Callable
import importlib
from .base import BaseGNN


class ModelRegistry:
    """Registry for GNN models that allows dynamic model instantiation."""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseGNN]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in models
        self._register_builtin_models()
    
    def register(self, name: str, model_class: Type[BaseGNN], 
                description: str = "", default_config: Optional[Dict[str, Any]] = None):
        """
        Register a model class.
        
        Args:
            name: Model name (e.g., 'gcn', 'gat')
            model_class: Model class that inherits from BaseGNN
            description: Description of the model
            default_config: Default configuration for the model
        """
        if not issubclass(model_class, BaseGNN):
            raise ValueError(f"Model class {model_class} must inherit from BaseGNN")
        
        self._models[name.lower()] = model_class
        self._model_info[name.lower()] = {
            'class': model_class,
            'description': description,
            'default_config': default_config or {}
        }
    
    def create_model(self, model_name: str, config: Dict[str, Any]) -> BaseGNN:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Configuration dictionary
            
        Returns:
            Model instance
        """
        model_name = model_name.lower()
        
        if model_name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
        
        model_class = self._models[model_name]
        
        # Merge with default config if available
        default_config = self._model_info[model_name].get('default_config', {})
        merged_config = {**default_config, **config}
        
        return model_class(merged_config)
    
    def create_model_from_config_file(self, model_name: str, config_name: str, 
                                    config_path: str = 'configs/gcn_configs.yaml') -> BaseGNN:
        """
        Create a model from a configuration file.
        
        Args:
            model_name: Name of the model to create
            config_name: Name of the configuration in the YAML file
            config_path: Path to the configuration file
            
        Returns:
            Model instance
        """
        # Import utils from parent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from utils import get_config_by_name
        
        config = get_config_by_name(config_name, config_path)
        return self.create_model(model_name, config)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        return self._model_info.copy()
    
    def list_models(self) -> None:
        """Print information about all available models."""
        print("Available Models:")
        print("=" * 50)
        for name, info in self._model_info.items():
            description = info.get('description', 'No description available')
            print(f"{name.upper():<10}: {description}")
        print()
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in the registry."""
        return model_name.lower() in self._models
    
    def _register_builtin_models(self):
        """Register built-in models."""
        try:
            # Import and register GCN
            from .gcn import GCN, GCNWithSkipConnections
            self.register(
                'gcn', 
                GCN, 
                'Graph Convolutional Network',
                {'num_layers': 2, 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 0.01}
            )
            self.register(
                'gcn_skip', 
                GCNWithSkipConnections, 
                'GCN with Skip Connections',
                {'num_layers': 3, 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 0.01}
            )
        except ImportError:
            pass
        
        try:
            # Import and register GAT
            from .gat import GAT, GATWithSkipConnections
            self.register(
                'gat', 
                GAT, 
                'Graph Attention Network',
                {'num_layers': 2, 'dropout': 0.6, 'heads': 8, 'optimizer': 'adam', 'learning_rate': 0.005}
            )
            self.register(
                'gat_skip', 
                GATWithSkipConnections, 
                'GAT with Skip Connections',
                {'num_layers': 3, 'dropout': 0.6, 'heads': 8, 'optimizer': 'adam', 'learning_rate': 0.005}
            )
        except ImportError:
            pass

        try:
            # Import and register RGCN
            from .rgcn import RGCN
            self.register(
                'rgcn',
                RGCN,
                'Relational Graph Convolutional Network',
                {'num_layers': 2, 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 0.01}
            )
        except ImportError:
            pass

        try:
            # Import and register TGCN
            from .tgcn import TGCN
            self.register(
                'tgcn',
                TGCN,
                'Temporal Graph Convolutional Network',
                {'num_layers': 2, 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 0.01}
            )
        except ImportError:
            pass
        
        try:
            # Import and register STGNN
            from .stgnn import STGNN
            self.register(
                'stgnn',
                STGNN,
                'Spatio-Temporal Graph Neural Network (with CaFO support)',
                {'num_layers': 2, 'hidden_dim': 64, 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 0.001}
            )
        except ImportError:
            pass

        try:
            # Import and register BoxE
            from .boxe import BoxE
            self.register(
                'boxe',
                BoxE,
                'BoxE model (PyKEEN wrapper)',
                {'dataset': 'wn18rr', 'model_kwargs': {'embedding_dim': 50}}
            )
        except ImportError:
            pass

        try:
            # Import and register RGCNEncoderDecoder
            from .mpqe import RGCNEncoderDecoder
            self.register(
                'mpqe',
                RGCNEncoderDecoder,
                'MPQE for query reasoning',
                {
                    'embed_dim': 128,
                    'num_layers': 2, 
                    'dropout': 0.5, 
                    'optimizer': 'adam', 
                    'learning_rate': 0.01,
                    'readout': 'mp',
                    'scatter_op': 'add',
                    'shared_layers': True,
                    'adaptive': True,
                    'batch_size': 512,
                    'inter_weight': 0.005,
                    'path_weight': 0.01
                }
            )
        except ImportError:
            pass


# Global registry instance
_registry = ModelRegistry()


def register_model(name: str, model_class: Type[BaseGNN], 
                  description: str = "", default_config: Optional[Dict[str, Any]] = None):
    """
    Register a model in the global registry.
    
    Args:
        name: Model name
        model_class: Model class
        description: Model description
        default_config: Default configuration
    """
    _registry.register(name, model_class, description, default_config)


def create_model(model_name: str, config: Dict[str, Any]) -> BaseGNN:
    """
    Create a model using the global registry.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    return _registry.create_model(model_name, config)


def create_model_from_config(model_name: str, config_name: str, 
                           config_path: str = 'configs/gcn_configs.yaml') -> BaseGNN:
    """
    Create a model from a configuration file using the global registry.
    
    Args:
        model_name: Name of the model
        config_name: Name of the configuration
        config_path: Path to configuration file
        
    Returns:
        Model instance
    """
    return _registry.create_model_from_config_file(model_name, config_name, config_path)


def list_available_models() -> None:
    """List all available models."""
    _registry.list_models()


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get information about available models."""
    return _registry.get_available_models()


def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    return _registry.is_model_available(model_name)


# Convenience function for the most common use case
def get_model(model_name: str, config_name: str, 
              config_path: str = 'configs/gcn_configs.yaml') -> BaseGNN:
    """
    Convenience function to get a model by name and config.
    
    Args:
        model_name: Name of the model (e.g., 'gcn', 'gat')
        config_name: Name of the configuration in YAML
        config_path: Path to configuration file
        
    Returns:
        Model instance ready for training
        
    Example:
        >>> model = get_model('gat', 'basic_gat')
        >>> model = get_model('gcn', 'lightweight_gcn')
    """
    return create_model_from_config(model_name, config_name, config_path) 