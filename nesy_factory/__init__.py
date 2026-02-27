"""
NeSy Factory - Neural Symbolic Factory Library

A comprehensive framework for GNNs, Language Models, and CNNs with unified interfaces
and configuration-driven workflows.

Main modules:
- GNNs: Core GNN model implementations
- CNNs: CNN model implementations and registry (NEW)
- utils: Utility functions for configuration, data handling, and training
- scripts: Example scripts and usage patterns
- model_initializer: Advanced model initialization and factory functions
- quick_start_examples: Ready-to-use examples for common scenarios

Architectures Available:
    • GNNs: GCN, GAT, RGCN, LightGCN, PinSAGE, Temporal GNN
    • Language Models: Transformer-based models
    • CNNs: BaseCNN, ResNet, EfficientNet, MobileNet, DenseNet, SimpleCNN (NEW)
"""

import os
import sys
from typing import Dict, Any, List

# Package version
__version__ = "2.0.0"
__author__ = "NeSy Factory Team"
__email__ = "contact@nesyfactory.com"

# Package metadata
__all__ = [
    '__version__',
    '__author__', 
    '__email__'
]

# Import subpackages with error handling
try:
    from . import models
    _MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: models subpackage not available: {e}")
    _MODELS_AVAILABLE = False

try:
    from . import CNNs  # NEW
    _CNNS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CNNs subpackage not available: {e}")
    _CNNS_AVAILABLE = False

try:
    from . import utils
    _UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: utils subpackage not available: {e}")
    _UTILS_AVAILABLE = False

try:
    from . import scripts
    _SCRIPTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: scripts subpackage not available: {e}")
    _SCRIPTS_AVAILABLE = False

# Import packages for organized access
from nesy_factory import scripts
try:
    from nesy_factory import VAE
    _VAE_AVAILABLE = True
except ImportError:
    _VAE_AVAILABLE = False

# =============================================================================
# GNN IMPORTS (from original nesy_factory) - FIXED CIRCULAR IMPORTS
# =============================================================================

# Core GNN models and registry
_GNN_AVAILABLE = False
try:
    from .GNNs import (  # Base class; Model classes; Registry functions - most commonly used; Convenience functions
        GAT,
        GCN,
        RGCN,
        BaseGNN,
        GATWithSkipConnections,
        GCNWithSkipConnections,
        LightGCN,
        LightGCNWithFeatures,
        PinSAGE,
        create_gat,
        create_gat_with_skip_connections,
        create_gcn,
        create_gcn_with_skip_connections,
        create_lightgcn,
        create_lightgcn_with_features,
        create_model,
        create_model_from_config,
        create_pinsage,
        get_available_models,
        get_model,
        is_model_available,
        list_available_models,
        register_model,
    )
    _GNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GNN subpackage not available: {e}")
    _GNN_AVAILABLE = False
    # Create dummy values to avoid import errors
    GAT = GCN = RGCN = BaseGNN = None
    get_available_models = list_available_models = lambda: {}

# Import temporal models if available
try:
    from .GNNs import TGCN
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False
    TGCN = None

# Advanced model initialization - import only if needed
_INITIALIZER_AVAILABLE = False
try:
    from .scripts.model_initializer import (  # Primary initialization functions; Utility functions
        get_model_info,
        initialize_model_for_task,
        initialize_model_from_config_dict,
        initialize_model_from_yaml,
        initialize_model_with_custom_config,
        initialize_multiple_models,
        list_all_available_options,
    )
    _INITIALIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model initializer not available: {e}")
    _INITIALIZER_AVAILABLE = False

# Utility functions - import only if available
_UTILS_FUNCTIONS_AVAILABLE = False
try:
    from .utils import (  # Configuration utilities; Data utilities; Training utilities
        compare_models_performance,
        create_sample_graph,
        get_config_by_name,
        get_experiment_config,
        get_training_config,
        load_yaml_config,
        print_model_summary,
        train_model_with_config,
        update_config_for_data,
    )
    _UTILS_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some utility functions not available: {e}")
    _UTILS_FUNCTIONS_AVAILABLE = False
    # Create dummy functions
    def compare_models_performance(*args, **kwargs):
        print("compare_models_performance not available")
        return {}
    def create_sample_graph(*args, **kwargs):
        print("create_sample_graph not available")
        return None
    def get_config_by_name(*args, **kwargs):
        print("get_config_by_name not available")
        return {}
    def get_experiment_config(*args, **kwargs):
        print("get_experiment_config not available")
        return {}
    def get_training_config(*args, **kwargs):
        print("get_training_config not available")
        return {}
    def load_yaml_config(*args, **kwargs):
        print("load_yaml_config not available")
        return {}
    def print_model_summary(*args, **kwargs):
        print("print_model_summary not available")
    def train_model_with_config(*args, **kwargs):
        print("train_model_with_config not available")
        return {}
    def update_config_for_data(*args, **kwargs):
        print("update_config_for_data not available")
        return args[0] if args else {}

# =============================================================================
# CNN IMPORTS (from nesy_factory_cnn)
# =============================================================================

if _CNNS_AVAILABLE:
    # Core CNN imports
    try:
        from .CNNs import BaseCNN
        __all__.append('BaseCNN')
    except ImportError as e:
        print(f"Warning: BaseCNN not available: {e}")

    # CNN Registry functions
    try:
        from .CNNs.registry import (
            get_available_models as get_available_cnn_models,
            list_available_models as list_available_cnn_models,
            is_model_available as is_cnn_model_available
        )
        __all__.extend([
            'get_available_cnn_models',
            'list_available_cnn_models', 
            'is_cnn_model_available'
        ])
    except ImportError as e:
        print(f"Warning: CNN Registry functions not available: {e}")

    # CNN Factory
    try:
        from .CNNs.factory import CNNFactory
        __all__.append('CNNFactory')
    except ImportError as e:
        print(f"Warning: CNNFactory not available: {e}")

    # CNN Utility imports
    if _UTILS_AVAILABLE:
        try:
            from .utils import (
                create_default_configs, 
                get_device,
                setup_training,
                create_optimizer
            )
            __all__.extend([
                'create_default_configs', 
                'get_device',
                'setup_training', 
                'create_optimizer'
            ])
        except ImportError as e:
            print(f"Warning: Some CNN utils not available: {e}")

# =============================================================================
# COMBINED __all__ EXPORTS
# =============================================================================

__all__.extend([
    # Core model creation (most commonly used)
    "get_model",
    "create_model",
    "create_model_from_config",
    # GNN Model classes
    "BaseGNN",
    "GCN",
    "GCNWithSkipConnections",
    "GAT",
    "GATWithSkipConnections",
    "RGCN",
    "LightGCN",
    "LightGCNWithFeatures",
    "PinSAGE",
    # GNN Registry functions
    "list_available_models",
    "get_available_models",
    "is_model_available",
    "register_model",
    # GNN Convenience creation functions
    "create_gcn",
    "create_gcn_with_skip_connections",
    "create_gat",
    "create_gat_with_skip_connections",
    "create_lightgcn",
    "create_lightgcn_with_features",
    "create_pinsage",
    # Configuration utilities
    "load_yaml_config",
    "get_config_by_name",
    "update_config_for_data",
    "get_training_config",
    "get_experiment_config",
    # Data utilities
    "create_sample_graph",
    "print_model_summary",
    # Training utilities
    "train_model_with_config",
    "compare_models_performance",
    # Advanced initialization
    "initialize_model_from_yaml",
    "initialize_model_from_config_dict",
    "initialize_model_with_custom_config",
    "initialize_multiple_models",
    "initialize_model_for_task",
    # Information functions
    "list_all_available_options",
    "get_model_info",
])

# Add packages to __all__
__all__.extend(["scripts"])
if _VAE_AVAILABLE:
    __all__.append("VAE")

# Add temporal models if available
if _TEMPORAL_AVAILABLE:
    __all__.append("TGCN")

# =============================================================================
# COMBINED CONVENIENCE FUNCTIONS
# =============================================================================

def quick_start(model_type: str = "gnn"):
    """
    Display quick start information for the Factory.
    
    Args:
        model_type: Type of model ('gnn', 'cnn', or 'all')
    """
    print("🚀 NeSy Factory - Quick Start")
    print("=" * 50)
    print()
    
    if model_type in ["gnn", "all"] and _GNN_AVAILABLE:
        print("🧠 GNN Usage:")
        print("1. Create a model from YAML config:")
        print("   from nesy_factory import get_model")
        print("   model = get_model('gcn', 'basic_gcn')")
        print()
        print("2. Create a custom GNN model:")
        print("   from nesy_factory import create_model")
        print("   config = {'input_dim': 128, 'hidden_dim': 64, 'output_dim': 10}")
        print("   model = create_model('gcn', config)")
        print()
    
    if model_type in ["cnn", "all"] and _CNNS_AVAILABLE:
        print("🖼️ CNN Usage:")
        print("1. Create a CNN model:")
        print("   from nesy_factory import BaseCNN")
        print("   from nesy_factory.utils import create_default_configs")
        print("   config = create_default_configs()['mnist_basic']")
        print("   model = BaseCNN(config)")
        print()
        print("2. Use CNN Factory for multiple architectures:")
        print("   from nesy_factory.CNNs.factory import CNNFactory")
        print("   model = CNNFactory.create_model('resnet', config)")
        print()
    
    print("3. List available models:")
    if _GNN_AVAILABLE:
        print("   from nesy_factory import list_available_models")
        print("   list_available_models()")
        print()
    if _CNNS_AVAILABLE:
        print("   from nesy_factory import list_available_cnn_models")
        print("   list_available_cnn_models()")
        print()
    
    print("For more examples, run:")
    print("   python quick_start_examples.py")
    if _CNNS_AVAILABLE:
        print("   python quick_start_examples_cnn.py")


def show_available():
    """
    Show all available models and configurations.
    """
    print("📋 Available Models and Configurations")
    print("=" * 50)
    
    if _GNN_AVAILABLE:
        print("🧠 Graph Neural Networks:")
        list_available_models()
        print()
    
    if _CNNS_AVAILABLE:
        print("🖼️ Convolutional Neural Networks:")
        list_available_cnn_models()
        print()
    
    print("💡 For detailed information, run:")
    print("   python list_available_options.py")
    if _CNNS_AVAILABLE:
        print("   python list_available_options_cnn.py")


def run_examples():
    """
    Information on how to run examples.
    """
    print("📖 Running Examples")
    print("=" * 30)
    print()
    
    if _GNN_AVAILABLE:
        print("🧠 GNN Examples:")
        print("   Quick start: python quick_start_examples.py")
        print("   Model-specific:")
        print("     python examples/example_gcn.py")
        print("     python examples/example_light_gcn.py")
        print("     python examples/example_pinsage.py")
        print("     python examples/example_tgcn.py")
        print()
    
    if _CNNS_AVAILABLE:
        print("🖼️ CNN Examples:")
        print("   Quick start: python quick_start_examples_cnn.py")
        print("   Architecture comparison: python architecture_selector.py")
        print()
    
    print("List all available options:")
    print("   python list_available_options.py")
    if _CNNS_AVAILABLE:
        print("   python list_available_options_cnn.py")


# GNN Convenience functions (from original)
def create_basic_gcn(input_dim: int, output_dim: int, **kwargs):
    """Quick function to create a basic GCN model."""
    if not _GNN_AVAILABLE:
        print("GNN models not available")
        return None
    config = {
        "input_dim": input_dim,
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "output_dim": output_dim,
        "num_layers": kwargs.get("num_layers", 2),
        "dropout": kwargs.get("dropout", 0.5),
        "optimizer": kwargs.get("optimizer", "adam"),
        "learning_rate": kwargs.get("learning_rate", 0.01),
        **kwargs,
    }
    return create_model("gcn", config)


def create_basic_gat(input_dim: int, output_dim: int, **kwargs):
    """Quick function to create a basic GAT model."""
    if not _GNN_AVAILABLE:
        print("GNN models not available")
        return None
    config = {
        "input_dim": input_dim,
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "output_dim": output_dim,
        "num_layers": kwargs.get("num_layers", 2),
        "dropout": kwargs.get("dropout", 0.6),
        "heads": kwargs.get("heads", 8),
        "optimizer": kwargs.get("optimizer", "adam"),
        "learning_rate": kwargs.get("learning_rate", 0.005),
        **kwargs,
    }
    return create_model("gat", config)


# CNN Convenience functions (from CNN version)
def cnn_quick_start(architecture: str = "BaseCNN"):
    """
    Quick start guide for CNN models.
    
    Args:
        architecture: Architecture to demonstrate ('BaseCNN', 'ResNet', 'EfficientNet', etc.)
    """
    if not _CNNS_AVAILABLE:
        print("❌ CNN subpackage not available")
        return
    
    print(f"📷 CNN Quick Start - {architecture}")
    print("=" * 60)
    
    if architecture.lower() == "basecnn":
        print("from nesy_factory import BaseCNN")
        print("from nesy_factory.utils import create_default_configs")
        print("")
        print("# Create configuration")
        print("config = create_default_configs()['mnist_basic']")
        print("")
        print("# Create custom model")
        print("class MyCNN(BaseCNN):")
        print("    def forward(self, x):")
        print("        for conv_block in self.conv_blocks:")
        print("            x = conv_block(x)")
        print("            x = self.pool(x)")
        print("        x = x.view(x.size(0), -1)")
        print("        return self.classifier(x)")
        print("")
        print("model = MyCNN(config)")
    
    else:
        print("from nesy_factory.CNNs.factory import CNNFactory")
        print("from nesy_factory.scripts.model_initializer_cnn import create_ui_analysis_config")
        print("")
        print("# Create configuration")
        print(f"config = create_ui_analysis_config(architecture='{architecture}')")
        print("")
        print("# Create model using factory")
        print(f"model = CNNFactory.create_model('{architecture.lower()}', config)")
        print("")
        print(f"print(f'✅ Created {architecture} model with {{model.get_num_parameters():,}} parameters')")
    
    print("")
    print("💡 Use list_available_options_cnn.py to see all architectures and configurations")


def get_available_architectures() -> List[str]:
    """Get list of available CNN architectures."""
    if not _CNNS_AVAILABLE:
        return []
    
    try:
        from .CNNs.registry import get_available_models
        return list(get_available_models().keys())
    except ImportError:
        return ['BaseCNN']  # Fallback


# Add all convenience functions to __all__
__all__.extend([
    "quick_start",
    "show_available",
    "run_examples",
    "create_basic_gcn",
    "create_basic_gat",
    "cnn_quick_start",
    "get_available_architectures",
])

# Package information
def get_package_info():
    """Get information about the package."""
    info = {
        "name": "NeSy Factory",
        "version": __version__,
        "description": "Unified framework for GNNs, Language Models, and CNNs",
        "gnn_available": _GNN_AVAILABLE,
        "cnn_available": _CNNS_AVAILABLE,
        "vae_available": _VAE_AVAILABLE,
        "temporal_available": _TEMPORAL_AVAILABLE,
    }
    
    if _GNN_AVAILABLE:
        info["gnn_models"] = list(get_available_models().keys())
    
    if _CNNS_AVAILABLE:
        info["cnn_architectures"] = get_available_architectures()
    
    return info


def print_package_info():
    """Print comprehensive information about the package."""
    info = get_package_info()
    
    print("🚀 NeSy Factory Package Info")
    print("=" * 60)
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    
    if info['gnn_available']:
        print(f"GNN Models: {', '.join(info['gnn_models'])}")
    else:
        print("GNN Models: Not available")
    
    if info['cnn_available']:
        print(f"CNN Architectures: {', '.join(info['cnn_architectures'])}")
    else:
        print("CNN Architectures: Not available")
    
    print(f"VAE Available: {info['vae_available']}")
    print(f"Temporal Models: {info['temporal_available']}")
    print()
    print("📖 Quick Start:")
    if info['gnn_available']:
        print("  GNN: quick_start('gnn')")
    if info['cnn_available']:
        print("  CNN: quick_start('cnn')")
    print("  All: quick_start('all')")
    print("=" * 60)


# Add to exports
__all__.extend(['get_package_info', 'print_package_info'])

# Print info on import (optional)
def _print_welcome():
    """Print welcome message on import."""
    print("✨ nesy_factory imported successfully!")
    print("💡 Use print_package_info() to see available features")
    print("💡 Use quick_start() for usage examples")

# Uncomment for welcome message
# _print_welcome()
