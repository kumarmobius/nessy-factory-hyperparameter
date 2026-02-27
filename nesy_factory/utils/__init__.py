"""
Unified Utilities for NeSy Factory Models (GNN + CNN)

This package provides utility functions for both GNN/Language Models and CNN architectures,
including configuration handling, data creation, training utilities, and model management.
"""

from typing import Dict, Any, List, Optional
import importlib

# Package configuration
PACKAGE_CONFIG = {
    "gnn_available": False,
    "cnn_available": False,
    "current_mode": "auto"  # 'auto', 'gnn', or 'cnn'
}

# =============================================================================
# GNN UTILITIES (from original nesy_factory) - FIXED IMPORTS
# =============================================================================

_GNN_UTILS_AVAILABLE = False
try:
    from .utils import (
        # YAML CONFIGURATION UTILITIES
        load_yaml_config as load_yaml_config_gnn,
        get_config_by_name as get_config_by_name_gnn,
        update_config_for_data as update_config_for_data_gnn,
    )
    _GNN_UTILS_AVAILABLE = True
    
    # Try to import optional functions
    try:
        from .utils import (
            get_training_config,
            get_experiment_config,
            list_available_configs,
            get_model_config_names,
            setup_experiment_environment,
            set_random_seed,
            create_sample_graph,
            create_bipartite_graph,
            create_recommendation_graph,
            print_model_summary as print_model_summary_gnn,
            train_model_with_config,
            compare_models_performance,
            create_example_models_from_configs,
            print_config_comparison,
        )
    except ImportError as e:
        print(f"Warning: Some GNN utility functions not available: {e}")
        # Create dummy functions for missing ones
        def create_bipartite_graph(*args, **kwargs):
            print("create_bipartite_graph not available")
            return None
        def create_recommendation_graph(*args, **kwargs):
            print("create_recommendation_graph not available") 
            return None
        def get_training_config(*args, **kwargs):
            print("get_training_config not available")
            return {}
        def get_experiment_config(*args, **kwargs):
            print("get_experiment_config not available")
            return {}
        def list_available_configs(*args, **kwargs):
            print("list_available_configs not available")
            return []
        def get_model_config_names(*args, **kwargs):
            print("get_model_config_names not available")
            return []
        def setup_experiment_environment(*args, **kwargs):
            print("setup_experiment_environment not available")
            return {}
        def set_random_seed(*args, **kwargs):
            print("set_random_seed not available")
        def train_model_with_config(*args, **kwargs):
            print("train_model_with_config not available")
            return {}
        def compare_models_performance(*args, **kwargs):
            print("compare_models_performance not available")
            return {}
        def create_example_models_from_configs(*args, **kwargs):
            print("create_example_models_from_configs not available")
            return []
        def print_config_comparison(*args, **kwargs):
            print("print_config_comparison not available")
        
except ImportError as e:
    print(f"Warning: GNN utilities not available: {e}")
    # Create essential dummy functions
    def load_yaml_config_gnn(*args, **kwargs):
        print("GNN YAML config loader not available")
        return {}
    def get_config_by_name_gnn(*args, **kwargs):
        print("GNN config by name not available")
        return {}
    def update_config_for_data_gnn(*args, **kwargs):
        print("GNN config update not available")
        return kwargs[0] if args else {}
    # Create dummy functions for all expected GNN utilities
    def create_sample_graph(*args, **kwargs):
        print("create_sample_graph not available")
        return None
    def print_model_summary_gnn(*args, **kwargs):
        print("print_model_summary not available")

PACKAGE_CONFIG["gnn_available"] = _GNN_UTILS_AVAILABLE

# =============================================================================
# CNN UTILITIES (from nesy_factory_cnn)
# =============================================================================

_CNN_UTILS_AVAILABLE = False
try:
    from .utils_cnn import (
        # Configuration management
        load_yaml_config as load_yaml_config_cnn,
        save_yaml_config,
        get_config_by_name as get_config_by_name_cnn,
        update_config_for_data as update_config_for_data_cnn,
        
        # Data utilities
        create_sample_image_data,
        calculate_output_size,
        
        # Model utilities
        estimate_parameters,
        count_model_parameters,
        get_device,
        setup_training,
        create_optimizer,
        
        # Configuration helpers
        create_default_configs,
        save_default_configs,
        print_config_summary,
        validate_config,
        
        # Multi-architecture utilities
        get_architecture_comparison,
        estimate_architecture_parameters,
        validate_architecture_config,
        create_architecture_presets
    )
    _CNN_UTILS_AVAILABLE = True
    PACKAGE_CONFIG["cnn_available"] = True
except ImportError as e:
    print(f"Warning: CNN utilities not available: {e}")
    # Create dummy functions for CNN utilities
    def load_yaml_config_cnn(*args, **kwargs):
        print("CNN YAML config loader not available")
        return {}
    def save_yaml_config(*args, **kwargs):
        print("save_yaml_config not available")
    def get_config_by_name_cnn(*args, **kwargs):
        print("CNN config by name not available")
        return {}
    def update_config_for_data_cnn(*args, **kwargs):
        print("CNN config update not available")
        return kwargs[0] if args else {}
    def create_sample_image_data(*args, **kwargs):
        print("create_sample_image_data not available")
        return None, None
    def calculate_output_size(*args, **kwargs):
        print("calculate_output_size not available")
        return (0, 0)
    def estimate_parameters(*args, **kwargs):
        print("estimate_parameters not available")
        return 0
    def count_model_parameters(*args, **kwargs):
        print("count_model_parameters not available")
        return 0
    def get_device(*args, **kwargs):
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    def setup_training(*args, **kwargs):
        print("setup_training not available")
        return kwargs[0] if args else {}
    def create_optimizer(*args, **kwargs):
        print("create_optimizer not available")
        return None
    def create_default_configs(*args, **kwargs):
        print("create_default_configs not available")
        return {}
    def save_default_configs(*args, **kwargs):
        print("save_default_configs not available")
    def print_config_summary(*args, **kwargs):
        print("print_config_summary not available")
    def validate_config(*args, **kwargs):
        print("validate_config not available")
        return True, []
    def get_architecture_comparison(*args, **kwargs):
        print("get_architecture_comparison not available")
        return {}
    def estimate_architecture_parameters(*args, **kwargs):
        print("estimate_architecture_parameters not available")
        return 0
    def validate_architecture_config(*args, **kwargs):
        print("validate_architecture_config not available")
        return True, []
    def create_architecture_presets(*args, **kwargs):
        print("create_architecture_presets not available")
        return {}

try:
    from .config_validator import (
        ConfigValidator,
        validate_and_fix_config
    )
except ImportError:
    # Fallback if config_validator not available
    ConfigValidator = None
    validate_and_fix_config = None
    print("Warning: ConfigValidator not available")

# =============================================================================
# UNIFIED INTERFACE FUNCTIONS
# =============================================================================

def load_yaml_config(config_path: str, config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified YAML config loader that works with both GNN and CNN configurations.
    
    Args:
        config_path: Path to YAML configuration file
        config_type: Type of config ('gnn', 'cnn', or None for auto-detect)
        
    Returns:
        Configuration dictionary
    """
    if config_type == "gnn" and PACKAGE_CONFIG["gnn_available"]:
        return load_yaml_config_gnn(config_path)
    elif config_type == "cnn" and PACKAGE_CONFIG["cnn_available"]:
        return load_yaml_config_cnn(config_path)
    elif PACKAGE_CONFIG["gnn_available"]:
        return load_yaml_config_gnn(config_path)  # Default to GNN
    elif PACKAGE_CONFIG["cnn_available"]:
        return load_yaml_config_cnn(config_path)
    else:
        raise ImportError("No utilities available")

def get_config_by_name(config_name: str, config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified config getter that works with both GNN and CNN configurations.
    
    Args:
        config_name: Name of the configuration
        config_type: Type of config ('gnn', 'cnn', or None for auto-detect)
        
    Returns:
        Configuration dictionary
    """
    if config_type == "gnn" and PACKAGE_CONFIG["gnn_available"]:
        return get_config_by_name_gnn(config_name)
    elif config_type == "cnn" and PACKAGE_CONFIG["cnn_available"]:
        return get_config_by_name_cnn(config_name)
    elif PACKAGE_CONFIG["gnn_available"]:
        try:
            return get_config_by_name_gnn(config_name)
        except:
            if PACKAGE_CONFIG["cnn_available"]:
                return get_config_by_name_cnn(config_name)
            raise
    elif PACKAGE_CONFIG["cnn_available"]:
        return get_config_by_name_cnn(config_name)
    else:
        raise ImportError("No utilities available")

def update_config_for_data(config: Dict[str, Any], data, config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified config updater for data characteristics.
    
    Args:
        config: Original configuration
        data: Data object with attributes
        config_type: Type of config ('gnn', 'cnn', or None for auto-detect)
        
    Returns:
        Updated configuration
    """
    if config_type == "gnn" and PACKAGE_CONFIG["gnn_available"]:
        return update_config_for_data_gnn(config, data)
    elif config_type == "cnn" and PACKAGE_CONFIG["cnn_available"]:
        return update_config_for_data_cnn(config, data)
    elif PACKAGE_CONFIG["gnn_available"]:
        return update_config_for_data_gnn(config, data)  # Default to GNN
    elif PACKAGE_CONFIG["cnn_available"]:
        return update_config_for_data_cnn(config, data)
    else:
        raise ImportError("No utilities available")

def print_model_summary(model, config_type: Optional[str] = None):
    """
    Unified model summary printer.
    
    Args:
        model: Model instance
        config_type: Type of model ('gnn', 'cnn', or None for auto-detect)
    """
    if config_type == "gnn" and PACKAGE_CONFIG["gnn_available"]:
        return print_model_summary_gnn(model)
    elif config_type == "cnn" and PACKAGE_CONFIG["cnn_available"]:
        # Use parameter counting for CNN
        param_count = count_model_parameters(model)
        print(f"Model Parameters: {param_count:,}")
        return param_count
    else:
        # Auto-detect based on model type
        if hasattr(model, 'graph_layers') or hasattr(model, 'conv_layers'):  # Likely GNN
            if PACKAGE_CONFIG["gnn_available"]:
                return print_model_summary_gnn(model)
        else:  # Likely CNN
            if PACKAGE_CONFIG["cnn_available"]:
                param_count = count_model_parameters(model)
                print(f"Model Parameters: {param_count:,}")
                return param_count
        print("Cannot determine model type for summary")

# =============================================================================
# MODE MANAGEMENT
# =============================================================================

def set_mode(mode: str):
    """
    Set the current operating mode.
    
    Args:
        mode: Operating mode ('auto', 'gnn', 'cnn')
    """
    if mode in ['auto', 'gnn', 'cnn']:
        PACKAGE_CONFIG["current_mode"] = mode
    else:
        raise ValueError("Mode must be 'auto', 'gnn', or 'cnn'")

def get_mode() -> str:
    """
    Get current operating mode.
    
    Returns:
        Current mode string
    """
    return PACKAGE_CONFIG["current_mode"]

def get_available_backends() -> Dict[str, bool]:
    """
    Get available backends.
    
    Returns:
        Dictionary showing available backends
    """
    return {
        "gnn": PACKAGE_CONFIG["gnn_available"],
        "cnn": PACKAGE_CONFIG["cnn_available"]
    }

# =============================================================================
# COMBINED __all__ EXPORTS
# =============================================================================

__all__ = [
    # Unified functions
    "load_yaml_config",
    "get_config_by_name", 
    "update_config_for_data",
    "print_model_summary",
    
    # Mode management
    "set_mode",
    "get_mode",
    "get_available_backends",
    
    # Package info
    "PACKAGE_CONFIG",
]

# Add GNN-specific functions if available
if PACKAGE_CONFIG["gnn_available"]:
    __all__.extend([
        # GNN Configuration utilities
        "get_training_config",
        "get_experiment_config", 
        "list_available_configs",
        "get_model_config_names",
        "setup_experiment_environment",
        
        # GNN Reproducibility
        "set_random_seed",
        
        # GNN Data utilities
        "create_sample_graph",
        "create_bipartite_graph",
        "create_recommendation_graph",
        
        # GNN Training utilities
        "train_model_with_config",
        "compare_models_performance",
        
        # GNN Example utilities
        "create_example_models_from_configs",
        "print_config_comparison",
    ])

# Add CNN-specific functions if available  
if PACKAGE_CONFIG["cnn_available"]:
    __all__.extend([
        # CNN Configuration management
        "save_yaml_config",
        
        # CNN Data utilities
        "create_sample_image_data",
        "calculate_output_size",
        
        # CNN Model utilities
        "estimate_parameters",
        "count_model_parameters", 
        "get_device",
        "setup_training",
        "create_optimizer",
        
        # CNN Configuration helpers
        "create_default_configs",
        "save_default_configs",
        "print_config_summary",
        "validate_config",
        
        # CNN Multi-architecture utilities
        "get_architecture_comparison",
        "estimate_architecture_parameters", 
        "validate_architecture_config",
        "create_architecture_presets",
    ])

# Add config validator if available
if ConfigValidator is not None:
    __all__.extend([
        "ConfigValidator",
        "validate_and_fix_config",
    ])

# Package metadata
__version__ = "2.0.0-unified"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def demonstrate_capabilities():
    """Demonstrate available capabilities."""
    backends = get_available_backends()
    print("🎯 Unified Utilities Capabilities:")
    print("=" * 50)
    print(f"🧠 GNN Backend: {'✅ Available' if backends['gnn'] else '❌ Not Available'}")
    print(f"🖼️ CNN Backend: {'✅ Available' if backends['cnn'] else '❌ Not Available'}")
    print(f"🔧 Current Mode: {get_mode()}")
    
    if backends['gnn']:
        print("\n🧠 GNN Functions Available:")
        print("  - Graph data creation (create_sample_graph, etc.)")
        print("  - GNN model training utilities")
        print("  - Experiment configuration helpers")
        
    if backends['cnn']:
        print("\n🖼️ CNN Functions Available:")
        print("  - Image data utilities (create_sample_image_data)")
        print("  - Multi-architecture support")
        print("  - Advanced configuration validation")
        print("  - Parameter estimation and counting")

def create_sample_data(data_type: str = "graph", **kwargs):
    """
    Unified sample data creation.
    
    Args:
        data_type: Type of data ('graph', 'image')
        **kwargs: Additional parameters
        
    Returns:
        Sample data
    """
    if data_type == "graph" and PACKAGE_CONFIG["gnn_available"]:
        return create_sample_graph(**kwargs)
    elif data_type == "image" and PACKAGE_CONFIG["cnn_available"]:
        return create_sample_image_data(**kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

# Add to exports
__all__.extend(["demonstrate_capabilities", "create_sample_data"])

# Print info on import
def _print_welcome():
    """Print welcome message on import."""
    backends = get_available_backends()
    if any(backends.values()):
        print("✨ nesy_factory.utils imported successfully!")
        if backends['gnn'] and backends['cnn']:
            print("💡 Both GNN and CNN utilities available")
        elif backends['gnn']:
            print("💡 GNN utilities available")
        elif backends['cnn']:
            print("💡 CNN utilities available")
        print("💡 Use demonstrate_capabilities() to see all features")

# Uncomment for welcome message
# _print_welcome()

if __name__ == "__main__":
    demonstrate_capabilities()
