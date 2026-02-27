"""
Model Initializer for GNN Factory
Comprehensive examples of how to initialize models by name and config.
"""

import torch
from typing import Dict, Any, Optional, List
from ..GNNs import (
    create_model, create_model_from_config, get_model, 
    list_available_models, get_available_models, is_model_available
)
from ..utils import get_config_by_name, update_config_for_data, load_yaml_config

# ============================================================================
# METHOD 1: EASIEST - Use get_model() with YAML configs
# ============================================================================

def initialize_model_from_yaml(model_name: str, config_name: str, data=None, config_path: str = 'configs/gcn_configs.yaml'):
    """
    Easiest way to initialize a model using YAML configurations.
    
    Args:
        model_name: Name of the model ('gcn', 'gat', 'rgcn', 'tgcn', 'lightgcn', 'pinsage')
        config_name: Name of the configuration in YAML ('basic_gcn', 'lightweight_gcn', etc.)
        data: Optional data object to update input dimensions
        config_path: Path to YAML config file
        
    Returns:
        Initialized model instance
        
    Example:
        >>> model = initialize_model_from_yaml('gcn', 'basic_gcn')
        >>> model = initialize_model_from_yaml('gat', 'basic_gat')
    """
    if not is_model_available(model_name):
        available = list(get_available_models().keys())
        raise ValueError(f"Model '{model_name}' not available. Available models: {available}")
    
    # Get the configuration and optionally update with data dimensions
    config = get_config_by_name(config_name, config_path)
    if data is not None:
        config = update_config_for_data(config, data)
    
    # Use the convenience function
    model = create_model_from_config(model_name, config_name, config_path)
    
    print(f"✅ Created {model_name.upper()} model with '{config_name}' configuration")
    print(f"   Parameters: {model.get_num_parameters()}")
    return model

# ============================================================================
# METHOD 2: DIRECT CONFIG - Use create_model() with config dict
# ============================================================================

def initialize_model_from_config_dict(model_name: str, config: Dict[str, Any]):
    """
    Initialize model directly from a configuration dictionary.
    
    Args:
        model_name: Name of the model to create
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized model instance
        
    Example:
        >>> config = {
        ...     'input_dim': 10,
        ...     'hidden_dim': 64,
        ...     'output_dim': 3,
        ...     'num_layers': 2,
        ...     'dropout': 0.5,
        ...     'optimizer': 'adam',
        ...     'learning_rate': 0.01
        ... }
        >>> model = initialize_model_from_config_dict('gcn', config)
    """
    if not is_model_available(model_name):
        available = list(get_available_models().keys())
        raise ValueError(f"Model '{model_name}' not available. Available models: {available}")
    
    model = create_model(model_name, config)
    
    print(f"✅ Created {model_name.upper()} model from config dictionary")
    print(f"   Parameters: {model.get_num_parameters()}")
    return model

# ============================================================================
# METHOD 3: CUSTOM CONFIGS - Modify existing YAML configs
# ============================================================================

def initialize_model_with_custom_config(model_name: str, base_config_name: str, 
                                       custom_params: Dict[str, Any], data=None):
    """
    Initialize model by modifying an existing YAML configuration.
    
    Args:
        model_name: Name of the model to create
        base_config_name: Base configuration to start from
        custom_params: Dictionary of parameters to override
        data: Optional data object to update input dimensions
        
    Returns:
        Initialized model instance
        
    Example:
        >>> custom_params = {'dropout': 0.8, 'learning_rate': 0.001}
        >>> model = initialize_model_with_custom_config('gcn', 'basic_gcn', custom_params)
    """
    # Load base configuration
    config = get_config_by_name(base_config_name)
    
    # Update with custom parameters
    config.update(custom_params)
    
    # Update with data dimensions if provided
    if data is not None:
        config = update_config_for_data(config, data)
    
    model = create_model(model_name, config)
    
    print(f"✅ Created {model_name.upper()} model with custom configuration")
    print(f"   Base config: {base_config_name}")
    print(f"   Custom params: {custom_params}")
    print(f"   Parameters: {model.get_num_parameters()}")
    return model

# ============================================================================
# METHOD 4: BATCH INITIALIZATION - Multiple models at once
# ============================================================================

def initialize_multiple_models(model_specs: List[Dict[str, Any]], data=None):
    """
    Initialize multiple models at once for comparison/ensemble.
    
    Args:
        model_specs: List of dictionaries with 'model_name' and 'config_name' or 'config'
        data: Optional data object to update input dimensions
        
    Returns:
        Dictionary of initialized models
        
    Example:
        >>> specs = [
        ...     {'model_name': 'gcn', 'config_name': 'basic_gcn'},
        ...     {'model_name': 'gat', 'config_name': 'basic_gat'},
        ...     {'model_name': 'gcn', 'config': {'input_dim': 10, 'hidden_dim': 32, 'output_dim': 3}}
        ... ]
        >>> models = initialize_multiple_models(specs)
    """
    models = {}
    
    for i, spec in enumerate(model_specs):
        model_name = spec['model_name']
        
        if 'config_name' in spec:
            # Initialize from YAML config
            config_name = spec['config_name']
            model = initialize_model_from_yaml(model_name, config_name, data)
            model_key = f"{model_name}_{config_name}"
        elif 'config' in spec:
            # Initialize from config dict
            config = spec['config'].copy()
            if data is not None:
                config = update_config_for_data(config, data)
            model = initialize_model_from_config_dict(model_name, config)
            model_key = f"{model_name}_{i}"
        else:
            raise ValueError(f"Model spec {i} must contain either 'config_name' or 'config'")
        
        models[model_key] = model
    
    print(f"✅ Created {len(models)} models: {list(models.keys())}")
    return models

# ============================================================================
# METHOD 5: AUTO-DETECTION - Smart initialization based on task
# ============================================================================

def initialize_model_for_task(task_type: str, dataset_info: Dict[str, Any], 
                             performance_priority: str = 'balanced'):
    """
    Automatically select and initialize a model based on task requirements.
    
    Args:
        task_type: Type of task ('node_classification', 'graph_classification', 'link_prediction')
        dataset_info: Dictionary with dataset information (num_nodes, num_features, num_classes, etc.)
        performance_priority: 'speed', 'accuracy', or 'balanced'
        
    Returns:
        Initialized model instance with suitable configuration
        
    Example:
        >>> dataset_info = {'num_features': 128, 'num_classes': 7, 'num_nodes': 2708}
        >>> model = initialize_model_for_task('node_classification', dataset_info, 'accuracy')
    """
    # Model selection based on task and priority
    if task_type == 'node_classification':
        if performance_priority == 'speed':
            model_name, config_name = 'gcn', 'lightweight_gcn'
        elif performance_priority == 'accuracy':
            model_name, config_name = 'gat', 'basic_gat'  # Assuming GAT config exists
        else:  # balanced
            model_name, config_name = 'gcn', 'basic_gcn'
    elif task_type == 'graph_classification':
        model_name, config_name = 'gcn', 'heavy_gcn'
    elif task_type == 'link_prediction':
        model_name, config_name = 'gcn', 'expanding_gcn'
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Load and customize configuration
    config = get_config_by_name(config_name)
    config['input_dim'] = dataset_info.get('num_features', config['input_dim'])
    config['output_dim'] = dataset_info.get('num_classes', config['output_dim'])
    
    # Adjust architecture based on dataset size
    if dataset_info.get('num_nodes', 0) > 10000:
        config['dropout'] = min(config.get('dropout', 0.5) + 0.1, 0.8)  # Higher dropout for larger graphs
    
    model = create_model(model_name, config)
    
    print(f"✅ Auto-selected {model_name.upper()} for {task_type}")
    print(f"   Configuration: {config_name} (priority: {performance_priority})")
    print(f"   Parameters: {model.get_num_parameters()}")
    return model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_all_available_options():
    """Display all available models and configurations."""
    print("🔍 Available Models and Configurations")
    print("=" * 50)
    
    # Show available models
    print("\n📦 Available Models:")
    list_available_models()
    
    # Show available configurations
    print("⚙️  Available Configurations:")
    try:
        configs = load_yaml_config()
        config_names = [k for k in configs.keys() if not k.startswith(('training', 'experiment'))]
        for config_name in config_names:
            config = configs[config_name]
            hidden_dim = config.get('hidden_dim', 'N/A')
            layers = config.get('num_layers', 'N/A')
            print(f"  {config_name:<20}: {layers} layers, {hidden_dim} hidden dims")
    except Exception as e:
        print(f"  Error loading configs: {e}")

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    if not is_model_available(model_name):
        return {"error": f"Model '{model_name}' not available"}
    
    models_info = get_available_models()
    return models_info.get(model_name.lower(), {})

# ============================================================================
# EXAMPLES AND DEMONSTRATIONS
# ============================================================================

def demo_all_initialization_methods():
    """Demonstrate all initialization methods."""
    print("🚀 GNN Model Initialization Demo")
    print("=" * 50)
    
    # First, show what's available
    list_all_available_options()
    
    print("\n" + "="*50)
    print("DEMONSTRATION OF INITIALIZATION METHODS")
    print("="*50)
    
    # Method 1: YAML-based initialization
    print("\n1️⃣  YAML-based initialization:")
    try:
        model1 = initialize_model_from_yaml('gcn', 'basic_gcn')
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: Config dictionary
    print("\n2️⃣  Config dictionary initialization:")
    config_dict = {
        'input_dim': 10,
        'hidden_dim': 64,
        'output_dim': 3,
        'num_layers': 2,
        'dropout': 0.5,
        'optimizer': 'adam',
        'learning_rate': 0.01
    }
    try:
        model2 = initialize_model_from_config_dict('gcn', config_dict)
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Custom configuration
    print("\n3️⃣  Custom configuration:")
    try:
        custom_params = {'dropout': 0.8, 'learning_rate': 0.001}
        model3 = initialize_model_with_custom_config('gcn', 'basic_gcn', custom_params)
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 4: Multiple models
    print("\n4️⃣  Multiple models initialization:")
    try:
        model_specs = [
            {'model_name': 'gcn', 'config_name': 'basic_gcn'},
            {'model_name': 'gcn', 'config_name': 'lightweight_gcn'},
        ]
        models = initialize_multiple_models(model_specs)
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 5: Auto-detection
    print("\n5️⃣  Auto-detection for task:")
    try:
        dataset_info = {'num_features': 128, 'num_classes': 7, 'num_nodes': 2708}
        model5 = initialize_model_for_task('node_classification', dataset_info, 'balanced')
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    demo_all_initialization_methods() 