"""
Quick Start: Model Initialization Examples
Simple examples of how to initialize GNN models by name and config.
"""

from ..GNNs import create_model, get_model, list_available_models, is_model_available
from ..utils import get_config_by_name, create_sample_graph

# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

def quick_example_1_simplest():
    """Simplest way: Use existing YAML config"""
    print("🚀 Method 1: Simplest - Using YAML config")
    
    # This is the easiest way - uses pre-defined YAML configurations
    model = get_model('gcn', 'basic_gcn')
    print(f"✅ Created GCN model with {model.get_num_parameters()} parameters")
    return model

def quick_example_2_custom_config():
    """Quick way: Custom configuration dictionary"""
    print("\n🔧 Method 2: Custom configuration dictionary")
    
    # Define your own configuration
    config = {
        'input_dim': 128,      # Number of input features
        'hidden_dim': 64,      # Hidden layer dimensions
        'output_dim': 7,       # Number of output classes
        'num_layers': 3,       # Number of layers
        'dropout': 0.5,        # Dropout rate
        'optimizer': 'adam',   # Optimizer
        'learning_rate': 0.01  # Learning rate
    }
    
    model = create_model('gcn', config)
    print(f"✅ Created custom GCN model with {model.get_num_parameters()} parameters")
    return model

def quick_example_3_with_data():
    """Practical way: Initialize with actual data"""
    print("\n📊 Method 3: Initialize with actual graph data")
    
    # Create sample data
    data, train_mask, val_mask, test_mask = create_sample_graph()
    print(f"Sample graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.x.size(1)} features")
    
    # Load config and update with data dimensions
    config = get_config_by_name('basic_gcn')
    config['input_dim'] = data.x.size(1)    # Automatically set input features
    config['output_dim'] = data.y.max().item() + 1  # Set number of classes
    
    model = create_model('gcn', config)
    print(f"✅ Created GCN model adapted to data: {model.get_num_parameters()} parameters")
    return model, data

def quick_example_4_multiple_models():
    """Compare multiple models"""
    print("\n🏆 Method 4: Create multiple models for comparison")
    
    models = {}
    
    # Create different model architectures
    model_configs = [
        ('gcn', 'lightweight_gcn'),  # Fast model
        ('gcn', 'basic_gcn'),        # Standard model  
        ('gcn', 'heavy_gcn'),        # Powerful model
    ]
    
    for model_name, config_name in model_configs:
        if is_model_available(model_name):
            model = get_model(model_name, config_name)
            models[f"{model_name}_{config_name}"] = model
            print(f"✅ {config_name}: {model.get_num_parameters()} parameters")
    
    return models

def quick_example_5_best_practices():
    """Best practices for model initialization"""
    print("\n⭐ Method 5: Best practices with error handling")
    
    # Check if model is available first
    model_name = 'gcn'
    config_name = 'basic_gcn'
    
    if not is_model_available(model_name):
        print(f"❌ Model '{model_name}' not available")
        print("Available models:")
        list_available_models()
        return None
    
    try:
        # Load and customize configuration
        config = get_config_by_name(config_name)
        
        # Customize for your needs
        config.update({
            'dropout': 0.3,        # Lower dropout
            'learning_rate': 0.005 # Lower learning rate
        })
        
        model = create_model(model_name, config)
        print(f"✅ Successfully created {model_name.upper()} model")
        print(f"   Parameters: {model.get_num_parameters()}")
        print(f"   Configuration: {config}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

# ============================================================================
# COMMON PATTERNS
# ============================================================================

def create_model_by_name_and_config(model_name: str, config_name: str = None, **custom_params):
    """
    Universal function to create any model with optional customization.
    
    Args:
        model_name: Name of the model ('gcn', 'gat', 'rgcn', etc.)
        config_name: Name of YAML config (optional, defaults to basic config)
        **custom_params: Any parameters to override
        
    Returns:
        Initialized model
        
    Examples:
        >>> model = create_model_by_name_and_config('gcn')
        >>> model = create_model_by_name_and_config('gcn', 'lightweight_gcn')
        >>> model = create_model_by_name_and_config('gcn', 'basic_gcn', dropout=0.8)
    """
    # Default configs for each model
    default_configs = {
        'gcn': 'basic_gcn',
        'gat': 'basic_gat',  # Assuming this exists in GAT configs
        'rgcn': 'basic_rgcn',
        'tgcn': 'basic_tgcn',
        'lightgcn': 'basic_lightgcn',
        'pinsage': 'basic_pinsage'
    }
    
    # Use provided config or default
    if config_name is None:
        config_name = default_configs.get(model_name.lower(), 'basic_gcn')
    
    try:
        # Load base configuration
        config = get_config_by_name(config_name)
        
        # Apply custom parameters
        config.update(custom_params)
        
        # Create model
        model = create_model(model_name, config)
        
        print(f"✅ Created {model_name.upper()} model")
        if custom_params:
            print(f"   Custom parameters: {custom_params}")
        print(f"   Total parameters: {model.get_num_parameters()}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating {model_name} model: {e}")
        print("Available models:")
        list_available_models()
        return None

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all quick start examples"""
    print("🎯 GNN Model Factory - Quick Start Examples")
    print("=" * 60)
    
    # Show available models first
    print("📋 Available Models:")
    list_available_models()
    print()
    
    # Run examples
    quick_example_1_simplest()
    quick_example_2_custom_config()
    quick_example_3_with_data()
    quick_example_4_multiple_models()
    quick_example_5_best_practices()
    
    print("\n" + "=" * 60)
    print("🎉 Quick start examples completed!")
    print("💡 Use these patterns in your own code for model initialization.")

if __name__ == "__main__":
    main() 