"""
Model registry for CNN Factory.

Provides dynamic registration and creation of CNN architectures.
"""

from typing import Dict, Any, Type, Optional, Callable
from .base import BaseCNN

# Global registry for all CNN models
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_model(
    name: str, 
    model_class: Type[BaseCNN], 
    description: str = "",
    tags: Optional[list] = None
) -> None:
    """
    Register a CNN model class in the global registry.
    
    Args:
        name: Model name (e.g., 'resnet', 'efficientnet')
        model_class: The model class to register
        description: Brief description of the model
        tags: Optional tags for categorization
    """
    if name in _MODEL_REGISTRY:
        print(f"Warning: Model '{name}' is already registered. Overwriting.")
    
    _MODEL_REGISTRY[name] = {
        'class': model_class,
        'description': description,
        'tags': tags or [],
        'module': model_class.__module__
    }

def create_model(name: str, config: Dict[str, Any]) -> BaseCNN:
    """
    Create a model instance by name.
    
    Args:
        name: Model name from registry
        config: Configuration dictionary
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name is not registered
    """
    if name not in _MODEL_REGISTRY:
        available_models = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available_models}"
        )
    
    model_info = _MODEL_REGISTRY[name]
    model_class = model_info['class']
    
    try:
        return model_class(config)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{name}': {e}")

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered models with their information.
    
    Returns:
        Dictionary of model names to model info
    """
    return _MODEL_REGISTRY.copy()

def list_available_models() -> None:
    """Print all available models in a formatted way."""
    if not _MODEL_REGISTRY:
        print("❌ No models registered in the registry.")
        return
    
    print("🏗️  AVAILABLE CNN MODELS")
    print("=" * 80)
    print(f"{'Name':<15} {'Description':<40} {'Tags':<20}")
    print("-" * 80)
    
    for name, info in _MODEL_REGISTRY.items():
        desc = info['description'][:38] + "..." if len(info['description']) > 38 else info['description']
        tags = ", ".join(info['tags'][:2])  # Show first 2 tags
        if len(info['tags']) > 2:
            tags += "..."
        print(f"{name:<15} {desc:<40} {tags:<20}")

def is_model_available(name: str) -> bool:
    """
    Check if a model is available in the registry.
    
    Args:
        name: Model name to check
        
    Returns:
        True if model is registered
    """
    return name in _MODEL_REGISTRY

def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        name: Model name
        
    Returns:
        Model information dictionary
        
    Raises:
        ValueError: If model is not registered
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    
    return _MODEL_REGISTRY[name].copy()

def unregister_model(name: str) -> bool:
    """
    Remove a model from the registry.
    
    Args:
        name: Model name to remove
        
    Returns:
        True if model was removed, False if not found
    """
    if name in _MODEL_REGISTRY:
        del _MODEL_REGISTRY[name]
        print(f"🗑️  Unregistered model: {name}")
        return True
    return False

def get_models_by_tag(tag: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all models that have a specific tag.
    
    Args:
        tag: Tag to filter by
        
    Returns:
        Dictionary of matching models
    """
    return {
        name: info for name, info in _MODEL_REGISTRY.items()
        if tag in info.get('tags', [])
    }

def clear_registry() -> None:
    """Clear all models from the registry."""
    _MODEL_REGISTRY.clear()
    print("🧹 Cleared all models from registry.")

# Pre-register common model tags for categorization
MODEL_TAGS = {
    'RESNET': 'resnet',
    'EFFICIENT': 'efficient',
    'MOBILE': 'mobile',
    'LIGHTWEIGHT': 'lightweight',
    'HEAVY': 'heavy',
    'PRETRAINED': 'pretrained_available',
    'CUSTOM': 'custom',
    'PRODUCTION': 'production',
    'RESEARCH': 'research',
    'EDUCATIONAL': 'educational',
    'FLEXIBLE': 'flexible',
    'PROTOTYPING': 'prototyping',
    'BEGINNER_FRIENDLY': 'beginner_friendly',
    'FAST': 'fast',
    'SIMPLE': 'simple'
}

def register_model_with_tags(
    name: str,
    model_class: Type[BaseCNN],
    description: str = "",
    tags: Optional[list] = None
) -> None:
    """
    Register a model with automatic tag categorization.
    
    Args:
        name: Model name
        model_class: Model class
        description: Model description
        tags: Additional tags
    """
    # Auto-detect tags based on model name and characteristics
    auto_tags = []
    
    name_lower = name.lower()
    
    # Architecture-based tags
    if 'resnet' in name_lower:
        auto_tags.append(MODEL_TAGS['RESNET'])
    if 'efficient' in name_lower:
        auto_tags.append(MODEL_TAGS['EFFICIENT'])
    if 'mobile' in name_lower:
        auto_tags.append(MODEL_TAGS['MOBILE'])
        auto_tags.append(MODEL_TAGS['LIGHTWEIGHT'])
    if 'simple' in name_lower:
        auto_tags.append(MODEL_TAGS['SIMPLE'])
        auto_tags.append(MODEL_TAGS['BEGINNER_FRIENDLY'])
        auto_tags.append(MODEL_TAGS['EDUCATIONAL'])
    if 'base' in name_lower or 'custom' in name_lower:
        auto_tags.append(MODEL_TAGS['CUSTOM'])
        auto_tags.append(MODEL_TAGS['FLEXIBLE'])
        auto_tags.append(MODEL_TAGS['EDUCATIONAL'])
        auto_tags.append(MODEL_TAGS['PROTOTYPING'])
    if 'dense' in name_lower:
        auto_tags.append(MODEL_TAGS['EFFICIENT'])
    
    # Size-based tags (rough estimation)
    if any(size in name_lower for size in ['b0', 'b1', 'small', 'v2', '18', '34', 'simple']):
        auto_tags.append(MODEL_TAGS['LIGHTWEIGHT'])
        auto_tags.append(MODEL_TAGS['FAST'])
    if any(size in name_lower for size in ['b7', '152', '201', 'large', '101', '169']):
        auto_tags.append(MODEL_TAGS['HEAVY'])
    
    # Add pretrained availability tag for non-base models
    if 'base' not in name_lower and 'custom' not in name_lower and 'simple' not in name_lower:
        auto_tags.append(MODEL_TAGS['PRETRAINED'])
    
    # Combine with user-provided tags
    all_tags = list(set(auto_tags + (tags or [])))
    
    register_model(name, model_class, description, all_tags)

# Convenience function for batch registration
def register_models(model_dict: Dict[str, Type[BaseCNN]]) -> None:
    """
    Register multiple models at once.
    
    Args:
        model_dict: Dictionary of model names to model classes
    """
    for name, model_class in model_dict.items():
        # Generate description from class docstring
        description = model_class.__doc__.split('\n')[0] if model_class.__doc__ else ""
        register_model_with_tags(name, model_class, description)

# Export public API
__all__ = [
    'register_model',
    'create_model',
    'get_available_models',
    'list_available_models',
    'is_model_available',
    'get_model_info',
    'unregister_model',
    'get_models_by_tag',
    'clear_registry',
    'register_model_with_tags',
    'register_models',
    'MODEL_TAGS'
]

# =============================================================================
# AUTO-REGISTRATION OF ALL AVAILABLE MODELS
# =============================================================================

# 1. Register BaseCNN first (foundational model)
try:
    from .base import BaseCNN
    register_model_with_tags(
        'basecnn',
        BaseCNN,
        'Customizable CNN for general purposes and prototyping',
        ['customizable', 'prototyping', 'educational', 'flexible', 'general_purpose']
    )
    print("✅ Registered model: basecnn - Customizable CNN for general purposes and prototyping")
except ImportError as e:
    print(f"⚠️ Could not auto-register BaseCNN: {e}")
except Exception as e:
    print(f"⚠️ BaseCNN registration had issues: {e}")

# 2. Register ResNet
try:
    from .resnet import ResNet
    register_model_with_tags(
        'resnet',
        ResNet,
        'Residual Networks with skip connections for deep learning',
        ['proven', 'classification', 'deep_learning', 'transfer_learning']
    )
    print("✅ Registered model: resnet - Residual Networks with skip connections for deep learning")
except ImportError as e:
    print(f"⚠️ Could not auto-register ResNet: {e}")
except Exception as e:
    print(f"⚠️ ResNet registration had issues: {e}")

# 3. Register EfficientNet
try:
    from .efficientnet import EfficientNet
    register_model_with_tags(
        'efficientnet', 
        EfficientNet,
        'EfficientNet with compound scaling for optimal performance',
        ['modern', 'scalable', 'efficient', 'state_of_the_art']
    )
    print("✅ Registered model: efficientnet - EfficientNet with compound scaling for optimal performance")
except ImportError as e:
    print(f"⚠️ Could not auto-register EfficientNet: {e}")
except Exception as e:
    print(f"⚠️ EfficientNet registration had issues (downloads may fail): {e}")

# 4. Register MobileNet - FIXED: Proper registration
try:
    from .mobilenet import MobileNet
    register_model_with_tags(
        'mobilenet',
        MobileNet,
        'MobileNet for efficient mobile and edge vision tasks',
        ['lightweight', 'mobile', 'fast', 'edge_computing', 'pretrained_available']
    )
    print("✅ Registered model: mobilenet - MobileNet for efficient mobile and edge vision tasks")
except ImportError as e:
    print(f"⚠️ Could not auto-register MobileNet: {e}")
except Exception as e:
    print(f"⚠️ MobileNet registration had issues: {e}")

# 5. Register DenseNet
try:
    from .densenet import DenseNet, DenseNetCustom
    register_model_with_tags(
        'densenet',
        DenseNet,
        'DenseNet with dense connections for feature reuse',
        ['efficient', 'dense_connections', 'feature_reuse', 'parameter_efficient']
    )
    print("✅ Registered model: densenet - DenseNet with dense connections for feature reuse")
    
    register_model_with_tags(
        'densenet_custom',
        DenseNetCustom, 
        'Custom DenseNet with flexible configuration',
        ['flexible', 'customizable', 'dense_connections', 'configurable']
    )
    print("✅ Registered model: densenet_custom - Custom DenseNet with flexible configuration")
except ImportError as e:
    print(f"⚠️ Could not auto-register DenseNet: {e}")
except Exception as e:
    print(f"⚠️ DenseNet registration had issues: {e}")

# 6. Register SimpleCNN - FIXED: Proper registration for both variants
try:
    from .simple_cnn import SimpleCNN, SimpleCNNV2
    register_model_with_tags(
        'simple_cnn',
        SimpleCNN,
        'Simple CNN for basic tasks and educational purposes',
        ['lightweight', 'educational', 'fast', 'prototyping', 'beginner_friendly', 'simple']
    )
    print("✅ Registered model: simple_cnn - Simple CNN for basic tasks and educational purposes")
    
    register_model_with_tags(
        'simple_cnn_v2',  # FIXED: Use underscore instead of camelcase
        SimpleCNNV2,
        'Enhanced SimpleCNN with advanced features',
        ['lightweight', 'configurable', 'advanced', 'flexible', 'educational']
    )
    print("✅ Registered model: simple_cnn_v2 - Enhanced SimpleCNN with advanced features")
except ImportError as e:
    print(f"⚠️ Could not auto-register SimpleCNN: {e}")
except Exception as e:
    print(f"⚠️ SimpleCNN registration had issues: {e}")

# Demo function
def demo_registry():
    """Demonstrate the registry functionality."""
    print("\n🎯 CNN MODEL REGISTRY DEMO")
    print("=" * 50)
    
    # List available models
    list_available_models()
    
    # Show model counts
    available = get_available_models()
    print(f"\n📊 Total models registered: {len(available)}")
    
    # Show models by category
    categories = {
        'Simple Models': ['simple_cnn', 'simple_cnn_v2'],
        'Base Models': ['basecnn'],
        'Production Models': ['resnet', 'efficientnet', 'densenet'],
        'Mobile Models': ['mobilenet'],
        'Custom Models': ['densenet_custom']
    }
    
    print(f"\n📁 Model Categories:")
    for category, models in categories.items():
        available_models = [m for m in models if m in available]
        if available_models:
            print(f"  {category}: {', '.join(available_models)}")
    
    # Show models by tag
    print(f"\n🏷️  Models by Tag:")
    lightweight_models = get_models_by_tag('lightweight')
    if lightweight_models:
        print(f"  Lightweight: {list(lightweight_models.keys())}")
    
    educational_models = get_models_by_tag('educational')
    if educational_models:
        print(f"  Educational: {list(educational_models.keys())}")
    
    efficient_models = get_models_by_tag('efficient')
    if efficient_models:
        print(f"  Efficient: {list(efficient_models.keys())}")
    
    customizable_models = get_models_by_tag('customizable')
    if customizable_models:
        print(f"  Customizable: {list(customizable_models.keys())}")

# Print summary on import
def print_registry_summary():
    """Print a summary of the registry when imported."""
    available = get_available_models()
    if available:
        print(f"✨ CNN Model Registry loaded with {len(available)} architectures:")
        for name in sorted(available.keys()):
            info = available[name]
            tags = ", ".join(info['tags'][:3])
            print(f"   • {name}: {info['description'][:50]}... [{tags}]")
    else:
        print("❌ No models registered in CNN Model Registry")

# Run demo if this file is executed directly
if __name__ == "__main__":
    demo_registry()

# Print summary when module is imported
print_registry_summary()
