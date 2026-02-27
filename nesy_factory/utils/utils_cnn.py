"""
Utility functions for CNN Factory with multi-architecture support.
"""

import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import os
import json


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")


def get_config_by_name(config_name: str, config_path: str = 'configs/cnn_configs.yaml') -> Dict[str, Any]:
    """
    Get specific configuration by name from YAML file.
    
    Args:
        config_name: Name of the configuration
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    configs = load_yaml_config(config_path)
    return configs.get(config_name, {})


def update_config_for_data(config: Dict[str, Any], data) -> Dict[str, Any]:
    """
    Update configuration based on data characteristics.
    
    Args:
        config: Original configuration
        data: Data object with attributes
        
    Returns:
        Updated configuration
    """
    updated_config = config.copy()
    
    # Update input dimensions based on data
    if hasattr(data, 'shape'):
        # For tensor data
        if len(data.shape) == 4:  # Batch x Channels x Height x Width
            updated_config['input_channels'] = data.shape[1]
            updated_config['input_size'] = (data.shape[2], data.shape[3])
    elif hasattr(data, 'num_channels'):
        updated_config['input_channels'] = data.num_channels
    elif hasattr(data, 'channels'):
        updated_config['input_channels'] = data.channels
    
    return updated_config


def create_sample_image_data(batch_size: int = 4, channels: int = 3, 
                           height: int = 224, width: int = 224, 
                           num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample image data for testing.
    
    Args:
        batch_size: Number of samples
        channels: Number of channels
        height: Image height
        width: Image width
        num_classes: Number of output classes
        
    Returns:
        Tuple of (images, labels)
    """
    images = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


def calculate_output_size(input_size: Tuple[int, int], kernel_size: int = 3, 
                         stride: int = 1, padding: int = 1, pooling: int = 2) -> Tuple[int, int]:
    """
    Calculate output size after convolution and pooling.
    
    Args:
        input_size: (height, width) of input
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        pooling: Pooling kernel size
        
    Returns:
        Output size (height, width)
    """
    height, width = input_size
    
    # After convolution
    height = (height + 2 * padding - kernel_size) // stride + 1
    width = (width + 2 * padding - kernel_size) // stride + 1
    
    # After pooling
    height = height // pooling
    width = width // pooling
    
    return height, width


def estimate_parameters(config: Dict[str, Any]) -> int:
    """
    Estimate number of parameters for a given configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Estimated parameter count
    """
    # If architecture is specified and not BaseCNN, use architecture-specific estimation
    architecture = config.get('architecture', 'BaseCNN')
    if architecture.lower() != 'basecnn':
        return estimate_architecture_parameters(config)
    
    # Original BaseCNN estimation
    input_channels = config.get('input_channels', 3)
    base_channels = config.get('base_channels', 64)
    num_blocks = config.get('num_blocks', 4)
    hidden_dims = config.get('hidden_dims', [512, 256])
    use_batch_norm = config.get('use_batch_norm', True)
    
    total_params = 0
    
    # Convolutional layers
    current_channels = input_channels
    for i in range(num_blocks):
        out_channels = base_channels * (2 ** i)
        
        # Conv layer parameters
        kernel_size = config.get('kernel_size', 3)
        conv_params = (kernel_size * kernel_size * current_channels * out_channels)
        if not use_batch_norm:  # Bias if no batch norm
            conv_params += out_channels
        total_params += conv_params
        
        # Batch norm parameters (2 per channel: weight + bias)
        if use_batch_norm:
            total_params += 2 * out_channels
            
        current_channels = out_channels
    
    # Calculate spatial size after conv blocks
    input_size = config.get('input_size', (224, 224))
    spatial_reduction = (2 ** num_blocks)  # Assuming pooling at each block
    spatial_size = (input_size[0] // spatial_reduction) * (input_size[1] // spatial_reduction)
    fc_input_size = current_channels * spatial_size
    
    # Fully connected layers
    prev_dim = fc_input_size
    for hidden_dim in hidden_dims:
        # Linear layer parameters
        total_params += (prev_dim * hidden_dim) + hidden_dim
        prev_dim = hidden_dim
    
    # Output layer
    output_dim = config.get('output_dim', 10)
    total_params += (prev_dim * output_dim) + output_dim
    
    return total_params


def estimate_architecture_parameters(config: Dict[str, Any]) -> int:
    """
    Estimate parameters for specific architectures.
    
    Args:
        config: Model configuration with architecture info
        
    Returns:
        Estimated parameter count
    """
    architecture = config.get('architecture', 'BaseCNN').lower()
    variant = config.get('variant', '')
    
    # Rough parameter estimates for common architectures
    architecture_params = {
        'resnet': {
            'resnet18': 11689512,
            'resnet34': 21797672,
            'resnet50': 25557032,
            'resnet101': 44549160,
            'resnet152': 60192808
        },
        'efficientnet': {
            'efficientnet_b0': 5288548,
            'efficientnet_b1': 7794184,
            'efficientnet_b2': 9109994,
            'efficientnet_b3': 12233232,
            'efficientnet_b4': 19341616,
            'efficientnet_b5': 30389784,
            'efficientnet_b6': 43040704,
            'efficientnet_b7': 66347960
        },
        'mobilenet': {
            'mobilenet_v2': 3504872,
            'mobilenet_v3_small': 2542856,
            'mobilenet_v3_large': 5485352
        },
        'densenet': {
            'densenet121': 7978856,
            'densenet169': 14149480,
            'densenet201': 20013928
        }
    }
    
    # Get base parameters for architecture variant
    if architecture in architecture_params:
        if variant in architecture_params[architecture]:
            base_params = architecture_params[architecture][variant]
            # Adjust for output dimension (rough adjustment)
            output_dim = config.get('output_dim', 1000)  # Default ImageNet classes
            if output_dim != 1000:
                # Rough adjustment: last layer change
                adjustment = (output_dim - 1000) * 2048  # Assuming 2048 features before output
                return base_params + adjustment
            return base_params
    
    # Fallback to BaseCNN estimation
    return estimate_parameters(config)


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def setup_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup training environment based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with device info
    """
    updated_config = config.copy()
    
    # Set device
    device = get_device()
    updated_config['device'] = device
    
    architecture = config.get('architecture', 'BaseCNN')
    variant = config.get('variant', 'default')
    
    print(f"🚀 Training setup:")
    print(f"   Architecture: {architecture}")
    if variant != 'default':
        print(f"   Variant: {variant}")
    print(f"   Device: {device}")
    print(f"   Optimizer: {config.get('optimizer', 'adam')}")
    print(f"   Learning rate: {config.get('learning_rate', 0.001)}")
    
    return updated_config


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    # Architecture-specific learning rate adjustments
    architecture = config.get('architecture', 'BaseCNN')
    if architecture.lower() != 'basecnn' and config.get('pretrained', True):
        # Lower learning rate for fine-tuning pre-trained models
        learning_rate = learning_rate * 0.1
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer: {optimizer_type}, using Adam")
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


def create_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create default configurations for common use cases.
    
    Returns:
        Dictionary of configurations
    """
    configs = {
        'mnist_basic': {
            'input_channels': 1,
            'input_size': (28, 28),
            'output_dim': 10,
            'base_channels': 32,
            'num_blocks': 3,
            'hidden_dims': [128, 64],
            'dropout_rate': 0.3,
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        'cifar_basic': {
            'input_channels': 3,
            'input_size': (32, 32),
            'output_dim': 10,
            'base_channels': 64,
            'num_blocks': 4,
            'hidden_dims': [256, 128],
            'dropout_rate': 0.4,
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        'ui_analysis': {
            'input_channels': 3,
            'input_size': (224, 224),
            'output_dim': 2,
            'base_channels': 64,
            'num_blocks': 4,
            'hidden_dims': [512, 256],
            'dropout_rate': 0.5,
            'optimizer': 'adamw',
            'learning_rate': 0.0005,
            'task_type': 'classification'
        },
        'creative_scoring': {
            'input_channels': 3,
            'input_size': (224, 224),
            'output_dim': 1,
            'base_channels': 64,
            'num_blocks': 5,
            'hidden_dims': [1024, 512, 256],
            'dropout_rate': 0.4,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'task_type': 'regression'
        }
    }
    
    return configs


def create_architecture_presets() -> Dict[str, Dict[str, Any]]:
    """
    Create architecture-specific configuration presets.
    
    Returns:
        Dictionary of architecture presets
    """
    presets = {
        'resnet_ui_analysis': {
            'architecture': 'resnet',
            'variant': 'resnet50',
            'input_channels': 3,
            'input_size': (224, 224),
            'output_dim': 2,
            'pretrained': True,
            'task_type': 'classification',
            'optimizer': 'adamw',
            'learning_rate': 0.0001
        },
        'efficientnet_creative': {
            'architecture': 'efficientnet',
            'variant': 'efficientnet_b4',
            'input_channels': 3,
            'input_size': (224, 224),
            'output_dim': 1,
            'pretrained': True,
            'task_type': 'regression',
            'optimizer': 'adam',
            'learning_rate': 0.0005
        },
        'mobilenet_accessibility': {
            'architecture': 'mobilenet',
            'variant': 'mobilenet_v3_small',
            'input_channels': 3,
            'input_size': (224, 224),
            'output_dim': 4,
            'pretrained': True,
            'task_type': 'classification',
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        'densenet_ocr': {
            'architecture': 'densenet',
            'variant': 'densenet121',
            'input_channels': 1,
            'input_size': (128, 128),
            'output_dim': 5,
            'pretrained': True,
            'task_type': 'classification',
            'optimizer': 'adam',
            'learning_rate': 0.001
        }
    }
    
    return presets


def save_default_configs(config_dir: str = 'configs'):
    """
    Save default configurations to YAML files.
    
    Args:
        config_dir: Directory to save configurations
    """
    os.makedirs(config_dir, exist_ok=True)
    
    # Save BaseCNN configs
    configs = create_default_configs()
    config_path = os.path.join(config_dir, 'cnn_configs.yaml')
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(configs, f, default_flow_style=False)
        print(f"✅ BaseCNN configurations saved to {config_path}")
    except Exception as e:
        print(f"❌ Error saving configurations: {e}")
    
    # Save architecture presets
    presets = create_architecture_presets()
    presets_path = os.path.join(config_dir, 'architecture_presets.yaml')
    
    try:
        with open(presets_path, 'w') as f:
            yaml.dump(presets, f, default_flow_style=False)
        print(f"✅ Architecture presets saved to {presets_path}")
    except Exception as e:
        print(f"❌ Error saving architecture presets: {e}")


def print_config_summary(config: Dict[str, Any], title: str = "Configuration"):
    """
    Print a formatted summary of configuration.
    
    Args:
        config: Configuration dictionary
        title: Title for the summary
    """
    print(f"📋 {title}")
    print("=" * 50)
    
    # Architecture info
    architecture = config.get('architecture', 'BaseCNN')
    variant = config.get('variant', 'default')
    
    print(f"🏗️  Architecture: {architecture}")
    if variant != 'default':
        print(f"   Variant: {variant}")
    print(f"   Pretrained: {config.get('pretrained', False)}")
    
    # Architecture parameters
    arch_params = ['input_channels', 'input_size', 'output_dim', 'base_channels', 
                  'num_blocks', 'hidden_dims', 'task_type']
    
    print("\n📐 Parameters:")
    for key in arch_params:
        if key in config:
            print(f"  {key:<15}: {config[key]}")
    
    # Training parameters
    train_params = ['optimizer', 'learning_rate', 'weight_decay', 'dropout_rate', 'epochs']
    
    print("\n⚙️  Training:")
    for key in train_params:
        if key in config:
            print(f"  {key:<15}: {config[key]}")
    
    # Estimate parameters
    estimated = estimate_parameters(config)
    print(f"\n📊 Estimated parameters: {estimated:,}")
    print("=" * 50)


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required parameters
    required = ['input_channels', 'input_size', 'output_dim']
    for param in required:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Type checks
    if 'input_size' in config and not isinstance(config['input_size'], (tuple, list)):
        errors.append("input_size must be a tuple or list (height, width)")
    
    if 'hidden_dims' in config and not isinstance(config['hidden_dims'], (list, tuple)):
        errors.append("hidden_dims must be a list or tuple")
    
    # Value checks
    if 'input_channels' in config and config['input_channels'] <= 0:
        errors.append("input_channels must be positive")
    
    if 'output_dim' in config and config['output_dim'] <= 0:
        errors.append("output_dim must be positive")
    
    # Architecture-specific validation
    architecture = config.get('architecture', 'BaseCNN')
    if architecture.lower() != 'basecnn':
        valid_architectures = ['resnet', 'efficientnet', 'mobilenet', 'densenet']
        if architecture.lower() not in valid_architectures:
            errors.append(f"Unknown architecture: {architecture}. Valid: {valid_architectures}")
    
    return len(errors) == 0, errors


def validate_architecture_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate architecture-specific configuration.
    
    Args:
        config: Configuration with architecture info
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    architecture = config.get('architecture', 'BaseCNN')
    
    if architecture.lower() == 'resnet':
        valid_variants = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        variant = config.get('variant', 'resnet50')
        if variant not in valid_variants:
            errors.append(f"Invalid ResNet variant: {variant}. Valid: {valid_variants}")
    
    elif architecture.lower() == 'efficientnet':
        valid_variants = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                         'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                         'efficientnet_b6', 'efficientnet_b7']
        variant = config.get('variant', 'efficientnet_b0')
        if variant not in valid_variants:
            errors.append(f"Invalid EfficientNet variant: {variant}. Valid: {valid_variants}")
    
    elif architecture.lower() == 'mobilenet':
        valid_variants = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']
        variant = config.get('variant', 'mobilenet_v3_small')
        if variant not in valid_variants:
            errors.append(f"Invalid MobileNet variant: {variant}. Valid: {valid_variants}")
    
    elif architecture.lower() == 'densenet':
        valid_variants = ['densenet121', 'densenet169', 'densenet201']
        variant = config.get('variant', 'densenet121')
        if variant not in valid_variants:
            errors.append(f"Invalid DenseNet variant: {variant}. Valid: {valid_variants}")
    
    return len(errors) == 0, errors


def get_architecture_comparison() -> Dict[str, Dict[str, Any]]:
    """
    Get comparison information for different architectures.
    
    Returns:
        Dictionary with architecture comparisons
    """
    comparison = {
        'BaseCNN': {
            'parameters': '1M-10M',
            'speed': 'Medium',
            'accuracy': 'Good',
            'best_for': ['Custom implementations', 'Prototyping', 'Education'],
            'pretrained': False
        },
        'ResNet': {
            'parameters': '11M-60M',
            'speed': 'Medium-Fast',
            'accuracy': 'Excellent',
            'best_for': ['High accuracy', 'Transfer learning', 'Research'],
            'pretrained': True
        },
        'EfficientNet': {
            'parameters': '5M-66M',
            'speed': 'Fast',
            'accuracy': 'State-of-the-art',
            'best_for': ['Production systems', 'Balanced requirements'],
            'pretrained': True
        },
        'MobileNet': {
            'parameters': '2M-6M',
            'speed': 'Very Fast',
            'accuracy': 'Good',
            'best_for': ['Mobile/edge devices', 'Real-time applications'],
            'pretrained': True
        },
        'DenseNet': {
            'parameters': '8M-20M',
            'speed': 'Medium',
            'accuracy': 'Excellent',
            'best_for': ['Feature-rich tasks', 'Parameter efficiency'],
            'pretrained': True
        }
    }
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Test the utility functions
    print("🧪 Testing CNN Utilities with Multi-Architecture Support")
    
    # Test architecture comparison
    comparison = get_architecture_comparison()
    print("\n📊 Architecture Comparison:")
    for arch, info in comparison.items():
        print(f"  {arch}: {info['parameters']} params, {info['speed']} speed, {info['accuracy']} accuracy")
    
    # Test architecture presets
    presets = create_architecture_presets()
    print(f"\n🎯 Architecture Presets: {list(presets.keys())}")
    
    # Test parameter estimation for different architectures
    test_configs = [
        {'architecture': 'resnet', 'variant': 'resnet50', 'output_dim': 10},
        {'architecture': 'efficientnet', 'variant': 'efficientnet_b3', 'output_dim': 2},
        {'architecture': 'mobilenet', 'variant': 'mobilenet_v3_small', 'output_dim': 5}
    ]
    
    print("\n🔢 Parameter Estimation:")
    for config in test_configs:
        estimated = estimate_architecture_parameters(config)
        print(f"  {config['architecture']} ({config['variant']}): {estimated:,} parameters")
    
    # Save default configs
    save_default_configs('test_configs')