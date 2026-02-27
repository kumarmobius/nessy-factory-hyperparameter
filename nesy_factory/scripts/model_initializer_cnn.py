#!/usr/bin/env python3
"""
Model initializer for CNN Factory with missing functions
"""

import torch
from typing import Dict, Any, List

def initialize_cnn_from_config_dict(config_dict: Dict[str, Any]) -> torch.nn.Module:
    """Initialize CNN from configuration dictionary"""
    from nesy_factory.CNNs.factory import CNNFactory
    architecture = config_dict.get('architecture', 'BaseCNN')
    return CNNFactory.create_model(architecture, config_dict)

def initialize_cnn_from_yaml(yaml_path: str) -> torch.nn.Module:
    """Initialize CNN from YAML configuration"""
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return initialize_cnn_from_config_dict(config)

def initialize_cnn_with_custom_config(base_config: Dict[str, Any], **kwargs) -> torch.nn.Module:
    """Initialize CNN with custom configuration overrides"""
    config = base_config.copy()
    config.update(kwargs)
    return initialize_cnn_from_config_dict(config)

def initialize_multiple_cnns(configs: List[Dict[str, Any]]) -> List[torch.nn.Module]:
    """Initialize multiple CNN models"""
    return [initialize_cnn_from_config_dict(config) for config in configs]

def list_all_available_cnn_options():
    """List all available CNN options"""
    from nesy_factory.CNNs.registry import list_available_models
    print("Available CNN Models:")
    list_available_models()

# Keep your existing functions below...
def create_simple_cnn_config(output_dim: int = 10, variant: str = 'simple_cnn') -> Dict[str, Any]:
    """Create configuration for SimpleCNN models."""
    base_config = {
        'architecture': 'simple_cnn',
        'input_channels': 1,
        'input_size': (28, 28),
        'output_dim': output_dim,
        'num_conv_layers': 3,
        'conv_channels': [32, 64, 128],
        'hidden_dims': [128, 64],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'use_dropout': True,
        'optimizer': 'adam',
        'learning_rate': 0.001
    }
    
    if variant == 'simple_cnn_v2':
        base_config['architecture'] = 'simple_cnn_v2'
        base_config.update({
            'kernel_sizes': [3, 3, 3],
            'strides': [1, 1, 1],
            'paddings': [1, 1, 1],
            'use_global_pooling': False,
            'activation': 'relu'
        })
    
    return base_config

def create_mnist_config(output_dim: int = 10, architecture: str = 'BaseCNN') -> Dict[str, Any]:
    """Create configuration for MNIST-like datasets."""
    base_config = {
        'input_channels': 1,
        'input_size': (28, 28),
        'output_dim': output_dim,
        'base_channels': 32,
        'num_blocks': 3,
        'hidden_dims': [128, 64],
        'dropout_rate': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.001
    }
    
    # Architecture-specific adjustments
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        if architecture.lower() == 'mobilenet':
            base_config['variant'] = 'mobilenet_v3_small'
        elif architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b0'
        elif architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet18'
    
    return base_config

def create_cifar_config(output_dim: int = 10, architecture: str = 'BaseCNN') -> Dict[str, Any]:
    """Create configuration for CIFAR-like datasets."""
    base_config = {
        'input_channels': 3,
        'input_size': (32, 32),
        'output_dim': output_dim,
        'base_channels': 64,
        'num_blocks': 4,
        'hidden_dims': [256, 128],
        'dropout_rate': 0.4,
        'optimizer': 'adam',
        'learning_rate': 0.001
    }
    
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        if architecture.lower() == 'mobilenet':
            base_config['variant'] = 'mobilenet_v3_small'
        elif architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b0'
        elif architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet34'
    
    return base_config

def create_ui_analysis_config(output_dim: int = 2, architecture: str = 'EfficientNet') -> Dict[str, Any]:
    """Create configuration for UI analysis tasks."""
    base_config = {
        'input_channels': 3,
        'input_size': (224, 224),
        'output_dim': output_dim,
        'base_channels': 64,
        'num_blocks': 4,
        'hidden_dims': [512, 256],
        'dropout_rate': 0.5,
        'optimizer': 'adamw',
        'learning_rate': 0.0005,
        'task_type': 'classification'
    }
    
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        base_config['pretrained'] = True
        
        if architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b3'
        elif architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet50'
        elif architecture.lower() == 'mobilenet':
            base_config['variant'] = 'mobilenet_v3_large'
    
    return base_config

def create_creative_scoring_config(output_dim: int = 1, task_type: str = 'regression', architecture: str = 'EfficientNet') -> Dict[str, Any]:
    """Create configuration for creative analysis tasks."""
    base_config = {
        'input_channels': 3,
        'input_size': (224, 224),
        'output_dim': output_dim,
        'base_channels': 64,
        'num_blocks': 5,
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.4,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'task_type': task_type
    }
    
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        base_config['pretrained'] = True
        
        if architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b4'
        elif architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet101'
        elif architecture.lower() == 'densenet':
            base_config['architecture'] = 'densenet'
            base_config['variant'] = 'densenet169'
    
    return base_config

def create_ocr_config(output_dim: int = 10, architecture: str = 'DenseNet') -> Dict[str, Any]:
    """Create configuration for OCR and text detection tasks."""
    base_config = {
        'input_channels': 1,  # Grayscale for text
        'input_size': (128, 128),
        'output_dim': output_dim,
        'base_channels': 32,
        'num_blocks': 4,
        'hidden_dims': [256, 128],
        'dropout_rate': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'task_type': 'classification'
    }
    
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        base_config['pretrained'] = True
        
        if architecture.lower() == 'densenet':
            base_config['variant'] = 'densenet121'
        elif architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet34'
        elif architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b2'
    
    return base_config

def create_custom_config(input_channels: int, input_size: tuple, output_dim: int, 
                        complexity: str = 'medium', architecture: str = 'BaseCNN') -> Dict[str, Any]:
    """Create custom configuration based on complexity."""
    complexity_settings = {
        'light': {'base_channels': 32, 'num_blocks': 3, 'hidden_dims': [128]},
        'medium': {'base_channels': 64, 'num_blocks': 4, 'hidden_dims': [256, 128]},
        'heavy': {'base_channels': 128, 'num_blocks': 5, 'hidden_dims': [512, 256, 128]}
    }
    
    if complexity not in complexity_settings:
        complexity = 'medium'
    
    settings = complexity_settings[complexity]
    
    base_config = {
        'input_channels': input_channels,
        'input_size': input_size,
        'output_dim': output_dim,
        'base_channels': settings['base_channels'],
        'num_blocks': settings['num_blocks'],
        'hidden_dims': settings['hidden_dims'],
        'dropout_rate': 0.5 if complexity == 'heavy' else 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.001
    }
    
    if architecture.lower() != 'basecnn':
        base_config['architecture'] = architecture
        base_config['pretrained'] = True
        
        # Set appropriate variants based on complexity
        if architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet18' if complexity == 'light' else 'resnet50' if complexity == 'medium' else 'resnet101'
        elif architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b0' if complexity == 'light' else 'efficientnet_b3' if complexity == 'medium' else 'efficientnet_b5'
        elif architecture.lower() == 'mobilenet':
            base_config['variant'] = 'mobilenet_v3_small' if complexity == 'light' else 'mobilenet_v3_large'
    
    return base_config

def initialize_model_for_task(task_type: str, architecture: str = 'auto', **kwargs) -> Dict[str, Any]:
    """
    Automatically create configuration for specific task types.
    
    Args:
        task_type: Type of task ('mnist', 'cifar', 'ui_analysis', 'creative', 'ocr')
        architecture: Architecture to use ('auto' for automatic selection)
        **kwargs: Additional parameters like output_dim, input_size, etc.
    """
    
    # Automatic architecture selection
    if architecture == 'auto':
        auto_arch_map = {
            'mnist': 'BaseCNN',
            'cifar': 'BaseCNN', 
            'ui_analysis': 'EfficientNet',
            'creative': 'EfficientNet',
            'ocr': 'DenseNet'
        }
        architecture = auto_arch_map.get(task_type, 'BaseCNN')
    
    task_initializers = {
        'mnist': create_mnist_config,
        'cifar': create_cifar_config,
        'ui_analysis': create_ui_analysis_config,
        'creative': create_creative_scoring_config,
        'ocr': create_ocr_config
    }
    
    if task_type not in task_initializers:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(task_initializers.keys())}")
    
    return task_initializers[task_type](architecture=architecture, **kwargs)

def create_model_from_config(config: Dict[str, Any]):
    """
    Create a model instance from configuration using CNNFactory.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model instance
    """
    try:
        from nesy_factory.CNNs.factory import CNNFactory
        
        architecture = config.get('architecture', 'BaseCNN')
        # UPDATE: Use the new registry-based factory
        return CNNFactory.create_model(architecture, config)
        
    except ImportError as e:
        print(f"❌ Error creating model: {e}")
        print("Falling back to BaseCNN...")
        from nesy_factory.CNNs import BaseCNN
        
        # Remove architecture-specific keys
        base_config = {k: v for k, v in config.items() if k not in ['architecture', 'variant', 'pretrained']}
        
        class CustomModel(BaseCNN):
            def forward(self, x):
                for conv_block in self.conv_blocks:
                    x = conv_block(x)
                    x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return CustomModel(base_config)

def print_config_summary(config: Dict[str, Any], task_name: str = "Custom"):
    """Print a summary of the configuration."""
    print(f"✅ {task_name.upper()} CONFIGURATION")
    print("=" * 40)
    
    key_params = ['architecture', 'input_channels', 'input_size', 'output_dim', 
                  'base_channels', 'num_blocks', 'hidden_dims', 'task_type', 
                  'optimizer', 'learning_rate', 'variant', 'pretrained']
    
    for key in key_params:
        if key in config:
            print(f"  {key:<15}: {config[key]}")
    
    print()

def demo_initializers():
    """Demonstrate all initializer functions with different architectures."""
    print("🚀 CNN CONFIGURATION INITIALIZER DEMO")
    print("=" * 50)
    
    # Demo different task configurations with recommended architectures
    tasks = [
        ('MNIST Digit Classification', create_mnist_config(architecture='BaseCNN')),
        ('CIFAR-10 (ResNet)', create_cifar_config(architecture='resnet')),
        ('UI Diff Testing (EfficientNet)', create_ui_analysis_config(architecture='efficientnet')),
        ('Creative Quality (EfficientNet)', create_creative_scoring_config(architecture='efficientnet')),
        ('Chart OCR (DenseNet)', create_ocr_config(architecture='densenet', output_dim=5)),
        ('Mobile App (MobileNet)', create_custom_config(3, (224, 224), 10, 'light', 'mobilenet'))
    ]
    
    for task_name, config in tasks:
        print_config_summary(config, task_name)
        
        # Demonstrate model creation
        try:
            model = create_model_from_config(config)
            print(f"  Model created successfully!")
            print(f"  Parameters: {model.get_num_parameters():,}")
            print()
        except Exception as e:
            print(f"  ❌ Model creation failed: {e}")
            print()

if __name__ == "__main__":
    demo_initializers()
