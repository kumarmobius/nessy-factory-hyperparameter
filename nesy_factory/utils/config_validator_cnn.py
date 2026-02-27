#!/usr/bin/env python3
"""
Configuration validator for CNN Factory architectures.
Provides comprehensive validation for different CNN architectures.
"""

from typing import Dict, Any, List, Tuple, Optional
from .utils_cnn import validate_config, validate_architecture_config


class ConfigValidator:
    """
    Comprehensive configuration validator for CNN architectures.
    """
    
    def __init__(self):
        self.valid_architectures = ['basecnn', 'resnet', 'efficientnet', 'mobilenet', 'densenet']
        self.architecture_variants = {
            'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
            'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                           'efficientnet_b6', 'efficientnet_b7'],
            'mobilenet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
            'densenet': ['densenet121', 'densenet169', 'densenet201']
        }
    
    def validate_complete_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Perform complete configuration validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Basic validation
        is_basic_valid, basic_errors = validate_config(config)
        if not is_basic_valid:
            errors.extend(basic_errors)
        
        # Architecture-specific validation
        is_arch_valid, arch_errors = validate_architecture_config(config)
        if not is_arch_valid:
            errors.extend(arch_errors)
        
        # Additional validations
        errors.extend(self._validate_training_params(config))
        errors.extend(self._validate_data_params(config))
        errors.extend(self._validate_architecture_compatibility(config))
        
        return len(errors) == 0, errors
    
    def _validate_training_params(self, config: Dict[str, Any]) -> List[str]:
        """Validate training parameters."""
        errors = []
        
        # Learning rate validation
        lr = config.get('learning_rate', 0.001)
        if lr <= 0:
            errors.append("Learning rate must be positive")
        elif lr > 1.0:
            errors.append("Learning rate seems too high (should typically be < 1.0)")
        
        # Optimizer validation
        valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad']
        optimizer = config.get('optimizer', 'adam')
        if optimizer.lower() not in valid_optimizers:
            errors.append(f"Invalid optimizer: {optimizer}. Valid: {valid_optimizers}")
        
        return errors
    
    def _validate_data_params(self, config: Dict[str, Any]) -> List[str]:
        """Validate data-related parameters."""
        errors = []
        
        # Input size validation
        input_size = config.get('input_size', (224, 224))
        if isinstance(input_size, (tuple, list)):
            h, w = input_size
            if h < 16 or w < 16:
                errors.append("Input size too small (minimum 16x16)")
            if h > 1024 or w > 1024:
                errors.append("Input size too large (maximum 1024x1024)")
        
        # Input channels validation
        input_channels = config.get('input_channels', 3)
        if input_channels not in [1, 3]:
            errors.append("Input channels should be 1 (grayscale) or 3 (RGB)")
        
        return errors
    
    def _validate_architecture_compatibility(self, config: Dict[str, Any]) -> List[str]:
        """Validate architecture compatibility with other parameters."""
        errors = []
        
        architecture = config.get('architecture', 'BaseCNN').lower()
        input_size = config.get('input_size', (224, 224))
        
        # Check input size compatibility
        if architecture != 'basecnn':
            if isinstance(input_size, (tuple, list)):
                h, w = input_size
                # Most pre-trained models expect at least 224x224
                if h < 224 or w < 224:
                    errors.append(f"{architecture} typically expects input size >= 224x224")
        
        # Check task type compatibility
        task_type = config.get('task_type', 'classification')
        if task_type not in ['classification', 'regression']:
            errors.append(f"Unsupported task type: {task_type}")
        
        return errors
    
    def get_recommended_config(self, architecture: str, use_case: str) -> Dict[str, Any]:
        """
        Get recommended configuration for architecture and use case.
        
        Args:
            architecture: Target architecture
            use_case: Target use case
            
        Returns:
            Recommended configuration
        """
        base_config = {
            'architecture': architecture,
            'pretrained': True,
            'optimizer': 'adamw',
            'learning_rate': 0.001
        }
        
        # Use case specific configurations
        use_case_configs = {
            'ui_analysis': {
                'input_size': (224, 224),
                'input_channels': 3,
                'task_type': 'classification',
                'output_dim': 2
            },
            'creative_quality': {
                'input_size': (224, 224),
                'input_channels': 3, 
                'task_type': 'regression',
                'output_dim': 1
            },
            'chart_ocr': {
                'input_size': (128, 128),
                'input_channels': 1,
                'task_type': 'classification',
                'output_dim': 5
            },
            'accessibility': {
                'input_size': (224, 224),
                'input_channels': 3,
                'task_type': 'classification', 
                'output_dim': 4
            }
        }
        
        use_case_config = use_case_configs.get(use_case, {})
        
        # Architecture-specific adjustments
        if architecture.lower() == 'resnet':
            base_config['variant'] = 'resnet50'
            base_config['learning_rate'] = 0.0001
        elif architecture.lower() == 'efficientnet':
            base_config['variant'] = 'efficientnet_b3'
            base_config['learning_rate'] = 0.0005
        elif architecture.lower() == 'mobilenet':
            base_config['variant'] = 'mobilenet_v3_small'
            base_config['learning_rate'] = 0.001
        elif architecture.lower() == 'densenet':
            base_config['variant'] = 'densenet121'
            base_config['learning_rate'] = 0.001
        
        return {**base_config, **use_case_config}


def validate_and_fix_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate configuration and attempt to fix common issues.
    
    Args:
        config: Configuration to validate and fix
        
    Returns:
        Tuple of (fixed_config, warnings)
    """
    validator = ConfigValidator()
    warnings = []
    
    # Create a copy to modify
    fixed_config = config.copy()
    
    # Validate first
    is_valid, errors = validator.validate_complete_config(config)
    
    if not is_valid:
        # Try to fix common issues
        for error in errors:
            if "input_size" in error and "small" in error:
                fixed_config['input_size'] = (224, 224)
                warnings.append("Fixed input_size to 224x224")
            elif "learning_rate" in error and "high" in error:
                fixed_config['learning_rate'] = 0.001
                warnings.append("Fixed learning_rate to 0.001")
            elif "architecture" in error and "unknown" in error:
                fixed_config['architecture'] = 'BaseCNN'
                warnings.append("Fixed architecture to BaseCNN")
    
    return fixed_config, warnings


# Example usage
if __name__ == "__main__":
    validator = ConfigValidator()
    
    # Test configurations
    test_configs = [
        {'architecture': 'resnet', 'variant': 'resnet50', 'input_size': (224, 224), 'output_dim': 10},
        {'architecture': 'unknown', 'input_size': (10, 10), 'learning_rate': 10.0},
        {'architecture': 'efficientnet', 'variant': 'efficientnet_b3', 'input_channels': 3}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n🔧 Testing Config {i+1}:")
        is_valid, errors = validator.validate_complete_config(config)
        if is_valid:
            print("  ✅ Configuration is valid")
        else:
            print("  ❌ Configuration has errors:")
            for error in errors:
                print(f"    - {error}")
            
            # Try to fix
            fixed_config, warnings = validate_and_fix_config(config)
            if warnings:
                print("  🔧 Fixed configuration:")
                for warning in warnings:
                    print(f"    - {warning}")
                print(f"  Fixed config: {fixed_config}")