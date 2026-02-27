"""
CNN Model Factory Package

This package provides implementations of various Convolutional Neural Network models
with a unified interface based on PyTorch.
"""

from .base import BaseCNN
from .factory import CNNFactory
from .registry import (
    register_model, create_model, 
    list_available_models, get_available_models, is_model_available
)

# Import all model classes for direct access
from .resnet import ResNet
from .efficientnet import EfficientNet
from .mobilenet import MobileNet
from .densenet import DenseNet, DenseNetCustom
from .simple_cnn import SimpleCNN, SimpleCNNV2

__all__ = [
    # Base class
    'BaseCNN',
    
    # Factory
    'CNNFactory',
    
    # Registry functions
    'register_model',
    'create_model', 
    'list_available_models',
    'get_available_models',
    'is_model_available',
    
    # Model classes
    'ResNet',
    'EfficientNet', 
    'MobileNet',
    'DenseNet',
    'DenseNetCustom',
    'SimpleCNN',
    'SimpleCNNV2'
]

__version__ = '1.1.0'
