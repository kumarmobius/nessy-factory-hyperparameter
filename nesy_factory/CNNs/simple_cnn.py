"""
Simple CNN implementation extending BaseCNN.
A lightweight, easy-to-understand CNN for basic tasks and education.
"""

import torch
import torch.nn as nn
from .base import BaseCNN
from typing import Dict, Any, List

class SimpleCNN(BaseCNN):
    """
    Simple CNN implementation for basic tasks and educational purposes.
    
    Features:
    - Clean, easy-to-understand architecture
    - Lightweight and fast training
    - Perfect for prototyping and learning
    - No complex components like residual connections
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SimpleCNN model.
        
        Args:
            config: Configuration dictionary
        """
        # SimpleCNN-specific parameters
        self.num_conv_layers = config.get('num_conv_layers', 3)
        self.conv_channels = config.get('conv_channels', [32, 64, 128])
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_dropout = config.get('use_dropout', True)
        
        # Initialize BaseCNN first
        super().__init__(config)
    
    def _build_layers(self):
        """Build SimpleCNN architecture."""
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        # Use provided channels or generate based on num_conv_layers
        if isinstance(self.conv_channels, (list, tuple)) and len(self.conv_channels) >= self.num_conv_layers:
            channels = self.conv_channels[:self.num_conv_layers]
        else:
            # Generate channel progression: 32, 64, 128, 256, etc.
            channels = [32 * (2 ** i) for i in range(self.num_conv_layers)]
        
        for i, out_channels in enumerate(channels):
            conv_block = self._create_conv_block(in_channels, out_channels)
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate FC input size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_channels, *self.input_size)
            for conv_layer in self.conv_layers:
                dummy_input = conv_layer(dummy_input)
                dummy_input = self.pool(dummy_input)
            fc_input_size = dummy_input.view(1, -1).size(1)
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        
        # Determine hidden dimensions
        if hasattr(self, 'hidden_dims') and self.hidden_dims:
            hidden_dims = self.hidden_dims
        else:
            # Auto-generate hidden dimensions based on complexity
            if self.num_conv_layers <= 2:
                hidden_dims = [128]
            elif self.num_conv_layers <= 4:
                hidden_dims = [256, 128]
            else:
                hidden_dims = [512, 256, 128]
        
        # Build FC layers
        prev_dim = fc_input_size
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                self.fc_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
    
    def _create_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a simple convolutional block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            
        Returns:
            Sequential block of layers
        """
        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=not self.use_batch_norm
        ))
        
        # Batch normalization
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SimpleCNN.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor
        """
        # Convolutional layers with pooling
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract feature maps from specific convolutional layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last conv layer)
            
        Returns:
            Feature maps tensor
        """
        self.eval()
        with torch.no_grad():
            x = self._to_device(x)
            
            # Handle negative indexing
            if layer_idx < 0:
                layer_idx = len(self.conv_layers) + layer_idx
            
            if layer_idx < 0 or layer_idx >= len(self.conv_layers):
                raise ValueError(f"Layer index {layer_idx} out of range")
            
            # Forward pass until target layer
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if i == layer_idx:
                    return x
                x = self.pool(x)
            
            return x

class SimpleCNNV2(BaseCNN):
    """
    Enhanced SimpleCNN with additional features.
    
    Features:
    - Optional global average pooling
    - Configurable kernel sizes
    - Multiple activation functions
    - Advanced regularization options
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SimpleCNNV2 model.
        
        Args:
            config: Configuration dictionary
        """
        # Enhanced parameters - INITIALIZE THESE BEFORE CALLING SUPER()
        self.num_conv_layers = config.get('num_conv_layers', 3)
        self.conv_channels = config.get('conv_channels', [32, 64, 128])  # FIXED: Initialize before super()
        self.kernel_sizes = config.get('kernel_sizes', [3, 3, 3])
        self.strides = config.get('strides', [1, 1, 1])
        self.paddings = config.get('paddings', [1, 1, 1])
        self.use_global_pooling = config.get('use_global_pooling', False)
        self.activation = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_dropout = config.get('use_dropout', True)
        
        # Initialize BaseCNN
        super().__init__(config)
    
    def _build_layers(self):
        """Build enhanced SimpleCNN architecture."""
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        # Determine channels - FIXED: Use self.conv_channels which is now properly initialized
        if isinstance(self.conv_channels, (list, tuple)) and len(self.conv_channels) >= self.num_conv_layers:
            channels = self.conv_channels[:self.num_conv_layers]
        else:
            # Generate channel progression: 32, 64, 128, 256, etc.
            channels = [32 * (2 ** i) for i in range(self.num_conv_layers)]
        
        # Ensure kernel_sizes, strides, paddings match num_conv_layers
        kernel_sizes = self._ensure_list_length(self.kernel_sizes, self.num_conv_layers, 3)
        strides = self._ensure_list_length(self.strides, self.num_conv_layers, 1)
        paddings = self._ensure_list_length(self.paddings, self.num_conv_layers, 1)
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
            zip(channels, kernel_sizes, strides, paddings)
        ):
            conv_block = self._create_enhanced_conv_block(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global pooling if enabled
        if self.use_global_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate FC input size
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_channels, *self.input_size)
            for conv_layer in self.conv_layers:
                dummy_input = conv_layer(dummy_input)
                dummy_input = self.pool(dummy_input)
            
            if self.use_global_pooling:
                dummy_input = self.global_pool(dummy_input)
            
            fc_input_size = dummy_input.view(1, -1).size(1)
        
        # Build classifier
        self.classifier = self._build_classifier(fc_input_size)
    
    def _ensure_list_length(self, lst, target_length, default_value):
        """Ensure list has target length by padding or truncating."""
        if not isinstance(lst, (list, tuple)):
            return [default_value] * target_length
        if len(lst) < target_length:
            return lst + [default_value] * (target_length - len(lst))
        return lst[:target_length]
    
    def _create_enhanced_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create enhanced convolutional block."""
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=not self.use_batch_norm
        ))
        
        # Batch normalization
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function
        if self.activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        elif self.activation == 'selu':
            layers.append(nn.SELU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self, input_size):
        """Build classifier layers."""
        layers = []
        prev_dim = input_size
        
        # Use provided hidden_dims or default
        if hasattr(self, 'hidden_dims') and self.hidden_dims:
            hidden_dims = self.hidden_dims
        else:
            hidden_dims = [512, 256] if self.num_conv_layers >= 3 else [256, 128]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate) if self.use_dropout else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for enhanced SimpleCNN."""
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.pool(x)
        
        # Global pooling if enabled
        if self.use_global_pooling:
            x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x

# Factory functions
def create_simple_cnn(config: Dict[str, Any]) -> SimpleCNN:
    """
    Factory function to create SimpleCNN model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SimpleCNN model instance
    """
    return SimpleCNN(config)

def create_simple_cnn_v2(config: Dict[str, Any]) -> SimpleCNNV2:
    """
    Factory function to create enhanced SimpleCNN model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SimpleCNNV2 model instance
    """
    return SimpleCNNV2(config)
