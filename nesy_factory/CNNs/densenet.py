"""
DenseNet implementation extending BaseCNN.
Densely Connected Convolutional Networks with feature reuse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCNN
from typing import Dict, Any, List, Tuple

class DenseNet(BaseCNN):
    """
    DenseNet implementation with dense connections between layers.
    
    Features:
    - Dense connectivity pattern for feature reuse
    - Reduced number of parameters
    - Improved gradient flow
    - Strong feature propagation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DenseNet model.
        
        Args:
            config: Configuration dictionary
        """
        # DenseNet-specific parameters
        self.variant = config.get('variant', 'densenet121')
        self.pretrained = config.get('pretrained', True)
        self.growth_rate = config.get('growth_rate', 32)
        self.compression = config.get('compression', 0.5)  # For transition layers
        
        # DenseNet architecture configurations
        self.densenet_configs = {
            'densenet121': [6, 12, 24, 16],
            'densenet169': [6, 12, 32, 32],
            'densenet201': [6, 12, 48, 32],
            'densenet161': [6, 12, 36, 24]  # Larger growth rate
        }
        
        # Initialize BaseCNN first
        super().__init__(config)
    
    def _build_layers(self):
        """Build DenseNet architecture."""
        try:
            import torchvision.models as models
            
            variant_map = {
                'densenet121': models.densenet121,
                'densenet169': models.densenet169,
                'densenet201': models.densenet201,
                'densenet161': models.densenet161
            }
            
            if self.variant in variant_map:
                # Load pretrained DenseNet
                densenet_model = variant_map[self.variant](pretrained=self.pretrained)
                
                # Replace the final classifier
                in_features = densenet_model.classifier.in_features
                densenet_model.classifier = nn.Linear(in_features, self.output_dim)
                
                # Use the entire DenseNet as our model
                self.model = densenet_model
                return
                
        except ImportError:
            print("torchvision not available, using custom DenseNet implementation")
        
        # Custom DenseNet implementation
        self._build_custom_densenet()
    
    def _build_custom_densenet(self):
        """Build custom DenseNet when torchvision is not available."""
        # Get block configuration
        if self.variant in self.densenet_configs:
            block_config = self.densenet_configs[self.variant]
        else:
            block_config = self.densenet_configs['densenet121']  # Default
        
        # Initial convolution
        num_init_features = 64
        self.features = nn.Sequential()
        
        # Initial convolution and pooling
        self.features.add_module('conv0', nn.Conv2d(
            self.input_channels, num_init_features, 
            kernel_size=7, stride=2, padding=3, bias=False
        ))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        ))
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=self.growth_rate,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * self.growth_rate
            
            # Transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * self.compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * self.compression)
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, self.output_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for DenseNet."""
        if hasattr(self, 'model'):
            # Use torchvision model
            return self.model(x)
        
        # Custom DenseNet forward
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class _DenseLayer(nn.Module):
    """Single layer within a dense block."""
    
    def __init__(self, num_input_features, growth_rate, use_batch_norm=True, dropout_rate=0.0):
        super().__init__()
        
        layers = []
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_input_features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(
            num_input_features, growth_rate, 
            kernel_size=3, stride=1, padding=1, bias=False
        ))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    """Dense block containing multiple dense layers."""
    
    def __init__(self, num_layers, num_input_features, growth_rate, 
                 use_batch_norm=True, dropout_rate=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    """Transition layer between dense blocks (compression)."""
    
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_input_features, num_output_features,
                kernel_size=1, stride=1, bias=False
            ),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)

class DenseNetCustom(BaseCNN):
    """
    Custom DenseNet implementation with BaseCNN integration.
    Provides more flexibility than the standard DenseNet.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize custom DenseNet with BaseCNN integration.
        
        Args:
            config: Configuration dictionary
        """
        # Custom DenseNet parameters
        self.growth_rate = config.get('growth_rate', 32)
        self.block_depths = config.get('block_depths', [6, 12, 24, 16])  # DenseNet-121
        self.compression_factor = config.get('compression_factor', 0.5)
        self.bottleneck = config.get('bottleneck', True)  # Use bottleneck layers
        self.use_dropout = config.get('use_dropout', True)
        
        super().__init__(config)
    
    def _build_layers(self):
        """Build custom DenseNet architecture."""
        # Initial convolution
        self.conv1 = nn.Conv2d(
            self.input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()
        
        num_features = 64
        for i, depth in enumerate(self.block_depths):
            # Dense block
            dense_block = self._make_dense_block(
                num_features, depth, 
                growth_rate=self.growth_rate,
                bottleneck=self.bottleneck
            )
            self.dense_blocks.append(dense_block)
            
            # Update feature count
            num_features += depth * self.growth_rate
            
            # Transition block (except after last block)
            if i < len(self.block_depths) - 1:
                transition = self._make_transition(
                    num_features, 
                    int(num_features * self.compression_factor)
                )
                self.transition_blocks.append(transition)
                num_features = int(num_features * self.compression_factor)
        
        # Final layers
        self.bn_final = nn.BatchNorm2d(num_features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, self.output_dim)
    
    def _make_dense_block(self, in_channels, depth, growth_rate, bottleneck=True):
        """Create a dense block with specified depth."""
        layers = []
        for i in range(depth):
            if bottleneck:
                # Bottleneck layer: 1x1 conv followed by 3x3 conv
                layers.append(
                    _BottleneckLayer(in_channels + i * growth_rate, growth_rate)
                )
            else:
                # Standard dense layer
                layers.append(
                    _DenseLayer(in_channels + i * growth_rate, growth_rate)
                )
        return nn.Sequential(*layers)
    
    def _make_transition(self, in_channels, out_channels):
        """Create a transition layer for compression."""
        return _Transition(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for custom DenseNet."""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Dense blocks with transitions
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            
            # Apply transition if not the last block
            if i < len(self.transition_blocks):
                x = self.transition_blocks[i](x)
        
        # Final layers
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class _BottleneckLayer(nn.Module):
    """Bottleneck layer for DenseNet (1x1 conv reduces computation)."""
    
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        # 1x1 bottleneck convolution
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 
                              kernel_size=1, bias=False)
        
        # 3x3 convolution
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate,
                              kernel_size=3, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Bottleneck
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        # Main convolution
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Concatenate with input
        return torch.cat([x, out], 1)

# Factory function for easy creation
def create_densenet(config: Dict[str, Any]) -> DenseNet:
    """
    Factory function to create DenseNet model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DenseNet model instance
    """
    return DenseNet(config)

def create_densenet_custom(config: Dict[str, Any]) -> DenseNetCustom:
    """
    Factory function to create custom DenseNet model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DenseNetCustom model instance
    """
    return DenseNetCustom(config)
