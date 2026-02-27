"""
ResNet implementation extending BaseCNN.
"""

import torch
import torch.nn as nn
from .base import BaseCNN
from typing import Dict, Any, List, Tuple

class ResNet(BaseCNN):
    """
    Residual Network implementation.
    
    Supports various ResNet variants with skip connections for deep networks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ResNet model.
        
        Args:
            config: Configuration dictionary
        """
        # ResNet-specific parameters
        self.variant = config.get('variant', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.use_pretrained_features = config.get('use_pretrained_features', True)
        
        # Initialize BaseCNN first
        super().__init__(config)
    
    def _build_layers(self):
        """Build ResNet architecture."""
        # Use torchvision ResNet if available and pretrained
        try:
            import torchvision.models as models
            
            # Map variant names to torchvision models
            variant_map = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34, 
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152
            }
            
            if self.variant in variant_map:
                # Load pretrained ResNet
                resnet_model = variant_map[self.variant](pretrained=self.pretrained)
                
                # Replace the final fully connected layer
                in_features = resnet_model.fc.in_features
                resnet_model.fc = nn.Linear(in_features, self.output_dim)
                
                # Use the entire ResNet as our model
                self.model = resnet_model
                return
                
        except ImportError:
            print("torchvision not available, using custom ResNet implementation")
        
        # Custom ResNet implementation
        self._build_custom_resnet()
    
    def _build_custom_resnet(self):
        """Build custom ResNet when torchvision is not available."""
        # Basic ResNet components
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers based on variant
        self.layers = nn.ModuleList()
        
        if self.variant == 'resnet18':
            layers = [2, 2, 2, 2]
        elif self.variant == 'resnet34':
            layers = [3, 4, 6, 3]
        elif self.variant == 'resnet50':
            layers = [3, 4, 6, 3]
        elif self.variant == 'resnet101':
            layers = [3, 4, 23, 3]
        elif self.variant == 'resnet152':
            layers = [3, 8, 36, 3]
        else:
            layers = [2, 2, 2, 2]  # Default to ResNet18
        
        # Build ResNet layers
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        # Adaptive pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.output_dim)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a ResNet layer with residual blocks."""
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResNet."""
        if hasattr(self, 'model'):
            # Use torchvision model
            return self.model(x)
        
        # Custom ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        
        out += identity
        out = self.relu(out)
        
        return out