"""
EfficientNet implementation extending BaseCNN.
"""

import torch
import torch.nn as nn
from .base import BaseCNN
from typing import Dict, Any, List, Tuple

class EfficientNet(BaseCNN):
    """
    EfficientNet implementation with compound scaling.
    
    Balances network depth, width, and resolution for optimal performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EfficientNet model.
        
        Args:
            config: Configuration dictionary
        """
        # EfficientNet-specific parameters
        self.variant = config.get('variant', 'efficientnet_b0')
        self.pretrained = config.get('pretrained', True)
        
        super().__init__(config)
    
    def _build_layers(self):
        """Build EfficientNet architecture."""
        try:
            import torchvision.models as models
            
            # Map variants to torchvision models with new weights API
            variant_map = {
                'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
                'efficientnet_b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
                'efficientnet_b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
                'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
                'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
                'efficientnet_b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
                'efficientnet_b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
                'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
            }
            
            if self.variant in variant_map:
                model_func, weights = variant_map[self.variant]
                
                if self.pretrained:
                    efficientnet_model = model_func(weights=weights)
                else:
                    efficientnet_model = model_func(weights=None)
                
                # Replace classifier
                in_features = efficientnet_model.classifier[1].in_features
                efficientnet_model.classifier[1] = nn.Linear(in_features, self.output_dim)
                
                self.model = efficientnet_model
                return
                
        except ImportError:
            print("torchvision not available, using custom EfficientNet implementation")
        except Exception as e:
            print(f"Torchvision EfficientNet failed: {e}, using custom implementation")
        
        # Custom implementation
        self._build_custom_efficientnet()
    
    def _build_custom_efficientnet(self):
        """Build custom EfficientNet implementation."""
        # MBConv block configurations for different variants
        mbconv_configs = {
            'efficientnet_b0': [
                # (expansion, channels, layers, kernel_size, stride)
                (1, 16, 1, 3, 1),
                (6, 24, 2, 3, 2),
                (6, 40, 2, 5, 2),
                (6, 80, 3, 3, 2),
                (6, 112, 3, 5, 1),
                (6, 192, 4, 5, 2),
                (6, 320, 1, 3, 1)
            ],
            'efficientnet_b3': [
                (1, 24, 1, 3, 1),
                (6, 32, 2, 3, 2),
                (6, 48, 2, 5, 2),
                (6, 96, 3, 3, 2),
                (6, 136, 3, 5, 1),
                (6, 232, 4, 5, 2),
                (6, 384, 1, 3, 1)
            ]
        }
        
        config = mbconv_configs.get(self.variant, mbconv_configs['efficientnet_b0'])
        
        # Build stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        input_channels = 32
        
        for expansion, channels, num_layers, kernel_size, stride in config:
            for i in range(num_layers):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(input_channels, channels, expansion, 
                               kernel_size, block_stride)
                )
                input_channels = channels
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(self.dropout_rate),
            nn.Flatten(),
            nn.Linear(1280, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EfficientNet."""
        if hasattr(self, 'model'):
            return self.model(x)
        
        # Custom forward
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block."""
    
    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride):
        super().__init__()
        
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        expanded_channels = in_channels * expansion
        
        layers = []
        
        # Expansion phase
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze and excitation
        layers.append(SELayer(expanded_channels))
        
        # Output phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class SELayer(nn.Module):
    """Squeeze and Excitation layer."""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y