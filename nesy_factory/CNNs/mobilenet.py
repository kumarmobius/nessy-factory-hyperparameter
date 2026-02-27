"""
MobileNet implementation extending BaseCNN.
"""

import torch
import torch.nn as nn
from .base import BaseCNN
from typing import Dict, Any, List, Tuple

class MobileNet(BaseCNN):
    """
    MobileNet implementation for efficient mobile vision tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MobileNet model.
        
        Args:
            config: Configuration dictionary
        """
        # Set default variant if not provided
        self.variant = config.get('variant', 'mobilenet_v3_small')
        self.pretrained = config.get('pretrained', True)
        
        # Initialize BaseCNN first
        super().__init__(config)
    
    def _build_layers(self):
        """Build MobileNet architecture."""
        try:
            import torchvision.models as models
            
            variant_map = {
                'mobilenet_v2': (models.mobilenet_v2, None),
                'mobilenet_v3_small': (models.mobilenet_v3_small, None),
                'mobilenet_v3_large': (models.mobilenet_v3_large, None)
            }
            
            if self.variant in variant_map:
                model_func, _ = variant_map[self.variant]
                mobilenet_model = model_func(pretrained=self.pretrained)
                
                # Replace classifier based on variant
                if self.variant == 'mobilenet_v2':
                    in_features = mobilenet_model.classifier[1].in_features
                    mobilenet_model.classifier[1] = nn.Linear(in_features, self.output_dim)
                else:  # v3 variants
                    in_features = mobilenet_model.classifier[3].in_features
                    mobilenet_model.classifier[3] = nn.Linear(in_features, self.output_dim)
                
                self.model = mobilenet_model
                return
                
        except ImportError as e:
            print("torchvision not available, using custom MobileNet implementation")
        except Exception as e:
            print(f"Torchvision MobileNet failed: {e}, using custom implementation")
        
        # Custom implementation
        self._build_custom_mobilenet()
    
    def _build_custom_mobilenet(self):
        """Build custom MobileNet implementation."""
        print(f"Building custom MobileNet: {self.variant}")
        
        # Basic MobileNet components
        if self.variant == 'mobilenet_v2':
            self._build_mobilenet_v2()
        else:
            self._build_mobilenet_v3()
    
    def _build_mobilenet_v2(self):
        """Build MobileNetV2 architecture."""
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks configuration for MobileNetV2
        config = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        ]
        
        input_channels = 32
        for t, c, n, s in config:
            output_channels = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channels, output_channels, stride, t)
                )
                input_channels = output_channels
        
        # Final layers
        self.features.append(
            nn.Conv2d(input_channels, 1280, 1, bias=False)
        )
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(nn.ReLU6(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.output_dim)
        )
    
    def _build_mobilenet_v3(self):
        """Build MobileNetV3 architecture."""
        # Simplified MobileNetV3 implementation
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(self.input_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            
            # Bottleneck blocks
            InvertedResidual(16, 16, 1, 1, use_hs=True),
            InvertedResidual(16, 24, 2, 4.5, use_hs=False),
            InvertedResidual(24, 24, 1, 3.7, use_hs=False),
            InvertedResidual(24, 40, 2, 4, use_hs=True),
            InvertedResidual(40, 40, 1, 6, use_hs=True),
            InvertedResidual(40, 80, 2, 6, use_hs=False),
            InvertedResidual(80, 80, 1, 3.5, use_hs=False),
        )
        
        if self.variant == 'mobilenet_v3_large':
            self.features.extend([
                InvertedResidual(80, 112, 1, 6, use_hs=True),
                InvertedResidual(112, 112, 1, 6, use_hs=True),
                InvertedResidual(112, 160, 2, 6, use_hs=True),
                InvertedResidual(160, 160, 1, 6, use_hs=True),
            ])
        
        # Final layers
        self.features.append(
            nn.Conv2d(160 if self.variant == 'mobilenet_v3_large' else 80, 
                     960, 1, bias=False)
        )
        self.features.append(nn.BatchNorm2d(960))
        self.features.append(nn.Hardswish(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(1280, self.output_dim, 1, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MobileNet."""
        if hasattr(self, 'model'):
            return self.model(x)
        
        x = self.features(x)
        if hasattr(self, 'classifier'):
            x = self.classifier(x)
            x = x.view(x.size(0), -1)
        return x

class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNet."""
    
    def __init__(self, in_channels, out_channels, stride, expansion, use_hs=False):
        super().__init__()
        
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        hidden_channels = int(round(in_channels * expansion))
        
        layers = []
        if expansion != 1:
            # Pointwise convolution for expansion
            layers.append(nn.Conv2d(in_channels, hidden_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.Hardswish(inplace=True) if use_hs else nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, 
                               stride, 1, groups=hidden_channels, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Hardswish(inplace=True) if use_hs else nn.ReLU6(inplace=True))
        
        # Pointwise convolution for projection
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
