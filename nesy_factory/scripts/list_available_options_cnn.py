#!/usr/bin/env python3
"""
Script to list available CNN architectures and configurations.
Now supports multiple architectures beyond BaseCNN.
"""

import sys
import os
from typing import Dict, Any

# Add the current directory to path to import CNNs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_available_architectures():
    """Print available CNN architectures."""
    print("🏗️  CNN FACTORY - AVAILABLE ARCHITECTURES")
    print("=" * 60)
    
    try:
        # FIXED IMPORT - import the function directly
        from nesy_factory.CNNs.registry import list_available_models
        list_available_models()  # This will now work without duplication
            
    except ImportError as e:
        print("❌ Could not load available models. Make sure the CNN package is properly installed.")
        print(f"Error: {e}")

def print_architecture_recommendations():
    """Print architecture recommendations for different use cases."""
    print("\n🎯 ARCHITECTURE RECOMMENDATIONS BY USE CASE")
    print("=" * 60)
    
    recommendations = {
        # Agent: Monet
        "Visual Diff Testing": {
            "primary": "EfficientNet-B3",
            "alternatives": ["ResNet-50", "BaseCNN"],
            "reason": "Good balance of accuracy and speed for UI comparison"
        },
        "Accessibility Detection": {
            "primary": "MobileNetV3", 
            "alternatives": ["EfficientNet-B2", "BaseCNN"],
            "reason": "Fast inference for real-time accessibility checking"
        },
        "Chart OCR": {
            "primary": "DenseNet-121",
            "alternatives": ["ResNet-34", "BaseCNN"],
            "reason": "Excellent feature reuse for text and chart patterns"
        },
        
        # Agent: RunRun
        "Screenshot Heatmaps": {
            "primary": "HRNet-W18", 
            "alternatives": ["UNet", "BaseCNN"],
            "reason": "Maintains high resolution for precise localization"
        },
        "Log Pattern Vision": {
            "primary": "MobileNetV3-Small",
            "alternatives": ["EfficientNet-B0", "BaseCNN"],
            "reason": "Lightweight for fast log processing"
        },
        "UI Latency Hotspots": {
            "primary": "EfficientNet-B1",
            "alternatives": ["ResNet-34", "BaseCNN"],
            "reason": "Good performance/efficiency balance"
        },
        
        # Agent: AdWise
        "Creative Quality": {
            "primary": "EfficientNet-B4",
            "alternatives": ["ResNet-101", "BaseCNN"],
            "reason": "Excellent for image quality assessment"
        },
        "Brand Safety": {
            "primary": "ResNet-101",
            "alternatives": ["DenseNet-169", "BaseCNN"],
            "reason": "Deep features for complex scene understanding"
        },
        "Saliency Analysis": {
            "primary": "UNet",
            "alternatives": ["HRNet-W18", "BaseCNN"],
            "reason": "Segmentation architecture for heatmap generation"
        }
    }
    
    for use_case, info in recommendations.items():
        print(f"\n🔹 {use_case}:")
        print(f"   Primary: {info['primary']}")
        print(f"   Alternatives: {', '.join(info['alternatives'])}")
        print(f"   Reason: {info['reason']}")

def print_architecture_comparison():
    """Print comparison of different architectures."""
    print("\n📊 ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    architectures = {
        'BaseCNN': {
            'params': '1M-10M',
            'speed': 'Medium',
            'accuracy': 'Good',
            'use_case': 'General purpose, custom implementations'
        },
        'ResNet': {
            'params': '11M-60M', 
            'speed': 'Medium-Fast',
            'accuracy': 'Excellent',
            'use_case': 'Deep networks, transfer learning'
        },
        'EfficientNet': {
            'params': '5M-66M',
            'speed': 'Fast',
            'accuracy': 'State-of-the-art',
            'use_case': 'Best accuracy/efficiency trade-off'
        },
        'MobileNet': {
            'params': '2M-6M',
            'speed': 'Very Fast', 
            'accuracy': 'Good',
            'use_case': 'Mobile/edge devices, real-time'
        },
        'DenseNet': {
            'params': '8M-20M',
            'speed': 'Medium',
            'accuracy': 'Excellent',
            'use_case': 'Feature reuse, parameter efficiency'
        }
    }
    
    print(f"{'Architecture':<12} {'Params':<10} {'Speed':<12} {'Accuracy':<12} {'Best For':<30}")
    print("-" * 80)
    
    for arch, info in architectures.items():
        print(f"{arch:<12} {info['params']:<10} {info['speed']:<12} {info['accuracy']:<12} {info['use_case']:<30}")

def print_quick_start_commands():
    """Print quick start commands for different architectures."""
    print("\n🚀 QUICK START COMMANDS")
    print("=" * 60)
    
    examples = [
        ("BaseCNN (Custom)", """from nesy_factory.CNNs import BaseCNN

class MyModel(BaseCNN):
    def forward(self, x):
        # Your custom implementation
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = MyModel(config)"""),
        
        ("ResNet-50", """from nesy_factory.CNNs.factory import CNNFactory

config = {
    'architecture': 'resnet',
    'variant': 'resnet50',
    'output_dim': 10,
    'pretrained': True
}
model = CNNFactory.create_model('resnet', config)"""),
        
        ("EfficientNet-B3", """from nesy_factory.CNNs.factory import CNNFactory

config = {
    'architecture': 'efficientnet', 
    'variant': 'efficientnet_b3',
    'output_dim': 2,
    'pretrained': True
}
model = CNNFactory.create_model('efficientnet', config)"""),
        
        ("MobileNetV3", """from nesy_factory.CNNs.factory import CNNFactory

config = {
    'architecture': 'mobilenet',
    'variant': 'mobilenet_v3_small', 
    'output_dim': 5,
    'pretrained': True
}
model = CNNFactory.create_model('mobilenet', config)""")
    ]
    
    for title, code in examples:
        print(f"\n{title}:")
        print(code)

def main():
    """Main function to display all information."""
    print_available_architectures()
    print_architecture_recommendations()
    print_architecture_comparison() 
    print_quick_start_commands()
    
    print("\n" + "=" * 60)
    print("💡 USAGE TIPS:")
    print("  • Use CNNFactory.create_model() for quick architecture creation")
    print("  • Use CNNFactory.get_recommended_architecture() for use case guidance")  
    print("  • All architectures inherit from BaseCNN - same interface")
    print("  • Pre-trained weights available for most architectures")
    print("=" * 60)

if __name__ == "__main__":

    main()
