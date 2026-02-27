#!/usr/bin/env python3
"""
Quick start examples for CNN Factory with multiple architecture support.
"""

import torch
import torch.nn as nn

def example_1_mnist():
    """Example 1: MNIST digit classification with BaseCNN"""
    print("🔢 Example 1: MNIST Digit Classification (BaseCNN)")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_mnist_config, create_model_from_config
    
    # Create configuration
    config = create_mnist_config(output_dim=10, architecture='BaseCNN')
    
    # Create model
    model = create_model_from_config(config)
    
    print(f"✅ Created MNIST model with BaseCNN")
    print(f"   Architecture: {config.get('architecture', 'BaseCNN')}")
    print(f"   Input: 1x28x28")
    print(f"   Output: 10 classes")
    print(f"   Parameters: {model.get_num_parameters():,}")
    
    # Test with sample input
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    print(f"   Sample output shape: {output.shape}")
    
    return model

def example_2_ui_diff_testing():
    """Example 2: UI Visual Diff Testing with EfficientNet"""
    print("\n🖥️  Example 2: UI Visual Diff Testing (EfficientNet)")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_ui_analysis_config, create_model_from_config
    
    # Create configuration for binary classification
    config = create_ui_analysis_config(output_dim=2, architecture='EfficientNet')
    
    # Create model
    model = create_model_from_config(config)
    
    print(f"✅ Created UI Diff Testing model")
    print(f"   Architecture: {config.get('architecture', 'BaseCNN')}")
    print(f"   Variant: {config.get('variant', 'N/A')}")
    print(f"   Input: 3x224x224 (RGB screenshots)")
    print(f"   Output: 2 classes (different/same)")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Pretrained: {config.get('pretrained', False)}")
    
    # Test with sample input
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(f"   Sample output shape: {output.shape}")
    
    return model

def example_3_creative_scoring():
    """Example 3: Creative Quality Scoring with ResNet"""
    print("\n🎨 Example 3: Creative Quality Scoring (ResNet)")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_creative_scoring_config, create_model_from_config
    
    # Create configuration for regression
    config = create_creative_scoring_config(output_dim=1, task_type='regression', architecture='ResNet')
    
    # Create model
    model = create_model_from_config(config)
    
    print(f"✅ Created Creative Scoring model")
    print(f"   Architecture: {config.get('architecture', 'BaseCNN')}")
    print(f"   Variant: {config.get('variant', 'N/A')}")
    print(f"   Input: 3x224x224 (creative images)")
    print(f"   Output: 1 value (quality score)")
    print(f"   Task type: {config['task_type']}")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Pretrained: {config.get('pretrained', False)}")
    
    # Test with sample input
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(f"   Sample output: {output.item():.4f}")
    
    return model

def example_4_feature_extraction():
    """Example 4: Feature Extraction with DenseNet"""
    print("\n🔍 Example 4: Feature Extraction for OCR (DenseNet)")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_ocr_config, create_model_from_config
    
    # Create configuration for chart classification
    config = create_ocr_config(output_dim=5, architecture='DenseNet')
    
    # Create model
    model = create_model_from_config(config)
    
    print(f"✅ Created OCR model with DenseNet")
    print(f"   Architecture: {config.get('architecture', 'BaseCNN')}")
    print(f"   Variant: {config.get('variant', 'N/A')}")
    print(f"   Input: 1x128x128 (grayscale charts)")
    print(f"   Output: 5 chart types")
    print(f"   Parameters: {model.get_num_parameters():,}")
    
    # Demonstrate feature extraction
    sample_input = torch.randn(1, 1, 128, 128)
    
    if hasattr(model, 'get_feature_maps'):
        feature_maps = model.get_feature_maps(sample_input, layer_idx=-2)
        print(f"   Feature maps shape: {feature_maps.shape}")
        print(f"   Can be used for visualization and analysis")
    else:
        print(f"   Feature extraction not available for this architecture")
    
    return model

def example_5_custom_implementation():
    """Example 5: Custom Implementation with BaseCNN"""
    print("\n⚡ Example 5: Custom Implementation (BaseCNN)")
    print("-" * 50)
    
    class CustomCNN(nn.Module):
        def __init__(self, config):
            super().__init__()
            from nesy_factory.CNNs import BaseCNN
            self.base_cnn = BaseCNN(config)
            
        def forward(self, x):
            # Custom forward pass with intermediate feature storage
            features = []
            
            for conv_block in self.base_cnn.conv_blocks:
                x = conv_block(x)
                features.append(x)
                x = self.base_cnn.pool(x)
            
            x = x.view(x.size(0), -1)
            x = self.base_cnn.classifier(x)
            
            return x, features
    
    # Create configuration
    from nesy_factory.scripts.model_initializer_cnn import create_cifar_config
    config = create_cifar_config(output_dim=10, architecture='BaseCNN')
    
    # Create custom model
    model = CustomCNN(config)
    
    print(f"✅ Created custom CNN with BaseCNN")
    print(f"   Returns both predictions and intermediate features")
    print(f"   Useful for detailed analysis and visualization")
    print(f"   Parameters: {model.base_cnn.get_num_parameters():,}")
    
    # Test
    sample_input = torch.randn(1, 3, 32, 32)
    predictions, features = model(sample_input)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Number of feature maps: {len(features)}")
    print(f"   First feature map shape: {features[0].shape}")
    
    return model

def example_6_architecture_comparison():
    """Example 6: Compare different architectures for the same task"""
    print("\n📊 Example 6: Architecture Comparison for UI Analysis")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_ui_analysis_config, create_model_from_config
    
    architectures = ['BaseCNN', 'EfficientNet', 'MobileNet', 'ResNet']
    
    sample_input = torch.randn(1, 3, 224, 224)
    
    for arch in architectures:
        try:
            print(f"\n🔹 Testing {arch}:")
            config = create_ui_analysis_config(architecture=arch)
            model = create_model_from_config(config)
            
            # Test inference
            with torch.no_grad():
                output = model(sample_input)
            
            print(f"   ✅ Successfully created and tested")
            print(f"   Parameters: {model.get_num_parameters():,}")
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n💡 Comparison complete!")
    print(f"   Different architectures offer different trade-offs")
    print(f"   Choose based on your accuracy, speed, and memory requirements")
def example_7_simple_cnn_education():
    """Example 7: SimpleCNN for Education and Prototyping"""
    print("\n🎓 Example 7: SimpleCNN for Education (SimpleCNN)")
    print("-" * 50)
    
    from nesy_factory.scripts.model_initializer_cnn import create_simple_cnn_config, create_model_from_config
    
    # Create configuration for educational purposes
    config = create_simple_cnn_config(output_dim=10, variant='simple_cnn')
    
    # Create model
    model = create_model_from_config(config)
    
    print(f"✅ Created SimpleCNN model for education")
    print(f"   Architecture: {config.get('architecture', 'SimpleCNN')}")
    print(f"   Input: 1x28x28 (MNIST-like)")
    print(f"   Output: 10 classes")
    print(f"   Convolutional layers: {config.get('num_conv_layers', 3)}")
    print(f"   Channels: {config.get('conv_channels', [32, 64, 128])}")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Perfect for: Education, prototyping, beginner projects")
    
    # Test with sample input
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    print(f"   Sample output shape: {output.shape}")
    
    # Demonstrate feature extraction
    if hasattr(model, 'get_feature_maps'):
        feature_maps = model.get_feature_maps(sample_input, layer_idx=0)
        print(f"   First layer feature maps: {feature_maps.shape}")
    
    return model

def main():
    """Run all quick start examples."""
    print("🎯 CNN FACTORY - QUICK START EXAMPLES")
    print("=" * 60)
    print("These examples show how to use different CNN architectures:")
    print("• SimpleCNN for education and prototyping")  # NEW
    print("• BaseCNN for custom implementations")
    print("• EfficientNet for UI analysis")
    print("• ResNet for creative scoring") 
    print("• DenseNet for OCR tasks")
    print("• MobileNet for lightweight applications")
    print("• Architecture comparison for informed choices")
    print()
    
    # Run examples
    example_1_mnist()
    example_2_ui_diff_testing() 
    example_3_creative_scoring()
    example_4_feature_extraction()
    example_5_custom_implementation()
    example_6_architecture_comparison()
    example_7_simple_cnn_education()  # NEW
    
    print("\n" + "=" * 60)
    print("🚀 Quick start examples completed!")
    print("💡 Next steps:")
    print("  1. Use list_available_options_cnn.py to see all architectures")
    print("  2. Use model_initializer_cnn.py for quick configuration")
    print("  3. Try different architectures for your specific use case")
    print("  4. Use CNNFactory for automatic architecture selection")
    print("  5. Start with SimpleCNN for education and prototyping")  # NEW
    print("=" * 60)

if __name__ == "__main__":

    main()
