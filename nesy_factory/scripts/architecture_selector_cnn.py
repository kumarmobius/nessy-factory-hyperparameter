#!/usr/bin/env python3
"""
Architecture selection helper for CNN Factory.
Provides intelligent recommendations based on use cases and constraints.
"""

from typing import Dict, Any, List, Optional
from nesy_factory.CNNs.factory import CNNFactory

def recommend_architecture(use_case: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get detailed architecture recommendation for a use case.
    
    Args:
        use_case: Target use case
        constraints: Performance constraints
        
    Returns:
        Dictionary with recommendation details
    """
    constraints = constraints or {}
    
    # Get base recommendation
    architecture = CNNFactory.get_recommended_architecture(use_case, constraints)
    config = CNNFactory.get_use_case_config(use_case)
    
    # Architecture-specific details
    arch_details = {
        'BaseCNN': {
            'description': 'Customizable CNN for general purposes',
            'best_for': ['Prototyping', 'Custom implementations', 'Learning'],
            'parameters': '1M-10M',
            'inference_speed': 'Medium'
        },
        'EfficientNet': {
            'description': 'State-of-the-art efficiency and accuracy',
            'best_for': ['Production systems', 'Balanced requirements', 'Transfer learning'],
            'parameters': '5M-66M', 
            'inference_speed': 'Fast'
        },
        'ResNet': {
            'description': 'Proven architecture for deep networks',
            'best_for': ['High accuracy', 'Transfer learning', 'Research'],
            'parameters': '11M-60M',
            'inference_speed': 'Medium-Fast'
        },
        'MobileNet': {
            'description': 'Lightweight architecture for mobile/edge',
            'best_for': ['Real-time applications', 'Mobile devices', 'Low power'],
            'parameters': '2M-6M',
            'inference_speed': 'Very Fast'
        },
        'DenseNet': {
            'description': 'Parameter-efficient with feature reuse',
            'best_for': ['Feature-rich tasks', 'Parameter efficiency', 'Dense predictions'],
            'parameters': '8M-20M',
            'inference_speed': 'Medium'
        }
    }
    
    recommendation = {
        'architecture': architecture,
        'use_case': use_case,
        'config_template': config,
        'details': arch_details.get(architecture, {}),
        'constraints_applied': constraints
    }
    
    return recommendation

def compare_architectures(use_case: str, architectures: List[str]) -> Dict[str, Any]:
    """
    Compare multiple architectures for a specific use case.
    
    Args:
        use_case: Target use case
        architectures: List of architectures to compare
        
    Returns:
        Comparison results
    """
    config = CNNFactory.get_use_case_config(use_case)
    comparison = {}
    
    for arch in architectures:
        try:
            # Create model to get parameter count
            model_config = config.copy()
            model_config['architecture'] = arch
            model = CNNFactory.create_model(arch, model_config)
            
            comparison[arch] = {
                'parameters': model.get_num_parameters(),
                'config': model_config,
                'status': 'success'
            }
            
        except Exception as e:
            comparison[arch] = {
                'parameters': 0,
                'error': str(e),
                'status': 'failed'
            }
    
    return comparison

def get_architecture_info(architecture: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        Architecture information
    """
    info = {
        'SimpleCNN': {
            'type': 'Lightweight Educational',
            'parameters': '50K-2M',
            'use_cases': ['Education', 'Prototyping', 'Beginner projects', 'Simple tasks'],
            'strengths': ['Easy to understand', 'Fast training', 'Lightweight', 'Great for learning'],
            'limitations': ['Limited complexity', 'Not state-of-the-art for complex tasks'],
            'variants': ['SimpleCNN', 'SimpleCNNV2 (enhanced)']
        },
        'BaseCNN': {
            'type': 'Customizable',
            'parameters': 'Configurable',
            'use_cases': ['All purposes', 'Custom implementations', 'Education'],
            'strengths': ['Fully customizable', 'Simple interface', 'Educational'],
            'limitations': ['Manual optimization', 'No pre-trained weights'],
            'variants': ['Custom configurations only']
        },
        # ... keep existing architectures
    }
    
    return info.get(architecture, {})

def main():
    """Demo the architecture selection functionality."""
    print("🏗️  CNN ARCHITECTURE SELECTOR")
    print("=" * 50)
    
    # Demo recommendations for your use cases
    use_cases = [
        "visual_diff_testing",
        "accessibility_defects", 
        "chart_ocr",
        "creative_quality",
        "brand_safety"
    ]
    
    for use_case in use_cases:
        print(f"\n🎯 {use_case.replace('_', ' ').title()}:")
        recommendation = recommend_architecture(use_case)
        print(f"   Recommended: {recommendation['architecture']}")
        print(f"   Description: {recommendation['details'].get('description', 'N/A')}")
        print(f"   Best for: {', '.join(recommendation['details'].get('best_for', []))}")
    
    # Demo comparison
    print(f"\n📊 COMPARISON FOR UI ANALYSIS:")
    comparison = compare_architectures("visual_diff_testing", ["BaseCNN", "EfficientNet", "MobileNet"])
    for arch, info in comparison.items():
        if info['status'] == 'success':
            print(f"   {arch}: {info['parameters']:,} parameters")
        else:
            print(f"   {arch}: Failed - {info['error']}")

if __name__ == "__main__":

    main()
