"""
CNN Factory - Main factory class for creating CNN architectures.
"""

"""
CNN Factory - Main factory class for creating CNN architectures.
"""

from typing import Dict, Any, Optional
from .base import BaseCNN
from .registry import create_model, get_available_models, is_model_available
from typing import Dict, Any, List, Tuple

class CNNFactory:
    """
    Main factory class for creating CNN architectures with unified interface.
    """
    
    @staticmethod
    def create_model(architecture: str, config: Dict[str, Any], **kwargs) -> BaseCNN:
        """
        Create a CNN model instance.
        
        Args:
            architecture: Model architecture name
            config: Configuration dictionary
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Initialized model instance
        """
        # Update config with kwargs
        config.update(kwargs)
        
        # Use the registry to create the model
        if is_model_available(architecture):
            return create_model(architecture, config)
        else:
            raise ValueError(f"Architecture '{architecture}' not available. "
                           f"Available: {list(get_available_models().keys())}")
    
    @staticmethod
    def get_available_models() -> Dict[str, Any]:
        """
        Get all available models from the registry.
        
        Returns:
            Dictionary of available models
        """
        return get_available_models()

    @staticmethod
    def get_recommended_architecture(use_case: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """Get recommended architecture for specific use case."""
        constraints = constraints or {}
        
        # Use case to architecture mapping
        recommendations = {
            # Simple tasks and education
            "education": "SimpleCNN",
            "prototyping": "SimpleCNN",
            "beginner_projects": "SimpleCNN",
            
            # Agent: Monet
            "visual_diff_testing": "EfficientNet",
            "accessibility_defects": "MobileNet", 
            "chart_ocr": "DenseNet",
            
            # Agent: RunRun
            "screenshot_heatmaps": "UNet",
            "log_pattern_vision": "MobileNet",
            "ui_latency_hotspots": "EfficientNet",
            
            # Agent: AdWise
            "creative_quality": "EfficientNet",
            "brand_safety": "ResNet", 
            "saliency_analysis": "UNet",
            
            # Feature-rich tasks
            "medical_imaging": "DenseNet",
            "fine_grained_classification": "DenseNet",
            "texture_analysis": "DenseNet"
        }
        
        # Apply constraints
        base_arch = recommendations.get(use_case, "EfficientNet")
        
        if constraints.get('beginner_friendly') or constraints.get('educational'):
            return "SimpleCNN"
        elif constraints.get('feature_rich') or constraints.get('parameter_efficient'):
            return "DenseNet"
        elif constraints.get('high_accuracy'):
            return "ResNet"
        elif constraints.get('fast_inference'):
            return "MobileNet"
        elif constraints.get('memory_efficient'):
            return "MobileNet"
        elif constraints.get('simple') or constraints.get('prototyping'):
            return "SimpleCNN"
        
        return base_arch 
    @staticmethod
    def get_use_case_config(use_case: str) -> Dict[str, Any]:
        """
        Get pre-configured settings for specific use cases.
        
        Args:
            use_case: Target use case
            
        Returns:
            Configuration dictionary
        """
        configs = {
            "visual_diff_testing": {
                "input_size": [224, 224],
                "output_dim": 2,
                "pretrained": True,
                "task_type": "classification"
            },
            "accessibility_defects": {
                "input_size": [224, 224], 
                "output_dim": 4,
                "pretrained": True,
                "task_type": "classification"
            },
            "chart_ocr": {
                "input_size": [128, 128],
                "output_dim": 5,
                "pretrained": True,
                "task_type": "classification"
            },
            "screenshot_heatmaps": {
                "input_size": [256, 256],
                "output_dim": 1,
                "task_type": "segmentation",
                "use_attention": True
            },
            "creative_quality": {
                "input_size": [224, 224],
                "output_dim": 1,
                "task_type": "regression",
                "pretrained": True
            },
            "brand_safety": {
                "input_size": [224, 224],
                "output_dim": 3,
                "pretrained": True,
                "task_type": "classification"
            },
            "saliency_analysis": {
                "input_size": [256, 256],
                "output_dim": 1,
                "task_type": "regression"
            }
        }
        
        return configs.get(use_case, {})