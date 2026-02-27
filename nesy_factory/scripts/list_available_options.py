#!/usr/bin/env python3
"""
Script to list all available models, optimizers, and configurations in the GNN Factory.
Run this script to get an up-to-date overview of what's available.
"""

import sys
import os
from typing import Dict, Any, List
import yaml

# Add the current directory to path to import GNNs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from GNNs import get_available_models, list_available_models
    from utils import load_yaml_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def print_models_info():
    """Print detailed information about available models."""
    print("📦 AVAILABLE MODELS")
    print("=" * 60)
    
    try:
        models_info = get_available_models()
        
        print(f"{'Model Name':<15} {'Description':<45} {'Default Config'}")
        print("-" * 80)
        
        for model_name, info in models_info.items():
            description = info.get('description', 'No description')[:40] + "..."
            default_config = info.get('default_config', {})
            default_optimizer = default_config.get('optimizer', 'N/A')
            default_lr = default_config.get('learning_rate', 'N/A')
            
            print(f"{model_name:<15} {description:<45} {default_optimizer} (lr={default_lr})")
            
    except Exception as e:
        print(f"Error getting model info: {e}")
        print("Available models might not be properly registered.")

def print_optimizers_info():
    """Print information about available optimizers."""
    print("\n⚙️  AVAILABLE OPTIMIZERS")
    print("=" * 60)
    
    # This info is from the BaseGNN class
    optimizers = {
        'adam': {
            'description': 'Adam Optimizer (Default)',
            'parameters': ['learning_rate', 'betas', 'eps', 'weight_decay'],
            'use_case': 'General purpose, adaptive learning rate'
        },
        'adamw': {
            'description': 'AdamW (Decoupled weight decay)',
            'parameters': ['learning_rate', 'betas', 'eps', 'weight_decay'],
            'use_case': 'Better regularization than Adam'
        },
        'sgd': {
            'description': 'Stochastic Gradient Descent',
            'parameters': ['learning_rate', 'momentum', 'weight_decay'],
            'use_case': 'Simple, reliable, good for large datasets'
        },
        'rmsprop': {
            'description': 'RMSprop Optimizer',
            'parameters': ['learning_rate', 'alpha', 'eps', 'weight_decay'],
            'use_case': 'Good for RNNs and non-stationary objectives'
        },
        'adagrad': {
            'description': 'Adagrad Optimizer',
            'parameters': ['learning_rate', 'eps', 'weight_decay'],
            'use_case': 'Sparse gradients, early stopping'
        }
    }
    
    print(f"{'Optimizer':<12} {'Description':<35} {'Key Parameters'}")
    print("-" * 80)
    
    for opt_name, opt_info in optimizers.items():
        params = ', '.join(opt_info['parameters'][:3]) + "..."
        print(f"{opt_name:<12} {opt_info['description']:<35} {params}")
    
    print("\nOptimizer Details:")
    for opt_name, opt_info in optimizers.items():
        print(f"  {opt_name}: {opt_info['use_case']}")

def print_configurations_info():
    """Print information about available configurations."""
    print("\n📋 AVAILABLE CONFIGURATIONS")
    print("=" * 60)
    
    config_files = [
        'configs/gcn_configs.yaml',
        'configs/gat_configs.yaml', 
        'configs/light_gcn_configs.yaml',
        'configs/pinsage_configs.yaml',
        'configs/rgcn_configs.yaml',
        'configs/tgcn_configs.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n📄 {config_file.upper()}")
            print("-" * 40)
            
            try:
                with open(config_file, 'r') as f:
                    configs = yaml.safe_load(f)
                
                # Filter out training and experiment configs
                model_configs = {k: v for k, v in configs.items() 
                               if not k.startswith(('training', 'experiment', 'sampling_configs', 'dataset_configs', 'optimization'))}
                
                print(f"{'Config Name':<25} {'Layers':<8} {'Hidden Dim':<20} {'Optimizer':<10}")
                print("-" * 65)
                
                for config_name, config in model_configs.items():
                    if isinstance(config, dict):
                        layers = config.get('num_layers', 'N/A')
                        hidden_dim = str(config.get('hidden_dim', 'N/A'))[:18]
                        optimizer = config.get('optimizer', 'N/A')
                        
                        print(f"{config_name:<25} {layers:<8} {hidden_dim:<20} {optimizer:<10}")
                        
            except Exception as e:
                print(f"Error reading {config_file}: {e}")
        else:
            print(f"Configuration file {config_file} not found")

def print_model_selection_guide():
    """Print model selection guide."""
    print("\n🎯 MODEL SELECTION GUIDE")
    print("=" * 60)
    
    print("\nBy Task Type:")
    task_recommendations = {
        'Node Classification': ['gcn', 'gat', 'gcn_skip'],
        'Graph Classification': ['gcn', 'gat'],
        'Link Prediction': ['rgcn', 'gcn'],
        'Recommendation Systems': ['lightgcn', 'pinsage'],
        'Temporal Analysis': ['tgcn'],
        'Large-Scale Graphs': ['pinsage', 'lightgcn']
    }
    
    for task, models in task_recommendations.items():
        print(f"  {task:<25}: {', '.join(models)}")
    
    print("\nBy Dataset Size:")
    size_recommendations = {
        'Small (< 1K nodes)': {'models': ['gcn', 'gat'], 'configs': 'lightweight_*'},
        'Medium (1K-100K nodes)': {'models': ['gcn', 'gat', 'lightgcn'], 'configs': 'basic_*'},
        'Large (100K+ nodes)': {'models': ['lightgcn', 'pinsage'], 'configs': 'large_scale_*'}
    }
    
    for size, rec in size_recommendations.items():
        print(f"  {size:<25}: {', '.join(rec['models'])} ({rec['configs']})")
    
    print("\nBy Performance Priority:")
    perf_recommendations = {
        'Speed': {'models': ['lightgcn', 'gcn'], 'configs': 'lightweight_*', 'optimizers': ['adam', 'sgd']},
        'Accuracy': {'models': ['gat', 'pinsage'], 'configs': 'heavy_*, deep_*', 'optimizers': ['adamw', 'adam']},
        'Balanced': {'models': ['gcn', 'gat'], 'configs': 'basic_*', 'optimizers': ['adam']}
    }
    
    for priority, rec in perf_recommendations.items():
        models_str = ', '.join(rec['models'])
        opts_str = ', '.join(rec['optimizers'])
        print(f"  {priority:<10}: {models_str} | {rec['configs']} | {opts_str}")

def print_usage_examples():
    """Print usage examples."""
    print("\n📖 QUICK USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Basic model creation", "model = get_model('gcn', 'basic_gcn')"),
        ("Lightweight model", "model = get_model('gcn', 'lightweight_gcn')"),
        ("Attention model", "model = get_model('gat', 'basic_gat')"),
        ("Recommendation model", "model = get_model('lightgcn', 'basic_lightgcn')"),
        ("Custom config", """config = {
    'input_dim': 128, 'hidden_dim': 64, 'output_dim': 10,
    'num_layers': 3, 'optimizer': 'adamw', 'learning_rate': 0.005
}
model = create_model('gcn', config)""")
    ]
    
    for title, code in examples:
        print(f"\n{title}:")
        print(f"  {code}")

def list_config_details(config_file: str = None):
    """List detailed configuration information for a specific file."""
    if config_file is None:
        print("\n🔍 Available configuration files:")
        config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]
        for i, cf in enumerate(config_files, 1):
            print(f"  {i}. {cf}")
        return
    
    config_path = f"configs/{config_file}" if not config_file.startswith('configs/') else config_file
    
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found")
        return
    
    print(f"\n📋 DETAILED CONFIGURATIONS: {config_path.upper()}")
    print("=" * 60)
    
    try:
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        # Filter out training and experiment configs
        model_configs = {k: v for k, v in configs.items() 
                       if not k.startswith(('training', 'experiment', 'sampling_configs', 'dataset_configs', 'optimization'))}
        
        for config_name, config in model_configs.items():
            if isinstance(config, dict):
                print(f"\n{config_name}:")
                for key, value in config.items():
                    if key.startswith('#') or key in ['input_dim']:  # Skip comments and input_dim
                        continue
                    print(f"  {key:<20}: {value}")
                        
    except Exception as e:
        print(f"Error reading {config_path}: {e}")

def main():
    """Main function to display all information."""
    print("🚀 GNN FACTORY - AVAILABLE MODELS AND OPTIMIZERS")
    print("=" * 80)
    
    try:
        print_models_info()
        print_optimizers_info()
        print_configurations_info()
        print_model_selection_guide()
        print_usage_examples()
        
        print("\n" + "=" * 80)
        print("💡 TIPS:")
        print("  • Start with 'basic_*' configurations for new projects")
        print("  • Use 'lightweight_*' for development and testing")
        print("  • Scale to 'heavy_*' or 'deep_*' for production")
        print("  • Import: from GNNs import get_model, create_model")
        print("  • Run: python list_available_options.py --detail gcn_configs.yaml")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error displaying information: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='List available GNN models and configurations')
    parser.add_argument('--detail', type=str, help='Show detailed info for specific config file')
    parser.add_argument('--models-only', action='store_true', help='Show only model information')
    parser.add_argument('--configs-only', action='store_true', help='Show only configuration information')
    
    args = parser.parse_args()
    
    if args.detail:
        list_config_details(args.detail)
    elif args.models_only:
        print_models_info()
    elif args.configs_only:
        print_configurations_info()
    else:
        main() 