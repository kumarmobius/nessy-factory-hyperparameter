"""
Main interface for continual learning with GNNs

This module provides a simplified interface for performing continual learning
on GNN models with minimal setup required.
"""


import torch
from typing import Dict, Any, List, Optional, Union
from ..base import BaseGNN
from ..gcn import GCN
from ..gat import GAT
from .continual_gnn import ContinualGNN
from .strategies import create_continual_learning_strategy
from .utils import create_graph_scenario, create_synthetic_graph_scenario, evaluate_continual_model


def perform_continual_learning(
    model_type: str = 'GCN',
    model_config: Optional[Dict[str, Any]] = None,
    strategy: str = 'ewc',
    strategy_params: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    scenario_config: Optional[Dict[str, Any]] = None,
    num_tasks: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Simplified interface for performing continual learning on GNN models.
    
    This function provides a one-stop solution for setting up and running
    continual learning experiments with GNNs.
    
    Args:
        model_type: Type of GNN model ('GCN', 'GAT', or custom BaseGNN instance)
        model_config: Configuration dictionary for the model
        strategy: Continual learning strategy ('naive', 'ewc', 'si', 'lwf')
        strategy_params: Parameters for the continual learning strategy
        dataset_name: Name of real dataset to use (None for synthetic)
        scenario_config: Configuration for scenario creation
        num_tasks: Number of tasks in the continual learning scenario
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing results from continual learning evaluation
        
    Example:
        >>> results = perform_continual_learning(
        ...     model_type='GCN',
        ...     model_config={'input_dim': 32, 'hidden_dim': 64, 'output_dim': 2},
        ...     strategy='ewc',
        ...     strategy_params={'ewc_lambda': 0.4},
        ...     num_tasks=3
        ... )
        >>> print(f"Final accuracy: {results['average_accuracy'][-1]:.3f}")
    """
    
    if verbose:
        print("🚀 Starting Continual Learning with GNNs")
        print(f"Model: {model_type}, Strategy: {strategy.upper()}, Tasks: {num_tasks}")
    
    # Set default configurations
    if model_config is None:
        model_config = {
            'input_dim': 32,
            'hidden_dim': 64,
            'output_dim': 2,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.01
        }
    
    if strategy_params is None:
        strategy_params = {}
    
    if scenario_config is None:
        scenario_config = {}
    
    # Create the base GNN model
    if verbose:
        print("📊 Creating GNN model...")
    
    if isinstance(model_type, str):
        if model_type.upper() == 'GCN':
            base_gnn = GCN(model_config)
        elif model_type.upper() == 'GAT':
            # Add GAT-specific defaults if not provided
            if 'heads' not in model_config:
                model_config['heads'] = 4
            if 'concat' not in model_config:
                model_config['concat'] = True
            base_gnn = GAT(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'GCN', 'GAT', or provide BaseGNN instance")
    elif isinstance(model_type, BaseGNN):
        base_gnn = model_type
    else:
        raise ValueError("model_type must be string ('GCN', 'GAT') or BaseGNN instance")
    
    if verbose:
        print(f"   ✓ Created {base_gnn.__class__.__name__} with {base_gnn.get_num_parameters():,} parameters")
    
    # Wrap for continual learning
    continual_model = ContinualGNN(base_gnn, num_classes=model_config.get('output_dim', 2))
    
    # Create scenario
    if verbose:
        print("🎯 Creating continual learning scenario...")
    
    if dataset_name is not None:
        # Use real dataset
        default_scenario_config = {
            'dataset_name': dataset_name,
            'num_tasks': num_tasks,
            'task_type': 'class_incremental',
            'shuffle': True,
            'seed': 42
        }
        default_scenario_config.update(scenario_config)
        scenario = create_graph_scenario(**default_scenario_config)
        if verbose:
            print(f"   ✓ Created scenario from {dataset_name} dataset")
    else:
        # Use synthetic dataset
        default_scenario_config = {
            'num_tasks': num_tasks,
            'graphs_per_task': 50,
            'num_nodes_range': (10, 30),
            'num_features': model_config.get('input_dim', 32),
            'num_classes': model_config.get('output_dim', 2),
            'edge_prob': 0.3,
            'seed': 42
        }
        default_scenario_config.update(scenario_config)
        scenario = create_synthetic_graph_scenario(**default_scenario_config)
        if verbose:
            print(f"   ✓ Created synthetic scenario with {num_tasks} tasks")
    
    # Create strategy
    if verbose:
        print(f"🧠 Setting up {strategy.upper()} strategy...")
    
    cl_strategy = create_continual_learning_strategy(
        strategy_name=strategy,
        model=continual_model,
        strategy_params=strategy_params
    )
    
    if verbose:
        print(f"   ✓ Strategy configured with parameters: {strategy_params}")
    
    # Train and evaluate
    if verbose:
        print("🏋️ Training continual learning model...")
    
    results = evaluate_continual_model(continual_model, cl_strategy, scenario)
    
    if verbose:
        print("✅ Training completed!")
        print("\n📈 Results Summary:")
        print(f"   Final Average Accuracy: {results['average_accuracy'][-1]:.3f}")
        print(f"   Final Forgetting: {results['forgetting'][-1]:.3f}")
        print(f"   Per-task Accuracies: {[f'{acc:.3f}' for acc in results['final_accuracies']]}")
    
    # Add additional metadata to results
    results['config'] = {
        'model_type': model_type if isinstance(model_type, str) else model_type.__class__.__name__,
        'model_config': model_config,
        'strategy': strategy,
        'strategy_params': strategy_params,
        'num_tasks': num_tasks,
        'dataset_name': dataset_name
    }
    
    return results


def quick_comparison(
    strategies: Optional[List[str]] = None,
    model_type: str = 'GCN',
    model_config: Optional[Dict[str, Any]] = None,
    num_tasks: int = 3,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Quickly compare multiple continual learning strategies.
    
    Args:
        strategies: List of strategy names to compare
        model_type: Type of GNN model to use
        model_config: Configuration for the model
        num_tasks: Number of tasks in the scenario
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results for each strategy
        
    Example:
        >>> results = quick_comparison(
        ...     strategies=['naive', 'ewc', 'si'],
        ...     model_type='GCN',
        ...     num_tasks=3
        ... )
        >>> for strategy, result in results.items():
        ...     print(f"{strategy}: {result['average_accuracy'][-1]:.3f}")
    """
    
    if strategies is None:
        strategies = ['naive', 'ewc', 'si', 'lwf']
    
    if verbose:
        print(f"🔬 Comparing {len(strategies)} continual learning strategies")
        print(f"Strategies: {', '.join(strategies)}")
    
    all_results = {}
    
    # Default strategy parameters
    default_strategy_params = {
        'ewc': {'ewc_lambda': 0.4},
        'si': {'si_lambda': 0.01},
        'lwf': {'alpha': 1.0, 'temperature': 2.0}
    }
    
    for strategy in strategies:
        if verbose:
            print(f"\n🧪 Testing {strategy.upper()} strategy...")
        
        strategy_params = default_strategy_params.get(strategy, {})
        
        results = perform_continual_learning(
            model_type=model_type,
            model_config=model_config,
            strategy=strategy,
            strategy_params=strategy_params,
            num_tasks=num_tasks,
            verbose=False  # Reduce verbosity for comparison
        )
        
        all_results[strategy] = results
        
        if verbose:
            final_acc = results['average_accuracy'][-1]
            final_forgetting = results['forgetting'][-1]
            print(f"   ✓ {strategy.upper()}: Accuracy={final_acc:.3f}, Forgetting={final_forgetting:.3f}")
    
    if verbose:
        print(f"\n🏆 Strategy Comparison Summary:")
        print("Strategy\t\tAccuracy\tForgetting")
        print("-" * 40)
        for strategy in strategies:
            results = all_results[strategy]
            acc = results['average_accuracy'][-1]
            forgetting = results['forgetting'][-1]
            print(f"{strategy.upper():<12}\t{acc:.3f}\t\t{forgetting:.3f}")
    
    return all_results


def create_gnn_for_continual_learning(
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    **kwargs
) -> ContinualGNN:
    """
    Factory function to create a GNN model ready for continual learning.
    
    Args:
        model_type: Type of GNN ('GCN', 'GAT')
        input_dim: Input feature dimension
        output_dim: Output dimension (number of classes)
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        **kwargs: Additional model-specific parameters
        
    Returns:
        ContinualGNN instance ready for continual learning
        
    Example:
        >>> model = create_gnn_for_continual_learning(
        ...     model_type='GCN',
        ...     input_dim=32,
        ...     output_dim=3,
        ...     hidden_dim=64
        ... )
        >>> print(f"Model has {model.get_base_model().get_num_parameters()} parameters")
    """
    
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': num_layers,
        'dropout': kwargs.get('dropout', 0.5),
        'learning_rate': kwargs.get('learning_rate', 0.01),
        **kwargs
    }
    

    if model_type.upper() == 'GCN':
        base_gnn = GCN(config)
    elif model_type.upper() == 'GAT':
        # Add GAT-specific defaults
        config.setdefault('heads', 4)
        config.setdefault('concat', True)
        base_gnn = GAT(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return ContinualGNN(base_gnn, num_classes=output_dim)


# Convenience aliases
run_continual_learning = perform_continual_learning
compare_strategies = quick_comparison


