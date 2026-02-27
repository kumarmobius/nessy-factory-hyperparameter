"""
Continual Learning GNNs Module

This module provides continual learning capabilities for GNN models using the Avalanche framework.
It includes wrapper classes and strategies for lifelong learning on graph data.
"""

from .continual_gnn import ContinualGNN
from .strategies import (
    EWCStrategy, 
    SynapticIntelligenceStrategy, 
    LearningWithoutForgettingStrategy,
    create_continual_learning_strategy
)
from .utils import (
    create_graph_scenario,
    GraphDatasetAdapter,
    evaluate_continual_model,
    visualize_continual_learning_results,
    create_synthetic_graph_scenario
)

# Try to import NIScenario utilities
try:
    from .ni_scenario_utils import (
        RealWorldDatasetLoader,
        create_ni_scenario_from_real_dataset,
        create_multi_dataset_ni_scenario,
        create_node_classification_ni_scenario,
        create_graph_classification_ni_scenario_with_targets,
        demonstrate_ni_scenario_usage
    )
    NI_SCENARIO_AVAILABLE = True
except ImportError:
    NI_SCENARIO_AVAILABLE = False
from .main import (
    perform_continual_learning,
    quick_comparison,
    create_gnn_for_continual_learning,
    run_continual_learning,
    compare_strategies
)

__all__ = [
    # Core classes
    'ContinualGNN',
    'EWCStrategy',
    'SynapticIntelligenceStrategy', 
    'LearningWithoutForgettingStrategy',
    
    # Strategy and scenario creation
    'create_continual_learning_strategy',
    'create_graph_scenario',
    'create_synthetic_graph_scenario',
    
    # Utilities
    'GraphDatasetAdapter',
    'evaluate_continual_model',
    'visualize_continual_learning_results',
    
    # High-level interfaces
    'perform_continual_learning',
    'quick_comparison',
    'create_gnn_for_continual_learning',
    'run_continual_learning',
    'compare_strategies'
]

# Add NIScenario utilities to __all__ if available
if NI_SCENARIO_AVAILABLE:
    __all__.extend([
        'RealWorldDatasetLoader',
        'create_ni_scenario_from_real_dataset', 
        'create_multi_dataset_ni_scenario',
        'create_node_classification_ni_scenario',
        'create_graph_classification_ni_scenario_with_targets',
        'demonstrate_ni_scenario_usage'
    ])