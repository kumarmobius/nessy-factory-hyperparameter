"""
GNN Model Factory Package

This package provides implementations of various Graph Neural Network models
with a unified interface based on PyTorch and PyTorch Geometric.
It also includes continual learning capabilities using the Avalanche framework.
"""

from .base import BaseGNN
from .gcn import GCN, GCNWithSkipConnections, create_gcn, create_gcn_with_skip_connections
from .gat import GAT, GATWithSkipConnections, create_gat, create_gat_with_skip_connections
from .rgcn import RGCN
from .light_gcn import LightGCN, LightGCNWithFeatures, create_lightgcn, create_lightgcn_with_features
from .pinsage import PinSAGE, create_pinsage
from .boxe import BoxE, create_boxe

# Add STGNN (Spatio-Temporal GNN, optional)
try:
    from .stgnn import STGNN
    _STGNN_AVAILABLE = True
except ImportError:
    _STGNN_AVAILABLE = False

# Temporal GNN imports (optional, requires torch-geometric-temporal)
try:
    from .tgcn import TGCN
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False

from .registry import (
    register_model, create_model, create_model_from_config,
    list_available_models, get_available_models, is_model_available, get_model
)

# Continual Learning imports (optional, requires avalanche-lib)
try:
    from .continual_gnns import (
        ContinualGNN,
        EWCStrategy,
        SynapticIntelligenceStrategy,
        LearningWithoutForgettingStrategy,
        create_continual_learning_strategy,
        create_graph_scenario,
        GraphDatasetAdapter,
        evaluate_continual_model
    )
    _CONTINUAL_LEARNING_AVAILABLE = True
except ImportError:
    _CONTINUAL_LEARNING_AVAILABLE = False

__all__ = [
    # Base class
    'BaseGNN',
    # GCN models
    'GCN',
    'GCNWithSkipConnections',
    'create_gcn',
    'create_gcn_with_skip_connections',
    # GAT models
    'GAT',
    'GATWithSkipConnections',
    'create_gat',
    'create_gat_with_skip_connections',
    # RGCN model
    'RGCN',
    # LightGCN models
    'LightGCN',
    'LightGCNWithFeatures',
    'create_lightgcn',
    'create_lightgcn_with_features',
    # PinSAGE model
    'PinSAGE',
    'create_pinsage',
    # BoxE model
    'BoxE',
    'create_boxe',
    # Registry functions
    'register_model',
    'create_model',
    'create_model_from_config',
    'list_available_models',
    'get_available_models',
    'is_model_available',
    'get_model'
]

# Add temporal models if available
if _TEMPORAL_AVAILABLE:
    __all__.append('TGCN')

# Add STGNN if available
if _STGNN_AVAILABLE:
    __all__.append('STGNN')

# Add continual learning exports if available
if _CONTINUAL_LEARNING_AVAILABLE:
    __all__.extend([
        'ContinualGNN',
        'EWCStrategy',
        'SynapticIntelligenceStrategy',
        'LearningWithoutForgettingStrategy',
        'create_continual_learning_strategy',
        'create_graph_scenario',
        'GraphDatasetAdapter',
        'evaluate_continual_model'
    ])

__version__ = '1.0.0'
