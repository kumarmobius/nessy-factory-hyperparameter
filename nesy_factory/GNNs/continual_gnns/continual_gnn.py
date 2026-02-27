"""
Continual Learning GNN Wrapper

This module provides a wrapper for GNN models to enable continual learning capabilities
using the Avalanche framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from avalanche.models import BaseModel
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training import Naive, EWC, SynapticIntelligence, LwF

from ..base import BaseGNN


class ContinualGNN(BaseModel):
    """
    Wrapper class that enables continual learning for any GNN model.
    
    This class wraps a BaseGNN model to make it compatible with Avalanche's
    continual learning framework while preserving the graph-specific functionality.
    """
    
    def __init__(self, base_gnn: BaseGNN, num_classes: Optional[int] = None):
        """
        Initialize the continual learning GNN wrapper.
        
        Args:
            base_gnn: The base GNN model to wrap
            num_classes: Number of output classes (if different from base_gnn.output_dim)
        """
        super().__init__()
        
        self.base_gnn = base_gnn
        self.num_classes = num_classes or base_gnn.output_dim
        
        # Store original forward method
        self._original_forward = base_gnn.forward
        
        # Ensure the model has the required attributes for Avalanche
        if not hasattr(self, 'classifier'):
            # If the base GNN doesn't have a separate classifier, 
            # we'll use the whole model as the classifier
            self.classifier = self.base_gnn
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass that handles both graph data and standard tensor input.
        
        Args:
            x: Input tensor (node features or standard features)
            edge_index: Edge indices for graph data (optional)
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
        # If edge_index is provided, use graph-specific forward
        if edge_index is not None:
            return self.base_gnn.forward(x, edge_index, **kwargs)
        else:
            # For compatibility with Avalanche scenarios that don't use graphs
            # We might need to adapt the input format
            if hasattr(kwargs, 'data') and hasattr(kwargs['data'], 'edge_index'):
                return self.base_gnn.forward(x, kwargs['data'].edge_index, **kwargs)
            else:
                # If no graph structure, treat as regular tensor input
                # This might require the base GNN to handle this case
                return self.base_gnn.forward(x, torch.empty((2, 0), dtype=torch.long, device=x.device), **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the model callable."""
        return self.forward(*args, **kwargs)
    
    def get_features(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Extract features from the model. Required by Avalanche's BaseModel.
        
        Args:
            x: Input tensor (node features or standard features)
            edge_index: Edge indices for graph data (optional)
            **kwargs: Additional arguments
            
        Returns:
            Feature tensor
        """
        # For GNNs, we'll return the node embeddings before the final classification layer
        if hasattr(self.base_gnn, 'get_node_embeddings'):
            return self.base_gnn.get_node_embeddings(x, edge_index, **kwargs)
        else:
            # Fallback: run forward pass and return features
            with torch.no_grad():
                return self.forward(x, edge_index, **kwargs)
    
    def get_base_model(self) -> BaseGNN:
        """Get the underlying base GNN model."""
        return self.base_gnn
    
    def parameters(self, recurse: bool = True):
        """Return parameters from the base GNN model."""
        return self.base_gnn.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters from the base GNN model."""
        return self.base_gnn.named_parameters(prefix=prefix, recurse=recurse)
    
    def children(self):
        """Return children modules."""
        return self.base_gnn.children()
    
    def named_children(self):
        """Return named children modules."""
        return self.base_gnn.named_children()
    
    def modules(self):
        """Return all modules."""
        yield self
        yield from self.base_gnn.modules()
    
    def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
        """Return named modules."""
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
        for name, module in self.base_gnn.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate):
            yield name, module
    
    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self.base_gnn.train(mode)
        self.training = mode
        return self
    
    def eval(self):
        """Set the module in evaluation mode."""
        self.base_gnn.eval()
        self.training = False
        return self
    
    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.base_gnn.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        """Move model to CUDA."""
        self.base_gnn.cuda(device)
        return self
    
    def cpu(self):
        """Move model to CPU."""
        self.base_gnn.cpu()
        return self
    
    def reset_parameters(self):
        """Reset all parameters."""
        if hasattr(self.base_gnn, 'reset_parameters'):
            self.base_gnn.reset_parameters()
    
    def train_step(self, data, mask=None) -> float:
        """Delegate training step to base GNN."""
        return self.base_gnn.train_step(data, mask)
    
    def eval_step(self, data, mask=None) -> Dict[str, float]:
        """Delegate evaluation step to base GNN."""
        return self.base_gnn.eval_step(data, mask)
    
    def predict(self, data) -> torch.Tensor:
        """Delegate prediction to base GNN."""
        return self.base_gnn.predict(data)
    
    def save_model(self, path: str):
        """Save the model including continual learning state."""
        checkpoint = {
            'base_gnn_state_dict': self.base_gnn.state_dict(),
            'base_gnn_config': self.base_gnn.config,
            'num_classes': self.num_classes,
            'model_class': self.base_gnn.__class__.__name__
        }
        torch.save(checkpoint, path)
        print(f"Continual GNN model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path, map_location=self.base_gnn.device)
        self.base_gnn.load_state_dict(checkpoint['base_gnn_state_dict'])
        self.num_classes = checkpoint['num_classes']
        print(f"Continual GNN model loaded from {path}")


def create_continual_gnn(base_gnn: BaseGNN, num_classes: Optional[int] = None) -> ContinualGNN:
    """
    Factory function to create a continual learning GNN.
    
    Args:
        base_gnn: The base GNN model to wrap
        num_classes: Number of output classes
        
    Returns:
        ContinualGNN wrapper instance
    """
    return ContinualGNN(base_gnn, num_classes)


def setup_continual_learning_environment(
    model: ContinualGNN,
    strategy_name: str = 'naive',
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    evaluation_metrics: Optional[List[str]] = None
) -> tuple:
    """
    Set up a complete continual learning environment with model, strategy, and evaluation.
    
    Args:
        model: The continual GNN model
        strategy_name: Name of the continual learning strategy
        strategy_kwargs: Additional arguments for the strategy
        evaluation_metrics: List of metrics to evaluate
        
    Returns:
        Tuple of (strategy, evaluation_plugin)
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}
    
    if evaluation_metrics is None:
        evaluation_metrics = ['accuracy', 'loss', 'forgetting']
    
    # Set up evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger()]
    )
    
    # Create strategy based on name
    if strategy_name.lower() == 'naive':
        strategy = Naive(
            model=model,
            optimizer=model.base_gnn.optimizer or torch.optim.Adam(model.parameters()),
            criterion=model.base_gnn.criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            **strategy_kwargs
        )
    elif strategy_name.lower() == 'ewc':
        strategy = EWC(
            model=model,
            optimizer=model.base_gnn.optimizer or torch.optim.Adam(model.parameters()),
            criterion=model.base_gnn.criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            ewc_lambda=strategy_kwargs.get('ewc_lambda', 0.4),
            **{k: v for k, v in strategy_kwargs.items() if k != 'ewc_lambda'}
        )
    elif strategy_name.lower() == 'si':
        strategy = SynapticIntelligence(
            model=model,
            optimizer=model.base_gnn.optimizer or torch.optim.Adam(model.parameters()),
            criterion=model.base_gnn.criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            si_lambda=strategy_kwargs.get('si_lambda', 0.01),
            **{k: v for k, v in strategy_kwargs.items() if k != 'si_lambda'}
        )
    elif strategy_name.lower() == 'lwf':
        strategy = LwF(
            model=model,
            optimizer=model.base_gnn.optimizer or torch.optim.Adam(model.parameters()),
            criterion=model.base_gnn.criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            alpha=strategy_kwargs.get('alpha', 1.0),
            temperature=strategy_kwargs.get('temperature', 2.0),
            **{k: v for k, v in strategy_kwargs.items() if k not in ['alpha', 'temperature']}
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Supported strategies: naive, ewc, si, lwf")
    
    return strategy, eval_plugin