"""
Continual Learning Strategies for GNNs

This module provides specialized continual learning strategies that are optimized 
for graph neural networks using the Avalanche framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from avalanche.training import EWC, SynapticIntelligence, LwF, Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger

from .continual_gnn import ContinualGNN


class EWCStrategy:
    """
    Elastic Weight Consolidation (EWC) strategy specialized for GNNs.
    
    EWC prevents catastrophic forgetting by adding a regularization term
    that preserves important weights from previous tasks.
    """
    
    def __init__(
        self,
        model: ContinualGNN,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        ewc_lambda: float = 0.4,
        mode: str = 'separate',
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        **kwargs
    ):
        """
        Initialize EWC strategy for GNNs.
        
        Args:
            model: The continual GNN model
            optimizer: Optimizer for training
            criterion: Loss criterion
            ewc_lambda: Regularization strength for EWC
            mode: EWC mode ('separate' or 'online')
            decay_factor: Decay factor for online EWC
            keep_importance_data: Whether to keep importance data
            **kwargs: Additional arguments
        """
        
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Set up evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), TextLogger()]
        )
        
        # Initialize EWC strategy
        self.strategy = EWC(
            model=model,
            optimizer=optimizer or torch.optim.Adam(model.parameters()),
            criterion=criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            ewc_lambda=ewc_lambda,
            mode=mode,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
            **kwargs
        )
    
    def train(self, experiences, eval_streams=None, **kwargs):
        """Train the model on a sequence of experiences."""
        return self.strategy.train(experiences, eval_streams, **kwargs)
    
    def eval(self, exp_list, **kwargs):
        """Evaluate the model on experiences."""
        return self.strategy.eval(exp_list, **kwargs)


class SynapticIntelligenceStrategy:
    """
    Synaptic Intelligence (SI) strategy specialized for GNNs.
    
    SI estimates parameter importance based on their contribution to the loss
    during training and protects important parameters.
    """
    
    def __init__(
        self,
        model: ContinualGNN,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        si_lambda: float = 0.01,
        eps: float = 0.001,
        **kwargs
    ):
        """
        Initialize Synaptic Intelligence strategy for GNNs.
        
        Args:
            model: The continual GNN model
            optimizer: Optimizer for training
            criterion: Loss criterion
            si_lambda: Regularization strength for SI
            eps: Small constant for numerical stability
            **kwargs: Additional arguments
        """
        self.model = model
        self.si_lambda = si_lambda
        
        # Set up evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), TextLogger()]
        )
        
        # Initialize SI strategy
        self.strategy = SynapticIntelligence(
            model=model,
            optimizer=optimizer or torch.optim.Adam(model.parameters()),
            criterion=criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            si_lambda=si_lambda,
            eps=eps,
            **kwargs
        )
    
    def train(self, experiences, eval_streams=None, **kwargs):
        """Train the model on a sequence of experiences."""
        return self.strategy.train(experiences, eval_streams, **kwargs)
    
    def eval(self, exp_list, **kwargs):
        """Evaluate the model on experiences."""
        return self.strategy.eval(exp_list, **kwargs)


class LearningWithoutForgettingStrategy:
    """
    Learning without Forgetting (LwF) strategy specialized for GNNs.
    
    LwF uses knowledge distillation to preserve knowledge from previous tasks
    while learning new tasks.
    """
    
    def __init__(
        self,
        model: ContinualGNN,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        alpha: float = 1.0,
        temperature: float = 2.0,
        **kwargs
    ):
        """
        Initialize Learning without Forgetting strategy for GNNs.
        
        Args:
            model: The continual GNN model
            optimizer: Optimizer for training
            criterion: Loss criterion
            alpha: Weight for distillation loss
            temperature: Temperature for knowledge distillation
            **kwargs: Additional arguments
        """
        self.model = model
        self.alpha = alpha
        self.temperature = temperature
        
        # Set up evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), TextLogger()]
        )
        
        # Initialize LwF strategy
        self.strategy = LwF(
            model=model,
            optimizer=optimizer or torch.optim.Adam(model.parameters()),
            criterion=criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            alpha=alpha,
            temperature=temperature,
            **kwargs
        )
    
    def train(self, experiences, eval_streams=None, **kwargs):
        """Train the model on a sequence of experiences."""
        return self.strategy.train(experiences, eval_streams, **kwargs)
    
    def eval(self, exp_list, **kwargs):
        """Evaluate the model on experiences."""
        return self.strategy.eval(exp_list, **kwargs)


class NaiveStrategy:
    """
    Naive continual learning strategy for GNNs.
    
    This strategy simply trains on new tasks without any mechanism to prevent
    catastrophic forgetting. Useful as a baseline.
    """
    
    def __init__(
        self,
        model: ContinualGNN,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        **kwargs
    ):
        """
        Initialize Naive strategy for GNNs.
        
        Args:
            model: The continual GNN model
            optimizer: Optimizer for training
            criterion: Loss criterion
            **kwargs: Additional arguments
        """
        self.model = model
        
        # Set up evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), TextLogger()]
        )
        
        # Initialize Naive strategy
        self.strategy = Naive(
            model=model,
            optimizer=optimizer or torch.optim.Adam(model.parameters()),
            criterion=criterion or nn.CrossEntropyLoss(),
            evaluator=eval_plugin,
            **kwargs
        )
    
    def train(self, experiences, eval_streams=None, **kwargs):
        """Train the model on a sequence of experiences."""
        return self.strategy.train(experiences, eval_streams, **kwargs)
    
    def eval(self, exp_list, **kwargs):
        """Evaluate the model on experiences."""
        return self.strategy.eval(exp_list, **kwargs)


def create_continual_learning_strategy(
    strategy_name: str,
    model: ContinualGNN,
    optimizer: torch.optim.Optimizer = None,
    criterion: nn.Module = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Factory function to create continual learning strategies for GNNs.
    
    Args:
        strategy_name: Name of the strategy ('ewc', 'si', 'lwf', 'naive')
        model: The continual GNN model
        optimizer: Optimizer for training
        criterion: Loss criterion
        strategy_params: Strategy-specific parameters
        **kwargs: Additional arguments
        
    Returns:
        Continual learning strategy instance
    """
    if strategy_params is None:
        strategy_params = {}
    
    strategy_name = strategy_name.lower()
    
    if strategy_name == 'ewc':
        return EWCStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **strategy_params,
            **kwargs
        )
    elif strategy_name == 'si':
        return SynapticIntelligenceStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **strategy_params,
            **kwargs
        )
    elif strategy_name == 'lwf':
        return LearningWithoutForgettingStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **strategy_params,
            **kwargs
        )
    elif strategy_name == 'naive':
        return NaiveStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **strategy_params,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Supported strategies: ewc, si, lwf, naive")


def compare_strategies(
    strategies: List[str],
    model_factory,
    experiences,
    eval_streams=None,
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compare multiple continual learning strategies on the same task sequence.
    
    Args:
        strategies: List of strategy names to compare
        model_factory: Function that creates fresh model instances
        experiences: Training experiences
        eval_streams: Evaluation streams
        strategy_params: Parameters for each strategy
        
    Returns:
        Dictionary containing results for each strategy
    """
    if strategy_params is None:
        strategy_params = {}
    
    results = {}
    
    for strategy_name in strategies:
        print(f"\nTraining with {strategy_name} strategy...")
        
        # Create fresh model
        model = model_factory()
        continual_model = ContinualGNN(model)
        
        # Create strategy
        strategy = create_continual_learning_strategy(
            strategy_name=strategy_name,
            model=continual_model,
            strategy_params=strategy_params.get(strategy_name, {})
        )
        
        # Train and evaluate
        strategy.train(experiences, eval_streams)
        eval_results = strategy.eval(experiences)
        
        results[strategy_name] = eval_results
        
        print(f"Completed {strategy_name} strategy training")
    
    return results