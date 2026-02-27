"""
BoxE model implementation as a wrapper around the PyKEEN library.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Import pykeen if available, otherwise this model will not be usable
try:
    from pykeen.pipeline import pipeline
    from pykeen.datasets import get_dataset
except ImportError:
    pipeline = None
    get_dataset = None

from .base import BaseGNN

class BoxE(BaseGNN):
    """
    Wrapper for the PyKEEN BoxE model to integrate with the GNN factory.
    
    This class is a bit different from others as it wraps a full PyKEEN pipeline.
    To train this model, you should call the `train_model()` method first, which
    runs the PyKEEN pipeline. After training, you can get embeddings or score triples.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BoxE wrapper.
        
        Args:
            config: Configuration dictionary. It should contain parameters for the
                    PyKEEN pipeline, such as 'dataset', 'model_kwargs', 'training_kwargs', etc.
        """
        if pipeline is None:
            raise ImportError("PyKEEN is not installed. Please install it to use the BoxE model.")

        # We call super().__init__ but many BaseGNN parameters might not be directly used.
        super().__init__(config)
        
        self.pykeen_result = None
        self.pykeen_model = None
        self.dataset = None

        # Parameters for PyKEEN pipeline from config
        self.pykeen_dataset_name = config.get('dataset', 'wn18rr')
        self.model_kwargs = config.get('model_kwargs', {'embedding_dim': 50})
        self.optimizer_name = config.get('optimizer', 'Adam')
        self.optimizer_kwargs = config.get('optimizer_kwargs', {'lr': 0.01})
        self.training_kwargs = config.get('training_kwargs', {'num_epochs': 100, 'batch_size': 8192})
        self.random_seed = config.get('random_seed', 42)

    def train_model(self):
        """
        Run the PyKEEN training pipeline to train the BoxE model.
        This method must be called before using `forward` or `decode`.
        """
        print(f"Loading dataset: {self.pykeen_dataset_name}")
        self.dataset = get_dataset(dataset=self.pykeen_dataset_name)
        
        print("Starting PyKEEN pipeline for BoxE...")
        self.pykeen_result = pipeline(
            dataset=self.dataset,
            model="BoxE",
            model_kwargs=self.model_kwargs,
            optimizer=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            training_kwargs=self.training_kwargs,
            device=self.device,
            random_seed=self.random_seed,
        )
        self.pykeen_model = self.pykeen_result.model
        print("PyKEEN pipeline finished.")

    def forward(self, x: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Returns the learned entity embeddings from the trained PyKEEN model.
        
        Note: The standard GNN inputs (x, edge_index) are ignored.
        
        Returns:
            A tensor of entity embeddings of shape [num_entities, embedding_dim].
        
        Raises:
            RuntimeError: If the model has not been trained yet via `train_model()`.
        """
        if self.pykeen_model is None:
            raise RuntimeError("Model not trained. Please call `train_model()` first.")
        
        # BoxE has one entity representation
        return self.pykeen_model.entity_representations[0]()

    def decode(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Scores triples using the trained BoxE model.
        
        Args:
            h: Head entity indices, shape [batch_size]
            r: Relation indices, shape [batch_size]
            t: Tail entity indices, shape [batch_size]
            
        Returns:
            Scores for each triple, shape [batch_size].
        """
        if self.pykeen_model is None:
            raise RuntimeError("Model not trained. Please call `train_model()` first.")
        
        hrt_batch = torch.stack([h, r, t], dim=1)
        return self.pykeen_model.score_hrt(hrt_batch)

    def train_step(self, data, mask=None) -> float:
        """
        This method is not applicable for the BoxE wrapper, as training is
        handled by the `train_model()` method which runs the full PyKEEN pipeline.
        Calling this will raise a NotImplementedError.
        """
        raise NotImplementedError("For BoxE, call `train_model()` to run the full training pipeline.")

    def eval_step(self, data, mask=None) -> Dict[str, float]:
        """
        This method is not applicable for the BoxE wrapper.
        Evaluation metrics are available in `self.pykeen_result.metric_results`.
        """
        raise NotImplementedError("For BoxE, evaluation is part of the pipeline. Access results via `self.pykeen_result`.")

def create_boxe(config: Dict[str, Any]) -> BoxE:
    """
    Convenience function to create a BoxE model wrapper.
    
    Args:
        config: Configuration dictionary for the BoxE model and PyKEEN pipeline.
        
    Returns:
        A BoxE model instance.
    """
    return BoxE(config)