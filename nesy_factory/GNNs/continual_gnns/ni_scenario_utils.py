"""
NIScenario (New Instance) Utilities for Real-World Datasets

This module provides utilities for creating NIScenario instances from real-world datasets
using Avalanche's continual learning framework. NIScenario is particularly useful for
scenarios where new instances of the same classes appear over time, simulating realistic
continual learning situations.

Based on Avalanche documentation: 
https://avalanche-api.continualai.org/en/v0.6.0/generated/avalanche.benchmarks.scenarios.NIScenario.html
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence
from pathlib import Path
import warnings

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, Amazon, Coauthor
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from sklearn.model_selection import train_test_split

try:
    from avalanche.benchmarks.utils import AvalancheDataset, TaskAwareClassificationDataset
    from avalanche.benchmarks.scenarios import NIScenario
    from avalanche.benchmarks import ni_benchmark
    AVALANCHE_AVAILABLE = True
except ImportError:
    print("⚠️ Avalanche not available. Install with: pip install avalanche-lib")
    AVALANCHE_AVAILABLE = False

from .utils import convert_graph_data_to_tensor_dataset, GraphDatasetAdapter


class RealWorldDatasetLoader:
    """
    Loader for various real-world graph datasets with NIScenario support.
    
    Supports loading from multiple sources:
    - PyTorch Geometric datasets (TUDataset, Planetoid, etc.)
    - Custom datasets from files
    - Pre-processed datasets
    """
    
    def __init__(self, cache_dir: str = "/tmp/graph_datasets"):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset registry with metadata
        self.dataset_registry = {
            # TU Datasets (graph classification)
            'MUTAG': {
                'source': 'TUDataset',
                'task_type': 'graph_classification',
                'num_classes': 2,
                'avg_nodes': 18,
                'description': 'Mutagenic aromatic compounds'
            },
            'ENZYMES': {
                'source': 'TUDataset', 
                'task_type': 'graph_classification',
                'num_classes': 6,
                'avg_nodes': 33,
                'description': 'Protein tertiary structures'
            },
            'PROTEINS': {
                'source': 'TUDataset',
                'task_type': 'graph_classification', 
                'num_classes': 2,
                'avg_nodes': 39,
                'description': 'Protein structures'
            },
            'IMDB-BINARY': {
                'source': 'TUDataset',
                'task_type': 'graph_classification',
                'num_classes': 2,
                'avg_nodes': 20,
                'description': 'Movie collaboration networks'
            },
            'IMDB-MULTI': {
                'source': 'TUDataset',
                'task_type': 'graph_classification',
                'num_classes': 3,
                'avg_nodes': 13,
                'description': 'Movie collaboration networks (multi-class)'
            },
            'COLLAB': {
                'source': 'TUDataset',
                'task_type': 'graph_classification',
                'num_classes': 3,
                'avg_nodes': 74,
                'description': 'Scientific collaboration networks'
            },
            'REDDIT-BINARY': {
                'source': 'TUDataset',
                'task_type': 'graph_classification',
                'num_classes': 2,
                'avg_nodes': 430,
                'description': 'Reddit discussion threads'
            },
            
            # Planetoid datasets (node classification)
            'Cora': {
                'source': 'Planetoid',
                'task_type': 'node_classification',
                'num_classes': 7,
                'num_nodes': 2708,
                'description': 'Scientific publications'
            },
            'CiteSeer': {
                'source': 'Planetoid',
                'task_type': 'node_classification', 
                'num_classes': 6,
                'num_nodes': 3327,
                'description': 'Scientific publications'
            },
            'PubMed': {
                'source': 'Planetoid',
                'task_type': 'node_classification',
                'num_classes': 3,
                'num_nodes': 19717,
                'description': 'Biomedical publications'
            },
            
            # Amazon datasets
            'Amazon_Computers': {
                'source': 'Amazon',
                'task_type': 'node_classification',
                'num_classes': 10,
                'description': 'Amazon computer product co-purchasing'
            },
            'Amazon_Photo': {
                'source': 'Amazon',
                'task_type': 'node_classification', 
                'num_classes': 8,
                'description': 'Amazon photo product co-purchasing'
            },
            
            # Coauthor datasets
            'Coauthor_CS': {
                'source': 'Coauthor',
                'task_type': 'node_classification',
                'num_classes': 15,
                'description': 'Computer science co-authorship'
            },
            'Coauthor_Physics': {
                'source': 'Coauthor',
                'task_type': 'node_classification',
                'num_classes': 5,
                'description': 'Physics co-authorship'
            }
        }
    
    def load_dataset(self, dataset_name: str, **kwargs) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Tuple of (dataset, metadata)
        """
        if dataset_name not in self.dataset_registry:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available datasets: {list(self.dataset_registry.keys())}")
        
        config = self.dataset_registry[dataset_name]
        source = config['source']
        
        print(f"📁 Loading {dataset_name} from {source}...")
        
        if source == 'TUDataset':
            dataset = TUDataset(
                root=str(self.cache_dir / dataset_name),
                name=dataset_name,
                **kwargs
            )
        elif source == 'Planetoid':
            dataset = Planetoid(
                root=str(self.cache_dir / dataset_name),
                name=dataset_name,
                **kwargs
            )
        elif source == 'Amazon':
            name = dataset_name.split('_')[1]  # Extract 'Computers' or 'Photo'
            dataset = Amazon(
                root=str(self.cache_dir / dataset_name),
                name=name,
                **kwargs
            )
        elif source == 'Coauthor':
            name = dataset_name.split('_')[1]  # Extract 'CS' or 'Physics'
            dataset = Coauthor(
                root=str(self.cache_dir / dataset_name),
                name=name,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Update config with actual dataset statistics
        metadata = config.copy()
        metadata.update({
            'dataset_size': len(dataset),
            'actual_num_classes': self._get_num_classes(dataset),
            'feature_dim': self._get_feature_dim(dataset)
        })
        
        print(f"   ✓ Loaded {len(dataset)} samples")
        print(f"   ✓ Classes: {metadata['actual_num_classes']}")
        print(f"   ✓ Feature dim: {metadata['feature_dim']}")
        
        return dataset, metadata
    
    def _get_num_classes(self, dataset: Dataset) -> int:
        """Get the number of classes in the dataset."""
        try:
            # For graph-level tasks
            if hasattr(dataset[0], 'y') and dataset[0].y.dim() == 0:
                labels = [int(data.y.item()) for data in dataset]
                return len(set(labels))
            # For node-level tasks
            elif hasattr(dataset[0], 'y') and dataset[0].y.dim() == 1:
                return int(dataset[0].y.max().item()) + 1
            else:
                return 2  # Default binary classification
        except:
            return 2  # Default fallback
    
    def _get_feature_dim(self, dataset: Dataset) -> int:
        """Get the feature dimension of the dataset."""
        try:
            if hasattr(dataset[0], 'x') and dataset[0].x is not None:
                return dataset[0].x.shape[1]
            else:
                return 0  # No features
        except:
            return 0  # Fallback
    
    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all available datasets with their metadata."""
        return self.dataset_registry.copy()


def create_ni_scenario_from_real_dataset(
    dataset_name: str,
    n_experiences: int = 5,
    test_size: float = 0.2,
    task_labels: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    balance_experiences: bool = False,
    min_class_patterns_in_exp: int = 0,
    cache_dir: str = "/tmp/graph_datasets",
    max_samples: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[NIScenario, Dict[str, Any]]:
    """
    Create an NIScenario from a real-world dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        n_experiences: Number of experiences in the scenario
        test_size: Fraction of data to use for testing
        task_labels: Whether to use task labels
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        balance_experiences: Whether to balance experiences
        min_class_patterns_in_exp: Minimum patterns per class per experience
        cache_dir: Directory to cache datasets
        max_samples: Maximum number of samples to use (for large datasets)
        **dataset_kwargs: Additional arguments for dataset loading
        
    Returns:
        Tuple of (NIScenario, metadata)
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche is required for NIScenario creation. "
                         "Install with: pip install avalanche-lib")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Load the dataset
    loader = RealWorldDatasetLoader(cache_dir=cache_dir)
    dataset, metadata = loader.load_dataset(dataset_name, **dataset_kwargs)
    
    print(f"🔄 Creating NIScenario from {dataset_name}...")
    
    # Sample dataset if too large
    if max_samples is not None and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        sampled_data = [dataset[i] for i in sorted(indices)]
        dataset = sampled_data
        print(f"   ✓ Sampled to {max_samples} instances")
    
    # Convert to list if needed
    if not isinstance(dataset, list):
        dataset = [dataset[i] for i in range(len(dataset))]
    
    # Handle different task types
    task_type = metadata.get('task_type', 'graph_classification')
    
    if task_type == 'graph_classification':
        # For graph classification, each sample is a graph with a label
        graphs = []
        targets = []
        
        for data in dataset:
            # Ensure node features exist
            if not hasattr(data, 'x') or data.x is None:
                # Create dummy features if none exist
                data.x = torch.ones(data.num_nodes, 1)
            
            graphs.append(data)
            if hasattr(data, 'y'):
                targets.append(int(data.y.item()))
            else:
                targets.append(0)  # Default label
        
        # Convert to tensor dataset for Avalanche compatibility
        tensor_dataset = convert_graph_data_to_tensor_dataset(graphs, targets)
        
    elif task_type == 'node_classification':
        # For node classification, we need to adapt the approach
        # We'll treat each node as a separate instance
        graph_data = dataset[0]  # Assume single graph for node classification
        
        if not hasattr(graph_data, 'x') or graph_data.x is None:
            raise ValueError("Node features are required for node classification")
        
        # Extract node features and labels
        node_features = graph_data.x
        node_labels = graph_data.y
        
        # Create tensor dataset
        tensor_dataset = TensorDataset(node_features, node_labels)
        tensor_dataset.targets = node_labels
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Split into train and test
    total_size = len(tensor_dataset)
    train_size = int((1 - test_size) * total_size)
    test_size_actual = total_size - train_size
    
    # Create indices for splitting
    indices = list(range(total_size))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create train and test datasets
    train_dataset = Subset(tensor_dataset, train_indices)
    test_dataset = Subset(tensor_dataset, test_indices)
    
    # Convert to TaskAwareClassificationDataset for NIScenario
    # Add targets attribute to subsets if not present
    if not hasattr(train_dataset, 'targets'):
        train_targets = [tensor_dataset.targets[i] for i in train_indices]
        train_dataset.targets = torch.tensor(train_targets)
    
    if not hasattr(test_dataset, 'targets'):
        test_targets = [tensor_dataset.targets[i] for i in test_indices]
        test_dataset.targets = torch.tensor(test_targets)
    
    # Create AvalancheDataset instances with proper target formatting
    train_avalanche = AvalancheDataset([tensor_dataset], indices=train_indices)
    test_avalanche = AvalancheDataset([tensor_dataset], indices=test_indices)
    
    # Create NIScenario
    scenario = NIScenario(
        train_dataset=train_avalanche,
        test_dataset=test_avalanche,
        n_experiences=n_experiences,
        task_labels=task_labels,
        shuffle=shuffle,
        seed=seed,
        balance_experiences=balance_experiences,
        min_class_patterns_in_exp=min_class_patterns_in_exp
    )
    
    # Update metadata with scenario info
    scenario_metadata = metadata.copy()
    scenario_metadata.update({
        'n_experiences': n_experiences,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'task_labels': task_labels,
        'balance_experiences': balance_experiences,
        'scenario_type': 'NIScenario'
    })
    
    print(f"   ✓ Created NIScenario with {n_experiences} experiences")
    print(f"   ✓ Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    return scenario, scenario_metadata


def create_multi_dataset_ni_scenario(
    dataset_names: List[str],
    n_experiences: int = None,
    test_size: float = 0.2,
    task_labels: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    balance_experiences: bool = False,
    cache_dir: str = "/tmp/graph_datasets",
    max_samples_per_dataset: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[NIScenario, Dict[str, Any]]:
    """
    Create an NIScenario from multiple real-world datasets.
    
    This creates a more challenging scenario where different experiences
    contain data from different datasets, simulating domain shift.
    
    Args:
        dataset_names: List of dataset names to combine
        n_experiences: Number of experiences (defaults to number of datasets)
        test_size: Fraction of data to use for testing
        task_labels: Whether to use task labels
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        balance_experiences: Whether to balance experiences
        cache_dir: Directory to cache datasets
        max_samples_per_dataset: Maximum samples per dataset
        **dataset_kwargs: Additional arguments for dataset loading
        
    Returns:
        Tuple of (NIScenario, metadata)
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche is required for NIScenario creation.")
    
    if n_experiences is None:
        n_experiences = len(dataset_names)
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"🔄 Creating multi-dataset NIScenario from {len(dataset_names)} datasets...")
    
    # Load all datasets
    loader = RealWorldDatasetLoader(cache_dir=cache_dir)
    all_datasets = []
    all_targets = []
    combined_metadata = {
        'datasets': {},
        'total_samples': 0,
        'combined_classes': 0
    }
    
    current_class_offset = 0
    
    for dataset_name in dataset_names:
        dataset, metadata = loader.load_dataset(dataset_name, **dataset_kwargs)
        
        # Sample if needed
        if max_samples_per_dataset is not None and len(dataset) > max_samples_per_dataset:
            indices = np.random.choice(len(dataset), max_samples_per_dataset, replace=False)
            sampled_data = [dataset[i] for i in sorted(indices)]
            dataset = sampled_data
        
        # Convert to list if needed
        if not isinstance(dataset, list):
            dataset = [dataset[i] for i in range(len(dataset))]
        
        # Process based on task type
        task_type = metadata.get('task_type', 'graph_classification')
        
        if task_type == 'graph_classification':
            for data in dataset:
                # Ensure node features exist
                if not hasattr(data, 'x') or data.x is None:
                    data.x = torch.ones(data.num_nodes, 1)
                
                all_datasets.append(data)
                if hasattr(data, 'y'):
                    # Offset class labels to avoid conflicts between datasets
                    original_label = int(data.y.item())
                    new_label = original_label + current_class_offset
                    all_targets.append(new_label)
                else:
                    all_targets.append(current_class_offset)
        
        # Update class offset for next dataset
        current_class_offset += metadata.get('actual_num_classes', metadata.get('num_classes', 2))
        
        # Store metadata
        combined_metadata['datasets'][dataset_name] = metadata
        combined_metadata['total_samples'] += len(dataset)
    
    combined_metadata['combined_classes'] = current_class_offset
    
    # Convert to tensor dataset
    tensor_dataset = convert_graph_data_to_tensor_dataset(all_datasets, all_targets)
    
    # Split into train and test
    total_size = len(tensor_dataset)
    train_size = int((1 - test_size) * total_size)
    
    indices = list(range(total_size))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(tensor_dataset, train_indices)
    test_dataset = Subset(tensor_dataset, test_indices)
    
    # Add targets attributes
    train_targets = [tensor_dataset.targets[i] for i in train_indices]
    test_targets = [tensor_dataset.targets[i] for i in test_indices]
    train_dataset.targets = torch.tensor(train_targets)
    test_dataset.targets = torch.tensor(test_targets)
    
    # Create AvalancheDataset instances with proper target formatting
    train_avalanche = AvalancheDataset([tensor_dataset], indices=train_indices)
    test_avalanche = AvalancheDataset([tensor_dataset], indices=test_indices)
    
    # Create NIScenario
    scenario = NIScenario(
        train_dataset=train_avalanche,
        test_dataset=test_avalanche,
        n_experiences=n_experiences,
        task_labels=task_labels,
        shuffle=shuffle,
        seed=seed,
        balance_experiences=balance_experiences
    )
    
    # Update metadata
    combined_metadata.update({
        'n_experiences': n_experiences,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'scenario_type': 'Multi-dataset NIScenario',
        'source_datasets': dataset_names
    })
    
    print(f"   ✓ Created multi-dataset NIScenario with {n_experiences} experiences")
    print(f"   ✓ Combined {len(dataset_names)} datasets")
    print(f"   ✓ Total samples: {combined_metadata['total_samples']}")
    print(f"   ✓ Total classes: {combined_metadata['combined_classes']}")
    
    return scenario, combined_metadata


def create_node_classification_ni_scenario(
    dataset_name: str = 'Cora',
    n_experiences: int = 3,
    split_strategy: str = 'by_nodes',
    test_size: float = 0.2,
    task_labels: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    balance_experiences: bool = False,
    cache_dir: str = "/tmp/graph_datasets"
) -> Tuple[NIScenario, Dict[str, Any]]:
    """
    Create NIScenario from node classification dataset by splitting nodes into multiple instances.
    
    Args:
        dataset_name: Name of the node classification dataset ('Cora', 'CiteSeer', 'PubMed')
        n_experiences: Number of experiences to create
        split_strategy: How to split the dataset ('by_nodes', 'by_classes', 'temporal')
        test_size: Fraction of nodes to use for testing
        task_labels: Whether to use task labels
        shuffle: Whether to shuffle nodes before splitting
        seed: Random seed for reproducibility
        balance_experiences: Whether to balance experiences
        cache_dir: Directory to cache datasets
        
    Returns:
        Tuple of (NIScenario, metadata)
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche is required for NIScenario creation.")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"🔄 Creating node classification NIScenario from {dataset_name}...")
    
    # Load the dataset
    loader = RealWorldDatasetLoader(cache_dir=cache_dir)
    dataset, metadata = loader.load_dataset(dataset_name)
    
    # Get the graph (node classification datasets typically have one large graph)
    graph_data = dataset[0]
    
    if not hasattr(graph_data, 'x') or graph_data.x is None:
        raise ValueError("Node features are required for node classification")
    
    node_features = graph_data.x  # [num_nodes, feature_dim]
    node_labels = graph_data.y    # [num_nodes]
    num_nodes = node_features.shape[0]
    num_classes = int(node_labels.max().item()) + 1
    
    print(f"   ✓ Graph loaded: {num_nodes} nodes, {num_classes} classes")
    
    # Create node indices
    node_indices = list(range(num_nodes))
    if shuffle:
        np.random.shuffle(node_indices)
    
    # Split into train/test first
    train_size = int((1 - test_size) * num_nodes)
    train_node_indices = node_indices[:train_size]
    test_node_indices = node_indices[train_size:]
    
    print(f"   ✓ Split: {len(train_node_indices)} train nodes, {len(test_node_indices)} test nodes")
    
    # Create multiple instances based on split strategy
    if split_strategy == 'by_nodes':
        # Split nodes randomly into experiences
        nodes_per_exp = len(train_node_indices) // n_experiences
        train_experiences = []
        test_experiences = []
        
        for exp_id in range(n_experiences):
            start_idx = exp_id * nodes_per_exp
            if exp_id == n_experiences - 1:
                # Last experience gets remaining nodes
                end_idx = len(train_node_indices)
            else:
                end_idx = (exp_id + 1) * nodes_per_exp
            
            exp_train_nodes = train_node_indices[start_idx:end_idx]
            
            # For test, use proportional split
            test_nodes_per_exp = len(test_node_indices) // n_experiences
            test_start_idx = exp_id * test_nodes_per_exp
            if exp_id == n_experiences - 1:
                test_end_idx = len(test_node_indices)
            else:
                test_end_idx = (exp_id + 1) * test_nodes_per_exp
            
            exp_test_nodes = test_node_indices[test_start_idx:test_end_idx]
            
            train_experiences.append(exp_train_nodes)
            test_experiences.append(exp_test_nodes)
    
    elif split_strategy == 'by_classes':
        # Split by classes - each experience sees different classes
        unique_classes = torch.unique(node_labels)
        classes_per_exp = len(unique_classes) // n_experiences
        
        train_experiences = []
        test_experiences = []
        
        for exp_id in range(n_experiences):
            start_class = exp_id * classes_per_exp
            if exp_id == n_experiences - 1:
                end_class = len(unique_classes)
            else:
                end_class = (exp_id + 1) * classes_per_exp
            
            exp_classes = unique_classes[start_class:end_class]
            
            # Get nodes belonging to these classes
            exp_train_nodes = []
            exp_test_nodes = []
            
            for cls in exp_classes:
                cls_nodes = (node_labels == cls).nonzero().flatten().tolist()
                cls_train_nodes = [n for n in cls_nodes if n in train_node_indices]
                cls_test_nodes = [n for n in cls_nodes if n in test_node_indices]
                exp_train_nodes.extend(cls_train_nodes)
                exp_test_nodes.extend(cls_test_nodes)
            
            train_experiences.append(exp_train_nodes)
            test_experiences.append(exp_test_nodes)
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    # Create tensor datasets for each experience
    from torch.utils.data import TensorDataset
    
    train_datasets = []
    test_datasets = []
    
    for exp_id in range(n_experiences):
        exp_train_nodes = train_experiences[exp_id]
        exp_test_nodes = test_experiences[exp_id]
        
        if not exp_train_nodes or not exp_test_nodes:
            print(f"   ⚠️ Skipping experience {exp_id} (empty)")
            continue
        
        # Extract features and labels for this experience
        exp_train_features = node_features[exp_train_nodes]
        exp_train_labels = node_labels[exp_train_nodes]
        exp_test_features = node_features[exp_test_nodes]
        exp_test_labels = node_labels[exp_test_nodes]
        
        # Create TensorDatasets with proper targets attribute
        train_tensor_dataset = TensorDataset(exp_train_features, exp_train_labels)
        train_tensor_dataset.targets = exp_train_labels  # Add targets for Avalanche
        
        test_tensor_dataset = TensorDataset(exp_test_features, exp_test_labels)
        test_tensor_dataset.targets = exp_test_labels    # Add targets for Avalanche
        
        train_datasets.append(train_tensor_dataset)
        test_datasets.append(test_tensor_dataset)
        
        print(f"   ✓ Experience {exp_id}: {len(exp_train_nodes)} train nodes, {len(exp_test_nodes)} test nodes")
    
    # Create NIScenario using nc_benchmark (more reliable than direct NIScenario)
    from avalanche.benchmarks import nc_benchmark
    
    scenario = nc_benchmark(
        train_datasets,
        test_datasets,
        n_experiences=len(train_datasets),
        task_labels=task_labels,
        shuffle=False,  # Already handled
        seed=seed
    )
    
    # Create metadata
    scenario_metadata = {
        'dataset_name': dataset_name,
        'original_dataset_info': metadata,
        'split_strategy': split_strategy,
        'n_experiences': len(train_datasets),
        'total_nodes': num_nodes,
        'total_classes': num_classes,
        'train_nodes': len(train_node_indices),
        'test_nodes': len(test_node_indices),
        'feature_dim': node_features.shape[1],
        'scenario_type': 'Node Classification NIScenario',
        'experiences_info': [
            {
                'exp_id': i,
                'train_nodes': len(train_datasets[i]),
                'test_nodes': len(test_datasets[i]),
                'train_classes': len(torch.unique(train_datasets[i].targets)),
                'test_classes': len(torch.unique(test_datasets[i].targets))
            }
            for i in range(len(train_datasets))
        ]
    }
    
    print(f"   ✓ Created NIScenario with {len(train_datasets)} experiences")
    print(f"   ✓ Strategy: {split_strategy}")
    print(f"   ✓ Feature dimension: {node_features.shape[1]}")
    
    return scenario, scenario_metadata


def create_graph_classification_ni_scenario_with_targets(
    dataset_name: str = 'MUTAG',
    n_experiences: int = 3,
    test_size: float = 0.2,
    task_labels: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    balance_experiences: bool = False,
    cache_dir: str = "/tmp/graph_datasets",
    max_samples: Optional[int] = None
) -> Tuple[NIScenario, Dict[str, Any]]:
    """
    Create NIScenario from graph classification dataset with proper targets field.
    
    This is an improved version that ensures the targets field is properly set.
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche is required for NIScenario creation.")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"🔄 Creating graph classification NIScenario from {dataset_name}...")
    
    # Load the dataset
    loader = RealWorldDatasetLoader(cache_dir=cache_dir)
    dataset, metadata = loader.load_dataset(dataset_name)
    
    # Sample if needed
    if max_samples is not None and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        sampled_data = [dataset[i] for i in sorted(indices)]
        dataset = sampled_data
        print(f"   ✓ Sampled to {max_samples} instances")
    
    # Convert to list if needed
    if not isinstance(dataset, list):
        dataset = [dataset[i] for i in range(len(dataset))]
    
    # Extract features and labels
    graphs = []
    targets = []
    
    for data in dataset:
        # Ensure node features exist
        if not hasattr(data, 'x') or data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        
        graphs.append(data)
        if hasattr(data, 'y'):
            targets.append(int(data.y.item()))
        else:
            targets.append(0)
    
    # Convert to tensor dataset with enhanced features
    tensor_dataset = convert_graph_data_to_tensor_dataset(graphs, targets)
    
    # Ensure targets attribute is set correctly
    if not hasattr(tensor_dataset, 'targets'):
        tensor_dataset.targets = torch.tensor(targets, dtype=torch.long)
    
    print(f"   ✓ Converted to tensor dataset: {len(tensor_dataset)} samples")
    print(f"   ✓ Targets field present: {hasattr(tensor_dataset, 'targets')}")
    print(f"   ✓ Targets shape: {tensor_dataset.targets.shape}")
    print(f"   ✓ Unique classes: {torch.unique(tensor_dataset.targets)}")
    
    # Split into train/test
    total_size = len(tensor_dataset)
    train_size = int((1 - test_size) * total_size)
    
    indices = list(range(total_size))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create experience datasets by splitting classes
    unique_classes = torch.unique(tensor_dataset.targets)
    num_classes = len(unique_classes)
    classes_per_exp = max(1, num_classes // n_experiences)
    
    from torch.utils.data import TensorDataset, Subset
    
    train_datasets = []
    test_datasets = []
    
    for exp_id in range(n_experiences):
        start_class = exp_id * classes_per_exp
        if exp_id == n_experiences - 1:
            end_class = num_classes
        else:
            end_class = (exp_id + 1) * classes_per_exp
        
        exp_classes = unique_classes[start_class:end_class]
        
        # Get indices for this experience's classes
        exp_train_indices = []
        exp_test_indices = []
        
        for cls in exp_classes:
            cls_indices = (tensor_dataset.targets == cls).nonzero().flatten().tolist()
            cls_train_indices = [idx for idx in cls_indices if idx in train_indices]
            cls_test_indices = [idx for idx in cls_indices if idx in test_indices]
            exp_train_indices.extend(cls_train_indices)
            exp_test_indices.extend(cls_test_indices)
        
        if exp_train_indices and exp_test_indices:
            # Create subsets
            exp_train_subset = Subset(tensor_dataset, exp_train_indices)
            exp_test_subset = Subset(tensor_dataset, exp_test_indices)
            
            # Add targets attribute to subsets (required by Avalanche)
            exp_train_subset.targets = tensor_dataset.targets[exp_train_indices]
            exp_test_subset.targets = tensor_dataset.targets[exp_test_indices]
            
            train_datasets.append(exp_train_subset)
            test_datasets.append(exp_test_subset)
            
            print(f"   ✓ Experience {exp_id}: classes {exp_classes.tolist()}, "
                  f"{len(exp_train_indices)} train, {len(exp_test_indices)} test")
    
    # Create scenario using nc_benchmark (more reliable)
    from avalanche.benchmarks import nc_benchmark
    
    scenario = nc_benchmark(
        train_datasets,
        test_datasets,
        n_experiences=len(train_datasets),
        task_labels=task_labels,
        shuffle=False,  # Already handled
        seed=seed
    )
    
    # Create metadata
    scenario_metadata = metadata.copy()
    scenario_metadata.update({
        'dataset_name': dataset_name,
        'n_experiences': len(train_datasets),
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'total_classes': num_classes,
        'scenario_type': 'Graph Classification NIScenario with Targets',
        'experiences_info': [
            {
                'exp_id': i,
                'train_samples': len(train_datasets[i]),
                'test_samples': len(test_datasets[i]),
                'classes': torch.unique(train_datasets[i].targets).tolist()
            }
            for i in range(len(train_datasets))
        ]
    })
    
    print(f"   ✓ Created NIScenario with {len(train_datasets)} experiences")
    
    return scenario, scenario_metadata


def demonstrate_ni_scenario_usage():
    """
    Demonstrate how to use NIScenario with real-world datasets.
    """
    print("=== NIScenario Real-World Dataset Demo ===\n")
    
    # Example 1: Single dataset NIScenario
    print("1. Creating NIScenario from MUTAG dataset...")
    try:
        scenario, metadata = create_ni_scenario_from_real_dataset(
            dataset_name='MUTAG',
            n_experiences=3,
            test_size=0.2,
            shuffle=True,
            seed=42
        )
        
        print("   📊 Scenario created successfully!")
        print(f"   📊 Metadata: {metadata}")
        
        # Demonstrate iterating through experiences
        print("\n   Iterating through training experiences:")
        for i, experience in enumerate(scenario.train_stream):
            print(f"     Experience {i}: {len(experience.dataset)} samples")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Multi-dataset NIScenario  
    print("2. Creating multi-dataset NIScenario...")
    try:
        scenario, metadata = create_multi_dataset_ni_scenario(
            dataset_names=['MUTAG', 'ENZYMES'],
            n_experiences=4,
            test_size=0.2,
            max_samples_per_dataset=50,  # Limit for demo
            seed=42
        )
        
        print("   📊 Multi-dataset scenario created successfully!")
        print(f"   📊 Source datasets: {metadata['source_datasets']}")
        print(f"   📊 Total classes: {metadata['combined_classes']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: List available datasets
    print("3. Available datasets:")
    loader = RealWorldDatasetLoader()
    datasets = loader.list_available_datasets()
    
    for name, info in datasets.items():
        print(f"   📚 {name}: {info['description']}")
        print(f"      Source: {info['source']}, Classes: {info['num_classes']}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_ni_scenario_usage()
