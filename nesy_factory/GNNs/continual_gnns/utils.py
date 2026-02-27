"""
Utility functions for continual learning with GNNs

This module provides utilities for creating scenarios, adapting datasets,
and evaluating continual learning models on graph data.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset, Planetoid
from torch.utils.data import TensorDataset, Subset
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import nc_benchmark, ni_benchmark, CLScenario
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .continual_gnn import ContinualGNN


def convert_graph_data_to_tensor_dataset(data_list, targets):
    """Convert graph data to tensor format for Avalanche compatibility"""
    # For simplicity, we'll flatten the graph features
    # In a real scenario, you might want to use graph pooling or other methods
    tensor_data = []
    max_features = 0
    
    # First pass: determine maximum feature dimension
    for graph_data in data_list:
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            features = graph_data.x.mean(dim=0)  # Global pooling
            max_features = max(max_features, features.size(0))
        else:
            max_features = max(max_features, 10)  # Default feature size
    
    # Second pass: create tensors with consistent dimensions
    for graph_data in data_list:
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            # Use node features - take mean pooling for graph-level tasks
            features = graph_data.x.mean(dim=0)  # Global pooling
            
            # Pad or truncate to max_features size
            if features.size(0) < max_features:
                # Pad with zeros
                padding = torch.zeros(max_features - features.size(0))
                features = torch.cat([features, padding])
            elif features.size(0) > max_features:
                # Truncate
                features = features[:max_features]
        else:
            # Create dummy features if no node features
            features = torch.randn(max_features)
        tensor_data.append(features)
    
    # Stack into tensor
    x_tensor = torch.stack(tensor_data)
    # Ensure targets are plain integers for better Avalanche compatibility
    y_targets = []
    for target in targets:
        if isinstance(target, torch.Tensor):
            y_targets.append(int(target.item()))
        else:
            y_targets.append(int(target))
    y_tensor = torch.tensor(y_targets, dtype=torch.long)
    
    # Create TensorDataset and add targets attribute for Avalanche compatibility
    tensor_dataset = TensorDataset(x_tensor, y_tensor)
    tensor_dataset.targets = y_tensor  # Add targets attribute that Avalanche expects
    
    return tensor_dataset


class GraphDatasetAdapter:
    """
    Adapter class to convert PyTorch Geometric datasets to Avalanche-compatible format.
    
    This class handles the conversion of graph datasets for use in continual learning
    scenarios with the Avalanche framework.
    """
    
    def __init__(self, dataset: Dataset, task_labels: Optional[List[int]] = None):
        """
        Initialize the graph dataset adapter.
        
        Args:
            dataset: PyTorch Geometric dataset
            task_labels: Optional task labels for each sample
        """
        self.dataset = dataset
        self.task_labels = task_labels or [0] * len(dataset)
        
    def to_avalanche_dataset(self, transform=None) -> AvalancheDataset:
        """
        Convert PyTorch Geometric dataset to Avalanche dataset.
        
        Args:
            transform: Optional transform to apply
            
        Returns:
            AvalancheDataset instance
        """
        # Extract data and targets
        data_list = []
        targets = []
        
        for i, graph_data in enumerate(self.dataset):
            # For graph-level tasks
            if hasattr(graph_data, 'y') and graph_data.y.dim() == 0:
                data_list.append(graph_data)
                targets.append(graph_data.y.item())
            # For node-level tasks
            elif hasattr(graph_data, 'y') and graph_data.y.dim() == 1:
                data_list.append(graph_data)
                targets.append(graph_data.y)
            else:
                data_list.append(graph_data)
                targets.append(0)  # Default target
        
        # Convert to tensor dataset for Avalanche compatibility
        tensor_dataset = convert_graph_data_to_tensor_dataset(data_list, targets)
        return AvalancheDataset([tensor_dataset])
    
    def split_by_tasks(self, num_tasks: int, shuffle: bool = True) -> List[AvalancheDataset]:
        """
        Split dataset into multiple tasks.
        
        Args:
            num_tasks: Number of tasks to create
            shuffle: Whether to shuffle the data before splitting
            
        Returns:
            List of AvalancheDataset instances, one per task
        """
        indices = list(range(len(self.dataset)))
        if shuffle:
            np.random.shuffle(indices)
        
        task_size = len(indices) // num_tasks
        task_datasets = []
        
        for i in range(num_tasks):
            start_idx = i * task_size
            end_idx = (i + 1) * task_size if i < num_tasks - 1 else len(indices)
            task_indices = indices[start_idx:end_idx]
            
            task_data = [self.dataset[idx] for idx in task_indices]
            task_targets = [self.dataset[idx].y.item() if self.dataset[idx].y.dim() == 0 
                           else self.dataset[idx].y for idx in task_indices]
            task_labels = [i] * len(task_indices)
            
            task_tensor_dataset = convert_graph_data_to_tensor_dataset(task_data, task_targets)
            task_dataset = AvalancheDataset([task_tensor_dataset])
            task_datasets.append(task_dataset)
        
        return task_datasets


def create_graph_scenario(
    dataset_name: str = 'MUTAG',
    num_tasks: int = 3,
    num_classes_per_task: int = 2,
    task_type: str = 'class_incremental',
    shuffle: bool = True,
    seed: int = 42
) -> CLScenario:
    """
    Create a continual learning scenario from graph datasets.
    
    Args:
        dataset_name: Name of the dataset ('MUTAG', 'ENZYMES', 'Cora', etc.)
        num_tasks: Number of tasks in the scenario
        num_classes_per_task: Number of classes per task (for class incremental)
        task_type: Type of continual learning scenario
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        
    Returns:
        CLScenario instance for continual learning
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset based on name
    if dataset_name in ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY']:
        dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create adapter
    adapter = GraphDatasetAdapter(dataset)
    
    if task_type == 'class_incremental':
        # Create class incremental scenario
        # Split classes across tasks
        unique_labels = torch.unique(torch.tensor([
            data.y.item() if data.y.dim() == 0 else data.y.max().item() 
            for data in dataset
        ]))
        
        num_classes = len(unique_labels)
        classes_per_task = num_classes // num_tasks
        
        train_datasets = []
        test_datasets = []
        
        for task_id in range(num_tasks):
            start_class = task_id * classes_per_task
            end_class = (task_id + 1) * classes_per_task if task_id < num_tasks - 1 else num_classes
            task_classes = unique_labels[start_class:end_class]
            
            # Filter data for this task
            task_indices = []
            for i, data in enumerate(dataset):
                label = data.y.item() if data.y.dim() == 0 else data.y.max().item()
                if label in task_classes:
                    task_indices.append(i)
            
            # Split into train/test
            train_indices, test_indices = train_test_split(
                task_indices, test_size=0.2, random_state=seed, shuffle=shuffle
            )
            
            # Create task datasets
            train_data = [dataset[idx] for idx in train_indices]
            test_data = [dataset[idx] for idx in test_indices]
            
            train_targets = [dataset[idx].y.item() if dataset[idx].y.dim() == 0 
                           else dataset[idx].y for idx in train_indices]
            test_targets = [dataset[idx].y.item() if dataset[idx].y.dim() == 0 
                          else dataset[idx].y for idx in test_indices]
            
            train_tensor_dataset = convert_graph_data_to_tensor_dataset(train_data, train_targets)
            test_tensor_dataset = convert_graph_data_to_tensor_dataset(test_data, test_targets)
            
            train_datasets.append(train_tensor_dataset)
            test_datasets.append(test_tensor_dataset)
        
        scenario = nc_benchmark(
            train_datasets,
            test_datasets,
            n_experiences=num_tasks,
            task_labels=True,
            shuffle=False  # We already handled shuffling
        )
    
    elif task_type == 'domain_incremental':
        # Create domain incremental scenario
        # Split dataset temporally or randomly
        adapter = GraphDatasetAdapter(dataset)
        task_datasets = adapter.split_by_tasks(num_tasks, shuffle=shuffle)
        
        train_datasets = []
        test_datasets = []
        
        for task_dataset in task_datasets:
            # Split each task into train/test
            # task_dataset is already an AvalancheDataset containing TensorDataset
            # Extract the underlying tensor dataset
            if hasattr(task_dataset, 'datasets') and len(task_dataset.datasets) > 0:
                underlying_dataset = task_dataset.datasets[0]
            else:
                underlying_dataset = task_dataset
            
            total_size = len(underlying_dataset)
            train_size = int(0.8 * total_size)
            
            train_indices = list(range(train_size))
            test_indices = list(range(train_size, total_size))
            
            # Create subset tensor datasets
            train_subset = Subset(underlying_dataset, train_indices)
            test_subset = Subset(underlying_dataset, test_indices)

            train_datasets.append(train_subset)
            test_datasets.append(test_subset)
        
        # Create scenario using the same approach as class incremental
        scenario = nc_benchmark(
            train_datasets,
            test_datasets,
            n_experiences=num_tasks,
            task_labels=True,
            shuffle=False
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return scenario


def evaluate_continual_model(
    model: ContinualGNN,
    strategy,
    scenario,  # Can be CLScenario or list of scenarios
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate a continual learning model on a scenario.
    
    Args:
        model: The continual GNN model
        strategy: The continual learning strategy
        scenario: The continual learning scenario or list of scenarios
        metrics: List of metrics to compute
        
    Returns:
        Dictionary containing evaluation results
    """
    if metrics is None:
        metrics = ['accuracy', 'loss', 'forgetting']
    
    results = {
        'task_accuracies': [],
        'average_accuracy': [],
        'forgetting': [],
        'final_accuracies': []
    }
    
    # Handle both single scenario and list of scenarios
    if isinstance(scenario, list):
        # Multi-task case: list of individual scenarios
        num_tasks = len(scenario)
        
        for i in range(num_tasks):
            print(f"Training on experience {i}")
            # Train on the i-th task
            current_scenario = scenario[i]
            train_exp = current_scenario.train_stream[0]  # Single experience per scenario
            strategy.train(train_exp)
            
            # Evaluate on all tasks seen so far
            task_accs = []
            for j in range(i + 1):
                eval_scenario = scenario[j]
                eval_exp = eval_scenario.test_stream[0]
                eval_results = strategy.eval(eval_exp)
                
                # Extract accuracy from Avalanche evaluation results
                acc = extract_accuracy_from_results(eval_results, model, eval_exp)
                
                task_accs.append(acc)
            
            results['task_accuracies'].append(task_accs.copy())
            results['average_accuracy'].append(np.mean(task_accs))
            
            # Compute forgetting (only for tasks seen before)
            if i > 0:
                forgetting = []
                for j in range(i):
                    # Forgetting = max accuracy on task j - current accuracy on task j
                    max_acc = max([results['task_accuracies'][k][j] for k in range(j, i+1)])
                    current_acc = task_accs[j]
                    forgetting.append(max_acc - current_acc)
                results['forgetting'].append(np.mean(forgetting) if forgetting else 0.0)
            else:
                results['forgetting'].append(0.0)
    else:
        # Single scenario case: use original logic
        for i, experience in enumerate(scenario.train_stream):
            print(f"Training on experience {i}")
            strategy.train(experience)
            
            # Evaluate on all previous experiences
            task_accs = []
            for j in range(i + 1):
                eval_exp = scenario.test_stream[j]
                eval_results = strategy.eval(eval_exp)
                
                # Extract accuracy from Avalanche evaluation results
                acc = extract_accuracy_from_results(eval_results, model, eval_exp)
                
                task_accs.append(acc)
            
            results['task_accuracies'].append(task_accs.copy())
            results['average_accuracy'].append(np.mean(task_accs))
            
            # Compute forgetting (only for tasks seen before)
            if i > 0:
                forgetting = []
                for j in range(i):
                    # Forgetting = max accuracy on task j - current accuracy on task j
                    max_acc = max([results['task_accuracies'][k][j] for k in range(j, i+1)])
                    current_acc = task_accs[j]
                    forgetting.append(max_acc - current_acc)
                results['forgetting'].append(np.mean(forgetting) if forgetting else 0.0)
            else:
                results['forgetting'].append(0.0)
    
    # Final accuracies on all tasks
    results['final_accuracies'] = results['task_accuracies'][-1]
    
    return results


def extract_accuracy_from_results(eval_results, model: ContinualGNN, experience) -> float:
    """
    Extract accuracy from Avalanche evaluation results.
    
    Args:
        eval_results: Results from strategy.eval()
        model: The continual GNN model (fallback)
        experience: The evaluation experience (fallback)
        
    Returns:
        Accuracy value
    """
    # Debug: print what we're getting from Avalanche
    # print(f"DEBUG: eval_results type: {type(eval_results)}")
    # print(f"DEBUG: eval_results: {eval_results}")
    
    # The key insight: Avalanche eval() returns None, but the metrics are logged elsewhere
    # We need to access the strategy's evaluation plugin to get the metrics
    
    # Actually, let's just use the fallback computation for now since it should work
    # and the Avalanche internal metrics are complex to extract
    return compute_accuracy(model, experience)


def compute_accuracy(model: ContinualGNN, experience) -> float:
    """
    Manually compute accuracy for an experience.
    
    Args:
        model: The continual GNN model
        experience: The evaluation experience
        
    Returns:
        Accuracy value
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(experience.dataset):
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                # Tensor format from our converted dataset
                # Format: [x, y] or [x, y, task_label]
                x = batch[0]  # Features tensor
                y = batch[1]  # Target label
                
                # Ensure x has the right dimensions for the model
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # Add batch dimension
                
                # Forward pass - for tensor data, we don't provide edge_index
                output = model(x)
                pred = output.argmax(dim=-1)  # Get predicted class
                
                # Handle the target value (could be int or tensor)
                if isinstance(y, (int, float)):
                    # Target is already an integer
                    y_target = int(y)
                    pred_class = pred.item() if hasattr(pred, 'item') else int(pred[0])
                    correct = 1 if pred_class == y_target else 0
                    total_samples += 1
                elif hasattr(y, 'dim') and y.dim() == 0:
                    # Single sample tensor
                    y_target = y.item() if hasattr(y, 'item') else int(y)
                    pred_class = pred.item() if hasattr(pred, 'item') else int(pred[0])
                    correct = 1 if pred_class == y_target else 0
                    total_samples += 1
                else:
                    # Multiple samples (shouldn't happen with our current setup, but handle it)
                    y_tensor = torch.tensor(y) if not hasattr(y, 'dim') else y
                    correct = (pred == y_tensor).sum().item()
                    total_samples += len(y_tensor)
                    
                total_correct += correct
                
            elif isinstance(batch, Data):
                # PyTorch Geometric format (original graph data)
                output = model(batch.x, batch.edge_index)
                pred = output.argmax(dim=-1)
                
                if batch.y.dim() == 0:
                    # Graph-level task
                    correct = (pred == batch.y).sum().item()
                    total_samples += 1
                else:
                    # Node-level task  
                    correct = (pred == batch.y).sum().item()
                    total_samples += len(batch.y)
                total_correct += correct
            else:
                # Unknown format - skip
                # print(f"DEBUG: Skipping unknown batch format: {type(batch)}")
                continue
    
    return total_correct / total_samples if total_samples > 0 else 0.0


def visualize_continual_learning_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Visualize continual learning results.
    
    Args:
        results: Results dictionary from evaluate_continual_model
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Task accuracies over time
    axes[0, 0].set_title('Task Accuracies Over Time')
    num_tasks = len(results['final_accuracies'])
    for task_id in range(num_tasks):
        task_accs = [results['task_accuracies'][exp][task_id] 
                    for exp in range(task_id, len(results['task_accuracies']))]
        axes[0, 0].plot(range(task_id, len(results['task_accuracies'])), 
                       task_accs, marker='o', label=f'Task {task_id}')
    axes[0, 0].set_xlabel('Experience')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Average accuracy over time
    axes[0, 1].set_title('Average Accuracy Over Time')
    axes[0, 1].plot(results['average_accuracy'], marker='o', color='red')
    axes[0, 1].set_xlabel('Experience')
    axes[0, 1].set_ylabel('Average Accuracy')
    axes[0, 1].grid(True)
    
    # Plot 3: Forgetting over time
    axes[1, 0].set_title('Forgetting Over Time')
    axes[1, 0].plot(results['forgetting'], marker='o', color='orange')
    axes[1, 0].set_xlabel('Experience')
    axes[1, 0].set_ylabel('Average Forgetting')
    axes[1, 0].grid(True)
    
    # Plot 4: Final accuracies
    axes[1, 1].set_title('Final Accuracies per Task')
    axes[1, 1].bar(range(len(results['final_accuracies'])), results['final_accuracies'])
    axes[1, 1].set_xlabel('Task')
    axes[1, 1].set_ylabel('Final Accuracy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_synthetic_graph_scenario(
    num_tasks: int = 3,
    graphs_per_task: int = 100,
    num_nodes_range: Tuple[int, int] = (10, 50),
    num_features: int = 32,
    num_classes: int = 2,
    edge_prob: float = 0.3,
    seed: int = 42
) -> CLScenario:
    """
    Create a synthetic graph dataset scenario for continual learning.
    
    Args:
        num_tasks: Number of tasks
        graphs_per_task: Number of graphs per task
        num_nodes_range: Range of number of nodes per graph
        num_features: Number of node features
        num_classes: Number of classes
        edge_prob: Edge probability for random graphs
        seed: Random seed
        
    Returns:
        CLScenario instance
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_datasets = []
    test_datasets = []
    
    for task_id in range(num_tasks):
        train_graphs = []
        test_graphs = []
        train_targets = []
        test_targets = []
        
        # Generate training graphs
        for _ in range(graphs_per_task):
            num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1] + 1)
            
            # Generate node features with task-specific variation
            task_feature_shift = task_id * 0.3  # Small shift per task
            x = torch.randn(num_nodes, num_features) + task_feature_shift
            
            # Generate edges with task-specific connectivity
            task_edge_prob = edge_prob + (task_id * 0.05)  # Slight increase for later tasks
            task_edge_prob = min(task_edge_prob, 0.7)  # Cap connectivity
            edge_index = torch.randperm(num_nodes * num_nodes)[:int(num_nodes * num_nodes * task_edge_prob)]
            edge_index = torch.stack([edge_index // num_nodes, edge_index % num_nodes])
            
            # Remove self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            # Generate label (keep same distribution across tasks)
            y = torch.randint(0, num_classes, (1,))[0]
            # For multi-task scenarios, we'll rely on different graph structures
            # rather than modifying labels to create task differences
            
            graph = Data(x=x, edge_index=edge_index, y=y)
            train_graphs.append(graph)
            train_targets.append(int(y.item()))
        
        # Generate test graphs (similar process)
        for _ in range(graphs_per_task // 4):  # Smaller test set
            num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1] + 1)
            # Generate test features with same task-specific variation
            task_feature_shift = task_id * 0.3  # Match training data
            x = torch.randn(num_nodes, num_features) + task_feature_shift
            
            # Generate test edges with same task-specific connectivity
            task_edge_prob = edge_prob + (task_id * 0.05)
            task_edge_prob = min(task_edge_prob, 0.7)
            edge_index = torch.randperm(num_nodes * num_nodes)[:int(num_nodes * num_nodes * task_edge_prob)]
            edge_index = torch.stack([edge_index // num_nodes, edge_index % num_nodes])
            
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            y = torch.randint(0, num_classes, (1,))[0]
            # Keep same label distribution across tasks
            
            graph = Data(x=x, edge_index=edge_index, y=y)
            test_graphs.append(graph)
            test_targets.append(int(y.item()))
        
        # Create Avalanche datasets  
        train_tensor_dataset = convert_graph_data_to_tensor_dataset(train_graphs, train_targets)
        test_tensor_dataset = convert_graph_data_to_tensor_dataset(test_graphs, test_targets)
        
        train_datasets.append(train_tensor_dataset)
        test_datasets.append(test_tensor_dataset)
    
    # For now, use a simpler approach: create individual single-task scenarios
    # and manually handle multi-task training in the training loop
    if num_tasks == 1:
        scenario = nc_benchmark(
            train_datasets,
            test_datasets,
            n_experiences=num_tasks,
            task_labels=True,
            shuffle=False
        )
    else:
        # For multi-task, create individual scenarios for each task
        # This avoids complex Avalanche class remapping issues
        scenario_list = []
        for i in range(num_tasks):
            single_scenario = nc_benchmark(
                [train_datasets[i]],
                [test_datasets[i]], 
                n_experiences=1,
                task_labels=True,
                shuffle=False
            )
            scenario_list.append(single_scenario)
        
        # Store the list of scenarios and handle multi-task training separately
        scenario = scenario_list
    
    return scenario