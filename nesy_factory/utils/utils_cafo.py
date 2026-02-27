"""
Utilities for GNN Model Factory.
Contains functions for YAML configuration handling and other helper utilities.
Enhanced with Cascaded Forward (CaFo) training capabilities.

⚠️  DEPRECATED: CaFO functionality in this file is DEPRECATED.
Use enhanced_cafo_blocks.py for modern CaFO implementation with:
- Better performance and cleaner API
- Support for STGNN and specialized models  
- True layer-wise CaFO training
- Continual learning integration

This file is kept for backward compatibility only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import numpy as np
import random
from typing import Dict, Any, List, Union, Tuple, Optional
from torch_geometric.data import Data
from collections import defaultdict
import time

# ============================================================================
# YAML CONFIGURATION UTILITIES
# ============================================================================

def load_yaml_config(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_config_by_name(config_name: str, config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Get a specific configuration by name from YAML file."""
    all_configs = load_yaml_config(config_path)
    
    if config_name not in all_configs:
        available_configs = [k for k in all_configs.keys() if not k.startswith(('training', 'experiment', 'cafo'))]
        raise ValueError(f"Configuration '{config_name}' not found. Available configs: {available_configs}")
    
    return all_configs[config_name].copy()

def update_config_for_data(config: Dict[str, Any], data: Data) -> Dict[str, Any]:
    """Update configuration with actual data dimensions."""
    config = config.copy()  # Don't modify original
    config['input_dim'] = data.x.size(1)
    return config

def get_training_config(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Get training configuration from YAML file."""
    all_configs = load_yaml_config(config_path)
    return all_configs.get('training', {})

def get_cafo_config(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Get CaFo training configuration from YAML file."""
    all_configs = load_yaml_config(config_path)
    return all_configs.get('cafo', {})

def get_experiment_config(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Get experiment configuration from YAML file."""
    all_configs = load_yaml_config(config_path)
    return all_configs.get('experiment', {})

def list_available_configs(config_path: str = 'configs/gcn_configs.yaml') -> None:
    """List all available model configurations."""
    all_configs = load_yaml_config(config_path)
    model_configs = {k: v for k, v in all_configs.items() 
                    if not k.startswith(('training', 'experiment', 'cafo'))}
    
    print("Available GCN Configurations:")
    print("=" * 50)
    for name, config in model_configs.items():
        hidden_str = str(config['hidden_dim'])
        layers = config.get('num_layers', 'N/A')
        optimizer = config.get('optimizer', 'N/A')
        print(f"{name:20}: Hidden={hidden_str:25} Layers={layers} Optimizer={optimizer}")
    print()

def get_model_config_names(config_path: str = 'configs/gcn_configs.yaml') -> List[str]:
    """Get list of available model configuration names."""
    all_configs = load_yaml_config(config_path)
    return [k for k in all_configs.keys() if not k.startswith(('training', 'experiment', 'cafo'))]

def setup_experiment_environment(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Setup experiment environment based on YAML configuration."""
    experiment_config = get_experiment_config(config_path)
    
    # Set random seed
    if experiment_config.get('random_seed'):
        torch.manual_seed(experiment_config['random_seed'])
        print(f"Random seed set to: {experiment_config['random_seed']}")
    
    # Set device
    device = experiment_config.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    experiment_config['device'] = device
    if experiment_config.get('verbose', True):
        print(f"Device set to: {device}")
    
    return experiment_config

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_random_seed(seed: int, verbose: bool = True):
    """Set random seeds for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # The following two lines are often recommended for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"Random seed set to {seed}")

# ============================================================================
# DATA UTILITIES
# ============================================================================

def create_sample_graph(num_nodes: int = 15, num_features: int = 10, num_classes: int = 4) -> tuple:
    """Create a sample graph for testing with configurable parameters."""
    x = torch.randn(num_nodes, num_features)
    
    # Create a more connected graph
    edge_list = []
    # Ring connections + additional connections for better connectivity
    for i in range(num_nodes):
        # Ring
        edge_list.extend([[i, (i + 1) % num_nodes], [(i + 1) % num_nodes, i]])
        # Additional connections
        for j in range(i + 1, min(i + 4, num_nodes)):
            edge_list.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create masks for train/val/test
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool) 
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True  
    test_mask[int(0.8 * num_nodes):] = True
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, train_mask, val_mask, test_mask

def print_model_summary(model, config: Dict[str, Any], data: Data) -> None:
    """Print a comprehensive summary of the model."""
    print(f"Model: {model.__class__.__name__}")
    
    if isinstance(config['hidden_dim'], list):
        arch_str = f"{config['input_dim']} -> {' -> '.join(map(str, config['hidden_dim']))} -> {config['output_dim']}"
    else:
        layers = config.get('num_layers', 2)
        hidden_layers = [config['hidden_dim']] * (layers - 1) if layers > 1 else []
        arch_str = f"{config['input_dim']} -> {' -> '.join(map(str, hidden_layers))} -> {config['output_dim']}"
    
    print(f"Architecture: {arch_str}")
    print(f"Parameters: {model.get_num_parameters()}")
    print(f"Optimizer: {model.get_optimizer_info()}")
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

# ============================================================================
# CASCADED FORWARD (CAFO) TRAINING UTILITIES
# ============================================================================

class CaFoBlock(nn.Module):
    """A single block in the CaFo training, representing one layer of graph convolution."""

    def __init__(self, embedding_dim: int, n_nodes: int, block_type: str = 'linear'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_nodes = n_nodes
        self.block_type = block_type

        # Local classifier for this block
        if block_type == 'linear':
            self.classifier = nn.Linear(embedding_dim, embedding_dim)
        elif block_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # Initialize classifier
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def graph_convolution(self, embeddings: torch.Tensor, adj_matrix: torch.sparse.FloatTensor) -> torch.Tensor:
        """Perform one layer of graph convolution"""
        return torch.sparse.mm(adj_matrix, embeddings)

    def predict_scores(self, embeddings: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        """Predict scores using local classifier"""
        node_features = self.classifier(embeddings[node_ids])
        return node_features

    def train_block(self, embeddings: torch.Tensor, adj_matrix: torch.sparse.FloatTensor, 
                   data: Data, train_mask: torch.Tensor, 
                   epochs: int = 50, lr: float = 0.001, weight_decay: float = 1e-4) -> torch.Tensor:
        """Train this block's classifier while keeping embeddings fixed"""

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Perform graph convolution for this layer
        current_embeddings = self.graph_convolution(embeddings, adj_matrix)
        current_embeddings = current_embeddings.detach()  # Prevent gradient flow

        best_loss = float('inf')
        best_embeddings = current_embeddings.clone()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Get predictions for training nodes
            train_nodes = torch.where(train_mask)[0]
            if len(train_nodes) == 0:
                continue

            node_features = self.predict_scores(current_embeddings, train_nodes)
            
            # For node classification, we need to add a final prediction layer
            # This is a simplified version - you might need to adapt based on your specific task
            if not hasattr(self, 'output_layer'):
                self.output_layer = nn.Linear(self.embedding_dim, data.y.max().item() + 1).to(embeddings.device)
            
            predictions = self.output_layer(node_features)
            targets = data.y[train_nodes]

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_embeddings = current_embeddings.clone()

            if epoch % 10 == 0:
                print(f"    Block epoch {epoch:2d}/{epochs} - Loss: {loss.item():.6f}")

        return best_embeddings


def create_adjacency_matrix(data: Data, normalize: bool = True) -> torch.sparse.FloatTensor:
    """Create normalized adjacency matrix from PyTorch Geometric data."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Create adjacency matrix
    adj_indices = edge_index
    adj_values = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
    
    adj_matrix = torch.sparse_coo_tensor(
        adj_indices, adj_values, (num_nodes, num_nodes), dtype=torch.float
    ).coalesce()

    if normalize:
        # Add self-loops
        eye_indices = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(num_nodes, dtype=torch.float, device=edge_index.device)
        eye = torch.sparse_coo_tensor(eye_indices, eye_values, (num_nodes, num_nodes))
        adj_matrix = adj_matrix + eye

        # Coalesce before normalization
        adj_matrix = adj_matrix.coalesce()
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum + 1e-6, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

        indices = adj_matrix.indices()
        values = adj_matrix.values()
        normalized_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]

        adj_matrix = torch.sparse_coo_tensor(
            indices, normalized_values, adj_matrix.size(), dtype=torch.float
        ).coalesce()

    return adj_matrix


def train_model_with_cafo(model, data: Data, train_mask: torch.Tensor, val_mask: torch.Tensor, 
                         test_mask: torch.Tensor, cafo_config: Dict[str, Any], 
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Train a GNN model using Cascaded Forward (CaFo) algorithm.
    
    Args:
        model: The GNN model to train
        data: PyTorch Geometric data object
        train_mask: Training node mask
        val_mask: Validation node mask
        test_mask: Test node mask
        cafo_config: CaFo configuration parameters
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing training results and final embeddings
    """
    
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Extract CaFo parameters
    epochs_per_block = cafo_config.get('epochs_per_block', 50)
    block_lr = cafo_config.get('block_lr', 0.001)
    block_weight_decay = cafo_config.get('block_weight_decay', 1e-4)
    block_type = cafo_config.get('block_type', 'linear')
    n_blocks = cafo_config.get('n_blocks', 3)
    
    if verbose:
        print(f"Starting CaFo training with {n_blocks} blocks...")
        print(f"Epochs per block: {epochs_per_block}")
        print(f"Block learning rate: {block_lr}")
    
    # Create normalized adjacency matrix
    adj_matrix = create_adjacency_matrix(data, normalize=True)
    
    # Get initial embeddings from model
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_embeddings'):
            current_embeddings = model.get_embeddings(data.x, data.edge_index)
        else:
            # Fallback: use the input features
            current_embeddings = data.x
    
    # Store all layer embeddings
    all_embeddings = [current_embeddings.clone()]
    
    # Create and train blocks
    blocks = []
    results = {
        'block_losses': [],
        'val_accuracies': [],
        'training_time': 0
    }
    
    start_time = time.time()
    
    for i in range(n_blocks):
        if verbose:
            print(f"\n--- Training Block {i+1}/{n_blocks} ---")
        
        # Create new block
        block = CaFoBlock(
            embedding_dim=current_embeddings.size(1),
            n_nodes=data.num_nodes,
            block_type=block_type
        ).to(device)
        
        # Train block and get updated embeddings
        current_embeddings = block.train_block(
            current_embeddings, adj_matrix, data, train_mask,
            epochs=epochs_per_block, lr=block_lr, weight_decay=block_weight_decay
        )
        
        blocks.append(block)
        all_embeddings.append(current_embeddings.clone())
        
        # Evaluate current performance
        if hasattr(block, 'output_layer'):
            block.eval()
            with torch.no_grad():
                val_nodes = torch.where(val_mask)[0]
                if len(val_nodes) > 0:
                    val_features = block.predict_scores(current_embeddings, val_nodes)
                    val_preds = block.output_layer(val_features)
                    val_acc = (val_preds.argmax(dim=1) == data.y[val_nodes]).float().mean().item()
                    results['val_accuracies'].append(val_acc)
                    
                    if verbose:
                        print(f"Block {i+1} validation accuracy: {val_acc:.4f}")
    
    end_time = time.time()
    results['training_time'] = end_time - start_time
    
    # Combine all layer embeddings (mean aggregation)
    final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
    
    # Final evaluation
    if blocks and hasattr(blocks[-1], 'output_layer'):
        blocks[-1].eval()
        with torch.no_grad():
            test_nodes = torch.where(test_mask)[0]
            if len(test_nodes) > 0:
                test_features = blocks[-1].predict_scores(final_embeddings, test_nodes)
                test_preds = blocks[-1].output_layer(test_features)
                test_acc = (test_preds.argmax(dim=1) == data.y[test_nodes]).float().mean().item()
                results['test_accuracy'] = test_acc
                
                if verbose:
                    print(f"\nFinal test accuracy: {test_acc:.4f}")
                    print(f"CaFo training completed in {results['training_time']:.2f} seconds")
    
    results['final_embeddings'] = final_embeddings
    results['all_embeddings'] = all_embeddings
    results['blocks'] = blocks
    
    return results


# ============================================================================
# TRAINING UTILITIES (ENHANCED)
# ============================================================================

def train_model_with_config(model, data, train_mask, val_mask, test_mask, 
                          training_config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Train a model using training configuration from YAML."""
    training_method = training_config.get('method', 'standard')
    
    if training_method == 'cafo':
        # Use CaFo training
        cafo_config = training_config.get('cafo_params', {})
        return train_model_with_cafo(model, data, train_mask, val_mask, test_mask, cafo_config, verbose)
    else:
        # Standard training
        epochs = training_config.get('epochs', 10)
        val_freq = training_config.get('validation_frequency', 5)
        
        results = {
            'train_losses': [],
            'val_accuracies': [],
            'test_accuracy': None
        }
        
        if verbose:
            print(f"Training for {epochs} epochs with validation every {val_freq} epochs")
        
        for epoch in range(epochs):
            loss = model.train_step(data, train_mask)
            results['train_losses'].append(loss)
            
            if epoch % val_freq == 0 or epoch == epochs - 1:
                val_results = model.eval_step(data, val_mask)
                results['val_accuracies'].append(val_results['accuracy'])
                
                if verbose:
                    print(f"Epoch {epoch:3d}: Loss {loss:.4f}, Val Acc {val_results['accuracy']:.4f}")
        
        # Final test evaluation
        test_results = model.eval_step(data, test_mask)
        results['test_accuracy'] = test_results['accuracy']
        
        if verbose:
            print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
        
        return results


def compare_training_methods(model_configs: List[str], data, train_mask, val_mask, test_mask,
                           standard_epochs: int = 10, cafo_blocks: int = 3) -> Dict[str, Dict[str, Any]]:
    """Compare standard training vs CaFo training for multiple model configurations."""
    from GNNs import GCN  # Assuming this is your GNN implementation
    
    results = {}
    
    for config_name in model_configs:
        print(f"\n{'='*60}")
        print(f"Training model: {config_name}")
        print(f"{'='*60}")
        
        # Get model configuration
        config = get_config_by_name(config_name)
        config = update_config_for_data(config, data)
        
        # Standard training
        print(f"\n--- Standard Training ---")
        model_std = GCN(config)
        start_time = time.time()
        std_results = train_model_with_config(
            model_std, data, train_mask, val_mask, test_mask,
            {'method': 'standard', 'epochs': standard_epochs, 'validation_frequency': 5}
        )
        std_time = time.time() - start_time
        std_results['training_time'] = std_time
        
        # CaFo training
        print(f"\n--- CaFo Training ---")
        model_cafo = GCN(config)
        cafo_config = {
            'epochs_per_block': standard_epochs // cafo_blocks,
            'n_blocks': cafo_blocks,
            'block_lr': 0.001,
            'block_weight_decay': 1e-4,
            'block_type': 'linear'
        }
        cafo_results = train_model_with_cafo(
            model_cafo, data, train_mask, val_mask, test_mask, cafo_config
        )
        
        results[config_name] = {
            'standard': std_results,
            'cafo': cafo_results
        }
        
        # Summary
        print(f"\n--- {config_name} Results Summary ---")
        print(f"Standard - Test Acc: {std_results['test_accuracy']:.4f}, Time: {std_results['training_time']:.2f}s")
        print(f"CaFo     - Test Acc: {cafo_results['test_accuracy']:.4f}, Time: {cafo_results['training_time']:.2f}s")
    
    return results


def compare_models_performance(models: Dict[str, Any], data, train_mask, val_mask, test_mask,
                             epochs: int = 10) -> Dict[str, Dict[str, float]]:
    """Compare performance of multiple models."""
    results = {}
    
    print("Training comparison:")
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        for name, model in models.items():
            loss = model.train_step(data, train_mask)
            val_results = model.eval_step(data, val_mask)
            
            if name not in results:
                results[name] = {'losses': [], 'val_accuracies': []}
            
            results[name]['losses'].append(loss)
            results[name]['val_accuracies'].append(val_results['accuracy'])
            
            print(f"  {name:15}: Loss {loss:.4f}, Val Acc {val_results['accuracy']:.4f}")
    
    # Final test evaluation for all models
    print("\nFinal Test Results:")
    for name, model in models.items():
        test_results = model.eval_step(data, test_mask)
        results[name]['test_accuracy'] = test_results['accuracy']
        print(f"  {name:15}: Test Acc {test_results['accuracy']:.4f}")
    
    return results

# ============================================================================
# EXAMPLE UTILITIES
# ============================================================================

def create_example_models_from_configs(config_names: List[str], data: Data) -> Dict[str, Any]:
    """Create multiple models from YAML configurations."""
    from GNNs import GCN
    
    models = {}
    
    for name in config_names:
        config = get_config_by_name(name)
        config = update_config_for_data(config, data)
        models[name] = GCN(config)
    
    return models

def print_config_comparison(config_names: List[str], data: Data) -> None:
    """Print comparison table of different configurations."""
    print("Configuration Comparison:")
    print("=" * 80)
    print(f"{'Config Name':<20} {'Architecture':<35} {'Params':<10} {'Optimizer':<10}")
    print("-" * 80)
    
    for name in config_names:
        config = get_config_by_name(name)
        config = update_config_for_data(config, data)
        
        from GNNs import GCN
        model = GCN(config)
        
        if isinstance(config['hidden_dim'], list):
            arch = f"{config['input_dim']} -> {' -> '.join(map(str, config['hidden_dim']))} -> {config['output_dim']}"
        else:
            arch = f"{config['input_dim']} -> {config['hidden_dim']} -> {config['output_dim']}"
        
        arch = arch[:33] + "..." if len(arch) > 35 else arch
        
        print(f"{name:<20} {arch:<35} {model.get_num_parameters():<10} {config['optimizer']:<10}")
    
    print()

# ============================================================================
# CAFO-SPECIFIC EXAMPLE CONFIGURATIONS
# ============================================================================

def get_default_cafo_config() -> Dict[str, Any]:
    """Get default CaFo configuration parameters."""
    return {
        'epochs_per_block': 50,
        'n_blocks': 3,
        'block_lr': 0.001,
        'block_weight_decay': 1e-4,
        'block_type': 'linear'  # or 'mlp'
    }

def run_cafo_experiment(config_names: List[str], data: Data, train_mask: torch.Tensor,
                       val_mask: torch.Tensor, test_mask: torch.Tensor) -> None:
    """Run a complete CaFo experiment with multiple configurations."""
    print("=" * 70)
    print("CaFo Training Experiment")
    print("=" * 70)
    
    # Load experiment configuration
    experiment_config = setup_experiment_environment()
    cafo_config = get_default_cafo_config()
    
    print(f"Device: {experiment_config['device']}")
    print(f"CaFo Config: {cafo_config}")
    print()
    
    for config_name in config_names:
        print(f"Training {config_name} with CaFo...")
        
        # Create model
        config = get_config_by_name(config_name)
        config = update_config_for_data(config, data)
        
        from GNNs import GCN
        model = GCN(config)
        
        # Train with CaFo
        results = train_model_with_cafo(
            model, data, train_mask, val_mask, test_mask, cafo_config, verbose=True
        )
        
        print(f"{config_name} - Final accuracy: {results['test_accuracy']:.4f}")
        print("-" * 50)
