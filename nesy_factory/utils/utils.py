"""
Utilities for GNN Model Factory.
Contains functions for YAML configuration handling and other helper utilities.
"""

import torch
import yaml
import os
import numpy as np
import random
from typing import Dict, Any, List, Union
from torch_geometric.data import Data

from sklearn.decomposition import PCA
from tqdm import tqdm


## Data loading classes STGNN

class DataWrapper:
    def __init__(self, data_dict):
        self.__dict__.update(data_dict)

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

# data loading classes PPI
class DataContainer:
    """A container to hold datasets and mimic the single-graph data object structure."""
    def __init__(self, train_dataset, test_dataset):
        self.dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_mask = None
        self.test_mask = None

# data loading classes 


# ============================================================================
# Anomaly Detection Utilities (for STGNN)
# ============================================================================

class AnomalyDetector:
    def __init__(self, train_obs, val_obs, test_obs,
                 train_forecast, val_forecast, test_forecast,
                 window_length=None, batch_size=512, root_cause=False):
        self.train_obs = train_obs
        self.val_obs = val_obs
        self.test_obs = test_obs
        self.train_forecast = train_forecast
        self.val_forecast = val_forecast
        self.test_forecast = test_forecast
        self.root_cause = root_cause
        if window_length is None:
            self.window_length = len(train_obs) + len(val_obs)
        else:
            self.window_length = window_length
        self.batch_size = batch_size

        if self.root_cause:
            self.val_re_full = None
            self.test_re_full = None

    def pca_model(self, val_error, test_error, dim_size=1):
        pca = PCA(n_components=dim_size, svd_solver='full')
        pca.fit(val_error)

        transf_val_error = pca.inverse_transform(pca.transform(val_error))
        transf_test_error = pca.inverse_transform(pca.transform(test_error))
        val_re_full = np.absolute(transf_val_error - val_error)
        val_re = val_re_full.sum(axis=1)
        test_re_full = np.absolute(transf_test_error - test_error)
        test_re = test_re_full.sum(axis=1)

        return val_re, test_re, val_re_full, test_re_full

    def scorer(self, num_components):
        val_abs = np.absolute(self.val_obs - self.val_forecast)
        full_obs = np.concatenate((self.train_obs, self.val_obs, self.test_obs), axis=0)
        full_forecast = np.concatenate((self.train_forecast, self.val_forecast, self.test_forecast), axis=0)
        full_abs = np.absolute(full_obs - full_forecast)
        
        val_norm = error_normalizer(val_abs)
        test_norm = error_sw_normalizer(full_abs, self.window_length, self.batch_size, len(self.test_obs))

        val_re, test_re, val_re_full, test_re_full = self.pca_model(val_norm, test_norm, num_components)

        if self.root_cause:
            self.val_re_full = val_re_full
            self.test_re_full = test_re_full

        realtime_indicator = test_re
        anomaly_prediction = test_re > val_re.max()

        return realtime_indicator, anomaly_prediction

def error_normalizer(error_mat):
    median = np.median(error_mat, axis=0)
    q1 = np.quantile(error_mat, q=0.25, axis=0)
    q3 = np.quantile(error_mat, q=0.75, axis=0)
    iqr = q3 - q1 + 1e-2
    norm_error = (error_mat - median) / iqr
    return norm_error

def sliding_window(error_mat, window_size):
    return error_mat[np.arange(window_size)[None, :] + np.arange(error_mat.shape[0] - window_size)[:, None]]

def error_sw_normalizer(error_mat, window_size, batch_size, test_size):
    data_size = error_mat.shape[0]
    num_batch = int(test_size / batch_size) + 1
    norm_error_mat = []

    for i in tqdm(range(num_batch), desc="Normalizing errors"):
        start_idx = i * batch_size + (data_size - test_size)
        end_idx = (i + 1) * batch_size + (data_size - test_size)
        batch_error_mat = error_mat[start_idx:end_idx, :]
        sw_error_mat = sliding_window(error_mat[(start_idx - window_size):end_idx, :], window_size)

        median = np.median(sw_error_mat, axis=1)
        q1 = np.quantile(sw_error_mat, q=0.25, axis=1)
        q3 = np.quantile(sw_error_mat, q=0.75, axis=1)
        iqr = q3 - q1 + 1e-2
        batch_norm_error = (batch_error_mat - median) / iqr
        norm_error_mat.append(batch_norm_error)
        
    norm_error_mat = np.concatenate(norm_error_mat)
    return norm_error_mat

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
        available_configs = [k for k in all_configs.keys() if not k.startswith(('training', 'experiment'))]
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

def get_experiment_config(config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Get experiment configuration from YAML file."""
    all_configs = load_yaml_config(config_path)
    return all_configs.get('experiment', {})

def list_available_configs(config_path: str = 'configs/gcn_configs.yaml') -> None:
    """List all available model configurations."""
    all_configs = load_yaml_config(config_path)
    model_configs = {k: v for k, v in all_configs.items() 
                    if not k.startswith(('training', 'experiment'))}
    
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
    return [k for k in all_configs.keys() if not k.startswith(('training', 'experiment'))]

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

def create_sample_graph(num_nodes: int = 15, num_features: int = 10, num_classes: int = 3) -> tuple:
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

def create_bipartite_graph(num_users: int = 50, num_items: int = 30, num_features: int = 10) -> tuple:
    """Create a sample bipartite graph for recommendation systems."""
    # Create user and item features
    user_features = torch.randn(num_users, num_features)
    item_features = torch.randn(num_items, num_features)
    
    # Combine all features
    x = torch.cat([user_features, item_features], dim=0)
    total_nodes = num_users + num_items
    
    # Create bipartite edges (users to items only)
    edge_list = []
    num_edges = min(200, num_users * num_items // 4)  # Reasonable number of edges
    
    for _ in range(num_edges):
        user_idx = np.random.randint(0, num_users)
        item_idx = np.random.randint(num_users, total_nodes)
        edge_list.extend([[user_idx, item_idx], [item_idx, user_idx]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    
    # Create binary labels for user-item interactions (for link prediction)
    y = torch.randint(0, 2, (total_nodes,))
    
    # Create masks
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool) 
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)
    
    train_mask[:int(0.6 * total_nodes)] = True
    val_mask[int(0.6 * total_nodes):int(0.8 * total_nodes)] = True  
    test_mask[int(0.8 * total_nodes):] = True
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, train_mask, val_mask, test_mask

def create_recommendation_graph(num_users: int = 100, num_items: int = 50, num_features: int = 16, 
                              sparsity: float = 0.1) -> tuple:
    """Create a sample recommendation graph for PinSAGE."""
    # Create user and item features
    user_features = torch.randn(num_users, num_features)
    item_features = torch.randn(num_items, num_features)
    
    # Combine features
    x = torch.cat([user_features, item_features], dim=0)
    total_nodes = num_users + num_items
    
    # Create sparse user-item interactions
    edge_list = []
    num_interactions = int(num_users * num_items * sparsity)
    
    for _ in range(num_interactions):
        user_idx = np.random.randint(0, num_users)
        item_idx = np.random.randint(num_users, total_nodes)
        edge_list.extend([[user_idx, item_idx], [item_idx, user_idx]])
    
    # Add some item-item similarity edges
    num_item_edges = min(100, num_items * 2)
    for _ in range(num_item_edges):
        item1 = np.random.randint(num_users, total_nodes)
        item2 = np.random.randint(num_users, total_nodes)
        if item1 != item2:
            edge_list.extend([[item1, item2], [item2, item1]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    
    # Create ratings/labels (for rating prediction)
    y = torch.randint(1, 6, (total_nodes,))  # Ratings from 1-5
    
    # Create masks
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool) 
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)
    
    train_mask[:int(0.6 * total_nodes)] = True
    val_mask[int(0.6 * total_nodes):int(0.8 * total_nodes)] = True  
    test_mask[int(0.8 * total_nodes):] = True
    
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
# TRAINING UTILITIES
# ============================================================================

def train_model_with_config(model, data, train_mask, val_mask, test_mask, 
                          training_config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Train a model using training configuration from YAML."""
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

def create_example_models_from_configs(config_names: List[str], data: Data, config_path: str = 'configs/gcn_configs.yaml') -> Dict[str, Any]:
    """Create multiple models from YAML configurations."""
    from nesy_factory.GNNs import GCN
    
    models = {}
    
    for name in config_names:
        config = get_config_by_name(name, config_path)
        config = update_config_for_data(config, data)
        models[name] = GCN(config)
    
    return models

def print_config_comparison(config_names: List[str], data: Data, config_path: str = 'configs/gcn_configs.yaml') -> None:
    """Print comparison table of different configurations."""
    print("Configuration Comparison:")
    print("=" * 80)
    print(f"{'Config Name':<20} {'Architecture':<35} {'Params':<10} {'Optimizer':<10}")
    print("-" * 80)
    
    for name in config_names:
        config = get_config_by_name(name, config_path)
        config = update_config_for_data(config, data)
        
        from nesy_factory.GNNs import GCN
        model = GCN(config)
        
        if isinstance(config['hidden_dim'], list):
            arch = f"{config['input_dim']} -> {' -> '.join(map(str, config['hidden_dim']))} -> {config['output_dim']}"
        else:
            arch = f"{config['input_dim']} -> {config['hidden_dim']} -> {config['output_dim']}"
        
        arch = arch[:33] + "..." if len(arch) > 35 else arch
        
        print(f"{name:<20} {arch:<35} {model.get_num_parameters():<10} {config['optimizer']:<10}")
    
    print() 