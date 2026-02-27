#!/usr/bin/env python3
"""
Enhanced CaFO Blocks for GNNs

This module implements layer-by-layer CaFO (Constrained and Frozen Optimization) training 
for Graph Neural Networks, adapted for both standard GNNs and spatio-temporal models like STGNN.

Key Features:
- True CaFO algorithm with closed-form solutions
- Support for standard GNNs (GCN, GAT, etc.) 
- Specialized STGNN CaFO implementation for forecasting
- Layer-wise progressive training
- Continual learning compatibility

Classes:
- GNNCaFoBlock: CaFO block for standard GNNs
- LayerWiseCaFoGNN: Layer-wise CaFO training for GNNs
- STGNNCaFoBlock: Specialized CaFO block for STGNN
- LayerWiseCaFoSTGNN: Layer-wise CaFO training for STGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple


# Utility functions for CaFO training
def one_hot(y, num_labels):
    """Convert integer labels to one-hot vectors for classification tasks."""
    h = torch.zeros(size=(y.shape[0], num_labels), dtype=torch.float32, device=y.device)
    h[range(y.shape[0]), y] = 1.0
    return h

def loss_MSE(outputs, labels):
    """MSE loss for CaFO gradient descent training."""
    diff = outputs - labels
    loss = torch.sqrt(diff * diff).sum(1).mean()
    return loss

def jaccobian_MSE(outputs, labels):
    """MSE gradient for CaFO gradient descent training."""
    jacc = 2 * (outputs - labels)
    return jacc

def loss_cross_entropy(outputs, labels):
    """Cross-entropy loss for CaFO gradient descent training."""
    loss = -(labels * torch.log(F.softmax(outputs, dim=1) + 1e-8)).sum(1).mean()
    return loss

def jaccobian_cross_entropy(outputs, labels):
    """Cross-entropy gradient for CaFO gradient descent training."""
    s = F.softmax(outputs, dim=1)
    jacc = s - labels
    return jacc


class GNNCaFoBlock(nn.Module):
    """
    A CaFO block for GNNs implementing layer-by-layer CaFO training.
    
    Uses the actual CaFO algorithm with closed-form solutions for MSE
    and gradient descent for cross-entropy loss.
    """
    
    def __init__(self, gnn_layer, in_channels: int, out_channels: int, 
                 num_classes: int, cafo_config: Dict[str, Any]):
        """
        Initialize a GNN CaFO block.
        
        Args:
            gnn_layer: The GNN layer (GCNConv, GATConv, etc.)
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_classes: Number of classes for classification
            cafo_config: CaFO-specific configuration
        """
        super().__init__()
        
        self.gnn_layer = gnn_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        # CaFO configuration
        self.cafo_loss_fn = cafo_config.get('loss_fn', 'MSE')
        self.cafo_lamda = cafo_config.get('lambda', 0.001)
        self.cafo_num_epochs = cafo_config.get('num_epochs', 100)
        self.cafo_step = cafo_config.get('step', 0.01)
        self.cafo_num_batches = cafo_config.get('num_batches', 1)
        
        # CaFO weights (will be learned during training)
        self.cafo_weights = None
        
        # Local classifier for this block
        self.local_classifier = nn.Linear(out_channels, num_classes, bias=False)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the GNN layer.
        
        Args:
            x: Input node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output features [num_nodes, out_channels]
        """
        features = self.gnn_layer(x, edge_index, **kwargs)
        
        # Apply activation if not already applied
        if not hasattr(self.gnn_layer, 'activation') or self.gnn_layer.activation is None:
            features = F.relu(features)
            
        return features
    
    def cafo_predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using CaFO weights.
        
        Args:
            features: Input features [num_nodes, out_channels]
            
        Returns:
            Predictions [num_nodes, num_classes]
        """
        if self.cafo_weights is None:
            raise ValueError("CaFO weights not trained yet. Call train_cafo_block first.")
            
        # Add bias term
        if features.dim() == 2:
            bias_term = torch.ones(features.shape[0], 1, device=features.device)
            features_with_bias = torch.cat([features, bias_term], dim=1)
        else:
            features_with_bias = features
            
        return torch.matmul(features_with_bias, self.cafo_weights)
    
    def train_cafo_closeform_mse(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Train CaFO weights using closed-form MSE solution.
        
        Args:
            features: Input features [num_nodes, out_channels]
            labels: Target labels [num_nodes] (integer class indices)
            
        Returns:
            Trained CaFO weights
        """
        device = features.device
        inputs = features.to(device)
        
        # Convert to one-hot
        labels_onehot = one_hot(labels, num_labels=self.num_classes).to(device)
        
        # Add bias term
        if inputs.dim() == 2:
            bias_term = torch.ones(inputs.shape[0], 1, device=device)
            inputs = torch.cat([inputs, bias_term], dim=1)
        
        # Ridge regression solution
        n_samples, n_features = inputs.shape
        
        # Compute matrices
        A = torch.matmul(inputs.t(), inputs)
        B = torch.matmul(inputs.t(), labels_onehot)
        
        # Add regularization
        reg_strength = max(self.cafo_lamda, 1e-8)
        A += reg_strength * torch.eye(A.shape[0], device=device)
        
        # Solve linear system
        try:
            self.cafo_weights = torch.linalg.solve(A, B)
        except Exception as e:
            print(f"CaFO Block: Linear solve failed ({e}), using pseudo-inverse")
            self.cafo_weights = torch.matmul(torch.pinverse(A), B)
        
        return self.cafo_weights
    
    def train_cafo_gradient_descent(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Train CaFO weights using gradient descent.
        
        Args:
            features: Input features [num_nodes, out_channels]
            labels: Target labels [num_nodes] (integer class indices)
            
        Returns:
            Trained CaFO weights
        """
        device = features.device
        inputs = features.to(device)
        
        # Convert to one-hot
        labels_onehot = one_hot(labels, num_labels=self.num_classes).to(device)
        
        # Add bias term
        if inputs.dim() == 2:
            bias_term = torch.ones(inputs.shape[0], 1, device=device)
            inputs = torch.cat([inputs, bias_term], dim=1)
        
        # Initialize weights
        if self.cafo_weights is None:
            self.cafo_weights = torch.randn(inputs.shape[1], self.num_classes, 
                                          dtype=torch.float32, device=device) * 0.01
        
        # Gradient descent training
        num_samples = inputs.shape[0]
        step = self.cafo_step
        
        for epoch in range(self.cafo_num_epochs):
            batch_size = max(1, int(num_samples / self.cafo_num_batches))
            
            for k in range(self.cafo_num_batches):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, num_samples)
                
                input_batch = inputs[start_idx:end_idx]
                label_batch = labels_onehot[start_idx:end_idx]
                
                # Forward pass
                outputs_batch = torch.matmul(input_batch, self.cafo_weights)
                
                # Compute gradient
                if self.cafo_loss_fn == 'MSE':
                    jacc = jaccobian_MSE(outputs_batch, label_batch)
                elif self.cafo_loss_fn == 'CE':
                    jacc = jaccobian_cross_entropy(outputs_batch, label_batch)
                else:
                    jacc = torch.zeros_like(outputs_batch)
                
                # Compute weight update
                jacc = jacc.unsqueeze(1)
                xx = input_batch.unsqueeze(-1)
                delta_W = torch.matmul(xx, jacc).mean(0)
                
                # Add regularization
                if self.cafo_lamda > 1e-4:
                    delta_W += self.cafo_lamda * self.cafo_weights
                
                # Update weights
                self.cafo_weights -= step * delta_W
            
            # Print progress
            if epoch % 20 == 0:
                outputs = torch.matmul(inputs, self.cafo_weights)
                if self.cafo_loss_fn == 'MSE':
                    loss = loss_MSE(outputs, labels_onehot)
                elif self.cafo_loss_fn == 'CE':
                    loss = loss_cross_entropy(outputs, labels_onehot)
                else:
                    loss = torch.tensor(0.0)
                print(f"CaFO Block Epoch {epoch}: Loss = {loss.item():.6f}")
        
        return self.cafo_weights
    
    def train_cafo_block(self, features: torch.Tensor, edge_index: torch.Tensor, 
                        labels: torch.Tensor, train_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Train the entire CaFO block: GNN layer + CaFO classifier.
        
        Args:
            features: Input features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            labels: Target labels [num_nodes]
            train_mask: Optional training mask [num_nodes]
            
        Returns:
            Output features from this block [num_nodes, out_channels]
        """
        # Step 1: Forward pass through GNN layer (frozen after first training)
        gnn_features = self.forward(features, edge_index)
        
        # Step 2: Select training data
        if train_mask is not None:
            train_features = gnn_features[train_mask]
            train_labels = labels[train_mask]
        else:
            train_features = gnn_features
            train_labels = labels
        
        # Step 3: Train CaFO classifier
        if self.cafo_loss_fn == 'MSE':
            self.train_cafo_closeform_mse(train_features, train_labels)
        else:
            self.train_cafo_gradient_descent(train_features, train_labels)
        
        print(f"CaFO Block trained with {self.cafo_loss_fn} loss")
        
        # Return features for next block (detached to prevent gradients)
        return gnn_features.detach()


class LayerWiseCaFoGNN(nn.Module):
    """
    A GNN that implements layer-by-layer CaFO training.
    
    This trains each GNN layer progressively using the CaFO algorithm,
    similar to the original CaFO paper but for graph neural networks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the layer-wise CaFO GNN.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Model parameters
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config.get('num_layers', 2)
        self.hidden_dims = config.get('hidden_dims', [64] * (self.num_layers - 1))
        
        # GNN layer configuration
        self.gnn_type = config.get('gnn_type', 'gcn').lower()
        self.gnn_config = config.get('gnn_config', {})
        
        # CaFO configuration
        self.cafo_config = {
            'loss_fn': config.get('cafo_loss_fn', 'MSE'),
            'lambda': config.get('cafo_lambda', 0.001),
            'num_epochs': config.get('cafo_num_epochs', 100),
            'step': config.get('cafo_step', 0.01),
            'num_batches': config.get('cafo_num_batches', 1)
        }
        
        # Create CaFO blocks
        self.cafo_blocks = nn.ModuleList()
        self._create_cafo_blocks()
        
        # Track training state
        self.blocks_trained = 0
        
        self.to(self.device)
    
    def _create_gnn_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """
        Create a GNN layer based on configuration.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            
        Returns:
            GNN layer instance
        """
        if self.gnn_type == 'gcn':
            from torch_geometric.nn import GCNConv
            return GCNConv(
                in_dim, out_dim,
                normalize=self.gnn_config.get('normalize', True),
                add_self_loops=self.gnn_config.get('add_self_loops', True),
                bias=self.gnn_config.get('bias', True)
            )
        elif self.gnn_type == 'gat':
            from torch_geometric.nn import GATConv
            return GATConv(
                in_dim, out_dim,
                heads=self.gnn_config.get('heads', 1),
                concat=self.gnn_config.get('concat', True),
                dropout=self.gnn_config.get('dropout', 0.0),
                add_self_loops=self.gnn_config.get('add_self_loops', True),
                bias=self.gnn_config.get('bias', True)
            )
        elif self.gnn_type == 'sage':
            from torch_geometric.nn import SAGEConv
            return SAGEConv(
                in_dim, out_dim,
                normalize=self.gnn_config.get('normalize', False),
                root_weight=self.gnn_config.get('root_weight', True),
                bias=self.gnn_config.get('bias', True)
            )
        elif self.gnn_type == 'gin':
            from torch_geometric.nn import GINConv
            # GIN requires an MLP as input
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(mlp, train_eps=self.gnn_config.get('train_eps', False))
        elif self.gnn_type == 'rgcn':
            from torch_geometric.nn import RGCNConv
            num_relations = self.gnn_config.get('num_relations', 1)
            return RGCNConv(
                in_dim, out_dim, num_relations,
                num_blocks=self.gnn_config.get('num_blocks', None),
                bias=self.gnn_config.get('bias', True)
            )
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}. "
                           f"Supported types: gcn, gat, sage, gin, rgcn")

    def _create_cafo_blocks(self):
        """Create CaFO blocks for each layer."""
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # Create GNN layer based on configuration
            gnn_layer = self._create_gnn_layer(in_dim, out_dim)
            
            # Create CaFO block
            cafo_block = GNNCaFoBlock(
                gnn_layer=gnn_layer,
                in_channels=in_dim,
                out_channels=out_dim,
                num_classes=self.output_dim,
                cafo_config=self.cafo_config
            )
            
            self.cafo_blocks.append(cafo_block)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all trained blocks.
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Final predictions [num_nodes, output_dim]
        """
        current_features = x
        predictions = torch.zeros(x.shape[0], self.output_dim, device=self.device)
        
        for i, block in enumerate(self.cafo_blocks):
            if i < self.blocks_trained:
                # Use trained CaFO weights
                current_features = block.forward(current_features, edge_index)
                block_predictions = block.cafo_predict(current_features)
                predictions += block_predictions
            else:
                # Block not trained yet
                break
        
        return predictions
    
    def train_next_block(self, x: torch.Tensor, edge_index: torch.Tensor, 
                        y: torch.Tensor, train_mask: Optional[torch.Tensor] = None):
        """
        Train the next CaFO block in sequence.
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            y: Target labels [num_nodes]
            train_mask: Optional training mask [num_nodes]
        """
        if self.blocks_trained >= len(self.cafo_blocks):
            print("All blocks already trained!")
            return
        
        print(f"Training CaFO Block {self.blocks_trained + 1}/{len(self.cafo_blocks)}")
        
        # Get features for current block
        current_features = x
        for i in range(self.blocks_trained):
            current_features = self.cafo_blocks[i].forward(current_features, edge_index)
        
        # Train current block
        block = self.cafo_blocks[self.blocks_trained]
        output_features = block.train_cafo_block(current_features, edge_index, y, train_mask)
        
        self.blocks_trained += 1
        print(f"Block {self.blocks_trained} training completed!")
    
    def train_all_blocks(self, x: torch.Tensor, edge_index: torch.Tensor, 
                        y: torch.Tensor, train_mask: Optional[torch.Tensor] = None):
        """
        Train all CaFO blocks sequentially.
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            y: Target labels [num_nodes]
            train_mask: Optional training mask [num_nodes]
        """
        print("Starting layer-wise CaFO training...")
        
        for i in range(len(self.cafo_blocks)):
            self.train_next_block(x, edge_index, y, train_mask)
        
        print("All CaFO blocks trained successfully!")
    
    def evaluate(self, x: torch.Tensor, edge_index: torch.Tensor, 
                y: torch.Tensor, eval_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges] 
            y: Target labels [num_nodes]
            eval_mask: Optional evaluation mask [num_nodes]
            
        Returns:
            Dictionary with accuracy and loss
        """
        self.eval()
        
        with torch.no_grad():
            predictions = self.forward(x, edge_index)
            
            if eval_mask is not None:
                predictions = predictions[eval_mask]
                labels = y[eval_mask]
            else:
                labels = y
            
            pred_classes = predictions.argmax(dim=1)
            accuracy = (pred_classes == labels).float().mean().item()
            
            # Simple loss (could be improved)
            loss = F.cross_entropy(predictions, labels).item()
        
        return {'accuracy': accuracy, 'loss': loss}


def create_layer_wise_cafo_gnn(config: Dict[str, Any]) -> LayerWiseCaFoGNN:
    """
    Factory function to create a layer-wise CaFO GNN.
    
    Args:
        config: Configuration dictionary containing:
            - input_dim: Input feature dimension
            - output_dim: Output dimension (number of classes)
            - num_layers: Number of layers (default: 2)
            - hidden_dims: List of hidden dimensions (default: [64])
            - gnn_type: Type of GNN layer ('gcn', 'gat', 'sage', 'gin', 'rgcn')
            - gnn_config: GNN-specific configuration parameters
            - cafo_loss_fn: CaFO loss function ('MSE' or 'CE')
            - cafo_lambda: Regularization strength
            - cafo_num_epochs: Training epochs per block
            - cafo_step: Learning rate for gradient descent
            - device: Training device
        
    Returns:
        LayerWiseCaFoGNN instance
    """
    return LayerWiseCaFoGNN(config)


class STGNNCaFoBlock(GNNCaFoBlock):
    """
    A specialized CaFO block for STGNN (Spatio-Temporal Graph Neural Networks).
    
    Inherits from GNNCaFoBlock and adapts it for spatio-temporal forecasting tasks.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_targets: int, cafo_config: Dict[str, Any]):
        """
        Initialize an STGNN CaFO block.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension  
            num_targets: Number of prediction targets (for forecasting)
            cafo_config: CaFO-specific configuration
        """
        # Initialize parent with dummy GNN layer (not used for STGNN)
        super().__init__(
            gnn_layer=None, 
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_targets,  # Use num_targets as num_classes
            cafo_config=cafo_config
        )
        
        self.num_targets = num_targets
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Forward pass (simplified for STGNN - just pass through features)."""
        return x
    
    def train_cafo_block(self, features: torch.Tensor, adj: torch.Tensor, 
                        targets: torch.Tensor) -> torch.Tensor:
        """
        Train the STGNN CaFO block for forecasting.
        
        Args:
            features: Input features [batch, channels, nodes, time]
            adj: Adjacency matrix (not used)
            targets: Target values for forecasting
            
        Returns:
            Output features from this block
        """
        # Reshape for CaFO training: [batch*nodes*time, channels]
        batch_size, channels, num_nodes, seq_len = features.shape
        reshaped_features = features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Reshape targets: [batch*nodes*time, out_dim]
        if targets.dim() == 4:  # [batch, out_dim, nodes, time]
            reshaped_targets = targets.permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        else:
            reshaped_targets = targets.reshape(-1, targets.shape[-1])
        
        # Ensure same number of samples
        min_samples = min(reshaped_features.shape[0], reshaped_targets.shape[0])
        reshaped_features = reshaped_features[:min_samples]
        reshaped_targets = reshaped_targets[:min_samples]
        
        # Train CaFO weights using ridge regression
        self._train_ridge_regression(reshaped_features, reshaped_targets)
        
        return features.detach()
    
    def _train_ridge_regression(self, features: torch.Tensor, targets: torch.Tensor):
        """Train CaFO weights using ridge regression for forecasting."""
        device = features.device
        
        # Add bias term
        bias_term = torch.ones(features.shape[0], 1, device=device)
        inputs = torch.cat([features, bias_term], dim=1)
        
        # Ridge regression: (X^T X + λI)^(-1) X^T y
        A = torch.matmul(inputs.t(), inputs)
        B = torch.matmul(inputs.t(), targets)
        
        # Add regularization
        reg_strength = max(self.cafo_lamda, 1e-8)
        A += reg_strength * torch.eye(A.shape[0], device=device)
        
        # Solve linear system
        try:
            self.cafo_weights = torch.linalg.solve(A, B)
        except Exception:
            self.cafo_weights = torch.matmul(torch.pinverse(A), B)


class LayerWiseCaFoSTGNN(nn.Module):
    """A simplified layer-wise CaFO implementation for STGNN forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the layer-wise CaFO STGNN."""
        super().__init__()
        
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Extract STGNN parameters
        stgnn_config = config.get('stgnn_config', {})
        self.num_nodes = stgnn_config.get('num_nodes', 50)
        self.out_dim = stgnn_config.get('out_dim', 3)
        self.conv_channels = stgnn_config.get('conv_channels', 32)
        self.num_layers = config.get('num_layers', 3)
        
        # CaFO configuration
        self.cafo_config = {
            'loss_fn': config.get('cafo_loss_fn', 'MSE'),
            'lambda': config.get('cafo_lambda', 0.001),
            'num_epochs': config.get('cafo_num_epochs', 100),
            'step': config.get('cafo_step', 0.01),
            'num_batches': config.get('cafo_num_batches', 1)
        }
        
        # Create CaFO blocks
        self.cafo_blocks = nn.ModuleList([
            STGNNCaFoBlock(
                in_channels=self.conv_channels,
                out_channels=self.conv_channels,
                num_targets=1,  # Single target for forecasting
                cafo_config=self.cafo_config
            ) for _ in range(self.num_layers)
        ])
        
        self.blocks_trained = 0
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through all trained blocks."""
        batch_size = x.shape[0]
        predictions = torch.zeros(batch_size, self.out_dim, self.num_nodes, 1, device=self.device)
        
        # Simple ensemble: each trained block contributes to predictions
        current_features = x.permute(0, 3, 2, 1)  # [batch, channels, nodes, seq_len]
        
        for i, block in enumerate(self.cafo_blocks[:self.blocks_trained]):
            if block.cafo_weights is not None:
                try:
                    # Reshape for prediction: [batch*nodes*time, channels]
                    reshaped_features = current_features.permute(0, 2, 3, 1).reshape(-1, current_features.shape[1])
                    
                    # Get block predictions
                    block_predictions = block.cafo_predict(reshaped_features)
                    
                    # Reshape and aggregate: [batch, nodes, 1] -> [batch, out_dim, nodes, 1]
                    block_pred_reshaped = block_predictions.reshape(batch_size, self.num_nodes, -1)
                    block_pred_final = block_pred_reshaped.mean(dim=2, keepdim=True)  # Average over time
                    
                    # Add to predictions
                    predictions += block_pred_final.unsqueeze(1).expand(-1, self.out_dim, -1, -1)
                    
                except RuntimeError:
                    continue  # Skip failed predictions
        
        return predictions
    
    def train_next_block(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor = None):
        """Train the next CaFO block in sequence."""
        if self.blocks_trained >= len(self.cafo_blocks):
            return
        
        # Convert to STGNN format: [batch, channels, nodes, seq_len]
        features = x.permute(0, 3, 2, 1)
        
        # Train current block
        block = self.cafo_blocks[self.blocks_trained]
        block.train_cafo_block(features, adj, y)
        
        self.blocks_trained += 1
    
    def train_all_blocks(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor = None):
        """Train all CaFO blocks sequentially."""
        for _ in range(len(self.cafo_blocks)):
            self.train_next_block(x, y, adj)
    
    def evaluate_forecasting(self, x: torch.Tensor, y: torch.Tensor, 
                           adj: torch.Tensor = None) -> Dict[str, float]:
        """Evaluate the model for forecasting."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, adj)
            mae = torch.abs(predictions - y).mean().item()
            mse = ((predictions - y) ** 2).mean().item()
            rmse = mse ** 0.5
        return {'mae': mae, 'mse': mse, 'rmse': rmse}


def create_layer_wise_cafo_stgnn(config: Dict[str, Any]) -> LayerWiseCaFoSTGNN:
    """Factory function to create a layer-wise CaFO STGNN."""
    return LayerWiseCaFoSTGNN(config)
