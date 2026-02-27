"""
Base GNN class that all GNN models should inherit from.

Note: Legacy CaFO implementation in this file is DEPRECATED.
For CaFO functionality, use enhanced_cafo_blocks.py which provides:
- Better performance
- Cleaner API
- Support for STGNN and other specialized models
- True layer-wise CaFO training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import sys
import os

# Add CaFo module to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
cafo_path = os.path.join(os.path.dirname(current_dir), 'CaFo')
if cafo_path not in sys.path:
    sys.path.append(cafo_path)

try:
    from utils import one_hot, loss_MSE, jaccobian_MSE, loss_cross_entropy, jaccobian_cross_entropy, loss_sparsemax, jaccobian_sparsemax, sparsemax
except ImportError:
    # If CaFo utils not available, define dummy functions
    def one_hot(y, num_labels):
        h = torch.zeros(size=(y.shape[0], num_labels), dtype=torch.float32, device=y.device)
        h[range(y.shape[0]), y] = 1.0
        return h
    
    def loss_MSE(outputs, labels):
        diff = outputs - labels
        loss = torch.sqrt(diff * diff).sum(1).mean()
        return loss
    
    def jaccobian_MSE(outputs, labels):
        jacc = 2 * (outputs - labels)
        return jacc
    
    def loss_cross_entropy(outputs, labels):
        loss = -(labels * torch.log(F.softmax(outputs, dim=1))).sum(1).mean()
        return loss
    
    def jaccobian_cross_entropy(outputs, labels):
        s = F.softmax(outputs, dim=1)
        jacc = s - labels
        return jacc
    
    def sparsemax(z):
        sorted_z = z.sort(descending=True)[0]
        k_z = torch.zeros(size=(sorted_z.shape[0], 1), dtype=torch.int64, device=z.device)
        acc_z = sorted_z.clone()
        for k in range(1, sorted_z.shape[1]):
            acc_z[:, k] = acc_z[:, k - 1] + sorted_z[:, k]
            valid = (1.0 + (k + 1) * sorted_z[:, k]) > acc_z[:, k]
            k_z[valid] = k
        k_z = k_z.squeeze(-1)
        ta_z = (acc_z[range(acc_z.shape[0]), k_z] - 1.0) / (1 + k_z).float()
        ta_z = ta_z.unsqueeze(-1).repeat(1, z.shape[1])
        delta = z - ta_z
        p = F.relu(delta, inplace=True)
        return p, ta_z
    
    def loss_sparsemax(outputs, labels):
        p, ta_z = sparsemax(outputs)
        valid = (p > 0.0).float()
        loss = -outputs * labels + 0.5 * valid * (outputs * outputs - ta_z * ta_z) + 0.5
        loss = loss.sum(1).mean()
        return loss
    
    def jaccobian_sparsemax(outputs, labels):
        s, _ = sparsemax(outputs)
        jacc = s - labels
        return jacc


# Old CaFoBlock removed - use enhanced_cafo_blocks.py for CaFO functionality


class BaseGNN(nn.Module, ABC):
    """
    Abstract base class for all GNN models.
    
    This class provides the common interface and initialization that all GNN models
    should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base GNN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(BaseGNN, self).__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Model parameters
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.5)
        
        # Handle hidden dimensions - can be single value or list
        hidden_dim = config['hidden_dim']
        if isinstance(hidden_dim, (list, tuple)):
            self.hidden_dims = list(hidden_dim)
            self.hidden_dim = hidden_dim[0]  # For backward compatibility
        else:
            self.hidden_dim = hidden_dim
            self.hidden_dims = [hidden_dim] * max(1, self.num_layers - 1)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.epochs = config.get('epochs', 500)
        self.optimizer_type = config.get('optimizer', 'adam').lower()
        
        # Learning algorithm configuration
        self.learning_algo = config.get('learning_algo', 'backprop').lower()  # 'backprop' or 'cafo'
        
        # Optimizer-specific parameters
        self.momentum = config.get('momentum', 0.9)  # For SGD
        self.alpha = config.get('alpha', 0.99)  # For RMSprop
        self.eps = config.get('eps', 1e-8)  # For Adam, AdamW, RMSprop
        self.betas = config.get('betas', (0.9, 0.999))  # For Adam, AdamW
        
        # Note: For CaFO functionality, use enhanced_cafo_blocks.py
        # Legacy CaFO parameters removed for cleaner interface
        
        # Initialize optimizer and loss function
        self.optimizer = None
        self.criterion = None
        
        # Move model to device
        self.to(self.device)
    
    def _init_optimizer_and_criterion(self):
        """Initialize optimizer and loss criterion."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        if self.criterion is None:
            loss_function_name = self.config.get('loss_function', 'cross_entropy').lower()

            if loss_function_name == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss()
            elif loss_function_name == 'mse':
                self.criterion = nn.MSELoss()
            elif loss_function_name == 'bce_with_logits':
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unsupported loss function: {loss_function_name}")
    
    def _create_optimizer(self):
        """Create optimizer based on the specified type."""
        params = self.parameters()
        
        if self.optimizer_type == 'adam':
            return optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps
            )
        elif self.optimizer_type == 'sgd':
            return optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                alpha=self.alpha,
                eps=self.eps
            )
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps
            )
        elif self.optimizer_type == 'adagrad':
            return optim.Adagrad(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}. "
                           f"Supported types: adam, sgd, rmsprop, adamw, adagrad")
    
    def set_optimizer(self, optimizer_type: str, **optimizer_kwargs):
        """
        Change the optimizer type and parameters.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', 'adamw', 'adagrad')
            **optimizer_kwargs: Additional optimizer parameters to override defaults
        """
        self.optimizer_type = optimizer_type.lower()
        
        # Update optimizer parameters if provided
        if 'learning_rate' in optimizer_kwargs:
            self.learning_rate = optimizer_kwargs['learning_rate']
        if 'weight_decay' in optimizer_kwargs:
            self.weight_decay = optimizer_kwargs['weight_decay']
        if 'momentum' in optimizer_kwargs:
            self.momentum = optimizer_kwargs['momentum']
        if 'alpha' in optimizer_kwargs:
            self.alpha = optimizer_kwargs['alpha']
        if 'eps' in optimizer_kwargs:
            self.eps = optimizer_kwargs['eps']
        if 'betas' in optimizer_kwargs:
            self.betas = optimizer_kwargs['betas']
        
        # Recreate optimizer with new parameters
        self.optimizer = self._create_optimizer()
        print(f"Optimizer changed to {self.optimizer_type}")
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the current optimizer."""
        info = {
            'type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer_type == 'sgd':
            info['momentum'] = self.momentum
        elif self.optimizer_type == 'rmsprop':
            info['alpha'] = self.alpha
            info['eps'] = self.eps
        elif self.optimizer_type in ['adam', 'adamw']:
            info['betas'] = self.betas
            info['eps'] = self.eps
        elif self.optimizer_type == 'adagrad':
            info['eps'] = self.eps
            
        return info
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices tensor of shape [2, num_edges]
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Output tensor of shape [num_nodes, output_dim]
        """
        pass
    
    def _get_hidden_features(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the hidden layer(s) for CaFO training.
        
        This method should be overridden by concrete implementations to return
        features from the appropriate hidden layer instead of the final output layer.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices tensor of shape [2, num_edges]
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Hidden features tensor of shape [num_nodes, hidden_dim]
        """
        # Default implementation: use forward pass but issue warning
        print("Warning: Using default _get_hidden_features. "
              "Concrete models should override this for better CaFO performance.")
        output = self.forward(x, edge_index, **kwargs)
        
        # If output is in wrong dimension space, try to adapt
        if output.shape[-1] == self.output_dim and self.output_dim != self.hidden_dim:
            # Try to project back to hidden space using a linear layer
            if not hasattr(self, '_hidden_projection'):
                self._hidden_projection = nn.Linear(self.output_dim, self.hidden_dim).to(self.device)
            output = self._hidden_projection(output)
            print(f"Projected output from {self.output_dim} to {self.hidden_dim} dimensions for CaFO")
        
        return output
    
    def _cafo_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Linear transformation for CaFO with automatic bias handling."""
        if self.cafo_weights is None:
            raise ValueError("CaFO weights not initialized. Run CaFO training first.")
        
        # Add bias term to match training procedure
        if x.dim() == 2 and x.shape[1] == self.cafo_weights.shape[0] - 1:
            bias_term = torch.ones(x.shape[0], 1, device=x.device)
            x = torch.cat([x, bias_term], dim=1)
        
        # print(f"Debug _cafo_linear: x.shape={x.shape}, cafo_weights.shape={self.cafo_weights.shape}")
        
        # Handle potential dimension mismatches
        if x.shape[-1] != self.cafo_weights.shape[0]:
            # If feature dimension doesn't match, try transpose
            if x.shape[-1] == self.cafo_weights.shape[1]:
                # Transpose weights to match
                result = torch.matmul(x, self.cafo_weights.T)
            else:
                print(f"Shape mismatch warning: input {x.shape} vs weights {self.cafo_weights.shape}")
                # Try to handle gracefully
                if x.shape[-1] > self.cafo_weights.shape[0]:
                    x = x[..., :self.cafo_weights.shape[0]]
                else:
                    padding_size = self.cafo_weights.shape[0] - x.shape[-1]
                    padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device)
                    x = torch.cat([x, padding], dim=-1)
                result = torch.matmul(x, self.cafo_weights)
        else:
            result = torch.matmul(x, self.cafo_weights)
        
        # print(f"CAFO forward: input {x.shape} -> output {result.shape}")
        return result
    
    def _cafo_train_closeform_mse(self, features: torch.Tensor, y: torch.Tensor):
        """Train using closed-form MSE solution for CaFO with improved stability."""
        inputs = features.to(self.device)
        labels = one_hot(y, num_labels=self.output_dim).to(self.device)
        
        # Debug: print shapes for troubleshooting (comment out for production)
        # print(f"CAFO Debug - Features: {inputs.shape}, Labels: {labels.shape}")
        
        # Add bias term for better fitting
        if inputs.dim() == 2:
            bias_term = torch.ones(inputs.shape[0], 1, device=self.device)
            inputs = torch.cat([inputs, bias_term], dim=1)
        
        # Improved solution with regularization for numerical stability
        n_samples, n_features = inputs.shape
        
        if n_samples < n_features:
            # print(f"Warning: Few samples ({n_samples}) vs features ({n_features}). Using regularized pseudo-inverse.")
            # Add ridge regularization for better conditioning
            reg_strength = max(self.cafo_lamda, 1e-6)
            A = torch.matmul(inputs.t(), inputs) + reg_strength * torch.eye(n_features, device=self.device)
            B = torch.matmul(inputs.t(), labels)
            self.cafo_weights = torch.matmul(torch.pinverse(A), B)
        else:
            # Standard ridge regression solution for better stability
            A = torch.matmul(inputs.t(), inputs)
            B = torch.matmul(inputs.t(), labels)
            
            # Add regularization for numerical stability
            reg_strength = max(self.cafo_lamda, 1e-8)
            A += reg_strength * torch.eye(A.shape[0], device=self.device)
            
            try:
                self.cafo_weights = torch.linalg.solve(A, B)
                # print(f"CAFO: Solved linear system successfully")
            except Exception as e:
                print(f"CAFO: Linear solve failed ({e}), using pseudo-inverse")
                self.cafo_weights = torch.matmul(torch.pinverse(A), B)
        
        # print(f"CAFO weights shape: {self.cafo_weights.shape}")
        return self.cafo_weights
    
    def _cafo_compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CaFO loss with detailed reporting."""
        # Convert labels to appropriate format based on loss function
        if self.cafo_loss_fn == 'MSE':
            if labels.dim() == 1:
                labels = one_hot(labels, num_labels=self.output_dim).to(labels.device)
            data_loss = loss_MSE(outputs, labels)
        elif self.cafo_loss_fn == 'CE':
            if labels.dim() == 1:
                labels = one_hot(labels, num_labels=self.output_dim).to(labels.device)
            data_loss = loss_cross_entropy(outputs, labels)
        elif self.cafo_loss_fn == 'SL':
            if labels.dim() == 1:
                labels = one_hot(labels, num_labels=self.output_dim).to(labels.device)
            data_loss = loss_sparsemax(outputs, labels)
        else:
            data_loss = torch.tensor(0.0, device=outputs.device)
        
        # Add regularization term
        reg_loss = torch.tensor(0.0, device=outputs.device)
        if self.cafo_lamda > 1e-6 and self.cafo_weights is not None:
            reg_loss = 0.5 * self.cafo_lamda * self.cafo_weights.norm(p='fro')
        
        total_loss = data_loss + reg_loss
        
        # Print detailed loss breakdown for debugging (comment out for production)
        # print(f"CAFO Loss - Data: {data_loss.item():.6f}, Reg: {reg_loss.item():.6f}, Total: {total_loss.item():.6f}")
        
        return total_loss
    
    def _cafo_compute_gradient(self, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute gradients for CaFO."""
        if self.cafo_loss_fn == 'MSE':
            jacc = jaccobian_MSE(outputs, labels)
        elif self.cafo_loss_fn == 'CE':
            jacc = jaccobian_cross_entropy(outputs, labels)
        elif self.cafo_loss_fn == 'SL':
            jacc = jaccobian_sparsemax(outputs, labels)
        else:
            jacc = torch.zeros_like(outputs)
        
        jacc = jacc.unsqueeze(1)
        xx = inputs.unsqueeze(-1)
        delta_W = torch.matmul(xx, jacc).mean(0)
        
        # Add regularization term
        if self.cafo_lamda > 1e-4 and self.cafo_weights is not None:
            delta_W += self.cafo_lamda * self.cafo_weights
        
        return delta_W
    
    def _cafo_train_gradient_descent(self, features: torch.Tensor, y: torch.Tensor):
        """Train using gradient descent for CaFO."""
        num_samples = features.shape[0]
        num_dims = features.shape[1]
        
        # Initialize weights
        self.cafo_weights = torch.ones(size=(num_dims, self.output_dim), 
                                     dtype=torch.float32, device=self.device)
        
        inputs = features.to(self.device)
        labels = one_hot(y, num_labels=self.output_dim).to(self.device)
        
        step = self.cafo_step
        for idx in range(self.cafo_num_epochs):
            batch_size = max(1, int(num_samples / self.cafo_num_batches))
            
            for k in range(self.cafo_num_batches):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, num_samples)
                
                input_batch = inputs[start_idx:end_idx, :]
                label_batch = labels[start_idx:end_idx, :]
                
                outputs_batch = self._cafo_linear(input_batch)
                delta_W = self._cafo_compute_gradient(input_batch, outputs_batch, label_batch)
                self.cafo_weights += -step * delta_W
            
            if idx % 100 == 0:
                outputs = self._cafo_linear(inputs)
                loss = self._cafo_compute_loss(outputs, labels)
                print(f"CaFO Epoch {idx}: Loss = {loss.item():.6f}")
        
        return self.cafo_weights
    
    def _cafo_train_gradient_descent_incremental(self, features: torch.Tensor, y: torch.Tensor):
        """Incremental gradient descent training for CaFO - performs a few steps per call."""
        # Initialize weights if not already done
        if not hasattr(self, 'cafo_weights') or self.cafo_weights is None:
            num_dims = features.shape[1]
            self.cafo_weights = torch.randn(size=(num_dims, self.output_dim), 
                                         dtype=torch.float32, device=self.device) * 0.01
        
        inputs = features.to(self.device)
        labels = one_hot(y, num_labels=self.output_dim).to(self.device)
        
        step = self.cafo_step
        # Do a few gradient steps per train_step call (instead of all epochs at once)
        num_steps = min(5, self.cafo_num_epochs)  # 5 steps per call
        
        for _ in range(num_steps):
            outputs = self._cafo_linear(inputs)
            delta_W = self._cafo_compute_gradient(inputs, outputs, labels)
            self.cafo_weights += -step * delta_W
        
        return self.cafo_weights
    
    def reset_cafo_state(self):
        """Reset CAFO training state for fresh training."""
        if hasattr(self, '_cafo_training_initialized'):
            delattr(self, '_cafo_training_initialized')
        if hasattr(self, '_cafo_current_epoch'):
            delattr(self, '_cafo_current_epoch')
        if hasattr(self, '_cafo_accumulated_features'):
            delattr(self, '_cafo_accumulated_features')
        if hasattr(self, '_cafo_accumulated_labels'):
            delattr(self, '_cafo_accumulated_labels')
        self.cafo_weights = None
    
    def train_step(self, data, mask=None, learning_algo=None) -> float:
        """
        Perform a single training step.
        
        Args:
            data: Training data with x (node features), edge_index (edges), and y (labels)
            mask: Optional mask for selecting specific nodes for training
            learning_algo: Override the default learning algorithm ('backprop' or 'cafo')
            
        Returns:
            Loss value for this training step
        """
        # Use provided learning_algo or fall back to instance setting
        algo = learning_algo if learning_algo is not None else self.learning_algo
        
        # Move data to device
        data = self._to_device(data)
        
        if algo == 'cafo':
            return self._cafo_train_step(data, mask)
        else:
            return self._backprop_train_step(data, mask)
    
    def _backprop_train_step(self, data, mask=None) -> float:
        """Standard backpropagation training step."""
        self._init_optimizer_and_criterion()
        self.train()
        self.optimizer.zero_grad()

        # Handle multi-graph datasets like PPI
        if hasattr(data, 'dataset'):
            total_loss = 0
            for graph in data.dataset:
                graph = self._to_device(graph)
                node_embeddings = self.forward(graph.x, graph.edge_index)
                
                # For multi-label, BCEWithLogitsLoss is more appropriate
                if graph.y.dim() > 1 and graph.y.shape[1] > 1:
                    loss = F.binary_cross_entropy_with_logits(node_embeddings, graph.y.float())
                else:
                    loss = self.criterion(node_embeddings, graph.y)

                total_loss += loss.item()
                loss.backward()
            self.optimizer.step()
            return total_loss / len(data.dataset)

        # Forward pass for single graph
        node_embeddings = self.forward(data.x, data.edge_index)

        # Compute loss
        if mask is not None:
            mask = mask.to(self.device)
            out = node_embeddings[mask]
            labels = data.y[mask]
        else:
            # Graph-level or node-level prediction for a single graph
            if hasattr(data, 'batch') and data.batch is not None:
                from torch_geometric.nn import global_mean_pool
                out = global_mean_pool(node_embeddings, data.batch)
                labels = data.y
            elif hasattr(data, 'y') and (data.y.dim() == 0 or data.y.shape[0] != node_embeddings.shape[0]):
                out = node_embeddings.mean(dim=0, keepdim=True)
                labels = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y
            else:
                out = node_embeddings
                labels = data.y
        
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _cafo_train_step(self, data, mask=None) -> float:
        """Enhanced CaFO training step with block-based training."""
        algo_type = getattr(self, 'cafo_training_type', 'enhanced')  # 'classic' or 'enhanced'
        
        if algo_type == 'enhanced':
            return self._enhanced_cafo_train_step(data, mask)
        else:
            return self._classic_cafo_train_step(data, mask)
    
    def _classic_cafo_train_step(self, data, mask=None) -> float:
        """Classic CaFO training step with incremental training support."""
        self.train()
        
        # Initialize CAFO training state if not already done
        if not hasattr(self, '_cafo_training_initialized'):
            self._cafo_training_initialized = True
            self._cafo_current_epoch = 0
            self._cafo_accumulated_features = None
            self._cafo_accumulated_labels = None
        
        # Forward pass to get hidden layer embeddings (not final output)
        # This is crucial: CaFO needs features from hidden layer, not output layer
        node_embeddings = self._get_hidden_features(data.x, data.edge_index)
        
        # Apply mask if provided or handle graph-level prediction
        if mask is not None:
            mask = mask.to(self.device)
            features = node_embeddings[mask]
            labels = data.y[mask]
        else:
            # Check if this is graph-level prediction (single label for entire graph)
            if hasattr(data, 'batch') or (hasattr(data, 'y') and data.y.dim() == 0):
                # Graph-level task: aggregate node embeddings to graph embedding
                if hasattr(data, 'batch'):
                    # Batched graphs - use simple mean pooling to avoid dimension issues
                    try:
                        from torch_geometric.nn import global_mean_pool
                        features = global_mean_pool(node_embeddings, data.batch)
                    except ImportError:
                        # Fallback to manual mean pooling
                        batch_size = data.batch.max().item() + 1
                        features = []
                        for i in range(batch_size):
                            mask = data.batch == i
                            graph_feat = node_embeddings[mask].mean(dim=0, keepdim=True)
                            features.append(graph_feat)
                        features = torch.cat(features, dim=0)
                    labels = data.y
                else:
                    # Single graph: use simple mean pooling to avoid dimension issues
                    features = node_embeddings.mean(dim=0, keepdim=True)
                    labels = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y
                    
                # print(f"Graph-level features shape: {features.shape}")
            else:
                # Node-level task
                features = node_embeddings
                labels = data.y
        
        # Accumulate features and labels for batch training
        if self._cafo_accumulated_features is None:
            self._cafo_accumulated_features = features.detach()
            self._cafo_accumulated_labels = labels.detach()
        else:
            self._cafo_accumulated_features = torch.cat([self._cafo_accumulated_features, features.detach()], dim=0)
            self._cafo_accumulated_labels = torch.cat([self._cafo_accumulated_labels, labels.detach()], dim=0)
        
        # Incremental CAFO training
        if self.cafo_loss_fn == 'MSE':
            # For MSE, retrain with accumulated data every few steps
            if self._cafo_current_epoch % 5 == 0:  # Retrain every 5 epochs
                self._cafo_train_closeform_mse(self._cafo_accumulated_features, self._cafo_accumulated_labels)
        else:
            # For gradient descent, do a few iterations per step
            self._cafo_train_gradient_descent_incremental(features, labels)
        
        self._cafo_current_epoch += 1
        
        # Compute and return loss using current accumulated data
        outputs = self._cafo_linear(features)
        if labels.dim() == 1:
            labels_for_loss = one_hot(labels, num_labels=self.output_dim).to(self.device)
        else:
            labels_for_loss = labels
        loss = self._cafo_compute_loss(outputs, labels_for_loss)
        
        return loss.item()
    
    def _enhanced_cafo_train_step(self, data, mask=None) -> float:
        """Enhanced CaFO training step using block-based approach."""
        if self.cafo_blocks is None:
            self._init_cafo_blocks(data, mask)
        
        return self._train_cafo_blocks(data, mask)
    
    def _init_cafo_blocks(self, data, mask=None):
        """Initialize CaFo blocks based on the model's architecture."""
        # This should be implemented by concrete GNN classes
        # For now, we'll create a basic block structure
        if not hasattr(self, '_get_gnn_layers'):
            raise NotImplementedError("Model must implement _get_gnn_layers() method for enhanced CaFo training")
        
        gnn_layers = self._get_gnn_layers()
        self.cafo_blocks = nn.ModuleList()
        
        # Create CaFo blocks for each GNN layer
        for i, layer in enumerate(gnn_layers):
            if i == 0:
                in_channels = self.input_dim
            else:
                in_channels = self.hidden_dims[i-1] if hasattr(self, 'hidden_dims') else self.hidden_dim
            
            out_channels = self.hidden_dims[i] if hasattr(self, 'hidden_dims') and i < len(self.hidden_dims) else self.hidden_dim
            
            block = CaFoBlock(layer, in_channels, out_channels, self.output_dim)
            self.cafo_blocks.append(block)
    
    def _train_cafo_blocks(self, data, mask=None) -> float:
        """Train the model layer-by-layer using the enhanced CaFo algorithm."""
        data = self._to_device(data)
        current_features = data.x
        
        final_predictions = torch.zeros((current_features.shape[0], self.output_dim), device=self.device)
        total_loss = 0.0
        
        for i, block in enumerate(self.cafo_blocks):
            print(f"--- Training CaFo Block {i+1}/{len(self.cafo_blocks)} for {self.cafo_epochs_per_block} epochs ---")
            
            # Train the current block and get its output features for the next block
            current_features = block.train_block(
                current_features, 
                data.edge_index, 
                data, 
                mask,
                epochs=self.cafo_epochs_per_block,
                lr=self.cafo_block_lr,
                weight_decay=self.cafo_block_weight_decay
            )
            
            # Get predictions from this block and add to total
            with torch.no_grad():
                block_preds = block.predict(current_features)
                final_predictions += block_preds
                
                # Calculate loss for this block
                if mask is not None:
                    mask_device = mask.to(self.device)
                    block_loss = F.cross_entropy(block_preds[mask_device], data.y[mask_device].to(self.device))
                else:
                    block_loss = F.cross_entropy(block_preds, data.y.to(self.device))
                total_loss += block_loss.item()
            
            print(f"Block {i+1} trained. Block loss: {block_loss.item():.4f}")
        
        # Store final predictions for evaluation
        self.cafo_final_predictions = final_predictions.argmax(dim=1)
        
        return total_loss / len(self.cafo_blocks)
    
    def eval_step(self, data, mask=None) -> Dict[str, float]:
        """
        Perform evaluation step.
        
        Args:
            data: Evaluation data
            mask: Optional mask for selecting specific nodes for evaluation
            
        Returns:
            Dictionary containing loss and accuracy
        """
        self.eval()
        
        with torch.no_grad():
            # Move data to device
            data = self._to_device(data)
            
            # Forward pass
            if self.learning_algo == 'cafo':
                if hasattr(self, 'cafo_final_predictions') and self.cafo_final_predictions is not None:
                    # Use enhanced CaFo predictions
                    if mask is not None:
                        mask = mask.to(self.device)
                        pred = self.cafo_final_predictions[mask]
                        labels = data.y[mask].to(self.device)
                    else:
                        pred = self.cafo_final_predictions
                        labels = data.y.to(self.device)
                    
                    correct = pred.eq(labels).sum().item()
                    accuracy = correct / labels.size(0)
                    loss = torch.tensor(0.0)  # Loss already computed during training
                    
                elif self.cafo_weights is not None:
                    # Use classic CaFo weights with hidden features
                    node_embeddings = self._get_hidden_features(data.x, data.edge_index)
                    if mask is not None:
                        mask = mask.to(self.device)
                        features = node_embeddings[mask]
                        labels = data.y[mask]
                    else:
                        # Check if this is graph-level prediction
                        if hasattr(data, 'batch') or (hasattr(data, 'y') and data.y.dim() == 0):
                            # Graph-level task: aggregate node embeddings to graph embedding
                            if hasattr(data, 'batch'):
                                # Batched graphs
                                from torch_geometric.nn import global_mean_pool
                                features = global_mean_pool(node_embeddings, data.batch)
                                labels = data.y
                            else:
                                # Single graph: use mean pooling
                                features = node_embeddings.mean(dim=0, keepdim=True)
                                labels = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y
                        else:
                            # Node-level task
                            features = node_embeddings
                            labels = data.y
                    
                    out = self._cafo_linear(features)
                    labels_onehot = one_hot(labels, num_labels=self.output_dim).to(self.device)
                    loss = self._cafo_compute_loss(out, labels_onehot)
                    pred = out.argmax(dim=1)
                    correct = pred.eq(labels).sum().item()
                    accuracy = correct / labels.size(0)
                else:
                    # CaFo not trained yet, fall back to standard evaluation
                    self._init_optimizer_and_criterion()
                    node_embeddings = self.forward(data.x, data.edge_index)
                    
                    if mask is not None:
                        mask = mask.to(self.device)
                        out = node_embeddings[mask]
                        labels = data.y[mask].to(self.device)
                        loss = self.criterion(out, labels)
                        pred = out.argmax(dim=1)
                        correct = pred.eq(labels).sum().item()
                        accuracy = correct / mask.sum().item()
                    else:
                        # Check if this is graph-level prediction
                        if hasattr(data, 'batch') or (hasattr(data, 'y') and data.y.dim() == 0):
                            # Graph-level task: aggregate node embeddings to graph embedding
                            if hasattr(data, 'batch'):
                                # Batched graphs
                                from torch_geometric.nn import global_mean_pool
                                out = global_mean_pool(node_embeddings, data.batch)
                                labels = data.y.to(self.device)
                            else:
                                # Single graph: use mean pooling
                                out = node_embeddings.mean(dim=0, keepdim=True)
                                labels = data.y.unsqueeze(0).to(self.device) if data.y.dim() == 0 else data.y.to(self.device)
                        else:
                            # Node-level task
                            out = node_embeddings
                            labels = data.y.to(self.device)
                        
                        loss = self.criterion(out, labels)
                        pred = out.argmax(dim=1)
                        correct = pred.eq(labels).sum().item()
                        accuracy = correct / labels.size(0)
            else:
                # Standard backpropagation evaluation
                self._init_optimizer_and_criterion()

                # Handle multi-graph datasets like PPI
                if hasattr(data, 'dataset'):
                    total_loss = 0
                    total_correct = 0
                    total_nodes = 0
                    for graph in data.dataset:
                        graph = self._to_device(graph)
                        node_embeddings = self.forward(graph.x, graph.edge_index)

                        if graph.y.dim() > 1 and graph.y.shape[1] > 1:
                            loss = F.binary_cross_entropy_with_logits(node_embeddings, graph.y.float())
                            pred = (torch.sigmoid(node_embeddings) > 0.5).float()
                            total_correct += pred.eq(graph.y).sum().item()
                            total_nodes += graph.y.numel() # Count all labels
                        else:
                            loss = self.criterion(node_embeddings, graph.y)
                            pred = node_embeddings.argmax(dim=1)
                            total_correct += pred.eq(graph.y).sum().item()
                            total_nodes += graph.num_nodes

                        total_loss += loss.item()

                    loss = torch.tensor(total_loss / len(data.dataset))
                    accuracy = total_correct / total_nodes
                else:
                    node_embeddings = self.forward(data.x, data.edge_index)
                    # Compute loss and accuracy
                    if mask is not None:
                        mask = mask.to(self.device)
                        out = node_embeddings[mask]
                        labels = data.y[mask]
                        loss = self.criterion(out, labels)
                        pred = out.argmax(dim=1)
                        correct = pred.eq(labels).sum().item()
                        accuracy = correct / mask.sum().item()
                    else:
                        # Check if this is graph-level prediction
                        if hasattr(data, 'batch') or (hasattr(data, 'y') and data.y.dim() == 0):
                            # Graph-level task: aggregate node embeddings to graph embedding
                            if hasattr(data, 'batch'):
                                # Batched graphs
                                from torch_geometric.nn import global_mean_pool
                                out = global_mean_pool(node_embeddings, data.batch)
                                labels = data.y
                            else:
                                # Single graph: use mean pooling
                                out = node_embeddings.mean(dim=0, keepdim=True)
                                labels = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y
                        else:
                            # Node-level task
                            out = node_embeddings
                            labels = data.y
                        
                        loss = self.criterion(out, labels)
                        pred = out.argmax(dim=1)
                        correct = pred.eq(labels).sum().item()
                        accuracy = correct / labels.size(0)
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def predict(self, data) -> torch.Tensor:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            data = self._to_device(data)
            
            if self.learning_algo == 'cafo':
                if self.cafo_blocks is not None:
                    # Use enhanced CaFo blocks for prediction
                    return self._predict_with_cafo_blocks(data)
                elif self.cafo_weights is not None:
                    # Use classic CaFo weights
                    node_embeddings = self.forward(data.x, data.edge_index)
                    return self._cafo_linear(node_embeddings)
                else:
                    # CaFo not trained, fall back to standard forward
                    return self.forward(data.x, data.edge_index)
            else:
                # Standard forward pass
                return self.forward(data.x, data.edge_index)
    
    def _predict_with_cafo_blocks(self, data) -> torch.Tensor:
        """Make predictions using trained CaFo blocks."""
        current_features = data.x
        final_predictions = torch.zeros((current_features.shape[0], self.output_dim), device=self.device)
        
        for block in self.cafo_blocks:
            block.eval()
            # Forward through block
            current_features = block.forward(current_features, data.edge_index)
            # Get predictions from this block and add to total
            block_preds = block.predict(current_features)
            final_predictions += block_preds
            
        return final_predictions
    
    def _to_device(self, data):
        """Move data to the model's device."""
        if hasattr(data, 'to'):
            return data.to(self.device)
        return data
    
    def save_model(self, path: str):
        """Save the model state."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'criterion_state_dict': self.criterion.state_dict() if self.criterion else None
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.criterion and checkpoint.get('criterion_state_dict'):
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            
        print(f"Model loaded from {path}")
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        info = {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_parameters': self.get_num_parameters(),
            'device': str(self.device),
            'learning_algo': self.learning_algo,
            'optimizer': self.get_optimizer_info()
        }
        
        # Add CaFO-specific information if using CaFO
        if self.learning_algo == 'cafo':
            info['cafo_config'] = {
                'loss_fn': self.cafo_loss_fn,
                'num_epochs': self.cafo_num_epochs,
                'num_batches': self.cafo_num_batches,
                'step': self.cafo_step,
                'lambda': self.cafo_lamda,
                'weights_initialized': self.cafo_weights is not None,
                'enhanced_cafo': {
                    'epochs_per_block': self.cafo_epochs_per_block,
                    'num_blocks': self.cafo_num_blocks,
                    'block_lr': self.cafo_block_lr,
                    'block_weight_decay': self.cafo_block_weight_decay,
                    'blocks_initialized': self.cafo_blocks is not None
                }
            }
        
        return info
    
    def set_learning_algorithm(self, learning_algo: str, **cafo_kwargs):
        """
        Change the learning algorithm.
        
        Args:
            learning_algo: Type of learning algorithm ('backprop' or 'cafo')
            **cafo_kwargs: Additional CaFO parameters to override defaults
        """
        if learning_algo.lower() not in ['backprop', 'cafo']:
            raise ValueError(f"Unsupported learning algorithm: {learning_algo}. "
                           f"Supported types: backprop, cafo")
        
        self.learning_algo = learning_algo.lower()
        
        # Update CaFO parameters if provided
        if 'cafo_loss_fn' in cafo_kwargs:
            self.cafo_loss_fn = cafo_kwargs['cafo_loss_fn']
        if 'cafo_num_epochs' in cafo_kwargs:
            self.cafo_num_epochs = cafo_kwargs['cafo_num_epochs']
        if 'cafo_num_batches' in cafo_kwargs:
            self.cafo_num_batches = cafo_kwargs['cafo_num_batches']
        if 'cafo_step' in cafo_kwargs:
            self.cafo_step = cafo_kwargs['cafo_step']
        if 'cafo_lamda' in cafo_kwargs:
            self.cafo_lamda = cafo_kwargs['cafo_lamda']
            
        # Update enhanced CaFo parameters if provided
        if 'cafo_epochs_per_block' in cafo_kwargs:
            self.cafo_epochs_per_block = cafo_kwargs['cafo_epochs_per_block']
        if 'cafo_num_blocks' in cafo_kwargs:
            self.cafo_num_blocks = cafo_kwargs['cafo_num_blocks']
        if 'cafo_block_lr' in cafo_kwargs:
            self.cafo_block_lr = cafo_kwargs['cafo_block_lr']
        if 'cafo_block_weight_decay' in cafo_kwargs:
            self.cafo_block_weight_decay = cafo_kwargs['cafo_block_weight_decay']
        if 'cafo_training_type' in cafo_kwargs:
            self.cafo_training_type = cafo_kwargs['cafo_training_type']
        
        # Reset CaFO weights and blocks when switching algorithms
        if learning_algo.lower() == 'cafo':
            self.cafo_weights = None
            self.cafo_blocks = None
            self.cafo_final_predictions = None
            
        print(f"Learning algorithm changed to {self.learning_algo}")