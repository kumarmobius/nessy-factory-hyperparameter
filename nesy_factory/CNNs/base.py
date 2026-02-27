"""
Base CNN class that all CNN models should inherit from.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union

class BaseCNN(nn.Module, ABC):
    """
    Abstract base class for all CNN models.
    
    This class provides the common interface and initialization that all CNN models
    should implement. It supports various computer vision tasks including classification,
    segmentation, and object detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base CNN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(BaseCNN, self).__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Model architecture parameters
        self.input_channels = config.get('input_channels', 3)  # RGB by default
        self.input_size = config.get('input_size', (224, 224))  # (height, width)
        self.output_dim = config['output_dim']  # Number of classes for classification
        
        # CNN architecture parameters
        self.num_blocks = config.get('num_blocks', 4)  # Number of convolutional blocks
        self.base_channels = config.get('base_channels', 64)  # Base number of channels
        self.channel_multiplier = config.get('channel_multiplier', 2)  # How channels increase per block
        
        # Handle custom channel dimensions
        channels = config.get('channels')
        if isinstance(channels, (list, tuple)):
            self.channels = list(channels)
        else:
            self.channels = self._compute_channel_dims()
        
        # Convolutional parameters
        self.kernel_size = config.get('kernel_size', 3)
        self.stride = config.get('stride', 1)
        self.padding = config.get('padding', 1)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.activation = config.get('activation', 'relu')
        self.pooling = config.get('pooling', 'max')  # 'max', 'avg', or 'adaptive'
        self.pool_size = config.get('pool_size', 2)
        
        # Fully connected parameters
        self.use_dropout = config.get('use_dropout', True)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.hidden_dims = config.get('hidden_dims', [512, 256])  # FC layer dimensions
        
        # Task-specific parameters
        self.task_type = config.get('task_type', 'classification')  # 'classification', 'segmentation', 'detection'
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.epochs = config.get('epochs', 100)
        self.optimizer_type = config.get('optimizer', 'adam').lower()
        
        # Optimizer-specific parameters
        self.momentum = config.get('momentum', 0.9)  # For SGD
        self.alpha = config.get('alpha', 0.99)  # For RMSprop
        self.eps = config.get('eps', 1e-8)  # For Adam, AdamW, RMSprop
        self.betas = config.get('betas', (0.9, 0.999))  # For Adam, AdamW
        
        # Initialize optimizer and loss function
        self.optimizer = None
        self.criterion = None
        
        # Build the model architecture
        self._build_layers()
        
        # Move model to device
        self.to(self.device)
    
    def _compute_channel_dims(self) -> List[int]:
        """Compute channel dimensions for each block."""
        channels = []
        current_channels = self.base_channels
        for i in range(self.num_blocks):
            channels.append(current_channels)
            current_channels = int(current_channels * self.channel_multiplier)
        return channels
    
    def _build_layers(self):
        """Build the CNN architecture based on configuration."""
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(self.channels):
            conv_block = self.create_conv_block(in_channels, out_channels)
            self.conv_blocks.append(conv_block)
            in_channels = out_channels
        
        # Pooling layer
        self.pool = self.create_pooling_layer()
        
        # Calculate FC input size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_channels, *self.input_size)
            for conv_block in self.conv_blocks:
                dummy_input = conv_block(dummy_input)
                dummy_input = self.pool(dummy_input)
            fc_input_size = dummy_input.view(1, -1).size(1)
        
        # Fully connected layers
        fc_layers = []
        prev_dim = fc_input_size
        
        for hidden_dim in self.hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate) if self.use_dropout else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        fc_layers.append(nn.Linear(prev_dim, self.output_dim))
        self.classifier = nn.Sequential(*fc_layers)
    
    def _init_optimizer_and_criterion(self):
        """Initialize optimizer and loss criterion based on task type."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        if self.criterion is None:
            if self.task_type == 'classification':
                self.criterion = nn.CrossEntropyLoss()
            elif self.task_type == 'segmentation':
                self.criterion = nn.CrossEntropyLoss()  # Can be changed to DiceLoss, etc.
            elif self.task_type == 'detection':
                self.criterion = nn.MSELoss()  # Placeholder, should be task-specific
            else:
                self.criterion = nn.MSELoss()
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor shape depends on task type:
            - Classification: [batch_size, output_dim]
            - Segmentation: [batch_size, num_classes, height, width]
            - Detection: [batch_size, ...] (task-specific)
        """
        # Default implementation for BaseCNN
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def create_conv_block(self, in_channels: int, out_channels: int, 
                         kernel_size: int = None, stride: int = None, 
                         padding: int = None) -> nn.Sequential:
        """
        Create a convolutional block with optional batch norm and activation.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            
        Returns:
            Sequential block of layers
        """
        kernel_size = kernel_size or self.kernel_size
        stride = stride or self.stride
        padding = padding or self.padding
        
        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride, padding, bias=not self.use_batch_norm))
        
        # Batch normalization
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if self.activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        elif self.activation == 'selu':
            layers.append(nn.SELU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def create_pooling_layer(self) -> nn.Module:
        """Create pooling layer based on configuration."""
        if self.pooling == 'max':
            return nn.MaxPool2d(self.pool_size)
        elif self.pooling == 'avg':
            return nn.AvgPool2d(self.pool_size)
        elif self.pooling == 'adaptive':
            return nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
    
    def train_step(self, data, targets=None) -> float:
        """
        Perform a single training step.
        
        Args:
            data: Input tensor of shape [batch_size, channels, height, width]
            targets: Target labels/tensors
            
        Returns:
            Loss value for this training step
        """
        # Initialize optimizer and criterion if not already done
        self._init_optimizer_and_criterion()
        
        self.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = self._to_device(data)
        if targets is not None:
            targets = self._to_device(targets)
        
        # Forward pass
        outputs = self.forward(data)
        
        # Compute loss
        if targets is not None:
            loss = self.criterion(outputs, targets)
        else:
            # Assume data is a tuple of (inputs, targets)
            loss = self.criterion(outputs, data[1])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, targets=None) -> Dict[str, float]:
        """
        Perform evaluation step.
        
        Args:
            data: Evaluation data
            targets: Target labels/tensors
            
        Returns:
            Dictionary containing loss and accuracy/metrics
        """
        self._init_optimizer_and_criterion()
        self.eval()
        
        with torch.no_grad():
            # Move data to device
            data = self._to_device(data)
            if targets is not None:
                targets = self._to_device(targets)
            
            # Forward pass
            outputs = self.forward(data)
            
            # Compute loss and metrics
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                # Assume data is a tuple of (inputs, targets)
                loss = self.criterion(outputs, data[1])
                targets = data[1]
            
            # Compute accuracy for classification tasks
            metrics = {'loss': loss.item()}
            
            if self.task_type == 'classification':
                pred = outputs.argmax(dim=1)
                correct = pred.eq(targets).sum().item()
                accuracy = correct / targets.size(0)
                metrics['accuracy'] = accuracy
            
            # Add task-specific metrics here
            
        return metrics
    
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
            return self.forward(data)
    
    def predict_proba(self, data) -> torch.Tensor:
        """
        Make probability predictions on new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Probability tensor
        """
        self.eval()
        with torch.no_grad():
            data = self._to_device(data)
            outputs = self.forward(data)
            if self.task_type == 'classification':
                return torch.softmax(outputs, dim=1)
            return outputs
    
    def _to_device(self, data):
        """Move data to the model's device."""
        if hasattr(data, 'to'):
            return data.to(self.device)
        elif isinstance(data, (tuple, list)):
            return tuple(item.to(self.device) if hasattr(item, 'to') else item for item in data)
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
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'output_dim': self.output_dim,
            'num_blocks': self.num_blocks,
            'base_channels': self.base_channels,
            'channels': self.channels,
            'task_type': self.task_type,
            'num_parameters': self.get_num_parameters(),
            'device': str(self.device),
            'optimizer': self.get_optimizer_info()
        }
        return info
    
    def get_feature_maps(self, data: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract feature maps from a specific layer.
        
        Args:
            data: Input tensor
            layer_idx: Layer index to extract features from (-1 for last layer)
            
        Returns:
            Feature maps tensor
        """
        self.eval()
        with torch.no_grad():
            data = self._to_device(data)
            
            # Hook to capture feature maps
            feature_maps = {}
            
            def get_features(name):
                def hook(model, input, output):
                    feature_maps[name] = output
                return hook
            
            # Get all layers including conv_blocks and classifier
            all_layers = []
            
            # Add conv_blocks layers
            for i, conv_block in enumerate(self.conv_blocks):
                all_layers.append(conv_block)
            
            # Add classifier layers if needed
            if hasattr(self, 'classifier'):
                for i, layer in enumerate(self.classifier):
                    all_layers.append(layer)
            
            # Handle negative indexing
            if layer_idx < 0:
                layer_idx = len(all_layers) + layer_idx
            
            # Check if layer_idx is valid
            if layer_idx < 0 or layer_idx >= len(all_layers):
                raise ValueError(f"Layer index {layer_idx} out of range. Available layers: 0 to {len(all_layers)-1}")
            
            target_layer = all_layers[layer_idx]
            
            # Register hook and run forward pass
            handle = target_layer.register_forward_hook(get_features(f'layer_{layer_idx}'))
            _ = self.forward(data)
            handle.remove()
            
            if f'layer_{layer_idx}' not in feature_maps:
                raise KeyError(f"Feature maps for layer {layer_idx} not captured. Available layers: {list(feature_maps.keys())}")
            
            return feature_maps[f'layer_{layer_idx}']
