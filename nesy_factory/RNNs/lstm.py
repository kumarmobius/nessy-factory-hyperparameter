import torch
import torch.nn as nn
import time
import numpy as np
from .base import BaseRNN
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class ForwardForwardLSTMBlock(nn.Module):
    """A single LSTM block for Forward Forward training."""
    
    def __init__(self, input_dim: int, hidden_dim: int, threshold: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        # Single LSTM layer per block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Normalization layer for goodness computation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)  # Set forget gate bias to 1
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass through the LSTM block."""
        batch_size = x.size(0)
        
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            hidden_state = (h0, c0)
        
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out = self.dropout_layer(lstm_out)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out, new_hidden
    
    def compute_goodness(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute goodness score for Forward Forward training."""
        # Use the sum of squares of hidden activations as goodness measure
        # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]
        goodness = torch.sum(hidden_states ** 2, dim=-1)
        # Average over sequence length: [batch_size, seq_len] -> [batch_size]
        goodness = torch.mean(goodness, dim=1)
        return goodness
    
    def forward_forward_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """Compute Forward Forward loss for this block."""
        # Positive examples should have goodness > threshold
        pos_loss = torch.log(1 + torch.exp(-(pos_goodness - self.threshold)))
        # Negative examples should have goodness < threshold  
        neg_loss = torch.log(1 + torch.exp(neg_goodness - self.threshold))
        
        return torch.mean(pos_loss) + torch.mean(neg_loss)


class CaFoLSTMBlock(nn.Module):
    """A single LSTM block for CAFO training."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Single LSTM layer per block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Local predictor for this block
        self.local_predictor = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)  # Set forget gate bias to 1
        
        nn.init.xavier_uniform_(self.local_predictor.weight)
        if self.local_predictor.bias is not None:
            nn.init.zeros_(self.local_predictor.bias)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass through the LSTM block."""
        batch_size = x.size(0)
        
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            hidden_state = (h0, c0)
        
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out = self.dropout_layer(lstm_out)
        
        return lstm_out, new_hidden
    
    def predict_local(self, hidden_states: torch.Tensor, task_type: str = 'sequence_to_one'):
        """Make predictions using the local predictor."""
        if task_type == 'sequence_to_one':
            last_hidden = hidden_states[:, -1, :]
            predictions = self.local_predictor(last_hidden)
        else:  # sequence_to_sequence
            predictions = self.local_predictor(hidden_states)
        return predictions


class LSTM(BaseRNN):
    def __init__(self, config: Dict[str, Any]):
        super(LSTM, self).__init__(config)
        
        # Training method selection
        self.use_cafo = config.get('use_cafo', False)
        self.use_forward_forward = config.get('use_forward_forward', False)
        
        # Validate training method selection
        if sum([self.use_cafo, self.use_forward_forward]) > 1:
            raise ValueError("Only one training method can be selected: use_cafo or use_forward_forward")
        
        # CAFO-specific parameters
        self.cafo_blocks = config.get('cafo_blocks', 3) if self.use_cafo else self.num_layers
        self.epochs_per_block = config.get('epochs_per_block', 50)
        self.block_lr = config.get('block_lr', 0.001)
        self.task_type = config.get('task_type', 'sequence_to_one')
        
        # Forward Forward specific parameters
        self.ff_blocks = config.get('ff_blocks', 3) if self.use_forward_forward else self.num_layers
        self.ff_threshold = config.get('ff_threshold', 2.0)
        self.ff_epochs_per_block = config.get('ff_epochs_per_block', 100)
        self.ff_lr = config.get('ff_lr', 0.03)
        
        if self.use_cafo:
            # Create CAFO blocks
            self.blocks = nn.ModuleList()
            self._create_cafo_blocks()
            self.cafo_trained = False
        elif self.use_forward_forward:
            # Create Forward Forward blocks
            self.ff_blocks_list = nn.ModuleList()
            self._create_ff_blocks()
            self.ff_trained = False
            # Final classifier for Forward Forward (use last block's hidden dim)
            final_hidden_dim = self.hidden_dim * 2 if self.ff_blocks > 1 else self.hidden_dim
            self.ff_classifier = nn.Linear(final_hidden_dim, self.output_dim)
        else:
            # Traditional LSTM
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
        self._init_optimizer_and_criterion()
    
    def _create_cafo_blocks(self):
        """Create CAFO LSTM blocks."""
        # Calculate hidden dimensions for each block
        hidden_dims = []
        if self.cafo_blocks == 1:
            hidden_dims = [self.hidden_dim]
        else:
            # Gradually increase then decrease hidden dimensions
            max_hidden = self.hidden_dim * 2
            for i in range(self.cafo_blocks):
                if i < self.cafo_blocks // 2:
                    # Increasing phase
                    dim = self.hidden_dim + (max_hidden - self.hidden_dim) * i // (self.cafo_blocks // 2)
                else:
                    # Decreasing phase
                    remaining = self.cafo_blocks - i - 1
                    dim = self.hidden_dim + (max_hidden - self.hidden_dim) * remaining // (self.cafo_blocks // 2)
                hidden_dims.append(int(dim))
        
        # Create blocks
        for i in range(self.cafo_blocks):
            input_dim = self.input_dim if i == 0 else hidden_dims[i-1]
            hidden_dim = hidden_dims[i]
            output_dim = self.output_dim  # All blocks predict to final output for training
            
            block = CaFoLSTMBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=self.dropout
            )
            self.blocks.append(block)
    
    def _create_ff_blocks(self):
        """Create Forward Forward LSTM blocks."""
        # Calculate hidden dimensions for each block
        hidden_dims = []
        if self.ff_blocks == 1:
            hidden_dims = [self.hidden_dim]
        else:
            # Gradually increase hidden dimensions
            for i in range(self.ff_blocks):
                dim = self.hidden_dim + (self.hidden_dim * i) // (self.ff_blocks - 1)
                hidden_dims.append(int(dim))
        
        # Create Forward Forward blocks
        for i in range(self.ff_blocks):
            input_dim = self.input_dim if i == 0 else hidden_dims[i-1]
            hidden_dim = hidden_dims[i]
            
            block = ForwardForwardLSTMBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                threshold=self.ff_threshold,
                dropout=self.dropout
            )
            self.ff_blocks_list.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cafo:
            if not self.cafo_trained:
                raise RuntimeError("CAFO model must be trained using train_cafo() before inference")
            return self._forward_cafo(x)
        elif self.use_forward_forward:
            if not self.ff_trained:
                raise RuntimeError("Forward Forward model must be trained using train_forward_forward() before inference")
            return self._forward_ff(x)
        else:
            return self._forward_traditional(x)
    
    def _forward_traditional(self, x: torch.Tensor) -> torch.Tensor:
        """Traditional LSTM forward pass."""
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out
    
    def _forward_cafo(self, x: torch.Tensor) -> torch.Tensor:
        """CAFO LSTM forward pass through all blocks."""
        current_input = x
        
        for i, block in enumerate(self.blocks):
            hidden_states, _ = block.forward(current_input)
            
            if i == len(self.blocks) - 1:  # Last block
                predictions = block.predict_local(hidden_states, self.task_type)
                return predictions
            else:
                current_input = hidden_states
        
        return current_input
    
    def _forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Forward LSTM forward pass through all blocks."""
        current_input = x
        
        # Pass through all FF blocks
        for i, block in enumerate(self.ff_blocks_list):
            hidden_states, _ = block.forward(current_input)
            current_input = hidden_states
        
        # Use last time step for classification
        final_hidden = current_input[:, -1, :]
        predictions = self.ff_classifier(final_hidden)
        return predictions
    
    def train_cafo(self, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM using CAFO methodology.
        
        Args:
            X_train: Training sequences [batch_size, sequence_length, input_dim]
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training results and metrics
        """
        if not self.use_cafo:
            raise ValueError("CAFO training is only available when use_cafo=True")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        if y_val is not None:
            y_val = y_val.to(self.device)
        
        if verbose:
            print(f"Starting CAFO LSTM training with {len(self.blocks)} blocks...")
            print(f"Architecture: {[self.input_dim] + [block.hidden_dim for block in self.blocks]} -> {self.output_dim}")
            print(f"Epochs per block: {self.epochs_per_block}")
        
        # Training results
        results = {
            'block_results': [],
            'total_training_time': 0,
            'all_hidden_states': []
        }
        
        start_time = time.time()
        current_input = X_train
        val_input = X_val
        
        # Train each block sequentially
        for i, block in enumerate(self.blocks):
            if verbose:
                print(f"\n--- Training Block {i+1}/{len(self.blocks)} ---")
                print(f"Input dim: {current_input.shape[-1]}, Hidden dim: {block.hidden_dim}")
            
            # Train current block
            block_results = self._train_single_block(
                block=block,
                X=current_input,
                y=y_train,
                val_X=val_input,
                val_y=y_val,
                verbose=verbose
            )
            
            results['block_results'].append(block_results)
            results['all_hidden_states'].append(block_results['final_hidden_states'])
            
            # Prepare input for next block (use hidden states as input)
            if i < len(self.blocks) - 1:  # Not the last block
                current_input = block_results['final_hidden_states'].detach()
                
                # Process validation input through current block
                if val_input is not None:
                    block.eval()
                    with torch.no_grad():
                        val_hidden_states, _ = block.forward(val_input)
                        val_input = val_hidden_states
        
        results['total_training_time'] = time.time() - start_time
        self.cafo_trained = True
        
        if verbose:
            print(f"\nCAFO LSTM training completed in {results['total_training_time']:.2f} seconds")
        
        return results
    
    def _generate_negative_examples(self, X_pos: torch.Tensor, y_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate negative examples by corrupting positive examples."""
        batch_size = X_pos.size(0)
        seq_len = X_pos.size(1)
        
        # Method 1: Shuffle features within each sequence
        X_neg = X_pos.clone()
        for i in range(batch_size):
            for t in range(seq_len):
                # Randomly permute features at each time step
                perm_idx = torch.randperm(X_pos.size(-1))
                X_neg[i, t] = X_pos[i, t, perm_idx]
        
        # Create negative labels (wrong class)
        y_neg = (y_pos + torch.randint(1, self.output_dim, (batch_size,), device=y_pos.device)) % self.output_dim
        
        return X_neg, y_neg
    
    def train_forward_forward(self, X_train: torch.Tensor, y_train: torch.Tensor,
                             X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
                             verbose: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM using Forward Forward methodology.
        
        Args:
            X_train: Training sequences [batch_size, sequence_length, input_dim]
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training results and metrics
        """
        if not self.use_forward_forward:
            raise ValueError("Forward Forward training is only available when use_forward_forward=True")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        if y_val is not None:
            y_val = y_val.to(self.device)
        
        if verbose:
            print(f"Starting Forward Forward LSTM training with {len(self.ff_blocks_list)} blocks...")
            print(f"Architecture: {[self.input_dim] + [block.hidden_dim for block in self.ff_blocks_list]}")
            print(f"Threshold: {self.ff_threshold}, Epochs per block: {self.ff_epochs_per_block}")
        
        # Generate negative examples
        X_neg, y_neg = self._generate_negative_examples(X_train, y_train)
        
        # Training results
        results = {
            'block_results': [],
            'total_training_time': 0
        }
        
        start_time = time.time()
        current_pos_input = X_train
        current_neg_input = X_neg
        
        # Train each block sequentially
        for i, block in enumerate(self.ff_blocks_list):
            if verbose:
                print(f"\n--- Training FF Block {i+1}/{len(self.ff_blocks_list)} ---")
                print(f"Input dim: {current_pos_input.shape[-1]}, Hidden dim: {block.hidden_dim}")
            
            # Train current block
            block_results = self._train_single_ff_block(
                block=block,
                X_pos=current_pos_input,
                X_neg=current_neg_input,
                verbose=verbose
            )
            
            results['block_results'].append(block_results)
            
            # Prepare input for next block (use hidden states as input)
            if i < len(self.ff_blocks_list) - 1:  # Not the last block
                # Process positive examples
                block.eval()
                with torch.no_grad():
                    pos_hidden, _ = block.forward(current_pos_input)
                    neg_hidden, _ = block.forward(current_neg_input)
                current_pos_input = pos_hidden.detach()
                current_neg_input = neg_hidden.detach()
        
        # Train final classifier
        self._train_ff_classifier(X_train, y_train, X_val, y_val, verbose)
        
        results['total_training_time'] = time.time() - start_time
        self.ff_trained = True
        
        if verbose:
            print(f"\nForward Forward LSTM training completed in {results['total_training_time']:.2f} seconds")
        
        return results
    
    def _train_single_ff_block(self, block: ForwardForwardLSTMBlock, 
                              X_pos: torch.Tensor, X_neg: torch.Tensor,
                              verbose: bool = True) -> Dict[str, Any]:
        """Train a single Forward Forward block."""
        # Setup optimizer for this block
        optimizer = torch.optim.SGD(block.parameters(), lr=self.ff_lr)
        
        # Training metrics
        losses = []
        
        # Training loop
        for epoch in range(self.ff_epochs_per_block):
            block.train()
            optimizer.zero_grad()
            
            # Forward pass for positive examples
            pos_hidden, _ = block.forward(X_pos)
            pos_goodness = block.compute_goodness(pos_hidden)
            
            # Forward pass for negative examples  
            neg_hidden, _ = block.forward(X_neg)
            neg_goodness = block.compute_goodness(neg_hidden)
            
            # Compute Forward Forward loss
            loss = block.forward_forward_loss(pos_goodness, neg_goodness)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % 20 == 0:
                pos_above_threshold = (pos_goodness > block.threshold).float().mean()
                neg_below_threshold = (neg_goodness < block.threshold).float().mean()
                print(f"    FF Epoch {epoch:2d}/{self.ff_epochs_per_block} - "
                      f"Loss: {loss.item():.6f}, "
                      f"Pos>Th: {pos_above_threshold:.3f}, "
                      f"Neg<Th: {neg_below_threshold:.3f}")
        
        return {
            'losses': losses,
            'final_loss': losses[-1] if losses else 0.0
        }
    
    def _train_ff_classifier(self, X_train: torch.Tensor, y_train: torch.Tensor,
                            X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
                            verbose: bool = True):
        """Train the final classifier for Forward Forward."""
        if verbose:
            print("\n--- Training Final Classifier ---")
        
        # Get features from trained FF blocks
        with torch.no_grad():
            current_input = X_train
            for block in self.ff_blocks_list:
                block.eval()
                hidden_states, _ = block.forward(current_input)
                current_input = hidden_states
            
            # Use last time step
            train_features = current_input[:, -1, :]
        
        # Setup classifier optimizer
        classifier_optimizer = torch.optim.Adam(self.ff_classifier.parameters(), lr=0.001)
        
        # Create data loader
        batch_size = 32
        train_dataset = TensorDataset(train_features, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train classifier with CrossEntropyLoss for classification
        classification_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(50):  # Fixed epochs for classifier
            self.ff_classifier.train()
            epoch_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                classifier_optimizer.zero_grad()
                
                predictions = self.ff_classifier(batch_features)
                loss = classification_criterion(predictions, batch_labels)
                
                loss.backward()
                classifier_optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and epoch % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"    Classifier Epoch {epoch:2d}/50 - Loss: {avg_loss:.6f}")
    
    def _train_single_block(self, block: CaFoLSTMBlock, X: torch.Tensor, y: torch.Tensor,
                           val_X: Optional[torch.Tensor] = None, val_y: Optional[torch.Tensor] = None,
                           verbose: bool = True) -> Dict[str, Any]:
        """Train a single CAFO block."""
        # Setup optimizer and criterion for this block
        optimizer = torch.optim.Adam(block.parameters(), lr=self.block_lr, weight_decay=1e-4)
        
        # Create data loaders
        batch_size = 32
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_X is not None and val_y is not None:
            val_dataset = TensorDataset(val_X, val_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Training loop
        for epoch in range(self.epochs_per_block):
            # Training phase
            block.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                hidden_states, _ = block.forward(batch_X)
                predictions = block.predict_local(hidden_states, self.task_type)
                
                # Compute loss - use CrossEntropyLoss for classification
                if isinstance(self.criterion, nn.MSELoss):
                    # Override with CrossEntropyLoss for classification tasks
                    classification_criterion = nn.CrossEntropyLoss()
                    loss = classification_criterion(predictions, batch_y)
                else:
                    loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                block.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        hidden_states, _ = block.forward(batch_X)
                        predictions = block.predict_local(hidden_states, self.task_type)
                        loss = self.criterion(predictions, batch_y)
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Track best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                
                if verbose and epoch % 10 == 0:
                    print(f"    Block Epoch {epoch:2d}/{self.epochs_per_block} - "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"    Block Epoch {epoch:2d}/{self.epochs_per_block} - Train Loss: {avg_train_loss:.6f}")
        
        # Generate final hidden representations for the entire dataset
        block.eval()
        with torch.no_grad():
            final_hidden_states, _ = block.forward(X)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_hidden_states': final_hidden_states,
            'final_model_state': block.state_dict()
        }