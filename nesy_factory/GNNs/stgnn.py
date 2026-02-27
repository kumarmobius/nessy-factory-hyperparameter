"""
Spatio-Temporal Graph Neural Network (STGNN) for time-series forecasting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numbers
from typing import Dict, Any
import numpy as np
import json

from .base import BaseGNN
from ..utils.utils import AnomalyDetector

# Helper functions and classes from stgnn_standalone.py

def masked_mse(preds, labels, null_val=0.0):
    """
    Loss Function 1: Your Original Masked MSE Loss
    """
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def bce_with_logits_pos_weight(logits, targets):
    """
    Loss Function 2: BCEWithLogits + pos_weight
    Logit → risk: z = f_STGNN(X, A); risk = σ(z) = 1/(1+e^{-z})
    Loss: BCEWithLogits(z, y; pos_weight=neg/pos)
    Decision: risk ≥ τ (e.g., 0.5) ⇒ predicted outage
    """
    # Calculate pos_weight = neg_count / pos_count
    pos_count = targets.sum()
    total_count = targets.numel()
    neg_count = total_count - pos_count
    pos_weight = (neg_count / (pos_count + 1e-8)).clamp(min=1e-6, max=1e6)
    
    # BCEWithLogits loss with pos_weight
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return loss


def bce_calibrated_probability(logits, targets):
    """
    Loss Function 3: BCEWithLogits + pos_weight (Calibrated Probability)
    Same as left: risk = σ(z), trained with BCEWithLogits + pos_weight
    Risk is calibrated probability of next-step degradation
    """
    # Same implementation as function 2 but with different interpretation
    pos_count = targets.sum()
    total_count = targets.numel()
    neg_count = total_count - pos_count
    pos_weight = (neg_count / (pos_count + 1e-8)).clamp(min=1e-6, max=1e6)
    
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return loss


def class_weighted_bce(logits, targets):
    """
    Loss Function 4: Class-weighted BCE (focuses on rarer outage class)
    Same: risk = σ(z); class-weighted BCE focuses on the rarer outage class; τ=0.5 (or cost-aware)
    """
    # Calculate class frequencies
    pos_count = targets.sum()  # Outage class (rarer)
    neg_count = targets.numel() - pos_count  # Normal class
    total = pos_count + neg_count
    
    # Compute inverse frequency weights (focuses on rarer outage class)
    weight_0 = total / (2.0 * neg_count + 1e-8)  # Weight for normal class
    weight_1 = total / (2.0 * pos_count + 1e-8)  # Weight for outage class (higher)
    
    # Apply weights to each sample
    weights = torch.where(targets == 1, weight_1, weight_0)
    
    # Compute BCE loss with class weights
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weighted_loss = (bce_loss * weights).mean()
    
    return weighted_loss

LOSS_FUNCTIONS = {
    "masked_mse": masked_mse,
    "bce_with_logits_pos_weight": bce_with_logits_pos_weight,
    "bce_calibrated_probability": bce_calibrated_probability,
    "class_weighted_bce": class_weighted_bce,
}

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

class STGNN(BaseGNN):
    def __init__(self, config: Dict[str, Any]):
        # Manually set those attributes that are not part of the BaseGNN
        self.num_nodes = config['num_nodes']
        self.in_dim = config.get('in_dim', 1)
        config['input_dim'] = self.num_nodes * self.in_dim # BaseGNN expects input_dim
        config['output_dim'] = self.num_nodes * config.get('seq_out_len', 1) # BaseGNN expects output_dim
        super(STGNN, self).__init__(config)

        self.gcn_true = config.get('gcn_true', True)
        self.buildA_true = config.get('buildA_true', True)
        self.predefined_A = config.get('predefined_A', None)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 1)
        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)
        self.seq_length = config.get('seq_in_len', 12)
        self.out_dim = config.get('seq_out_len', 12)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.layer_norm_affline = config.get('layer_norm_affline', True)
        self.static_feat = config.get('static_feat', None)
        self.error_percentage_threshold = config.get('error_percentage_threshold', 0.1)
        self.error_absolute_threshold = config.get('error_absolute_threshold', 0.1)
        self.loss_function_name = config.get('loss_function', 'masked_mse')
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, self.node_dim, self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (self.dilation_exponential**self.layers - 1) / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (self.dilation_exponential**self.layers - 1) / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (self.dilation_exponential**j - 1) / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1), elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1), elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                             out_channels=self.end_channels,
                                             kernel_size=(1, 1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                             out_channels=self.out_dim,
                                             kernel_size=(1, 1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(self.device)
        self._init_optimizer_and_criterion() # Initialize optimizer and criterion here

    def _init_optimizer_and_criterion(self):
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        if self.criterion is None:
            self.criterion = LOSS_FUNCTIONS.get(self.loss_function_name, masked_mse)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs):
        # The input x from the dataloader is (batch_size, seq_len, num_nodes, in_dim)
        # The model expects (batch_size, in_dim, num_nodes, seq_len)
        input = x.permute(0, 3, 2, 1)

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if 'idx' in kwargs:
                    adp = self.gc(kwargs['idx'])
                else:
                    adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if 'idx' in kwargs:
                x = self.norm[i](x, kwargs['idx'])
            else:
                x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

    def _get_hidden_features(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Extract features from hidden layer for CaFO training.
        
        For STGNN, we extract features from the middle layer (skip connections)
        which captures spatio-temporal patterns suitable for CaFO training.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, in_dim]
            edge_index: Not used for STGNN (uses predefined adjacency)
            **kwargs: Additional arguments
            
        Returns:
            Hidden features tensor for CaFO training
        """
        # Convert input format to STGNN expected format
        input = x.permute(0, 3, 2, 1)  # [batch, in_dim, num_nodes, seq_len]
        
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if 'idx' in kwargs:
                    adp = self.gc(kwargs['idx'])
                else:
                    adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        x_hidden = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        
        # Extract features from the middle layer for CaFO training
        for i in range(min(self.layers // 2, self.layers)):  # Use first half of layers
            residual = x_hidden
            filter = self.filter_convs[i](x_hidden)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x_hidden)
            gate = torch.sigmoid(gate)
            x_hidden = filter * gate
            x_hidden = F.dropout(x_hidden, self.dropout, training=self.training)
            s = x_hidden
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x_hidden = self.gconv1[i](x_hidden, adp) + self.gconv2[i](x_hidden, adp.transpose(1, 0))
            else:
                x_hidden = self.residual_convs[i](x_hidden)

            x_hidden = x_hidden + residual[:, :, :, -x_hidden.size(3):]
            if 'idx' in kwargs:
                x_hidden = self.norm[i](x_hidden, kwargs['idx'])
            else:
                x_hidden = self.norm[i](x_hidden, self.idx)
        
        # Convert to appropriate format for CaFO training
        # [batch, channels, nodes, time] -> [batch*nodes*time, channels]
        batch_size, channels, num_nodes, time_steps = x_hidden.shape
        hidden_features = x_hidden.permute(0, 2, 3, 1).reshape(-1, channels)
        
        return hidden_features

    def _get_gnn_layers(self):
        """
        Get the GNN layers for enhanced CaFO block training.
        
        For STGNN, we return the graph convolution layers.
        
        Returns:
            List of mixprop layers (graph convolution layers)
        """
        return list(self.gconv1) + list(self.gconv2)

    def enable_layer_wise_cafo(self, cafo_config: dict = None):
        """
        Enable layer-wise CaFO training for this STGNN.
        
        This creates CaFO blocks specifically designed for spatio-temporal 
        forecasting tasks using the enhanced STGNN CaFO implementation.
        
        Args:
            cafo_config: CaFO configuration dictionary
                - loss_fn: 'MSE' or 'CE' (default: 'MSE')
                - lambda: regularization strength (default: 0.001)
                - num_epochs: training epochs per block (default: 100)
                - step: learning rate for gradient descent (default: 0.01)
        
        Returns:
            LayerWiseCaFoSTGNN instance for this STGNN
        """
        from .enhanced_cafo_blocks import create_layer_wise_cafo_stgnn
        
        if cafo_config is None:
            cafo_config = {}
        
        # Create configuration for layer-wise CaFO STGNN
        layer_wise_config = {
            'num_layers': self.layers,
            'stgnn_config': {
                'num_nodes': self.num_nodes,
                'out_dim': self.out_dim,
                'conv_channels': self.conv_channels,
            },
            'cafo_loss_fn': cafo_config.get('loss_fn', 'MSE'),
            'cafo_lambda': cafo_config.get('lambda', 0.001),
            'cafo_num_epochs': cafo_config.get('num_epochs', 100),
            'cafo_step': cafo_config.get('step', 0.01),
            'cafo_num_batches': cafo_config.get('num_batches', 1),
            'device': str(self.device)
        }
        
        print(f"🔧 Creating layer-wise CaFO STGNN with {self.layers} blocks")
        layer_wise_model = create_layer_wise_cafo_stgnn(layer_wise_config)
        
        print(f"✅ Layer-wise CaFO enabled for STGNN! Use .train_all_blocks() to train progressively")
        return layer_wise_model

    def enable_cafo_training(self, cafo_config: dict = None):
        """
        Enable CaFO training for this STGNN.
        
        This enables both classic CaFO training and enhanced layer-wise CaFO training.
        
        Args:
            cafo_config: CaFO configuration dictionary
                - loss_fn: 'MSE' or 'CE' (default: 'MSE')
                - lambda: regularization strength (default: 0.001)
                - num_epochs: training epochs (default: 100)
                - step: learning rate (default: 0.01)
                - algorithm_type: 'classic' or 'enhanced' (default: 'enhanced')
        
        Returns:
            Self for method chaining, or LayerWiseCaFoSTGNN if enhanced mode
        """
        if cafo_config is None:
            cafo_config = {}
        
        # Set default CaFO parameters specific to STGNN forecasting
        self.cafo_lambda = cafo_config.get('lambda', 0.001)
        self.cafo_num_epochs = cafo_config.get('num_epochs', 100)
        self.cafo_step = cafo_config.get('step', 0.01)
        self.cafo_loss_fn = cafo_config.get('loss_fn', 'MSE')
        self.cafo_algorithm_type = cafo_config.get('algorithm_type', 'enhanced')
        
        # Initialize CaFO weights for classic mode
        self.cafo_weights = None
        self.cafo_blocks = None
        
        print(f"🚀 CaFO training enabled for STGNN with {self.cafo_loss_fn} loss")
        print(f"   Algorithm: {self.cafo_algorithm_type}")
        print(f"   Lambda: {self.cafo_lambda}, Epochs: {self.cafo_num_epochs}")
        
        # If enhanced mode, create layer-wise CaFO
        if self.cafo_algorithm_type == 'enhanced':
            return self.enable_layer_wise_cafo(cafo_config)
        
        return self

    def _cafo_linear(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply CaFO linear transformation for classic CaFO training.
        
        Args:
            features: Input features tensor
            
        Returns:
            CaFO predictions
        """
        if self.cafo_weights is None:
            raise ValueError("CaFO weights not initialized. Enable CaFO training first.")
        
        # Add bias term if needed
        if features.dim() == 2:
            bias_term = torch.ones(features.shape[0], 1, device=features.device)
            features_with_bias = torch.cat([features, bias_term], dim=1)
        else:
            features_with_bias = features
        
        return torch.matmul(features_with_bias, self.cafo_weights)

    def _cafo_train_closeform_mse(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Train CaFO weights using closed-form MSE solution for STGNN forecasting.
        
        Args:
            features: Input features tensor
            targets: Target values for forecasting
        """
        device = features.device
        inputs = features.to(device)
        
        # For STGNN forecasting, targets are continuous values, not discrete classes
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Add bias term
        if inputs.dim() == 2:
            bias_term = torch.ones(inputs.shape[0], 1, device=device)
            inputs = torch.cat([inputs, bias_term], dim=1)
        
        # Ridge regression solution: (X^T X + λI)^(-1) X^T y
        A = torch.matmul(inputs.t(), inputs)
        B = torch.matmul(inputs.t(), targets.float())
        
        # Add regularization
        reg_strength = max(self.cafo_lambda, 1e-8)
        A += reg_strength * torch.eye(A.shape[0], device=device)
        
        # Solve linear system
        try:
            self.cafo_weights = torch.linalg.solve(A, B)
        except Exception as e:
            print(f"STGNN CaFO: Linear solve failed ({e}), using pseudo-inverse")
            self.cafo_weights = torch.matmul(torch.pinverse(A), B)

    def _cafo_compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CaFO loss for STGNN forecasting.
        
        Args:
            outputs: Model predictions
            targets: Target values
            
        Returns:
            Loss tensor
        """
        if self.cafo_loss_fn == 'MSE':
            return F.mse_loss(outputs, targets.float())
        else:
            # For other loss types, fall back to MSE for forecasting
            return F.mse_loss(outputs, targets.float())

    def train_step(self, data, mask=None):
        # Check if CaFO training is enabled
        if hasattr(self, 'cafo_lambda') and self.cafo_lambda is not None:
            return self._cafo_train_step(data, mask)
        
        # Regular training step
        self.train()
        total_loss = 0
        dataloader = data.train_loader
        scaler = data.scaler
        dataloader.shuffle()
        for i, (x, y) in enumerate(dataloader.get_iterator()):
            self.optimizer.zero_grad()
            x = torch.Tensor(x).to(self.device)
            y = torch.Tensor(y).to(self.device)
            
            y_pred_raw = self(x)
            y_pred = y_pred_raw.transpose(1,3)
            
            y_true_proc = y.permute(0, 3, 2, 1)

            if 'bce' in self.loss_function_name:
                loss = self.criterion(y_pred, y_true_proc)
            else:
                y_pred_scaled = scaler.inverse_transform(y_pred)
                loss = self.criterion(y_pred_scaled, y_true_proc)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / (i + 1) if i >= 0 else 0

    def _cafo_train_step(self, data, mask=None) -> float:
        """
        CaFO training step for STGNN.
        
        Args:
            data: Training data object with train_loader
            mask: Optional mask (not typically used for STGNN)
            
        Returns:
            Average loss for this training step
        """
        if not hasattr(self, 'cafo_algorithm_type'):
            self.cafo_algorithm_type = 'classic'
        
        # Use enhanced CaFO if enabled
        if self.cafo_algorithm_type == 'enhanced':
            return self._enhanced_cafo_train_step(data, mask)
        
        # Classic CaFO training
        self.train()
        total_loss = 0
        dataloader = data.train_loader
        scaler = data.scaler
        
        # Initialize CaFO training state if not already done
        if not hasattr(self, '_cafo_training_initialized'):
            self._cafo_training_initialized = True
            self._cafo_current_epoch = 0
            self._cafo_accumulated_features = None
            self._cafo_accumulated_targets = None
        
        dataloader.shuffle()
        for i, (x, y) in enumerate(dataloader.get_iterator()):
            x = torch.Tensor(x).to(self.device)
            y = torch.Tensor(y).to(self.device)
            
            # Get hidden features for CaFO training
            try:
                hidden_features = self._get_hidden_features(x)
            except Exception as e:
                print(f"Warning: Failed to extract hidden features for CaFO: {e}")
                # Fallback to regular training for this batch
                continue
            
            # Prepare targets - flatten to match feature format
            y_target = y.permute(0, 3, 2, 1).reshape(-1, y.shape[1])  # [batch*nodes*time, out_dim]
            
            # Ensure same number of samples
            min_samples = min(hidden_features.shape[0], y_target.shape[0])
            hidden_features = hidden_features[:min_samples]
            y_target = y_target[:min_samples]
            
            # Accumulate features and targets for batch training
            if self._cafo_accumulated_features is None:
                self._cafo_accumulated_features = hidden_features.detach()
                self._cafo_accumulated_targets = y_target.detach()
            else:
                self._cafo_accumulated_features = torch.cat([self._cafo_accumulated_features, hidden_features.detach()], dim=0)
                self._cafo_accumulated_targets = torch.cat([self._cafo_accumulated_targets, y_target.detach()], dim=0)
            
            # Train CaFO weights periodically
            if self._cafo_current_epoch % 5 == 0:  # Retrain every 5 epochs
                try:
                    self._cafo_train_closeform_mse(self._cafo_accumulated_features, self._cafo_accumulated_targets)
                except Exception as e:
                    print(f"Warning: CaFO training failed: {e}")
            
            # Compute loss using CaFO predictions
            if self.cafo_weights is not None:
                try:
                    cafo_outputs = self._cafo_linear(hidden_features)
                    loss = self._cafo_compute_loss(cafo_outputs, y_target)
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Warning: CaFO prediction failed: {e}")
                    total_loss += 0.0
            else:
                total_loss += 0.0
        
        self._cafo_current_epoch += 1
        
        return total_loss / (i + 1) if i >= 0 else 0

    def _enhanced_cafo_train_step(self, data, mask=None) -> float:
        """Enhanced CaFO training step using block-based approach for STGNN."""
        if self.cafo_blocks is None:
            self._init_cafo_blocks(data, mask)
        
        return self._train_cafo_blocks(data, mask)

    def _init_cafo_blocks(self, data, mask=None):
        """Initialize CaFO blocks for STGNN."""
        try:
            from .enhanced_cafo_blocks import LayerWiseCaFoSTGNN
            
            # Create layer-wise CaFO STGNN if not already done
            if not hasattr(self, '_layer_wise_cafo_model'):
                config = {
                    'num_layers': self.layers,
                    'stgnn_config': {
                        'num_nodes': self.num_nodes,
                        'out_dim': self.out_dim,
                        'conv_channels': self.conv_channels,
                    },
                    'cafo_loss_fn': getattr(self, 'cafo_loss_fn', 'MSE'),
                    'cafo_lambda': getattr(self, 'cafo_lambda', 0.001),
                    'cafo_num_epochs': getattr(self, 'cafo_num_epochs', 100),
                    'cafo_step': getattr(self, 'cafo_step', 0.01),
                    'device': str(self.device)
                }
                self._layer_wise_cafo_model = LayerWiseCaFoSTGNN(config)
            
            self.cafo_blocks = self._layer_wise_cafo_model.cafo_blocks
        except Exception as e:
            print(f"Warning: Failed to initialize enhanced CaFO blocks: {e}")
            self.cafo_blocks = None

    def _train_cafo_blocks(self, data, mask=None) -> float:
        """Train CaFO blocks for STGNN."""
        if self.cafo_blocks is None or not hasattr(self, '_layer_wise_cafo_model'):
            return 0.0
        
        try:
            dataloader = data.train_loader
            total_loss = 0
            count = 0
            
            for i, (x, y) in enumerate(dataloader.get_iterator()):
                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device)
                
                # Train next block
                self._layer_wise_cafo_model.train_next_block(x, y)
                
                # Compute loss using current trained blocks
                try:
                    predictions = self._layer_wise_cafo_model.forward(x)
                    loss = F.mse_loss(predictions, y)
                    total_loss += loss.item()
                    count += 1
                except Exception:
                    continue
            
            return total_loss / count if count > 0 else 0.0
        except Exception as e:
            print(f"Warning: Enhanced CaFO training failed: {e}")
            return 0.0

    def eval_step(self, data, mask=None):
        self.eval()
        total_eval_loss = 0
        total_correct_predictions = 0
        total_predictions = 0
        all_abs_errors = []
        dataloader = data.test_loader
        scaler = data.scaler
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader.get_iterator()):
                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device)
                
                y_pred_raw = self(x)
                y_pred = y_pred_raw.transpose(1,3)
                
                y_true_proc = y.permute(0, 3, 2, 1)

                if 'bce' in self.loss_function_name:
                    loss = self.criterion(y_pred, y_true_proc)
                    y_pred_scaled = torch.sigmoid(y_pred)
                else:
                    y_pred_scaled = scaler.inverse_transform(y_pred)
                    loss = self.criterion(y_pred_scaled, y_true_proc)

                total_eval_loss += loss.item()


                if 'bce' in self.loss_function_name:
                    predicted_classes = (y_pred_scaled > 0.5).float()
                    total_correct_predictions += (predicted_classes == y_true_proc).sum().item()
                    total_predictions += y_true_proc.numel()
                    abs_error = torch.abs(y_pred_scaled - y_true_proc)
                    all_abs_errors.append(abs_error.cpu())
                else:
                    # Calculate accuracy based on percentage or absolute error
                    abs_error = torch.abs(y_pred_scaled - y_true_proc)
                    all_abs_errors.append(abs_error.cpu())

                    # Condition 1: Absolute error is within the absolute threshold
                    correct_by_abs = abs_error <= self.error_absolute_threshold
                    
                    # Condition 2: Percentage error is within the percentage threshold
                    # To avoid division by zero, we only calculate relative error for non-zero true values
                    non_zero_mask = y_true_proc != 0
                    correct_by_perc = torch.zeros_like(y_true_proc, dtype=torch.bool)
                    if torch.any(non_zero_mask):
                        relative_error = abs_error[non_zero_mask] / torch.abs(y_true_proc[non_zero_mask])
                        correct_by_perc[non_zero_mask] = relative_error <= self.error_percentage_threshold
                    
                    # A prediction is considered correct if it satisfies either condition
                    correct_predictions = correct_by_abs | correct_by_perc

                    total_correct_predictions += correct_predictions.sum().item()
                    total_predictions += y_true_proc.numel()

        accuracy = (total_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Calculate and log error statistics
        if all_abs_errors:
            all_abs_errors = torch.cat(all_abs_errors)
            error_mean = all_abs_errors.mean().item()
            error_median = all_abs_errors.median().item()
            error_max = all_abs_errors.max().item()
            error_std = all_abs_errors.std().item()
        else:
            error_mean = error_median = error_max = error_std = 0.0


        print(f"\n--- Evaluation Error Statistics ---")
        print(f"Mean Absolute Error: {error_mean:.4f}")
        print(f"Median Absolute Error: {error_median:.4f}")
        print(f"Max Absolute Error: {error_max:.4f}")
        print(f"Std Dev of Absolute Error: {error_std:.4f}")
        print(f"------------------------------------")

        return {
            'loss': total_eval_loss / (i + 1) if i >= 0 else 0,
            'accuracy': accuracy,
            'error_mean': error_mean,
            'error_median': error_median,
            'error_max': error_max,
            'error_std': error_std
        }
    def infer_step(self, single_x, single_y_true, val_loader, y_val, scaler, num_components=8):
        """
        Performs inference on a single data point, including anomaly detection and RCA.

        Args:
            single_x (torch.Tensor): Input tensor for a single data point.
            single_y_true (np.array): Ground truth for the single data point.
            val_loader: DataLoader for validation data.
            y_val (np.array): Ground truth for validation data.
            scaler: The scaler object used for data normalization.
            num_components (int): Number of PCA components for anomaly detection.

        Returns:
            str: A JSON string with detailed inference and anomaly detection results.
        """
        self.eval()

        # --- 1. Single Point Inference ---
        with torch.no_grad():
            if single_x.dim() == 3:
                single_x = single_x.unsqueeze(0)
            
            pred_y_raw = self(single_x)
            pred_y = pred_y_raw.transpose(1, 3)

        pred_y_unscaled = scaler.inverse_transform(pred_y)
        pred_y_numpy = pred_y_unscaled.squeeze().cpu().detach().numpy()

        # --- 2. Get Validation Predictions for Anomaly Detection ---
        val_outputs = []
        for i, (x, y) in enumerate(val_loader.get_iterator()):
            val_x = torch.Tensor(x).to(self.device)
            with torch.no_grad():
                preds_raw = self(val_x)
            val_outputs.append(preds_raw.transpose(1, 3))

        val_yhat = torch.cat(val_outputs, dim=0)
        val_realy = torch.Tensor(y_val).to(self.device)
        val_yhat = val_yhat[:val_realy.size(0), ...]

        val_pred_unscaled = scaler.inverse_transform(val_yhat)
        val_pred_numpy = val_pred_unscaled.squeeze().cpu().detach().numpy()
        val_label_numpy = val_realy.squeeze().cpu().detach().numpy()

        # --- 3. Anomaly Detection ---
        anomaly_detector = AnomalyDetector(
            train_obs=np.array([]).reshape(0, val_label_numpy.shape[1]),
            val_obs=val_label_numpy,
            test_obs=np.array([single_y_true.squeeze()]),
            train_forecast=np.array([]).reshape(0, val_pred_numpy.shape[1]),
            val_forecast=val_pred_numpy,
            test_forecast=np.array([pred_y_numpy]),
            window_length=None,
            root_cause=True
        )

        indicator, prediction = anomaly_detector.scorer(num_components=num_components)

        # --- 4. Root Cause Analysis ---
        root_cause_analysis = {}
        if prediction[0]:
            error_contribution = anomaly_detector.test_re_full[0]
            sorted_indices = np.argsort(error_contribution)[::-1]
            root_cause_analysis['top_contributors'] = []
            for i in range(min(3, len(sorted_indices))):
                param_index = sorted_indices[i]
                root_cause_analysis['top_contributors'].append({
                    "parameter_index": int(param_index),
                    "contribution": float(error_contribution[param_index])
                })

        # --- 5. Compile Results into JSON ---
        result = {
            "inference": {
                "predicted_value": pred_y_numpy.tolist(),
                "true_value": single_y_true.squeeze().tolist()
            },
            "anomaly_detection": {
                "anomaly_score": float(indicator[0]),
                "result": 'Anomaly Detected' if prediction[0] else 'No Anomaly Detected'
            },
            "root_cause_analysis": root_cause_analysis if prediction[0] else "Not applicable"
        }

        return json.dumps(result, indent=2)
