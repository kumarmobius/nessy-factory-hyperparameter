import torch
import torch.nn as nn
import os
import pandas as pd
import torch

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {name} - choose from 'relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid', or 'elu'")


def add_noise(x, noise_type="gaussian", noise_factor=0.2):
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * noise_factor
        return x + noise
    elif noise_type == "salt_pepper":
        mask = torch.rand_like(x)
        x_noisy = x.clone()
        x_noisy[mask < (noise_factor / 2)] = 0.0  # pepper
        x_noisy[mask > (1 - noise_factor / 2)] = 1.0  # salt
        return x_noisy
    elif noise_type == "none":
        return x
    else:
        raise ValueError(f"Unsupported noise type: {noise_type} - choose from 'gaussian', 'salt_pepper', or 'none'")
    
    
def build_mlp(input_dim, layer_dims, activation,
              dropout=0.0, batch_norm=False):

    layers = []
    prev_dim = input_dim

    for units in layer_dims:
        layers.append(nn.Linear(prev_dim, units))

        if batch_norm:
            layers.append(nn.BatchNorm1d(units))

        layers.append(get_activation(activation))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        prev_dim = units

    return nn.Sequential(*layers), prev_dim


def get_optimizer(name, model, lr, weight_decay, momentum=0.9):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name} - choose from 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta', or 'adamax'")
    
    

def compute_vae_loss(model, x_clean, x_input=None):
    if x_input is None:
        x_input = x_clean
    recon, mu, logvar, _ = model(x_input)
    recon_loss = nn.functional.mse_loss(recon, x_clean)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def compute_dae_loss(model, x_clean, noise_type, noise_factor):
    x_noisy = add_noise(x_clean, noise_type=noise_type, noise_factor=noise_factor)
    
    output = model(x_noisy)
    
    # Unpack if model returns a tuple (recon, latent, ...)
    recon = output[0] if isinstance(output, tuple) else output
    
    return nn.functional.mse_loss(recon, x_clean)


def parse_units(units_str):
    return [int(x.strip()) for x in units_str.split(",")]
import torch.nn as nn

def init_weights(module, method="kaiming", bias_init="zeros"):
    """
    Initialize weights of a module (used with model.apply()).
    """

    if isinstance(module, nn.Linear):

        if method == "xavier":
            nn.init.xavier_uniform_(module.weight)

        elif method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

        elif method == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif method == "orthogonal":
            nn.init.orthogonal_(module.weight)

        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)

        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

        else:
            raise ValueError(
                f"Unsupported initialization method: {method}"
            )

        # Bias init
        if module.bias is not None:
            if bias_init == "zeros":
                nn.init.zeros_(module.bias)
            elif bias_init == "normal":
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(module.bias)


def load_tensor_generic(path):
    """
    Generic loader that attempts multiple formats:
    1. CSV
    2. Parquet
    3. Torch tensor (.pt / .pth)
    4. Excel (.xlsx / .xls)

    Returns:
        torch.FloatTensor
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
        print(f"[Loader] Loaded as CSV: {path}")
        return torch.tensor(df.values, dtype=torch.float32)
    except Exception:
        pass

    try:
        df = pd.read_parquet(path)
        print(f"[Loader] Loaded as Parquet: {path}")
        return torch.tensor(df.values, dtype=torch.float32)
    except Exception:
        pass

    try:
        data = torch.load(path, map_location="cpu")

        if isinstance(data, torch.Tensor):
            print(f"[Loader] Loaded as Torch Tensor: {path}")
            return data.float()

        # If checkpoint dict
        if isinstance(data, dict):
            for key in ["data", "tensor", "features"]:
                if key in data and isinstance(data[key], torch.Tensor):
                    print(f"[Loader] Loaded Tensor from dict key '{key}'")
                    return data[key].float()

        raise ValueError("Torch file does not contain usable tensor.")

    except Exception:
        pass

    try:
        df = pd.read_excel(path)
        print(f"[Loader] Loaded as Excel: {path}")
        return torch.tensor(df.values, dtype=torch.float32)
    except Exception:
        pass
    raise ValueError(
        f"Unsupported file format or corrupted file: {path}\n"
        "Supported formats: CSV, Parquet, Torch (.pt/.pth), Excel"
    )
    

def get_loss_fn(name):
    return {
        "mseloss": nn.MSELoss(),
        "maeloss": nn.L1Loss(),
        "huberloss": nn.HuberLoss(),
    }[name]
def compute_rnn_loss(preds, y, loss_fn, learning_method="backprop"):
    """
    backprop        → preds is tensor (B, output_dim)       → standard loss
    cafo            → preds is List[(B, output_dim)]        → sum losses per layer
    forward_forward → preds is List[(B, T, H)] raw outputs  → goodness-based loss
    """
    if learning_method == "forward_forward":
        # Goodness = mean squared activation per layer, summed across layers
        # Minimize -goodness to maximize goodness on real data (FF objective)
        total_goodness = sum(out.pow(2).mean() for out in preds)
        return -total_goodness

    elif isinstance(preds, list):
        # CAFO: sum standard loss across all layer-wise predictions
        return sum(loss_fn(p, y) for p in preds)

    else:
        # Standard backprop: single tensor output
        return loss_fn(preds, y)