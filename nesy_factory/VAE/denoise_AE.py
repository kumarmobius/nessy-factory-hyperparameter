import torch
import torch.nn as nn
from nesy_factory.utils.helper import get_activation, add_noise, build_mlp,init_weights


class DenoisingAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        noise_type: str = "gaussian",
        noise_factor: float = 0.2,
        output_activation: str = None,
        weight_init: str = None,
    ):
        super().__init__()

        self.noise_type = noise_type
        self.noise_factor = noise_factor

        # ---------------- Encoder ----------------
        self.encoder_body, encoder_out_dim = build_mlp(
            input_dim=input_dim,
            layer_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.latent_layer = nn.Linear(encoder_out_dim, latent_dim)

        # ---------------- Decoder ----------------
        self.decoder_body, decoder_out_dim = build_mlp(
            input_dim=latent_dim,
            layer_dims=list(reversed(hidden_dims)),
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.output_layer = nn.Linear(decoder_out_dim, input_dim)

        # Optional output activation
        self.output_activation = (
            get_activation(output_activation)
            if output_activation
            else None
        )
        if weight_init:
            init_weights(self, weight_init)

    def forward(self, x):

        if self.training:
            x = add_noise(x, self.noise_type, self.noise_factor)

        h = self.encoder_body(x)
        z = self.latent_layer(h)

        h_dec = self.decoder_body(z)
        recon = self.output_layer(h_dec)

        if self.output_activation:
            recon = self.output_activation(recon)

        return recon, z