import torch
import torch.nn as nn
from nesy_factory.utils.helper import get_activation, build_mlp, init_weights


class MaskAwareAutoencoder(nn.Module):
    """
    Mask-Aware Autoencoder

    Encoder input = concat(x, mask)
    Input shape:
        x    -> (batch_size, feature_dim)
        mask -> (batch_size, feature_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        encoder_layers: list,
        decoder_layers: list,
        encoder_activation: str = "relu",
        decoder_activation: str = "relu",
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        encoder_batch_norm: bool = False,
        decoder_batch_norm: bool = False,
        weight_init: str = None,
        bias_init: str = "zeros",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder
        # Input dimension = 2 * feature_dim (x + mask)
        encoder_input_dim = 2 * feature_dim

        self.encoder_body, encoder_out_dim = build_mlp(
            input_dim=encoder_input_dim,
            layer_dims=encoder_layers,
            activation=encoder_activation,
            dropout=encoder_dropout,
            batch_norm=encoder_batch_norm,
        )

        self.latent_layer = nn.Linear(encoder_out_dim, latent_dim)

        # Decoder
        self.decoder_body, decoder_out_dim = build_mlp(
            input_dim=latent_dim,
            layer_dims=decoder_layers,
            activation=decoder_activation,
            dropout=decoder_dropout,
            batch_norm=decoder_batch_norm,
        )

        self.output_layer = nn.Linear(decoder_out_dim, feature_dim)

        # Weight Initialization (Optional)
        if weight_init is not None:
            self.apply(lambda m: init_weights(m, weight_init, bias_init))

    # Forward
    def forward(self, x, mask, return_latent=False):
        x_input = torch.cat([x, mask], dim=1)

        # Encoder
        h = self.encoder_body(x_input)
        z = self.latent_layer(h)

        # Decoder
        h_dec = self.decoder_body(z)
        recon = self.output_layer(h_dec)

        if return_latent:
            return recon, z
        return recon

    # Utilities
    def encode(self, x, mask):
        x_input = torch.cat([x, mask], dim=1)
        h = self.encoder_body(x_input)
        return self.latent_layer(h)

    def decode(self, z):
        h_dec = self.decoder_body(z)
        return self.output_layer(h_dec)