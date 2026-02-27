import torch
import torch.nn as nn
from nesy_factory.utils.helper import get_activation,build_mlp,init_weights



class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_layers,
        decoder_layers,
        encoder_activation="relu",
        decoder_activation="relu",
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        encoder_batch_norm=False,
        decoder_batch_norm=False,
        output_activation=None,
        weight_init: str = None,
        
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---------------- Encoder ----------------
        self.encoder_body, encoder_out_dim = build_mlp(
            input_dim=input_dim,
            layer_dims=encoder_layers,
            activation=encoder_activation,
            dropout=encoder_dropout,
            batch_norm=encoder_batch_norm,
        )

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        # ---------------- Decoder ----------------
        self.decoder_body, decoder_out_dim = build_mlp(
            input_dim=latent_dim,
            layer_dims=decoder_layers,
            activation=decoder_activation,
            dropout=decoder_dropout,
            batch_norm=decoder_batch_norm,
        )

        self.output_layer = nn.Linear(decoder_out_dim, input_dim)

        self.output_activation = (
            get_activation(output_activation)
            if output_activation
            else None
        )

        # 🔥 Apply weight initialization if provided
        if weight_init:
            init_weights(self, weight_init)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        h_dec = self.decoder_body(z)
        recon = self.output_layer(h_dec)

        return recon, mu, logvar, z

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def encode(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        h = self.decoder_body(z)
        return self.output_layer(h)

    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decode(z)