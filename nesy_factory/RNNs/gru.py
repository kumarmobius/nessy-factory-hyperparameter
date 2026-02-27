# import torch
# import torch.nn as nn
# from typing import List, Optional
# from .base import BaseRNN
# from nesy_factory.utils.helper import init_weights


# class GRU(BaseRNN):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dims: List[int],
#         output_dim: int,
#         dropout: float = 0.0,
#         bidirectional: bool = False,
#         pooling: str = "last",
#         learn_init_hidden: bool = False,
#         nonlinearity: str = "tanh",
#         weight_init: Optional[str] = None,
#         **config
#     ):
#         config.update({
#             "input_dim": input_dim,
#             "hidden_dim": hidden_dims[-1],
#             "output_dim": output_dim,
#             "num_layers": len(hidden_dims),
#         })

#         super().__init__(config)

#         self.hidden_dims = hidden_dims
#         self.bidirectional = bidirectional
#         self.pooling = pooling
#         self.learn_init_hidden = learn_init_hidden
#         self.nonlinearity = nonlinearity

#         self.layers = nn.ModuleList()
#         self.dropouts = nn.ModuleList()

#         for i in range(self.num_layers):
#             in_features = (
#                 input_dim if i == 0
#                 else hidden_dims[i - 1] * (2 if bidirectional else 1)
#             )

#             self.layers.append(
#                 nn.GRU(
#                     input_size=in_features,
#                     hidden_size=hidden_dims[i],
#                     num_layers=1,
#                     batch_first=True,
#                     bidirectional=bidirectional,
#                 )
#             )

#             self.dropouts.append(
#                 nn.Dropout(dropout if i < self.num_layers - 1 else 0.0)
#             )

#         final_dim = hidden_dims[-1] * (2 if bidirectional else 1)
#         self.fc = nn.Linear(final_dim, output_dim)

#         # CAFO predictors
#         self.cafo_predictors = nn.ModuleList([
#             nn.Linear(
#                 hidden_dims[i] * (2 if bidirectional else 1),
#                 output_dim
#             )
#             for i in range(self.num_layers)
#         ])

#         if weight_init:
#             init_weights(self, weight_init)

#         self._init_optimizer_and_criterion()

#     # -------------------------------------------------
#     # CAFO Parameter Extraction
#     # -------------------------------------------------
#     def get_layer_parameters(self, i: int, num_outputs: int):

#         if i < self.num_layers:
#             return (
#                 list(self.layers[i].parameters())
#                 + list(self.cafo_predictors[i].parameters())
#             )
#         elif i == self.num_layers:
#             return list(self.fc.parameters())
#         else:
#             raise IndexError("Invalid layer index")

#     # -------------------------------------------------
#     # Forward
#     # -------------------------------------------------
#     def forward(self, x, ff_pass=False):

#         current_input = x
#         layer_outputs = []

#         for i, layer in enumerate(self.layers):

#             directions = 2 if self.bidirectional else 1
#             h_init = torch.zeros(
#                 directions,
#                 x.size(0),
#                 self.hidden_dims[i],
#                 device=x.device,
#             )

#             out, _ = layer(current_input, h_init)
#             out = self.dropouts[i](out)

#             layer_outputs.append(out)
#             current_input = out

#         # ---------- Forward-Forward ----------
#         if ff_pass:
#             return layer_outputs

#         # ---------- CAFO ----------
#         if self.learning_method == "cafo" and self.training:

#             preds = []
#             for i, out in enumerate(layer_outputs):
#                 pooled = self._pool(out)
#                 preds.append(self.cafo_predictors[i](pooled))

#             final_pred = self.fc(self._pool(layer_outputs[-1]))
#             preds.append(final_pred)

#             return preds

#         # ---------- Standard Backprop ----------
#         pooled = self._pool(layer_outputs[-1])
#         return self.fc(pooled)

#     # -------------------------------------------------
#     def _pool(self, out):
#         if self.pooling == "last":
#             return out[:, -1, :]
#         elif self.pooling == "mean":
#             return out.mean(dim=1)
#         elif self.pooling == "max":
#             return out.max(dim=1)[0]
#         else:
#             return out[:, -1, :]


import torch
import torch.nn as nn
from typing import List, Optional
from nesy_factory.RNNs.base import BaseRNN
from nesy_factory.utils.helper import init_weights


class GRU(BaseRNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: str = "last",
        learn_init_hidden: bool = False,
        weight_init: Optional[str] = None,
        # BaseRNN-level params
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        loss_function: str = "mseloss",
        learning_method: str = "backprop",
        grad_clip: float = None,
        device: str = None,
    ):
        if isinstance(hidden_dims, str):
            hidden_dims = [int(x) for x in hidden_dims.split(",")]

        # ── Call BaseRNN with individual named args (NOT a dict) ──
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            num_layers=len(hidden_dims),
            dropout=dropout,
            bidirectional=bidirectional,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            loss_function=loss_function,
            learning_method=learning_method,
            grad_clip=grad_clip,
            weight_init=None,           # handled below after layers are built
            device=device,
        )

        # ── GRU-specific attributes ──
        self.hidden_dims = hidden_dims
        self.pooling = pooling
        self.learn_init_hidden = learn_init_hidden
        self.weight_init = weight_init

        # ── GRU layers ──
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            in_features = (
                input_dim if i == 0
                else hidden_dims[i - 1] * (2 if bidirectional else 1)
            )
            self.layers.append(
                nn.GRU(
                    input_size=in_features,
                    hidden_size=hidden_dims[i],
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            # no dropout after final layer
            self.dropouts.append(
                nn.Dropout(dropout if i < self.num_layers - 1 else 0.0)
            )

        final_dim = hidden_dims[-1] * (2 if bidirectional else 1)

        # ── Output head ──
        self.fc = nn.Linear(final_dim, output_dim)

        # ── CAFO: one predictor per layer ──
        self.cafo_predictors = nn.ModuleList([
            nn.Linear(hidden_dims[i] * (2 if bidirectional else 1), output_dim)
            for i in range(self.num_layers)
        ])

        # ── Learnable initial hidden state ──
        if learn_init_hidden:
            directions = 2 if bidirectional else 1
            self.h0 = nn.ParameterList([
                nn.Parameter(torch.zeros(directions, 1, h))
                for h in hidden_dims
            ])
        else:
            self.h0 = None

        # ── Weight initialization (after all layers are built) ──
        if weight_init:
            init_weights(self, weight_init)

    # -------------------------------------------------
    # Internal: run all GRU layers, return per-layer outputs
    # -------------------------------------------------
    def _forward_layers(self, x: torch.Tensor) -> List[torch.Tensor]:

        batch_size = x.size(0)
        current_input = x
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            directions = 2 if self.bidirectional else 1

            if self.learn_init_hidden and self.h0 is not None:
                h_init = self.h0[i].repeat(1, batch_size, 1)
            else:
                h_init = torch.zeros(
                    directions, batch_size, self.hidden_dims[i],
                    device=x.device,
                )

            out, _ = layer(current_input, h_init)
            out = self.dropouts[i](out)
            layer_outputs.append(out)
            current_input = out

        return layer_outputs

    # -------------------------------------------------
    # Pooling
    def _pool(self, out: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return out[:, -1, :]
        elif self.pooling == "mean":
            return out.mean(dim=1)
        elif self.pooling == "max":
            return out.max(dim=1)[0]
        else:
            return out[:, -1, :]

    #   learning_method = "backprop"        → returns tensor  (B, output_dim)
    #   learning_method = "cafo"            → returns List[(B, output_dim)]  per-layer + final
    #   learning_method = "forward_forward" → returns List[(B, T, H)]  raw layer outputs

    def forward(self, x: torch.Tensor):

        layer_outputs = self._forward_layers(x)

        if self.learning_method == "forward_forward":
            return layer_outputs

        if self.learning_method == "cafo" and self.training:
            preds = []
            for i, out in enumerate(layer_outputs):
                pooled = self._pool(out)
                preds.append(self.cafo_predictors[i](pooled))
            preds.append(self.fc(self._pool(layer_outputs[-1])))
            return preds

        pooled = self._pool(layer_outputs[-1])
        return self.fc(pooled)

    def get_layer_parameters(self, i: int):
        if i < self.num_layers:
            return (
                list(self.layers[i].parameters())
                + list(self.cafo_predictors[i].parameters())
            )
        elif i == self.num_layers:
            return list(self.fc.parameters())
        else:
            raise IndexError(f"Invalid layer index {i}, model has {self.num_layers} layers")


    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        layer_outputs = self._forward_layers(x)
        return self._pool(layer_outputs[-1])