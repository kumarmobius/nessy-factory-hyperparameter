import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from sklearn.metrics import precision_recall_fscore_support
from nesy_factory.utils.helper import get_optimizer, init_weights


class BaseRNN(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        loss_function: str = "mseloss",
        learning_method: str = "backprop",
        grad_clip: float = None,
        weight_init: str = None,
        device: str = None,
    ):
        super().__init__()

        # -------------------------
        # Core config
        # -------------------------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.loss_function_type = loss_function.lower()
        self.learning_method = learning_method.lower()
        self.grad_clip = grad_clip

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.optimizer = None
        self.criterion = None

        # Weight initialization if needed
        if weight_init:
            init_weights(self, weight_init)

        self.to(self.device)

    # -------------------------------------------------
    # Optimizer + Loss Setup
    # -------------------------------------------------

    def _init_optimizer_and_criterion(self):

        if self.optimizer is None:
            self.optimizer = get_optimizer(
                self.optimizer_name,
                self,
                self.learning_rate,
                self.weight_decay,
                momentum=self.momentum,
            )

        if self.criterion is None:

            if self.loss_function_type == "mseloss":
                self.criterion = nn.MSELoss()

            elif self.loss_function_type == "l1loss":
                self.criterion = nn.L1Loss()

            elif self.loss_function_type == "bcewithlogitsloss":
                self.criterion = nn.BCEWithLogitsLoss()

            elif self.loss_function_type == "crossentropyloss":
                self.criterion = nn.CrossEntropyLoss()

            else:
                raise ValueError(f"Unsupported loss: {self.loss_function_type}")

    # -------------------------------------------------
    # Abstract Forward
    # -------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    # -------------------------------------------------
    # Training Step
    # -------------------------------------------------

    def train_step(self, batch):

        self._init_optimizer_and_criterion()
        self.train()

        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    def eval_step(self, data_loader):

        self.eval()
        self._init_optimizer_and_criterion()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                if self.loss_function_type == "bcewithlogitsloss":
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    preds = outputs

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        if self.loss_function_type in ["bcewithlogitsloss"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            accuracy = (all_preds == all_labels).mean()

            return {
                "loss": float(avg_loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }

        else:
            mse = ((all_preds - all_labels) ** 2).mean()
            mae = np.abs(all_preds - all_labels).mean()

            return {
                "loss": float(avg_loss),
                "mse": float(mse),
                "mae": float(mae),
            }