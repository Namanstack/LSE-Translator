"""
LSE Temporal Sequence Classifier

Processes temporal feature representations (e.g. from fused modalities) across
sequence frames (T=30) and classifies the sign language gesture into target vocabulary logits.

Structure:
    Fused Sequence (B, T, D_in)
          ↓
    Temporal LSTM
          ↓
    Final Representation
          ↓
    Fully Connected Classification Head
          ↓
    Class Logits (B, num_classes)
"""

import torch
import torch.nn as nn
from typing import Tuple


class TemporalClassifier(nn.Module):
    """
    Temporal Classifier using stacked LSTM and fully-connected head.

    Args:
        input_dim (int): Dimension of input features per frame (e.g., fusion_dim = 128).
        hidden_dim (int): Number of LSTM hidden units (default: 128).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        num_classes (int): Number of target classification categories (configurable).
        dropout (float): Dropout probability.
        bidirectional (bool): Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super(TemporalClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 1. Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 2. Classification Head
        fc_in = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal for recurrent weights and Xavier for linear weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Fused sequence tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            logits (torch.Tensor): Unnormalized class scores of shape (batch_size, num_classes).
            final_temporal_repr (torch.Tensor): Final time-step feature embedding (batch_size, hidden_dim).
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        if not self.bidirectional:
            final_temporal_repr = lstm_out[:, -1, :]  # (B, hidden_dim)
        else:
            final_temporal_repr = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)

        logits = self.classifier(final_temporal_repr)
        return logits, final_temporal_repr
