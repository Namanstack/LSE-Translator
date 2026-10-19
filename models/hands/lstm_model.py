"""
LSE Hands-Only LSTM Baseline Architecture

Processes spatio-temporal hand landmark sequences (30 frames x 126 features)
and predicts gesture class logits using a multi-layer Bidirectional/Unidirectional LSTM.

Input Shape:  (batch_size, sequence_length=30, input_size=126)
Output Shape: (batch_size, num_classes)
"""

import torch
import torch.nn as nn
from typing import Tuple


class HandsLSTMClassifier(nn.Module):
    """
    Hands-only LSTM Classifier for temporal sign language recognition.

    Attributes:
        input_size (int): Dimension of per-frame features (126 = 2 hands * 21 landmarks * 3 coords).
        hidden_size (int): Number of hidden units per LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        num_classes (int): Number of output vocabulary target classes.
        dropout (float): Dropout probability between LSTM layers and classification head.
        bidirectional (bool): Whether to use bidirectional LSTM (default: False for baseline).
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super(HandsLSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ---------------------------------------------------------------------
        # 1. Temporal Feature Extractor: Stacked LSTM
        # ---------------------------------------------------------------------
        # batch_first=True expects input shape: (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # ---------------------------------------------------------------------
        # 2. Classification Head
        # ---------------------------------------------------------------------
        fc_input_dim = hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        # Initialize network weights
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialization for linear layers and orthogonal for LSTM weights.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Initialize forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gesture classification.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, sequence_length, input_size)
                              e.g., (batch_size, 30, 126).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, num_classes).
        """
        # x: (batch_size, seq_len, input_size)
        # out: (batch_size, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        # c_n: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract the final temporal representation
        if not self.bidirectional:
            # For unidirectional LSTM, take the output at the last time step
            # lstm_out[:, -1, :] shape: (batch_size, hidden_size)
            final_repr = lstm_out[:, -1, :]
        else:
            # For bidirectional LSTM, concatenate the forward last step and backward first step
            # or concatenate last hidden states from the top layer
            forward_h = h_n[-2, :, :]
            backward_h = h_n[-1, :, :]
            final_repr = torch.cat([forward_h, backward_h], dim=1)

        # Pass through the classification head to obtain logits
        logits = self.classifier(final_repr)
        return logits


def get_model_summary(model: nn.Module) -> Tuple[str, int]:
    """
    Computes total trainable parameters and formatted architecture summary string.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_str = f"{model}\nTotal Trainable Parameters: {total_params:,}"
    return summary_str, total_params


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Architecture Test & Verification with Dummy Tensor
    # -------------------------------------------------------------------------
    batch_size = 2
    sequence_length = 30
    input_size = 126
    num_classes = 5

    print("=" * 70)
    print(" LSE HANDS-ONLY LSTM BASELINE ARCHITECTURE TEST")
    print("=" * 70)

    # Instantiate model
    model = HandsLSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    )

    summary_text, total_params = get_model_summary(model)
    print(f"Model Structure:\n{summary_text}\n")
    print("-" * 70)

    # Create dummy random input tensor: (2, 30, 126)
    dummy_input = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float32)
    print(f"Dummy Input Tensor Shape:  {dummy_input.shape} (batch_size={batch_size}, seq_len={sequence_length}, input_size={input_size})")

    # Run forward pass in eval mode
    model.eval()
    with torch.no_grad():
        output_logits = model(dummy_input)

    print(f"Output Logits Tensor Shape: {output_logits.shape} (batch_size={batch_size}, num_classes={num_classes})")
    print(f"Sample Output Logits (Sample 0):\n{output_logits[0].numpy()}")
    print("-" * 70)

    # Verification checks
    assert output_logits.shape == (batch_size, num_classes), f"Output shape mismatch: {output_logits.shape} vs ({batch_size}, {num_classes})"
    print("[PASSED] Architecture forward-pass test completed successfully.")
    print("=" * 70)
