"""
LSE Facial Landmark Feature Encoder - Modality 2

Lightweight frame-wise MLP encoder that projects dense facial landmark vectors
(1404 features per frame) into a compact latent embedding space (128 features per frame)
while preserving the temporal sequence dimension.

Input Shape:  (batch_size, sequence_length=30, input_dim=1404)
Output Shape: (batch_size, sequence_length=30, output_dim=128)
"""

import torch
import torch.nn as nn
from typing import Tuple


class FaceFeatureEncoder(nn.Module):
    """
    Frame-wise MLP Encoder for facial landmarks.

    Applies linear projection, activation, and regularization independently to each
    temporal frame (time-distributed manner) to produce compact frame embeddings.

    Architecture:
        Linear(input_dim -> hidden_dim)
        ReLU()
        Dropout(p=dropout)
        Linear(hidden_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim: int = 1404,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super(FaceFeatureEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Frame-wise projection network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming/Xavier initialization for linear layers.
        """
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes sequence of facial landmark frames.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, sequence_length, input_dim)
                              e.g., (batch_size, 30, 1404).

        Returns:
            torch.Tensor: Encoded frame embeddings of shape (batch_size, sequence_length, output_dim)
                          e.g., (batch_size, 30, 128).
        """
        # PyTorch Linear layers automatically operate on the last dimension,
        # preserving batch_size and sequence_length seamlessly:
        # (B, T, D_in) -> (B, T, D_out)
        encoded_seq = self.encoder(x)
        return encoded_seq


def get_encoder_summary(model: nn.Module) -> Tuple[str, int]:
    """
    Computes total trainable parameters and formatted architecture summary string.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_str = f"{model}\nTotal Trainable Parameters: {total_params:,}"
    return summary_str, total_params


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Face Feature Encoder Architecture Verification Test
    # -------------------------------------------------------------------------
    batch_size = 2
    sequence_length = 30
    input_dim = 1404
    hidden_dim = 256
    output_dim = 128
    dropout = 0.2

    print("=" * 70)
    print(" LSE FACE FEATURE ENCODER - ARCHITECTURE TEST")
    print("=" * 70)

    # 1. Instantiate encoder
    encoder = FaceFeatureEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout
    )

    summary_text, total_params = get_encoder_summary(encoder)
    print(f"Model Structure:\n{summary_text}\n")
    print("-" * 70)

    # 2. Create dummy input tensor: (2, 30, 1404)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim, dtype=torch.float32)
    print(f"Dummy Input Tensor Shape:  {dummy_input.shape} (batch_size={batch_size}, seq_len={sequence_length}, input_dim={input_dim})")

    # 3. Perform forward pass
    encoder.eval()
    with torch.no_grad():
        output_embeddings = encoder(dummy_input)

    print(f"Output Embeddings Shape:   {output_embeddings.shape} (batch_size={batch_size}, seq_len={sequence_length}, output_dim={output_dim})")
    print("-" * 70)

    # 4. Shape validation check
    expected_output_shape = (batch_size, sequence_length, output_dim)
    assert output_embeddings.shape == expected_output_shape, (
        f"Shape mismatch: expected {expected_output_shape}, received {output_embeddings.shape}"
    )

    print("[PASSED] Face feature encoder forward-pass verification completed successfully.")
    print("=" * 70)
