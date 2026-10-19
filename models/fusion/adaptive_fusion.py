"""
LSE Adaptive Multimodal Fusion Module

Combines Hand and Face feature embeddings across temporal frames via a learnable
modality-weighting mechanism.

Concept:
    hand_embeddings (B, T, D_h) -> Projection (B, T, D_fused) ─┐
                                                                 ├→ Modality Gating / Softmax Weights → Fused (B, T, D_fused)
    face_embeddings (B, T, D_f) -> Projection (B, T, D_fused) ─┘

Formula:
    w = Softmax([Score_hand, Score_face])
    Fused = w_hand * Hand_proj + w_face * Face_proj

Note:
    This is an implementation candidate for empirical testing. Research and patent
    prior-art analysis will be conducted later to assess novelty and baseline comparisons.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AdaptiveModalityFusion(nn.Module):
    """
    Adaptive Modality Fusion with learnable frame-level or sequence-level attention gating.

    Args:
        hand_dim (int): Input feature dimension of hand embeddings (default: 128).
        face_dim (int): Input feature dimension of face embeddings (default: 128).
        fusion_dim (int): Target fused embedding dimension (default: 128).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        hand_dim: int = 128,
        face_dim: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.2
    ):
        super(AdaptiveModalityFusion, self).__init__()

        self.hand_dim = hand_dim
        self.face_dim = face_dim
        self.fusion_dim = fusion_dim

        # 1. Modality-specific linear projections to align embedding dimensions
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.face_proj = nn.Sequential(
            nn.Linear(face_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 2. Modality Gating Network (estimates confidence/weight per modality)
        # Produces 2 scalar scores [score_hand, score_face] per frame
        self.gating_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2)
        )

        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in [self.hand_proj, self.face_proj, self.gating_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def forward(
        self,
        hand_emb: torch.Tensor,
        face_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multimodal adaptive fusion.

        Args:
            hand_emb (torch.Tensor): Hand features (batch_size, seq_len, hand_dim).
            face_emb (torch.Tensor): Face features (batch_size, seq_len, face_dim).

        Returns:
            fused_features (torch.Tensor): Fused sequence representation (batch_size, seq_len, fusion_dim).
            weights (torch.Tensor): Normalized modality weights (batch_size, seq_len, 2)
                                    where weights[..., 0] = w_hand, weights[..., 1] = w_face.
        """
        # Project each modality to common fusion dimension: (B, T, fusion_dim)
        h_proj = self.hand_proj(hand_emb)
        f_proj = self.face_proj(face_emb)

        # Concatenate projected features for joint gating decision: (B, T, fusion_dim * 2)
        joint_feat = torch.cat([h_proj, f_proj], dim=-1)

        # Compute unnormalized gating logits: (B, T, 2)
        gate_logits = self.gating_network(joint_feat)

        # Softmax normalization ensures w_hand + w_face = 1.0 per frame: (B, T, 2)
        weights = self.softmax(gate_logits)

        # Extract weights
        w_hand = weights[..., 0:1]  # (B, T, 1)
        w_face = weights[..., 1:2]  # (B, T, 1)

        # Weighted adaptive combination: (B, T, fusion_dim)
        fused_features = w_hand * h_proj + w_face * f_proj

        return fused_features, weights


if __name__ == "__main__":
    # Test adaptive fusion module
    B, T, D_h, D_f, D_fuse = 2, 30, 128, 128, 128
    hand_dummy = torch.randn(B, T, D_h)
    face_dummy = torch.randn(B, T, D_f)

    fusion_layer = AdaptiveModalityFusion(hand_dim=D_h, face_dim=D_f, fusion_dim=D_fuse)
    fused, weights = fusion_layer(hand_dummy, face_dummy)

    print("Fusion output shape:", fused.shape)       # Expected: (2, 30, 128)
    print("Fusion weights shape:", weights.shape)   # Expected: (2, 30, 2)
    print("Sample weights frame 0:", weights[0, 0].detach().numpy())
    assert fused.shape == (B, T, D_fuse)
    assert weights.shape == (B, T, 2)
    print("[PASSED] AdaptiveModalityFusion test.")
