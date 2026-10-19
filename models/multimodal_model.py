"""
LSE End-to-End Multimodal Hand + Face Architecture

Combines:
1. Hand Landmark Encoder (models.hands.hand_encoder.HandFeatureEncoder)
2. Face Landmark Encoder (models.face.face_encoder.FaceFeatureEncoder)
3. Adaptive Modality Fusion (models.fusion.adaptive_fusion.AdaptiveModalityFusion)
4. Temporal Sequence Classifier (models.temporal_classifier.TemporalClassifier)

Forward Interface:
    Input:
        hand_input: (batch_size, sequence_length=30, hand_feature_dim=126)
        face_input: (batch_size, sequence_length=30, face_feature_dim=1404)
    Output:
        MultimodalOutput dataclass containing:
        - logits: (batch_size, num_classes)
        - hand_embeddings: (batch_size, sequence_length, hand_emb_dim)
        - face_embeddings: (batch_size, sequence_length, face_emb_dim)
        - fused_features: (batch_size, sequence_length, fusion_dim)
        - fusion_weights: (batch_size, sequence_length, 2)
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig, DEFAULT_CONFIG
from models.hands.hand_encoder import HandFeatureEncoder
from models.face.face_encoder import FaceFeatureEncoder
from models.fusion.adaptive_fusion import AdaptiveModalityFusion
from models.temporal_classifier import TemporalClassifier


@dataclass
class MultimodalOutput:
    """Structured container for end-to-end multimodal predictions and intermediate representations."""
    logits: torch.Tensor                    # (B, num_classes)
    hand_embeddings: torch.Tensor          # (B, T, hand_emb_dim)
    face_embeddings: torch.Tensor          # (B, T, face_emb_dim)
    fused_features: torch.Tensor           # (B, T, fusion_dim)
    fusion_weights: torch.Tensor           # (B, T, 2) [w_hand, w_face]
    final_temporal_repr: torch.Tensor      # (B, hidden_dim)


class MultimodalSignModel(nn.Module):
    """
    End-to-End Multimodal Sign Language Recognition Model (Hands + Face).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super(MultimodalSignModel, self).__init__()
        self.config = config or DEFAULT_CONFIG

        # 1. Modality Encoders (Frame-wise)
        self.hand_encoder = HandFeatureEncoder(
            input_dim=self.config.hand_feature_dim,
            hidden_dim=self.config.hand_embedding_dim,
            output_dim=self.config.hand_embedding_dim,
            dropout=self.config.encoder_dropout
        )

        self.face_encoder = FaceFeatureEncoder(
            input_dim=self.config.face_feature_dim,
            hidden_dim=256,
            output_dim=self.config.face_embedding_dim,
            dropout=self.config.encoder_dropout
        )

        # 2. Adaptive Modality Fusion Layer
        self.fusion = AdaptiveModalityFusion(
            hand_dim=self.config.hand_embedding_dim,
            face_dim=self.config.face_embedding_dim,
            fusion_dim=self.config.fusion_dim,
            dropout=self.config.fusion_dropout
        )

        # 3. Temporal Classifier (LSTM + Classification Head)
        self.temporal_classifier = TemporalClassifier(
            input_dim=self.config.fusion_dim,
            hidden_dim=self.config.temporal_hidden_dim,
            num_layers=self.config.temporal_num_layers,
            num_classes=self.config.num_classes,
            dropout=self.config.temporal_dropout,
            bidirectional=self.config.bidirectional
        )

    def forward(
        self,
        hand_input: torch.Tensor,
        face_input: torch.Tensor
    ) -> MultimodalOutput:
        """
        Forward pass of the complete multimodal pipeline.

        Args:
            hand_input (torch.Tensor): Hand landmark sequence (B, T, 126).
            face_input (torch.Tensor): Face landmark sequence (B, T, 1404).

        Returns:
            MultimodalOutput: Container with logits and intermediate representations.
        """
        # 1. Encode modalities independently
        hand_emb = self.hand_encoder(hand_input)  # (B, T, 128)
        face_emb = self.face_encoder(face_input)  # (B, T, 128)

        # 2. Adaptively fuse embeddings
        fused_feat, fusion_weights = self.fusion(hand_emb, face_emb)  # (B, T, 128), (B, T, 2)

        # 3. Temporal sequence classification
        logits, final_repr = self.temporal_classifier(fused_feat)  # (B, num_classes), (B, 128)

        return MultimodalOutput(
            logits=logits,
            hand_embeddings=hand_emb,
            face_embeddings=face_emb,
            fused_features=fused_feat,
            fusion_weights=fusion_weights,
            final_temporal_repr=final_repr
        )


def get_multimodal_model_summary(model: nn.Module) -> Tuple[str, int]:
    """Computes total trainable parameter count and architecture summary."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_str = f"{model}\nTotal Trainable Parameters: {total_params:,}"
    return summary_str, total_params
