"""
LSE Central Configuration Module

Stores centralized architectural dimensions, sequence lengths, training defaults,
and path constants to ensure consistent interfaces across all modalities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    # -------------------------------------------------------------------------
    # Input Sequence & Landmark Dimensions
    # -------------------------------------------------------------------------
    sequence_length: int = 30           # Temporal frames per gesture sequence
    hand_feature_dim: int = 126         # 2 hands * 21 landmarks * 3 coords (x, y, z)
    face_feature_dim: int = 1404        # 468 landmarks * 3 coords (x, y, z)

    # -------------------------------------------------------------------------
    # Modality Encoder Embeddings
    # -------------------------------------------------------------------------
    hand_embedding_dim: int = 128       # Output embedding dim from HandFeatureEncoder
    face_embedding_dim: int = 128       # Output embedding dim from FaceFeatureEncoder
    encoder_dropout: float = 0.2

    # -------------------------------------------------------------------------
    # Adaptive Fusion Module
    # -------------------------------------------------------------------------
    fusion_dim: int = 128               # Shared projection dimension for fusion
    fusion_dropout: float = 0.2

    # -------------------------------------------------------------------------
    # Temporal Classifier (LSTM / GRU)
    # -------------------------------------------------------------------------
    temporal_hidden_dim: int = 128      # LSTM hidden units
    temporal_num_layers: int = 2        # Stacked LSTM layers
    temporal_dropout: float = 0.3
    bidirectional: bool = False

    # -------------------------------------------------------------------------
    # Classification & Vocabulary
    # -------------------------------------------------------------------------
    num_classes: int = 5                # Configurable vocabulary size (default 5 for testing)
    target_vocab: List[str] = field(default_factory=lambda: ["HOLA", "GRACIAS", "POR_FAVOR", "BUENOS_DIAS", "ADIOS"])

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42


# Default global instance
DEFAULT_CONFIG = ModelConfig()
