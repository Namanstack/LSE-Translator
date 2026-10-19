"""
LSE Training Configuration
===========================
Single source-of-truth for all hyperparameters and path constants used by the
multimodal training pipeline.

Usage
-----
    from training.train_config import TrainConfig, DEFAULT_TRAIN_CONFIG

    cfg = TrainConfig(lr=5e-4, epochs=50)
    # or from CLI args:
    cfg = TrainConfig.from_args(parsed_args)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """
    Hyperparameters and path settings for the multimodal training pipeline.

    All fields have sensible defaults; override as needed for each experiment.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = os.path.join("data", "processed")
    """Root directory searched recursively for .npz multimodal recordings."""

    train_signers: List[str] = field(default_factory=list)
    """Signer IDs exclusively assigned to the training split."""

    val_signers: List[str] = field(default_factory=list)
    """Signer IDs exclusively assigned to the validation split."""

    test_signers: List[str] = field(default_factory=list)
    """Signer IDs exclusively assigned to the held-out test split (optional)."""

    require_face: bool = True
    """If True, samples without face_features are skipped."""

    # ── Optimisation ─────────────────────────────────────────────────────────
    batch_size: int = 16
    """Number of sequences per gradient update."""

    lr: float = 1e-3
    """Initial learning rate for AdamW."""

    weight_decay: float = 1e-4
    """L2 regularisation coefficient for AdamW."""

    epochs: int = 50
    """Maximum number of training epochs."""

    # ── Architecture overrides (passed to MultimodalSignModel via ModelConfig) ─
    hidden_size: int = 128
    """LSTM hidden units (temporal_hidden_dim in ModelConfig)."""

    num_lstm_layers: int = 2
    """Number of stacked LSTM layers (temporal_num_layers in ModelConfig)."""

    dropout: float = 0.3
    """Dropout probability applied in encoders, fusion, and classifier."""

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    lr_scheduler: str = "cosine"
    """Learning rate schedule: 'cosine' | 'step' | 'none'."""

    lr_step_size: int = 10
    """Step size for StepLR scheduler (only used when lr_scheduler='step')."""

    lr_gamma: float = 0.5
    """Decay factor for StepLR scheduler (only used when lr_scheduler='step')."""

    # ── Early Stopping ────────────────────────────────────────────────────────
    patience: int = 10
    """Number of epochs with no val-loss improvement before early stopping."""

    min_delta: float = 1e-4
    """Minimum improvement in val loss to reset the patience counter."""

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = os.path.join("models", "checkpoints")
    """Directory where per-epoch checkpoints and the best model are saved."""

    save_every_n_epochs: int = 5
    """Save a checkpoint every N epochs (in addition to the best model)."""

    resume_from: Optional[str] = None
    """Path to a .pt checkpoint file to resume training from."""

    # ── Metrics / History ────────────────────────────────────────────────────
    metrics_dir: str = os.path.join("results", "metrics")
    """Directory where training_history.json is written after real training."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for Python, NumPy, and PyTorch."""

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"
    """Training device: 'auto' (use CUDA if available, else CPU), 'cpu', 'cuda'."""

    # ─────────────────────────────────────────────────────────────────────────
    # Factories
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        """
        Construct a TrainConfig from a parsed argparse.Namespace object.

        Only attributes that actually exist on *args* are applied; missing
        attributes fall back to the dataclass defaults.
        """
        cfg = cls()
        for field_name in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
            if hasattr(args, field_name):
                setattr(cfg, field_name, getattr(args, field_name))
        return cfg

    def resolve_device(self) -> str:
        """Returns 'cuda' or 'cpu' after resolving 'auto'."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def __post_init__(self) -> None:
        """Validate critical numeric constraints."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be ≥ 1, got {self.epochs}")
        if self.patience < 1:
            raise ValueError(f"patience must be ≥ 1, got {self.patience}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.lr_scheduler not in {"cosine", "step", "none"}:
            raise ValueError(
                f"lr_scheduler must be 'cosine', 'step', or 'none', "
                f"got '{self.lr_scheduler}'"
            )


# Default configuration instance
DEFAULT_TRAIN_CONFIG = TrainConfig()
