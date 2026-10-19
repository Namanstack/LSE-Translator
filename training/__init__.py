"""
LSE Training Package

Contains the complete multimodal training infrastructure for the
LSE Sign Language Recognition system.

Modules:
    multimodal_dataset  – PyTorch Dataset for paired hand + face .npz recordings.
    signer_split        – Signer-independent train/val/test split builder.
    train_config        – TrainConfig dataclass with all hyperparameters.
    trainer             – Core Trainer class (training loop, early stopping, checkpointing).
    train               – CLI entry-point for launching real training runs.

NOTE: Real training requires multimodal .npz files in data/processed/ containing
both 'hand_features' and 'face_features' keys.  This package will refuse to
train on fabricated data and will never report fake metrics.
"""
