"""
LSE Multimodal Trainer
=======================
Core training engine for the end-to-end multimodal sign-language model.

Responsibilities
----------------
* Training loop   – one epoch of gradient updates on the training split.
* Validation loop – one epoch of evaluation on the validation split.
* NaN/Inf detection – immediately halts on corrupted tensors.
* Best-model selection – saves the checkpoint with the lowest val loss.
* Periodic checkpointing – saves every N epochs for resume capability.
* Early stopping – halts when val loss stagnates for `patience` epochs.
* Resume support – loads weights + optimizer + scheduler state from a .pt file.
* History logging – appends per-epoch metrics to results/metrics/training_history.json
  (only written during real training runs; never fabricated).

IMPORTANT
---------
This trainer NEVER generates or reports fake loss or accuracy values.
All numbers in training_history.json come exclusively from real forward passes
on real data.  Calling `Trainer.train()` without a valid dataset raises
`DatasetMissingError` before any computation begins.

Usage (once real data is available)
-------------------------------------
    from training.trainer import Trainer
    from training.train_config import TrainConfig

    cfg = TrainConfig(
        data_dir="data/processed",
        train_signers=["signer_01", "signer_02"],
        val_signers=["signer_03"],
        epochs=50,
    )
    trainer = Trainer(cfg)
    trainer.train()
"""

import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel
from training.multimodal_dataset import LSEMultimodalDataset
from training.signer_split import make_signer_independent_splits, summarise_splits
from training.train_config import TrainConfig

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lse.trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DatasetMissingError(RuntimeError):
    """Raised when no valid .npz recordings are found in data_dir."""


class InsufficientClassesError(ValueError):
    """Raised when fewer than 2 classes are present in the dataset."""


class InvalidTensorError(RuntimeError):
    """Raised when NaN or Inf values are detected in model inputs or outputs."""


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise InvalidTensorError if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        raise InvalidTensorError(
            f"NaN detected in '{name}'.  Check your data normalisation or "
            "learning rate – it may have diverged."
        )
    if torch.isinf(tensor).any():
        raise InvalidTensorError(
            f"Inf detected in '{name}'.  Check your data normalisation or "
            "learning rate – it may have diverged."
        )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Construct the LR scheduler specified by cfg.lr_scheduler."""
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
        )
    if cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
    return None  # "none"


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the complete multimodal training loop.

    Args:
        cfg (TrainConfig): All hyperparameters and path settings.

    The Trainer does not build datasets or the model until ``train()`` or
    ``validate()`` is explicitly called, keeping construction lightweight and
    testable.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.history: List[Dict] = []
        self._best_val_loss: float = math.inf
        self._epochs_no_improve: int = 0
        self._start_epoch: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the complete training loop.

        Raises:
            DatasetMissingError  – no .npz files found.
            InsufficientClassesError – fewer than 2 classes discovered.
            RuntimeError         – empty train or val split.
            InvalidTensorError   – NaN/Inf detected during training.
        """
        cfg = self.cfg
        set_seed(cfg.seed)
        device = torch.device(cfg.resolve_device())
        logger.info("Device: %s", device)

        # ── Validate dataset presence ────────────────────────────────────────
        self._assert_dataset_present(cfg.data_dir)

        # ── Build signer-independent splits ──────────────────────────────────
        train_ds, val_ds, test_ds, label_to_idx = make_signer_independent_splits(
            data_dir=cfg.data_dir,
            train_signers=cfg.train_signers,
            val_signers=cfg.val_signers,
            test_signers=cfg.test_signers if cfg.test_signers else None,
            require_face=cfg.require_face,
        )

        # ── Class count guard ────────────────────────────────────────────────
        num_classes = len(label_to_idx)
        if num_classes < 2:
            raise InsufficientClassesError(
                f"Only {num_classes} class(es) found.  Need ≥ 2 to train a classifier."
            )

        logger.info("\n%s", summarise_splits(train_ds, val_ds, test_ds))

        # ── DataLoaders ──────────────────────────────────────────────────────
        num_workers = 0  # 0 = main process (safe on Windows)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # ── Model ────────────────────────────────────────────────────────────
        model_cfg = ModelConfig(
            num_classes=num_classes,
            temporal_hidden_dim=cfg.hidden_size,
            temporal_num_layers=cfg.num_lstm_layers,
            temporal_dropout=cfg.dropout,
            encoder_dropout=cfg.dropout,
            fusion_dropout=cfg.dropout,
        )
        model = MultimodalSignModel(config=model_cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: %s", f"{total_params:,}")

        # ── Optimizer & scheduler ────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = _build_scheduler(optimizer, cfg)
        criterion = nn.CrossEntropyLoss()

        # ── Resume from checkpoint ───────────────────────────────────────────
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from, model, optimizer, scheduler)

        # ── Ensure output directories exist ──────────────────────────────────
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.metrics_dir, exist_ok=True)

        # ─────────────────────────────────────────────────────────────────────
        # Training loop
        # ─────────────────────────────────────────────────────────────────────
        logger.info("=" * 64)
        logger.info("Starting training – %d epochs max, patience=%d", cfg.epochs, cfg.patience)
        logger.info("=" * 64)

        for epoch in range(self._start_epoch, cfg.epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = self._validate_one_epoch(
                model, val_loader, criterion, device
            )

            if scheduler is not None:
                scheduler.step()

            # ── Per-epoch log ────────────────────────────────────────────────
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | lr=%.2e",
                epoch + 1, cfg.epochs, train_loss, val_loss, val_acc, current_lr,
            )

            # ── History ──────────────────────────────────────────────────────
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                }
            )
            self._save_history(cfg.metrics_dir)

            # ── Best-model checkpoint ────────────────────────────────────────
            improved = (self._best_val_loss - val_loss) > cfg.min_delta
            if improved:
                self._best_val_loss = val_loss
                self._epochs_no_improve = 0
                best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                self._save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch + 1, val_loss
                )
                logger.info("  ✓ Best model saved  (val_loss=%.4f)", val_loss)
            else:
                self._epochs_no_improve += 1

            # ── Periodic checkpoint ──────────────────────────────────────────
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt"
                )
                self._save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch + 1, val_loss
                )

            # ── Early stopping ───────────────────────────────────────────────
            if self._epochs_no_improve >= cfg.patience:
                logger.info(
                    "Early stopping triggered after %d epochs with no improvement.",
                    cfg.patience,
                )
                break

        logger.info("=" * 64)
        logger.info("Training complete.")
        logger.info("Best val_loss: %.4f", self._best_val_loss)
        logger.info("History saved to: %s", os.path.join(cfg.metrics_dir, "training_history.json"))

    # ─────────────────────────────────────────────────────────────────────────
    # Core loops (also usable independently in unit tests)
    # ─────────────────────────────────────────────────────────────────────────

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Run one training epoch.

        Returns:
            (mean_loss, accuracy)  – computed from real batch data only.
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (hand_batch, face_batch, label_batch, _) in enumerate(loader):
            hand_batch = hand_batch.to(device)
            face_batch = face_batch.to(device)
            label_batch = label_batch.to(device)

            # ── NaN / Inf guards on inputs ───────────────────────────────────
            _check_tensor(f"batch[{batch_idx}].hand", hand_batch)
            _check_tensor(f"batch[{batch_idx}].face", face_batch)

            optimizer.zero_grad()
            output = model(hand_batch, face_batch)
            logits = output.logits  # (B, num_classes)

            # ── NaN / Inf guards on outputs ──────────────────────────────────
            _check_tensor(f"batch[{batch_idx}].logits", logits)

            loss = criterion(logits, label_batch)
            _check_tensor(f"batch[{batch_idx}].loss", loss)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item() * hand_batch.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)

        mean_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return mean_loss, accuracy

    def _validate_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Run one validation epoch (no gradient updates).

        Returns:
            (mean_loss, accuracy)  – computed from real batch data only.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (hand_batch, face_batch, label_batch, _) in enumerate(loader):
                hand_batch = hand_batch.to(device)
                face_batch = face_batch.to(device)
                label_batch = label_batch.to(device)

                _check_tensor(f"val_batch[{batch_idx}].hand", hand_batch)
                _check_tensor(f"val_batch[{batch_idx}].face", face_batch)

                output = model(hand_batch, face_batch)
                logits = output.logits

                _check_tensor(f"val_batch[{batch_idx}].logits", logits)

                loss = criterion(logits, label_batch)

                running_loss += loss.item() * hand_batch.size(0)
                _, preds = torch.max(logits, dim=1)
                correct += (preds == label_batch).sum().item()
                total += label_batch.size(0)

        mean_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return mean_loss, accuracy

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save training state to a .pt file."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "history": self.history,
            "config": self.cfg.__dict__,
        }
        torch.save(state, path)

    def _load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> None:
        """Load training state from a .pt checkpoint file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {path}"
            )
        logger.info("Resuming from checkpoint: %s", path)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._start_epoch = ckpt.get("epoch", 0)
        self.history = ckpt.get("history", [])
        self._best_val_loss = min(
            (e["val_loss"] for e in self.history), default=math.inf
        )
        logger.info("Resumed from epoch %d (best val_loss so far: %.4f)",
                    self._start_epoch, self._best_val_loss)

    # ─────────────────────────────────────────────────────────────────────────
    # History / metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _save_history(self, metrics_dir: str) -> None:
        """Append current history to training_history.json (real data only)."""
        path = os.path.join(metrics_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Safeguard helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assert_dataset_present(data_dir: str) -> None:
        """
        Verify that at least one .npz file is reachable under data_dir.

        Raises:
            DatasetMissingError – with a clear, actionable message.
        """
        import glob
        pattern = os.path.join(data_dir, "**", "*.npz")
        found = glob.glob(pattern, recursive=True)
        if not found:
            raise DatasetMissingError(
                f"\n"
                f"  ╔══════════════════════════════════════════════════╗\n"
                f"  ║          DATASET NOT FOUND – CANNOT TRAIN        ║\n"
                f"  ╠══════════════════════════════════════════════════╣\n"
                f"  ║  No .npz files were found under:                 ║\n"
                f"  ║    {data_dir:<46}║\n"
                f"  ║                                                  ║\n"
                f"  ║  Each file must contain:                         ║\n"
                f"  ║    hand_features  shape (30, 126)  float32       ║\n"
                f"  ║    face_features  shape (30, 1404) float32       ║\n"
                f"  ║    label          str                            ║\n"
                f"  ║    signer         str                            ║\n"
                f"  ║    recording_id   str                            ║\n"
                f"  ║                                                  ║\n"
                f"  ║  Collect recordings first:                       ║\n"
                f"  ║    python preprocessing/collect_hand_landmarks.py║\n"
                f"  ║    python preprocessing/collect_face_landmarks.py║\n"
                f"  ╚══════════════════════════════════════════════════╝\n"
            )
