"""
LSE Multimodal Training Entry-Point
=====================================
Launch real model training from the command line.

Usage (once a real dataset is available)
-----------------------------------------
    python training/train.py \\
        --data_dir data/processed \\
        --train_signers signer_01 signer_02 \\
        --val_signers signer_03 \\
        --test_signers signer_04 \\
        --epochs 50 \\
        --batch_size 16 \\
        --lr 1e-3 \\
        --hidden_size 128 \\
        --num_lstm_layers 2 \\
        --dropout 0.3 \\
        --seed 42 \\
        --patience 10 \\
        --checkpoint_dir models/checkpoints \\
        --metrics_dir results/metrics

Resume from a checkpoint
--------------------------
    python training/train.py \\
        --data_dir data/processed \\
        --train_signers signer_01 signer_02 \\
        --val_signers signer_03 \\
        --resume_from models/checkpoints/checkpoint_epoch_0020.pt \\
        --epochs 100

IMPORTANT
---------
This script will EXIT WITH CODE 1 and print a clear error if:
    • No .npz dataset files are found in data_dir.
    • Fewer than 2 classes are present.
    • Any required split (train/val) is empty after signer filtering.
    • Overlapping signers between splits are detected (data leakage).

It will NEVER fabricate training metrics or claim training succeeded without
completing a genuine training run on real data.
"""

import argparse
import os
import sys

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.train_config import TrainConfig
from training.trainer import (
    Trainer,
    DatasetMissingError,
    InsufficientClassesError,
    InvalidTensorError,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "LSE Multimodal Sign-Language Training Pipeline.\n\n"
            "Trains the end-to-end MultimodalSignModel (hand + face → fusion → LSTM) "
            "using signer-independent dataset splits.  Requires real .npz recordings "
            "in --data_dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Root directory containing processed .npz recordings. (default: %(default)s)",
    )
    data_grp.add_argument(
        "--train_signers",
        nargs="+",
        required=True,
        metavar="SIGNER",
        help="Signer IDs for the training partition (space-separated).",
    )
    data_grp.add_argument(
        "--val_signers",
        nargs="+",
        required=True,
        metavar="SIGNER",
        help="Signer IDs for the validation partition (space-separated).",
    )
    data_grp.add_argument(
        "--test_signers",
        nargs="+",
        default=[],
        metavar="SIGNER",
        help="Optional signer IDs for a held-out test partition.",
    )
    data_grp.add_argument(
        "--no_face",
        action="store_true",
        help="Allow samples without face_features (face tensor set to zeros).",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument(
        "--epochs", type=int, default=50, help="Max training epochs. (default: %(default)s)"
    )
    train_grp.add_argument(
        "--batch_size", type=int, default=16, help="Batch size. (default: %(default)s)"
    )
    train_grp.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate. (default: %(default)s)"
    )
    train_grp.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. (default: %(default)s)",
    )
    train_grp.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience (epochs without improvement). (default: %(default)s)",
    )
    train_grp.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum val-loss improvement to reset patience counter. (default: %(default)s)",
    )

    # ── Architecture ──────────────────────────────────────────────────────────
    arch_grp = parser.add_argument_group("Architecture")
    arch_grp.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="LSTM hidden units. (default: %(default)s)",
    )
    arch_grp.add_argument(
        "--num_lstm_layers",
        type=int,
        default=2,
        help="Stacked LSTM layers. (default: %(default)s)",
    )
    arch_grp.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability for encoders, fusion, and classifier. (default: %(default)s)",
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    sched_grp = parser.add_argument_group("LR Scheduler")
    sched_grp.add_argument(
        "--lr_scheduler",
        choices=["cosine", "step", "none"],
        default="cosine",
        help="LR scheduler type. (default: %(default)s)",
    )
    sched_grp.add_argument(
        "--lr_step_size",
        type=int,
        default=10,
        help="StepLR step size (only used when --lr_scheduler=step). (default: %(default)s)",
    )
    sched_grp.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5,
        help="StepLR decay factor (only used when --lr_scheduler=step). (default: %(default)s)",
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_grp = parser.add_argument_group("Checkpointing")
    ckpt_grp.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join("models", "checkpoints"),
        help="Directory for saving checkpoints. (default: %(default)s)",
    )
    ckpt_grp.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs. (default: %(default)s)",
    )
    ckpt_grp.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_grp = parser.add_argument_group("Metrics")
    metrics_grp.add_argument(
        "--metrics_dir",
        type=str,
        default=os.path.join("results", "metrics"),
        help="Directory where training_history.json is saved. (default: %(default)s)",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    misc_grp = parser.add_argument_group("Miscellaneous")
    misc_grp.add_argument(
        "--seed", type=int, default=42, help="Random seed. (default: %(default)s)"
    )
    misc_grp.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: 'auto', 'cpu', or 'cuda'. (default: %(default)s)",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Build config from CLI args ────────────────────────────────────────────
    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_signers=args.train_signers,
        val_signers=args.val_signers,
        test_signers=args.test_signers,
        require_face=not args.no_face,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        resume_from=args.resume_from,
        metrics_dir=args.metrics_dir,
        seed=args.seed,
        device=args.device,
    )

    # ── Launch training ───────────────────────────────────────────────────────
    try:
        trainer = Trainer(cfg)
        trainer.train()
    except DatasetMissingError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except InsufficientClassesError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except InvalidTensorError as exc:
        print(f"[CRITICAL] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
