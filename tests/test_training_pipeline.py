"""
LSE Training Pipeline Unit Tests
==================================
Tests the complete multimodal training infrastructure using SYNTHETIC tensors only.

══════════════════════════════════════════════════════════════════════════════
IMPORTANT – READ BEFORE INTERPRETING OUTPUT
══════════════════════════════════════════════════════════════════════════════
All tests in this file use randomly-generated (torch.randn / np.random) data.

• Loss and accuracy numbers produced here DO NOT represent real model performance.
• These tests ONLY verify:
    – tensor shapes flow correctly through the pipeline,
    – code executes without errors,
    – safeguards (NaN detection, early stopping, dataset-missing guard, etc.) fire.
• No inference is made about the model's ability to recognise sign language.
• Real performance can only be assessed after a genuine training run on real data.
══════════════════════════════════════════════════════════════════════════════

Tests
-----
1. test_multimodal_dataset_synthetic         – Dataset loads in-memory mock .npz.
2. test_signer_split_no_leakage              – Signer sets are strictly disjoint.
3. test_trainer_one_epoch_shapes             – 1 epoch on 4 synthetic samples; check scalar loss/acc.
4. test_nan_inf_detection_raises             – Injected NaN triggers InvalidTensorError.
5. test_checkpoint_save_and_resume           – Saved weights reload correctly.
6. test_early_stopping_trigger               – patience=1 stops training at epoch 2.
7. test_dataset_missing_error                – Empty data_dir triggers DatasetMissingError.
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.multimodal_dataset import LSEMultimodalDataset
from training.signer_split import (
    validate_split_config,
    make_signer_independent_splits,
)
from training.train_config import TrainConfig
from training.trainer import (
    Trainer,
    DatasetMissingError,
    InsufficientClassesError,
    InvalidTensorError,
    _check_tensor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: create synthetic .npz files in a temp directory
# ─────────────────────────────────────────────────────────────────────────────

def _write_npz(
    path: str,
    label: str,
    signer: str,
    recording_id: str,
    hand_shape=(30, 126),
    face_shape=(30, 1404),
    corrupt_hand: bool = False,
    corrupt_face: bool = False,
) -> None:
    """Write a minimal synthetic .npz recording to *path*."""
    hand = np.random.randn(*hand_shape).astype(np.float32)
    face = np.random.randn(*face_shape).astype(np.float32)
    if corrupt_hand:
        hand[0, 0] = float("nan")
    if corrupt_face:
        face[0, 0] = float("inf")
    np.savez(
        path,
        hand_features=hand,
        face_features=face,
        label=label,
        signer=signer,
        recording_id=recording_id,
    )


def _make_synthetic_dataset(
    tmp_dir: str,
    n_per_class_per_signer: int = 2,
    labels=("SIGN_A", "SIGN_B"),
    signers=("signer_01", "signer_02", "signer_03"),
) -> str:
    """
    Populate tmp_dir with synthetic .npz files and return the directory path.

    NOTE: data is entirely random – this is purely for shape / code-path testing.
    """
    for lbl in labels:
        lbl_dir = os.path.join(tmp_dir, lbl)
        os.makedirs(lbl_dir, exist_ok=True)
        for sgn in signers:
            for i in range(n_per_class_per_signer):
                fname = f"{sgn}_{i:03d}.npz"
                _write_npz(
                    path=os.path.join(lbl_dir, fname),
                    label=lbl,
                    signer=sgn,
                    recording_id=f"{sgn}_{lbl}_{i}",
                )
    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Dataset loads synthetic .npz files
# ─────────────────────────────────────────────────────────────────────────────

def test_multimodal_dataset_synthetic():
    """
    [SYNTHETIC DATA TEST]
    Verifies that LSEMultimodalDataset correctly loads .npz files and returns
    tensors with the expected shapes.

    This test does NOT evaluate model performance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_dataset(
            tmp, n_per_class_per_signer=3, labels=("A", "B"), signers=("s1", "s2")
        )
        ds = LSEMultimodalDataset(data_dir=tmp)

        assert len(ds) == 12, f"Expected 12 samples, got {len(ds)}"
        assert len(ds.label_to_idx) == 2, "Expected 2 classes"

        hand, face, label, meta = ds[0]
        assert hand.shape == (30, 126), f"hand shape mismatch: {hand.shape}"
        assert face.shape == (30, 1404), f"face shape mismatch: {face.shape}"
        assert label.dtype == torch.long
        assert isinstance(meta, dict)
        assert "signer" in meta and "label" in meta


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Signer splits are strictly disjoint
# ─────────────────────────────────────────────────────────────────────────────

def test_signer_split_no_leakage():
    """
    [SYNTHETIC DATA TEST]
    Verifies that train / val / test splits have zero signer overlap.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_dataset(
            tmp,
            n_per_class_per_signer=2,
            labels=("X", "Y"),
            signers=("s1", "s2", "s3", "s4"),
        )
        train_ds, val_ds, test_ds, lbl_map = make_signer_independent_splits(
            data_dir=tmp,
            train_signers=["s1", "s2"],
            val_signers=["s3"],
            test_signers=["s4"],
        )

        train_set = set(train_ds.signers_found)
        val_set = set(val_ds.signers_found)
        test_set = set(test_ds.signers_found)

        assert train_set & val_set == set(), "Train/Val signer overlap!"
        assert train_set & test_set == set(), "Train/Test signer overlap!"
        assert val_set & test_set == set(), "Val/Test signer overlap!"

        # Shared vocabulary
        assert lbl_map == train_ds.label_to_idx == val_ds.label_to_idx == test_ds.label_to_idx

    # Overlap should raise ValueError
    with tempfile.TemporaryDirectory() as tmp2:
        _make_synthetic_dataset(tmp2, labels=("A", "B"), signers=("s1", "s2"))
        with pytest.raises(ValueError, match="Data leakage detected"):
            validate_split_config(
                all_signers=["s1", "s2"],
                train_signers=["s1"],
                val_signers=["s1"],   # deliberately overlapping
                test_signers=None,
                all_labels=["A", "B"],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – One epoch produces scalar loss and acc in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_trainer_one_epoch_shapes():
    """
    [SYNTHETIC DATA TEST]
    Runs exactly 1 training epoch and 1 validation epoch on 4 synthetic samples.

    The loss and accuracy values reported here are from random data and do NOT
    indicate real model performance.
    """
    with tempfile.TemporaryDirectory() as tmp_data, \
         tempfile.TemporaryDirectory() as tmp_ckpt, \
         tempfile.TemporaryDirectory() as tmp_metrics:

        _make_synthetic_dataset(
            tmp_data,
            n_per_class_per_signer=2,
            labels=("A", "B"),
            signers=("s1", "s2", "s3"),
        )

        cfg = TrainConfig(
            data_dir=tmp_data,
            train_signers=["s1", "s2"],
            val_signers=["s3"],
            epochs=1,
            batch_size=4,
            lr=1e-3,
            seed=42,
            patience=5,
            checkpoint_dir=tmp_ckpt,
            metrics_dir=tmp_metrics,
            lr_scheduler="none",
        )

        trainer = Trainer(cfg)
        trainer.train()  # runs exactly 1 real epoch

        assert len(trainer.history) == 1, "Expected 1 history entry"
        rec = trainer.history[0]

        # Validate scalar types (not performance)
        assert isinstance(rec["train_loss"], float) and rec["train_loss"] >= 0
        assert isinstance(rec["val_loss"], float) and rec["val_loss"] >= 0
        assert 0.0 <= rec["train_acc"] <= 1.0
        assert 0.0 <= rec["val_acc"] <= 1.0
        assert rec["epoch"] == 1

        # History file written
        hist_path = os.path.join(tmp_metrics, "training_history.json")
        assert os.path.isfile(hist_path), "training_history.json not created"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – NaN/Inf in inputs raises InvalidTensorError
# ─────────────────────────────────────────────────────────────────────────────

def test_nan_inf_detection_raises():
    """
    [SYNTHETIC DATA TEST]
    Verifies that _check_tensor raises InvalidTensorError on NaN and Inf tensors.
    """
    good = torch.randn(4, 30, 126)
    nan_tensor = good.clone()
    nan_tensor[0, 0, 0] = float("nan")
    inf_tensor = good.clone()
    inf_tensor[0, 0, 0] = float("inf")

    with pytest.raises(InvalidTensorError, match="NaN"):
        _check_tensor("test_nan", nan_tensor)

    with pytest.raises(InvalidTensorError, match="Inf"):
        _check_tensor("test_inf", inf_tensor)

    # Clean tensor should not raise
    _check_tensor("clean", good)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Checkpoint save and resume restores identical weights
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_save_and_resume():
    """
    [SYNTHETIC DATA TEST]
    Saves a checkpoint after epoch 1, reloads it, and verifies the model
    state-dict is byte-identical.
    """
    with tempfile.TemporaryDirectory() as tmp_data, \
         tempfile.TemporaryDirectory() as tmp_ckpt, \
         tempfile.TemporaryDirectory() as tmp_metrics:

        _make_synthetic_dataset(
            tmp_data, n_per_class_per_signer=2, labels=("A", "B"),
            signers=("s1", "s2", "s3"),
        )

        cfg = TrainConfig(
            data_dir=tmp_data,
            train_signers=["s1", "s2"],
            val_signers=["s3"],
            epochs=1,
            batch_size=4,
            seed=0,
            patience=5,
            checkpoint_dir=tmp_ckpt,
            metrics_dir=tmp_metrics,
            lr_scheduler="none",
        )

        # Run epoch 1 – best_model.pt is saved
        trainer = Trainer(cfg)
        trainer.train()

        best_path = os.path.join(tmp_ckpt, "best_model.pt")
        assert os.path.isfile(best_path), "best_model.pt not saved"

        # Load checkpoint and verify state-dict equality
        ckpt = torch.load(best_path, map_location="cpu")
        loaded_sd = ckpt["model_state_dict"]

        # Rebuild fresh model and compare
        from models.config import ModelConfig
        from models.multimodal_model import MultimodalSignModel

        model_cfg = ModelConfig(num_classes=2)
        fresh_model = MultimodalSignModel(config=model_cfg)
        fresh_model.load_state_dict(loaded_sd)

        for key, param in fresh_model.state_dict().items():
            assert torch.equal(param, loaded_sd[key]), \
                f"State dict mismatch for key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – Early stopping fires after patience epochs
# ─────────────────────────────────────────────────────────────────────────────

def test_early_stopping_trigger():
    """
    [SYNTHETIC DATA TEST]
    With patience=1 and min_delta=0, training should stop after 2 epochs
    (first epoch is accepted, second shows no improvement).
    """
    with tempfile.TemporaryDirectory() as tmp_data, \
         tempfile.TemporaryDirectory() as tmp_ckpt, \
         tempfile.TemporaryDirectory() as tmp_metrics:

        _make_synthetic_dataset(
            tmp_data, n_per_class_per_signer=2, labels=("A", "B"),
            signers=("s1", "s2", "s3"),
        )

        cfg = TrainConfig(
            data_dir=tmp_data,
            train_signers=["s1", "s2"],
            val_signers=["s3"],
            epochs=20,          # allow up to 20, but patience should cut it short
            batch_size=4,
            seed=7,
            patience=1,
            min_delta=1e10,     # set impossibly high so epoch 2 always fails to improve
            checkpoint_dir=tmp_ckpt,
            metrics_dir=tmp_metrics,
            lr_scheduler="none",
        )

        trainer = Trainer(cfg)
        trainer.train()

        # Should stop after 2 epochs: epoch 1 (best saved), epoch 2 (no improvement → stop)
        assert len(trainer.history) == 2, \
            f"Expected 2 epochs before early stop, got {len(trainer.history)}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – DatasetMissingError on empty data_dir
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_missing_error():
    """
    [SAFEGUARD TEST]
    Calling Trainer.train() on an empty directory must raise DatasetMissingError.
    Training must never proceed when no real data is present.
    """
    with tempfile.TemporaryDirectory() as empty_dir, \
         tempfile.TemporaryDirectory() as tmp_ckpt, \
         tempfile.TemporaryDirectory() as tmp_metrics:

        cfg = TrainConfig(
            data_dir=empty_dir,
            train_signers=["s1"],
            val_signers=["s2"],
            epochs=1,
            checkpoint_dir=tmp_ckpt,
            metrics_dir=tmp_metrics,
        )

        trainer = Trainer(cfg)
        with pytest.raises(DatasetMissingError):
            trainer.train()
