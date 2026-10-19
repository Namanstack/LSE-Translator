"""
LSE Signer-Independent Split Builder
=====================================
Constructs non-overlapping train / val / test dataset splits by assigning
entire signers exclusively to one partition.

Rationale
---------
Random sample-level splitting leaks signer anatomy, speed, and style cues into
the validation set, artificially inflating reported accuracy.  Partitioning by
signer ID is the standard protocol for evaluating genuine generalisation in
sign-language recognition research.

Public API
----------
    make_signer_independent_splits(...)  →  (train_ds, val_ds, test_ds, label_to_idx)
    validate_split_config(...)           →  raises ValueError on bad configuration
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path when running this file directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.multimodal_dataset import LSEMultimodalDataset, get_all_signers_and_labels


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_split_config(
    all_signers: List[str],
    train_signers: List[str],
    val_signers: List[str],
    test_signers: Optional[List[str]],
    all_labels: List[str],
) -> None:
    """
    Validates signer split configuration before building datasets.

    Checks:
        1. train_signers is non-empty.
        2. val_signers is non-empty.
        3. No signer appears in more than one partition (data leakage guard).
        4. All specified signers actually exist in the dataset.
        5. At least 2 unique labels are available (minimum for classification).

    Raises:
        ValueError with a descriptive message on any failure.
    """
    if not train_signers:
        raise ValueError("train_signers must not be empty.")
    if not val_signers:
        raise ValueError("val_signers must not be empty.")

    train_set = {s.lower() for s in train_signers}
    val_set = {s.lower() for s in val_signers}
    test_set = {s.lower() for s in test_signers} if test_signers else set()

    # ── Overlap / leakage checks ──────────────────────────────────────────────
    tv_overlap = train_set & val_set
    if tv_overlap:
        raise ValueError(
            f"Data leakage detected: signers appear in both train and val "
            f"partitions: {sorted(tv_overlap)}"
        )

    tt_overlap = train_set & test_set
    if tt_overlap:
        raise ValueError(
            f"Data leakage detected: signers appear in both train and test "
            f"partitions: {sorted(tt_overlap)}"
        )

    vt_overlap = val_set & test_set
    if vt_overlap:
        raise ValueError(
            f"Data leakage detected: signers appear in both val and test "
            f"partitions: {sorted(vt_overlap)}"
        )

    # ── Unknown signer checks ─────────────────────────────────────────────────
    all_signer_set = {s.lower() for s in all_signers}
    unknown = (train_set | val_set | test_set) - all_signer_set
    if unknown:
        raise ValueError(
            f"Specified signer(s) not found in data_dir: {sorted(unknown)}. "
            f"Available signers: {sorted(all_signer_set)}"
        )

    # ── Minimum class requirement ─────────────────────────────────────────────
    if len(all_labels) < 2:
        raise ValueError(
            f"Dataset contains only {len(all_labels)} label(s).  "
            "At least 2 classes are required for classification."
        )


def make_signer_independent_splits(
    data_dir: str,
    train_signers: List[str],
    val_signers: List[str],
    test_signers: Optional[List[str]] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
    require_face: bool = True,
) -> Tuple[
    LSEMultimodalDataset,
    LSEMultimodalDataset,
    Optional[LSEMultimodalDataset],
    Dict[str, int],
]:
    """
    Build signer-independent train / val / test dataset splits.

    A single, shared ``label_to_idx`` vocabulary is used across all splits so
    that class indices are consistent and no information leaks between partitions.

    Args:
        data_dir (str):
            Root directory searched recursively for .npz files.
        train_signers (list[str]):
            Signers assigned exclusively to the training split.
        val_signers (list[str]):
            Signers assigned exclusively to the validation split.
        test_signers (list[str] | None):
            Optional signers for a held-out test split.
        label_to_idx (dict | None):
            Optional pre-built vocabulary.  If None, the vocabulary is derived
            from the *complete* (unfiltered) dataset to guarantee all labels are
            covered regardless of how signers are partitioned.
        require_face (bool):
            Passed through to LSEMultimodalDataset (default True).

    Returns:
        train_ds     : LSEMultimodalDataset – training partition.
        val_ds       : LSEMultimodalDataset – validation partition.
        test_ds      : LSEMultimodalDataset or None – test partition.
        label_to_idx : dict[str, int] – shared vocabulary mapping.

    Raises:
        ValueError : on invalid split configuration (see validate_split_config).
        RuntimeError: if any mandatory split is empty after filtering.
    """
    # ── Discover all signers and labels from the raw data ─────────────────────
    all_signers, all_labels = get_all_signers_and_labels(data_dir)

    # ── Validate configuration before touching data ───────────────────────────
    validate_split_config(all_signers, train_signers, val_signers, test_signers, all_labels)

    # ── Build shared vocabulary from the full (unfiltered) dataset ───────────
    if label_to_idx is None:
        full_ds = LSEMultimodalDataset(
            data_dir=data_dir,
            allowed_signers=None,
            label_to_idx=None,
            require_face=require_face,
        )
        label_to_idx = full_ds.label_to_idx

    # ── Build individual splits ───────────────────────────────────────────────
    train_ds = LSEMultimodalDataset(
        data_dir=data_dir,
        allowed_signers=train_signers,
        label_to_idx=label_to_idx,
        require_face=require_face,
    )
    val_ds = LSEMultimodalDataset(
        data_dir=data_dir,
        allowed_signers=val_signers,
        label_to_idx=label_to_idx,
        require_face=require_face,
    )
    test_ds: Optional[LSEMultimodalDataset] = None
    if test_signers:
        test_ds = LSEMultimodalDataset(
            data_dir=data_dir,
            allowed_signers=test_signers,
            label_to_idx=label_to_idx,
            require_face=require_face,
        )

    # ── Ensure mandatory splits are non-empty ────────────────────────────────
    if len(train_ds) == 0:
        raise RuntimeError(
            f"Training split is empty after filtering to signers: {train_signers}. "
            "Check that the specified signers have data in data_dir."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"Validation split is empty after filtering to signers: {val_signers}. "
            "Check that the specified signers have data in data_dir."
        )

    return train_ds, val_ds, test_ds, label_to_idx


def summarise_splits(
    train_ds: LSEMultimodalDataset,
    val_ds: LSEMultimodalDataset,
    test_ds: Optional[LSEMultimodalDataset],
) -> str:
    """Returns a human-readable summary string for logging."""
    lines = [
        "─" * 60,
        " Signer-Independent Dataset Splits",
        "─" * 60,
        f"  Train : {len(train_ds):>5} samples  |  signers: {train_ds.signers_found}",
        f"  Val   : {len(val_ds):>5} samples  |  signers: {val_ds.signers_found}",
    ]
    if test_ds is not None:
        lines.append(
            f"  Test  : {len(test_ds):>5} samples  |  signers: {test_ds.signers_found}"
        )
    lines += [
        f"  Vocabulary ({len(train_ds.label_to_idx)} classes): "
        f"{sorted(train_ds.label_to_idx.keys())}",
        "─" * 60,
    ]
    return "\n".join(lines)
