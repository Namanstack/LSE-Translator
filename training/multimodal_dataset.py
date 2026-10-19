"""
LSE Multimodal Dataset
======================
PyTorch Dataset for paired hand + face landmark recordings.

Each .npz sample must contain:
    hand_features  : np.ndarray  shape (30, 126)   float32
    face_features  : np.ndarray  shape (30, 1404)  float32
    label          : str         e.g. "HOLA"
    signer         : str         e.g. "signer_01"
    recording_id   : str         e.g. "rec_0042"

__getitem__ returns:
    hand_tensor  : torch.Tensor  (30, 126)
    face_tensor  : torch.Tensor  (30, 1404)
    label_tensor : torch.Tensor  scalar long
    metadata     : dict  {label, signer, recording_id, filepath}

Safeguards enforced at load time:
    - Incorrect hand or face shape  → warning, file skipped.
    - NaN or Inf values             → warning, file skipped.
    - Signer not in allowed_signers → filtered out silently.
    - label not in label_to_idx     → warning, file skipped.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# Expected tensor contracts (must match ModelConfig)
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_SEQ_LEN: int = 30
EXPECTED_HAND_DIM: int = 126
EXPECTED_FACE_DIM: int = 1404


class LSEMultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal (hand + face) LSE landmark sequences.

    Args:
        data_dir (str):
            Root directory that is searched recursively for .npz files.
        allowed_signers (list[str] | None):
            If provided, only recordings from these signers are loaded.
            Comparison is case-insensitive.
        label_to_idx (dict[str, int] | None):
            Pre-built vocabulary mapping.  If None, the mapping is inferred
            from all labels discovered in data_dir (after signer filtering).
            Pass the same mapping to all splits to prevent leakage.
        require_face (bool):
            If True (default), samples without face_features are skipped.
            Set to False to load hand-only recordings (face tensor will be zeros).
    """

    def __init__(
        self,
        data_dir: str,
        allowed_signers: Optional[List[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        require_face: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        self.data_dir = data_dir
        self.require_face = require_face
        self.transform = transform
        self.allowed_signers = (
            {s.lower() for s in allowed_signers} if allowed_signers else None
        )

        self.samples: List[Dict] = []
        labels_found: set = set()
        signers_found: set = set()

        pattern = os.path.join(data_dir, "**", "*.npz")
        all_files = sorted(glob.glob(pattern, recursive=True))

        for filepath in all_files:
            record = self._load_and_validate(filepath)
            if record is None:
                continue

            # ── Signer filter ────────────────────────────────────────────────
            if self.allowed_signers is not None:
                if record["signer"] not in self.allowed_signers:
                    continue

            labels_found.add(record["label"])
            signers_found.add(record["signer"])
            self.samples.append(record)

        # ── Build vocabulary ─────────────────────────────────────────────────
        if label_to_idx is not None:
            self.label_to_idx: Dict[str, int] = label_to_idx
        else:
            sorted_labels = sorted(labels_found)
            self.label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

        self.idx_to_label: Dict[int, str] = {
            idx: lbl for lbl, idx in self.label_to_idx.items()
        }
        self.signers_found: List[str] = sorted(signers_found)

        # ── Drop samples whose label is outside the provided vocabulary ──────
        before = len(self.samples)
        self.samples = [
            s for s in self.samples if s["label"] in self.label_to_idx
        ]
        dropped = before - len(self.samples)
        if dropped > 0:
            print(
                f"[Warning] Dropped {dropped} sample(s) with labels not in "
                "label_to_idx.  This can happen when a shared vocabulary built "
                "from all signers is applied to a signer-filtered split."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_and_validate(self, filepath: str) -> Optional[Dict]:
        """
        Load a single .npz file and validate its contents.

        Returns a dict on success or None if the file must be skipped.
        """
        try:
            data = np.load(filepath, allow_pickle=True)
        except Exception as exc:
            print(f"[Warning] Cannot open {filepath}: {exc}  Skipping.")
            return None

        # ── Hand features ────────────────────────────────────────────────────
        if "hand_features" not in data:
            print(
                f"[Warning] {filepath} missing 'hand_features' key.  Skipping."
            )
            return None

        hand = data["hand_features"].astype(np.float32)
        if hand.shape != (EXPECTED_SEQ_LEN, EXPECTED_HAND_DIM):
            print(
                f"[Warning] {filepath}: hand_features shape {hand.shape} "
                f"!= ({EXPECTED_SEQ_LEN}, {EXPECTED_HAND_DIM}).  Skipping."
            )
            return None

        if np.isnan(hand).any() or np.isinf(hand).any():
            print(f"[Warning] {filepath}: NaN/Inf in hand_features.  Skipping.")
            return None

        # ── Face features ────────────────────────────────────────────────────
        if "face_features" not in data:
            if self.require_face:
                print(
                    f"[Warning] {filepath} missing 'face_features' key "
                    "(require_face=True).  Skipping."
                )
                return None
            face = np.zeros((EXPECTED_SEQ_LEN, EXPECTED_FACE_DIM), dtype=np.float32)
        else:
            face = data["face_features"].astype(np.float32)
            if face.shape != (EXPECTED_SEQ_LEN, EXPECTED_FACE_DIM):
                print(
                    f"[Warning] {filepath}: face_features shape {face.shape} "
                    f"!= ({EXPECTED_SEQ_LEN}, {EXPECTED_FACE_DIM}).  Skipping."
                )
                return None
            if np.isnan(face).any() or np.isinf(face).any():
                print(
                    f"[Warning] {filepath}: NaN/Inf in face_features.  Skipping."
                )
                return None

        # ── Metadata fields ──────────────────────────────────────────────────
        try:
            label = str(data["label"])
            signer = str(data["signer"]).lower()
            rec_id = str(data["recording_id"])
        except KeyError as exc:
            print(f"[Warning] {filepath} missing metadata key {exc}.  Skipping.")
            return None

        return {
            "hand_features": hand,
            "face_features": face,
            "label": label,
            "signer": signer,
            "recording_id": rec_id,
            "filepath": filepath,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, str]]:
        """
        Returns:
            hand_tensor  (30, 126)   float32
            face_tensor  (30, 1404)  float32
            label_tensor scalar      int64
            metadata     dict
        """
        item = self.samples[idx]
        hand_raw = item["hand_features"]
        face_raw = item["face_features"]

        if self.transform is not None:
            hand_raw, face_raw = self.transform(hand_raw, face_raw)

        hand_tensor = torch.tensor(hand_raw, dtype=torch.float32)
        face_tensor = torch.tensor(face_raw, dtype=torch.float32)

        label_idx = self.label_to_idx[item["label"]]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        metadata: Dict[str, str] = {
            "label": item["label"],
            "signer": item["signer"],
            "recording_id": item["recording_id"],
            "filepath": item["filepath"],
        }

        return hand_tensor, face_tensor, label_tensor, metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────────────

    def class_distribution(self) -> Dict[str, int]:
        """Returns a dict mapping each label to sample count."""
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery helper
# ─────────────────────────────────────────────────────────────────────────────

def get_all_signers_and_labels(
    data_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Scans data_dir recursively for .npz files and returns sorted lists of
    unique (signer, label) values.  Does not load feature arrays.

    Returns:
        signers (list[str]): Sorted unique signer IDs (lower-cased).
        labels  (list[str]): Sorted unique label strings.
    """
    signers: set = set()
    labels: set = set()
    pattern = os.path.join(data_dir, "**", "*.npz")
    for fp in sorted(glob.glob(pattern, recursive=True)):
        try:
            data = np.load(fp, allow_pickle=True)
            signers.add(str(data["signer"]).lower())
            labels.add(str(data["label"]))
        except Exception:
            continue
    return sorted(signers), sorted(labels)
