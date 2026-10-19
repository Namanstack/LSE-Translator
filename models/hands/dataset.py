"""
LSE Hands-Only PyTorch Dataset Module

Loads normalized spatio-temporal hand landmark recordings (.npz) from data/processed/,
preserves signer identities for signer-independent validation, and produces PyTorch tensors
with shape (30, 126).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any


EXPECTED_SEQ_LEN = 30
EXPECTED_FEATURE_DIM = 126


class LSEHandsDataset(Dataset):
    """
    PyTorch Dataset for LSE Hand Landmark sequences.

    Each sample corresponds to a single gesture sequence:
        X: Tensor of shape (30, 126)
        y: Integer class index
        metadata: Dict containing label, signer, recording_id, and source_filepath.
    """

    def __init__(
        self,
        data_dir: str = os.path.join("data", "processed"),
        allowed_signers: Optional[List[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Any] = None
    ):
        """
        Args:
            data_dir (str): Directory containing processed label folders with .npz files.
            allowed_signers (list, optional): Filter dataset for specific signers (e.g. for signer-independent split).
            label_to_idx (dict, optional): Mapping from string label to integer class index. If None, built automatically.
            transform (callable, optional): Optional data augmentation transform applied to features.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.allowed_signers = [s.lower() for s in allowed_signers] if allowed_signers else None

        # Scan for all .npz files recursively
        pattern = os.path.join(data_dir, "**", "*.npz")
        all_files = glob.glob(pattern, recursive=True)

        self.samples = []
        labels_found = set()
        signers_found = set()

        for filepath in sorted(all_files):
            try:
                data = np.load(filepath, allow_pickle=True)
                features = data["features"]
                label = str(data["label"])
                signer = str(data["signer"]).lower()
                rec_id = str(data["recording_id"])
            except Exception as e:
                print(f"[Warning] Skipping unreadable .npz file {filepath}: {e}")
                continue

            # Validation checks
            if features.shape != (EXPECTED_SEQ_LEN, EXPECTED_FEATURE_DIM):
                print(f"[Warning] Invalid shape {features.shape} in {filepath}. Expected ({EXPECTED_SEQ_LEN}, {EXPECTED_FEATURE_DIM}). Skipping.")
                continue

            if np.isnan(features).any() or np.isinf(features).any():
                print(f"[Warning] NaN or Inf found in {filepath}. Skipping.")
                continue

            # Filter by signer if specified
            if self.allowed_signers is not None and signer not in self.allowed_signers:
                continue

            labels_found.add(label)
            signers_found.add(signer)

            self.samples.append({
                "features": features.astype(np.float32),
                "label": label,
                "signer": signer,
                "recording_id": rec_id,
                "filepath": filepath
            })

        # Build or assign label encoder mapping
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            sorted_labels = sorted(list(labels_found))
            self.label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        self.signers_found = sorted(list(signers_found))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        item = self.samples[idx]
        feat_data = item["features"]
        if self.transform is not None:
            feat_data = self.transform(feat_data)
        feature_tensor = torch.tensor(feat_data, dtype=torch.float32)  # (30, 126)
        label_str = item["label"]
        label_idx = self.label_to_idx.get(label_str, -1)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        metadata = {
            "label": label_str,
            "signer": item["signer"],
            "recording_id": item["recording_id"],
            "filepath": item["filepath"]
        }

        return feature_tensor, label_tensor, metadata


def get_dataset_signers(data_dir: str = os.path.join("data", "processed")) -> Tuple[List[str], List[str]]:
    """
    Scans processed directory and returns all unique signers and labels discovered.
    """
    pattern = os.path.join(data_dir, "**", "*.npz")
    all_files = glob.glob(pattern, recursive=True)
    signers = set()
    labels = set()

    for fp in all_files:
        try:
            data = np.load(fp, allow_pickle=True)
            labels.add(str(data["label"]))
            signers.add(str(data["signer"]).lower())
        except Exception:
            continue

    return sorted(list(signers)), sorted(list(labels))
