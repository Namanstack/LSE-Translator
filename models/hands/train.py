"""
LSE Hands-Only Model Training Pipeline

Provides signer-independent dataset splitting, PyTorch DataLoader configuration,
training and validation loops, and a standalone non-training dry-run validation mode.

Signer-Independent Splitting Rationale:
In sign language recognition, random sample-level splitting often causes data leakage
because the network easily memorizes individual signer anatomy, speed, or idiosyncratic
hand posture. Partitioning datasets strictly by signer ID ensures that the model is
evaluated on unseen signers, measuring true generalizability.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hands.dataset import LSEHandsDataset, get_dataset_signers
from models.hands.lstm_model import HandsLSTMClassifier, get_model_summary


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_signer_independent_splits(
    data_dir: str,
    train_signers: List[str],
    val_signers: List[str],
    test_signers: Optional[List[str]] = None
) -> Tuple[LSEHandsDataset, LSEHandsDataset, Optional[LSEHandsDataset], Dict[str, int]]:
    """
    Constructs separate dataset splits based strictly on signer identifiers to avoid data leakage.
    Shares a consistent label-to-index mapping across splits.
    """
    # Build common vocabulary mapping across all data
    full_dataset = LSEHandsDataset(data_dir=data_dir)
    label_to_idx = full_dataset.label_to_idx

    train_ds = LSEHandsDataset(data_dir=data_dir, allowed_signers=train_signers, label_to_idx=label_to_idx)
    val_ds = LSEHandsDataset(data_dir=data_dir, allowed_signers=val_signers, label_to_idx=label_to_idx)
    test_ds = LSEHandsDataset(data_dir=data_dir, allowed_signers=test_signers, label_to_idx=label_to_idx) if test_signers else None

    return train_ds, val_ds, test_ds, label_to_idx


def run_dry_run(data_dir: str = os.path.join("data", "processed")) -> bool:
    """
    Executes a dry-run test without training:
    1. Discovers processed recordings, labels, and signers.
    2. Instantiates LSEHandsDataset and PyTorch DataLoader.
    3. Executes one forward pass through HandsLSTMClassifier.
    4. Validates input and output tensor dimensions.
    """
    print("=" * 70)
    print(" LSE HANDS-ONLY TRAINING PIPELINE - DRY RUN VERIFICATION")
    print("=" * 70)

    # 1. Discover dataset metadata
    all_signers, all_labels = get_dataset_signers(data_dir)
    dataset = LSEHandsDataset(data_dir=data_dir)

    print(f" Data Directory:       {data_dir}")
    print(f" Recordings Discovered: {len(dataset)}")
    print(f" Labels Discovered ({len(all_labels)}):   {all_labels}")
    print(f" Signers Discovered ({len(all_signers)}):  {all_signers}")
    print("-" * 70)

    if len(dataset) == 0:
        print("[Error] No valid processed recordings found for dry-run.", file=sys.stderr)
        return False

    # 2. Inspect a single sample tensor
    sample_x, sample_y, meta = dataset[0]
    print(f" Single Sample Tensor:  Shape = {sample_x.shape} (dtype: {sample_x.dtype})")
    print(f" Single Sample Label:   {meta['label']} (Class Index: {sample_y.item()})")
    print(f" Signer:                {meta['signer']}")
    print(f" Recording ID:          {meta['recording_id']}")
    print("-" * 70)

    # 3. Create DataLoader
    batch_size = min(4, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_x, batch_y, batch_meta = next(iter(loader))
    print(f" DataLoader Batch X:    Shape = {batch_x.shape}")
    print(f" DataLoader Batch Y:    Shape = {batch_y.shape}")
    print("-" * 70)

    # 4. Instantiate model with discovered classes (minimum 1)
    num_classes = max(1, len(dataset.label_to_idx))
    model = HandsLSTMClassifier(
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    )
    model.eval()

    # 5. Perform forward pass (no backprop / no optimizer)
    with torch.no_grad():
        logits = model(batch_x)

    print(f" Model Output Logits:   Shape = {logits.shape} (batch_size={batch_x.shape[0]}, num_classes={num_classes})")
    print(f" Sample Output Vector:  {logits[0].numpy()}")
    print("-" * 70)

    # Shape validation
    assert batch_x.shape == (batch_size, 30, 126), f"Unexpected batch_x shape: {batch_x.shape}"
    assert logits.shape == (batch_size, num_classes), f"Unexpected logits shape: {logits.shape}"

    print("[PASSED] Pipeline dry-run completed successfully. Zero optimizer steps taken.")
    print("=" * 70)
    return True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Reusable training loop for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch, _ in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Reusable validation/evaluation loop."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch, _ in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    eval_loss = running_loss / total if total > 0 else 0.0
    eval_acc = (correct / total) if total > 0 else 0.0
    return eval_loss, eval_acc


def main():
    parser = argparse.ArgumentParser(description="LSE Hands-Only Model Training Pipeline.")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"), help="Processed data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", default=True, help="Execute dry-run without training")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dry_run:
        passed = run_dry_run(args.data_dir)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
