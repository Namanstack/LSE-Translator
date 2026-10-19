"""
LSE Model Evaluation Framework

Calculates classification metrics (Accuracy, Macro Precision, Recall, F1, per-class stats),
computes confusion matrices, measures per-sequence inference latency, and saves reports
under results/metrics/ and results/confusion_matrices/.

Reusable across all modalities (Hands-only, Hands+Face, Hands+Face+Body).
Requires an existing trained model checkpoint and valid test dataset.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Ensure project root is available in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hands.dataset import LSEHandsDataset


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "LSE Confusion Matrix"
) -> None:
    """
    Plots and saves a formatted confusion matrix image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cell counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Runs model inference on the test DataLoader, computes all metrics, and exports reports.
    """
    metrics_dir = os.path.join(results_dir, "metrics")
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_signers = []
    all_rec_ids = []
    inference_times = []

    with torch.no_grad():
        for x_batch, y_batch, meta in dataloader:
            x_batch = x_batch.to(device)

            # Measure inference latency per sequence/batch
            start_time = time.perf_counter()
            logits = model(x_batch)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Store per-sample latency
            batch_sz = x_batch.size(0)
            inference_times.extend([elapsed_ms / batch_sz] * batch_sz)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y_batch.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

            # Metadata tracking
            if isinstance(meta["signer"], (list, tuple)):
                all_signers.extend(meta["signer"])
                all_rec_ids.extend(meta["recording_id"])
            else:
                all_signers.append(meta["signer"])
                all_rec_ids.append(meta["recording_id"])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    total_samples = len(all_targets)
    avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0

    # 1. Compute classification metrics
    acc = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    # 2. Detailed per-class metrics
    labels_idx = list(range(len(class_names)))
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_idx, average=None, zero_division=0
    )

    # 3. Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=labels_idx)

    # 4. Text classification report
    report_txt = classification_report(
        all_targets, all_preds, labels=labels_idx, target_names=class_names, zero_division=0
    )

    # -------------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------------
    # A. Save confusion matrix image
    cm_path = os.path.join(cm_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, class_names, cm_path)

    # B. Save classification report text
    report_path = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
        f.write(f"\nAverage Inference Latency: {avg_inference_ms:.2f} ms / sequence\n")

    # C. Save prediction records to CSV
    pred_labels = [class_names[p] if p < len(class_names) else f"Class_{p}" for p in all_preds]
    true_labels = [class_names[t] if t < len(class_names) else f"Class_{t}" for t in all_targets]

    pred_df = pd.DataFrame({
        "signer": all_signers,
        "recording_id": all_rec_ids,
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "correct": (all_preds == all_targets)
    })
    pred_csv_path = os.path.join(metrics_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # D. Save JSON summary metrics
    summary_metrics = {
        "num_test_samples": total_samples,
        "num_classes": len(class_names),
        "accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "avg_inference_time_ms": avg_inference_ms,
        "per_class": {
            class_names[i]: {
                "precision": float(p_class[i]),
                "recall": float(r_class[i]),
                "f1_score": float(f1_class[i]),
                "support": int(support_class[i])
            }
            for i in range(len(class_names))
        }
    }
    json_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)

    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description="LSE Model Evaluation Framework.")
    parser.add_argument("--model_path", type=str, default="", help="Path to trained PyTorch model checkpoint (.pth / .pt)")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"), help="Processed dataset directory")
    parser.add_argument("--test_signers", nargs="+", default=None, help="Signer IDs designated for test split")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    print("=" * 70)
    print(" LSE EVALUATION FRAMEWORK")
    print("=" * 70)

    # 1. Validation check for trained model checkpoint
    if not args.model_path or not os.path.exists(args.model_path):
        print("[Status: CANNOT EVALUATE]")
        print("Reason: No trained model checkpoint was provided or found.")
        print(f"Provided model path: '{args.model_path}'")
        print("\nNote: Evaluation requires a real trained checkpoint file (e.g. models/hands/hands_lstm.pth).")
        print("      Synthetic or placeholder evaluation results are strictly disabled.")
        print("=" * 70)
        sys.exit(0)

    # 2. Validation check for test dataset
    if not os.path.exists(args.data_dir):
        print("[Status: CANNOT EVALUATE]")
        print(f"Reason: Processed data directory '{args.data_dir}' does not exist.")
        print("=" * 70)
        sys.exit(0)

    test_dataset = LSEHandsDataset(data_dir=args.data_dir, allowed_signers=args.test_signers)

    if len(test_dataset) < 2:
        print("[Status: CANNOT EVALUATE]")
        print("Reason: Insufficient test dataset size.")
        print(f"Found {len(test_dataset)} sample(s). Evaluation requires a multi-sample, multi-class test split.")
        print("=" * 70)
        sys.exit(0)

    print(f"Test samples found: {len(test_dataset)}")
    print("Ready to run evaluation when model is trained.")


if __name__ == "__main__":
    main()
