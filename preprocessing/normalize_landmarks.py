"""
LSE Landmark Normalization & Preprocessing Pipeline

Transforms raw landmark CSV recordings into translation-invariant, scale-invariant,
and fixed-length (30 frames) numerical tensors suitable for sequence models (LSTM).

Outputs processed samples to: data/processed/<label>/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


EXPECTED_FRAMES = 30
LANDMARKS_PER_HAND = 21
COORDS_PER_LM = 3  # x, y, z
HANDS_ORDER = ["Left", "Right"]
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LM  # 63
TOTAL_FEATURES_HANDS = len(HANDS_ORDER) * FEATURES_PER_HAND  # 126


# ==============================================================================
# Modular Normalization & Feature Extraction Functions
# ==============================================================================

def normalize_hand_geometry(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes a single hand landmark array of shape (21, 3).
    1. Wrist centering (subtract landmark 0).
    2. Scale normalization (divide by maximum euclidean distance to wrist).

    If all landmarks are NaN (hand undetected), returns array of zeros (21, 3).
    """
    if np.isnan(landmarks).all():
        return np.zeros((LANDMARKS_PER_HAND, COORDS_PER_LM), dtype=np.float32)

    # Impute any sporadic NaN with 0 before normalization
    clean_lms = np.nan_to_num(landmarks, nan=0.0)

    # 1. Translation: Wrist (index 0) reference
    wrist = clean_lms[0, :]
    translated = clean_lms - wrist

    # 2. Scale: Maximum Euclidean distance from wrist to any landmark
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)

    eps = 1e-6
    if max_dist > eps:
        normalized = translated / max_dist
    else:
        normalized = translated

    return normalized.astype(np.float32)


def extract_frame_features(frame_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts a fixed 126-dimensional feature vector for a single frame:
    [Left_Hand (63 values), Right_Hand (63 values)].
    """
    frame_features = []

    for hand in HANDS_ORDER:
        hand_rows = frame_df[frame_df["hand"] == hand].sort_values("landmark_index")

        if len(hand_rows) == LANDMARKS_PER_HAND:
            lms = hand_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        else:
            # Missing hand or incomplete landmark entries
            lms = np.full((LANDMARKS_PER_HAND, COORDS_PER_LM), np.nan, dtype=np.float32)

        norm_lms = normalize_hand_geometry(lms)
        frame_features.append(norm_lms.flatten())

    # Concatenate Left (63) + Right (63) -> 126
    return np.concatenate(frame_features, axis=0)


def process_recording_file(filepath: str) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[str]]:
    """
    Validates, parses, normalizes, and extracts a (30, 126) sequence matrix from a CSV file.
    Returns: (feature_matrix, metadata_dict, error_message)
    """
    rel_path = os.path.relpath(filepath)
    parts = rel_path.split(os.sep)

    # Metadata extraction
    label = parts[-2] if len(parts) >= 2 else "UNKNOWN"
    filename = os.path.basename(filepath)
    signer = filename.split("_")[0] if "_" in filename else "UNKNOWN"
    rec_id = filename.split("_")[1].replace(".csv", "") if "_" in filename else "001"

    metadata = {
        "label": label,
        "signer": signer,
        "recording_id": rec_id,
        "source_file": filepath,
        "sequence_length": EXPECTED_FRAMES,
        "feature_dim": TOTAL_FEATURES_HANDS
    }

    # 1. Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return None, metadata, f"Read error: {e}"

    # 2. Schema check
    required_cols = {"frame", "hand", "landmark_index", "x", "y", "z"}
    if not required_cols.issubset(set(df.columns)):
        return None, metadata, f"Missing required columns: {required_cols - set(df.columns)}"

    # 3. Numeric check
    for col in ["frame", "landmark_index", "x", "y", "z"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None, metadata, f"Column '{col}' is not numeric."

    # 4. Check frame count
    unique_frames = sorted(df["frame"].dropna().unique().tolist())
    if len(unique_frames) != EXPECTED_FRAMES:
        return None, metadata, f"Expected {EXPECTED_FRAMES} frames, found {len(unique_frames)}"

    # 5. Extract features across all 30 frames
    sequence_data = []
    for frame_idx in range(EXPECTED_FRAMES):
        frame_df = df[df["frame"] == frame_idx]
        frame_vec = extract_frame_features(frame_df)
        sequence_data.append(frame_vec)

    feature_matrix = np.array(sequence_data, dtype=np.float32)  # Shape: (30, 126)

    # Final validation on shape
    if feature_matrix.shape != (EXPECTED_FRAMES, TOTAL_FEATURES_HANDS):
        return None, metadata, f"Invalid matrix shape: {feature_matrix.shape}"

    return feature_matrix, metadata, None


# ==============================================================================
# Pipeline Processor & File Exporter
# ==============================================================================

def normalize_dataset(raw_dir: str, processed_dir: str) -> None:
    """
    Scans raw dataset, normalizes each recording, and writes processed .npz and .csv files.
    """
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory '{raw_dir}' does not exist.", file=sys.stderr)
        return

    csv_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"No CSV recordings found in '{raw_dir}'.")
        return

    print("=" * 70)
    print(" LSE LANDMARK NORMALIZATION & PREPROCESSING PIPELINE")
    print(f" Raw Directory:       {raw_dir}")
    print(f" Processed Directory: {processed_dir}")
    print(f" Total Recordings:    {len(csv_files)}")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for filepath in sorted(csv_files):
        features, meta, error = process_recording_file(filepath)

        if error:
            print(f"[FAILED] {meta['source_file']}: {error}")
            fail_count += 1
            continue

        label_out_dir = os.path.join(processed_dir, meta["label"])
        os.makedirs(label_out_dir, exist_ok=True)

        base_name = f"{meta['signer']}_{meta['recording_id']}"
        npz_out_path = os.path.join(label_out_dir, f"{base_name}.npz")
        csv_out_path = os.path.join(label_out_dir, f"{base_name}_processed.csv")

        # 1. Save tensor and metadata in compressed NumPy format (.npz)
        np.savez_compressed(
            npz_out_path,
            features=features,
            label=meta["label"],
            signer=meta["signer"],
            recording_id=meta["recording_id"]
        )

        # 2. Also save tabular processed CSV for inspection
        flat_df = pd.DataFrame(features)
        # Create descriptive column names: L_0_x .. L_20_z, R_0_x .. R_20_z
        col_names = []
        for side in ["L", "R"]:
            for i in range(LANDMARKS_PER_HAND):
                for axis in ["x", "y", "z"]:
                    col_names.append(f"{side}_lm{i}_{axis}")
        flat_df.columns = col_names
        flat_df.insert(0, "frame", range(EXPECTED_FRAMES))
        flat_df.to_csv(csv_out_path, index=False)

        print(f"[PROCESSED] Label: {meta['label']:<10} Signer: {meta['signer']:<10} Shape: {features.shape} -> {npz_out_path}")
        success_count += 1

    print("-" * 70)
    print(f"Processing Complete: {success_count} succeeded, {fail_count} failed.\n")


def main():
    parser = argparse.ArgumentParser(description="Normalize LSE Landmark Dataset.")
    parser.add_argument("--raw_dir", type=str, default=os.path.join("data", "raw"), help="Raw recordings dir")
    parser.add_argument("--processed_dir", type=str, default=os.path.join("data", "processed"), help="Processed output dir")
    args = parser.parse_args()

    normalize_dataset(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
