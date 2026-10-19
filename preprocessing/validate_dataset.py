"""
LSE Dataset Validation Utility

Scans data/raw/ recursively to validate consistency, structure, and completeness
of all captured landmark CSV recordings.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {"frame", "hand", "landmark_index", "x", "y", "z"}
EXPECTED_FRAMES = 30
EXPECTED_LANDMARK_COUNT = 21  # per hand (0..20)


def validate_single_recording(filepath: str) -> dict:
    """
    Validates an individual recording CSV file.
    Returns a dictionary containing summary metadata and validation status.
    """
    rel_path = os.path.relpath(filepath)
    parts = rel_path.split(os.sep)

    # Infer sign label and signer id from path structure:
    # Expected: data/raw/<label>/<signer>_<rec_id>.csv
    label = parts[-2] if len(parts) >= 2 else "UNKNOWN"
    filename = os.path.basename(filepath)
    signer = filename.split("_")[0] if "_" in filename else "UNKNOWN"

    result = {
        "filepath": filepath,
        "filename": filename,
        "label": label,
        "signer": signer,
        "num_frames": 0,
        "hands_detected": set(),
        "is_valid": False,
        "error_messages": []
    }

    # 1. Readability check
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        result["error_messages"].append(f"Failed to read CSV: {e}")
        return result

    # 2. Required columns check
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        result["error_messages"].append(f"Missing columns: {sorted(list(missing_cols))}")
        return result

    # 3. Numeric types check for coordinates and indices
    for col in ["frame", "landmark_index", "x", "y", "z"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            result["error_messages"].append(f"Column '{col}' is not numeric.")

    # 4. Frame sequence check (0 to EXPECTED_FRAMES - 1)
    unique_frames = sorted(df["frame"].dropna().unique().tolist())
    result["num_frames"] = len(unique_frames)

    if result["num_frames"] != EXPECTED_FRAMES:
        result["error_messages"].append(
            f"Invalid frame count: expected {EXPECTED_FRAMES}, found {result['num_frames']}"
        )
    elif unique_frames != list(range(EXPECTED_FRAMES)):
        result["error_messages"].append(
            f"Frame sequence is non-contiguous or does not start at 0: {unique_frames}"
        )

    # 5. Landmark indices check (each must be in 0..20)
    invalid_indices = df[~df["landmark_index"].isin(range(EXPECTED_LANDMARK_COUNT))]
    if not invalid_indices.empty:
        result["error_messages"].append(
            f"Found {len(invalid_indices)} rows with invalid landmark_index outside 0..{EXPECTED_LANDMARK_COUNT - 1}"
        )

    # 6. Hand sides present and non-empty coordinate inspection
    # Check hands that have at least one non-NaN coordinate row
    valid_coords = df.dropna(subset=["x", "y", "z"])
    detected_hands = set(valid_coords["hand"].dropna().unique())
    result["hands_detected"] = sorted(list(detected_hands)) if detected_hands else ["None"]

    # 7. Overall status
    if not result["error_messages"]:
        result["is_valid"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate LSE Hand Landmark Dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Root directory of raw dataset (default: data/raw)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Target directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Find all CSV files recursively
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"No CSV recordings found in '{data_dir}'.")
        sys.exit(0)

    # Validate each recording
    results = [validate_single_recording(f) for f in sorted(csv_files)]

    total_recordings = len(results)
    valid_recordings = sum(1 for r in results if r["is_valid"])
    invalid_recordings = total_recordings - valid_recordings
    all_labels = {r["label"] for r in results}
    all_signers = {r["signer"] for r in results}

    # Print summary report
    print("\n" + "=" * 70)
    print("                 DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f" Total Recordings:   {total_recordings}")
    print(f" Valid Recordings:   {valid_recordings}")
    print(f" Invalid Recordings: {invalid_recordings}")
    print(f" Total Labels ({len(all_labels)}):   {', '.join(sorted(all_labels))}")
    print(f" Total Signers ({len(all_signers)}):  {', '.join(sorted(all_signers))}")
    print("-" * 70)
    print(" INDIVIDUAL RECORDINGS:")
    print("-" * 70)

    for r in results:
        status_str = "VALID" if r["is_valid"] else "INVALID"
        hands_str = ", ".join(r["hands_detected"])
        print(f" Filename:        {r['filename']}")
        print(f" Label:           {r['label']}")
        print(f" Signer:          {r['signer']}")
        print(f" Frames:          {r['num_frames']}/{EXPECTED_FRAMES}")
        print(f" Hands Detected:  {hands_str}")
        print(f" Status:          {status_str}")
        if not r["is_valid"]:
            for err in r["error_messages"]:
                print(f"   -> [Error] {err}")
        print("-" * 70)

    # Return non-zero exit code if any recording is invalid
    if invalid_recordings > 0:
        print(f"[FAILED] Found {invalid_recordings} invalid recording(s).\n")
        sys.exit(1)
    else:
        print("[PASSED] All recordings are valid.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
