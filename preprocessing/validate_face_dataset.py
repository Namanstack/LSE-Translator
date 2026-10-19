"""
LSE Face Dataset Validation Utility

Scans data/processed/ recursively to validate consistency, structure, and numerical integrity
of all processed face landmark recordings (.npz).

Expected tensor shape: (30, 1404) -> 30 frames x (468 landmarks * 3 coordinates)
"""

import os
import sys
import glob
import argparse
import numpy as np
from typing import Dict, Any, List


EXPECTED_SEQ_LEN = 30
EXPECTED_FACE_LANDMARKS = 468
EXPECTED_COORDS = 3
EXPECTED_FEATURE_DIM = EXPECTED_FACE_LANDMARKS * EXPECTED_COORDS  # 1404
EXPECTED_SHAPE = (EXPECTED_SEQ_LEN, EXPECTED_FEATURE_DIM)


def validate_single_face_file(filepath: str) -> Dict[str, Any]:
    """
    Validates an individual processed face .npz file.
    Returns metadata dict with validation status and error messages.
    """
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath)

    result = {
        "filepath": filepath,
        "filename": filename,
        "rel_path": rel_path,
        "label": "UNKNOWN",
        "signer": "UNKNOWN",
        "recording_id": "UNKNOWN",
        "shape": None,
        "valid_frames": None,
        "is_valid": False,
        "error_messages": []
    }

    # 1. Read .npz archive
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        result["error_messages"].append(f"Failed to read .npz file: {e}")
        return result

    # 2. Check essential keys
    required_keys = {"features", "label", "signer", "recording_id"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        result["error_messages"].append(f"Missing required metadata keys: {sorted(list(missing_keys))}")
        return result

    try:
        features = data["features"]
        label = str(data["label"]).strip()
        signer = str(data["signer"]).strip()
        recording_id = str(data["recording_id"]).strip()

        result["label"] = label if label else "UNKNOWN"
        result["signer"] = signer if signer else "UNKNOWN"
        result["recording_id"] = recording_id if recording_id else "UNKNOWN"
        result["shape"] = features.shape

        if "valid_frames" in data.files:
            result["valid_frames"] = int(data["valid_frames"])
    except Exception as e:
        result["error_messages"].append(f"Error accessing metadata: {e}")
        return result

    # 3. Label & Signer validation
    if not label or label == "UNKNOWN":
        result["error_messages"].append("Missing or invalid sign label.")

    if not signer or signer == "UNKNOWN":
        result["error_messages"].append("Missing or invalid signer ID.")

    if not recording_id or recording_id == "UNKNOWN":
        result["error_messages"].append("Missing or invalid recording ID.")

    # 4. Shape validation (30, 1404)
    if features.shape != EXPECTED_SHAPE:
        result["error_messages"].append(
            f"Invalid tensor shape: {features.shape}. Expected exactly {EXPECTED_SHAPE}."
        )

    # 5. Numerical integrity: No NaN or Inf
    if not np.issubdtype(features.dtype, np.number):
        result["error_messages"].append(f"Non-numeric data type: {features.dtype}")
    else:
        if np.isnan(features).any():
            nan_count = int(np.isnan(features).sum())
            result["error_messages"].append(f"Contains {nan_count} NaN values.")

        if np.isinf(features).any():
            inf_count = int(np.isinf(features).sum())
            result["error_messages"].append(f"Contains {inf_count} Infinite values.")

    # 6. Overall validity
    if not result["error_messages"]:
        result["is_valid"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate LSE Processed Face Dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Root processed dataset directory (default: data/processed)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Processed data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Search for all face .npz files recursively (files starting with 'face_')
    pattern = os.path.join(data_dir, "**", "face_*.npz")
    face_files = glob.glob(pattern, recursive=True)

    if not face_files:
        print(f"No processed face recordings ('face_*.npz') found in '{data_dir}'.")
        sys.exit(0)

    # Run validation across all files
    results = [validate_single_face_file(f) for f in sorted(face_files)]

    total_recordings = len(results)
    valid_recordings = sum(1 for r in results if r["is_valid"])
    invalid_recordings = total_recordings - valid_recordings
    all_labels = {r["label"] for r in results if r["label"] != "UNKNOWN"}
    all_signers = {r["signer"] for r in results if r["signer"] != "UNKNOWN"}
    total_frames = sum(r["shape"][0] for r in results if r["shape"] and len(r["shape"]) >= 1)

    # Summary Report Output
    print("\n" + "=" * 70)
    print("                 FACE DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f" Total Face Recordings: {total_recordings}")
    print(f" Valid Recordings:      {valid_recordings}")
    print(f" Invalid Recordings:    {invalid_recordings}")
    print(f" Expected Shape:        {EXPECTED_SHAPE} (30 frames x 1404 features)")
    print(f" Total Frames:          {total_frames}")
    print(f" Labels Discovered ({len(all_labels)}): {', '.join(sorted(all_labels)) if all_labels else 'None'}")
    print(f" Signers Discovered ({len(all_signers)}):{', '.join(sorted(all_signers)) if all_signers else 'None'}")
    print("-" * 70)
    print(" INDIVIDUAL RECORDING AUDIT:")
    print("-" * 70)

    for r in results:
        status_str = "VALID" if r["is_valid"] else "INVALID"
        valid_frames_str = f"{r['valid_frames']}/{EXPECTED_SEQ_LEN}" if r["valid_frames"] is not None else "N/A"
        print(f" Filename:       {r['filename']}")
        print(f" Path:           {r['rel_path']}")
        print(f" Label:          {r['label']}")
        print(f" Signer:         {r['signer']}")
        print(f" Recording ID:   {r['recording_id']}")
        print(f" Shape:          {r['shape']}")
        print(f" Valid Frames:   {valid_frames_str}")
        print(f" Status:         {status_str}")

        if not r["is_valid"]:
            for err in r["error_messages"]:
                print(f"   -> [Error] {err}")
        print("-" * 70)

    # Return exit code
    if invalid_recordings > 0:
        print(f"[FAILED] Found {invalid_recordings} invalid face recording(s).\n")
        sys.exit(1)
    else:
        print("[PASSED] All face recordings are valid.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
