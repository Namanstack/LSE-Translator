"""
LSE Face Landmark Dataset Collector - Modality 2

Captures 30-frame facial landmark sequences using MediaPipe Face Mesh,
normalizes 468 landmarks relative to the nose tip (468 x 3 = 1404 features per frame),
and saves both:
1. Raw structured CSVs under data/raw/<LABEL>/<SIGNER>/face_<recording_id>.csv
2. Processed .npz arrays with shape (30, 1404) under data/processed/<LABEL>/face_<signer>_<recording_id>.npz
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Tuple, Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.webcam_face import (
    extract_and_normalize_face_landmarks,
    NOSE_REFERENCE_INDEX,
    TOTAL_FACE_LANDMARKS
)

TOTAL_FACE_FEATURES = TOTAL_FACE_LANDMARKS * 3  # 468 * 3 = 1404
DEFAULT_SEQ_LEN = 30


def get_unique_filepath(base_path: str) -> str:
    """Ensures recording files are not accidentally overwritten."""
    if not os.path.exists(base_path):
        return base_path

    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{name}_copy{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def validate_face_sequence_tensor(
    tensor: np.ndarray,
    expected_seq_len: int = DEFAULT_SEQ_LEN,
    expected_feature_dim: int = TOTAL_FACE_FEATURES
) -> Tuple[bool, Optional[str]]:
    """
    Validates the generated facial feature tensor:
    - Shape must be exactly (seq_len, 1404)
    - No NaN values
    - No infinite values
    """
    if tensor.shape != (expected_seq_len, expected_feature_dim):
        return False, f"Invalid tensor shape: {tensor.shape}, expected ({expected_seq_len}, {expected_feature_dim})"

    if np.isnan(tensor).any():
        return False, "Tensor contains NaN values."

    if np.isinf(tensor).any():
        return False, "Tensor contains Infinite values."

    return True, None


def save_face_recording(
    raw_records: list,
    processed_tensor: np.ndarray,
    metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Saves raw CSV in data/raw/<LABEL>/<SIGNER>/ and processed .npz in data/processed/<LABEL>/
    """
    label = metadata["label"]
    signer = metadata["signer"]
    rec_id = metadata["recording_id"]

    # 1. Raw output path: data/raw/<LABEL>/<SIGNER>/face_<rec_id>.csv
    raw_dir = os.path.join("data", "raw", label, signer)
    os.makedirs(raw_dir, exist_ok=True)
    raw_filename = f"face_{rec_id:03d}.csv"
    raw_path = get_unique_filepath(os.path.join(raw_dir, raw_filename))

    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(raw_path, index=False)

    # 2. Processed output path: data/processed/<LABEL>/face_<signer>_<rec_id>.npz
    proc_dir = os.path.join("data", "processed", label)
    os.makedirs(proc_dir, exist_ok=True)
    proc_filename = f"face_{signer}_{rec_id:03d}.npz"
    proc_path = get_unique_filepath(os.path.join(proc_dir, proc_filename))

    np.savez_compressed(
        proc_path,
        features=processed_tensor,
        label=label,
        signer=signer,
        recording_id=str(rec_id),
        sequence_length=metadata["sequence_length"],
        feature_dimension=metadata["feature_dimension"],
        valid_frames=metadata["valid_frames"],
        modality="face"
    )

    return raw_path, proc_path


def main():
    parser = argparse.ArgumentParser(description="Collect LSE Face Landmark Dataset.")
    parser.add_argument("--label", type=str, default="HOLA", help="Sign vocabulary label (e.g. HOLA, GRACIAS)")
    parser.add_argument("--signer", type=str, default="signer01", help="Signer ID (e.g. signer01)")
    parser.add_argument("--recording_id", type=int, default=1, help="Recording number (e.g. 1)")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length in frames (default: 30)")
    args = parser.parse_args()

    sign_label = args.label.strip().upper()
    signer_id = args.signer.strip().lower()
    recording_id = args.recording_id
    seq_len = args.seq_len

    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for face dataset collection.", file=sys.stderr)
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print("=" * 70)
    print(" LSE FACE LANDMARK DATASET COLLECTOR")
    print(f" Sign Label:     {sign_label}")
    print(f" Signer ID:      {signer_id}")
    print(f" Recording ID:   {recording_id:03d}")
    print(f" Sequence Len:   {seq_len} frames")
    print(f" Feature Shape:  ({seq_len}, {TOTAL_FACE_FEATURES})")
    print(" Controls:       [SPACE] Start Capture | [R] Reset | [Q] Quit")
    print("=" * 70)

    is_recording = False
    current_frame_idx = 0
    raw_sequence_records = []
    processed_frame_vectors = []
    valid_face_frames_count = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Camera stream disconnected. Exiting...", file=sys.stderr)
                break

            # 1. Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # 2. MediaPipe RGB conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            face_detected = bool(results.multi_face_landmarks)

            # 3. Draw face mesh if present
            if face_detected:
                primary_face = results.multi_face_landmarks[0]
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=primary_face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=primary_face,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # 4. Handle recording buffer
            if is_recording:
                if face_detected:
                    primary_face = results.multi_face_landmarks[0]
                    valid_face_frames_count += 1
                    # Extract normalized (468, 3) matrix
                    norm_face = extract_and_normalize_face_landmarks(primary_face)
                    frame_flat = norm_face.flatten()  # (1404,)

                    # Store raw unflattened coordinates for auditability
                    for idx, lm in enumerate(primary_face.landmark[:TOTAL_FACE_LANDMARKS]):
                        raw_sequence_records.append({
                            "frame": current_frame_idx,
                            "landmark_index": idx,
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "face_detected": 1
                        })
                else:
                    # Missing face detection in this frame: safe zero-imputation
                    norm_face = extract_and_normalize_face_landmarks(None)
                    frame_flat = norm_face.flatten()  # (1404,) with zeros

                    for idx in range(TOTAL_FACE_LANDMARKS):
                        raw_sequence_records.append({
                            "frame": current_frame_idx,
                            "landmark_index": idx,
                            "x": float("nan"),
                            "y": float("nan"),
                            "z": float("nan"),
                            "face_detected": 0
                        })

                processed_frame_vectors.append(frame_flat)
                current_frame_idx += 1

                # Complete recording when sequence length is reached
                if current_frame_idx >= seq_len:
                    is_recording = False
                    sample_tensor = np.array(processed_frame_vectors, dtype=np.float32)

                    # Validation
                    is_valid, err_msg = validate_face_sequence_tensor(sample_tensor, seq_len, TOTAL_FACE_FEATURES)

                    if is_valid:
                        meta = {
                            "label": sign_label,
                            "signer": signer_id,
                            "recording_id": recording_id,
                            "sequence_length": seq_len,
                            "feature_dimension": TOTAL_FACE_FEATURES,
                            "valid_frames": valid_face_frames_count
                        }
                        raw_saved, proc_saved = save_face_recording(raw_sequence_records, sample_tensor, meta)
                        print(f"\n[Saved] Face sequence successfully recorded ({seq_len} frames, {valid_face_frames_count} valid face frames):")
                        print(f"        Raw CSV:   {raw_saved}")
                        print(f"        Processed: {proc_saved}")
                        print("Ready for next sequence. Press [SPACE] to capture again or [Q] to quit.\n")

                        # Increment recording counter
                        recording_id += 1
                    else:
                        print(f"\n[Warning] Validation failed: {err_msg}. Sequence discarded.")

                    # Reset buffers
                    current_frame_idx = 0
                    raw_sequence_records = []
                    processed_frame_vectors = []
                    valid_face_frames_count = 0

            # 5. On-Screen Display (OSD) Overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (20, 20, 20), -1)

            face_status_str = "FACE DETECTED" if face_detected else "NO FACE IN VIEW"
            face_status_col = (0, 255, 0) if face_detected else (0, 0, 255)
            rec_status_str = f"RECORDING [{current_frame_idx}/{seq_len}]" if is_recording else "IDLE (Ready)"
            rec_status_col = (0, 0, 255) if is_recording else (0, 255, 0)

            line1 = f"Label: {sign_label} | Signer: {signer_id} | Rec ID: {recording_id:03d}"
            line2 = f"Status: {rec_status_str} | {face_status_str}"

            cv2.putText(frame, line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, line2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, rec_status_col, 2, cv2.LINE_AA)
            cv2.putText(frame, "[SPACE] Record | [R] Reset | [Q] Quit", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("LSE Face Tracking - Dataset Collector", frame)

            # 6. Keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACE
                if not is_recording:
                    is_recording = True
                    current_frame_idx = 0
                    raw_sequence_records = []
                    processed_frame_vectors = []
                    valid_face_frames_count = 0
                    print(f"[Started] Capturing {seq_len} face frames for '{sign_label}'...")
            elif key in (ord('r'), ord('R')):  # R: Reset
                is_recording = False
                current_frame_idx = 0
                raw_sequence_records = []
                processed_frame_vectors = []
                valid_face_frames_count = 0
                print("[Reset] Current sequence cancelled.")
            elif key in (ord('q'), ord('Q')):  # Q: Quit
                print("[Quit] Exiting face dataset collector...")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Face dataset collector closed safely.")


if __name__ == "__main__":
    main()
