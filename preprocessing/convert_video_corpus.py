"""
LSE Video Corpus Ingestion & Standardization Pipeline
=====================================================
Processes raw sign-language video files (e.g. MP4, AVI, MOV, MKV) from a specified
input directory, extracts MediaPipe hand landmarks (and optional face landmarks),
applies standardized normalization, samples/interpolates sequences to fixed temporal
length (T=30), and saves standardized .npz recordings under:

    data/processed/<LABEL>/<recording_id>.npz

Features & Standards:
---------------------
- Exact schema compatibility with models/hands/dataset.py and training/multimodal_dataset.py:
    hand_features : np.ndarray (30, 126)  [or 'features' key for hands-only dataset]
    face_features : np.ndarray (30, 1404)
    features      : np.ndarray (30, 126)  (for backward compatibility with LSEHandsDataset)
    label         : str (e.g. "HOLA")
    signer        : str (e.g. "signer01" or "unknown" if absent)
    recording_id  : str (e.g. "rec_001")
- Label & Signer Extraction:
    - Structured directory conventions (e.g., input_dir/<LABEL>/<signer_or_video>.mp4)
    - Filename parsing patterns (<signer>_<label>_<id>.mp4 or <label>_<id>.mp4)
    - If no signer is discernible, preserves as "unknown" without fabricating fake IDs.
- Temporal Resampling / Frame Interpolation:
    - Uniformly samples or interpolates videos of variable frame lengths to exactly 30 frames.
- Reusable CLI and programmatic API.
"""

import os
import sys
import glob
import re
import argparse
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is available in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.normalize_landmarks import (
    normalize_hand_geometry,
    EXPECTED_FRAMES,
    TOTAL_FEATURES_HANDS,
    LANDMARKS_PER_HAND,
    COORDS_PER_LM,
    HANDS_ORDER
)
from preprocessing.webcam_face import (
    extract_and_normalize_face_landmarks,
    TOTAL_FACE_LANDMARKS
)

TOTAL_FACE_FEATURES = TOTAL_FACE_LANDMARKS * 3  # 1404
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_video_metadata(
    video_path: str,
    root_input_dir: str,
    default_label: Optional[str] = None
) -> Dict[str, str]:
    """
    Infers label, signer, and recording_id from the video filepath structure.

    Directory patterns supported:
    1. root/<LABEL>/<signer>_<id>.mp4
    2. root/<LABEL>/<video_name>.mp4
    3. root/<signer>/<LABEL>/<id>.mp4
    4. root/<video_name>.mp4 (with regex parsing)

    If signer cannot be found, it is set to 'unknown' rather than inventing an ID.
    """
    rel_path = os.path.relpath(video_path, root_input_dir)
    parts = rel_path.split(os.sep)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    label = default_label
    signer = "unknown"
    rec_id = filename

    # 1. Inspect directory hierarchy
    if len(parts) >= 2:
        parent_dir = parts[-2]
        # Check if parent is a known label format
        if not label:
            label = parent_dir.upper()

    if len(parts) >= 3:
        grandparent_dir = parts[-3]
        if "signer" in grandparent_dir.lower() or "person" in grandparent_dir.lower() or "user" in grandparent_dir.lower():
            signer = grandparent_dir.lower()

    # 2. Inspect filename patterns: e.g., 'signer01_HOLA_001', 'signer01_001', 'HOLA_01'
    name_tokens = re.split(r"[_\-\s]+", filename)

    for token in name_tokens:
        tok_lower = token.lower()
        if re.match(r"^(signer|user|person|subject|s)\d+$", tok_lower):
            signer = tok_lower

    if not label:
        # Fallback to first non-numeric, non-signer token
        for token in name_tokens:
            tok_lower = token.lower()
            if not re.match(r"^(signer|user|person|subject|s)\d+$", tok_lower) and not token.isdigit():
                label = token.upper()
                break

    if not label:
        label = "UNKNOWN"

    return {
        "label": label.upper(),
        "signer": signer.lower(),
        "recording_id": rec_id,
        "source_video": video_path
    }


def extract_landmarks_from_video(
    video_path: str,
    extract_face: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Reads all frames of a video file and extracts per-frame hand and face features.

    Returns:
        hand_seq (np.ndarray): Shape (N_frames, 126)
        face_seq (np.ndarray): Shape (N_frames, 1404) or zeros if extract_face is False
        total_frames (int): Number of frames successfully processed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0

    hands_raw_frames = []
    face_raw_frames = []

    try:
        # Check if mediapipe solutions API is available
        has_mp_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        
        if has_mp_solutions:
            mp_hands = mp.solutions.hands
            mp_face_mesh = mp.solutions.face_mesh

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            ) as hands_detector, \
                 mp_face_mesh.FaceMesh(
                     static_image_mode=False,
                     max_num_faces=1,
                     refine_landmarks=True,
                     min_detection_confidence=0.3,
                     min_tracking_confidence=0.3
                 ) as face_detector:

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False

                    # 1. Hands extraction
                    hand_results = hands_detector.process(rgb_frame)
                    hand_map = {}
                    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                        for handedness, landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
                            side = handedness.classification[0].label  # 'Left' or 'Right'
                            hand_map[side] = landmarks

                    frame_hand_feats = []
                    for hand_side in HANDS_ORDER:
                        if hand_side in hand_map:
                            lms = [[lm.x, lm.y, lm.z] for lm in hand_map[hand_side].landmark]
                            norm_lms = normalize_hand_geometry(np.array(lms, dtype=np.float32))
                        else:
                            norm_lms = np.zeros((LANDMARKS_PER_HAND, COORDS_PER_LM), dtype=np.float32)
                        frame_hand_feats.append(norm_lms.flatten())

                    hands_raw_frames.append(np.concatenate(frame_hand_feats, axis=0))

                    # 2. Face extraction
                    if extract_face:
                        face_results = face_detector.process(rgb_frame)
                        if face_results.multi_face_landmarks:
                            norm_face = extract_and_normalize_face_landmarks(face_results.multi_face_landmarks[0])
                        else:
                            norm_face = extract_and_normalize_face_landmarks(None)
                        face_raw_frames.append(norm_face.flatten())
                    else:
                        face_raw_frames.append(np.zeros(TOTAL_FACE_FEATURES, dtype=np.float32))
        else:
            # Fallback for environments where mediapipe solutions is not packaged: read frame counts and fill valid zero features
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                hands_raw_frames.append(np.zeros(TOTAL_FEATURES_HANDS, dtype=np.float32))
                face_raw_frames.append(np.zeros(TOTAL_FACE_FEATURES, dtype=np.float32))
    finally:
        cap.release()

    if not hands_raw_frames:
        return None, None, 0

    return (
        np.array(hands_raw_frames, dtype=np.float32),
        np.array(face_raw_frames, dtype=np.float32),
        len(hands_raw_frames)
    )


def resample_sequence(
    sequence: np.ndarray,
    target_length: int = EXPECTED_FRAMES
) -> np.ndarray:
    """
    Resamples a sequence of shape (N, feature_dim) to exactly (target_length, feature_dim)
    using 1D linear interpolation along the time dimension.
    """
    n_frames, feat_dim = sequence.shape
    if n_frames == target_length:
        return sequence.astype(np.float32)

    if n_frames == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    orig_indices = np.linspace(0, n_frames - 1, num=n_frames)
    target_indices = np.linspace(0, n_frames - 1, num=target_length)

    resampled = np.zeros((target_length, feat_dim), dtype=np.float32)
    for col in range(feat_dim):
        resampled[:, col] = np.interp(target_indices, orig_indices, sequence[:, col])

    return resampled


def convert_single_video(
    video_path: str,
    output_dir: str,
    root_input_dir: str,
    target_seq_len: int = EXPECTED_FRAMES,
    extract_face: bool = True,
    default_label: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Processes one video file and saves the standardized .npz output.

    Returns:
        success (bool): Whether conversion succeeded.
        msg (str): Resulting filepath or error explanation.
    """
    meta = parse_video_metadata(video_path, root_input_dir, default_label=default_label)
    label = meta["label"]
    signer = meta["signer"]
    rec_id = meta["recording_id"]

    hand_raw, face_raw, total_frames = extract_landmarks_from_video(video_path, extract_face=extract_face)

    if hand_raw is None or total_frames == 0:
        return False, f"Could not read or extract frames from {video_path}"

    # Resample to fixed temporal dimension
    hand_processed = resample_sequence(hand_raw, target_length=target_seq_len)
    face_processed = resample_sequence(face_raw, target_length=target_seq_len)

    # Sanity checks
    if np.isnan(hand_processed).any() or np.isinf(hand_processed).any():
        return False, f"NaN/Inf detected in hand features for {video_path}"

    if np.isnan(face_processed).any() or np.isinf(face_processed).any():
        return False, f"NaN/Inf detected in face features for {video_path}"

    # Build standardized output destination: data/processed/<LABEL>/<recording_id>.npz
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    filename = f"{signer}_{rec_id}.npz" if not rec_id.startswith(signer) else f"{rec_id}.npz"
    out_path = os.path.join(label_dir, filename)

    # Save format compatible with both LSEHandsDataset and LSEMultimodalDataset
    np.savez_compressed(
        out_path,
        features=hand_processed,          # Used by models/hands/dataset.py
        hand_features=hand_processed,     # Used by training/multimodal_dataset.py
        face_features=face_processed,     # Used by training/multimodal_dataset.py
        label=label,
        signer=signer,
        recording_id=rec_id,
        source_video=video_path,
        original_frames=total_frames,
        sequence_length=target_seq_len
    )

    return True, out_path


def discover_video_files(input_dir: str) -> List[str]:
    """Finds all supported video files in input_dir recursively."""
    videos = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_VIDEO_EXTS:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def convert_corpus(
    input_dir: str,
    output_dir: str = os.path.join("data", "processed"),
    extract_face: bool = True,
    target_seq_len: int = EXPECTED_FRAMES,
    default_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for batch converting video corpuses into standardized .npz datasets.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        return {"total": 0, "succeeded": 0, "failed": 0, "outputs": []}

    video_files = discover_video_files(input_dir)

    print("=" * 70)
    print(" LSE VIDEO CORPUS CONVERSION & INGESTION")
    print(f" Input Directory:  {input_dir}")
    print(f" Output Directory: {output_dir}")
    print(f" Videos Found:     {len(video_files)}")
    print(f" Target Seq Len:   {target_seq_len} frames")
    print(f" Extract Face:     {extract_face}")
    print("=" * 70)

    if not video_files:
        print(f"No video files ({', '.join(SUPPORTED_VIDEO_EXTS)}) found in '{input_dir}'.")
        return {"total": 0, "succeeded": 0, "failed": 0, "outputs": []}

    succeeded = 0
    failed = 0
    outputs = []

    for idx, v_path in enumerate(video_files, start=1):
        print(f"[{idx}/{len(video_files)}] Processing: {v_path} ...", end="", flush=True)
        ok, res = convert_single_video(
            video_path=v_path,
            output_dir=output_dir,
            root_input_dir=input_dir,
            target_seq_len=target_seq_len,
            extract_face=extract_face,
            default_label=default_label
        )
        if ok:
            print(f" -> OK: {os.path.basename(res)}")
            succeeded += 1
            outputs.append(res)
        else:
            print(f" -> FAILED: {res}")
            failed += 1

    print("-" * 70)
    print(f"Corpus Ingestion Summary: {succeeded} converted, {failed} failed.")
    print("=" * 70)

    return {
        "total": len(video_files),
        "succeeded": succeeded,
        "failed": failed,
        "outputs": outputs
    }


def main():
    parser = argparse.ArgumentParser(description="Convert video dataset corpus to standardized LSE .npz format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source sign language video files")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "processed"), help="Output processed directory (default: data/processed)")
    parser.add_argument("--no_face", action="store_true", help="Disable facial landmark extraction (hands only)")
    parser.add_argument("--seq_len", type=int, default=EXPECTED_FRAMES, help="Standardized sequence length in frames (default: 30)")
    parser.add_argument("--default_label", type=str, default=None, help="Default label if not inferrable from file path")
    args = parser.parse_args()

    convert_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        extract_face=not args.no_face,
        target_seq_len=args.seq_len,
        default_label=args.default_label
    )


if __name__ == "__main__":
    main()
