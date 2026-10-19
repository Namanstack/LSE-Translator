"""
LSE Translator - Multimodal Live Web Demonstration
===================================================
Browser-based live demonstration interface for Spanish Sign Language (LSE).

Architecture:
1. Camera Input (Webcam Video Stream)
2. Real-Time MediaPipe Hand (2x21 LMs) & Facial Mesh (468 LMs) Tracking
3. Spatio-Temporal Normalization (Wrist-centering & scale-invariance)
4. Temporal Buffer (0/30 → 30/30 Sliding Window) with manual reset
5. Dual Recognition Section:
   - ML Model Recognition: Explicitly marked as "Training Pending"
   - Verification Prototype: Multi-Reference Template Matching (10-sign vocabulary library)
6. Multi-Reference Landmark Distance Comparison:
   - Evaluates Mean Absolute Error (MAE) against all loaded reference sample sequences
   - If distance <= threshold: Routes matched sign to LSETranslator (Spanish & English)
   - If distance > threshold: Displays NO MATCH with actual calculated distance and closest candidate
   - If hands are missing: Explicitly returns NO MATCH — HANDS NOT DETECTED and clears buffer
7. Reference Sample Recording & Validation:
   - Interactive UI to record live 30-frame landmark sequences for any vocabulary sign
   - Validates sequence length, landmark dimensions (30, 126), and continuous hand tracking
   - Safe storage preventing accidental overwrites

Usage:
    python app/web_demo.py [--port PORT]
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
import mediapipe as mp
import gradio as gr

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.normalize_landmarks import normalize_hand_geometry
from preprocessing.webcam_face import extract_and_normalize_face_landmarks
from recognition.realtime_inference import (
    MultimodalTemporalBuffer,
    process_hands_frame,
    process_face_frame
)
from translation.translator import LSETranslator, CANONICAL_VOCABULARY

# Default distance threshold for template matching
# Values below 0.50 indicate strong spatial trajectory alignment with the reference sequence
DEFAULT_MATCH_THRESHOLD = 0.50


def compute_sequence_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Computes legitimate Mean Absolute Error (MAE) distance between two (30, 126) sequences.
    Both sequences are assumed to be normalized (wrist-centered & scale-invariant).
    """
    if seq_a.shape != seq_b.shape:
        raise ValueError(f"Shape mismatch: {seq_a.shape} vs {seq_b.shape}")
    return float(np.mean(np.abs(seq_a - seq_b)))


class LiveMultimodalProcessor:
    """
    Stateful processor that handles incoming webcam frames from Gradio,
    maintains a rolling 30-frame temporal buffer, performs multi-reference
    template matching across the 10-sign vocabulary library, supports live
    reference sample recording, and connects to the translation layer.
    """

    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self.frame_count = 0
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.match_threshold = match_threshold
        self.init_error: Optional[str] = None

        # Temporal Buffer (T=30)
        self.buffer = MultimodalTemporalBuffer(sequence_length=30, hand_dim=126, face_dim=1404)
        self.translator = LSETranslator()

        # Multi-Reference Library: Dict[label, List[Dict[str, Any]]]
        self.reference_library: Dict[str, List[Dict[str, Any]]] = {
            label: [] for label in CANONICAL_VOCABULARY
        }
        # Backward compatibility mapping: label -> primary np.ndarray
        self.reference_samples: Dict[str, np.ndarray] = {}

        # Live face tracking & expression diagnostic telemetry
        self.prev_face_coords: Optional[np.ndarray] = None
        self.prev_face_feat: Optional[np.ndarray] = None

        # Load all valid reference samples from data/processed/
        self.load_reference_library()

        # Initialize MediaPipe Solutions
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands") and hasattr(mp.solutions, "face_mesh"):
                self.mp_hands = mp.solutions.hands
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_styles = mp.solutions.drawing_styles

                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.face_detector = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.is_ready = True
            else:
                self.is_ready = False
                self.init_error = "MediaPipe Solutions API not present in current Python environment."
                self.hands_detector = None
                self.face_detector = None
        except Exception as e:
            self.is_ready = False
            self.init_error = f"MediaPipe Initialization Error: {e}"
            self.hands_detector = None
            self.face_detector = None

    def load_reference_library(self) -> Dict[str, int]:
        """
        Scans data/processed/<LABEL>/ for valid reference sample .npz files.
        Validates dimensions (30, 126), non-zero, and valid numerical values.
        Populates self.reference_library and self.reference_samples.
        Returns a dictionary mapping label to count of valid samples loaded.
        """
        self.reference_library = {label: [] for label in CANONICAL_VOCABULARY}
        self.reference_samples = {}
        sample_counts = {label: 0 for label in CANONICAL_VOCABULARY}

        processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
        if not os.path.exists(processed_dir):
            return sample_counts

        # Check all canonical vocabulary labels plus any existing subdirectories
        candidate_labels = list(CANONICAL_VOCABULARY)
        for entry in os.listdir(processed_dir):
            entry_path = os.path.join(processed_dir, entry)
            if os.path.isdir(entry_path) and entry not in candidate_labels:
                candidate_labels.append(entry)

        for label in candidate_labels:
            label_dir = os.path.join(processed_dir, label)
            if not os.path.isdir(label_dir):
                continue

            # Look for hand sequence .npz files (ignore face_* files)
            npz_files = sorted([
                f for f in os.listdir(label_dir)
                if f.endswith(".npz") and not f.startswith("face_")
            ])

            valid_samples: List[Dict[str, Any]] = []
            for fname in npz_files:
                fpath = os.path.join(label_dir, fname)
                try:
                    data = np.load(fpath, allow_pickle=True)
                    if "features" not in data:
                        continue
                    feats = data["features"].astype(np.float32)

                    # Validation: exactly (30, 126), not all zeros, no NaN/Inf
                    if feats.shape != (30, 126):
                        continue
                    if np.isnan(feats).any() or np.isinf(feats).any():
                        continue
                    if np.all(feats == 0):
                        continue

                    signer_id = str(data["signer"]) if "signer" in data else "signer01"
                    recording_id = str(data["recording_id"]) if "recording_id" in data else "001"

                    valid_samples.append({
                        "filename": fname,
                        "filepath": fpath,
                        "features": feats,
                        "signer": signer_id,
                        "recording_id": recording_id
                    })

                    # Maintain backward compatibility for primary sample
                    if label not in self.reference_samples:
                        self.reference_samples[label] = feats

                except Exception as exc:
                    print(f"[Warning] Failed loading reference {fpath}: {exc}")

            self.reference_library[label] = valid_samples
            sample_counts[label] = len(valid_samples)

        return sample_counts

    def get_active_vocabulary_str(self) -> str:
        """Returns concise summary of loaded reference vocabulary."""
        active = [lbl for lbl in CANONICAL_VOCABULARY if len(self.reference_library.get(lbl, [])) > 0]
        if not active:
            return "No reference samples active (0/10)"
        return f"{len(active)}/10 Active ({', '.join(active)})"

    def get_library_status_markdown(self) -> str:
        """
        Generates a transparent Markdown table showing status of all 10 vocabulary signs.
        Explicitly distinguishes between recorded reference samples and pending recordings.
        """
        total_recorded = sum(1 for lbl in CANONICAL_VOCABULARY if len(self.reference_library.get(lbl, [])) > 0)
        total_vocab = len(CANONICAL_VOCABULARY)

        rows = [
            f"### Reference Vocabulary Status: **{total_recorded}/{total_vocab} Signs Recorded**",
            "",
            "| # | Sign Label | Meaning (ES / EN) | Reference Status | Samples | Recorded File(s) |",
            "| :-: | :--- | :--- | :--- | :-: | :--- |"
        ]

        for idx, label in enumerate(CANONICAL_VOCABULARY, 1):
            samples = self.reference_library.get(label, [])
            es_meaning = self.translator.translate_label(label, "es")
            en_meaning = self.translator.translate_label(label, "en")
            meaning_str = f"{es_meaning} / {en_meaning}"

            if len(samples) > 0:
                status_badge = "**RECORDED**"
                files_str = ", ".join(f"`{s['filename']}`" for s in samples[:2])
                if len(samples) > 2:
                    files_str += f" *(+{len(samples)-2} more)*"
            else:
                status_badge = "*Needs Recording*"
                files_str = "*(None - Recording Required)*"

            rows.append(f"| {idx} | **{label}** | {meaning_str} | {status_badge} | {len(samples)} | {files_str} |")

        rows.append("")
        rows.append("*Integrity notice: Template matching compares live hand landmarks against verified reference samples only. Signs marked 'Needs Recording' are not guessed or fabricated.*")
        return "\n".join(rows)

    def save_current_buffer_as_reference(
        self, label: str, signer_id: str = "signer01"
    ) -> Tuple[str, str, str]:
        """
        Validates the current 30-frame live buffer and saves it as a new reference
        sample under data/processed/<label>/<signer_id>_<idx>.npz.
        Guarantees that existing reference samples (like HOLA/signer01_001.npz)
        are NEVER overwritten.
        Returns:
            feedback_message, updated_library_status_markdown, updated_active_vocab_str
        """
        clean_label = label.strip().upper() if label else ""
        if not clean_label:
            return (
                "[ERROR] Please select or enter a valid sign label.",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        clean_signer = signer_id.strip().lower() if signer_id else "signer01"

        if not self.buffer.is_full() or self.buffer.current_length < 30:
            return (
                f"[ERROR] Buffer has only {self.buffer.current_length}/30 frames. "
                "Please perform the sign in front of the camera until the buffer reaches 30/30 frames.",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        hand_seq = np.array(self.buffer.hand_buffer, dtype=np.float32)  # (30, 126)
        if hand_seq.shape != (30, 126):
            return (
                f"[ERROR] Invalid hand feature shape {hand_seq.shape}. Expected (30, 126).",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        # Validate continuous hand tracking (no all-zero frames)
        zero_frames = [i for i, f in enumerate(hand_seq) if np.all(f == 0)]
        if len(zero_frames) > 0:
            return (
                f"[ERROR] Hand landmarks were missing in {len(zero_frames)}/30 frames (e.g., frame {zero_frames[0]+1}). "
                "Both hands or the active hand must remain clearly visible in camera throughout all 30 frames.",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        # Numerical integrity check
        if np.isnan(hand_seq).any() or np.isinf(hand_seq).any():
            return (
                "[ERROR] Hand features contain invalid numerical values (NaN/Inf).",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        # Output directory: data/processed/<LABEL>/
        label_dir = os.path.join(PROJECT_ROOT, "data", "processed", clean_label)
        os.makedirs(label_dir, exist_ok=True)

        # Find next available recording ID to strictly prevent any overwrite
        idx = 1
        while True:
            target_filename = f"{clean_signer}_{idx:03d}.npz"
            target_path = os.path.join(label_dir, target_filename)
            if not os.path.exists(target_path):
                break
            idx += 1

        try:
            # Save in standard format identical to existing HOLA reference
            np.savez(
                target_path,
                features=hand_seq,
                label=clean_label,
                signer=clean_signer,
                recording_id=f"{idx:03d}"
            )
        except Exception as exc:
            return (
                f"[ERROR] Error saving reference file: {exc}",
                self.get_library_status_markdown(),
                self.get_active_vocabulary_str()
            )

        # Reload reference library dynamically so it's immediately active
        self.load_reference_library()

        msg = (
            f"[SUCCESS] Successfully saved reference sample for '{clean_label}'!\n"
            f"Path: data/processed/{clean_label}/{target_filename}\n"
            f"Shape: {hand_seq.shape} (30 frames x 126 normalized features)\n"
            f"Reference library reloaded: '{clean_label}' is now active for template matching."
        )
        return msg, self.get_library_status_markdown(), self.get_active_vocabulary_str()

    def reset_buffer(self) -> Tuple[str, str, str, str]:
        """
        Clears the 30-frame temporal buffer and resets comparison telemetry.
        """
        self.buffer.clear()
        self.prev_face_coords = None
        self.prev_face_feat = None
        return (
            "0/30 frames (Reset)",
            "Awaiting 30 frames",
            "...",
            "..."
        )

    def process_frame(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, str, str, str, str, str, str, str, str, str, str, str]:
        """
        Processes a single incoming webcam frame.
        Returns:
            annotated_img, fps, hands_cnt, faces_cnt, hand_status, face_status,
            buffer_status, pipeline_status, ml_status, cmp_status, trans_es, trans_en
        """
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA OFFLINE / AWAITING FEED", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            return (
                blank, "0.0", "0", "0",
                "Awaiting Feed", "Awaiting Feed", "0/30 frames",
                "Idle", "Training Pending", "Awaiting 30 frames", "...", "..."
            )

        self.frame_count += 1
        curr_time = time.perf_counter()
        dt = curr_time - self.prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps
        self.prev_time = curr_time

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)
        else:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

        annotated = frame.copy()
        num_hands = 0
        num_faces = 0

        if not self.is_ready:
            err_msg = self.init_error or "MediaPipe not initialized"
            cv2.putText(annotated, "ERROR: MEDIAPIPE UNAVAILABLE", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            return (
                annotated, f"{self.fps:.1f}", "0", "0",
                "Error", "Error", "0/30 frames",
                f"Error: {err_msg}", "Training Pending", "Error", "...", "..."
            )

        # Process with MediaPipe
        rgb_frame = frame.copy()
        rgb_frame.flags.writeable = False

        hand_res = self.hands_detector.process(rgb_frame)
        face_res = self.face_detector.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # Extract landmark detection results independently for CURRENT frame
        current_face_landmarks = None
        face_landmarks_updated = False
        face_lm_count = 0
        mae_delta = 0.0
        mouth_open_ratio = 0.0
        mouth_width_ratio = 0.0
        eyebrow_ratio = 0.0
        feat_changed = False

        if face_res and face_res.multi_face_landmarks and len(face_res.multi_face_landmarks) > 0:
            num_faces = len(face_res.multi_face_landmarks)
            current_face_landmarks = face_res.multi_face_landmarks[0]
            face_landmarks_updated = True
            face_lm_count = len(current_face_landmarks.landmark)

            # Extract current frame coordinates shape (468, 3)
            raw_face_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in current_face_landmarks.landmark[:468]],
                dtype=np.float32
            )

            # Compute consecutive frame coordinate difference (ΔMAE)
            if self.prev_face_coords is not None and self.prev_face_coords.shape == raw_face_coords.shape:
                mae_delta = float(np.mean(np.abs(raw_face_coords - self.prev_face_coords)))
            else:
                mae_delta = 0.0
            self.prev_face_coords = raw_face_coords.copy()

            # Dynamic scale normalization (Forehead center 10 to Chin 152)
            face_scale = float(np.linalg.norm(raw_face_coords[10] - raw_face_coords[152]))
            if face_scale < 1e-4:
                face_scale = 1.0

            # Scale-normalized mouth opening distance (Inner upper lip 13 to inner lower lip 14)
            mouth_open_dist = float(np.linalg.norm(raw_face_coords[13] - raw_face_coords[14]))
            mouth_open_ratio = mouth_open_dist / face_scale

            # Scale-normalized mouth width / smile distance (Left corner 61 to right corner 291)
            mouth_width_dist = float(np.linalg.norm(raw_face_coords[61] - raw_face_coords[291]))
            mouth_width_ratio = mouth_width_dist / face_scale

            # Scale-normalized eyebrow distance (Left brow 105 to eye 159, Right brow 334 to eye 386)
            left_brow_dist = float(np.linalg.norm(raw_face_coords[105] - raw_face_coords[159]))
            right_brow_dist = float(np.linalg.norm(raw_face_coords[334] - raw_face_coords[386]))
            eyebrow_dist = 0.5 * (left_brow_dist + right_brow_dist)
            eyebrow_ratio = eyebrow_dist / face_scale
        else:
            num_faces = 0
            current_face_landmarks = None
            face_landmarks_updated = False
            self.prev_face_coords = None
            self.prev_face_feat = None

        if hand_res and hand_res.multi_hand_landmarks:
            num_hands = len(hand_res.multi_hand_landmarks)
        else:
            num_hands = 0

        # 1. Face Landmarks Visualization (Drawn first so hand landmarks appear in foreground if occluding)
        if face_landmarks_updated and current_face_landmarks is not None:
            # A. Full 3D Face Mesh Tesselation (all 2,556 surface triangles across cheeks, forehead, mouth)
            self.mp_drawing.draw_landmarks(
                annotated,
                current_face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style()
            )
            # B. Face Mesh Contours (prominent outlines for eyes, eyebrows, lips, face oval)
            self.mp_drawing.draw_landmarks(
                annotated,
                current_face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
            )
            # C. Key expression landmark markers for unmistakable expression visual tracking
            # 13, 14: inner lips (yellow) | 61, 291: mouth corners (cyan) | 105, 334: brows (lime) | 1: nose tip (red)
            h_img, w_img, _ = annotated.shape
            expr_points = [
                (13, (0, 255, 255)),
                (14, (0, 255, 255)),
                (61, (255, 200, 0)),
                (291, (255, 200, 0)),
                (105, (0, 255, 0)),
                (334, (0, 255, 0)),
                (1, (0, 0, 255))
            ]
            for l_idx, col in expr_points:
                if l_idx < len(current_face_landmarks.landmark):
                    lm = current_face_landmarks.landmark[l_idx]
                    px, py = int(lm.x * w_img), int(lm.y * h_img)
                    cv2.circle(annotated, (px, py), 3, col, -1)

        # 2. Hand Landmarks Visualization (Foreground overlay)
        if num_hands > 0:
            for hl in hand_res.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hl,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

        # 3. Extract Spatio-Temporal Normalized Vectors
        hand_feat, hands_detected = process_hands_frame(hand_res)
        face_feat, face_detected = process_face_frame(face_res)

        # Track whether current face feature array changed
        if face_detected:
            if self.prev_face_feat is not None and self.prev_face_feat.shape == face_feat.shape:
                feat_diff = float(np.mean(np.abs(face_feat - self.prev_face_feat)))
                feat_changed = feat_diff > 1e-5
            else:
                feat_changed = True
            self.prev_face_feat = face_feat.copy()
        else:
            feat_changed = False
            self.prev_face_feat = None

        # 4. Update Temporal Buffer (0/30 -> 30/30)
        self.buffer.append(hand_feat, face_feat)
        buf_len = self.buffer.current_length
        buffer_str = f"{buf_len}/30 frames {'(Full)' if self.buffer.is_full() else '(Collecting)'}"

        # 5. Modality Alignment Status & Diagnostics
        hand_status = f"Detected ({num_hands} hand{'s' if num_hands > 1 else ''})" if num_hands > 0 else "No Hands Detected"
        if face_landmarks_updated:
            feat_status_str = "Changed" if feat_changed else "Stable"
            face_status = (
                f"Detected ({num_faces} face | {face_lm_count} landmarks updated) | "
                f"ΔMAE: {mae_delta:.4f} | Feat: {feat_status_str} | "
                f"Mouth: {mouth_open_ratio:.3f} | Brow: {eyebrow_ratio:.3f}"
            )
        else:
            face_status = "FACE NOT DETECTED"

        if num_hands > 0 and face_landmarks_updated:
            pipeline_status = "Active - Dual Modality (Hands + Face Synchronized)"
        elif num_hands > 0:
            pipeline_status = "Active - Hand Modality Only (Face Occluded / Not Detected)"
        elif face_landmarks_updated:
            pipeline_status = "Active - Facial Modality Only"
        else:
            pipeline_status = "Active - Searching for Signer"

        # 6. ML Recognition Status (Strictly guarded)
        ml_status = "Training Pending (Real multi-signer dataset required)"

        # 7. Verification Prototype — Multi-Reference Template Comparison
        cmp_status = "Awaiting 30 frames"
        trans_es = "..."
        trans_en = "..."
        best_label: Optional[str] = None
        best_distance = float("inf")

        if self.buffer.is_full():
            curr_hand_seq = np.array(self.buffer.hand_buffer, dtype=np.float32)  # (30, 126)

            # Detect any frame where hand landmarks are all zeros (missing hands)
            if np.any(np.all(curr_hand_seq == 0, axis=1)):
                # Invalidate comparison when hands are absent
                cmp_status = "NO MATCH — HANDS NOT DETECTED"
                # Clear buffer to avoid stale data influencing future matches
                self.buffer.clear()
            else:
                # Find best matching candidate across all loaded reference samples in the library
                best_ref_file = ""
                total_samples = 0

                # Check all canonical vocabulary labels
                for lbl in CANONICAL_VOCABULARY:
                    samples = self.reference_library.get(lbl, [])
                    for sample in samples:
                        total_samples += 1
                        dist = compute_sequence_distance(curr_hand_seq, sample["features"])
                        if dist < best_distance:
                            best_distance = dist
                            best_label = lbl
                            best_ref_file = sample["filename"]

                # Also check any extra labels outside canonical list if present
                for lbl, samples in self.reference_library.items():
                    if lbl not in CANONICAL_VOCABULARY:
                        for sample in samples:
                            total_samples += 1
                            dist = compute_sequence_distance(curr_hand_seq, sample["features"])
                            if dist < best_distance:
                                best_distance = dist
                                best_label = lbl
                                best_ref_file = sample["filename"]

                if total_samples == 0:
                    cmp_status = "NO MATCH — No reference templates loaded in library"
                elif best_distance <= self.match_threshold:
                    cmp_status = f"MATCH: {best_label} (Distance: {best_distance:.3f} <= {self.match_threshold:.2f} | Ref: {best_ref_file})"
                    trans_es = self.translator.translate_label(best_label, "es")
                    trans_en = self.translator.translate_label(best_label, "en")
                else:
                    cmp_status = f"NO MATCH (Closest: {best_label} @ Distance: {best_distance:.3f} > {self.match_threshold:.2f})"

        # 8. On-screen HUD overlay & current-frame diagnostics
        h, w, _ = annotated.shape
        cv2.rectangle(annotated, (0, 0), (w, 60), (20, 24, 30), -1)
        if face_landmarks_updated:
            face_diag_hud = f"Faces: {num_faces} (LMs: {face_lm_count} | ΔMAE: {mae_delta:.4f} | M: {mouth_open_ratio:.2f} B: {eyebrow_ratio:.2f})"
        else:
            face_diag_hud = "Faces: 0 (FACE NOT DETECTED)"
        hud_line1 = f"FPS: {self.fps:.1f} | Hands: {num_hands} | {face_diag_hud} | Buffer: {buf_len}/30"
        hud_color = (0, 255, 160) if (num_hands > 0 or face_landmarks_updated) else (200, 200, 200)
        cv2.putText(annotated, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_color, 2, cv2.LINE_AA)

        if not face_landmarks_updated:
            cv2.putText(annotated, "[FACE NOT DETECTED]", (w - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2, cv2.LINE_AA)

        if cmp_status.startswith("MATCH:") and best_label:
            hud_line2 = f"PROTOTYPE MATCH: {best_label} -> ES: {trans_es} | EN: {trans_en} (MAE: {best_distance:.2f})"
            cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        elif cmp_status.startswith("MATCH ("):
            hud_line2 = f"PROTOTYPE MATCH: HOLA -> ES: {trans_es} | EN: {trans_en}"
            cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            hud_line2 = f"STATUS: {pipeline_status}"
            cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        return (
            annotated,
            f"{self.fps:.1f}",
            str(num_hands),
            str(num_faces),
            hand_status,
            face_status,
            buffer_str,
            pipeline_status,
            ml_status,
            cmp_status,
            trans_es,
            trans_en
        )


def build_demo_app(match_threshold: float = DEFAULT_MATCH_THRESHOLD):
    """
    Constructs the Gradio UI conforming to scientific integrity guidelines:
    - Clear distinction between "ML Model — Training Pending" and "Verification Prototype — Multi-Reference Matching"
    - Legitimate numerical distance calculation (MAE) against 10-sign reference library
    - Working Reset Sequence control
    - Live reference sequence recording with dimension validation and overwrite protection
    - Transparent vocabulary status matrix
    """
    processor = LiveMultimodalProcessor(match_threshold=match_threshold)

    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        max-width: 1250px !important;
        margin: auto;
    }
    .main-title {
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 16px;
    }
    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .badge-prototype {
        background-color: #0284c7;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-pending {
        background-color: #d97706;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pipeline-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .integrity-note {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 6px;
    }
    """

    with gr.Blocks(title="LSE Translator - Faculty Demo") as demo:
        gr.HTML(f"""
        <style>{custom_css}</style>
        <div class="main-title">
            <h1 style="font-size: 2.2rem; font-weight: 700; margin: 0;">LSE Translator</h1>
        </div>
        <div class="subtitle">
            Multimodal Spanish Sign Language Recognition & Translation Pipeline
        </div>
        <div class="badge-bar">
            <span class="badge-prototype">VERIFICATION PROTOTYPE: MULTI-REFERENCE TEMPLATE MATCHING</span>
            <span class="badge-pending">ML MODEL TRAINING PENDING REAL DATASET</span>
        </div>
        """)

        with gr.Row():
            # Left Column: Webcam and Detection Feed
            with gr.Column(scale=3):
                gr.Markdown("### 📹 Live Camera & Landmark Stream")
                camera_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam Feed"
                )
                annotated_output = gr.Image(
                    type="numpy",
                    label="Annotated Landmark Stream (Hands + Face Mesh)"
                )
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Sequence Buffer (0/30)", variant="secondary")

            # Right Column: Telemetry, Recognition Modes & Translation
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Telemetry & Temporal Buffer")
                with gr.Group():
                    with gr.Row():
                        fps_stat = gr.Textbox(label="Live FPS", value="0.0", interactive=False)
                        hands_stat = gr.Textbox(label="Hands Detected", value="0", interactive=False)
                        faces_stat = gr.Textbox(label="Faces Detected", value="0", interactive=False)

                    with gr.Row():
                        hand_lm_stat = gr.Textbox(label="Hand Landmark Status", value="Awaiting Feed", interactive=False)
                        face_lm_stat = gr.Textbox(label="Face Landmark Status", value="Awaiting Feed", interactive=False)

                    with gr.Row():
                        buffer_stat = gr.Textbox(label="Temporal Buffer", value="0/30 frames", interactive=False)
                        pipe_stat = gr.Textbox(label="Multimodal Alignment", value="Idle", interactive=False)

                gr.Markdown("### 🧠 Recognition Modes")
                with gr.Group():
                    # Mode 1: ML Model
                    ml_stat = gr.Textbox(
                        label="Recognition Mode: ML Model",
                        value="Training Pending (Real multi-signer dataset required)",
                        interactive=False
                    )
                    gr.HTML("<div class='integrity-note'>*Neural network weights are pending collection of the multi-signer LSE corpus. No fake ML predictions are generated.</div>")

                    # Mode 2: Verification Prototype Template Matcher
                    gr.Markdown("---")
                    gr.Markdown("**Verification Prototype — Multi-Reference Template Matching**")
                    active_vocab_box = gr.Textbox(
                        label="Active Reference Vocabulary",
                        value=processor.get_active_vocabulary_str(),
                        interactive=False
                    )
                    cmp_stat = gr.Textbox(
                        label="Template Match Result & Distance",
                        value="Awaiting 30 frames",
                        interactive=False
                    )
                    gr.HTML(f"<div class='integrity-note'>*Compares live 30-frame hand trajectory against active reference templates. Match threshold: &le; {match_threshold:.2f} (MAE).</div>")

                gr.Markdown("### 🌐 Prototype Reference Translation")
                with gr.Group():
                    with gr.Row():
                        trans_es_box = gr.Textbox(label="Spanish (ES)", value="...", interactive=False)
                        trans_en_box = gr.Textbox(label="English (EN)", value="...", interactive=False)
                    gr.HTML("<div class='integrity-note'>*Translation layer maps the matched reference token through LSETranslator. Produced via template comparison, not trained ML weights.</div>")

                gr.Markdown("### ⚙️ Pipeline Flow")
                gr.HTML("""
                <div class="pipeline-box">
                    <strong>1. Camera Feed (Webcam Stream)</strong><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    <strong>2. Hand (126) + Face (1404) Preprocessing</strong><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    <strong>3. Temporal Buffer (0/30 → 30/30 Sliding Window)</strong><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    <strong>4. Multi-Reference Template Matching (10-Sign Library)</strong><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    <strong>5. LSETranslator (Spanish & English Text Generation)</strong>
                </div>
                """)

        # Reference Library Management & Live Recording Section
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                vocab_status_box = gr.Markdown(value=processor.get_library_status_markdown())
                refresh_lib_btn = gr.Button("🔄 Refresh Reference Library", variant="secondary", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Record New Reference Sign Sample")
                gr.Markdown(
                    "**How to record a reference sign:**<br>"
                    "1. Select the target sign and signer identifier below.<br>"
                    "2. Perform the sign in front of the camera until buffer reads **30/30 frames (Full)**.<br>"
                    "3. Click **Save Live Sequence as Reference Sample**.<br>"
                    "4. The system validates landmarks (dimensions, continuity) and saves to `data/processed/<SIGN>/` without overwriting existing files."
                )
                with gr.Row():
                    record_label_dropdown = gr.Dropdown(
                        choices=CANONICAL_VOCABULARY,
                        value="GRACIAS",
                        label="Target Sign Label"
                    )
                    record_signer_input = gr.Textbox(
                        value="signer01",
                        label="Signer Identifier",
                        placeholder="signer01"
                    )
                save_ref_btn = gr.Button(
                    "💾 Save Current 30-Frame Sequence as Reference Sample",
                    variant="primary"
                )
                record_status_box = gr.Textbox(
                    label="Recording & Validation Feedback",
                    value="Ready to record. Perform gesture in camera feed to fill 30-frame buffer.",
                    interactive=False,
                    lines=4
                )

        # Stream update event
        camera_input.stream(
            fn=processor.process_frame,
            inputs=[camera_input],
            outputs=[
                annotated_output,
                fps_stat,
                hands_stat,
                faces_stat,
                hand_lm_stat,
                face_lm_stat,
                buffer_stat,
                pipe_stat,
                ml_stat,
                cmp_stat,
                trans_es_box,
                trans_en_box
            ]
        )

        # Reset button event
        reset_btn.click(
            fn=processor.reset_buffer,
            inputs=[],
            outputs=[
                buffer_stat,
                cmp_stat,
                trans_es_box,
                trans_en_box
            ]
        )

        # Wire recording and library refresh events
        save_ref_btn.click(
            fn=processor.save_current_buffer_as_reference,
            inputs=[record_label_dropdown, record_signer_input],
            outputs=[record_status_box, vocab_status_box, active_vocab_box]
        )

        refresh_lib_btn.click(
            fn=lambda: (
                processor.load_reference_library(),
                processor.get_library_status_markdown(),
                processor.get_active_vocabulary_str()
            )[1:],
            inputs=[],
            outputs=[vocab_status_box, active_vocab_box]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LSE Translator Live Web Demonstration.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")), help="Server port (default: 7860)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="Template match distance threshold (default: 0.50)")
    args = parser.parse_args()

    print("=" * 70)
    print(" LSE TRANSLATOR - LIVE WEB DEMONSTRATION")
    print(" Multimodal Sign Language Recognition & Translation Pipeline")
    print(f" Port: {args.port} | Match Threshold: {args.threshold:.2f}")
    print(" Recognition Mode: 'ML Model — Training Pending'")
    print(" Verification Prototype: 'Multi-Reference Template Matching (10-Sign Library)'")
    print("=" * 70)

    app = build_demo_app(match_threshold=args.threshold)
    
    # Launch with specified port; if port is busy, Gradio automatically finds next open port
    try:
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False)
    except OSError as err:
        print(f"[Notice] Port {args.port} occupied ({err}). Searching for next open port...")
        app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
