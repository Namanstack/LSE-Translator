"""
LSE Real-Time Multimodal Inference & Sliding Buffer Engine - Milestone 4

Connects:
1. MediaPipe Hands + MediaPipe Face Mesh
2. Exact translation-invariant and scale-invariant normalizations from Milestones 1-3
3. Sliding temporal buffer (T=30 frames)
4. Fallback strategy for missing modalities (zero-vector imputation + modality gating)
5. End-to-end MultimodalSignModel with clean checkpoint loading
6. Untrained / Demo safeguard (explicitly displays 'MODEL STATUS: NOT TRAINED - INFERENCE DISABLED' when checkpoint is missing)
7. Decoupled LSE Translation Layer for Spanish & English text output
"""

import os
import sys
import time
import argparse
from collections import deque
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch

# Ensure project root is available
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig, DEFAULT_CONFIG
from models.multimodal_model import MultimodalSignModel
from translation.translator import LSETranslator
from preprocessing.normalize_landmarks import normalize_hand_geometry
from preprocessing.webcam_face import extract_and_normalize_face_landmarks


# -----------------------------------------------------------------------------
# Modality Landmark Extraction & Preprocessing Helpers
# -----------------------------------------------------------------------------

def process_hands_frame(hand_results) -> Tuple[np.ndarray, bool]:
    """
    Extracts and normalizes hand landmarks for a single video frame.
    Matches exact preprocessing logic from normalize_landmarks.py.

    Returns:
        hand_features (np.ndarray): Shape (126,) float32 [Left 63, Right 63].
        hands_detected (bool): True if at least one hand is detected.
    """
    hand_order = ["Left", "Right"]
    detected_map = {}

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for handedness, landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            detected_map[label] = landmarks

    hands_detected = len(detected_map) > 0
    frame_features = []

    for hand_side in hand_order:
        if hand_side in detected_map:
            lms_raw = [[lm.x, lm.y, lm.z] for lm in detected_map[hand_side].landmark]
            lms_arr = np.array(lms_raw, dtype=np.float32)
            norm_lms = normalize_hand_geometry(lms_arr)
        else:
            # Missing hand fallback: zeros of shape (21, 3)
            norm_lms = np.zeros((21, 3), dtype=np.float32)

        frame_features.append(norm_lms.flatten())

    return np.concatenate(frame_features, axis=0).astype(np.float32), hands_detected


def process_face_frame(face_results) -> Tuple[np.ndarray, bool]:
    """
    Extracts and normalizes facial landmarks for a single video frame.
    Matches exact preprocessing logic from webcam_face.py.

    Returns:
        face_features (np.ndarray): Shape (1404,) float32.
        face_detected (bool): True if face is detected.
    """
    if face_results.multi_face_landmarks:
        primary_face = face_results.multi_face_landmarks[0]
        norm_face = extract_and_normalize_face_landmarks(primary_face)
        return norm_face.flatten().astype(np.float32), True
    else:
        # Missing face fallback: zero vector of shape (1404,)
        norm_face = extract_and_normalize_face_landmarks(None)
        return norm_face.flatten().astype(np.float32), False


# -----------------------------------------------------------------------------
# Rolling Multimodal Temporal Buffer
# -----------------------------------------------------------------------------

class MultimodalTemporalBuffer:
    """
    Maintains synchronized rolling FIFO queues for hand and face landmark frames.
    """

    def __init__(self, sequence_length: int = 30, hand_dim: int = 126, face_dim: int = 1404):
        self.sequence_length = sequence_length
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        self.hand_buffer = deque(maxlen=sequence_length)
        self.face_buffer = deque(maxlen=sequence_length)

    def append(self, hand_feat: np.ndarray, face_feat: np.ndarray):
        """Appends one frame's extracted feature vectors."""
        self.hand_buffer.append(hand_feat)
        self.face_buffer.append(face_feat)

    def is_full(self) -> bool:
        """Returns True only when target sequence length has been collected."""
        return len(self.hand_buffer) == self.sequence_length

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns batched PyTorch tensors:
            hand_tensor: (1, sequence_length, 126)
            face_tensor: (1, sequence_length, 1404)
        """
        hand_arr = np.array(self.hand_buffer, dtype=np.float32)  # (T, 126)
        face_arr = np.array(self.face_buffer, dtype=np.float32)  # (T, 1404)

        hand_tensor = torch.tensor(hand_arr).unsqueeze(0)  # (1, T, 126)
        face_tensor = torch.tensor(face_arr).unsqueeze(0)  # (1, T, 1404)
        return hand_tensor, face_tensor

    def clear(self):
        """Clears all stored frames."""
        self.hand_buffer.clear()
        self.face_buffer.clear()

    @property
    def current_length(self) -> int:
        return len(self.hand_buffer)


# -----------------------------------------------------------------------------
# Real-Time Inference Controller
# -----------------------------------------------------------------------------

class RealtimeInferenceEngine:
    """
    Coordinates model loading, rolling buffer updates, modality checks, and UI rendering.
    """

    def __init__(
        self,
        config: ModelConfig = DEFAULT_CONFIG,
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.6
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.buffer = MultimodalTemporalBuffer(
            sequence_length=config.sequence_length,
            hand_dim=config.hand_feature_dim,
            face_dim=config.face_feature_dim
        )
        self.translator = LSETranslator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        self.model = MultimodalSignModel(config=config).to(self.device)
        self.model.eval()

        # Checkpoint validation
        self.is_checkpoint_loaded = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_checkpoint_loaded = True
                print(f"[Loaded] Checkpoint successfully restored from: {checkpoint_path}")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint: {e}. Running in untrained mode.")
                self.is_checkpoint_loaded = False
        else:
            print("[Info] No valid checkpoint specified/found. Running in UNTRAINED/DEMO mode.")
            self.is_checkpoint_loaded = False

    def process_frame(
        self,
        hand_results,
        face_results
    ) -> Dict[str, Any]:
        """
        Extracts features, updates buffer, and executes forward pass if buffer is full.
        """
        hand_feat, hands_detected = process_hands_frame(hand_results)
        face_feat, face_detected = process_face_frame(face_results)

        # Update sliding temporal buffer
        self.buffer.append(hand_feat, face_feat)

        result = {
            "hands_detected": hands_detected,
            "face_detected": face_detected,
            "buffer_status": f"{self.buffer.current_length}/{self.config.sequence_length}",
            "buffer_full": self.buffer.is_full(),
            "is_trained": self.is_checkpoint_loaded,
            "predicted_sign": None,
            "translation_es": None,
            "translation_en": None,
            "confidence": 0.0,
            "fusion_weights": None
        }

        # If buffer is full, run model forward pass
        if self.buffer.is_full():
            hand_tensor, face_tensor = self.buffer.get_tensors()
            hand_tensor = hand_tensor.to(self.device)
            face_tensor = face_tensor.to(self.device)

            with torch.no_grad():
                out = self.model(hand_tensor, face_tensor)

            # Modality weights (w_hand, w_face) for the most recent frame
            last_w = out.fusion_weights[0, -1].cpu().numpy()
            result["fusion_weights"] = last_w

            if self.is_checkpoint_loaded:
                # Real inference prediction only if an authentic checkpoint is loaded
                probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                if conf >= self.confidence_threshold:
                    sign_label = self.config.target_vocab[pred_idx] if pred_idx < len(self.config.target_vocab) else f"CLASS_{pred_idx}"
                    result["predicted_sign"] = sign_label
                    result["translation_es"] = self.translator.translate_label(sign_label, "es")
                    result["translation_en"] = self.translator.translate_label(sign_label, "en")
                    result["confidence"] = conf
                else:
                    result["predicted_sign"] = "LOW CONFIDENCE"
            else:
                # Untrained / Demo Mode safeguard
                result["predicted_sign"] = "INFERENCE DISABLED"
                result["translation_es"] = "MODEL STATUS: NOT TRAINED"
                result["translation_en"] = "MODEL STATUS: NOT TRAINED"

        return result


def draw_inference_hud(frame: np.ndarray, info: Dict[str, Any], fps: float):
    """
    Renders professional Head-Up Display (HUD) overlay for real-time monitoring.
    """
    h, w, _ = frame.shape
    # Top overlay bar
    cv2.rectangle(frame, (0, 0), (w, 140), (25, 25, 25), -1)

    # Status indicators
    hand_col = (0, 255, 0) if info["hands_detected"] else (0, 0, 255)
    face_col = (0, 255, 0) if info["face_detected"] else (0, 0, 255)
    hand_txt = "HAND: DETECTED" if info["hands_detected"] else "HAND: NOT DETECTED"
    face_txt = "FACE: DETECTED" if info["face_detected"] else "FACE: NOT DETECTED"

    cv2.putText(frame, hand_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_col, 2, cv2.LINE_AA)
    cv2.putText(frame, face_txt, (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, face_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"BUFFER: {info['buffer_status']}", (420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Inference & Translation outputs
    if info["is_trained"]:
        sign_str = f"SIGN: {info['predicted_sign'] or '...'}"
        trans_str = f"SPANISH: {info['translation_es'] or '...'} | ENGLISH: {info['translation_en'] or '...'}"
        cv2.putText(frame, sign_str, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, trans_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Untrained mode banner
        cv2.putText(frame, "MODEL STATUS: NOT TRAINED", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "INFERENCE: DISABLED (PIPELINE & TENSOR VERIFICATION ACTIVE)", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Modality weights display
    if info["fusion_weights"] is not None:
        w_h, w_f = info["fusion_weights"]
        weights_txt = f"Modality Weights: Hand={w_h:.2f}, Face={w_f:.2f}"
        cv2.putText(frame, weights_txt, (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 220, 150), 1, cv2.LINE_AA)


def run_webcam_inference(checkpoint_path: Optional[str] = None):
    """
    Main real-time loop capturing webcam, executing multimodal recognition, and displaying results.
    """
    engine = RealtimeInferenceEngine(checkpoint_path=checkpoint_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default camera (index 0).", file=sys.stderr)
        return

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    prev_time = time.perf_counter()

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        print("=" * 70)
        print(" LSE REAL-TIME MULTIMODAL INFERENCE PIPELINE")
        print(" Window: 'LSE Multimodal Recognition'")
        print(" Controls: Press [Q] to quit | [C] to clear buffer")
        print("=" * 70)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            hand_res = hands.process(rgb_frame)
            face_res = face_mesh.process(rgb_frame)

            rgb_frame.flags.writeable = True

            # Draw visual landmarks
            if hand_res.multi_hand_landmarks:
                for hl in hand_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())

            if face_res.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_res.multi_face_landmarks[0],
                                          mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

            # Pipeline processing & inference
            info = engine.process_frame(hand_res, face_res)

            # FPS calculation
            curr_time = time.perf_counter()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Render HUD
            draw_inference_hud(frame, info, fps)
            cv2.imshow("LSE Multimodal Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('c'), ord('C')):
                engine.buffer.clear()
                print("[Buffer Cleared]")

    cap.release()
    cv2.destroyAllWindows()
    print("Inference pipeline closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSE Real-Time Inference.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch checkpoint (.pth)")
    args = parser.parse_args()

    run_webcam_inference(checkpoint_path=args.checkpoint)
