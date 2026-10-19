"""
LSE Real-Time Pipeline Offline Verification Test (Using Real HOLA Sample)

Simulates the real-time sliding buffer and multimodal inference pipeline frame-by-frame
using the existing processed HOLA test recording without requiring a live webcam.

INTEGRATION TEST ONLY — NOT TRAINING/EVALUATION.
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel
from recognition.realtime_inference import MultimodalTemporalBuffer
from translation.translator import LSETranslator


def test_offline_temporal_streaming():
    print("=" * 70)
    print(" LSE OFFLINE TEMPORAL BUFFER & STREAMING INFERENCE TEST")
    print(" INTEGRATION TEST ONLY — NOT TRAINING/EVALUATION")
    print("=" * 70)

    # 1. Load real processed HOLA sample
    hand_path = os.path.join("data", "processed", "HOLA", "signer01_001.npz")
    face_path = os.path.join("data", "processed", "HOLA", "face_signer01_001.npz")

    hand_data = np.load(hand_path, allow_pickle=True)["features"]  # (30, 126)
    face_data = np.load(face_path, allow_pickle=True)["features"]  # (30, 1404)

    print(f" Loaded Stream Sample: {hand_data.shape[0]} frames of Hands + Face")
    print("-" * 70)

    # 2. Initialize Buffer & Model
    buffer = MultimodalTemporalBuffer(sequence_length=30, hand_dim=126, face_dim=1404)
    config = ModelConfig(num_classes=5)
    model = MultimodalSignModel(config=config)
    model.eval()

    translator = LSETranslator()

    # 3. Simulate Frame-by-Frame Streaming
    inference_triggered_count = 0
    for frame_idx in range(hand_data.shape[0]):
        h_frame = hand_data[frame_idx]
        f_frame = face_data[frame_idx]

        buffer.append(h_frame, f_frame)
        status_msg = f"Frame {frame_idx + 1:02d}/30 -> Buffer: {buffer.current_length:02d}/30 | Ready: {buffer.is_full()}"

        if buffer.is_full():
            hand_tensor, face_tensor = buffer.get_tensors()

            with torch.no_grad():
                out = model(hand_tensor, face_tensor)

            last_w = out.fusion_weights[0, -1].numpy()
            status_msg += f" | Forward Pass Executed! Modality Weights: [Hand={last_w[0]:.2f}, Face={last_w[1]:.2f}]"
            inference_triggered_count += 1

        print(status_msg)

    print("-" * 70)
    assert inference_triggered_count == 1, f"Expected exactly 1 trigger, got {inference_triggered_count}"
    assert buffer.is_full() is True
    print("[PASSED] Offline temporal streaming test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    test_offline_temporal_streaming()
