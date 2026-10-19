"""
LSE Real-Time Pipeline Unit Tests

Comprehensive tests covering:
1. Temporal buffer FIFO mechanics and rolling window behavior
2. Missing hand fallback (all zeros)
3. Missing face fallback (all zeros)
4. Both modalities missing fallback
5. Absence of trained checkpoint handling (ensures inference disabled mode)
6. Non-existent checkpoint loading error handling
7. Shape compatibility checks across modalities
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel
from recognition.realtime_inference import (
    MultimodalTemporalBuffer,
    RealtimeInferenceEngine,
    process_hands_frame,
    process_face_frame
)


class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockResults:
    def __init__(self, multi_landmarks=None, handedness=None):
        self.multi_hand_landmarks = multi_landmarks
        self.multi_face_landmarks = multi_landmarks
        self.multi_handedness = handedness


class TestRealtimeInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.seq_len = 30
        self.hand_dim = 126
        self.face_dim = 1404
        self.config = ModelConfig(
            sequence_length=self.seq_len,
            hand_feature_dim=self.hand_dim,
            face_feature_dim=self.face_dim,
            num_classes=5
        )

    def test_buffer_sliding_window(self):
        """Tests that buffer maintains FIFO queue of exact length 30."""
        buffer = MultimodalTemporalBuffer(sequence_length=self.seq_len, hand_dim=self.hand_dim, face_dim=self.face_dim)
        self.assertEqual(buffer.current_length, 0)
        self.assertFalse(buffer.is_full())

        for i in range(25):
            buffer.append(np.zeros(self.hand_dim), np.zeros(self.face_dim))
        self.assertEqual(buffer.current_length, 25)
        self.assertFalse(buffer.is_full())

        for i in range(10):
            buffer.append(np.full(self.hand_dim, i + 1), np.full(self.face_dim, i + 1))
        # Should be capped at 30
        self.assertEqual(buffer.current_length, 30)
        self.assertTrue(buffer.is_full())

        hand_tensor, face_tensor = buffer.get_tensors()
        self.assertEqual(hand_tensor.shape, (1, 30, self.hand_dim))
        self.assertEqual(face_tensor.shape, (1, 30, self.face_dim))

    def test_missing_modalities_fallback(self):
        """Tests that missing hands/faces produce well-conditioned zero vectors without crashing."""
        empty_res = MockResults(multi_landmarks=None)

        hand_feat, hand_detected = process_hands_frame(empty_res)
        self.assertFalse(hand_detected)
        self.assertEqual(hand_feat.shape, (self.hand_dim,))
        self.assertTrue(np.all(hand_feat == 0.0))

        face_feat, face_detected = process_face_frame(empty_res)
        self.assertFalse(face_detected)
        self.assertEqual(face_feat.shape, (self.face_dim,))
        self.assertTrue(np.all(face_feat == 0.0))

    def test_untrained_mode_safeguard(self):
        """Tests that missing checkpoint properly disables fake predictions."""
        engine = RealtimeInferenceEngine(config=self.config, checkpoint_path=None)
        self.assertFalse(engine.is_checkpoint_loaded)

        empty_res = MockResults(multi_landmarks=None)
        for _ in range(30):
            info = engine.process_frame(empty_res, empty_res)

        self.assertTrue(info["buffer_full"])
        self.assertFalse(info["is_trained"])
        self.assertEqual(info["predicted_sign"], "INFERENCE DISABLED")
        self.assertEqual(info["translation_es"], "MODEL STATUS: NOT TRAINED")

    def test_nonexistent_checkpoint_handling(self):
        """Tests that invalid checkpoint path does not crash the system."""
        engine = RealtimeInferenceEngine(config=self.config, checkpoint_path="models/checkpoints/non_existent.pth")
        self.assertFalse(engine.is_checkpoint_loaded)


if __name__ == "__main__":
    unittest.main()
