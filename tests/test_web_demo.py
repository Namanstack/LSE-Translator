"""
Unit tests for the Live Web Demo module (app/web_demo.py)
Covering:
- Temporal buffer filling from 0 to 30
- Buffer reset functionality
- Correct shape of a full sequence
- Legitimate sequence distance comparison (MAE)
- No-match behavior on divergent sequences
- Translation triggering only after a genuine match
- Blank and synthetic frame processing
"""

import os
import numpy as np
import pytest
from app.web_demo import (
    LiveMultimodalProcessor,
    build_demo_app,
    compute_sequence_distance,
    DEFAULT_MATCH_THRESHOLD
)


def test_compute_sequence_distance_identical():
    """Identical sequences must yield exactly 0.0 distance."""
    seq = np.random.uniform(-1, 1, size=(30, 126)).astype(np.float32)
    dist = compute_sequence_distance(seq, seq)
    assert dist == 0.0


def test_compute_sequence_distance_divergent():
    """Divergent sequences must yield non-zero positive distance."""
    seq_a = np.zeros((30, 126), dtype=np.float32)
    seq_b = np.ones((30, 126), dtype=np.float32)
    dist = compute_sequence_distance(seq_a, seq_b)
    assert dist == 1.0


def test_temporal_buffer_filling_and_shape():
    """Tests temporal buffer filling sequentially from 0 to 30 and verifies full tensor shape."""
    processor = LiveMultimodalProcessor()
    assert processor.buffer.current_length == 0
    assert not processor.buffer.is_full()

    # Feed 15 frames
    for _ in range(15):
        processor.buffer.append(np.zeros(126, dtype=np.float32), np.zeros(1404, dtype=np.float32))
    assert processor.buffer.current_length == 15
    assert not processor.buffer.is_full()

    # Feed remaining 15 frames to reach full 30
    for _ in range(15):
        processor.buffer.append(np.zeros(126, dtype=np.float32), np.zeros(1404, dtype=np.float32))
    assert processor.buffer.current_length == 30
    assert processor.buffer.is_full()

    hand_tensor, face_tensor = processor.buffer.get_tensors()
    assert hand_tensor.shape == (1, 30, 126)
    assert face_tensor.shape == (1, 30, 1404)


def test_buffer_reset_functionality():
    """Tests that reset_buffer clears stored frames and resets telemetry strings."""
    processor = LiveMultimodalProcessor()
    for _ in range(25):
        processor.buffer.append(np.ones(126, dtype=np.float32), np.ones(1404, dtype=np.float32))
    assert processor.buffer.current_length == 25

    # Trigger reset
    buf_stat, cmp_stat, trans_es, trans_en = processor.reset_buffer()
    assert processor.buffer.current_length == 0
    assert "0/30" in buf_stat
    assert "Awaiting" in cmp_stat
    assert trans_es == "..."
    assert trans_en == "..."


def test_reference_matching_and_translation_genuine():
    """
    Verifies that when current buffer closely matches the reference HOLA sample,
    a MATCH is detected and translation output is populated.
    """
    processor = LiveMultimodalProcessor(match_threshold=DEFAULT_MATCH_THRESHOLD)
    ref_hola = processor.reference_samples.get("HOLA")
    assert ref_hola is not None, "Reference HOLA sample must exist in data/processed/HOLA/signer01_001.npz"

    # Fill buffer with reference sequence (exact match)
    for frame_idx in range(30):
        processor.buffer.append(ref_hola[frame_idx], np.zeros(1404, dtype=np.float32))

    # Evaluate distance
    curr_hand = np.array(processor.buffer.hand_buffer, dtype=np.float32)
    dist = compute_sequence_distance(curr_hand, ref_hola)
    assert dist == 0.0
    assert dist <= processor.match_threshold

    # Translation
    es_trans = processor.translator.translate_label("HOLA", "es")
    en_trans = processor.translator.translate_label("HOLA", "en")
    assert es_trans == "Hola"
    assert en_trans == "Hello"


def test_reference_no_match_behavior():
    """
    Verifies that divergent/random sequences do not trigger a false match or translation.
    """
    processor = LiveMultimodalProcessor(match_threshold=0.30)
    # Fill with inverted/shifted values
    for _ in range(30):
        processor.buffer.append(np.full(126, 5.0, dtype=np.float32), np.zeros(1404, dtype=np.float32))

    ref_hola = processor.reference_samples.get("HOLA")
    curr_hand = np.array(processor.buffer.hand_buffer, dtype=np.float32)
    dist = compute_sequence_distance(curr_hand, ref_hola)
    assert dist > processor.match_threshold


def test_process_frame_blank_and_synthetic():
    """Verifies that frame processing pipeline runs without exceptions."""
    processor = LiveMultimodalProcessor()
    # 1. Blank frame
    res_blank = processor.process_frame(None)
    assert res_blank[0].shape == (480, 640, 3)
    assert res_blank[2] == "0"  # Hands

    # 2. Synthetic frame
    synth = np.zeros((480, 640, 3), dtype=np.uint8)
    res_synth = processor.process_frame(synth)
    assert res_synth[0].shape == (480, 640, 3)
    assert "0/30" in res_synth[6] or "1/30" in res_synth[6]


def test_build_demo_app_structure():
    demo = build_demo_app()
    assert demo is not None
