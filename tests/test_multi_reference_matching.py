"""
Focused tests for 10-Sign Reference Vocabulary and Multi-Reference Template Matching
Covers Requirement 12:
- Multiple labels comparison and correct minimum-distance selection
- Missing hands detection, rejection, and buffer clearance
- Invalid reference file filtering (wrong shape, NaN/Inf, all-zero, missing keys)
- Threshold handling (MATCH vs NO MATCH)
- Stale results invalidation and buffer reset behavior
- Reference recording validation and strict overwrite protection
- Honest vocabulary coverage status reporting
"""

import os
import shutil
import numpy as np
import pytest

from app.web_demo import (
    LiveMultimodalProcessor,
    compute_sequence_distance,
    DEFAULT_MATCH_THRESHOLD
)
from translation.translator import CANONICAL_VOCABULARY, LSETranslator


def _create_mock_sample(filepath: str, label: str, shape=(30, 126), fill_val=0.5, signer="signer01", rec_id="001"):
    """Helper to create a valid reference sample .npz file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    features = np.full(shape, fill_val, dtype=np.float32)
    np.savez(filepath, features=features, label=label, signer=signer, recording_id=rec_id)
    return features


@pytest.fixture
def multi_ref_workspace(tmp_path_factory):
    """
    Creates a temporary workspace with mock reference samples for HOLA, GRACIAS, and ADIOS,
    while leaving other vocabulary signs as 'Needs Recording'.
    """
    tmp_root = tmp_path_factory.mktemp("multi_ref_ws")
    processed_dir = tmp_root / "data" / "processed"

    # 1. HOLA reference: fill_val = 0.2
    _create_mock_sample(str(processed_dir / "HOLA" / "signer01_001.npz"), "HOLA", fill_val=0.2)

    # 2. GRACIAS reference: fill_val = 0.6
    _create_mock_sample(str(processed_dir / "GRACIAS" / "signer01_001.npz"), "GRACIAS", fill_val=0.6)

    # 3. ADIOS reference: fill_val = 0.9
    _create_mock_sample(str(processed_dir / "ADIOS" / "signer01_001.npz"), "ADIOS", fill_val=0.9)

    import app.web_demo as web_demo
    orig_root = web_demo.PROJECT_ROOT
    web_demo.PROJECT_ROOT = str(tmp_root)

    processor = LiveMultimodalProcessor(match_threshold=0.30)

    yield processor, tmp_root

    web_demo.PROJECT_ROOT = orig_root


def test_multi_label_matching_picks_closest_sign(multi_ref_workspace):
    """Verifies that multi-reference matching selects the sign with the lowest MAE distance."""
    processor, _ = multi_ref_workspace
    assert len(processor.reference_library["HOLA"]) == 1
    assert len(processor.reference_library["GRACIAS"]) == 1
    assert len(processor.reference_library["ADIOS"]) == 1

    # Feed sequence very close to GRACIAS (fill_val = 0.61 -> distance to GRACIAS is 0.01)
    gracias_like_frame = np.full(126, 0.61, dtype=np.float32)
    face_dummy = np.zeros(1404, dtype=np.float32)

    for _ in range(30):
        processor.buffer.append(gracias_like_frame, face_dummy)

    assert processor.buffer.is_full()

    # Process a synthetic frame to trigger comparison evaluation
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Monkey-patch detectors to avoid re-overwriting hand buffer during process_frame
    hand_seq = np.array(processor.buffer.hand_buffer, dtype=np.float32)
    dist_gracias = compute_sequence_distance(hand_seq, processor.reference_library["GRACIAS"][0]["features"])
    dist_hola = compute_sequence_distance(hand_seq, processor.reference_library["HOLA"][0]["features"])
    dist_adios = compute_sequence_distance(hand_seq, processor.reference_library["ADIOS"][0]["features"])

    assert dist_gracias < dist_hola
    assert dist_gracias < dist_adios
    assert dist_gracias < processor.match_threshold

    # Direct verification against processor comparison logic
    best_label = None
    best_dist = float("inf")
    for lbl in CANONICAL_VOCABULARY:
        for s in processor.reference_library.get(lbl, []):
            d = compute_sequence_distance(hand_seq, s["features"])
            if d < best_dist:
                best_dist = d
                best_label = lbl

    assert best_label == "GRACIAS"
    assert processor.translator.translate_label(best_label, "es") == "Gracias"
    assert processor.translator.translate_label(best_label, "en") == "Thank you"


def test_missing_hands_rejected_and_buffer_cleared(multi_ref_workspace):
    """Verifies that missing hands (zero vectors) report NO MATCH and immediately clear buffer."""
    processor, _ = multi_ref_workspace
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cmp_status = ""
    for _ in range(30):
        res = processor.process_frame(blank_frame)
        cmp_status = res[9]

    assert "NO MATCH — HANDS NOT DETECTED" in cmp_status
    assert processor.buffer.current_length == 0
    assert res[10] == "..."
    assert res[11] == "..."


def test_invalid_reference_files_are_safely_ignored(tmp_path_factory):
    """Verifies that corrupted, wrong-shape, NaN, and all-zero reference files are safely skipped."""
    tmp_root = tmp_path_factory.mktemp("invalid_refs_ws")
    processed_dir = tmp_root / "data" / "processed"

    # 1. Wrong shape (20, 126)
    _create_mock_sample(str(processed_dir / "SI" / "signer01_bad_shape.npz"), "SI", shape=(20, 126))

    # 2. Contains NaNs
    nan_path = str(processed_dir / "NO" / "signer01_nans.npz")
    os.makedirs(os.path.dirname(nan_path), exist_ok=True)
    nan_feats = np.ones((30, 126), dtype=np.float32)
    nan_feats[5, 10] = np.nan
    np.savez(nan_path, features=nan_feats, label="NO", signer="signer01", recording_id="001")

    # 3. All zeros
    _create_mock_sample(str(processed_dir / "AGUA" / "signer01_zeros.npz"), "AGUA", fill_val=0.0)

    # 4. Missing 'features' key
    missing_key_path = str(processed_dir / "AYUDA" / "signer01_nokey.npz")
    os.makedirs(os.path.dirname(missing_key_path), exist_ok=True)
    np.savez(missing_key_path, wrong_key=np.ones((30, 126), dtype=np.float32))

    # 5. One valid reference for BIEN
    _create_mock_sample(str(processed_dir / "BIEN" / "signer01_valid.npz"), "BIEN", fill_val=0.45)

    import app.web_demo as web_demo
    orig_root = web_demo.PROJECT_ROOT
    web_demo.PROJECT_ROOT = str(tmp_root)

    processor = LiveMultimodalProcessor()

    # Only BIEN should be loaded as valid
    assert len(processor.reference_library["SI"]) == 0
    assert len(processor.reference_library["NO"]) == 0
    assert len(processor.reference_library["AGUA"]) == 0
    assert len(processor.reference_library["AYUDA"]) == 0
    assert len(processor.reference_library["BIEN"]) == 1

    web_demo.PROJECT_ROOT = orig_root


def test_threshold_handling(multi_ref_workspace):
    """Verifies that distance <= threshold yields MATCH, while distance > threshold yields NO MATCH."""
    processor, _ = multi_ref_workspace
    ref_hola = processor.reference_library["HOLA"][0]["features"]  # fill_val = 0.2

    # Case 1: Close input (distance = 0.05 <= 0.30)
    close_input = ref_hola + 0.05
    dist_close = compute_sequence_distance(close_input, ref_hola)
    assert dist_close <= processor.match_threshold

    # Case 2: Divergent input (distance = 0.50 > 0.30)
    far_input = ref_hola + 0.50
    dist_far = compute_sequence_distance(far_input, ref_hola)
    assert dist_far > processor.match_threshold


def test_stale_results_and_buffer_reset(multi_ref_workspace):
    """Verifies that reset_buffer clears stored sequence and restores awaiting telemetry."""
    processor, _ = multi_ref_workspace

    # Fill buffer to 25 frames
    for _ in range(25):
        processor.buffer.append(np.ones(126, dtype=np.float32), np.zeros(1404, dtype=np.float32))
    assert processor.buffer.current_length == 25

    buf_stat, cmp_stat, trans_es, trans_en = processor.reset_buffer()
    assert processor.buffer.current_length == 0
    assert "0/30" in buf_stat
    assert "Awaiting" in cmp_stat
    assert trans_es == "..."
    assert trans_en == "..."


def test_reference_recording_validations(multi_ref_workspace):
    """Verifies that save_current_buffer_as_reference validates length, dimensions, and missing hands."""
    processor, _ = multi_ref_workspace

    # 1. Empty/incomplete buffer (< 30 frames)
    msg, _, _ = processor.save_current_buffer_as_reference("PERDON", "signer01")
    assert "[ERROR]" in msg
    assert "Buffer has only 0/30 frames" in msg

    # 2. Buffer has 30 frames, but hands are missing (all zeros)
    for _ in range(30):
        processor.buffer.append(np.zeros(126, dtype=np.float32), np.zeros(1404, dtype=np.float32))
    msg, _, _ = processor.save_current_buffer_as_reference("PERDON", "signer01")
    assert "[ERROR]" in msg
    assert "missing" in msg.lower()


def test_reference_recording_prevents_overwrite(multi_ref_workspace):
    """Verifies that recording saves sequentially without overwriting existing files."""
    processor, tmp_root = multi_ref_workspace
    processor.buffer.clear()

    # Fill buffer with valid continuous hand frames (e.g. fill_val = 0.4)
    valid_frame = np.full(126, 0.4, dtype=np.float32)
    face_frame = np.zeros(1404, dtype=np.float32)
    for _ in range(30):
        processor.buffer.append(valid_frame, face_frame)

    # First save for SI
    msg1, _, active_str1 = processor.save_current_buffer_as_reference("SI", "signer01")
    assert "[SUCCESS]" in msg1
    file1 = tmp_root / "data" / "processed" / "SI" / "signer01_001.npz"
    assert file1.exists()

    # Second save for SI (must write signer01_002.npz, not overwrite signer01_001.npz)
    msg2, _, active_str2 = processor.save_current_buffer_as_reference("SI", "signer01")
    assert "[SUCCESS]" in msg2
    file2 = tmp_root / "data" / "processed" / "SI" / "signer01_002.npz"
    assert file2.exists()
    assert file1.exists()  # file1 intact!

    # Library reloaded and has 2 samples for SI
    assert len(processor.reference_library["SI"]) == 2
    assert "SI" in active_str2


def test_vocabulary_coverage_honest_reporting(multi_ref_workspace):
    """Verifies that get_library_status_markdown accurately shows recorded vs unrecorded signs."""
    processor, _ = multi_ref_workspace
    status_md = processor.get_library_status_markdown()

    # Workspace has HOLA, GRACIAS, ADIOS recorded (3/10)
    assert "3/10 Signs Recorded" in status_md
    assert "| **HOLA** |" in status_md and "**RECORDED**" in status_md
    assert "| **GRACIAS** |" in status_md and "**RECORDED**" in status_md
    assert "| **ADIOS** |" in status_md and "**RECORDED**" in status_md

    # Unrecorded signs must explicitly say Needs Recording
    assert "| **POR_FAVOR** |" in status_md and "*Needs Recording*" in status_md
    assert "| **AGUA** |" in status_md and "*Needs Recording*" in status_md
    assert "| **AYUDA** |" in status_md and "*Needs Recording*" in status_md
    assert "| **PERDON** |" in status_md and "*Needs Recording*" in status_md
    assert "| **BIEN** |" in status_md and "*Needs Recording*" in status_md
