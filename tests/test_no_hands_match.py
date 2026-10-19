"""Test that the verification prototype does not report a MATCH when hands are missing.

The LiveMultimodalProcessor appends a zero-vector for missing hands (see
process_hands_frame). Previously the distance calculation would treat these
zeros as valid data, leading to a false positive MATCH. The bug fix adds a
check for any all-zero hand frame and clears the buffer, reporting
"NO MATCH — HANDS NOT DETECTED" instead.

This test creates a temporary reference HOLA sample and runs the processor
through 30 blank frames (no hands detected). It then asserts that the comparison
status contains the expected message and that the buffer is cleared.
"""

import os
import numpy as np
import pytest


def _create_dummy_reference(tmp_root: str) -> str:
    """Create a minimal HOLA reference .npz file."""
    ref_dir = os.path.join(tmp_root, "data", "processed", "HOLA")
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "signer01_001.npz")
    dummy_features = np.zeros((30, 126), dtype=np.float32)
    np.savez(ref_path, features=dummy_features)
    return ref_path


@pytest.fixture(scope="function")
def processor_with_dummy_ref(tmp_path_factory):
    """Create a LiveMultimodalProcessor with a dummy reference sample."""
    tmp_root = tmp_path_factory.mktemp("workspace")
    _create_dummy_reference(str(tmp_root))

    import importlib
    import app.web_demo as web_demo
    web_demo.PROJECT_ROOT = str(tmp_root)
    importlib.reload(web_demo)

    from app.web_demo import LiveMultimodalProcessor, DEFAULT_MATCH_THRESHOLD
    return LiveMultimodalProcessor(match_threshold=DEFAULT_MATCH_THRESHOLD)


def test_no_hands_match_is_rejected(processor_with_dummy_ref):
    """Run 30 blank frames and verify the comparison status."""
    proc = processor_with_dummy_ref
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cmp_status = None
    for _ in range(30):
        _, _, _, _, _, _, _, _, _, cmp_status, _, _ = proc.process_frame(blank_frame)
    assert "NO MATCH — HANDS NOT DETECTED" in cmp_status, f"Expected no-match message, got: {cmp_status}"
    assert proc.buffer.current_length == 0, "Buffer should be cleared after missing-hand detection"
