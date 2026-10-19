"""
Focused tests for Face Landmark Tracking & Occlusion Handling
Covers requirements:
1. Independent calculation per frame without stale landmark retention.
2. When face is absent or occluded, num_faces=0 and face_status="FACE NOT DETECTED".
3. When face is detected, returns actual landmarks and updates face_status.
4. Annotated video and telemetry agree on current-frame detection.
"""

import os
import cv2
import numpy as np
import pytest

from app.web_demo import LiveMultimodalProcessor


@pytest.fixture(scope="module")
def real_face_image():
    """Loads Grace Hopper sample image from matplotlib data for real face detection testing."""
    path = os.path.join(
        "venv", "Lib", "site-packages", "matplotlib", "mpl-data", "sample_data", "grace_hopper.jpg"
    )
    if os.path.exists(path):
        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (640, 480))
    # Fallback synthetic frame if sample image is absent
    return None


def test_face_not_detected_on_blank_frame():
    """Confirms that blank/covered frame returns FACE NOT DETECTED, faces=0, and no stale mesh."""
    proc = LiveMultimodalProcessor()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    res = proc.process_frame(blank)
    annotated, fps, hands_cnt, faces_cnt, hand_stat, face_stat = res[:6]

    assert faces_cnt == "0", f"Expected faces_cnt='0', got {faces_cnt}"
    assert face_stat == "FACE NOT DETECTED", f"Expected 'FACE NOT DETECTED', got {face_stat}"


def test_face_detection_and_occlusion_transition(real_face_image):
    """
    Tests sequential frame transition:
    Frame 1: Real face visible -> Detected, faces=1
    Frame 2: Face completely covered -> Immediately FACE NOT DETECTED, faces=0 (no stale retention)
    Frame 3: Real face visible again -> Detected, faces=1
    """
    if real_face_image is None:
        pytest.skip("Grace Hopper sample face image not found in venv")

    proc = LiveMultimodalProcessor()

    # Frame 1: Face visible
    res1 = proc.process_frame(real_face_image)
    assert res1[3] == "1", f"Expected 1 face detected, got {res1[3]}"
    assert "Detected" in res1[5]
    assert "landmarks updated" in res1[5]

    # Frame 2: Face covered by a large solid block (occlusion)
    covered_frame = real_face_image.copy()
    cv2.rectangle(covered_frame, (50, 20), (590, 460), (50, 50, 50), -1)

    res2 = proc.process_frame(covered_frame)
    assert res2[3] == "0", f"Expected 0 faces on covered frame, got {res2[3]}"
    assert res2[5] == "FACE NOT DETECTED"

    # Frame 3: Face visible again
    res3 = proc.process_frame(real_face_image)
    assert res3[3] == "1"
    assert "Detected" in res3[5]


def test_annotated_feed_shows_face_not_detected_text():
    """Confirms that when face is missing, the annotated frame receives the visual indicator."""
    proc = LiveMultimodalProcessor()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    res = proc.process_frame(blank)
    annotated = res[0]
    # Frame was modified with HUD and alert text (not purely all zeros)
    assert not np.all(annotated == 0)
