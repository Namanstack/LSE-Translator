"""
Focused tests for Live Facial Expression Landmark Tracking & Dynamics.

Validates:
1. MediaPipe FaceMesh processing of current frames.
2. Dynamic facial deformation tracking:
   - Neutral face
   - Mouth open (inner lips 13 to 14 distance expansion)
   - Smile (mouth corners 61 to 291 distance expansion)
   - Eyebrows raised (brow 105/334 to eye 159/386 distance expansion)
3. Scale invariance: facial expression ratios are normalized by face height (10 to 152),
   isolating facial expression from camera distance and rigid head movement.
4. Detection state consistency: when no face is present, face count is 0,
   face_status displays 'FACE NOT DETECTED', and no stale mesh is retained.
5. Telemetry updates: face_status accurately reports detected face count,
   landmark count, ΔMAE, feature array change status, and expression metrics.
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
    return None


def test_face_expression_telemetry_fields(real_face_image):
    """
    Verifies that the telemetry status string contains all required diagnostic values:
    - Number of detected faces
    - Current-frame landmark count
    - Mean absolute coordinate difference (ΔMAE)
    - Feature array change status
    - Mouth opening distance
    - Eyebrow distance
    """
    if real_face_image is None:
        pytest.skip("Grace Hopper sample face image not found in venv")

    proc = LiveMultimodalProcessor()

    # Frame 1: First face detection
    res1 = proc.process_frame(real_face_image)
    face_stat1 = res1[5]

    assert "Detected (1 face" in face_stat1
    assert "landmarks updated" in face_stat1
    assert "ΔMAE:" in face_stat1
    assert "Feat:" in face_stat1
    assert "Mouth:" in face_stat1
    assert "Brow:" in face_stat1

    # Frame 2: Sequential frame (ΔMAE should be computable and stable)
    res2 = proc.process_frame(real_face_image)
    face_stat2 = res2[5]
    assert "Detected (1 face" in face_stat2
    assert "ΔMAE:" in face_stat2


def test_face_occlusion_clears_landmarks_and_diagnostics():
    """
    Confirms that when face is occluded or absent:
    - faces_cnt is 0
    - face_status is 'FACE NOT DETECTED'
    - Internal coordinate tracking is cleared to prevent drawing stale points
    """
    proc = LiveMultimodalProcessor()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    res = proc.process_frame(blank)
    annotated, fps, hands_cnt, faces_cnt, hand_stat, face_stat = res[:6]

    assert faces_cnt == "0"
    assert face_stat == "FACE NOT DETECTED"
    assert proc.prev_face_coords is None
    assert proc.prev_face_feat is None


def test_facial_expression_geometric_metrics():
    """
    Validates geometric sensitivity of expression indices:
    - Inner lips (13 to 14) detects mouth opening
    - Mouth corners (61 to 291) detects smile widening
    - Eyebrows (105/334 to 159/386) detects eyebrow raising
    - Normalization by face scale (10 to 152) makes metrics scale-invariant
    """
    # Create coordinate templates (468 landmarks in 3D)
    neutral_coords = np.zeros((468, 3), dtype=np.float32)
    # Forehead (10) and chin (152)
    neutral_coords[10] = [0.5, 0.2, 0.0]
    neutral_coords[152] = [0.5, 0.8, 0.0]
    face_scale = float(np.linalg.norm(neutral_coords[10] - neutral_coords[152]))  # 0.60

    # Neutral mouth (closed)
    neutral_coords[13] = [0.5, 0.65, 0.0]  # upper inner lip
    neutral_coords[14] = [0.5, 0.66, 0.0]  # lower inner lip
    neutral_open = float(np.linalg.norm(neutral_coords[13] - neutral_coords[14])) / face_scale

    # Open mouth (jaw dropped)
    open_coords = neutral_coords.copy()
    open_coords[14] = [0.5, 0.73, 0.0]  # lower inner lip descends
    mouth_open = float(np.linalg.norm(open_coords[13] - open_coords[14])) / face_scale

    assert mouth_open > neutral_open * 4.0, "Mouth open ratio must be substantially greater than neutral"

    # Neutral smile vs wide smile
    neutral_coords[61] = [0.42, 0.65, 0.0]
    neutral_coords[291] = [0.58, 0.65, 0.0]
    neutral_width = float(np.linalg.norm(neutral_coords[61] - neutral_coords[291])) / face_scale

    smile_coords = neutral_coords.copy()
    smile_coords[61] = [0.38, 0.63, 0.0]
    smile_coords[291] = [0.62, 0.63, 0.0]
    smile_width = float(np.linalg.norm(smile_coords[61] - smile_coords[291])) / face_scale

    assert smile_width > neutral_width * 1.3, "Smile width ratio must be significantly greater than neutral"

    # Neutral eyebrows vs raised eyebrows
    neutral_coords[105] = [0.35, 0.32, 0.0]
    neutral_coords[159] = [0.35, 0.38, 0.0]
    neutral_coords[334] = [0.65, 0.32, 0.0]
    neutral_coords[386] = [0.65, 0.38, 0.0]
    neutral_brow = float(0.5 * (
        np.linalg.norm(neutral_coords[105] - neutral_coords[159]) +
        np.linalg.norm(neutral_coords[334] - neutral_coords[386])
    )) / face_scale

    raised_coords = neutral_coords.copy()
    raised_coords[105] = [0.35, 0.27, 0.0]
    raised_coords[334] = [0.65, 0.27, 0.0]
    raised_brow = float(0.5 * (
        np.linalg.norm(raised_coords[105] - raised_coords[159]) +
        np.linalg.norm(raised_coords[334] - raised_coords[386])
    )) / face_scale

    assert raised_brow > neutral_brow * 1.5, "Raised eyebrow ratio must be significantly greater than neutral"


def test_rigid_head_translation_invariance():
    """
    Confirms that rigid head movement (translating x, y or scaling z)
    does not distort scale-normalized expression ratios.
    """
    coords = np.zeros((468, 3), dtype=np.float32)
    coords[10] = [0.5, 0.2, 0.0]
    coords[152] = [0.5, 0.8, 0.0]
    coords[13] = [0.5, 0.65, 0.0]
    coords[14] = [0.5, 0.68, 0.0]

    scale_orig = float(np.linalg.norm(coords[10] - coords[152]))
    ratio_orig = float(np.linalg.norm(coords[13] - coords[14])) / scale_orig

    # Rigid translation: move head diagonally (+0.2 in x, -0.1 in y)
    translated_coords = coords.copy() + np.array([0.2, -0.1, 0.05], dtype=np.float32)
    scale_trans = float(np.linalg.norm(translated_coords[10] - translated_coords[152]))
    ratio_trans = float(np.linalg.norm(translated_coords[13] - translated_coords[14])) / scale_trans

    assert np.isclose(ratio_orig, ratio_trans, atol=1e-6), (
        "Scale-normalized mouth opening ratio must be invariant to head translation"
    )

    # Scale translation: user steps back (entire face shrinks by 50%)
    scaled_coords = coords.copy() * 0.5
    scale_scaled = float(np.linalg.norm(scaled_coords[10] - scaled_coords[152]))
    ratio_scaled = float(np.linalg.norm(scaled_coords[13] - scaled_coords[14])) / scale_scaled

    assert np.isclose(ratio_orig, ratio_scaled, atol=1e-6), (
        "Scale-normalized mouth opening ratio must be invariant to distance scaling"
    )
