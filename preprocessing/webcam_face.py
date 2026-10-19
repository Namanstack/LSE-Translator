"""
LSE Face Tracking & Landmark Extraction - Modality 2

Real-time facial landmark detection using OpenCV and MediaPipe Face Mesh.
Extracts 468 facial mesh landmarks, performs nose-bridge centering and scale normalization,
and displays live feed with detected face count and FPS overlay.
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


# Stable facial reference landmark: Nose bridge / Tip of nose (Landmark index 1 or 4)
# Landmark 1 is the tip of the nose, landmark 4 is the upper nasal bridge.
NOSE_REFERENCE_INDEX = 1
TOTAL_FACE_LANDMARKS = 468


def extract_and_normalize_face_landmarks(
    face_landmarks,
    reference_idx: int = NOSE_REFERENCE_INDEX
) -> np.ndarray:
    """
    Modular, reusable function to extract and normalize face landmarks.

    Args:
        face_landmarks: MediaPipe NormalizedLandmarkList for a detected face.
        reference_idx (int): Landmark index used as origin (default: 1, nose tip).

    Returns:
        np.ndarray: Normalized facial landmark matrix of shape (468, 3) where:
                    - Coordinates are centered at the nose reference point.
                    - Scale is normalized by the maximum facial distance (distance invariant).
                    If face_landmarks is None, returns zeros of shape (468, 3).
    """
    if face_landmarks is None:
        return np.zeros((TOTAL_FACE_LANDMARKS, 3), dtype=np.float32)

    # Convert landmarks to NumPy array of shape (468, 3)
    coords = []
    for lm in face_landmarks.landmark[:TOTAL_FACE_LANDMARKS]:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords, dtype=np.float32)

    # 1. Translation: Center relative to the stable nose reference landmark
    ref_point = coords[reference_idx, :]
    translated = coords - ref_point

    # 2. Scale: Normalize by max euclidean distance from reference point
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)

    eps = 1e-6
    if max_dist > eps:
        normalized = translated / max_dist
    else:
        normalized = translated

    return normalized.astype(np.float32)


def main():
    # Step 1: Initialize MediaPipe Face Mesh and drawing utilities
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Step 2: Open default webcam
    cap = cv2.VideoCapture(0)

    # Basic error handling
    if not cap.isOpened():
        print("Error: Could not open the default webcam (device index 0).", file=sys.stderr)
        print("Please check if the webcam is connected and accessible.", file=sys.stderr)
        return

    print("=" * 60)
    print(" LSE FACE TRACKING - READY")
    print(" Window:  'LSE Face Tracking'")
    print(" Controls: Press [Q] to quit")
    print("=" * 60)

    # Variables for FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0

    # Step 3: Configure MediaPipe FaceMesh (max_num_faces=1 for primary signer)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from webcam. Exiting...", file=sys.stderr)
                break

            # Step 4: Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Step 5: Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Step 6: Process facial landmarks
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            faces_detected_count = 0
            normalized_face_tensor = None

            # Step 7: Draw face mesh if detected
            if results.multi_face_landmarks:
                faces_detected_count = len(results.multi_face_landmarks)
                primary_face = results.multi_face_landmarks[0]

                # Draw facial mesh contours and tessellations
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=primary_face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=primary_face,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # Extract normalized coordinates (modular reusable call)
                normalized_face_tensor = extract_and_normalize_face_landmarks(primary_face)
            else:
                # Handle no-face detection safely
                normalized_face_tensor = extract_and_normalize_face_landmarks(None)

            # Step 8: Calculate FPS
            curr_frame_time = time.perf_counter()
            fps = 1.0 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0.0
            prev_frame_time = curr_frame_time

            # Step 9: Render status overlay
            # Semi-transparent dark background banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (20, 20, 20), -1)

            status_text = f"Faces Detected: {faces_detected_count} | FPS: {fps:.1f}"
            status_color = (0, 255, 0) if faces_detected_count > 0 else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

            footer_text = "Press [Q] to exit"
            cv2.putText(frame, footer_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            # Step 10: Show live feed window titled "LSE Face Tracking"
            cv2.imshow("LSE Face Tracking", frame)

            # Step 11: Safely handle quit key (Q / q)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("Quit key pressed. Closing face tracking...")
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed cleanly.")


if __name__ == "__main__":
    main()
