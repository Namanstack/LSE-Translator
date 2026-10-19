"""
LSE Controlled Hand Landmark Dataset Collector - Milestone 2

Captures spatio-temporal hand landmark sequences (30 frames per gesture sample),
normalizes landmark positions relative to the wrist, and saves structured CSV
files under data/raw/<sign_label>/<signer_id>_<recording_id>.csv.
"""

import os
import sys
import argparse
import pandas as pd
import cv2
import mediapipe as mp


def get_unique_filepath(base_path: str) -> str:
    """
    Ensures existing recording files are never overwritten.
    If the file exists, appends an incremented suffix _copy1, _copy2, etc.
    """
    if not os.path.exists(base_path):
        return base_path

    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    counter = 1

    while True:
        candidate = os.path.join(directory, f"{name}_copy{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def main():
    # ---------------------------------------------------------
    # Parse CLI Arguments for Sign Label, Signer ID, Recording ID
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Collect hand landmark sequence dataset for LSE.")
    parser.add_argument("--label", type=str, default="HOLA", help="Sign label (e.g., HOLA, GRACIAS, AGUA)")
    parser.add_argument("--signer", type=str, default="signer01", help="Signer identifier (e.g., signer01)")
    parser.add_argument("--recording", type=int, default=1, help="Recording number / trial ID (e.g., 1)")
    parser.add_argument("--seq_len", type=int, default=30, help="Target sequence length in frames (default: 30)")
    args = parser.parse_args()

    sign_label = args.label.strip().upper()
    signer_id = args.signer.strip().lower()
    recording_id = args.recording
    sequence_length = args.seq_len

    # Output directory: data/raw/<sign_label>/
    output_dir = os.path.join("data", "raw", sign_label)
    os.makedirs(output_dir, exist_ok=True)

    target_filename = f"{signer_id}_{recording_id:03d}.csv"
    target_filepath = os.path.join(output_dir, target_filename)
    safe_filepath = get_unique_filepath(target_filepath)

    if safe_filepath != target_filepath:
        print(f"[Notice] '{target_filename}' already exists. Output will be saved to: '{os.path.basename(safe_filepath)}'")

    # ---------------------------------------------------------
    # Initialize MediaPipe Hands & Video Capture
    # ---------------------------------------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default webcam (index 0).", file=sys.stderr)
        return

    print("=" * 60)
    print(" LSE DATASET COLLECTOR - READY")
    print(f" Sign Label:     {sign_label}")
    print(f" Signer ID:      {signer_id}")
    print(f" Recording File: {os.path.basename(safe_filepath)}")
    print(f" Target Length:  {sequence_length} frames")
    print(" Controls:       [SPACE] Start/Record | [R] Reset | [Q] Quit")
    print("=" * 60)

    # State tracking
    is_recording = False
    current_frame_idx = 0
    sequence_records = []  # List of dict rows for DataFrame

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera. Exiting...", file=sys.stderr)
                break

            # 1. Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # 2. Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # 3. Detect hand landmarks
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            detected_hands_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            # 4. Draw landmarks on preview
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            # ---------------------------------------------------------
            # 5. Capture Logic (when recording is active)
            # ---------------------------------------------------------
            if is_recording:
                # Map handedness label to landmark set
                # MediaPipe handedness on flipped frame: "Left" / "Right"
                detected_hands_map = {}
                if results.multi_hand_landmarks and results.multi_handedness:
                    for handedness, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                        label = handedness.classification[0].label  # "Left" or "Right"
                        detected_hands_map[label] = landmarks

                # For both hand slots (Left, Right), extract and normalize landmarks
                for hand_side in ["Left", "Right"]:
                    if hand_side in detected_hands_map:
                        lms = detected_hands_map[hand_side].landmark
                        # Wrist landmark (index 0) is the reference origin for normalization
                        wrist_x = lms[0].x
                        wrist_y = lms[0].y
                        wrist_z = lms[0].z

                        for idx, lm in enumerate(lms):
                            # Wrist-relative normalization
                            norm_x = lm.x - wrist_x
                            norm_y = lm.y - wrist_y
                            norm_z = lm.z - wrist_z

                            sequence_records.append({
                                "frame": current_frame_idx,
                                "hand": hand_side,
                                "landmark_index": idx,
                                "x": norm_x,
                                "y": norm_y,
                                "z": norm_z
                            })
                    else:
                        # If hand is not detected in this frame, populate with NaN for all 21 indices
                        for idx in range(21):
                            sequence_records.append({
                                "frame": current_frame_idx,
                                "hand": hand_side,
                                "landmark_index": idx,
                                "x": float("nan"),
                                "y": float("nan"),
                                "z": float("nan")
                            })

                current_frame_idx += 1

                # Check if target sequence length is reached
                if current_frame_idx >= sequence_length:
                    is_recording = False

                    # Validation: Only save if sequence is complete
                    if current_frame_idx == sequence_length and len(sequence_records) > 0:
                        df = pd.DataFrame(sequence_records)
                        df.to_csv(safe_filepath, index=False)
                        print(f"\n[Saved] Sequence successfully saved ({sequence_length} frames) to:")
                        print(f"        {safe_filepath}")
                        print("Ready for next sequence. Press [SPACE] to record again or [Q] to quit.\n")
                        # Advance recording ID for convenient subsequent captures
                        recording_id += 1
                        target_filename = f"{signer_id}_{recording_id:03d}.csv"
                        target_filepath = os.path.join(output_dir, target_filename)
                        safe_filepath = get_unique_filepath(target_filepath)
                    else:
                        print("\n[Warning] Incomplete sequence discarded.")

                    # Reset buffer
                    current_frame_idx = 0
                    sequence_records = []

            # ---------------------------------------------------------
            # 6. On-Screen Display (OSD) Overlay
            # ---------------------------------------------------------
            # Dark status bar at top for legibility
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (20, 20, 20), -1)

            # Metadata texts
            info_line1 = f"Label: {sign_label} | Signer: {signer_id} | Rec ID: {recording_id:03d}"
            rec_status = f"RECORDING ({current_frame_idx}/{sequence_length})" if is_recording else f"IDLE (Ready: {sequence_length} frames)"
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)
            hands_text = f"Hands in View: {detected_hands_count}"

            cv2.putText(frame, info_line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, rec_status, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
            cv2.putText(frame, hands_text, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # Controls footer at bottom
            footer_text = "[SPACE] Record/Capture | [R] Reset | [Q] Quit"
            cv2.putText(frame, footer_text, (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            # 7. Display Window
            cv2.imshow("LSE Hand Tracking - Dataset Collector", frame)

            # 8. Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACE: Start recording
                if not is_recording:
                    is_recording = True
                    current_frame_idx = 0
                    sequence_records = []
                    print(f"[Recording Started] Capturing {sequence_length} frames for '{sign_label}'...")
            elif key in (ord('r'), ord('R')):  # R: Reset
                is_recording = False
                current_frame_idx = 0
                sequence_records = []
                print("[Reset] Current sequence cancelled.")
            elif key in (ord('q'), ord('Q')):  # Q: Quit
                print("[Quit] Exiting dataset collector...")
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and application closed safely.")


if __name__ == "__main__":
    main()
