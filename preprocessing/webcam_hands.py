"""
LSE Hand Tracking - Milestone 1
Real-time hand tracking and landmark detection using OpenCV and MediaPipe Hands.
"""

import sys
import cv2
import mediapipe as mp


def main():
    # Step 1: Initialize MediaPipe Hands and drawing utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Step 2: Open the default webcam (device index 0)
    cap = cv2.VideoCapture(0)

    # Basic error handling: check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the default webcam (device index 0).", file=sys.stderr)
        print("Please check if the camera is connected and not in use by another program.", file=sys.stderr)
        return

    print("Webcam started successfully.")
    print("Window: 'LSE Hand Tracking'")
    print("Press 'q' or 'Q' to exit.")

    # Step 3: Configure MediaPipe Hands instance
    # max_num_hands=2 detects a maximum of 2 hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from webcam. Exiting...", file=sys.stderr)
                break

            # Step 4: Flip frame horizontally for a natural selfie-view / mirror display
            frame = cv2.flip(frame, 1)

            # Step 5: Convert BGR image to RGB as required by MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improve performance by marking the image as not writeable during processing
            rgb_frame.flags.writeable = False

            # Step 6: Process the frame and detect hand landmarks
            results = hands.process(rgb_frame)

            # Mark image writeable again for drawing
            rgb_frame.flags.writeable = True

            # Step 7: Count detected hands and draw landmarks/connections
            detected_hands_count = 0

            if results.multi_hand_landmarks:
                detected_hands_count = len(results.multi_hand_landmarks)

                # Draw 21 landmarks and hand connections for each detected hand
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Step 8: Display detected hands count overlay on screen
            status_text = f"Hands Detected: {detected_hands_count}"
            cv2.putText(
                img=frame,
                text=status_text,
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # Step 9: Show live feed in a window titled "LSE Hand Tracking"
            cv2.imshow("LSE Hand Tracking", frame)

            # Step 10: Exit safely when 'q' or 'Q' key is pressed
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("Exit key pressed. Closing application...")
                break

    # Step 11: Release camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed cleanly.")


if __name__ == "__main__":
    main()
