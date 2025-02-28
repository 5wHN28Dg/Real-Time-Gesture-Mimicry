import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Any


class SimpleHandTracker:
    def __init__(self):
        # Initialize MediaPipe Hands - we use Google's pre-trained hand detection model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False for video processing
            max_num_hands=1,  # We only need to track one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # Used for drawing hand landmarks on the image
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_hand_openness(self, hand_landmarks) -> tuple[Any, Any]:
        """
        Calculate how open the hand is based on the 3D distance between fingertips and palm center.
        Returns a value between 0 (closed hand) and 1 (open hand).
        """
        # Get palm center by retrieving key landmarks in 3D (wrist, index MCP, pinky MCP)
        wrist = np.array([
            hand_landmarks.landmark[0].x,
            hand_landmarks.landmark[0].y,
            hand_landmarks.landmark[0].z
        ])
        index_MCP = np.array([
            hand_landmarks.landmark[5].x,
            hand_landmarks.landmark[5].y,
            hand_landmarks.landmark[5].z
        ])
        pinky_MCP = np.array([
            hand_landmarks.landmark[17].x,
            hand_landmarks.landmark[17].y,
            hand_landmarks.landmark[17].z
        ])
        # Calculate a more robust palm center by averaging wrist, index MCP, and pinky MCP
        palm_center = np.mean( np.array([wrist, index_MCP, pinky_MCP]), axis=0 )

        # Use the distance between index MCP and pinky MCP as the hand scale factor
        hand_scale = np.linalg.norm(index_MCP - pinky_MCP)

        # Fingertip indices in MediaPipe hand model (thumb to pinky)
        fingertip_indices = [4, 8, 12, 16, 20]

        # Calculate average distance from fingertips to the palm center
        distances = []
        for tip_idx in fingertip_indices:
            tip_pos = np.array([
                hand_landmarks.landmark[tip_idx].x,
                hand_landmarks.landmark[tip_idx].y,
                hand_landmarks.landmark[tip_idx].z
            ])
            # Calculate Euclidean distance
            distance = np.linalg.norm(tip_pos - palm_center)
            distances.append(distance)

        # Average distances for the four fingers (excluding the thumb)
        avg_distance = np.mean(distances[1:])
        thumb_distance = distances[0]

        # Normalize the distances by the hand scale to get relative measures
        norm_avg = avg_distance / hand_scale
        norm_thumb = thumb_distance / hand_scale

        # Define empirical calibration values; adjust these based on your setup
        # They could differ between left and right hands if necessary using an if statement
        baseline_finger = 1   # Example baseline for left hand overall finger openness
        baseline_thumb = 0.45 # Example baseline for left thumb openness

        # Define a range for normalization (this represents the expected variation between closed and open)
        range_val = 0.9
        normalized_distance = np.clip((norm_avg - baseline_finger) / range_val, 0, 1)
        normalized_thumb_distance = np.clip((norm_thumb - baseline_thumb) / range_val, 0, 1)

        return normalized_distance, normalized_thumb_distance

    def process_frame(self, frame) -> Tuple[Any, int | Any, Any]:
        """
        Process a video frame and return the processed frame and hand openness value.

        Args:
            frame: Video frame from webcam

        Returns:
            Tuple containing:
            - Processed frame with drawings
            - Hand openness value (0 to 1) or 1 if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.hands.process(rgb_frame)

        hand_openness = 1  # Default value if no hand is detected
        thumb_openness = 1

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on frame for visualization
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Calculate hand openness
            hand_openness = self.calculate_hand_openness(hand_landmarks)[0]
            thumb_openness = self.calculate_hand_openness(hand_landmarks)[1]

            # Display the openness value on frame
            cv2.putText(frame, f"Hand openness: {hand_openness:.2f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.putText(frame, f"Thumb openness: {thumb_openness:.2f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        return frame, hand_openness, thumb_openness

    def run_test(self):
        """
        Test function to run hand tracking with webcam feed.
        Shows the video feed and prints the servo angle.
        """
        cap = cv2.VideoCapture(0)


        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break

            # Process the frame
            frame, hand_openness, thumb_openness = self.process_frame(frame)

            # Convert hand openness to servo angle (0 to 180 degrees)
            servo_angle = int(hand_openness * 180)

            # Display servo angle
            cv2.putText(frame, f"Servo angle: {servo_angle}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Hand Tracking', frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = SimpleHandTracker()
    tracker.run_test()