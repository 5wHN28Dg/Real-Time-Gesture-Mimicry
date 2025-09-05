import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Any


class SimpleHandTracker:
    def __init__(self):
        # Initialize MediaPipe Hands - we use Google's pre-trained hand detection model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False for video processing
            max_num_hands=1,  # We only need to track one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        # Used for drawing hand landmarks on the image
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_hand_openness(
        self, hand_landmarks
    ) -> Tuple[List[float], dict[str, np.ndarray]]:
        """
        Calculate how open each finger is based on the 3D distance between fingertips and key points on the palm.
        Returns a list of values between 0 (closed) and 1 (open) for each finger (pinky to thumb).
        """
        # Retrieve the MCPs of each finger and the wrist
        wrist = np.array(
            [
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z,
            ]
        )
        index_MCP = np.array(
            [
                hand_landmarks.landmark[5].x,
                hand_landmarks.landmark[5].y,
                hand_landmarks.landmark[5].z,
            ]
        )
        pinky_MCP = np.array(
            [
                hand_landmarks.landmark[17].x,
                hand_landmarks.landmark[17].y,
                hand_landmarks.landmark[17].z,
            ]
        )
        middle_finger_MCP = np.array(
            [
                hand_landmarks.landmark[9].x,
                hand_landmarks.landmark[9].y,
                hand_landmarks.landmark[9].z,
            ]
        )
        ring_finger_MCP = np.array(
            [
                hand_landmarks.landmark[13].x,
                hand_landmarks.landmark[13].y,
                hand_landmarks.landmark[13].z,
            ]
        )

        # Calculate a reference point for openness value for each finger
        index_finger_center = np.mean(np.array([wrist, index_MCP]), axis=0)
        middle_finger_center = np.mean(np.array([wrist, middle_finger_MCP]), axis=0)
        ring_finger_center = np.mean(np.array([wrist, ring_finger_MCP]), axis=0)
        pinky_finger_center = np.mean(np.array([wrist, pinky_MCP]), axis=0)

        custom_centers = {
            "index_center": index_finger_center,
            "middle_center": middle_finger_center,
            "ring_center": ring_finger_center,
            "pinky_center": pinky_finger_center,
        }

        # Order matches fingertip_indices: pinky, ring, middle, index, thumb
        centers = [
            pinky_finger_center,
            ring_finger_center,
            middle_finger_center,
            index_finger_center,
        ]

        # Use the distance between index MCP and pinky MCP as the hand scale factor
        hand_scale = np.linalg.norm(index_MCP - pinky_MCP) + 1e-6

        # Fingertip indices in MediaPipe hand model (pinky to thumb)
        fingertip_indices = [20, 16, 12, 8, 4]

        # Calculate average distance from fingertips to the palm center
        distances = []
        for i, tip_idx in enumerate(fingertip_indices):
            tip_pos = np.array(
                [
                    hand_landmarks.landmark[tip_idx].x,
                    hand_landmarks.landmark[tip_idx].y,
                    hand_landmarks.landmark[tip_idx].z,
                ]
            )

            # Calculate Euclidean distance from the thumb tip to the ring MCP
            if tip_idx == 4:  # Thumb
                distance = np.linalg.norm(tip_pos - ring_finger_MCP)
                distances.append(distance)
            # For all other fingers, use the corresponding center
            else:
                center_idx = min(
                    i, len(centers) - 1
                )  # Ensure we don't go out of bounds
                distance = np.linalg.norm(tip_pos - centers[center_idx])
                distances.append(distance)

        # Normalize the distances by the hand scale to get relative measures
        normalized_distances = [x / hand_scale for x in distances]

        # Define empirical calibration values; adjust these based on your setup
        baseline_finger = 1  # Example baseline for finger openness
        baseline_thumb = 0.45  # Example baseline for thumb openness

        # Define a range for normalization (this represents the expected variation between closed and open)
        range_val = 0.9

        # Apply different baselines for thumb vs. other fingers
        normalized_values = []
        for i, dist in enumerate(normalized_distances):
            if i == 4:  # Thumb (last element in our list)
                norm_val = np.clip((dist - baseline_thumb) / range_val, 0, 1)
            else:
                norm_val = np.clip((dist - baseline_finger) / range_val, 0, 1)
            normalized_values.append(norm_val)

        return normalized_values, custom_centers

    def process_frame(self, frame) -> Tuple[Any, float, float, float, float, float]:
        """
        Process a video frame and return the processed frame and hand openness values.

        Args:
            frame: Video frame from webcam

        Returns:
            Tuple containing:
            - Processed frame with drawings
            - Hand openness values (0 to 1) for pinky, ring, middle, index, thumb
              or default 1 values if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.hands.process(rgb_frame)
        h, w, _ = frame.shape

        # Default values if no hand is detected
        pinky_openness = 1
        ring_openness = 1
        middle_openness = 1
        index_openness = 1
        thumb_openness = 1

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on frame for visualization
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            # Calculate hand openness for all fingers (pinky to thumb)
            openness_values, custom_centers = self.calculate_hand_openness(
                hand_landmarks
            )

            # Assign values to individual fingers (order is pinky to thumb)
            pinky_openness = openness_values[0]
            ring_openness = openness_values[1]
            middle_openness = openness_values[2]
            index_openness = openness_values[3]
            thumb_openness = openness_values[4]

            # Display the openness values on frame
            cv2.putText(
                frame,
                f"Pinky openness: {pinky_openness:.2f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Ring openness: {ring_openness:.2f}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Middle finger openness: {middle_openness:.2f}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Index finger openness: {index_openness:.2f}",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Thumb openness: {thumb_openness:.2f}",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            for name, center in custom_centers.items():
                cx, cy = int(center[0] * w), int(center[1] * h)
                # Draw a filled circle
                cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

        return (
            frame,
            pinky_openness,
            ring_openness,
            middle_openness,
            index_openness,
            thumb_openness,
        )

    def run_test(self):
        """
        Test function to run hand tracking with webcam feed.
        Shows the video feed and prints the servo angles.
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break

            # Process the frame
            (
                frame,
                pinky_openness,
                ring_openness,
                middle_openness,
                index_openness,
                thumb_openness,
            ) = self.process_frame(frame)

            # Convert hand openness to servo angle (0 to 180 degrees)
            servo_angle1 = int(pinky_openness * 180)
            servo_angle2 = int(ring_openness * 180)
            servo_angle3 = int(middle_openness * 180)
            servo_angle4 = int(index_openness * 180)
            servo_angle5 = int(thumb_openness * 180)

            # Display servo angles with proper spacing
            cv2.putText(
                frame,
                f"Pinky servo angle: {servo_angle1}",
                (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Ring servo angle: {servo_angle2}",
                (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Middle servo angle: {servo_angle3}",
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Index servo angle: {servo_angle4}",
                (10, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Thumb servo angle: {servo_angle5}",
                (10, 370),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Show the frame
            cv2.imshow("Hand Tracking", frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = SimpleHandTracker()
    tracker.run_test()
