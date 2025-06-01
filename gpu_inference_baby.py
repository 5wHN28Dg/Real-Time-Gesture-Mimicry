import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Any


class GPUHandTracker:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize MediaPipe Hand Landmarker with GPU support using Tasks API.
        """
        # Download the model file first:
        # wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
        
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Configure GPU delegate
        if use_gpu:
            # For GPU acceleration
            base_options = python.BaseOptions(
                model_asset_path='hand_landmarker.task',
                delegate=python.BaseOptions.Delegate.GPU
            )
        else:
            # CPU fallback
            base_options = python.BaseOptions(
                model_asset_path='hand_landmarker.task'
            )
        
        # Hand landmarker options
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Pre-allocate arrays for performance
        self._landmark_array = np.zeros((21, 3), dtype=np.float32)
        self._fingertip_indices = np.array([20, 16, 12, 8, 4], dtype=np.int32)
        self._mcp_indices = np.array([17, 13, 9, 5], dtype=np.int32)
        
        # Calibration constants
        self.baseline_finger = 1.0
        self.baseline_thumb = 0.45
        self.range_val = 0.9

    def _extract_landmarks_vectorized(self, hand_landmarks) -> np.ndarray:
        """Extract landmarks into numpy array for vectorized operations."""
        for i, landmark in enumerate(hand_landmarks):
            self._landmark_array[i] = [landmark.x, landmark.y, landmark.z]
        return self._landmark_array

    def calculate_hand_openness(self, hand_landmarks) -> Tuple[List[float], dict[str, np.ndarray]]:
        """
        Vectorized calculation of finger openness.
        """
        # Extract all landmarks at once
        landmarks = self._extract_landmarks_vectorized(hand_landmarks)
        
        # Get key points using array indexing
        wrist = landmarks[0]
        mcps = landmarks[self._mcp_indices]  # [pinky_MCP, ring_MCP, middle_MCP, index_MCP]
        fingertips = landmarks[self._fingertip_indices]  # [pinky_tip, ring_tip, middle_tip, index_tip, thumb_tip]
        
        # Calculate centers vectorized
        centers = np.mean(np.stack([np.tile(wrist, (4, 1)), mcps]), axis=0)
        
        custom_centers = {
            "index_center": centers[3],
            "middle_center": centers[2], 
            "ring_center": centers[1],
            "pinky_center": centers[0]
        }
        
        # Hand scale calculation
        hand_scale = np.linalg.norm(mcps[3] - mcps[0]) + 1e-6
        
        # Calculate distances vectorized
        distances = np.zeros(5)
        
        # Four fingers (pinky to index) - vectorized distance calculation
        finger_distances = np.linalg.norm(fingertips[:4] - centers, axis=1)
        distances[:4] = finger_distances
        
        # Thumb distance (to ring MCP)
        distances[4] = np.linalg.norm(fingertips[4] - mcps[1])
        
        # Normalize distances
        normalized_distances = distances / hand_scale
        
        # Apply baselines vectorized
        baselines = np.array([self.baseline_finger] * 4 + [self.baseline_thumb])
        normalized_values = np.clip((normalized_distances - baselines) / self.range_val, 0, 1)
        
        return normalized_values.tolist(), custom_centers

    def process_frame(self, frame) -> Tuple[Any, float, float, float, float, float]:
        """
        Process frame using GPU-accelerated Tasks API.
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Default values
        pinky_openness = ring_openness = middle_openness = index_openness = thumb_openness = 1.0
        
        # Detect hand landmarks (GPU accelerated)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Get first hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Draw landmarks
            self._draw_landmarks(frame, hand_landmarks, w, h)
            
            # Calculate openness
            openness_values, custom_centers = self.calculate_hand_openness(hand_landmarks)
            
            pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness = openness_values
            
            # Display values
            self._display_values(frame, openness_values)
            self._draw_centers(frame, custom_centers, w, h)
        
        return frame, pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness

    def _draw_landmarks(self, frame, hand_landmarks, w, h):
        """Draw hand landmarks on frame."""
        # Convert normalized coordinates to pixel coordinates and draw
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    def _display_values(self, frame, openness_values):
        """Display openness values on frame."""
        finger_names = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
        for i, (name, value) in enumerate(zip(finger_names, openness_values)):
            y_pos = 110 + i * 15
            cv2.putText(frame, f"{name} openness: {value:.2f}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_centers(self, frame, custom_centers, w, h):
        """Draw finger centers on frame."""
        for center in custom_centers.values():
            cx, cy = int(center[0] * w), int(center[1] * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    def run_test(self):
        """Test function with webcam feed."""
        cap = cv2.VideoCapture(0)
        
        print("GPU Hand Tracker started. Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break
            
            # Process frame
            frame, pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness = self.process_frame(frame)
            
            # Convert to servo angles
            servo_angles = [int(openness * 180) for openness in 
                          [pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness]]
            
            # Display servo angles
            finger_names = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
            for i, (name, angle) in enumerate(zip(finger_names, servo_angles)):
                y_pos = 250 + i * 30
                cv2.putText(frame, f"{name} servo angle: {angle}",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('GPU Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # First download the model:
    # wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    
    try:
        tracker = GPUHandTracker(use_gpu=True)
        tracker.run_test()
    except Exception as e:
        print(f"GPU failed, falling back to CPU: {e}")
        tracker = GPUHandTracker(use_gpu=False)
        tracker.run_test()