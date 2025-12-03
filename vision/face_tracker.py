# vision/face_tracker.py
import cv2
import numpy as np
from typing import Optional, Tuple
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiSetAnglesRequest


class FaceTracker:
    """
    Face detection and tracking using NAO's camera with head movement control
    """

    def __init__(self, camera_manager):
        """
        Initialize face tracker

        Args:
            camera_manager: CameraManager instance (should use NAO camera)
        """
        self.camera_manager = camera_manager

        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.last_face_center = None
        self.face_lost_frames = 0
        self.max_lost_frames = 30

        # Head angle tracking
        self.current_head_yaw = 0.0
        self.current_head_pitch = 0.0

    def detect_face(self, frame):
        """
        Detect face in frame

        Returns:
            tuple: (face_detected, face_center, face_rect, annotated_frame)
                - face_detected: bool
                - face_center: (x, y) or None
                - face_rect: (x, y, w, h) or None
                - annotated_frame: frame with face rectangle drawn
        """
        if frame is None:
            return False, None, None, None

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        annotated_frame = frame.copy()

        if len(faces) > 0:
            # Use largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Calculate center
            face_center = (x + w // 2, y + h // 2)

            # Draw rectangle and center point
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(annotated_frame, face_center, 5, (0, 0, 255), -1)

            # Draw frame center crosshair
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            cv2.line(annotated_frame, (frame_center_x - 20, frame_center_y),
                     (frame_center_x + 20, frame_center_y), (255, 0, 0), 2)
            cv2.line(annotated_frame, (frame_center_x, frame_center_y - 20),
                     (frame_center_x, frame_center_y + 20), (255, 0, 0), 2)

            # Add status text
            cv2.putText(annotated_frame, "Face: Detected",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show offset from center
            offset_x = face_center[0] - frame_center_x
            offset_y = face_center[1] - frame_center_y
            cv2.putText(annotated_frame, f"Offset: ({offset_x}, {offset_y})",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show current head angles
            cv2.putText(annotated_frame, f"Head Yaw: {self.current_head_yaw:.2f} rad",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(annotated_frame, f"Head Pitch: {self.current_head_pitch:.2f} rad",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            self.last_face_center = face_center
            self.face_lost_frames = 0

            return True, face_center, largest_face, annotated_frame
        else:
            # No face detected
            self.face_lost_frames += 1

            cv2.putText(annotated_frame, "Face: Not detected",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.face_lost_frames > self.max_lost_frames:
                self.last_face_center = None

            return False, self.last_face_center, None, annotated_frame

    def get_head_movement_angles(self, face_center, frame_shape):
        """
        Calculate absolute head angles needed to center the face

        Args:
            face_center: (x, y) position of face center
            frame_shape: (height, width, channels) of frame

        Returns:
            tuple: (yaw_angle, pitch_angle) in radians
                - yaw: horizontal rotation (left/right)
                - pitch: vertical rotation (up/down)
        """
        if face_center is None:
            return self.current_head_yaw, self.current_head_pitch

        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        # Calculate normalized offset (-1 to 1)
        offset_x = (face_center[0] - frame_center_x) / (frame_width // 2)
        offset_y = (face_center[1] - frame_center_y) / (frame_height // 2)

        # Convert to angle adjustments (scale factor determines sensitivity)
        max_yaw = 0.4  # ~23 degrees max adjustment
        max_pitch = 0.25  # ~14 degrees max adjustment

        yaw_adjustment = -offset_x * max_yaw  # Negative for correct direction
        pitch_adjustment = offset_y * max_pitch

        # Calculate new absolute angles
        new_yaw = self.current_head_yaw + yaw_adjustment
        new_pitch = self.current_head_pitch + pitch_adjustment

        # Clamp to NAO's head limits
        # HeadYaw: -2.0857 to 2.0857 radians (-119.5° to 119.5°)
        # HeadPitch: -0.6720 to 0.5149 radians (-38.5° to 29.5°)
        new_yaw = max(-2.0, min(2.0, new_yaw))
        new_pitch = max(-0.6, min(0.5, new_pitch))

        return new_yaw, new_pitch

    def move_nao_head_to_face(self, nao, face_center, frame_shape):
        """
        Move NAO's head to center the detected face using absolute positioning

        Args:
            nao: NAO robot instance
            face_center: (x, y) position of face center
            frame_shape: (height, width, channels) of frame

        Returns:
            bool: True if head was moved, False if already centered or error
        """
        if face_center is None:
            return False

        # Check if already centered
        if self.is_face_centered(face_center, frame_shape, tolerance=50):
            return False

        # Calculate target angles
        target_yaw, target_pitch = self.get_head_movement_angles(face_center, frame_shape)

        try:
            # Move head using SetAngles (absolute positioning)
            nao.motion.request(
                NaoqiSetAnglesRequest(
                    names=["HeadYaw", "HeadPitch"],
                    angles=[target_yaw, target_pitch],
                    fractionMaxSpeed=0.1  # Slow, smooth movement (10% of max speed)
                ),
                block=False
            )

            # Update tracked angles
            self.current_head_yaw = target_yaw
            self.current_head_pitch = target_pitch

            return True

        except Exception as e:
            print(f"Head movement error: {e}")
            return False

    def reset_head_position(self, nao):
        """
        Reset NAO's head to center position (0, 0)

        Args:
            nao: NAO robot instance

        Returns:
            bool: True if successful
        """
        try:
            nao.motion.request(
                NaoqiSetAnglesRequest(
                    names=["HeadYaw", "HeadPitch"],
                    angles=[0.0, 0.0],
                    fractionMaxSpeed=0.2
                ),
                block=True
            )

            self.current_head_yaw = 0.0
            self.current_head_pitch = 0.0

            print("Head position reset to center")
            return True

        except Exception as e:
            print(f"Head reset error: {e}")
            return False

    def is_face_centered(self, face_center, frame_shape, tolerance=50):
        """
        Check if face is centered in frame

        Args:
            face_center: (x, y) position of face center
            frame_shape: (height, width, channels) of frame
            tolerance: pixel tolerance for "centered"

        Returns:
            bool: True if face is centered
        """
        if face_center is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        distance = np.sqrt(
            (face_center[0] - frame_center_x) ** 2 +
            (face_center[1] - frame_center_y) ** 2
        )

        return distance < tolerance

    def cleanup(self):
        """Cleanup resources"""
        pass