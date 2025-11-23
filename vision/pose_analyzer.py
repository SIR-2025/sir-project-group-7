import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

MODEL_PATH = r"../pose_landmarkers/pose_landmarker_full.task"

POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])


class PoseAnalyzer:
    """
    Analyzes body pose using MediaPipe and calculates joint angles
    """

    def __init__(self, camera_manager=None):
        self.camera_manager = camera_manager

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print("Pose analyzer initialized")

    def analyze_frame(self, frame):
        """
        Analyze pose in a single frame
        """
        if frame is None:
            return None, None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.landmarker.detect(mp_image)

            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                angles = self._calculate_angles(landmarks)
                annotated_frame = self._draw_landmarks(frame.copy(), landmarks)
                return angles, annotated_frame

            return None, frame

        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return None, frame

    def capture_and_analyze(self):
        """
        Capture frame from camera and analyze pose
        """
        if not self.camera_manager:
            print("No camera manager configured")
            return None, None

        frame = self.camera_manager.capture_frame()

        if frame is None:
            return None, None

        return self.analyze_frame(frame)

    def _calculate_angles(self, landmarks):
        """
        Calculate key joint angles from pose landmarks
        """
        angles = {}

        angles['left_knee'] = self._calculate_angle(
            landmarks[23],
            landmarks[25],
            landmarks[27]
        )

        angles['right_knee'] = self._calculate_angle(
            landmarks[24],
            landmarks[26],
            landmarks[28]
        )

        angles['left_hip'] = self._calculate_angle(
            landmarks[11],
            landmarks[23],
            landmarks[25]
        )

        angles['right_hip'] = self._calculate_angle(
            landmarks[12],
            landmarks[24],
            landmarks[26]
        )

        angles['back_angle'] = self._calculate_angle(
            landmarks[11],
            landmarks[23],
            landmarks[27]
        )

        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        knee_y = (landmarks[25].y + landmarks[26].y) / 2
        angles['squat_depth_ratio'] = abs(hip_y - knee_y)

        return angles

    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        """
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _draw_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks and connections on frame
        """
        h, w = frame.shape[:2]

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame

    def check_squat_form(self, angles, target_angles=None):
        """
        Check if squat form is correct based on angles
        """
        if target_angles is None:
            target_angles = {
                'left_knee': 90,
                'right_knee': 90,
                'left_hip': 90,
                'right_hip': 90,
                'back_angle': 10,
                'squat_depth_ratio': 0.15
            }

        analysis = {
            'joints': {},
            'overall_accuracy': 0,
            'is_correct': True
        }

        total_error = 0
        joint_count = 0

        for joint, target in target_angles.items():
            if joint == 'squat_depth_ratio':
                continue

            current = angles.get(joint, 0)
            error = abs(target - current)

            analysis['joints'][joint] = {
                'current_angle': round(current, 1),
                'target_angle': round(target, 1),
                'error_degrees': round(error, 1),
                'status': 'good' if error < 15 else 'needs_adjustment'
            }

            if error >= 15:
                analysis['is_correct'] = False

            total_error += error
            joint_count += 1

        if joint_count > 0:
            avg_error = total_error / joint_count
            analysis['overall_accuracy'] = max(0, 100 - avg_error)

        depth_ratio = angles.get('squat_depth_ratio', 0)
        analysis['squat_depth'] = {
            'ratio': round(depth_ratio, 3),
            'status': 'good' if depth_ratio > 0.12 else 'too_shallow'
        }

        return analysis

    def cleanup(self):
        cv2.destroyAllWindows()
        print("Pose analyzer resources released")