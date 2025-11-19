import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

MODEL_PATH = "../pose_landmarkers/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Skeleton
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])

# Landmarks
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class SquatValidator:
    def __init__(self):
        # Thresholds
        self.squat_threshold = 120
        self.hip_threshold = 90

    def calculate_angle(self, point1, point2, point3):
        # Angle calculation
        vector1 = (point1.x - point2.x, point1.y - point2.y)
        vector2 = (point3.x - point2.x, point3.y - point2.y)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        mag1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        mag2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if mag1 * mag2 == 0:
            return 180
        angle = math.acos(dot_product / (mag1 * mag2))
        return math.degrees(angle)

    def calculate_back_angle(self, landmarks):
        # Back straightness
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
        hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2

        vertical_diff = abs(shoulder_y - hip_y)
        horizontal_diff = abs(shoulder_x - hip_x)

        if vertical_diff == 0:
            return 90

        angle = math.degrees(math.atan(horizontal_diff / vertical_diff))
        return angle

    def is_valid_squat(self, landmarks):
        if len(landmarks) < 29:
            return False, ["Not enough landmarks"], {}

        # Angles
        left_knee_angle = self.calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right_knee_angle = self.calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        left_hip_angle = self.calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
        right_hip_angle = self.calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])
        back_angle = self.calculate_back_angle(landmarks)

        min_knee_angle = min(left_knee_angle, right_knee_angle)
        max_hip_angle = max(left_hip_angle, right_hip_angle)

        feedback = []
        is_valid = True

        # Validation
        if min_knee_angle > self.squat_threshold:
            feedback.append(f"Knees not bent enough ({min_knee_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Knee bend: {min_knee_angle:.1f}° ✓")

        if max_hip_angle < self.hip_threshold:
            feedback.append(f"Hips not low enough ({max_hip_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Hip position: {max_hip_angle:.1f}° ✓")

        if back_angle > 20:
            feedback.append(f"Back leaning ({back_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Back straight: {back_angle:.1f}° ✓")

        return is_valid, feedback, {
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'back_angle': back_angle
        }


def draw_angle_info(frame, landmarks, angles, w, h):
    # Angle display
    left_knee_pos = (int(landmarks[LEFT_KNEE].x * w), int(landmarks[LEFT_KNEE].y * h))
    right_knee_pos = (int(landmarks[RIGHT_KNEE].x * w), int(landmarks[RIGHT_KNEE].y * h))

    cv2.putText(frame, f"L: {angles['left_knee_angle']:.1f}°",
                (left_knee_pos[0] - 30, left_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"R: {angles['right_knee_angle']:.1f}°",
                (right_knee_pos[0] - 30, right_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    left_hip_pos = (int(landmarks[LEFT_HIP].x * w), int(landmarks[LEFT_HIP].y * h))
    cv2.putText(frame, f"Hip: {angles['left_hip_angle']:.1f}°",
                (left_hip_pos[0] - 30, left_hip_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_guidelines(frame, w, h):
    # Instructions
    guidelines = [
        "Squat Form Guide:",
        "1. Knees bent < 120°",
        "2. Hips low > 150°",
        "3. Back straight"
    ]

    for i, line in enumerate(guidelines):
        cv2.putText(frame, line, (w - 300, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# Main
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
squat_validator = SquatValidator()

with PoseLandmarker.create_from_options(options) as landmarker:
    ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts)

        squat_status = "No pose detected"
        feedback_lines = []
        angles_info = {}

        if result and result.pose_landmarks:
            for pose_lms in result.pose_landmarks:
                # Draw skeleton
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                        start = pose_lms[start_idx]
                        end = pose_lms[end_idx]
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                for lm in pose_lms:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                # Validate
                is_valid, feedback, angles = squat_validator.is_valid_squat(pose_lms)
                angles_info = angles

                squat_status = "VALID SQUAT! ✓" if is_valid else "INVALID SQUAT"
                feedback_lines = feedback

                draw_angle_info(frame, pose_lms, angles, w, h)

        draw_guidelines(frame, w, h)

        # Status display
        status_color = (0, 255, 0) if "VALID" in squat_status else (0, 0, 255)
        cv2.putText(frame, f"Status: {squat_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        y_offset = 60
        for line in feedback_lines:
            color = (255, 255, 255)
            if "✓" in line:
                color = (0, 255, 0)
            elif any(word in line.lower() for word in ["not", "leaning"]):
                color = (0, 165, 255)

            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        cv2.imshow("Squat Pose Validator", frame)
        ts += 33

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()