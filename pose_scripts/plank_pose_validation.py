import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

MODEL_PATH = "../pose_landmarkers/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])

# Landmark indices for key body parts
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


class PlankValidator:
    # thresholds
    def __init__(self):
        self.back_angle_threshold = 10
        self.hip_angle_threshold = 180
        self.shoulder_angle_threshold = 90
        self.elbow_angle_threshold = 90

    def calculate_angle(self, point1, point2, point3):

        # Get angles between specific points
        vector1 = (point1.x - point2.x, point1.y - point2.y)
        vector2 = (point3.x - point2.x, point3.y - point2.y)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        mag1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        mag2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate angle in degrees
        if mag1 * mag2 == 0:
            return 180
        angle = math.acos(dot_product / (mag1 * mag2))
        return math.degrees(angle)

    def calculate_back_straightness(self, landmarks):
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2

        vertical_diff = abs(shoulder_y - hip_y)

        if vertical_diff == 0:
            return 0

        return vertical_diff * 100

    def is_valid_plank(self, landmarks):
        if len(landmarks) < 29:
            return False, ["Not enough landmarks detected"], {}

        left_shoulder_angle = self.calculate_angle(
            landmarks[LEFT_ELBOW],
            landmarks[LEFT_SHOULDER],
            landmarks[LEFT_HIP]
        )
        right_shoulder_angle = self.calculate_angle(
            landmarks[RIGHT_ELBOW],
            landmarks[RIGHT_SHOULDER],
            landmarks[RIGHT_HIP]
        )

        left_elbow_angle = self.calculate_angle(
            landmarks[LEFT_WRIST],
            landmarks[LEFT_ELBOW],
            landmarks[LEFT_SHOULDER]
        )
        right_elbow_angle = self.calculate_angle(
            landmarks[RIGHT_WRIST],
            landmarks[RIGHT_ELBOW],
            landmarks[RIGHT_SHOULDER]
        )

        left_hip_angle = self.calculate_angle(
            landmarks[LEFT_SHOULDER],
            landmarks[LEFT_HIP],
            landmarks[LEFT_KNEE]
        )
        right_hip_angle = self.calculate_angle(
            landmarks[RIGHT_SHOULDER],
            landmarks[RIGHT_HIP],
            landmarks[RIGHT_KNEE]
        )

        back_straightness = self.calculate_back_straightness(landmarks)

        # Check plank criteria
        avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

        feedback = []
        is_valid = True

        # Shoulder position check
        if abs(avg_shoulder_angle - self.shoulder_angle_threshold) > 20:
            feedback.append(f"Shoulders not at 90° ({avg_shoulder_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Shoulders: {avg_shoulder_angle:.1f}° ✓")

        # Elbow position check
        if abs(avg_elbow_angle - self.elbow_angle_threshold) > 20:
            feedback.append(f"Elbows not at 90° ({avg_elbow_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Elbows: {avg_elbow_angle:.1f}° ✓")

        # Hip straightness check
        if abs(avg_hip_angle - self.hip_angle_threshold) > 15:
            feedback.append(f"Hips not straight ({avg_hip_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Hips straight: {avg_hip_angle:.1f}° ✓")

        # Back straightness check
        if back_straightness > 5:
            if landmarks[LEFT_HIP].y > landmarks[LEFT_SHOULDER].y:
                feedback.append(f"Hips sagging ({back_straightness:.1f})")
            else:
                feedback.append(f"Hips too high ({back_straightness:.1f})")
            is_valid = False
        else:
            feedback.append(f"Back straight: {back_straightness:.1f} ✓")

        return is_valid, feedback, {
            'left_shoulder_angle': left_shoulder_angle,
            'right_shoulder_angle': right_shoulder_angle,
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'back_straightness': back_straightness
        }


def draw_angle_info(frame, landmarks, angles, w, h):
    # Draw shoulder angles
    left_shoulder_pos = (int(landmarks[LEFT_SHOULDER].x * w), int(landmarks[LEFT_SHOULDER].y * h))
    cv2.putText(frame, f"S: {angles['left_shoulder_angle']:.1f}°",
                (left_shoulder_pos[0] - 30, left_shoulder_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw elbow angles
    left_elbow_pos = (int(landmarks[LEFT_ELBOW].x * w), int(landmarks[LEFT_ELBOW].y * h))
    cv2.putText(frame, f"E: {angles['left_elbow_angle']:.1f}°",
                (left_elbow_pos[0] - 30, left_elbow_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw hip angles
    left_hip_pos = (int(landmarks[LEFT_HIP].x * w), int(landmarks[LEFT_HIP].y * h))
    cv2.putText(frame, f"H: {angles['left_hip_angle']:.1f}°",
                (left_hip_pos[0] - 30, left_hip_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_plank_line(frame, landmarks, w, h):
    if len(landmarks) >= 29:
        # Get average shoulder and hip positions
        left_shoulder = (int(landmarks[LEFT_SHOULDER].x * w), int(landmarks[LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[RIGHT_SHOULDER].x * w), int(landmarks[RIGHT_SHOULDER].y * h))
        left_hip = (int(landmarks[LEFT_HIP].x * w), int(landmarks[LEFT_HIP].y * h))
        right_hip = (int(landmarks[RIGHT_HIP].x * w), int(landmarks[RIGHT_HIP].y * h))

        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
        avg_hip_y = (left_hip[1] + right_hip[1]) // 2

        # Draw reference line from shoulders to hips (special case)
        cv2.line(frame, (50, avg_shoulder_y), (w - 50, avg_shoulder_y), (0, 255, 255), 1)
        cv2.line(frame, (50, avg_hip_y), (w - 50, avg_hip_y), (0, 255, 255), 1)

        # Draw ideal alignment line
        ideal_y = (avg_shoulder_y + avg_hip_y) // 2
        cv2.line(frame, (50, ideal_y), (w - 50, ideal_y), (0, 255, 0), 2)


def draw_guidelines(frame, w, h):
    guidelines = [
        "Plank Form Guide:",
        "1. Shoulders directly over elbows",
        "2. Body in straight line",
        "3. Hips not sagging or raised",
        "4. Elbows at 90° angle"
    ]

    for i, line in enumerate(guidelines):
        cv2.putText(frame, line, (w - 350, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

plank_validator = PlankValidator()

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

        plank_status = "No pose detected"
        feedback_lines = []
        angles_info = {}

        if result and result.pose_landmarks:
            for pose_lms in result.pose_landmarks:
                # Draw pose landmarks and connections
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

                # Draw alignment line for plank
                draw_plank_line(frame, pose_lms, w, h)

                is_valid, feedback, angles = plank_validator.is_valid_plank(pose_lms)
                angles_info = angles

                plank_status = "VALID PLANK! ✓" if is_valid else "INVALID PLANK"
                feedback_lines = feedback

                draw_angle_info(frame, pose_lms, angles, w, h)

        draw_guidelines(frame, w, h)

        # Display plank valid or invalid status
        status_color = (0, 255, 0) if "VALID" in plank_status else (0, 0, 255)
        cv2.putText(frame, f"Status: {plank_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        y_offset = 60
        for line in feedback_lines:
            color = (255, 255, 255)  # white
            if "✓" in line:
                color = (0, 255, 0)  # green for good
            elif "not" in line.lower() or "sagging" in line.lower() or "too high" in line.lower():
                color = (0, 165, 255)  # orange for warning

            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        cv2.imshow("Plank Pose Validator", frame)
        ts += 33

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()