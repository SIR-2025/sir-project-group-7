
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

MODEL_PATH = "../pose_landmarkers/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Pose connections
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
LEFT_HEEL = 29
RIGHT_HEEL = 30


class LungeValidator:
    def __init__(self):
        # Thresholds
        self.front_knee_angle_min = 85
        self.front_knee_angle_max = 100
        self.back_knee_angle_min = 80
        self.back_knee_angle_max = 120
        self.hip_shoulder_alignment_threshold = 0.1
        self.torso_vertical_threshold = 15

    def calculate_angle(self, point1, point2, point3):
        # Vectors
        vector1 = (point1.x - point2.x, point1.y - point2.y)
        vector2 = (point3.x - point2.x, point3.y - point2.y)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        mag1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        mag2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if mag1 * mag2 == 0:
            return 180
        angle = math.acos(dot_product / (mag1 * mag2))
        return math.degrees(angle)

    def determine_lunge_side(self, landmarks):
        # Determine forward leg
        left_knee_forward = landmarks[LEFT_KNEE].x > landmarks[RIGHT_KNEE].x
        return "left" if left_knee_forward else "right"

    def calculate_torso_angle(self, landmarks):
        # Torso verticality
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
        hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2

        vertical_diff = shoulder_y - hip_y
        horizontal_diff = shoulder_x - hip_x

        if vertical_diff == 0:
            return 90

        angle = math.degrees(math.atan(abs(horizontal_diff) / abs(vertical_diff)))
        return angle

    def check_hip_shoulder_alignment(self, landmarks, lunge_side):
        # Hip-shoulder squared check
        left_hip_shoulder_diff = abs(landmarks[LEFT_HIP].x - landmarks[LEFT_SHOULDER].x)
        right_hip_shoulder_diff = abs(landmarks[RIGHT_HIP].x - landmarks[RIGHT_SHOULDER].x)
        avg_diff = (left_hip_shoulder_diff + right_hip_shoulder_diff) / 2
        return avg_diff

    def check_knee_over_ankle(self, landmarks, knee_idx, ankle_idx):
        # Knee alignment
        knee_x = landmarks[knee_idx].x
        ankle_x = landmarks[ankle_idx].x
        return abs(knee_x - ankle_x) < 0.05

    def is_valid_lunge(self, landmarks):
        if len(landmarks) < 31:
            return False, ["Not enough landmarks"], {}

        lunge_side = self.determine_lunge_side(landmarks)

        # Landmark assignment
        if lunge_side == "left":
            front_hip, front_knee, front_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            back_hip, back_knee, back_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        else:
            front_hip, front_knee, front_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            back_hip, back_knee, back_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

        # Angles
        front_knee_angle = self.calculate_angle(landmarks[front_hip], landmarks[front_knee], landmarks[front_ankle])
        back_knee_angle = self.calculate_angle(landmarks[back_hip], landmarks[back_knee], landmarks[back_ankle])
        torso_angle = self.calculate_torso_angle(landmarks)
        alignment_diff = self.check_hip_shoulder_alignment(landmarks, lunge_side)
        front_knee_over_ankle = self.check_knee_over_ankle(landmarks, front_knee, front_ankle)

        feedback = []
        is_valid = True

        # Validation checks
        if front_knee_angle < self.front_knee_angle_min:
            feedback.append(f"Front knee too bent ({front_knee_angle:.1f}°)")
            is_valid = False
        elif front_knee_angle > self.front_knee_angle_max:
            feedback.append(f"Front knee not bent enough ({front_knee_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Front knee: {front_knee_angle:.1f}° ✓")

        if back_knee_angle < self.back_knee_angle_min:
            feedback.append(f"Back knee too straight ({back_knee_angle:.1f}°)")
            is_valid = False
        elif back_knee_angle > self.back_knee_angle_max:
            feedback.append(f"Back knee too low ({back_knee_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Back knee: {back_knee_angle:.1f}° ✓")

        if torso_angle > self.torso_vertical_threshold:
            feedback.append(f"Torso leaning ({torso_angle:.1f}°)")
            is_valid = False
        else:
            feedback.append(f"Torso upright: {torso_angle:.1f}° ✓")

        if alignment_diff > self.hip_shoulder_alignment_threshold:
            feedback.append("Hips not squared")
            is_valid = False
        else:
            feedback.append("Hips squared ✓")

        if not front_knee_over_ankle:
            feedback.append("Front knee past toes")
            is_valid = False
        else:
            feedback.append("Knee over ankle ✓")

        return is_valid, feedback, {
            'lunge_side': lunge_side,
            'front_knee_angle': front_knee_angle,
            'back_knee_angle': back_knee_angle,
            'torso_angle': torso_angle,
            'alignment_diff': alignment_diff,
            'front_knee_over_ankle': front_knee_over_ankle
        }


def draw_angle_info(frame, landmarks, angles, w, h):
    # Angle display
    lunge_side = angles['lunge_side']

    if lunge_side == "left":
        front_knee_pos = (int(landmarks[LEFT_KNEE].x * w), int(landmarks[LEFT_KNEE].y * h))
        back_knee_pos = (int(landmarks[RIGHT_KNEE].x * w), int(landmarks[RIGHT_KNEE].y * h))
    else:
        front_knee_pos = (int(landmarks[RIGHT_KNEE].x * w), int(landmarks[RIGHT_KNEE].y * h))
        back_knee_pos = (int(landmarks[LEFT_KNEE].x * w), int(landmarks[LEFT_KNEE].y * h))

    cv2.putText(frame, f"Front: {angles['front_knee_angle']:.1f}°",
                (front_knee_pos[0] - 40, front_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.putText(frame, f"Back: {angles['back_knee_angle']:.1f}°",
                (back_knee_pos[0] - 40, back_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    shoulder_pos = (int(landmarks[LEFT_SHOULDER].x * w), int(landmarks[LEFT_SHOULDER].y * h))
    cv2.putText(frame, f"Torso: {angles['torso_angle']:.1f}°",
                (shoulder_pos[0] - 30, shoulder_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_lunge_guidelines(frame, landmarks, angles, w, h):
    # Alignment guides
    if len(landmarks) < 31:
        return

    lunge_side = angles['lunge_side']
    front_ankle = (int(landmarks[LEFT_ANKLE if lunge_side == "left" else RIGHT_ANKLE].x * w), 
                   int(landmarks[LEFT_ANKLE if lunge_side == "left" else RIGHT_ANKLE].y * h))
    front_knee = (int(landmarks[LEFT_KNEE if lunge_side == "left" else RIGHT_KNEE].x * w), 
                  int(landmarks[LEFT_KNEE if lunge_side == "left" else RIGHT_KNEE].y * h))

    # Vertical reference
    cv2.line(frame, (front_ankle[0], front_ankle[1]), (front_ankle[0], front_ankle[1] - 100), (0, 255, 0), 2)
    cv2.line(frame, front_knee, front_ankle, (255, 255, 0), 2)

    # Shoulder/hip lines
    left_shoulder = (int(landmarks[LEFT_SHOULDER].x * w), int(landmarks[LEFT_SHOULDER].y * h))
    right_shoulder = (int(landmarks[RIGHT_SHOULDER].x * w), int(landmarks[RIGHT_SHOULDER].y * h))
    left_hip = (int(landmarks[LEFT_HIP].x * w), int(landmarks[LEFT_HIP].y * h))
    right_hip = (int(landmarks[RIGHT_HIP].x * w), int(landmarks[RIGHT_HIP].y * h))

    cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 255), 2)
    cv2.line(frame, left_hip, right_hip, (0, 255, 255), 2)


def draw_guidelines(frame, w, h):
    # Instructions
    guidelines = [
        "Lunge Form Guide:",
        "1. Front knee at 90°, aligned over ankle",
        "2. Back knee bent, not touching ground",
        "3. Torso upright, shoulders back",
        "4. Hips squared forward",
        "5. Front knee: 85-100°, Back knee: 80-120°"
    ]

    for i, line in enumerate(guidelines):
        cv2.putText(frame, line, (w - 400, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# Main loop
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
lunge_validator = LungeValidator()

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

        lunge_status = "No pose detected"
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
                is_valid, feedback, angles = lunge_validator.is_valid_lunge(pose_lms)
                angles_info = angles

                lunge_status = f"VALID LUNGE ({angles['lunge_side'].upper()} LEG)! ✓" if is_valid else "INVALID LUNGE"
                feedback_lines = feedback

                draw_angle_info(frame, pose_lms, angles, w, h)
                draw_lunge_guidelines(frame, pose_lms, angles, w, h)

        draw_guidelines(frame, w, h)

        # Status display
        status_color = (0, 255, 0) if "VALID" in lunge_status else (0, 0, 255)
        cv2.putText(frame, f"Status: {lunge_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        y_offset = 60
        for line in feedback_lines:
            color = (255, 255, 255)
            if "✓" in line:
                color = (0, 255, 0)
            elif any(word in line.lower() for word in ["too", "not", "past", "leaning"]):
                color = (0, 165, 255)

            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        cv2.imshow("Lunge Pose Validator", frame)
        ts += 33

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()