import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarkers/pose_landmarker_full.task"

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

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

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
        
        if result and result.pose_landmarks:
            for pose_lms in result.pose_landmarks:
                
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
        
        cv2.imshow("Pose Landmarker", frame)
        ts += 33
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()