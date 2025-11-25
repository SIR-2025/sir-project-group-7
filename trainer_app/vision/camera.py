from ultralytics import YOLO
import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO("yolov8n-pose.pt")  # Lightweight + stable

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def detect_pose(self, frame=None):
        if frame is None:
            frame = self.get_frame()

        # YOLO pose inference
        results = self.model(frame, verbose=False)[0]

        if results.keypoints is None:
            return None

        # Return Nx(17) keypoints structure with xy coords
        return results.keypoints.xy.numpy()
