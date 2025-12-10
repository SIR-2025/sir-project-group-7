import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiSetAnglesRequest

    HAS_SIC = True
except ImportError:
    HAS_SIC = False


@dataclass
class FaceDetection:
    detected: bool
    center: Optional[Tuple[int, int]]
    rect: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    confidence: float


class SmoothValue:
    

    def __init__(self, alpha: float = 0.2, dead_zone: float = 0.0):
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
            return self.value

        if abs(new_value - self.value) < self.dead_zone:
            return self.value

        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = None):
        self.value = value


class FaceTracker:
    """
    Face detection and tracking using NAO's camera with smooth head movement.
    """

    def __init__(self, camera_manager=None,
                 smoothing_alpha: float = 0.20,
                 dead_zone_pixels: int = 50,
                 max_yaw: float = 0.15,
                 max_pitch: float = 0.10,
                 movement_speed: float = 0.15):
        """
            camera_manager: CameraManager instance (optional)
            smoothing_alpha: Head movement smoothing (0.1=smooth, 0.5=responsive)
            dead_zone_pixels: Ignore face movement smaller than this (pixels)
            max_yaw: Maximum yaw adjustment per update (radians) - smaller = smoother
            max_pitch: Maximum pitch adjustment per update (radians)
            movement_speed: NAO head movement speed (0.0-1.0)
        """
        self.camera_manager = camera_manager

        # try DNN first, fallback to Haar
        self.use_dnn = False
        self.face_net = None
        self.face_cascade = None

        try:
            from pathlib import Path

            possible_paths = [
                Path(__file__).parent.parent / "DDN_model_face_detection",
                Path(__file__).parent.parent / "dnn_model",
                Path(__file__).parent.parent,
                Path.cwd() / "DDN_model_face_detection",
                Path.cwd(),
            ]

            prototxt = None
            caffemodel = None

            for base_path in possible_paths:
                p = base_path / "deploy.prototxt"
                c = base_path / "res10_300x300_ssd_iter_140000.caffemodel"

                if p.exists() and c.exists():
                    prototxt = str(p)
                    caffemodel = str(c)
                    print(f"Found DNN model in: {base_path}")
                    break

            if prototxt and caffemodel:
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.use_dnn = True
                print("Using DNN face detector (high accuracy)")
            else:
                print("DNN model files not found, using Haar cascade")
                print("  Download DNN model:")
                print("    mkdir -p DDN_model_face_detection && cd DDN_model_face_detection")
                print("    wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
                print("    wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")

        except Exception as e:
            print(f"DNN loading failed: {e}")

        # Fallback to Haar cascade
        if not self.use_dnn:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("Using Haar cascade face detector")

        # Tracking parameters
        self.dead_zone_pixels = dead_zone_pixels
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.movement_speed = movement_speed

        # Smooth head angles
        self.smooth_yaw = SmoothValue(alpha=smoothing_alpha, dead_zone=0.02)
        self.smooth_pitch = SmoothValue(alpha=smoothing_alpha, dead_zone=0.02)

        # Current head position (radians)
        self.current_head_yaw = 0.0
        self.current_head_pitch = 0.0

        # Face tracking state
        self.last_face_center = None
        self.face_lost_frames = 0
        self.max_lost_frames = 15


        if self.use_dnn:
            self.smooth_face_x = SmoothValue(alpha=0.6, dead_zone=2)
            self.smooth_face_y = SmoothValue(alpha=0.6, dead_zone=2)
        else:
            self.smooth_face_x = SmoothValue(alpha=0.4, dead_zone=5)
            self.smooth_face_y = SmoothValue(alpha=0.4, dead_zone=5)

        # Control
        self.tracking_active = False

    def detect_face(self, frame: np.ndarray) -> FaceDetection:
        if frame is None:
            return FaceDetection(False, None, None, 0.0)

        if self.use_dnn and self.face_net is not None:
            return self._detect_face_dnn(frame)
        else:
            return self._detect_face_haar(frame)

    def _detect_face_dnn(self, frame: np.ndarray) -> FaceDetection:
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_face = None
        best_confidence = 0.3

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > best_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    best_face = (x1, y1, x2 - x1, y2 - y1)
                    best_confidence = confidence

        if best_face is not None:
            x, y, fw, fh = best_face
            center = (x + fw // 2, y + fh // 2)

            self.last_face_center = center
            self.face_lost_frames = 0

            smoothed_x = int(self.smooth_face_x.update(center[0]))
            smoothed_y = int(self.smooth_face_y.update(center[1]))
            smoothed_center = (smoothed_x, smoothed_y)

            return FaceDetection(True, smoothed_center, best_face, float(best_confidence))
        else:
            return self._handle_lost_face()

    def _detect_face_haar(self, frame: np.ndarray) -> FaceDetection:
        """Detect face using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=4,
            minSize=(40, 40),
            maxSize=(600, 600),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            largest = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = largest
            center = (x + w // 2, y + h // 2)

            frame_area = frame.shape[0] * frame.shape[1]
            face_area = w * h
            size_score = min(1.0, face_area / (frame_area * 0.08))

            aspect_ratio = w / h
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.5
            aspect_score = max(0.0, min(1.0, aspect_score))

            confidence = size_score * aspect_score

            self.last_face_center = center
            self.face_lost_frames = 0

            smoothed_x = int(self.smooth_face_x.update(center[0]))
            smoothed_y = int(self.smooth_face_y.update(center[1]))
            smoothed_center = (smoothed_x, smoothed_y)

            return FaceDetection(True, smoothed_center, (x, y, w, h), confidence)
        else:
            return self._handle_lost_face()

    def _handle_lost_face(self) -> FaceDetection:
        self.face_lost_frames += 1

        if self.face_lost_frames < self.max_lost_frames and self.last_face_center is not None:
            return FaceDetection(False, self.last_face_center, None, 0.0)
        else:
            self.last_face_center = None
            return FaceDetection(False, None, None, 0.0)

    def annotate_frame(self, frame: np.ndarray,
                       detection: FaceDetection = None) -> np.ndarray:
        if frame is None:
            return None

        if detection is None:
            detection = self.detect_face(frame)

        annotated = frame.copy()
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Crosshair in center
        cv2.line(annotated, (center_x - 30, center_y),
                 (center_x + 30, center_y), (100, 100, 100), 1)
        cv2.line(annotated, (center_x, center_y - 30),
                 (center_x, center_y + 30), (100, 100, 100), 1)

        # Dead zone circle
        cv2.circle(annotated, (center_x, center_y),
                   self.dead_zone_pixels, (50, 50, 50), 1)

        # Detector type indicator
        detector_type = "DNN" if self.use_dnn else "HAAR"
        cv2.putText(annotated, f"[{detector_type}]",
                    (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if detection.detected and detection.rect:
            x, y, fw, fh = detection.rect
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            if detection.center:
                cv2.circle(annotated, detection.center, 5, (0, 255, 0), -1)
                cv2.line(annotated, (center_x, center_y),
                         detection.center, (0, 255, 255), 1)

                offset_x = detection.center[0] - center_x
                offset_y = detection.center[1] - center_y
                cv2.putText(annotated, f"Offset: ({offset_x:+d}, {offset_y:+d})",
                            (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(annotated, f"Face: DETECTED ({detection.confidence:.0%})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif detection.center:
            cv2.circle(annotated, detection.center, 8, (0, 165, 255), 2)
            cv2.putText(annotated, "Face: TRACKING (lost)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(annotated, "Face: NOT DETECTED",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(annotated, f"Yaw: {self.current_head_yaw:+.2f} rad",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, f"Pitch: {self.current_head_pitch:+.2f} rad",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        status = "ACTIVE" if self.tracking_active else "PAUSED"
        color = (0, 255, 0) if self.tracking_active else (128, 128, 128)
        cv2.putText(annotated, f"Tracking: {status}",
                    (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated

    def get_head_movement_angles(self, face_center: Tuple[int, int],
                                 frame_shape: Tuple[int, ...]) -> Tuple[float, float]:
        """Calculate smooth head angles to center face."""
        if face_center is None:
            return self.current_head_yaw, self.current_head_pitch

        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2

        offset_x = face_center[0] - center_x
        offset_y = face_center[1] - center_y

        distance = np.sqrt(offset_x ** 2 + offset_y ** 2)

        if distance < self.dead_zone_pixels:
            return self.current_head_yaw, self.current_head_pitch

        norm_x = offset_x / (w / 2)
        norm_y = offset_y / (h / 2)

        norm_x = np.sign(norm_x) * (abs(norm_x) ** 1.5)
        norm_y = np.sign(norm_y) * (abs(norm_y) ** 1.5)

        yaw_adj = norm_x * self.max_yaw
        pitch_adj = norm_y * self.max_pitch

        target_yaw = self.current_head_yaw + yaw_adj
        target_pitch = self.current_head_pitch + pitch_adj

        target_yaw = np.clip(target_yaw, -1.5, 1.5)
        target_pitch = np.clip(target_pitch, -0.5, 0.4)

        smooth_yaw = self.smooth_yaw.update(target_yaw)
        smooth_pitch = self.smooth_pitch.update(target_pitch)

        return smooth_yaw, smooth_pitch

    def move_nao_head_to_face(self, nao, face_center: Tuple[int, int],
                              frame_shape: Tuple[int, ...]) -> bool:
        """Move NAO's head to track face with smooth motion."""
        if not self.tracking_active or nao is None or face_center is None:
            return False

        target_yaw, target_pitch = self.get_head_movement_angles(face_center, frame_shape)

        yaw_diff = abs(target_yaw - self.current_head_yaw)
        pitch_diff = abs(target_pitch - self.current_head_pitch)

        if yaw_diff < 0.02 and pitch_diff < 0.02:
            return False

        try:
            from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiSetAnglesRequest

            yaw = float(target_yaw)
            pitch = float(target_pitch)
            spd = float(self.movement_speed)

            nao.motion.request(
                NaoqiSetAnglesRequest(
                    names=["HeadYaw", "HeadPitch"],
                    angles=[yaw, pitch],
                    speed=spd
                ),
                block=False
            )

            self.current_head_yaw = target_yaw
            self.current_head_pitch = target_pitch
            return True

        except Exception as e:
            print(f"Head movement error: {e}")
            return False

    def reset_head_position(self, nao) -> bool:
        """Reset NAO's head to center position."""
        self.current_head_yaw = 0.0
        self.current_head_pitch = 0.0
        self.smooth_yaw.reset(0.0)
        self.smooth_pitch.reset(0.0)

        if nao is None:
            return True

        try:
            from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiSetAnglesRequest

            nao.motion.request(
                NaoqiSetAnglesRequest(
                    names=["HeadYaw", "HeadPitch"],
                    angles=[0.0, 0.0],
                    speed=0.2
                ),
                block=True
            )

            print("Head reset to center")
            return True

        except Exception as e:
            print(f"Head reset error: {e}")
            return False

    def start_tracking(self):
        """Enable head tracking"""
        self.tracking_active = True
        print("Face tracking STARTED")

    def stop_tracking(self):
        """Disable head tracking"""
        self.tracking_active = False
        print("Face tracking STOPPED")

    def is_face_centered(self, face_center: Tuple[int, int],
                         frame_shape: Tuple[int, ...],
                         tolerance: int = None) -> bool:
        if face_center is None:
            return False

        tol = tolerance or self.dead_zone_pixels
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2

        distance = np.sqrt(
            (face_center[0] - center_x) ** 2 +
            (face_center[1] - center_y) ** 2
        )

        return distance < tol

    def cleanup(self):
        self.stop_tracking()