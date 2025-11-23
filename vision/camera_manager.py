import cv2
import base64
from typing import Optional
import numpy as np


class CameraManager:
    """
    Manages camera capture from local webcam or NAO robot camera
    """

    def __init__(self, nao=None, use_local_camera=False):
        self.nao = nao
        self.use_local_camera = use_local_camera
        self.local_camera = None

        if use_local_camera:
            self._init_local_camera()
        elif nao:
            print("Using NAO camera")
        else:
            print("No camera configured")

    def _init_local_camera(self):
        try:
            self.local_camera = cv2.VideoCapture(0)

            if not self.local_camera.isOpened():
                try:
                    self.local_camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                except:
                    pass

            if self.local_camera.isOpened():
                print("Using LOCAL camera")
            else:
                raise RuntimeError("Could not open local camera")

        except Exception as e:
            print(f"Error initializing local camera: {e}")
            self.local_camera = None

    def capture_frame(self):
        """
        Capture a single frame from the camera
        """
        if self.use_local_camera:
            return self._capture_from_local()
        elif self.nao:
            return self._capture_from_nao()
        else:
            print("No camera configured")
            return None

    def _capture_from_local(self):
        """
        Capture frame from local webcam
        """
        if not self.local_camera or not self.local_camera.isOpened():
            print("Local camera not available")
            return None

        try:
            ret, frame = self.local_camera.read()
            if ret:
                return frame
            else:
                print("Failed to capture frame from local camera")
                return None
        except Exception as e:
            print(f"Local camera capture error: {e}")
            return None

    def _capture_from_nao(self):
        """
        Capture frame from NAO camera
        """
        if not self.nao:
            print("NAO not configured")
            return None

        try:
            frame = self.nao.camera.get_image()
            return frame
        except Exception as e:
            print(f"NAO camera capture error: {e}")
            return None

    def capture_image_base64(self):
        """
        Capture frame and return as base64 string
        """
        frame = self.capture_frame()

        if frame is None:
            return None

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def save_frame(self, filepath):
        """
        Capture frame and save to file
        """
        frame = self.capture_frame()

        if frame is None:
            return False

        try:
            cv2.imwrite(str(filepath), frame)
            print(f"Frame saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False

    def test_camera(self, duration=5):
        """
        Test camera by showing live feed
        """
        import time

        print(f"Testing camera for {duration} seconds...")
        print("Press ESC to stop early")

        start_time = time.time()

        try:
            while (time.time() - start_time) < duration:
                frame = self.capture_frame()

                if frame is not None:
                    cv2.imshow("Camera Test", frame)

                    if cv2.waitKey(30) & 0xFF == 27:
                        break
                else:
                    print("No frame captured")
                    time.sleep(0.1)

            cv2.destroyAllWindows()
            print("Camera test complete")

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\nCamera test interrupted")

    def is_available(self):
        """
        Check if camera is available
        """
        if self.use_local_camera:
            return self.local_camera is not None and self.local_camera.isOpened()
        elif self.nao:
            return True
        return False

    def cleanup(self):
        if self.local_camera:
            self.local_camera.release()
        cv2.destroyAllWindows()
        print("Camera resources released")