import cv2
import base64
from typing import Optional
import numpy as np
import threading
import time


class CameraManager:
    """
    Manages camera capture from local webcam, iPhone, or NAO robot camera
    """

    def __init__(self, nao=None, use_local_camera=False, camera_index=0, use_threading=False):
        """
        Initialize camera manager.
        
        Args:
            nao: NAO robot instance (if using NAO camera)
            use_local_camera: Whether to use local webcam/iPhone
            camera_index: Which camera to use (0=default, 1=iPhone, etc.)
            use_threading: Use background thread for continuous capture (better for iPhone)
        """
        self.nao = nao
        self.use_local_camera = use_local_camera
        self.camera_index = camera_index
        self.local_camera = None
        
        # Threading support for iPhone
        self.use_threading = use_threading
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.update_thread = None

        if use_local_camera:
            self._init_local_camera()
            if use_threading:
                self._start_capture_thread()
        elif nao:
            print("Using NAO camera")
        else:
            print("No camera configured")
    
    @staticmethod
    def list_cameras(max_cameras=5):
        """
        List available cameras.
        """
        print("\nScanning for available cameras...")
        available = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print("  Camera {}: {}x{}".format(i, width, height))
                    available.append(i)
                cap.release()
            else:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("  Camera {}: {}x{} (AVFoundation)".format(i, width, height))
                        available.append(i)
                    cap.release()
        
        return available

    def _init_local_camera(self):
        try:
            # Try with specified camera index
            self.local_camera = cv2.VideoCapture(self.camera_index)

            # If that doesn't work, try AVFoundation (Mac)
            if not self.local_camera.isOpened():
                try:
                    self.local_camera = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                except:
                    pass

            if self.local_camera.isOpened():
                # Test reading a frame
                for attempt in range(5):
                    ret, frame = self.local_camera.read()
                    if ret and frame is not None and frame.size > 0:
                        width = int(self.local_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.local_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("Using LOCAL camera {} - Resolution: {}x{}".format(
                            self.camera_index, width, height))
                        if self.use_threading:
                            with self.frame_lock:
                                self.current_frame = frame
                        return
                    time.sleep(0.5)
                
                raise RuntimeError("Could not read frames from camera {}".format(self.camera_index))
            else:
                raise RuntimeError("Could not open camera {}".format(self.camera_index))

        except Exception as e:
            print("Error initializing local camera: {}".format(e))
            self.local_camera = None
    
    def _start_capture_thread(self):
        """Start background thread for continuous frame capture"""
        if not self.local_camera or not self.local_camera.isOpened():
            return
        
        print("Starting camera capture thread...")
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        time.sleep(0.5)
        print("Camera thread started!")
    
    def _update_loop(self):
        """Continuously update current frame from camera"""
        frame_count = 0
        error_count = 0

        while self.running:
            try:
                ret, frame = self.local_camera.read()

                if ret and frame is not None and frame.size > 0:
                    frame_count += 1
                    error_count = 0

                    with self.frame_lock:
                        self.current_frame = frame

                    # Log every 100 frames
                    if frame_count % 100 == 0:
                        print("Camera: {} frames received".format(frame_count))
                else:
                    error_count += 1
                    if error_count > 10:
                        print("Warning: Multiple frame read failures")
                        error_count = 0
                    time.sleep(0.1)

            except Exception as e:
                print("Frame update error: {}".format(e))
                time.sleep(0.5)

    def capture_frame(self):
        """
        Capture a single frame from the camera
        """
        if self.use_local_camera:
            if self.use_threading:
                # Get from thread buffer
                with self.frame_lock:
                    if self.current_frame is not None:
                        return self.current_frame.copy()
                return None
            else:
                # Direct capture
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
            print("Local camera capture error: {}".format(e))
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
            print("NAO camera capture error: {}".format(e))
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
            print("Error encoding image: {}".format(e))
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
            print("Frame saved to {}".format(filepath))
            return True
        except Exception as e:
            print("Error saving frame: {}".format(e))
            return False

    def test_camera(self, duration=5):
        """
        Test camera by showing live feed
        """
        import time

        print("Testing camera for {} seconds...".format(duration))
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
            if self.use_threading:
                return self.current_frame is not None
            else:
                return self.local_camera is not None and self.local_camera.isOpened()
        elif self.nao:
            return True
        return False

    def cleanup(self):
        if self.use_threading and self.running:
            print("Stopping camera thread...")
            self.running = False
            if self.update_thread:
                self.update_thread.join(timeout=3)
        
        if self.local_camera:
            self.local_camera.release()
        cv2.destroyAllWindows()
        print("Camera resources released")