import cv2
import base64
import numpy as np
import threading
import time
import queue
from typing import Optional, Tuple


class CameraManager:


    def __init__(self, nao=None, use_local_camera=False, camera_index=0,
                 use_threading=True, window_name: str = None):
        """
            nao: NAO robot instance (if using NAO camera)
            use_local_camera: Whether to use local webcam
            camera_index: Which camera to use (0=default, 1=external, etc.)
            use_threading: Use background thread for continuous capture
            window_name: Optional window name for display
        """
        self.nao = nao
        self.use_local_camera = use_local_camera
        self.camera_index = camera_index
        self.local_camera = None
        self.window_name = window_name

        # For NAO camera callback system
        self.nao_frame_queue = queue.Queue(maxsize=3)

        # Threading support
        self.use_threading = use_threading
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.update_thread = None
        self.frame_count = 0
        self.start_time = time.time()

        # Window state
        self.window_created = False

        if use_local_camera:
            self._init_local_camera()
            if use_threading:
                self._start_capture_thread()
        elif nao:
            self._init_nao_camera()

    @staticmethod
    def list_cameras(max_cameras=5):
        print("\nScanning for available cameras...")
        available = []

        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  Camera {i}: {width}x{height}")
                    available.append(i)
                cap.release()
            else:
                # Try AVFoundation on macOS
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"  Camera {i}: {width}x{height} (AVFoundation)")
                        available.append(i)
                    cap.release()

        return available

    def _init_local_camera(self):
        try:
            self.local_camera = cv2.VideoCapture(self.camera_index)

            if not self.local_camera.isOpened():
                self.local_camera = cv2.VideoCapture(
                    self.camera_index, cv2.CAP_AVFOUNDATION
                )

            if self.local_camera.isOpened():
                # Warm up camera
                for _ in range(10):
                    ret, frame = self.local_camera.read()
                    if ret and frame is not None and frame.size > 0:
                        width = int(self.local_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.local_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"LOCAL camera {self.camera_index}: {width}x{height}")

                        with self.frame_lock:
                            self.current_frame = frame
                        return
                    time.sleep(0.1)

                raise RuntimeError(f"Could not read from camera {self.camera_index}")
            else:
                raise RuntimeError(f"Could not open camera {self.camera_index}")

        except Exception as e:
            print(f"✗ Local camera error: {e}")
            self.local_camera = None

    def _init_nao_camera(self):
        if not self.nao:
            return
        
        try:
            if hasattr(self.nao, 'top_camera'):
                self.nao.top_camera.register_callback(self._on_nao_image)
                print("NAO top_camera registered")

                start = time.time()
                while time.time() - start < 3.0:
                    if not self.nao_frame_queue.empty():
                        frame = self.nao_frame_queue.queue[0]
                        print(f"NAO camera ready: {frame.shape[1]}x{frame.shape[0]}")
                        return
                    time.sleep(0.1)

                print("NAO camera registered but no frames yet")

            elif hasattr(self.nao, 'bottom_camera'):
                self.nao.bottom_camera.register_callback(self._on_nao_image)
                print("NAO bottom_camera registered")
            else:
                print("NAO has no camera attribute")

        except Exception as e:
            print(f"NAO camera error: {e}")

    def _on_nao_image(self, image_message):
        """Callback for NAO camera images"""
        try:
            img = image_message.image

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #flip issues
            img_bgr = cv2.flip(img_bgr, 1)

            if self.nao_frame_queue.full():
                try:
                    self.nao_frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.nao_frame_queue.put(img_bgr)

            # Update current frame
            with self.frame_lock:
                self.current_frame = img_bgr
                self.frame_count += 1

        except Exception as e:
            print(f"NAO callback error: {e}")

    def _start_capture_thread(self):
        if not self.use_local_camera or not self.local_camera:
            return

        self.running = True
        self.update_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self.update_thread.start()
        print("Capture thread started")

    def _capture_loop(self):
        
        while self.running:
            try:
                if self.local_camera and self.local_camera.isOpened():
                    ret, frame = self.local_camera.read()

                    if ret and frame is not None:
                        self.frame_count += 1
                        with self.frame_lock:
                            self.current_frame = frame
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.1)

            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)

    def capture_frame(self) -> Optional[np.ndarray]:
        if self.use_local_camera:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
            return None
        else:

            try:
                frame = self.nao_frame_queue.get_nowait()
                with self.frame_lock:
                    self.current_frame = frame
                return frame.copy()
            except queue.Empty:
                with self.frame_lock:
                    if self.current_frame is not None:
                        return self.current_frame.copy()
                return None

    def create_window(self, name: str = None, x: int = 50, y: int = 100,
                      width: int = 640, height: int = 480):

        window_name = name or self.window_name or "Camera"
        self.window_name = window_name

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, x, y)
        cv2.resizeWindow(window_name, width, height)

        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (width // 3, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.imshow(window_name, placeholder)
        cv2.waitKey(1)

        self.window_created = True
        return window_name

    def show_frame(self, frame: np.ndarray, window_name: str = None):
        """Display frame in window"""
        name = window_name or self.window_name
        if name and frame is not None:
            cv2.imshow(name, frame)

    def show_placeholder(self, message: str = "No frame", window_name: str = None):
        name = window_name or self.window_name
        if name:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, message, (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow(name, placeholder)

    def capture_image_base64(self) -> Optional[str]:
        frame = self.capture_frame()
        if frame is None:
            return None

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Encoding error: {e}")
            return None

    def is_available(self) -> bool:
        if self.use_local_camera:
            with self.frame_lock:
                return self.current_frame is not None
        else:
            return (not self.nao_frame_queue.empty()) or (self.current_frame is not None)

    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def cleanup(self):
        if self.running:
            self.running = False
            if self.update_thread:
                self.update_thread.join(timeout=2)

        if self.local_camera:
            self.local_camera.release()

        # Clear NAO queue
        while not self.nao_frame_queue.empty():
            try:
                self.nao_frame_queue.get_nowait()
            except queue.Empty:
                break

        if self.window_created and self.window_name:
            cv2.destroyWindow(self.window_name)


class DualCameraManager:

    def __init__(self, nao=None, laptop_camera_index: int = 0):
        """
            nao: NAO robot instance (None for laptop-only mode)
            laptop_camera_index: Which laptop camera to use
        """
        self.nao = nao

        # Window names
        self.LAPTOP_WINDOW = "1. LAPTOP - Pose Detection"
        self.NAO_WINDOW = "2. NAO - Face Tracking"

        self._create_windows()


        self.laptop_camera = CameraManager(
            use_local_camera=True,
            camera_index=laptop_camera_index,
            use_threading=True,
            window_name=self.LAPTOP_WINDOW
        )

        self.nao_camera = None
        if nao:
            self.nao_camera = CameraManager(
                nao=nao,
                use_local_camera=False,
                window_name=self.NAO_WINDOW
            )

    def _create_windows(self):
        print("\n[Creating Windows]")

        # Left window - Laptop
        cv2.namedWindow(self.LAPTOP_WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.LAPTOP_WINDOW, 50, 100)
        cv2.resizeWindow(self.LAPTOP_WINDOW, 640, 480)
        print(f"  {self.LAPTOP_WINDOW}")

        # Right window - NAO
        cv2.namedWindow(self.NAO_WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.NAO_WINDOW, 720, 100)
        cv2.resizeWindow(self.NAO_WINDOW, 640, 480)
        print(f"  {self.NAO_WINDOW}")

        # Force windows to appear
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow(self.LAPTOP_WINDOW, placeholder)
        cv2.imshow(self.NAO_WINDOW, placeholder)
        cv2.waitKey(1)

    def get_laptop_frame(self) -> Optional[np.ndarray]:
        return self.laptop_camera.capture_frame()

    def get_nao_frame(self) -> Optional[np.ndarray]:
        if self.nao_camera:
            return self.nao_camera.capture_frame()
        return None

    def show_laptop_frame(self, frame: np.ndarray):
        if frame is not None:
            cv2.imshow(self.LAPTOP_WINDOW, frame)

    def show_nao_frame(self, frame: np.ndarray):
        if frame is not None:
            cv2.imshow(self.NAO_WINDOW, frame)

    def cleanup(self):
        if self.laptop_camera:
            self.laptop_camera.cleanup()
        if self.nao_camera:
            self.nao_camera.cleanup()
        cv2.destroyAllWindows()
