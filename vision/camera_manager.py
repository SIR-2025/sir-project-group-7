# vision/camera_manager.py
import cv2
import base64
from typing import Optional
import numpy as np
import threading
import time
import queue


class CameraManager:
    """
    Manages camera capture from local webcam or NAO robot camera with threading support
    """

    def __init__(self, nao=None, use_local_camera=False, camera_index=0, use_threading=False):
        """
        Initialize camera manager.

        Args:
            nao: NAO robot instance (if using NAO camera)
            use_local_camera: Whether to use local webcam
            camera_index: Which camera to use (0=default, 1=external, etc.)
            use_threading: Use background thread for continuous capture
        """
        self.nao = nao
        self.use_local_camera = use_local_camera
        self.camera_index = camera_index
        self.local_camera = None

        # For NAO camera callback system
        self.nao_frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames

        # Threading support
        self.use_threading = use_threading
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.update_thread = None
        self.frame_count = 0
        self.error_count = 0

        # Initialize camera
        if use_local_camera:
            self._init_local_camera()
            if use_threading:
                self._start_capture_thread()
        elif nao:
            self._init_nao_camera()
            # NAO camera is already "threaded" via callback - no need for separate thread
        else:
            print("No camera configured")
            return

    @staticmethod
    def list_cameras(max_cameras=5):
        """List available cameras."""
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

    def _on_nao_image(self, image_message):
        """
        Callback for NAO camera images (SIC framework style)

        Args:
            image_message: CompressedImageMessage from NAO
        """
        try:
            # Get image from message
            img = image_message.image

            # Convert RGB to BGR for OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = img[..., ::-1]  # RGB to BGR
            else:
                img_bgr = img

            # Update frame queue (remove old frames if full)
            if self.nao_frame_queue.full():
                try:
                    self.nao_frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.nao_frame_queue.put(img_bgr)

            # Update current frame for is_available() check
            with self.frame_lock:
                self.current_frame = img_bgr
                self.frame_count += 1

        except Exception as e:
            print(f"NAO camera callback error: {e}")

    def _init_local_camera(self):
        """Initialize local camera"""
        try:
            self.local_camera = cv2.VideoCapture(self.camera_index)

            if not self.local_camera.isOpened():
                try:
                    self.local_camera = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                except:
                    pass

            if self.local_camera.isOpened():
                for attempt in range(5):
                    ret, frame = self.local_camera.read()
                    if ret and frame is not None and frame.size > 0:
                        width = int(self.local_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.local_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"✓ LOCAL camera {self.camera_index} initialized - Resolution: {width}x{height}")
                        if self.use_threading:
                            with self.frame_lock:
                                self.current_frame = frame
                        return
                    time.sleep(0.1)

                raise RuntimeError(f"Could not read frames from camera {self.camera_index}")
            else:
                raise RuntimeError(f"Could not open camera {self.camera_index}")

        except Exception as e:
            print(f"Error initializing local camera: {e}")
            self.local_camera = None

    def _init_nao_camera(self):
        """Initialize NAO camera with callback registration"""
        if not self.nao:
            print("NAO not configured")
            return

        try:
            # Register callback for top camera (better for face tracking)
            if hasattr(self.nao, 'top_camera'):
                self.nao.top_camera.register_callback(self._on_nao_image)
                print("✓ NAO top_camera initialized with callback")

                # Wait a bit for first frame
                time.sleep(0.5)

                # Check if we got a frame
                if not self.nao_frame_queue.empty():
                    test_frame = self.nao_frame_queue.queue[0]
                    print(f"✓ NAO camera ready - Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                else:
                    print("⚠ NAO camera registered but no frame received yet")

            elif hasattr(self.nao, 'bottom_camera'):
                self.nao.bottom_camera.register_callback(self._on_nao_image)
                print("✓ NAO bottom_camera initialized with callback")
            else:
                available_attrs = [attr for attr in dir(self.nao) if 'camera' in attr.lower()]
                raise RuntimeError(f"NAO has no camera attribute. Available: {available_attrs}")

        except Exception as e:
            print(f"Error initializing NAO camera: {e}")
            import traceback
            traceback.print_exc()
            self.nao = None

    def _start_capture_thread(self):
        """Start background thread for continuous frame capture (local camera only)"""
        if not self.use_local_camera or not self.local_camera:
            return

        print(f"Starting camera capture thread (local)...")
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        # Wait for first frame
        max_wait = 5
        for i in range(max_wait * 10):
            with self.frame_lock:
                if self.current_frame is not None:
                    print(f"✓ Camera thread started and first frame captured!")
                    return
            time.sleep(0.1)

        print("Warning: Camera thread started but no frame captured yet")

    def _update_loop(self):
        """Continuously update current frame from LOCAL camera"""
        self.frame_count = 0
        self.error_count = 0
        last_log_time = time.time()

        while self.running:
            try:
                frame = self._capture_from_local_direct()

                if frame is not None and frame.size > 0:
                    self.frame_count += 1
                    self.error_count = 0

                    with self.frame_lock:
                        self.current_frame = frame.copy()

                    # Log every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        fps = self.frame_count / (current_time - last_log_time + 5.0)
                        print(f"LOCAL Camera thread: {self.frame_count} frames captured (~{fps:.1f} FPS)")
                        last_log_time = current_time
                else:
                    self.error_count += 1
                    if self.error_count > 10:
                        print(f"Warning: LOCAL camera - Multiple frame read failures ({self.error_count})")
                        self.error_count = 0
                    time.sleep(0.1)

            except Exception as e:
                print(f"LOCAL Frame update error: {e}")
                time.sleep(0.5)

    def _capture_from_local_direct(self):
        """Direct capture from local camera"""
        if not self.local_camera or not self.local_camera.isOpened():
            return None

        try:
            ret, frame = self.local_camera.read()
            if ret:
                return frame
            return None
        except Exception as e:
            print(f"Local camera capture error: {e}")
            return None

    def capture_frame(self):
        """
        Capture a single frame from the camera

        Returns:
            numpy array: The captured frame, or None if no frame available
        """
        if self.use_local_camera:
            # Local camera uses threading
            if self.use_threading:
                with self.frame_lock:
                    if self.current_frame is not None:
                        return self.current_frame.copy()
                return None
            else:
                return self._capture_from_local_direct()
        else:
            # NAO camera uses callback queue
            try:
                # Get latest frame from queue (non-blocking)
                frame = self.nao_frame_queue.get_nowait()

                # Put it back for next call (keep latest)
                if not self.nao_frame_queue.full():
                    self.nao_frame_queue.put(frame)

                return frame.copy()
            except queue.Empty:
                # No frame available yet
                return None

    def capture_image_base64(self):
        """Capture frame and return as base64 string"""
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
        """Capture frame and save to file"""
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
        """Test camera by showing live feed"""
        print(f"Testing camera for {duration} seconds...")
        print("Press ESC to stop early")

        start_time = time.time()
        frame_count = 0

        try:
            while (time.time() - start_time) < duration:
                frame = self.capture_frame()

                if frame is not None:
                    frame_count += 1

                    cv2.putText(frame, f"Frame: {frame_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    camera_type = "Local (Threading)" if self.use_local_camera else "NAO (Callback)"
                    cv2.putText(frame, f"Mode: {camera_type}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow("Camera Test", frame)

                    if cv2.waitKey(30) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.01)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.destroyAllWindows()
            print(f"Camera test complete: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\nCamera test interrupted")

    def is_available(self):
        """Check if camera is available and producing frames"""
        if self.use_local_camera:
            if self.use_threading:
                with self.frame_lock:
                    return self.current_frame is not None
            else:
                return self.local_camera is not None and self.local_camera.isOpened()
        else:
            # NAO camera - check if we have frames in queue
            return not self.nao_frame_queue.empty()

    def get_fps(self):
        """Get approximate capture FPS"""
        if self.use_threading:
            return self.frame_count / (time.time() + 1)
        return 0

    def cleanup(self):
        """Cleanup resources"""
        if self.use_threading and self.running:
            print("Stopping camera thread...")
            self.running = False
            if self.update_thread:
                self.update_thread.join(timeout=3)
            print("✓ Camera thread stopped")

        if self.local_camera:
            self.local_camera.release()
            print("✓ Local camera released")

        # Clear NAO frame queue
        if not self.use_local_camera:
            while not self.nao_frame_queue.empty():
                try:
                    self.nao_frame_queue.get_nowait()
                except queue.Empty:
                    break

        cv2.destroyAllWindows()