import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vision.camera_manager import CameraManager
from vision.face_tracker import FaceTracker


def test_laptop_only():

    window_name = "Laptop Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.waitKey(1)

    # Initialize camera
    camera = CameraManager(
        use_local_camera=True,
        camera_index=0,
        use_threading=True
    )

    face_tracker = FaceTracker()
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = camera.capture_frame()

            if frame is not None:
                frame_count += 1

                detection = face_tracker.detect_face(frame)
                display = face_tracker.annotate_frame(frame, detection)

                # Info overlay
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0.1 else 0
                cv2.putText(display, f"Frame: {frame_count} | FPS: {fps:.1f}",
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow(window_name, display)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.imshow(window_name, placeholder)

            # Check if window was closed (waitKey still needed for OpenCV)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.cleanup()
        cv2.destroyAllWindows()
        print(f"\nDone! {frame_count} frames")


def test_with_nao(nao_ip: str):
    print("DUAL CAMERA TEST - NAO Mode")
    print(f"Connecting to NAO at {nao_ip}...")
    print("Face tracking: ENABLED by default")
    print("Press 'r' to reset head, 't' to toggle tracking, 'q' to quit")
    print("=" * 60)

    # Import NAO
    try:
        from sic_framework.devices import Nao
        from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
        from sic_framework.devices.common_naoqi.naoqi_autonomous import (
            NaoWakeUpRequest, NaoRestRequest
        )
    except ImportError:
        print("ERROR: SIC Framework not installed!")
        return

    # Create windows FIRST before any camera init
    LAPTOP_WINDOW = "1. LAPTOP - Pose Detection"
    NAO_WINDOW = "2. NAO - Face Tracking"

    print("\n[Creating Windows]")
    cv2.namedWindow(LAPTOP_WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(LAPTOP_WINDOW, 50, 100)
    cv2.resizeWindow(LAPTOP_WINDOW, 640, 480)
    print(f"  ✓ {LAPTOP_WINDOW}")

    cv2.namedWindow(NAO_WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(NAO_WINDOW, 720, 100)
    cv2.resizeWindow(NAO_WINDOW, 640, 480)
    print(f"  ✓ {NAO_WINDOW}")

    # Show placeholders
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.imshow(LAPTOP_WINDOW, placeholder)
    cv2.imshow(NAO_WINDOW, placeholder)
    cv2.waitKey(100)

    # Connect to NAO with camera config
    print("\n[Connecting to NAO]")
    try:
        conf = NaoqiCameraConf(vflip=0)
        nao = Nao(ip=nao_ip, top_camera_conf=conf)
        nao.autonomous.request(NaoWakeUpRequest())
        time.sleep(1)
        print("  ✓ NAO connected and awake")
    except Exception as e:
        print(f"  ✗ NAO connection failed: {e}")
        print("\nFalling back to laptop-only...")
        cv2.destroyAllWindows()
        test_laptop_only()
        return

    print("\n[Initializing Laptop Camera]")
    laptop_camera = CameraManager(
        use_local_camera=True,
        camera_index=0,
        use_threading=True
    )

    print("\n[Initializing NAO Camera]")
    nao_camera = CameraManager(
        nao=nao,
        use_local_camera=False
    )

    face_tracker = FaceTracker(
        smoothing_alpha=0.50,
        dead_zone_pixels=20,
        max_yaw=0.40,
        max_pitch=0.25,
        movement_speed=0.50
    )
    face_tracker.reset_head_position(nao)
    face_tracker.start_tracking()

    print("\n" + "=" * 60)
    print("Running... Close windows or press 'q' to quit")
    print("=" * 60 + "\n")

    frame_count = 0
    start_time = time.time()
    last_track_time = 0
    TRACK_INTERVAL = 0.3

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            # === LAPTOP CAMERA (left window) ===
            laptop_frame = laptop_camera.capture_frame()
            if laptop_frame is not None:
                frame_count += 1
                fps = frame_count / elapsed if elapsed > 0.1 else 0

                display_laptop = laptop_frame.copy()
                cv2.putText(display_laptop, "LAPTOP - Pose Detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_laptop, f"Frame: {frame_count} | FPS: {fps:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow(LAPTOP_WINDOW, display_laptop)

            # === NAO CAMERA (right window) ===
            nao_frame = nao_camera.capture_frame()
            if nao_frame is not None:
                detection = face_tracker.detect_face(nao_frame)
                display_nao = face_tracker.annotate_frame(nao_frame, detection)

                cv2.putText(display_nao, "NAO - Face Tracking",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if now - last_track_time >= TRACK_INTERVAL:
                    if detection.center:
                        moved = face_tracker.move_nao_head_to_face(
                            nao, detection.center, nao_frame.shape
                        )
                        if moved:
                            last_track_time = now
                    else:
                        last_track_time = now

                cv2.imshow(NAO_WINDOW, display_nao)
            else:
                # Show waiting message for NAO
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "NAO Camera - Waiting...",
                            (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                cv2.imshow(NAO_WINDOW, waiting)

            # Check if windows are still open
            key = cv2.waitKey(1) & 0xFF
            laptop_open = cv2.getWindowProperty(LAPTOP_WINDOW, cv2.WND_PROP_VISIBLE) >= 1
            nao_open = cv2.getWindowProperty(NAO_WINDOW, cv2.WND_PROP_VISIBLE) >= 1

            # Keyboard controls
            if key == ord('r'):
                print("Resetting NAO head to center...")
                face_tracker.reset_head_position(nao)

            if key == ord('t'):
                if face_tracker.tracking_active:
                    print("Pausing face tracking...")
                    face_tracker.stop_tracking()
                else:
                    print("Resuming face tracking...")
                    face_tracker.start_tracking()

            if key == ord('q') or not laptop_open or not nao_open:
                print("\nWindow closed or 'q' pressed - shutting down...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("\n[Cleanup]")
        face_tracker.reset_head_position(nao)
        laptop_camera.cleanup()
        nao_camera.cleanup()
        try:
            from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
            nao.autonomous.request(NaoRestRequest())
            print("  ✓ NAO resting")
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"  ✓ Done! {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Test Dual Camera System")
    parser.add_argument("--nao", action="store_true", help="Enable NAO")
    parser.add_argument("--ip", type=str, default="10.0.0.241", help="NAO IP")
    args = parser.parse_args()

    if args.nao:
        test_with_nao(args.ip)
    else:
        test_laptop_only()


if __name__ == "__main__":
    main()
