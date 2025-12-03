# demos/nao/scenes/scene1_greeting_calibration.py
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoqiAnimationRequest,
    NaoqiSetAnglesRequest
)
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest

from dialogue.dialogue_manager import DialogueManager
from vision.camera_manager import CameraManager
from vision.pose_analyzer import PoseAnalyzer
from vision.face_tracker import FaceTracker
import time
import cv2
import numpy as np
import threading


class Scene1GreetingCalibration(SICApplication):
    """
    Scene 1: Greeting & Calibration with Dual Camera System + Non-blocking dialogue
    - Laptop camera: Pose detection (full body) - LEFT WINDOW
    - NAO camera: Face tracking (head movement) - RIGHT WINDOW
    """

    def __init__(self, use_nao=True, nao_ip="10.0.0.241"):
        super(Scene1GreetingCalibration, self).__init__()

        self.use_nao = use_nao
        self.nao_ip = nao_ip
        self.nao = None
        self.dialogue_manager = None

        # Dual camera system
        self.laptop_camera_manager = None
        self.nao_camera_manager = None
        self.pose_analyzer = None
        self.face_tracker = None

        # Window positioning
        self.laptop_window_name = "1. Laptop Camera - Pose Detection"
        self.nao_window_name = "2. NAO Camera - Face Tracking"

        # Threading for non-blocking dialogue
        self.listening_thread = None
        self.listening_active = False
        self.listening_result = None
        self.listening_lock = threading.Lock()

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        """Initialize dual camera system"""
        mode = "NAO Mode" if self.use_nao else "Laptop Test Mode"
        self.logger.info(f"Setting up Scene 1 ({mode})...")

        try:
            # Initialize NAO
            if self.use_nao:
                self.logger.info("Connecting to NAO...")
                self.nao = Nao(ip=self.nao_ip)
            else:
                self.logger.info("Skipping NAO connection (laptop mode)")
                self.nao = None

            # Initialize DialogueManager
            self.logger.info("Initializing dialogue system...")
            scene1_prompt = """You are Coach Nao, a confident fitness trainer robot.
You're meeting someone for the first time at their home.
Be friendly but slightly pushy about training.
Keep responses under 20 words.
Stay in character as Coach Nao."""

            self.dialogue_manager = DialogueManager(
                nao=self.nao if self.use_nao else None,
                use_local_mic=not self.use_nao,
                system_prompt=scene1_prompt
            )

            # CAMERA 1: LAPTOP - For pose detection
            self.logger.info("Initializing laptop camera for pose detection...")
            self.laptop_camera_manager = CameraManager(
                use_local_camera=True,
                camera_index=0,
                use_threading=True
            )

            if not self.laptop_camera_manager.is_available():
                raise RuntimeError("Laptop camera not available!")

            self.pose_analyzer = PoseAnalyzer(camera_manager=self.laptop_camera_manager)
            self.logger.info("✓ Laptop camera ready for pose detection")

            # CAMERA 2: NAO - For face tracking
            if self.use_nao and self.nao:
                self.logger.info("Initializing NAO camera for face tracking...")
                self.nao_camera_manager = CameraManager(
                    nao=self.nao,
                    use_local_camera=False,
                    use_threading=True
                )

                if self.nao_camera_manager.is_available():
                    self.face_tracker = FaceTracker(camera_manager=self.nao_camera_manager)

                    # Reset head to center position
                    self.logger.info("Resetting NAO head to center...")
                    self.face_tracker.reset_head_position(self.nao)

                    self.logger.info("✓ NAO camera ready for face tracking")
                else:
                    self.logger.warning("NAO camera not available, face tracking disabled")
            else:
                self.logger.info("NAO not enabled, face tracking disabled")

            self._setup_windows()
            self.logger.info(f"Scene 1 setup complete ({mode})")

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_windows(self):
        """Setup and position windows side by side"""
        cv2.namedWindow(self.laptop_window_name, cv2.WINDOW_NORMAL)

        if self.use_nao and self.nao_camera_manager:
            cv2.namedWindow(self.nao_window_name, cv2.WINDOW_NORMAL)

            # Position windows side by side
            cv2.moveWindow(self.laptop_window_name, 50, 50)
            cv2.resizeWindow(self.laptop_window_name, 640, 480)

            cv2.moveWindow(self.nao_window_name, 700, 50)
            cv2.resizeWindow(self.nao_window_name, 640, 480)

            self.logger.info("✓ Windows positioned side by side")

    def listen_for_person_async(self, max_duration=30.0):
        """
        Start listening in background thread (non-blocking)
        Returns immediately, check is_listening() and get_listening_result()
        """
        if self.listening_active:
            self.logger.warning("Already listening!")
            return False

        def listen_thread():
            self.logger.info("[LISTENING in background thread...]")
            result = self.dialogue_manager.listen_and_respond_auto(
                max_duration=max_duration,
                silence_threshold=0.01,
                silence_duration=2
            )

            with self.listening_lock:
                self.listening_result = result
                self.listening_active = False

            if result and 'user_input' in result:
                self.logger.info(f"[PERSON]: {result['user_input']}")
            else:
                self.logger.warning("[No speech detected]")

        # Start listening thread
        with self.listening_lock:
            self.listening_active = True
            self.listening_result = None

        self.listening_thread = threading.Thread(target=listen_thread, daemon=True)
        self.listening_thread.start()
        return True

    def is_listening(self):
        """Check if still listening"""
        with self.listening_lock:
            return self.listening_active

    def get_listening_result(self):
        """Get result from listening (blocks until ready)"""
        # Wait for listening to complete
        if self.listening_thread:
            self.listening_thread.join()

        with self.listening_lock:
            result = self.listening_result
            self.listening_result = None

        if result and 'user_input' in result:
            return result['user_input']
        return None

    def nao_speak(self, text, animation=None, wait=True):
        """Make NAO speak or print text"""
        self.logger.info(f"[NAO]: {text}")

        if self.use_nao and self.nao:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))

            if animation:
                self.logger.info(f"Animation: {animation.split('/')[-1]}")
                self.nao.motion.request(NaoqiAnimationRequest(animation), block=False)
        else:
            if animation:
                self.logger.info(f"Animation: {animation.split('/')[-1]}")

        if wait:
            time.sleep(len(text.split()) * 0.4)

    def nao_animate(self, animation):
        """Play NAO animation"""
        self.logger.info(f"Animation: {animation.split('/')[-1]}")

        if self.use_nao and self.nao:
            self.nao.motion.request(NaoqiAnimationRequest(animation), block=True)
        else:
            time.sleep(1)

    def run(self):
        """Execute Scene 1 with dual camera display and non-blocking dialogue"""
        try:
            mode = "NAO MODE" if self.use_nao else "LAPTOP TEST MODE"
            self.logger.info("\n" + "=" * 70)
            self.logger.info(f"SCENE 1: GREETING & CALIBRATION ({mode})")
            self.logger.info("Camera 1 (Laptop): Pose detection - LEFT WINDOW")
            if self.use_nao:
                self.logger.info("Camera 2 (NAO): Face tracking - RIGHT WINDOW")
            self.logger.info("Press 'q' to quit")
            self.logger.info("=" * 70 + "\n")

            # Wake up NAO
            if self.use_nao and self.nao:
                self.nao.autonomous.request(NaoWakeUpRequest())
                time.sleep(1)

            scene_step = 0
            step_start_time = time.time()
            frame_count = 0

            # Face tracking state
            face_tracking_active = False
            last_face_track_time = 0

            # Dialogue state
            waiting_for_response = False
            listening_started = False

            # ============================================================
            # MAIN CAMERA LOOP - Never blocks!
            # ============================================================
            while True:
                current_time = time.time()

                # ============================================================
                # CAMERA 1: Laptop camera (pose detection) - ALWAYS UPDATING
                # ============================================================
                laptop_frame = self.laptop_camera_manager.capture_frame()

                if laptop_frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1

                # Analyze pose
                angles, annotated_laptop_frame = self.pose_analyzer.analyze_frame(laptop_frame)
                display_laptop_frame = annotated_laptop_frame if annotated_laptop_frame is not None else laptop_frame

                # Add laptop camera info
                cv2.putText(display_laptop_frame, "LAPTOP CAMERA - Pose Detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(display_laptop_frame, f"Frame {frame_count}",
                            (display_laptop_frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Pose detection status
                if angles is not None:
                    cv2.putText(display_laptop_frame, "Pose: Detected",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_laptop_frame, "Pose: Not detected",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Scene info
                scene_names = [
                    "Waiting to start...",
                    "First greeting",
                    "Waiting for response",
                    "NAO introduction",
                    "Face calibration",
                    "Pose calibration",
                    "Ready check",
                    "Scene complete"
                ]
                scene_name = scene_names[min(scene_step, len(scene_names) - 1)]
                cv2.putText(display_laptop_frame, f"Step: {scene_name}",
                            (10, display_laptop_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Show listening status
                if self.is_listening():
                    cv2.putText(display_laptop_frame, "LISTENING...",
                                (10, display_laptop_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow(self.laptop_window_name, display_laptop_frame)

                # ============================================================
                # CAMERA 2: NAO camera (face tracking) - ALWAYS UPDATING
                # ============================================================
                if self.use_nao and self.nao_camera_manager and self.face_tracker:
                    nao_frame = self.nao_camera_manager.capture_frame()

                    if nao_frame is not None:
                        face_detected, face_center, face_rect, annotated_nao_frame = \
                            self.face_tracker.detect_face(nao_frame)

                        display_nao_frame = annotated_nao_frame if annotated_nao_frame is not None else nao_frame

                        cv2.putText(display_nao_frame, "NAO CAMERA - Face Tracking",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Face tracking status
                        if face_tracking_active:
                            cv2.putText(display_nao_frame, "Tracking: ACTIVE",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # Track face every 300ms
                            if current_time - last_face_track_time > 0.3:
                                if face_detected and face_center:
                                    self.face_tracker.move_nao_head_to_face(
                                        self.nao, face_center, nao_frame.shape
                                    )
                                    last_face_track_time = current_time
                        else:
                            cv2.putText(display_nao_frame, "Tracking: IDLE",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                        cv2.imshow(self.nao_window_name, display_nao_frame)
                    else:
                        # No frame - show black screen
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(black_frame, "NAO CAMERA - No frame",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(black_frame, "Waiting for camera...",
                                    (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow(self.nao_window_name, black_frame)

                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break

                # ============================================================
                # SCENE SCRIPT - Non-blocking!
                # ============================================================

                if scene_step == 0:
                    self.nao_speak(
                        "Hey there you are, almost did not see you!",
                        animation="animations/Stand/Gestures/Hey_1",
                        wait=False
                    )
                    scene_step = 1
                    step_start_time = current_time
                    waiting_for_response = False

                elif scene_step == 1:
                    if not waiting_for_response and current_time - step_start_time > 3:
                        self.logger.info("[Starting to listen for person's response...]")
                        self.listen_for_person_async(max_duration=30.0)
                        waiting_for_response = True

                    # Check if listening completed
                    if waiting_for_response and not self.is_listening():
                        person_response_1 = self.get_listening_result()
                        scene_step = 2
                        step_start_time = current_time
                        waiting_for_response = False

                elif scene_step == 2 and current_time - step_start_time > 1:
                    self.nao_speak(
                        "I'm Coach Nao, the smallest but smartest trainer in town.",
                        wait=False
                    )
                    time.sleep(3)

                    self.nao_speak(
                        "I will be your trainer today and guide you through the most intense session you will ever be doing.",
                        wait=False
                    )
                    time.sleep(4)

                    self.nao_animate("animations/Stand/Gestures/Yes_1")
                    time.sleep(2)

                    # Enable face tracking
                    face_tracking_active = True
                    self.logger.info("✓ Face tracking ACTIVATED")

                    self.nao_speak(
                        "Let me get a good look at you. Please look at me!",
                        wait=False
                    )

                    scene_step = 3
                    step_start_time = current_time

                elif scene_step == 3 and current_time - step_start_time > 5:
                    if self.use_nao and self.face_tracker:
                        nao_frame = self.nao_camera_manager.capture_frame()
                        if nao_frame is not None:
                            face_detected, _, _, _ = self.face_tracker.detect_face(nao_frame)

                            if face_detected:
                                self.nao_speak("Great! I can see your face clearly now!", wait=False)
                                time.sleep(3)
                                scene_step = 4
                                step_start_time = current_time
                            else:
                                self.nao_speak("Come on, let me see you!", wait=False)
                                step_start_time = current_time
                                time.sleep(2)
                        else:
                            step_start_time = current_time
                            time.sleep(0.5)
                    else:
                        scene_step = 4
                        step_start_time = current_time

                elif scene_step == 4 and current_time - step_start_time > 2:
                    self.nao_speak(
                        "Now, please stand in front of the laptop camera so I can see your full body movements.",
                        wait=False
                    )
                    time.sleep(5)
                    scene_step = 5
                    step_start_time = current_time

                elif scene_step == 5 and current_time - step_start_time > 2:
                    if angles is not None:
                        self.logger.info("✓ Pose detected successfully")
                        self.nao_speak("Perfect! I can see you clearly now.", wait=False)
                        time.sleep(3)
                        scene_step = 6
                        step_start_time = current_time
                    else:
                        self.nao_speak("Hmm, can you step back so I can see your full body?", wait=False)
                        time.sleep(3)
                        step_start_time = current_time

                elif scene_step == 6:
                    if not waiting_for_response and current_time - step_start_time > 1:
                        self.nao_speak(
                            "Perfect. Ready for round one?",
                            animation="animations/Stand/Gestures/Enthusiastic_1",
                            wait=False
                        )
                        time.sleep(3)

                        self.logger.info("[Starting to listen for person's response...]")
                        self.listen_for_person_async(max_duration=30.0)
                        waiting_for_response = True

                    # Check if listening completed
                    if waiting_for_response and not self.is_listening():
                        person_response = self.get_listening_result()

                        if person_response:
                            if any(word in person_response.lower() for word in ["yes", "yeah", "sure", "ready"]):
                                self.nao_speak("Excellent! Let's begin!", wait=False)
                            elif any(word in person_response.lower() for word in ["no", "wait", "not"]):
                                self.nao_speak("Too late! We're starting now!", wait=False)
                            else:
                                self.nao_speak("I'll take that as a yes!", wait=False)
                        else:
                            self.nao_speak("Silence means yes! Let's go!", wait=False)

                        time.sleep(3)
                        scene_step = 7
                        waiting_for_response = False

                elif scene_step == 7:
                    self.logger.info("\n" + "=" * 70)
                    self.logger.info("END OF SCENE 1")
                    self.logger.info("Dual camera system tested successfully!")
                    self.logger.info("=" * 70 + "\n")
                    time.sleep(2)
                    break

        except KeyboardInterrupt:
            self.logger.info("\nScene interrupted by user")
        except Exception as e:
            self.logger.error(f"Scene error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset head before cleanup
            if self.use_nao and self.face_tracker:
                self.logger.info("Resetting head to center...")
                self.face_tracker.reset_head_position(self.nao)

            cv2.destroyAllWindows()
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up...")

        try:
            # Stop any active listening thread
            if self.listening_thread and self.listening_thread.is_alive():
                self.logger.info("Waiting for listening thread to finish...")
                self.listening_thread.join(timeout=2)

            if self.dialogue_manager:
                self.dialogue_manager.cleanup()

            if self.laptop_camera_manager:
                self.laptop_camera_manager.cleanup()

            if self.nao_camera_manager:
                self.nao_camera_manager.cleanup()

            if self.pose_analyzer:
                self.pose_analyzer.cleanup()

            if self.face_tracker:
                self.face_tracker.cleanup()

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
        finally:
            self.logger.info("Scene 1 complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scene 1: Dual Camera System with Non-blocking Dialogue")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["nao", "laptop"],
        default="laptop",
        help="Mode: 'nao' for full system, 'laptop' for testing"
    )
    parser.add_argument(
        "--nao-ip",
        type=str,
        default="10.0.0.241",
        help="NAO robot IP address"
    )

    args = parser.parse_args()
    use_nao = (args.mode == "nao")

    print("\n" + "=" * 70)
    print("NAO FITNESS TRAINER - NON-BLOCKING DUAL CAMERA SYSTEM")
    print("Scene 1: Greeting & Calibration")
    print(f"Mode: {'NAO Robot' if use_nao else 'Laptop Testing'}")
    print("")
    print("DISPLAY:")
    print("  LEFT WINDOW  → Laptop camera (Pose detection)")
    if use_nao:
        print("  RIGHT WINDOW → NAO camera (Face tracking)")
    print("")
    print("THREADING:")
    print("  ✓ Camera feeds update continuously")
    print("  ✓ Dialogue runs in background thread")
    print("  ✓ Face tracking never freezes")
    print("")
    print("Controls:")
    print("  Press 'q' to quit")
    print("=" * 70 + "\n")

    scene = Scene1GreetingCalibration(use_nao=use_nao, nao_ip=args.nao_ip)
    scene.run()