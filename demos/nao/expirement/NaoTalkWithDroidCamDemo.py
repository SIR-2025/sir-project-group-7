from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from vision.camera_manager import CameraManager
from vision.pose_analyzer import PoseAnalyzer
from dialogue.dialogue_manager import DialogueManager
import cv2
import numpy as np
import time
import threading


class DroidCamManager:
    """
    DroidCam manager with improved connection handling.
    """

    def __init__(self, phone_ip, phone_port="4747"):
        self.phone_ip = phone_ip
        self.phone_port = phone_port

        # Try different URL formats
        self.alternative_urls = [
            "http://{}:{}/video".format(phone_ip, phone_port),
            "http://{}:{}/mjpegfeed".format(phone_ip, phone_port),
            "http://{}:{}/videofeed".format(phone_ip, phone_port),
        ]

        self.camera = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.update_thread = None
        self.connected = False
        self.connected_url = None

    def connect(self):
        print("\n" + "=" * 60)
        print("Connecting to DroidCam at {}:{}...".format(self.phone_ip, self.phone_port))
        print("=" * 60)

        for url in self.alternative_urls:
            print("\nTrying URL: {}".format(url))

            try:
                self.camera = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                print("  - Waiting for connection...")
                time.sleep(2)

                # Try multiple read attempts
                success = False
                for attempt in range(5):
                    ret, frame = self.camera.read()

                    if ret and frame is not None and frame.size > 0:
                        print("  - SUCCESS! Frame received: {}x{}".format(
                            frame.shape[1], frame.shape[0]))
                        self.connected = True
                        self.connected_url = url
                        self.current_frame = frame
                        return True

                    print("  - Attempt {}/5 failed, retrying...".format(attempt + 1))
                    time.sleep(0.5)

                print("  - Failed to read frames")
                self.camera.release()

            except Exception as e:
                print("  - Exception: {}".format(e))
                if self.camera:
                    self.camera.release()

        print("\n" + "=" * 60)
        print("ERROR: Could not connect to DroidCam")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Is DroidCam app running on phone?")
        print("2. Does http://{}:{}/video work in browser?".format(
            self.phone_ip, self.phone_port))
        print("3. Are phone and computer on same WiFi?")
        print("4. Try restarting DroidCam app")
        print("5. Disable 'Use HTTPS' in DroidCam settings")
        print("=" * 60)

        return False

    def start(self):
        if not self.connected:
            raise RuntimeError("DroidCam not connected! Call connect() first.")

        print("Starting DroidCam frame capture thread...")
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        time.sleep(1)
        print("DroidCam thread started!")

    def _update_loop(self):
        """Continuously update current frame from camera."""
        frame_count = 0
        error_count = 0

        while self.running:
            try:
                ret, frame = self.camera.read()

                if ret and frame is not None and frame.size > 0:
                    frame_count += 1
                    error_count = 0

                    with self.frame_lock:
                        self.current_frame = frame

                    # Log every 100 frames
                    if frame_count % 100 == 0:
                        print("DroidCam: {} frames received".format(frame_count))
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
        """Get current frame (thread-safe)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def capture_image_base64(self):
        """Capture and encode frame as base64."""
        import base64

        frame = self.capture_frame()
        if frame is None:
            return None

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            print("Encoding error: {}".format(e))
            return None

    def is_available(self):
        """Check if camera is available."""
        return self.current_frame is not None

    def stop(self):
        """Stop frame updating."""
        print("Stopping DroidCam...")
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=3)

    def cleanup(self):
        self.stop()
        if self.camera:
            self.camera.release()
        print("DroidCam cleaned up")


class NAOIntegratedSystemWithDroidCam(SICApplication):
    """
    Integrated NAO system with DroidCam.
    """

    def __init__(self, nao_ip="10.0.0.243", phone_ip="", pose_model_path=None):
        super(NAOIntegratedSystemWithDroidCam, self).__init__()

        self.nao_ip = nao_ip
        self.phone_ip = phone_ip
        self.nao = None
        self.pose_model_path = pose_model_path

        self.camera = None
        self.pose_analyzer = None
        self.dialogue_manager = None

        self.current_exercise = None
        self.session_active = False
        self.attempt_count = 0
        self.last_feedback_time = 0
        self.feedback_cooldown = 3.0  # seconds between feedback

        self.show_video = True
        self.show_pose = True

        self.set_log_level(sic_logging.INFO)

        self.setup()

    def setup(self):
        """Initialize all integrated components."""
        self.logger.info("=" * 60)
        self.logger.info("NAO Integrated Training System with DroidCam")
        self.logger.info("=" * 60)

        self.logger.info("Connecting to NAO at {}...".format(self.nao_ip))
        try:
            self.nao = Nao(ip=self.nao_ip)

            self.logger.info("Waking up NAO...")
            self.nao.autonomous.request(NaoWakeUpRequest())
            time.sleep(2)
            self.logger.info("NAO ready!")
        except Exception as e:
            self.logger.warning("Could not connect to NAO: {}".format(e))
            self.logger.warning("Continuing without NAO...")
            self.nao = None

        self.camera = DroidCamManager(self.phone_ip)

        if not self.camera.connect():
            self.logger.error("=" * 60)
            self.logger.error("FAILED TO CONNECT TO DROIDCAM!")
            self.logger.error("=" * 60)
            self.logger.error("")
            self.logger.error("Please check:")
            self.logger.error("1. DroidCam app is running on phone")
            self.logger.error("2. Test in browser: http://{}:4747/video".format(self.phone_ip))
            self.logger.error("3. Phone IP is correct: {}".format(self.phone_ip))
            self.logger.error("")
            raise RuntimeError("Failed to connect to DroidCam!")

        self.camera.start()

        self.logger.info("Waiting for camera to stabilize...")
        for i in range(10):
            if self.camera.is_available():
                self.logger.info("Camera frame available!")
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Camera not providing frames!")

        self.logger.info("DroidCam ready!")

        self.logger.info("Initializing pose analyzer...")

        if self.pose_model_path:
            import vision.pose_analyzer as pa_module
            original_path = getattr(pa_module, 'MODEL_PATH', None)
            pa_module.MODEL_PATH = self.pose_model_path
            self.logger.info("Using pose model: {}".format(self.pose_model_path))

        try:
            self.pose_analyzer = PoseAnalyzer(camera_manager=self.camera)
            self.logger.info("Pose analyzer ready!")
        except Exception as e:
            self.logger.error("Failed to initialize pose analyzer: {}".format(e))
            self.logger.error("Make sure the pose model file exists!")
            raise

        self.logger.info("Initializing dialogue manager...")
        self.dialogue_manager = DialogueManager(
            nao=self.nao,
            use_local_mic=False,
            camera_manager=self.camera,
            pose_analyzer=self.pose_analyzer
        )

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("ALL SYSTEMS READY!")
        self.logger.info("=" * 60)
        self.logger.info("")

    def start_training_session(self, exercise_name="squat"):
        """Start an integrated training session."""
        self.logger.info("Starting training session: {}".format(exercise_name))
        self.current_exercise = exercise_name
        self.session_active = True
        self.attempt_count = 0

        # Give greeting with animation
        greeting = self.dialogue_manager.get_greeting(exercise_name)
        self.logger.info("NAO: {}".format(greeting))
        self.speak(greeting, animated=True)

        time.sleep(2)

        exercise = {
            "name": exercise_name,
            "key_points": [
                "Feet shoulder-width apart",
                "Lower hips back and down",
                "Keep chest up and back straight",
                "Knees track over toes"
            ]
        }

        instructions = self.dialogue_manager.get_instructions(exercise)
        self.logger.info("NAO: {}".format(instructions))
        self.speak(instructions, animated=True)

        time.sleep(2)
        self.logger.info("Session started - monitoring your form...")

    def continuous_monitoring(self):
        """
        Continuously monitor pose and provide real-time feedback.
        """
        self.logger.info("Starting continuous monitoring...")
        self.logger.info("Controls: 'q'=quit, 's'=start session, 'r'=reset, 'p'=toggle pose")
        self.logger.info("")

        frame_count = 0
        try:
            while not self.shutdown_event.is_set():
                # Get current frame from DroidCam
                frame = self.camera.capture_frame()

                if frame is None:
                    print("No frame available, waiting...")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                angles, annotated_frame = self.pose_analyzer.analyze_frame(frame)
                display_frame = annotated_frame if annotated_frame is not None else frame
                cv2.putText(display_frame,
                            "DroidCam ({})".format(frame_count),
                            (display_frame.shape[1] - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                if self.session_active and angles is not None:
                    current_time = time.time()

                    analysis = self.pose_analyzer.check_squat_form(angles)
                    accuracy = analysis['overall_accuracy']

                    cv2.putText(display_frame,
                                "Accuracy: {:.1f}%".format(accuracy),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0) if accuracy > 85 else (0, 165, 255), 2)

                    cv2.putText(display_frame,
                                "Attempt: {}".format(self.attempt_count),
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)

                    if current_time - self.last_feedback_time > self.feedback_cooldown:
                        self.attempt_count += 1

                        feedback = self.dialogue_manager.get_feedback(
                            analysis,
                            self.current_exercise,
                            self.attempt_count
                        )

                        self.logger.info("Feedback #{}: {}".format(
                            self.attempt_count, feedback))
                        self.speak(feedback)

                        self.last_feedback_time = current_time

                        for joint, data in analysis['joints'].items():
                            if data['status'] != 'good':
                                self.logger.info("  - {}: {:.1f}° (target: {:.1f}°)".format(
                                    joint, data['current_angle'], data['target_angle']))

                elif self.session_active and angles is None:
                    cv2.putText(display_frame,
                                "Move into view!",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

                else:
                    cv2.putText(display_frame,
                                "Press 's' to start session",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                if self.show_video:
                    cv2.imshow("NAO Training with DroidCam", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    if not self.session_active:
                        self.start_training_session(self.current_exercise or "squat")
                elif key == ord('r'):
                    self.logger.info("Resetting session...")
                    self.session_active = False
                    self.attempt_count = 0
                    self.last_feedback_time = 0
                elif key == ord('p'):
                    self.show_pose = not self.show_pose
                    self.logger.info("Pose visualization: {}".format(
                        "ON" if self.show_pose else "OFF"))

        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted")
        finally:
            cv2.destroyAllWindows()
            if self.session_active:
                self.end_session()

    def end_session(self):
        """End the training session with summary."""
        if not self.session_active:
            return

        self.logger.info("Ending training session...")

        success = self.attempt_count >= 3
        final_accuracy = 75.0

        closing = self.dialogue_manager.get_closing(
            self.current_exercise,
            final_accuracy,
            success
        )

        self.logger.info("NAO: {}".format(closing))
        self.speak(closing)

        self.logger.info("Session summary:")
        self.logger.info("  Total attempts: {}".format(self.attempt_count))
        self.logger.info("  Exercise: {}".format(self.current_exercise))

        self.session_active = False

    def speak(self, text, animated=False):
        try:
            if self.nao:
                self.logger.info("[NAO SPEAKING] {}".format(text))
                self.nao.tts.request(NaoqiTextToSpeechRequest(text, animated=animated))
                time.sleep(0.5)
            else:
                self.logger.info("[SIMULATED SPEECH] {}".format(text))
        except Exception as e:
            self.logger.error("Speech error: {}".format(e))

    def run(self):
        self.logger.info("Starting integrated system with DroidCam...")

        try:
            self.continuous_monitoring()
        except Exception as e:
            self.logger.error("System error: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup_resources()

    def cleanup_resources(self):
        self.logger.info("Shutting down...")

        if self.session_active:
            self.end_session()

        if self.nao:
            try:
                self.nao.autonomous.request(NaoRestRequest())
            except:
                pass

        if self.camera:
            self.camera.cleanup()

        if self.pose_analyzer:
            self.pose_analyzer.cleanup()

        if self.dialogue_manager:
            self.dialogue_manager.cleanup()

        cv2.destroyAllWindows()
        self.shutdown()


if __name__ == "__main__":
    print("=" * 70)
    print("NAO Integrated Training System with DroidCam")
    print("=" * 70)
    print("")
    print("BEFORE RUNNING:")
    print("1. Start DroidCam app on phone")
    print("2. Test in browser: http://:4747/video")
    print("3. Make sure you see video in browser first!")
    print("")
    print("Configuration:")
    print("  NAO IP: 10.0.0.243")
    print("  DroidCam IP: ")
    print("=" * 70)

    input("Press ENTER when DroidCam is confirmed working in browser...")

    system = NAOIntegratedSystemWithDroidCam(
        nao_ip="10.0.0.243",
        phone_ip="",
        pose_model_path=r"..\..\pose_landmarkers\pose_landmarker_full.task"
    )
    system.run()