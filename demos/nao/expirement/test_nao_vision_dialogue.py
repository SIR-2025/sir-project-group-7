from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.core.message_python2 import CompressedImageMessage, AudioRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest

import sys
import os
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from vision.pose_analyzer import PoseAnalyzer
from dialogue.dialogue_manager import DialogueManager

import queue
import cv2
import numpy as np
import time
import threading


class IntegratedNAOCamera:
    def __init__(self, image_queue):
        self.image_queue = image_queue
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.update_thread = None

    def start(self):
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def _update_loop(self):
        while self.running:
            try:
                frame = self.image_queue.get(timeout=0.1)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                with self.frame_lock:
                    self.current_frame = frame_bgr

            except queue.Empty:
                continue
            except Exception as e:
                print("Frame update error: {}".format(e))

    def capture_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def capture_image_base64(self):
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
        return self.current_frame is not None

    def stop(self):
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)

    def cleanup(self):
        self.stop()


class NAOIntegratedSystem(SICApplication):

    def __init__(self, nao_ip="10.0.0.24", pose_model_path=None):
        super(NAOIntegratedSystem, self).__init__()

        self.nao_ip = nao_ip
        self.nao = None
        self.imgs = queue.Queue()
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

    def on_image(self, image_message: CompressedImageMessage):
        self.imgs.put(image_message.image)

    def setup(self):
        self.logger.info("=" * 60)
        self.logger.info("NAO Integrated Training System")
        self.logger.info("=" * 60)

        # Setup NAO camera
        self.logger.info("Connecting to NAO at {}...".format(self.nao_ip))
        camera_conf = NaoqiCameraConf(
            vflip=-1,
            brightness=55,
            contrast=32
        )

        self.nao = Nao(ip=self.nao_ip, top_camera_conf=camera_conf)
        self.nao.top_camera.register_callback(self.on_image)

        self.logger.info("Waking up NAO...")
        self.nao.autonomous.request(NaoWakeUpRequest())
        time.sleep(2)

        self.logger.info("Starting camera stream...")
        self.camera = IntegratedNAOCamera(self.imgs)
        self.camera.start()

        self.logger.info("Waiting for camera...")
        for _ in range(50):
            if self.camera.is_available():
                break
            time.sleep(0.1)

        if not self.camera.is_available():
            self.logger.error("Camera not available!")
            return

        self.logger.info("Camera ready!")

        self.logger.info("Initializing pose analyzer...")

        if self.pose_model_path:
            import vision.pose_analyzer as pa_module
            original_path = pa_module.MODEL_PATH
            pa_module.MODEL_PATH = self.pose_model_path
            self.logger.info("Using pose model: {}".format(self.pose_model_path))

        try:
            self.pose_analyzer = PoseAnalyzer(camera_manager=self.camera)
        except Exception as e:
            self.logger.error("Failed to initialize pose analyzer: {}".format(e))
            self.logger.error("Make sure the pose model file exists!")
            if self.pose_model_path:
                pa_module.MODEL_PATH = original_path
            raise

        self.logger.info("Initializing dialogue manager...")
        self.dialogue_manager = DialogueManager(
            nao=self.nao,
            use_local_mic=False,
            camera_manager=self.camera,
            pose_analyzer=self.pose_analyzer
        )

        self.logger.info("All systems ready!")
        self.logger.info("")

    def start_training_session(self, exercise_name="squat"):
        self.logger.info("Starting training session: {}".format(exercise_name))
        self.current_exercise = exercise_name
        self.session_active = True
        self.attempt_count = 0

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
        self.logger.info("Starting continuous monitoring...")
        self.logger.info("Press 'q' to stop, 's' to start session, 'r' to reset")

        try:
            while not self.shutdown_event.is_set():
                frame = self.camera.capture_frame()

                if frame is None:
                    time.sleep(0.1)
                    continue

                angles, annotated_frame = self.pose_analyzer.analyze_frame(frame)
                display_frame = annotated_frame if annotated_frame is not None else frame

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
                    cv2.imshow("NAO Integrated Training", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break

                elif key == ord('s'):
                    if not self.session_active:
                        self.start_training_session(self.current_exercise or "squat")
                    else:
                        self.logger.info("Session already active")

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
        if not self.session_active:
            return

        self.logger.info("Ending training session...")

        # Calculate final stats (simplified)
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
        """
        Text-to-speech using NAO's built-in TTS.
        Much simpler and faster than OpenAI TTS!
        """
        try:
            self.logger.info("[NAO SPEAKING] {}".format(text))

            self.nao.tts.request(NaoqiTextToSpeechRequest(text, animated=animated))

            time.sleep(0.5)

        except Exception as e:
            self.logger.error("Speech error: {}".format(e))
            self.logger.info("FALLBACK: Would say: {}".format(text))

    def run(self):
        self.logger.info("Starting integrated system...")
        self.logger.info("")

        try:
            self.continuous_monitoring()

        except Exception as e:
            self.logger.error("System error: {}".format(e))
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup_resources()

    def cleanup_resources(self):
        self.logger.info("Shutting down integrated system...")

        if self.session_active:
            self.end_session()

        try:
            self.logger.info("Putting NAO to rest...")
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

        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    system = NAOIntegratedSystem(
        nao_ip="10.0.0.243",
        pose_model_path=r"..\..\sir-project-group-7\pose_landmarkers\pose_landmarker_full.task"
    )
    system.run()