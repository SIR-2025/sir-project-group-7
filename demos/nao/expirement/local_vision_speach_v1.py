import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import os
from datetime import datetime
from pathlib import Path
import json
import time

# === CONFIGURATION ===
CONFIG = {
    "MODEL_PATH": "../../../pose_landmarkers/pose_landmarker_full.task",
    "WHISPER_MODEL": "base",

    "SQUAT_DEPTH_THRESHOLD": 100,  # degrees
    "SQUAT_STANDING_THRESHOLD": 150,  # degrees
    "REP_TARGET": 3,

    # Voice settings
    "USE_TTS": True,  # Robot speaks
    "USE_STT": True,  # Voice commands
    "VOICE_LISTEN_DURATION": 3,  # seconds
}

# === VOICE HANDLER ===
class VoiceHandler:
    """Handles TTS (Coqui) and STT (Whisper)"""

    def __init__(self, use_tts=True, use_stt=True, whisper_model="base"):
        self.use_tts = use_tts
        self.use_stt = use_stt
        self.tts_ready = False
        self.stt_ready = False

        if use_tts:
            try:
                from TTS.api import TTS
                import sounddevice as sd
                import soundfile as sf

                print("🔊 Loading TTS model...")
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                self.sd = sd
                self.sf = sf
                self.tts_ready = True
                print(" TTS ready")
            except Exception as e:
                print(f"  TTS not available: {e}")
                print("   Install: pip install TTS sounddevice soundfile")

        if use_stt:
            try:
                import whisper
                import pyaudio
                import wave

                print(f"🎤 Loading Whisper '{whisper_model}' model...")
                self.whisper = whisper.load_model(whisper_model)
                self.pyaudio = pyaudio
                self.wave = wave
                self.stt_ready = True
                print(" Whisper ready")
            except Exception as e:
                print(f"  Whisper not available: {e}")
                print("   Install: pip install openai-whisper pyaudio")

    def speak(self, text, block=True):
        """Convert text to speech"""
        print(f"\nCoach Nao: {text}")

        if not self.tts_ready:
            return

        try:
            temp_file = f"temp_tts_{time.time()}.wav"
            self.tts.tts_to_file(text=text, file_path=temp_file)

            if block:
                data, samplerate = self.sf.read(temp_file)
                self.sd.play(data, samplerate)
                self.sd.wait()

            if os.path.exists(temp_file):
                os.remove(temp_file)

        except Exception as e:
            print(f"  TTS error: {e}")

    def listen(self, duration=3):
        """Record audio and convert to text using Whisper"""
        if not self.stt_ready:
            print("  Whisper not available")
            return None

        print("🎤 Listening... (speak now)")

        try:
            audio_format = self.pyaudio.paInt16
            channels = 1
            rate = 16000
            chunk = 1024

            p = self.pyaudio.PyAudio()
            stream = p.open(
                format=audio_format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk
            )

            frames = []
            for _ in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            temp_file = f"temp_audio_{time.time()}.wav"
            wf = self.wave.open(temp_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            result = self.whisper.transcribe(temp_file, language="en", fp16=False)
            text = result["text"].strip().lower()

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if text:
                print(f"👤 You said: '{text}'")
                return text
            else:
                print("  No speech detected")
                return None

        except Exception as e:
            print(f"  Whisper error: {e}")
            return None


class SquatAnalyzer:
    """Analyzes squat form and detects reps"""

    def __init__(self, depth_threshold=100, standing_threshold=150):
        self.depth_threshold = depth_threshold
        self.standing_threshold = standing_threshold
        self.state = "standing"
        self.lowest_angle = 180

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze(self, landmarks):
        """Analyze squat form"""
        if len(landmarks) < 29:
            return None

        # Get key landmarks
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
        shoulder = landmarks[11]

        # Calculate angles
        knee_angle = self.calculate_angle(hip, knee, ankle)
        back_angle = self.calculate_angle(shoulder, hip, knee)

        # Check for form issues
        issues = []

        if knee_angle > self.depth_threshold:
            issues.append("shallow_depth")

        if knee.x < ankle.x - 0.05:
            issues.append("knees_in")

        if back_angle < 160:
            issues.append("back_rounding")

        return {
            "knee_angle": knee_angle,
            "back_angle": back_angle,
            "issues": issues
        }

    def detect_rep(self, knee_angle):
        """Detect when a rep is completed"""
        rep_completed = False

        # State machine: standing → squatting → standing
        if knee_angle < 90 and self.state == "standing":
            self.state = "squatting"
            self.lowest_angle = knee_angle

        elif knee_angle < self.lowest_angle and self.state == "squatting":
            self.lowest_angle = knee_angle

        elif knee_angle > self.standing_threshold and self.state == "squatting":
            self.state = "standing"
            rep_completed = True

        return rep_completed

    def reset(self):
        """Reset state for new set"""
        self.state = "standing"
        self.lowest_angle = 180


# === FEEDBACK GENERATOR ===
class FeedbackGenerator:
    PRAISE_MESSAGES = [
        "Great depth! Strong and stable!",
        "Excellent form! Keep that control.",
        "Perfect squat! That's how it's done.",
        "Nice work! Solid technique.",
        "Outstanding! You're on fire!",
    ]

    ISSUE_FEEDBACK = {
        "shallow_depth": [
            "Good effort. Try sitting deeper—imagine a low chair.",
            "Almost there! Go one inch lower next time.",
            "Nice try. Aim for parallel or below.",
        ],
        "knees_in": [
            "Watch those knees! Press them out over your toes.",
            "Good work. Keep knees aligned with your feet.",
            "Nice! Just push your knees out slightly.",
        ],
        "back_rounding": [
            "Great effort! Lift your chest—eyes forward.",
            "Good depth! Keep your spine tall.",
            "Almost perfect! Chest up, shoulders back.",
        ],
    }

    def __init__(self):
        self.praise_index = 0
        self.issue_indices = {issue: 0 for issue in self.ISSUE_FEEDBACK}

    def get_feedback(self, rep_number, analysis):
        """Generate contextual feedback"""
        issues = analysis.get("issues", [])

        # No issues - give praise
        if not issues:
            feedback = self.PRAISE_MESSAGES[self.praise_index % len(self.PRAISE_MESSAGES)]
            self.praise_index += 1
            return feedback

        primary_issue = issues[0]
        feedback_list = self.ISSUE_FEEDBACK[primary_issue]
        index = self.issue_indices[primary_issue]

        feedback = feedback_list[index % len(feedback_list)]
        self.issue_indices[primary_issue] += 1

        return feedback


# === MAIN COACH CLASS ===
class CoachNao:
    def __init__(self, config):
        self.config = config

        # Initialize MediaPipe
        print("Initializing Coach Nao...")
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=config["MODEL_PATH"]),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print(" MediaPipe Pose ready")

        # Initialize components
        self.voice = VoiceHandler(
            use_tts=config["USE_TTS"],
            use_stt=config["USE_STT"],
            whisper_model=config["WHISPER_MODEL"]
        )
        self.squat_analyzer = SquatAnalyzer(
            depth_threshold=config["SQUAT_DEPTH_THRESHOLD"],
            standing_threshold=config["SQUAT_STANDING_THRESHOLD"]
        )
        self.feedback_gen = FeedbackGenerator()

        # Session state
        self.phase = "warmup"
        self.rep_count = 0
        self.feedback_history = []
        self.timestamp = 0
        self.session_start = datetime.now()

        # Pose connections for drawing
        self.pose_connections = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ])

        print("Coach Nao ready!")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACE - Start workout")
        print("  V     - Voice command mode")
        print("  R     - Reset exercise")
        print("  Q     - Quit")
        print("="*60 + "\n")

    def speak(self, text):
        """Wrapper for voice output"""
        self.voice.speak(text)

    def handle_voice_command(self):
        """Process voice commands"""
        response = self.voice.listen(duration=self.config["VOICE_LISTEN_DURATION"])

        if not response:
            self.speak("I didn't hear anything. Try again.")
            return None

        # Command matching
        if any(word in response for word in ["start", "begin", "go", "let's go"]):
            if self.phase == "warmup":
                self.phase = "exercise"
                self.speak("Starting squats now. Three controlled reps. Go!")
                return "start"

        elif any(word in response for word in ["reset", "again", "restart", "over"]):
            self.rep_count = 0
            self.squat_analyzer.reset()
            self.phase = "exercise"
            self.feedback_history = []
            self.speak("Reset! Let's go again!")
            return "reset"

        elif any(word in response for word in ["pause", "stop", "wait", "hold"]):
            self.phase = "paused"
            self.speak("Paused. Say resume when you're ready.")
            return "pause"

        elif any(word in response for word in ["resume", "continue", "keep going"]):
            self.phase = "exercise"
            self.speak("Resuming! Let's go!")
            return "resume"

        elif any(word in response for word in ["quit", "exit", "done", "finish", "stop workout"]):
            self.speak("Great session! See you next time.")
            return "quit"

        else:
            self.speak("I didn't catch that. Try: start, reset, pause, resume, or quit.")
            return None

    def draw_pose(self, frame, landmarks):
        """Draw pose skeleton on frame"""
        h, w = frame.shape[:2]

        # Draw connections
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def save_session(self):
        session_data = {
            "date": self.session_start.isoformat(),
            "duration": (datetime.now() - self.session_start).seconds,
            "reps_completed": self.rep_count,
            "target_reps": self.config["REP_TARGET"],
            "feedback": self.feedback_history
        }

        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)

        # Save to JSON
        filename = f"data/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"\n💾 Session saved: {filename}")

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print(" Error: Cannot open camera")
            return

        self.speak("Stand in front of the camera. Ready to work out?")
        self.speak("Press space to start, or press V for voice commands.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]

                # Process frame with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(mp_image, self.timestamp)

                # Process pose
                if result and result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    self.draw_pose(frame, landmarks)

                    # Analyze exercise
                    if self.phase == "exercise":
                        analysis = self.squat_analyzer.analyze(landmarks)

                        if analysis:
                            # Detect rep completion
                            if self.squat_analyzer.detect_rep(analysis["knee_angle"]):
                                self.rep_count += 1
                                feedback = self.feedback_gen.get_feedback(self.rep_count, analysis)
                                self.speak(f"Rep {self.rep_count}: {feedback}")
                                self.feedback_history.append({
                                    "rep": self.rep_count,
                                    "feedback": feedback,
                                    "issues": analysis["issues"]
                                })

                                # Check if set complete
                                if self.rep_count >= self.config["REP_TARGET"]:
                                    self.speak("Great set! Shake out your legs.")
                                    self.phase = "done"

                            # Display metrics
                            cv2.putText(frame, f"Reps: {self.rep_count}/{self.config['REP_TARGET']}",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Knee Angle: {analysis['knee_angle']:.0f}°",
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                            if analysis["issues"]:
                                issues_text = ", ".join(analysis["issues"])
                                cv2.putText(frame, f"Issues: {issues_text}",
                                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Display instructions
                if self.phase == "warmup":
                    cv2.putText(frame, "SPACE=start | V=voice | Q=quit",
                               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                elif self.phase == "paused":
                    cv2.putText(frame, "PAUSED - Press V to resume",
                               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                elif self.phase == "done":
                    cv2.putText(frame, "Complete! Press Q to quit",
                               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Coach Nao - Fitness Trainer', frame)
                self.timestamp += 33

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break

                elif key == ord(' ') and self.phase == "warmup":
                    self.phase = "exercise"
                    self.speak("Starting squats now. Three controlled reps. Go!")

                elif key == ord('v'):
                    result = self.handle_voice_command()
                    if result == "quit":
                        break

                elif key == ord('r'):
                    self.rep_count = 0
                    self.squat_analyzer.reset()
                    self.phase = "exercise"
                    self.feedback_history = []
                    self.speak("Reset! Let's go again!")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()

            if self.rep_count > 0:
                self.save_session()

            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Duration: {(datetime.now() - self.session_start).seconds}s")
            print(f"Reps completed: {self.rep_count}/{self.config['REP_TARGET']}")
            print("\nFeedback given:")
            for item in self.feedback_history:
                print(f"  Rep {item['rep']}: {item['feedback']}")
                if item['issues']:
                    print(f"           Issues: {', '.join(item['issues'])}")
            print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  COACH NAO - PHASE 1 + 2")
    print("   Voice-Interactive Fitness Coach")
    print("="*60 + "\n")

    try:
        coach = CoachNao(CONFIG)
        coach.run()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()