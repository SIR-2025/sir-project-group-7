#!/usr/bin/env python3
"""
AI Personal Trainer - Voice Interactive Fitness Coach
A complete desktop application that uses OpenAI services for speech recognition,
natural language understanding, and text-to-speech to provide personalized workout guidance.

Features:
- Voice interaction using OpenAI Whisper (speech-to-text) and TTS
- Computer vision for pose detection and form correction
- Natural conversation with GPT-4 acting as a fitness coach
- Rep counting and workout tracking
- Motivational feedback and exercise guidance
- Gesture synchronization for future robot adaptation

Author: Built for NAO robot adaptation
Date: 2025
"""

import os
import sys
import time
import queue
import threading
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import logging

# Computer Vision and Media
import cv2
import numpy as np
import mediapipe as mp

# Audio Processing
import pyaudio
import wave
from io import BytesIO

# OpenAI Services
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the personal trainer application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = ""  # USER WILL PROVIDE
    OPENAI_MODEL = "gpt-4o"  # Using GPT-4o for better reasoning
    WHISPER_MODEL = "whisper-1"
    TTS_MODEL = "tts-1"
    TTS_VOICE = "nova"  # Energetic voice for a trainer
    
    # Audio Configuration
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 5  # Max recording length
    AUDIO_FORMAT = pyaudio.paInt16
    
    # Video Configuration
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Pose Detection Configuration
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Trainer Personality
    TRAINER_NAME = "Coach Alex"
    TRAINER_STYLE = "energetic and motivating"
    
    # System Prompts
    SYSTEM_PROMPT = f"""You are {TRAINER_NAME}, an enthusiastic and knowledgeable personal fitness trainer.
Your personality is {TRAINER_STYLE}. You are conducting a workout session with a client.

Your capabilities:
- Guide users through various exercises (push-ups, squats, lunges, planks, etc.)
- Count repetitions when requested
- Provide form corrections based on visual feedback
- Offer motivational encouragement
- Answer fitness-related questions
- Adjust workout intensity based on user feedback
- Maintain engaging conversation during rest periods

Guidelines:
1. Always be encouraging and positive
2. Prioritize safety - correct poor form immediately
3. Keep responses concise during active exercises
4. Be more conversational during rest periods
5. Use the user's name when known
6. Count reps out loud when doing an exercise (e.g., "1... 2... 3...")
7. Give specific, actionable form cues
8. Celebrate achievements

Current context: You can see the user through a camera and detect their body pose.
When you receive pose information, use it to provide relevant feedback.

Respond naturally as a personal trainer would in a gym setting."""


# ============================================================================
# POSE DETECTION AND ANALYSIS
# ============================================================================

class PoseAnalyzer:
    """Analyzes body pose using MediaPipe for exercise detection and form correction"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        
        # Exercise state tracking
        self.current_exercise = None
        self.rep_count = 0
        self.rep_stage = None  # For tracking rep phases (up/down)
        self.form_issues = []
        
        logger.info("PoseAnalyzer initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process a video frame for pose detection
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (annotated_frame, pose_data)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Extract pose data
        pose_data = None
        if results.pose_landmarks:
            pose_data = self._extract_pose_data(results.pose_landmarks)
            
        return annotated_frame, pose_data
    
    def _extract_pose_data(self, landmarks) -> Dict:
        """Extract relevant pose information from landmarks"""
        
        # Get key landmarks
        landmark_dict = {}
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_dict[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        # Calculate useful metrics
        pose_data = {
            'landmarks': landmark_dict,
            'is_standing': self._is_standing(landmark_dict),
            'body_angle': self._calculate_body_angle(landmark_dict),
            'arm_angles': self._calculate_arm_angles(landmark_dict),
            'leg_angles': self._calculate_leg_angles(landmark_dict),
            'timestamp': time.time()
        }
        
        return pose_data
    
    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle between three points"""
        a = np.array([point1['x'], point1['y']])
        b = np.array([point2['x'], point2['y']])
        c = np.array([point3['x'], point3['y']])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def _is_standing(self, landmarks: Dict) -> bool:
        """Determine if person is standing based on pose"""
        # Check if hips are above knees
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        return left_hip['y'] < left_knee['y'] - 0.1
    
    def _calculate_body_angle(self, landmarks: Dict) -> float:
        """Calculate forward lean angle"""
        shoulder = landmarks[11]  # Left shoulder
        hip = landmarks[23]  # Left hip
        knee = landmarks[25]  # Left knee
        
        return self._calculate_angle(shoulder, hip, knee)
    
    def _calculate_arm_angles(self, landmarks: Dict) -> Dict[str, float]:
        """Calculate arm bend angles"""
        # Left arm
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Right arm
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        return {'left': left_angle, 'right': right_angle}
    
    def _calculate_leg_angles(self, landmarks: Dict) -> Dict[str, float]:
        """Calculate leg bend angles"""
        # Left leg
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        left_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        # Right leg
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        right_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        return {'left': left_angle, 'right': right_angle}
    
    def analyze_exercise(self, pose_data: Dict, exercise_type: str) -> Dict:
        """
        Analyze pose for specific exercise type
        
        Args:
            pose_data: Pose information from process_frame
            exercise_type: Type of exercise (squat, pushup, etc.)
            
        Returns:
            Analysis results with rep counting and form feedback
        """
        if not pose_data:
            return {'status': 'no_pose_detected'}
        
        if exercise_type.lower() in ['squat', 'squats']:
            return self._analyze_squat(pose_data)
        elif exercise_type.lower() in ['pushup', 'pushups', 'push-up', 'push-ups']:
            return self._analyze_pushup(pose_data)
        elif exercise_type.lower() in ['lunge', 'lunges']:
            return self._analyze_lunge(pose_data)
        else:
            return {'status': 'exercise_not_implemented', 'exercise': exercise_type}
    
    def _analyze_squat(self, pose_data: Dict) -> Dict:
        """Analyze squat form and count reps"""
        leg_angles = pose_data['leg_angles']
        avg_leg_angle = (leg_angles['left'] + leg_angles['right']) / 2
        
        # Rep counting logic with hysteresis to avoid double counting
        rep_detected = False
        standing_threshold = 160
        squat_threshold = 110
        
        if avg_leg_angle > standing_threshold:  # Standing
            if self.rep_stage == 'down':
                self.rep_count += 1
                rep_detected = True
                logger.info(f"Rep counted! Total: {self.rep_count}")
            self.rep_stage = 'up'
        elif avg_leg_angle < squat_threshold:  # Squat depth reached
            self.rep_stage = 'down'
        
        # Detailed form analysis
        form_feedback = []
        
        # Check depth
        if self.rep_stage == 'down':
            if avg_leg_angle < 90:
                form_feedback.append("Excellent depth!")
            elif avg_leg_angle > 120:
                form_feedback.append("Go deeper - thighs parallel to ground")
        
        # Check knee alignment
        knee_difference = abs(leg_angles['left'] - leg_angles['right'])
        if knee_difference > 20:
            form_feedback.append("Keep both knees even")
        
        # Check torso angle
        body_angle = pose_data['body_angle']
        if body_angle < 140:
            form_feedback.append("Chest up! Keep back straight")
        
        # Check if knees are tracking over toes (using hip-knee-ankle alignment)
        landmarks = pose_data['landmarks']
        left_knee_x = landmarks[25]['x']
        left_ankle_x = landmarks[27]['x']
        if abs(left_knee_x - left_ankle_x) > 0.15:
            form_feedback.append("Knees over toes")
        
        return {
            'status': 'analyzing',
            'rep_count': self.rep_count,
            'rep_detected': rep_detected,
            'stage': self.rep_stage,
            'form_feedback': form_feedback,
            'metrics': {
                'leg_angle': avg_leg_angle,
                'body_angle': body_angle,
                'knee_difference': knee_difference
            }
        }
    
    def _analyze_pushup(self, pose_data: Dict) -> Dict:
        """Analyze push-up form and count reps"""
        arm_angles = pose_data['arm_angles']
        avg_arm_angle = (arm_angles['left'] + arm_angles['right']) / 2
        
        # Rep counting logic with hysteresis
        rep_detected = False
        extended_threshold = 160
        bent_threshold = 100
        
        if avg_arm_angle > extended_threshold:  # Arms extended
            if self.rep_stage == 'down':
                self.rep_count += 1
                rep_detected = True
                logger.info(f"Push-up rep counted! Total: {self.rep_count}")
            self.rep_stage = 'up'
        elif avg_arm_angle < bent_threshold:  # Arms bent
            self.rep_stage = 'down'
        
        # Form analysis
        form_feedback = []
        
        # Check arm symmetry
        arm_difference = abs(arm_angles['left'] - arm_angles['right'])
        if arm_difference > 20:
            form_feedback.append("Keep arms even - balance your weight")
        
        # Check body alignment (plank position)
        body_angle = pose_data['body_angle']
        if body_angle < 160:
            form_feedback.append("Keep core tight - body straight like a plank")
        elif body_angle > 190:
            form_feedback.append("Don't let hips sag")
        
        # Check depth
        if self.rep_stage == 'down' and avg_arm_angle < 80:
            form_feedback.append("Great depth!")
        elif self.rep_stage == 'down' and avg_arm_angle > 110:
            form_feedback.append("Go lower - chest near ground")
        
        # Check elbow position (should be ~45 degrees from body)
        landmarks = pose_data['landmarks']
        if landmarks:
            # This is a simplified check
            left_shoulder_y = landmarks[11]['y']
            left_elbow_y = landmarks[13]['y']
            if self.rep_stage == 'down' and abs(left_shoulder_y - left_elbow_y) < 0.05:
                form_feedback.append("Elbows at 45 degrees - not flared out")
        
        return {
            'status': 'analyzing',
            'rep_count': self.rep_count,
            'rep_detected': rep_detected,
            'stage': self.rep_stage,
            'form_feedback': form_feedback,
            'metrics': {
                'arm_angle': avg_arm_angle,
                'body_angle': body_angle,
                'arm_difference': arm_difference
            }
        }
    
    def _analyze_lunge(self, pose_data: Dict) -> Dict:
        """Analyze lunge form"""
        leg_angles = pose_data['leg_angles']
        
        form_feedback = []
        if leg_angles['left'] < 90 or leg_angles['right'] < 90:
            form_feedback.append("Good depth!")
        
        return {
            'status': 'analyzing',
            'form_feedback': form_feedback,
            'metrics': {
                'left_leg_angle': leg_angles['left'],
                'right_leg_angle': leg_angles['right']
            }
        }
    
    def reset_rep_count(self, exercise_type: str = None):
        """Reset rep counter for new exercise"""
        self.rep_count = 0
        self.rep_stage = None
        self.current_exercise = exercise_type
        logger.info(f"Rep counter reset for exercise: {exercise_type}")
    
    def close(self):
        """Clean up resources"""
        self.pose.close()


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioManager:
    """Manages audio recording and playback"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        
        logger.info("AudioManager initialized")
    
    def record_audio(self, duration: int = Config.RECORD_SECONDS) -> bytes:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            WAV audio bytes
        """
        logger.info(f"Recording audio for {duration} seconds...")
        
        stream = self.audio.open(
            format=Config.AUDIO_FORMAT,
            channels=Config.CHANNELS,
            rate=Config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=Config.CHUNK_SIZE
        )
        
        frames = []
        for _ in range(0, int(Config.SAMPLE_RATE / Config.CHUNK_SIZE * duration)):
            data = stream.read(Config.CHUNK_SIZE)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Convert to WAV format
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(Config.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(Config.AUDIO_FORMAT))
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        wav_buffer.seek(0)
        logger.info("Recording complete")
        return wav_buffer.read()
    
    def play_audio(self, audio_bytes: bytes):
        """
        Play audio bytes
        
        Args:
            audio_bytes: Audio data to play
        """
        try:
            # Save temporarily and play (simple approach)
            temp_file = "temp_audio.mp3"
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            # Use system player (platform-specific)
            if sys.platform == 'darwin':  # macOS
                os.system(f"afplay {temp_file}")
            elif sys.platform == 'linux':
                os.system(f"mpg123 -q {temp_file}")
            elif sys.platform == 'win32':
                os.system(f"start {temp_file}")
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def close(self):
        """Clean up audio resources"""
        self.audio.terminate()


# ============================================================================
# OPENAI SERVICE WRAPPER
# ============================================================================

class OpenAIService:
    """Wrapper for OpenAI API services (GPT, Whisper, TTS)"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        
        # Initialize with system prompt
        self.conversation_history.append({
            "role": "system",
            "content": Config.SYSTEM_PROMPT
        })
        
        logger.info("OpenAIService initialized")
    
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_bytes: Audio data in WAV format
            
        Returns:
            Transcribed text
        """
        try:
            # Create a file-like object
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            transcript = self.client.audio.transcriptions.create(
                model=Config.WHISPER_MODEL,
                file=audio_file
            )
            
            transcribed_text = transcript.text
            logger.info(f"Transcribed: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def get_trainer_response(
        self, 
        user_message: str, 
        pose_context: Optional[Dict] = None
    ) -> str:
        """
        Get response from GPT trainer
        
        Args:
            user_message: User's message
            pose_context: Optional pose analysis data for context
            
        Returns:
            Trainer's response
        """
        try:
            # Add pose context if available
            if pose_context:
                context_msg = f"\n[Pose Context: {json.dumps(pose_context, indent=2)}]"
                user_message += context_msg
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Get response
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=self.conversation_history,
                temperature=0.8,
                max_tokens=150  # Keep responses concise
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            logger.info(f"Trainer response: {assistant_message}")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error getting trainer response: {e}")
            return "Sorry, I'm having trouble thinking right now. Let's keep going!"
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using OpenAI TTS
        
        Args:
            text: Text to convert
            
        Returns:
            Audio bytes
        """
        try:
            response = self.client.audio.speech.create(
                model=Config.TTS_MODEL,
                voice=Config.TTS_VOICE,
                input=text
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""
    
    def add_context_message(self, context: str):
        """Add system context to conversation"""
        self.conversation_history.append({
            "role": "system",
            "content": context
        })
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = [{
            "role": "system",
            "content": Config.SYSTEM_PROMPT
        }]
        logger.info("Conversation history reset")


# ============================================================================
# VIDEO DISPLAY
# ============================================================================

class VideoDisplay:
    """Manages video capture and display"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        logger.info("VideoDisplay initialized")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.current_frame = frame
            return frame
        return None
    
    def display_frame(self, frame: np.ndarray, window_name: str = "AI Personal Trainer"):
        """Display frame in window"""
        cv2.imshow(window_name, frame)
    
    def close(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PersonalTrainerApp:
    """Main application orchestrating all components"""
    
    def __init__(self, api_key: str):
        logger.info("Initializing Personal Trainer Application...")
        
        # Initialize components
        self.openai_service = OpenAIService(api_key)
        self.audio_manager = AudioManager()
        self.video_display = VideoDisplay()
        self.pose_analyzer = PoseAnalyzer()
        
        # Application state
        self.is_running = False
        self.current_exercise = None
        self.workout_session = {
            'start_time': None,
            'exercises': [],
            'total_reps': 0
        }
        
        logger.info("Application initialized successfully")
    
    def greet_user(self):
        """Initial greeting from the trainer"""
        greeting = f"""Hey there! I'm {Config.TRAINER_NAME}, your AI personal trainer! 
I'm here to help you crush your workout today. I can see you through my camera 
and track your movements to help perfect your form. 

I can guide you through exercises like squats, push-ups, lunges, and more. 
I'll count your reps, correct your form, and keep you motivated!

So, what would you like to work on today?"""
        
        logger.info("Greeting user...")
        print(f"\n🏋️ {Config.TRAINER_NAME}: {greeting}\n")
        
        # Speak greeting
        audio = self.openai_service.text_to_speech(greeting)
        self.audio_manager.play_audio(audio)
    
    def listen_to_user(self, duration: int = 5) -> str:
        """
        Listen to user input
        
        Args:
            duration: Recording duration
            
        Returns:
            Transcribed text
        """
        print("\n🎤 Listening...")
        audio_bytes = self.audio_manager.record_audio(duration)
        transcription = self.openai_service.transcribe_audio(audio_bytes)
        
        if transcription:
            print(f"👤 You: {transcription}")
        
        return transcription
    
    def speak_response(self, text: str):
        """
        Speak trainer's response
        
        Args:
            text: Text to speak
        """
        print(f"\n🏋️ {Config.TRAINER_NAME}: {text}\n")
        audio = self.openai_service.text_to_speech(text)
        self.audio_manager.play_audio(audio)
    
    def process_interaction(self, user_input: str, pose_data: Optional[Dict] = None):
        """
        Process user interaction and respond
        
        Args:
            user_input: User's speech/text input
            pose_data: Current pose analysis data
        """
        # Get trainer response with pose context
        response = self.openai_service.get_trainer_response(user_input, pose_data)
        
        # Speak response
        self.speak_response(response)
        
        # Check if user is starting an exercise
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['squat', 'push-up', 'pushup', 'lunge']):
            for exercise in ['squat', 'push-up', 'lunge']:
                if exercise in user_lower or exercise.replace('-', '') in user_lower:
                    self.current_exercise = exercise
                    self.pose_analyzer.reset_rep_count(exercise)
                    self.workout_session['exercises'].append({
                        'name': exercise,
                        'start_time': time.time(),
                        'reps': 0
                    })
                    # Start monitoring mode
                    self.start_exercise_monitoring(exercise)
                    break
    
    def start_exercise_monitoring(self, exercise: str):
        """
        Start real-time monitoring of exercise performance
        
        Args:
            exercise: Type of exercise to monitor
        """
        logger.info(f"Starting monitoring for {exercise}")
        
        # Announce start
        start_msg = f"Alright! Starting {exercise}s. I'm watching your form. Begin when ready!"
        self.speak_response(start_msg)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_exercise_performance,
            args=(exercise,),
            daemon=True
        )
        monitor_thread.start()
    
    def _monitor_exercise_performance(self, exercise: str):
        """
        Continuously monitor exercise and provide feedback
        
        Args:
            exercise: Exercise being performed
        """
        last_rep_count = 0
        last_feedback_time = time.time()
        feedback_cooldown = 4  # Seconds between verbal feedback
        no_movement_threshold = 5  # Seconds of no movement to check if stopped
        last_movement_time = time.time()
        exercise_timeout = 60  # Stop monitoring after 60 seconds of no activity
        
        logger.info(f"Monitoring {exercise} performance...")
        
        while self.current_exercise == exercise and self.is_running:
            time.sleep(0.5)  # Check twice per second
            
            # Get current frame and analyze
            frame = self.video_display.get_frame()
            if frame is None:
                continue
            
            _, pose_data = self.pose_analyzer.process_frame(frame)
            
            if not pose_data:
                # No person detected
                current_time = time.time()
                if current_time - last_movement_time > exercise_timeout:
                    logger.info("No movement detected, stopping monitoring")
                    self.current_exercise = None
                    break
                continue
            
            # Analyze exercise
            analysis = self.pose_analyzer.analyze_exercise(pose_data, exercise)
            
            if analysis.get('status') != 'analyzing':
                continue
            
            last_movement_time = time.time()
            current_time = time.time()
            
            # Check for new rep
            if analysis.get('rep_detected'):
                new_count = analysis.get('rep_count', 0)
                if new_count > last_rep_count:
                    rep_msg = f"{new_count}!"
                    print(f"\n🏋️ {Config.TRAINER_NAME}: {rep_msg}")
                    
                    # Provide encouragement every 5 reps
                    if new_count % 5 == 0:
                        encouragement = self._get_encouragement(new_count)
                        self.speak_response(encouragement)
                        last_feedback_time = current_time
                    
                    last_rep_count = new_count
            
            # Check form and provide feedback
            if current_time - last_feedback_time > feedback_cooldown:
                form_feedback = analysis.get('form_feedback', [])
                
                if form_feedback:
                    # Give form correction
                    feedback_msg = form_feedback[0]  # One correction at a time
                    print(f"\n🏋️ {Config.TRAINER_NAME}: {feedback_msg}")
                    self.speak_response(feedback_msg)
                    last_feedback_time = current_time
                
                # Check if user seems to have stopped
                elif analysis.get('stage') == 'up' and current_time - last_movement_time > no_movement_threshold:
                    check_msg = "Still going? Keep it up!"
                    print(f"\n🏋️ {Config.TRAINER_NAME}: {check_msg}")
                    self.speak_response(check_msg)
                    last_feedback_time = current_time
        
        # Exercise monitoring ended
        final_count = self.pose_analyzer.rep_count
        if final_count > 0:
            completion_msg = f"Great work! You completed {final_count} {exercise}s! How do you feel?"
            self.speak_response(completion_msg)
            
            # Update session
            if self.workout_session['exercises']:
                self.workout_session['exercises'][-1]['reps'] = final_count
                self.workout_session['total_reps'] += final_count
    
    def _get_encouragement(self, rep_count: int) -> str:
        """
        Get contextual encouragement based on rep count
        
        Args:
            rep_count: Current rep count
            
        Returns:
            Encouragement message
        """
        encouragements = {
            5: ["Nice! That's 5!", "5 down! Looking good!", "5 reps! Keep going!"],
            10: ["10! You're crushing it!", "That's 10! Strong!", "10 reps! Excellent!"],
            15: ["15! You're on fire!", "15 down! Amazing!", "That's 15! Don't stop now!"],
            20: ["20! Incredible!", "20 reps! You're a machine!", "That's 20! Outstanding!"],
        }
        
        if rep_count in encouragements:
            return np.random.choice(encouragements[rep_count])
        elif rep_count > 20 and rep_count % 10 == 0:
            return f"{rep_count}! Keep pushing!"
        else:
            return f"{rep_count}!"
    
    def run_video_loop(self):
        """Main video processing loop"""
        logger.info("Starting video loop...")
        
        last_feedback_time = time.time()
        feedback_interval = 3  # Provide feedback every 3 seconds during exercise
        
        while self.is_running:
            frame = self.video_display.get_frame()
            
            if frame is None:
                continue
            
            # Process pose
            annotated_frame, pose_data = self.pose_analyzer.process_frame(frame)
            
            # Show monitoring status
            if self.current_exercise:
                # Show "MONITORING" indicator
                cv2.rectangle(
                    annotated_frame,
                    (10, 10),
                    (250, 60),
                    (0, 200, 0),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    "MONITORING ACTIVE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Analyze exercise if active
            if self.current_exercise and pose_data:
                exercise_analysis = self.pose_analyzer.analyze_exercise(
                    pose_data, 
                    self.current_exercise
                )
                
                # Display rep count prominently
                if exercise_analysis.get('rep_count') is not None:
                    rep_count = exercise_analysis['rep_count']
                    rep_text = f"Reps: {rep_count}"
                    
                    # Large rep counter
                    cv2.rectangle(
                        annotated_frame,
                        (annotated_frame.shape[1] - 200, 10),
                        (annotated_frame.shape[1] - 10, 100),
                        (50, 50, 50),
                        -1
                    )
                    cv2.putText(
                        annotated_frame, 
                        rep_text,
                        (annotated_frame.shape[1] - 190, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
                
                # Display current stage
                stage = exercise_analysis.get('stage', '')
                if stage:
                    stage_text = f"Stage: {stage.upper()}"
                    cv2.putText(
                        annotated_frame,
                        stage_text,
                        (annotated_frame.shape[1] - 190, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                
                # Display form feedback on screen
                if exercise_analysis.get('form_feedback'):
                    y_offset = 150
                    for feedback in exercise_analysis['form_feedback'][:3]:  # Show max 3
                        # Background for text
                        text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(
                            annotated_frame,
                            (5, y_offset - 25),
                            (15 + text_size[0], y_offset + 5),
                            (0, 0, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            feedback,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                        y_offset += 40
                
                # Display metrics for debugging
                metrics = exercise_analysis.get('metrics', {})
                if metrics:
                    y_offset = annotated_frame.shape[0] - 100
                    for key, value in metrics.items():
                        metric_text = f"{key}: {value:.1f}" if isinstance(value, float) else f"{key}: {value}"
                        cv2.putText(
                            annotated_frame,
                            metric_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1
                        )
                        y_offset += 20
            else:
                # Show "Ready to monitor" when no exercise active
                if pose_data:
                    cv2.putText(
                        annotated_frame,
                        "Ready - Say 'squats' or 'push-ups' to start",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            
            # Display exercise info
            if self.current_exercise:
                exercise_text = f"Exercise: {self.current_exercise.upper()}"
                cv2.rectangle(
                    annotated_frame,
                    (5, annotated_frame.shape[0] - 45),
                    (400, annotated_frame.shape[0] - 5),
                    (50, 50, 50),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    exercise_text,
                    (10, annotated_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
            # Display frame
            self.video_display.display_frame(annotated_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
                break
            elif key == ord('s'):
                # Stop current exercise
                if self.current_exercise:
                    logger.info(f"Stopping {self.current_exercise}")
                    self.current_exercise = None
    
    def run(self):
        """Main application loop"""
        self.is_running = True
        self.workout_session['start_time'] = datetime.now()
        
        # Start video loop in separate thread
        video_thread = threading.Thread(target=self.run_video_loop, daemon=True)
        video_thread.start()
        
        # Greet user
        self.greet_user()
        
        # Give quick start instructions
        quick_start = """Quick tip: You can also press 's' anytime to manually start an exercise, 
or just say 'let's do squats' or 'time for push-ups' and I'll start watching!"""
        print(f"\n💡 Tip: {quick_start}\n")
        
        # Main interaction loop
        try:
            while self.is_running:
                # Show instructions
                print("\n" + "="*60)
                if self.current_exercise:
                    print(f"MONITORING: {self.current_exercise.upper()} - Press 's' to stop exercise")
                else:
                    print("Press ENTER to speak | 's' to start exercise manually | 'q' to quit")
                print("="*60)
                
                user_input = input("\nYour choice: ").strip()
                
                if user_input.lower() == 'q':
                    self.is_running = False
                    break
                
                # Manual exercise start
                if user_input.lower() == 's':
                    if self.current_exercise:
                        # Stop current exercise
                        self.speak_response(f"Stopping {self.current_exercise}. Great work!")
                        self.current_exercise = None
                    else:
                        # Quick menu to start exercise
                        print("\nQuick Start Menu:")
                        print("1. Squats")
                        print("2. Push-ups")
                        print("3. Lunges")
                        choice = input("Choose (1-3): ").strip()
                        
                        exercise_map = {'1': 'squat', '2': 'push-up', '3': 'lunge'}
                        if choice in exercise_map:
                            self.current_exercise = exercise_map[choice]
                            self.pose_analyzer.reset_rep_count(self.current_exercise)
                            self.start_exercise_monitoring(self.current_exercise)
                    continue
                
                # Listen to user
                speech_input = self.listen_to_user()
                
                if not speech_input:
                    print("Didn't catch that. Try again!")
                    continue
                
                # Check for quit command
                if any(word in speech_input.lower() for word in ['quit', 'exit', 'goodbye', 'bye', 'done', 'finished']):
                    farewell = "Great work today! Keep it up and I'll see you next time!"
                    self.speak_response(farewell)
                    self.is_running = False
                    break
                
                # Check if stopping current exercise
                if self.current_exercise and any(word in speech_input.lower() for word in ['stop', 'done', 'finish', 'next']):
                    completion_msg = f"Nice work on those {self.current_exercise}s! What's next?"
                    self.speak_response(completion_msg)
                    self.current_exercise = None
                    continue
                
                # Get current pose for context
                frame = self.video_display.get_frame()
                pose_data = None
                if frame is not None:
                    _, pose_data = self.pose_analyzer.process_frame(frame)
                
                # Process interaction
                self.process_interaction(speech_input, pose_data)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up...")
        self.is_running = False
        
        # Show session summary
        if self.workout_session['start_time']:
            duration = datetime.now() - self.workout_session['start_time']
            print("\n" + "="*60)
            print("WORKOUT SUMMARY")
            print("="*60)
            print(f"Duration: {duration.seconds // 60} minutes")
            print(f"Exercises completed: {len(self.workout_session['exercises'])}")
            for ex in self.workout_session['exercises']:
                print(f"  - {ex['name'].title()}")
            print(f"Total reps tracked: {self.pose_analyzer.rep_count}")
            print("="*60)
        
        # Close all components
        self.pose_analyzer.close()
        self.video_display.close()
        self.audio_manager.close()
        
        logger.info("Cleanup complete")


# ============================================================================
# GESTURE CONTROLLER (for future NAO robot adaptation)
# ============================================================================

class GestureController:
    """
    Manages gesture synchronization with speech.
    This is designed for easy adaptation to NAO robot gestures.
    
    For NAO adaptation, this will interface with the SIC framework
    to send motion commands to the robot.
    """
    
    def __init__(self):
        self.gesture_queue = queue.Queue()
        self.is_gesturing = False
        
        # Define gesture mappings for different contexts
        self.gesture_library = {
            # Greeting gestures
            'wave': {'duration': 2.0, 'description': 'Wave hand'},
            'nod': {'duration': 1.0, 'description': 'Nod head'},
            
            # Encouragement gestures
            'thumbs_up': {'duration': 1.5, 'description': 'Thumbs up'},
            'clap': {'duration': 2.0, 'description': 'Clap hands'},
            'fist_pump': {'duration': 1.5, 'description': 'Fist pump'},
            
            # Instructional gestures
            'point_forward': {'duration': 1.0, 'description': 'Point forward'},
            'point_down': {'duration': 1.0, 'description': 'Point down'},
            'arms_wide': {'duration': 2.0, 'description': 'Spread arms wide'},
            
            # Exercise demonstration gestures
            'squat_demo': {'duration': 3.0, 'description': 'Demonstrate squat motion'},
            'pushup_demo': {'duration': 3.0, 'description': 'Demonstrate push-up motion'},
            'stretch': {'duration': 2.0, 'description': 'Stretching motion'},
            
            # Neutral/idle gestures
            'idle_shift': {'duration': 1.0, 'description': 'Subtle weight shift'},
            'look_around': {'duration': 2.0, 'description': 'Look around naturally'}
        }
        
        logger.info("GestureController initialized")
    
    def select_gesture_for_speech(self, speech_text: str) -> Optional[str]:
        """
        Select appropriate gesture based on speech content
        
        Args:
            speech_text: The text being spoken
            
        Returns:
            Gesture name or None
        """
        text_lower = speech_text.lower()
        
        # Greeting detection
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'welcome']):
            return 'wave'
        
        # Encouragement detection
        if any(word in text_lower for word in ['great', 'awesome', 'excellent', 'perfect', 'good job']):
            return 'thumbs_up'
        
        if any(word in text_lower for word in ['amazing', 'fantastic', 'incredible']):
            return 'fist_pump'
        
        # Instructional detection
        if any(word in text_lower for word in ['squat', 'down']):
            return 'squat_demo'
        
        if any(word in text_lower for word in ['push-up', 'pushup', 'push up']):
            return 'pushup_demo'
        
        if any(word in text_lower for word in ['look', 'see', 'watch']):
            return 'point_forward'
        
        # Default to subtle idle gesture
        return 'idle_shift'
    
    def execute_gesture(self, gesture_name: str):
        """
        Execute a gesture (currently logs for desktop, will control NAO in robot version)
        
        Args:
            gesture_name: Name of gesture to execute
        """
        if gesture_name not in self.gesture_library:
            logger.warning(f"Unknown gesture: {gesture_name}")
            return
        
        gesture = self.gesture_library[gesture_name]
        logger.info(f"Executing gesture: {gesture_name} - {gesture['description']}")
        
        # Desktop mode: just log
        # NAO mode: This will send commands via SIC framework
        # Example NAO code (to be implemented):
        # self.nao.motion.animate(gesture_name)
        
        self.is_gesturing = True
        time.sleep(gesture['duration'])
        self.is_gesturing = False
    
    def synchronize_gesture_with_speech(self, speech_text: str, tts_callback):
        """
        Synchronize gesture with speech output
        
        Args:
            speech_text: Text being spoken
            tts_callback: Function to call for TTS
        """
        # Select appropriate gesture
        gesture = self.select_gesture_for_speech(speech_text)
        
        if gesture:
            # Start gesture in separate thread
            gesture_thread = threading.Thread(
                target=self.execute_gesture,
                args=(gesture,),
                daemon=True
            )
            gesture_thread.start()
        
        # Execute speech
        tts_callback()
    
    def add_custom_gesture(self, name: str, duration: float, description: str):
        """Add custom gesture to library"""
        self.gesture_library[name] = {
            'duration': duration,
            'description': description
        }


# ============================================================================
# NAO ROBOT ADAPTER (Template for future implementation)
# ============================================================================

class NAORobotAdapter:
    """
    Adapter class for NAO robot integration using SIC framework.
    This provides the interface for transitioning from desktop to robot mode.
    
    To use with NAO:
    1. Install SIC framework: pip install social-interaction-cloud[nao]
    2. Ensure NAO robot is connected and services are running
    3. Replace desktop components with NAO equivalents
    """
    
    def __init__(self, robot_ip: str = "127.0.0.1"):
        """
        Initialize NAO robot connection
        
        Args:
            robot_ip: IP address of NAO robot
        """
        self.robot_ip = robot_ip
        self.is_connected = False
        
        # Placeholder for SIC components (to be implemented)
        # from sic_framework.devices.nao import Nao
        # from sic_framework.core.message_python2 import AudioRequest, SpeechRecognitionRequest
        
        logger.info(f"NAORobotAdapter initialized for robot at {robot_ip}")
    
    def connect(self):
        """Connect to NAO robot"""
        # Implementation:
        # self.nao = Nao(self.robot_ip)
        # self.nao.motion.wakeUp()
        # self.is_connected = True
        pass
    
    def say(self, text: str):
        """
        Make NAO speak
        
        Args:
            text: Text to speak
        """
        # Implementation:
        # self.nao.tts.say(text)
        logger.info(f"NAO would say: {text}")
    
    def move_head(self, yaw: float, pitch: float):
        """Move NAO's head"""
        # Implementation:
        # self.nao.motion.setAngles(["HeadYaw", "HeadPitch"], [yaw, pitch], 0.2)
        pass
    
    def perform_gesture(self, gesture_name: str):
        """Perform predefined gesture"""
        # Implementation:
        # self.nao.motion.animate(gesture_name)
        pass
    
    def get_camera_frame(self):
        """Get frame from NAO's camera"""
        # Implementation:
        # return self.nao.camera.get_frame()
        pass
    
    def listen(self, duration: float = 5.0) -> str:
        """
        Listen for speech input
        
        Args:
            duration: Listening duration
            
        Returns:
            Recognized text
        """
        # Implementation:
        # result = self.nao.speech_recognition.listen(duration)
        # return result.text
        return ""
    
    def set_eye_color(self, color: str):
        """Set eye LED color for expressiveness"""
        # Implementation:
        # color_map = {'red': 0xFF0000, 'green': 0x00FF00, 'blue': 0x0000FF}
        # self.nao.leds.fadeRGB("FaceLeds", color_map[color], 1.0)
        pass
    
    def disconnect(self):
        """Disconnect from NAO robot"""
        # Implementation:
        # self.nao.motion.rest()
        # self.is_connected = False
        pass


# ============================================================================
# CONVERSATION DESIGN PATTERNS (from lectures)
# ============================================================================

class ConversationPatterns:
    """
    Implements conversation design patterns from HRI research
    Based on lecture materials on dialog design and conversation flow
    """
    
    @staticmethod
    def specifications_pattern(question: str, options: List[str]) -> Dict:
        """
        Specifications pattern: Pair closed-ended with open-ended question
        Allows structured input while enabling elaboration
        """
        return {
            'closed_question': question,
            'options': options,
            'follow_up': "Tell me more about why you chose that?"
        }
    
    @staticmethod
    def repair_pattern(misunderstanding_context: str) -> str:
        """
        Repair pattern: Handle misunderstandings gracefully
        """
        return f"I'm not sure I understood that correctly. Could you rephrase?"
    
    @staticmethod
    def confirmation_pattern(action: str) -> str:
        """
        Confirmation pattern: Confirm before executing important actions
        """
        return f"Just to confirm, you want to {action}. Is that right?"
    
    @staticmethod
    def motivation_pattern(achievement: str) -> str:
        """
        Motivation pattern: Celebrate achievements
        """
        celebrations = [
            f"Awesome! {achievement}!",
            f"You're crushing it! {achievement}!",
            f"That's what I'm talking about! {achievement}!",
            f"Fantastic work! {achievement}!"
        ]
        return np.random.choice(celebrations)
    
    @staticmethod
    def progressive_disclosure(user_level: str, info: str) -> str:
        """
        Progressive disclosure: Adjust information based on user expertise
        """
        if user_level == 'beginner':
            return f"Let me break this down: {info}"
        elif user_level == 'intermediate':
            return info
        else:  # advanced
            return f"{info} - You know the drill!"


# ============================================================================
# WORKOUT SESSION MANAGER
# ============================================================================

class WorkoutSessionManager:
    """Manages workout session state and history"""
    
    def __init__(self):
        self.session = {
            'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': None,
            'end_time': None,
            'exercises': [],
            'total_reps': 0,
            'feedback_given': [],
            'user_responses': []
        }
        self.session_history = []
    
    def start_session(self):
        """Start new workout session"""
        self.session['start_time'] = datetime.now()
        logger.info(f"Workout session started: {self.session['id']}")
    
    def add_exercise(self, exercise_name: str, reps: int = 0, duration: float = 0):
        """Add exercise to session"""
        exercise_data = {
            'name': exercise_name,
            'reps': reps,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.session['exercises'].append(exercise_data)
        self.session['total_reps'] += reps
    
    def add_feedback(self, feedback: str):
        """Log feedback given during session"""
        self.session['feedback_given'].append({
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_user_response(self, response: str):
        """Log user responses"""
        self.session['user_responses'].append({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    def end_session(self):
        """End workout session"""
        self.session['end_time'] = datetime.now()
        duration = (self.session['end_time'] - self.session['start_time']).total_seconds()
        self.session['duration_seconds'] = duration
        
        # Save to history
        self.session_history.append(self.session.copy())
        
        logger.info(f"Workout session ended: {self.session['id']}, Duration: {duration}s")
        
        return self.get_session_summary()
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.session['start_time']:
            return {}
        
        duration = datetime.now() - self.session['start_time']
        
        return {
            'duration_minutes': duration.total_seconds() / 60,
            'exercises_count': len(self.session['exercises']),
            'total_reps': self.session['total_reps'],
            'exercises': [ex['name'] for ex in self.session['exercises']]
        }
    
    def save_session_to_file(self, filepath: str = "workout_sessions.json"):
        """Save session history to file"""
        try:
            # Load existing history
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add current session if not already saved
            if self.session not in history:
                # Convert datetime objects to strings
                session_copy = self.session.copy()
                if session_copy['start_time']:
                    session_copy['start_time'] = session_copy['start_time'].isoformat()
                if session_copy['end_time']:
                    session_copy['end_time'] = session_copy['end_time'].isoformat()
                
                history.append(session_copy)
            
            # Save
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Session saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")


# ============================================================================
# ENHANCED PERSONAL TRAINER APP WITH ALL FEATURES
# ============================================================================

class EnhancedPersonalTrainerApp(PersonalTrainerApp):
    """
    Enhanced version with gesture control and session management
    """
    
    def __init__(self, api_key: str, robot_mode: bool = False, robot_ip: str = None):
        super().__init__(api_key)
        
        # Additional components
        self.gesture_controller = GestureController()
        self.session_manager = WorkoutSessionManager()
        self.conversation_patterns = ConversationPatterns()
        
        # Robot mode
        self.robot_mode = robot_mode
        self.nao_adapter = None
        
        if robot_mode and robot_ip:
            self.nao_adapter = NAORobotAdapter(robot_ip)
            self.nao_adapter.connect()
        
        logger.info(f"EnhancedPersonalTrainerApp initialized (Robot mode: {robot_mode})")
    
    def speak_response(self, text: str):
        """
        Enhanced speak response with gesture synchronization
        
        Args:
            text: Text to speak
        """
        print(f"\n🏋️ {Config.TRAINER_NAME}: {text}\n")
        
        # Select and execute gesture
        gesture = self.gesture_controller.select_gesture_for_speech(text)
        
        if gesture:
            # Start gesture in background
            gesture_thread = threading.Thread(
                target=self.gesture_controller.execute_gesture,
                args=(gesture,),
                daemon=True
            )
            gesture_thread.start()
        
        # Speak (with robot or desktop)
        if self.robot_mode and self.nao_adapter:
            self.nao_adapter.say(text)
        else:
            audio = self.openai_service.text_to_speech(text)
            self.audio_manager.play_audio(audio)
    
    def run(self):
        """Enhanced main application loop with session management"""
        self.is_running = True
        self.session_manager.start_session()
        
        # Start video loop
        video_thread = threading.Thread(target=self.run_video_loop, daemon=True)
        video_thread.start()
        
        # Greet user
        self.greet_user()
        
        # Main interaction loop
        try:
            while self.is_running:
                print("\n" + "="*60)
                print("Press ENTER to speak, 'q' to quit, 's' for session summary")
                print("="*60)
                
                user_input = input("\nYour choice: ").strip()
                
                if user_input.lower() == 'q':
                    self.is_running = False
                    break
                
                if user_input.lower() == 's':
                    summary = self.session_manager.get_session_summary()
                    print("\n📊 SESSION SUMMARY:")
                    print(f"Duration: {summary.get('duration_minutes', 0):.1f} minutes")
                    print(f"Exercises: {summary.get('exercises_count', 0)}")
                    print(f"Total reps: {summary.get('total_reps', 0)}")
                    continue
                
                # Listen to user
                speech_input = self.listen_to_user()
                
                if not speech_input:
                    print("Didn't catch that. Try again!")
                    continue
                
                # Log user response
                self.session_manager.add_user_response(speech_input)
                
                # Check for quit
                if any(word in speech_input.lower() for word in ['quit', 'exit', 'goodbye', 'bye']):
                    # End session
                    summary = self.session_manager.end_session()
                    
                    farewell = f"""Great work today! You completed {summary['exercises_count']} exercises 
and {summary['total_reps']} reps in {summary['duration_minutes']:.1f} minutes. 
Keep it up and I'll see you next time!"""
                    
                    self.speak_response(farewell)
                    self.is_running = False
                    break
                
                # Get pose context
                frame = self.video_display.get_frame()
                pose_data = None
                if frame is not None:
                    _, pose_data = self.pose_analyzer.process_frame(frame)
                
                # Process interaction
                self.process_interaction(speech_input, pose_data)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Save session
            self.session_manager.save_session_to_file()
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with robot disconnection"""
        super().cleanup()
        
        if self.robot_mode and self.nao_adapter:
            self.nao_adapter.disconnect()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Personal Trainer - Voice Interactive Fitness Coach"
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=Config.OPENAI_API_KEY,
        help='OpenAI API key (or set in Config)'
    )
    
    parser.add_argument(
        '--robot-mode',
        action='store_true',
        help='Enable NAO robot mode (requires SIC framework)'
    )
    
    parser.add_argument(
        '--robot-ip',
        type=str,
        default='127.0.0.1',
        help='NAO robot IP address (for robot mode)'
    )
    
    parser.add_argument(
        '--trainer-name',
        type=str,
        default=Config.TRAINER_NAME,
        help='Trainer name'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update config with arguments
    if args.trainer_name:
        Config.TRAINER_NAME = args.trainer_name
    
    # Print welcome message
    print("\n" + "="*60)
    print("🏋️  AI PERSONAL TRAINER  🏋️")
    print("="*60)
    print("Voice-controlled fitness coach with computer vision")
    print("Powered by OpenAI GPT-4, Whisper, and TTS")
    print("="*60)
    
    # Check API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY', '')
    
    if not api_key:
        print("\n⚠️  ERROR: OpenAI API key required!")
        print("Set it via:")
        print("  1. --api-key argument")
        print("  2. OPENAI_API_KEY environment variable")
        print("  3. Config.OPENAI_API_KEY in code")
        sys.exit(1)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    required_modules = ['cv2', 'mediapipe', 'pyaudio', 'openai', 'numpy']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - NOT FOUND")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install opencv-python mediapipe pyaudio openai numpy")
        sys.exit(1)
    
    print("\n✓ All dependencies found!")
    
    # Create and run application
    try:
        print("\n🚀 Starting application...")
        
        app = EnhancedPersonalTrainerApp(
            api_key=api_key,
            robot_mode=args.robot_mode,
            robot_ip=args.robot_ip if args.robot_mode else None
        )
        
        print("\n✓ Application initialized successfully!")
        print("\nControls:")
        print("  - Press ENTER then speak your request")
        print("  - Press 'q' to quit")
        print("  - Press 's' for session summary")
        print("  - Press 'q' in video window to close")
        print("\n" + "="*60 + "\n")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()