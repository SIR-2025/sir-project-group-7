from openai import OpenAI
from typing import Dict, Any, Optional, Callable
import time
import base64
import cv2
import io
import wave
import threading
import numpy as np
from pathlib import Path
from utils import get_settings
from .prompts import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    get_greeting_prompt as default_greeting_prompt,
    get_instruction_prompt as default_instruction_prompt,
    get_feedback_prompt as default_feedback_prompt,
    get_closing_prompt as default_closing_prompt
)

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
OPENAI_MODEL = settings.openai_model
WHISPER_MODEL = settings.whisper_model
WHISPER_LANGUAGE = settings.whisper_language
WHISPER_TEMPERATURE = settings.whisper_temperature
MAX_TOKENS = settings.max_tokens
TEMPERATURE = settings.temperature
MAX_CONVERSATION_HISTORY = settings.max_conversation_history

try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class DialogueManager:
    def __init__(
        self, 
        nao=None, 
        use_local_mic=False,
        mic_device_index=None,
        camera_manager=None,
        pose_analyzer=None,
        system_prompt=None,
        greeting_prompt_fn=None,
        instruction_prompt_fn=None,
        feedback_prompt_fn=None,
        closing_prompt_fn=None
    ):
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not found in conf/.env")
            raise ValueError("OPENAI_API_KEY must be set in conf/.env")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.nao = nao
        self.nao_mic = nao.mic if nao else None
        self.use_local_mic = use_local_mic
        self.mic_device_index = mic_device_index 
        
        self.camera_manager = camera_manager
        self.pose_analyzer = pose_analyzer
        
        if use_local_mic and not SOUNDDEVICE_AVAILABLE:
            raise ImportError("Install: pip install sounddevice soundfile")
        
        if use_local_mic:
            if mic_device_index is not None:
                print(f"Using LAPTOP microphone (device index: {mic_device_index})")
            else:
                print("Using LAPTOP microphone (default device)")
        elif self.nao_mic:
            print("Using NAO microphone")

        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.greeting_prompt_fn = greeting_prompt_fn or default_greeting_prompt
        self.instruction_prompt_fn = instruction_prompt_fn or default_instruction_prompt
        self.feedback_prompt_fn = feedback_prompt_fn or default_feedback_prompt
        self.closing_prompt_fn = closing_prompt_fn or default_closing_prompt

        self.conversation_history = []
        self.current_exercise = None
        self._initialize_conversation()

        print(f"DialogueManager initialized")
        print(f"Model: {OPENAI_MODEL}")
        print(f"Whisper: {WHISPER_MODEL}")
        if camera_manager:
            print("Camera: Enabled")
        if pose_analyzer:
            print("Pose Analysis: Enabled")

    def set_nao(self, nao):
        self.nao = nao
        self.nao_mic = nao.mic
        self.use_local_mic = False
        print("Switched to NAO microphone")

    def listen_for_any_speech(self, max_duration=5.0, silence_threshold=0.04):
        """Quick speech detection without transcription"""
        try:
            if self.use_local_mic:
                return self._detect_speech_local(max_duration, silence_threshold)
            else:
                return self._detect_speech_nao(max_duration, silence_threshold)
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False
    
    def _detect_speech_local(self, max_duration, silence_threshold):
        """Quick local speech detection without transcription"""
        try:
            print("Listening for speech...")
            
            sample_rate = 16000
            chunk_duration = 0.1
            chunk_samples = int(sample_rate * chunk_duration)
            max_chunks = int(max_duration / chunk_duration)

            device_info = sd.query_devices(self.mic_device_index, 'input')
            num_channels = device_info['max_input_channels']
            
            stream = sd.InputStream(samplerate=sample_rate, channels=num_channels, dtype='int16', device=self.mic_device_index)
            
            with stream:
                for i in range(max_chunks):
                    chunk, _ = stream.read(chunk_samples)
                    audio_level = np.sqrt(np.mean(chunk.astype(float)**2)) / 32768.0
                    
                    if audio_level > silence_threshold:
                        print("Speech detected")
                        return True
                    
                    if i % 10 == 0:
                        print(".", end="", flush=True)
            
            print("\nNo speech detected")
            return False
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def _detect_speech_nao(self, max_duration, silence_threshold):
        """Quick NAO speech detection without transcription"""
        try:
            print("Listening for speech...")
            
            speech_detected = [False]
            recording_active = threading.Event()
            recording_active.set()
            
            def on_audio_message(message):
                if not recording_active.is_set():
                    return
                
                if hasattr(message, 'waveform') and message.waveform:
                    audio_chunk = bytes(message.waveform)
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_level = np.sqrt(np.mean(audio_array.astype(float)**2)) / 32768.0
                    
                    if audio_level > silence_threshold:
                        speech_detected[0] = True
                        recording_active.clear()
            
            self.nao_mic.register_callback(on_audio_message)
            
            start_time = time.time()
            while (time.time() - start_time) < max_duration and recording_active.is_set():
                time.sleep(0.1)
                if int((time.time() - start_time) * 10) % 10 == 0:
                    print(".", end="", flush=True)
            
            recording_active.clear()
            time.sleep(0.1)
            
            if speech_detected[0]:
                print("\nSpeech detected")
                return True
            else:
                print("\nNo speech detected")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False

    def set_greeting_prompt_fn(self, prompt_fn):
        self.greeting_prompt_fn = prompt_fn

    def set_instruction_prompt_fn(self, prompt_fn):
        self.instruction_prompt_fn = prompt_fn

    def set_feedback_prompt_fn(self, prompt_fn):
        self.feedback_prompt_fn = prompt_fn

    def set_closing_prompt_fn(self, prompt_fn):
        self.closing_prompt_fn = prompt_fn

    def _initialize_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def listen_and_transcribe(self, duration=5.0):
        try:
            if self.use_local_mic:
                print(f"Recording from laptop mic for {duration}s...")
                audio_buffer = self._record_laptop_mic(duration)
            elif self.nao_mic:
                print(f"Recording from NAO mic for {duration}s...")
                audio_buffer = self._record_from_nao_mic(duration)
            else:
                print("No microphone configured")
                return None

            if not audio_buffer:
                return None

            return self._transcribe_audio_buffer(audio_buffer)

        except Exception as e:
            print(f"Listen error: {e}")
            return None

    def _record_laptop_mic(self, duration):
        try:
            print(f"Recording for {duration} seconds... Speak now")

            device_info = sd.query_devices(self.mic_device_index, 'input')
            num_channels = device_info['max_input_channels']

            audio_data = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=num_channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()

            print("Recording complete")

            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 16000, format='WAV')
            buffer.seek(0)
            buffer.name = "recording.wav"

            return buffer

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def _record_from_nao_mic(self, duration):
        try:
            print(f"Recording from NAO for {duration} seconds...")
            
            audio_chunks = []
            recording_lock = threading.Lock()
            recording_active = threading.Event()
            recording_active.set()
            
            def on_audio_message(message):
                if not recording_active.is_set():
                    return
                    
                with recording_lock:
                    if hasattr(message, 'waveform') and message.waveform:
                        audio_chunks.append(bytes(message.waveform))
            
            self.nao_mic.register_callback(on_audio_message)
            
            time.sleep(duration)
            
            recording_active.clear()
            
            time.sleep(0.1)
            
            with recording_lock:
                if not audio_chunks:
                    print("No audio chunks received from NAO microphone")
                    return None
                
                audio_bytes = b''.join(audio_chunks)
            
            print(f"Received {len(audio_chunks)} chunks, {len(audio_bytes)} bytes total")
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            
            buffer.seek(0)
            buffer.name = "nao_recording.wav"
            
            print(f"NAO recording complete - {len(audio_bytes)} bytes")
            return buffer
            
        except Exception as e:
            print(f"NAO recording error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def listen_until_silence(self, max_duration=30.0, silence_threshold=0.02, silence_duration=1.5):
        """Listen until person stops speaking"""
        try:
            if self.use_local_mic:
                audio_buffer = self._record_until_silence_local(
                    max_duration, silence_threshold, silence_duration
                )
            elif self.nao_mic:
                audio_buffer = self._record_until_silence_nao(
                    max_duration, silence_threshold, silence_duration
                )
            else:
                print("No microphone configured")
                return None
            
            if not audio_buffer:
                return None
            
            return self._transcribe_audio_buffer(audio_buffer)
            
        except Exception as e:
            print(f"Listen error: {e}")
            return None

    def _record_until_silence_local(self, max_duration, silence_threshold, silence_duration):
        """Record from laptop mic until silence detected"""
        try:
            print(f"Listening... (waiting for speech to start)")
            
            sample_rate = 16000
            chunk_duration = 0.1
            chunk_samples = int(sample_rate * chunk_duration)

            device_info = sd.query_devices(self.mic_device_index, 'input')
            num_channels = device_info['max_input_channels']
            
            all_chunks = []
            silence_chunks = 0
            silence_chunks_needed = int(silence_duration / chunk_duration)
            
            max_chunks = int(max_duration / chunk_duration)
            chunks_recorded = 0
            
            speech_started = False
            
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=num_channels,
                dtype='int16',
                device=self.mic_device_index
            )
            
            with stream:
                while chunks_recorded < max_chunks:
                    chunk, _ = stream.read(chunk_samples)
                    chunks_recorded += 1
                    
                    audio_level = np.sqrt(np.mean(chunk.astype(float)**2)) / 32768.0
                    
                    if not speech_started:
                        if audio_level > silence_threshold:
                            speech_started = True
                            print("\nSpeech detected! Now recording until silence...")
                            all_chunks.append(chunk)
                            print(".", end="", flush=True)
                        else:
                            if chunks_recorded % 10 == 0:
                                print(".", end="", flush=True)
                        continue
                    
                    all_chunks.append(chunk)
                    
                    if audio_level < silence_threshold:
                        silence_chunks += 1
                        if silence_chunks >= silence_chunks_needed:
                            print(f"\nSilence detected after {chunks_recorded * chunk_duration:.1f}s")
                            break
                    else:
                        silence_chunks = 0
                        print(".", end="", flush=True)
            
            if not speech_started:
                print("\nNo speech detected within timeout")
                return None
            
            if not all_chunks:
                print("\nNo audio recorded")
                return None
            
            audio_data = np.concatenate(all_chunks)
            
            print(f"Recording complete ({len(audio_data) / sample_rate:.1f}s)")
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            buffer.seek(0)
            buffer.name = "recording.wav"
            
            return buffer
            
        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _record_until_silence_nao(self, max_duration, silence_threshold, silence_duration):
        """Record from NAO mic until silence detected"""
        try:
            print(f"Listening... (waiting for speech to start)")
            
            audio_chunks = []
            recording_lock = threading.Lock()
            recording_active = threading.Event()
            recording_active.set()
            
            silence_time = 0
            last_audio_time = time.time()
            speech_started = False
            
            def on_audio_message(message):
                nonlocal silence_time, last_audio_time, speech_started
                
                if not recording_active.is_set():
                    return
                
                with recording_lock:
                    if hasattr(message, 'waveform') and message.waveform:
                        audio_chunk = bytes(message.waveform)
                        
                        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_level = np.sqrt(np.mean(audio_array.astype(float)**2)) / 32768.0
                        
                        if not speech_started:
                            if audio_level > silence_threshold:
                                speech_started = True
                                print("\nSpeech detected! Now recording until silence...")
                                audio_chunks.append(audio_chunk)
                                last_audio_time = time.time()
                                silence_time = 0
                                print(".", end="", flush=True)
                            return
                        
                        audio_chunks.append(audio_chunk)
                        
                        if audio_level > silence_threshold:
                            last_audio_time = time.time()
                            silence_time = 0
                            print(".", end="", flush=True)
                        else:
                            silence_time = time.time() - last_audio_time
            
            self.nao_mic.register_callback(on_audio_message)
            
            start_time = time.time()
            while (time.time() - start_time) < max_duration:
                time.sleep(0.1)
                
                with recording_lock:
                    if speech_started and silence_time >= silence_duration:
                        print(f"\nSilence detected after {time.time() - start_time:.1f}s")
                        break
            
            recording_active.clear()
            time.sleep(0.1)
            
            with recording_lock:
                if not speech_started:
                    print("\nNo speech detected within timeout")
                    return None
                
                if not audio_chunks:
                    print("\nNo audio chunks received")
                    return None
                
                audio_bytes = b''.join(audio_chunks)
            
            print(f"Recording complete ({len(audio_chunks)} chunks, {len(audio_bytes)} bytes)")
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            
            buffer.seek(0)
            buffer.name = "nao_recording.wav"
            
            return buffer
            
        except Exception as e:
            print(f"NAO recording error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def listen_and_respond_auto(self, max_duration=30.0, silence_threshold=0.02, silence_duration=1.5, scene_context=""):
        print("\nAUTO-LISTENING MODE")
        
        user_input = self.listen_until_silence(
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration
        )
        
        result = {
            'user_input': user_input,
            'response': None,
            'detected': user_input is not None
        }
        
        if user_input:
            print(f"Transcribed: '{user_input}'")
            
            if scene_context:
                response = self.get_contextual_response(user_input, scene_context)
            else:
                response = self._get_response(user_input)
            
            result['response'] = response
            print(f"Generated response")
        else:
            print("No speech detected")
        
        return result

    def get_contextual_response(self, user_input="", scene_context="", max_tokens=MAX_TOKENS):
        """
        Get a response with scene context, even without user input.
        """
        try:
            if user_input and scene_context:
                message = f"[Context: {scene_context}]\nPerson said: \"{user_input}\"\n\nRespond in character."
            elif scene_context:
                message = f"[Context: {scene_context}]\n\nGenerate dialogue in character."
            elif user_input:
                message = user_input
            else:
                return "Let's continue!"
            
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            print(f"Sending contextual request to {OPENAI_MODEL}...")
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )
            
            assistant_message = response.choices[0].message.content
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            self._trim_history()
            return assistant_message
            
        except Exception as e:
            print(f"{OPENAI_MODEL} error in contextual response: {e}")
            return "Let's continue!"

    def _transcribe_audio_buffer(self, audio_buffer):
        try:
            print("Transcribing with Whisper...")

            transcript = self.client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_buffer,
                language=WHISPER_LANGUAGE,
                temperature=WHISPER_TEMPERATURE,
                response_format="text"
            )

            transcribed_text = transcript if isinstance(transcript, str) else transcript.text
            print(f"User said: '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            print(f"Whisper error: {e}")
            return None

    def listen_and_respond(self, duration=5.0):
        user_text = self.listen_and_transcribe(duration)
        if not user_text:
            return None
        return self._get_response(user_text, max_tokens=MAX_TOKENS)

    def get_greeting(self, exercise_name):
        self.current_exercise = exercise_name
        prompt = self.greeting_prompt_fn(exercise_name)
        return self._get_response(prompt, max_tokens=30, context="greeting")

    def get_instructions(self, exercise):
        prompt = self.instruction_prompt_fn(exercise)
        return self._get_response(prompt, max_tokens=50, context="instructions")

    def get_feedback(self, pose_analysis, exercise_name, attempt_number):
        prompt = self.feedback_prompt_fn(pose_analysis, exercise_name, attempt_number)
        response = self._get_response(prompt, max_tokens=MAX_TOKENS, context="feedback")

        accuracy = pose_analysis.get('overall_accuracy', 0)
        print(f"Feedback - Attempt: {attempt_number}, Accuracy: {accuracy:.1f}%")

        return response

    def get_closing(self, exercise_name, final_accuracy, success):
        prompt = self.closing_prompt_fn(exercise_name, final_accuracy, success)
        return self._get_response(prompt, max_tokens=40, context="closing")

    def describe_image(self, custom_prompt=None):
        """Capture image and get AI description"""
        if not self.camera_manager:
            print("No camera manager configured")
            return None

        img_base64 = self.camera_manager.capture_image_base64()
        if not img_base64:
            return None

        prompt = custom_prompt or "Describe what you see in this image."

        return self._get_vision_response(img_base64, prompt)

    def analyze_exercise_form_visual(self, exercise_name):
        """Capture image and analyze exercise form using vision only"""
        if not self.camera_manager:
            print("No camera manager configured")
            return None

        img_base64 = self.camera_manager.capture_image_base64()
        if not img_base64:
            return None

        prompt = f"""You are analyzing a {exercise_name} exercise.
Look at the person's form and provide brief feedback on:
1. Body positioning
2. One key correction if needed
Keep response under 20 words."""

        return self._get_vision_response(img_base64, prompt)

    def answer_visual_question(self, question):
        """Answer a question about what the camera sees"""
        if not self.camera_manager:
            print("No camera manager configured")
            return None

        img_base64 = self.camera_manager.capture_image_base64()
        if not img_base64:
            return None

        return self._get_vision_response(img_base64, question)

    def get_pose_feedback(self, exercise_name, show_frame=False):
        """Get comprehensive feedback combining MediaPipe + GPT Vision"""
        if not self.pose_analyzer:
            print("No pose analyzer configured")
            return None

        angles, annotated_frame = self.pose_analyzer.capture_and_analyze()

        if angles is None:
            return "Could not detect pose. Please ensure you're visible to the camera."

        analysis = self.pose_analyzer.check_squat_form(angles)

        if show_frame and annotated_frame is not None:
            cv2.imshow("Your Form", annotated_frame)
            cv2.waitKey(1)

        technical_feedback = self._generate_technical_feedback(analysis)

        if self.camera_manager and annotated_frame is not None:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            issues = [j for j, d in analysis['joints'].items() if d['status'] != 'good']
            issues_str = ', '.join(issues) if issues else 'none'

            visual_feedback = self._get_vision_response(
                img_base64,
                f"""You're a fitness trainer. The user is doing a {exercise_name}.

Technical analysis shows:
- Accuracy: {analysis['overall_accuracy']:.1f}%
- Issues: {issues_str}

Looking at their form, give ONE specific, encouraging correction. Max 20 words."""
            )

            return f"{technical_feedback} {visual_feedback}"

        return technical_feedback

    def _generate_technical_feedback(self, analysis):
        """Generate technical feedback from pose analysis"""
        accuracy = analysis['overall_accuracy']

        if accuracy >= 85:
            return f"Excellent form! {accuracy:.1f}% accuracy."

        issues = []
        for joint, data in analysis['joints'].items():
            if data['status'] != 'good':
                error = data['error_degrees']
                if error > 0:
                    issues.append(f"{joint.replace('_', ' ')}: adjust {error:.0f}°")

        depth_status = analysis.get('squat_depth', {}).get('status')
        if depth_status == 'too_shallow':
            issues.append("squat deeper")

        if issues:
            return f"{accuracy:.1f}% accuracy. Fix: {', '.join(issues[:2])}."

        return f"{accuracy:.1f}% accuracy. Good form!"

    def _get_vision_response(self, img_base64, prompt, max_tokens=150):
        """Get AI response based on image"""
        try:
            print(f"Sending vision request to {OPENAI_MODEL}...")

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )

            assistant_message = response.choices[0].message.content
            print(f"Vision response received")

            return assistant_message

        except Exception as e:
            print(f"Vision API error: {e}")
            return "I couldn't analyze the image. Please try again."

    def _get_response(self, user_message, max_tokens=MAX_TOKENS, context="general"):
        try:
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            print(f"Sending {context} request to {OPENAI_MODEL}...")

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )

            assistant_message = response.choices[0].message.content

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            self._trim_history()
            return assistant_message

        except Exception as e:
            print(f"{OPENAI_MODEL} error in {context}: {e}")
            return self._get_fallback_response(context)

    def _get_fallback_response(self, context):
        fallbacks = {
            "greeting": "Hello! Let's begin your training session.",
            "instructions": "Watch carefully and follow my movements.",
            "feedback": "Keep going, you're doing great!",
            "closing": "Well done! That completes your session."
        }
        return fallbacks.get(context, "Let's continue!")

    def _trim_history(self):
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
            self.conversation_history = [
                                            self.conversation_history[0]
                                        ] + self.conversation_history[-(MAX_CONVERSATION_HISTORY - 1):]

    def reset_conversation(self):
        self._initialize_conversation()
        self.current_exercise = None
        print("Conversation reset")

    def cleanup(self):
        print("DialogueManager resources released")