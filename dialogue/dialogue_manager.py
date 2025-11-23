from openai import OpenAI
from typing import Dict, Any, Optional, Callable
import time
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
        
        if use_local_mic and not SOUNDDEVICE_AVAILABLE:
            raise ImportError("Install: pip install sounddevice soundfile")
        
        if use_local_mic:
            print("Using LAPTOP microphone")
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
    
    def set_nao(self, nao):
        self.nao = nao
        self.nao_mic = nao.mic
        self.use_local_mic = False
        print("Switched to NAO microphone")
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self._initialize_conversation()
        print("System prompt updated")
    
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
            
            audio_data = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            
            print("Recording complete")
            
            import io
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
            import wave
            import io
            
            print(f"Recording from NAO for {duration} seconds...")
            
            audio_chunks = []
            num_samples = int(16000 * duration)
            samples_recorded = 0
            
            while samples_recorded < num_samples:
                chunk = self.nao_mic.read()
                if chunk:
                    audio_chunks.append(chunk)
                    samples_recorded += len(chunk) // 2
            
            audio_data = b''.join(audio_chunks)
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            
            buffer.seek(0)
            buffer.name = "nao_recording.wav"
            
            print("NAO recording complete")
            return buffer
            
        except Exception as e:
            print(f"NAO recording error: {e}")
            return None
    
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