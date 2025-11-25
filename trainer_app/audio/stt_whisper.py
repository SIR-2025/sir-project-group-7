import tempfile
import numpy as np
import wave
from openai import OpenAI
from config import OPENAI_API_KEY

class WhisperSTT:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def transcribe(self, audio_np):
        if len(audio_np) == 0:
            return ""

        # Save audio to proper WAV file (16-bit PCM, mono, 16 kHz)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)      # 16-bit PCM
                wf.setframerate(16000)
                wf.writeframes(audio_np.astype(np.int16).tobytes())

            # Correct OpenAI Whisper endpoint
            result = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp.name, "rb")
            )

        return result.text
