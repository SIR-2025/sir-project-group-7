import pyaudio
from openai import OpenAI
from config import OPENAI_API_KEY, TTS_VOICE

class OpenAITTS:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio = pyaudio.PyAudio()

    def speak(self, text: str):
        if not text:
            return

        # Request raw 16-bit PCM at 24 kHz from OpenAI TTS
        resp = self.client.audio.speech.create(
            model="gpt-4o-mini-tts",   # you can change if you want
            voice=TTS_VOICE,
            input=text,
            format="pcm",              # <<< IMPORTANT: raw PCM, not mp3
        )

        audio_bytes = resp.read()      # raw 16-bit PCM mono, 24000 Hz

        stream = self.audio.open(
            format=pyaudio.paInt16,    # 16-bit signed
            channels=1,
            rate=24000,
            output=True,
        )

        stream.write(audio_bytes)
        stream.stop_stream()
        stream.close()
