# -*- coding: utf-8 -*-
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.services.openai_whisper_stt.whisper_stt import (
    SICWhisper, GetTranscript, WhisperConf
)
from sic_framework.services.openai_gpt.gpt import GPT, GPTConf, GPTRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest

import time, os, random

class NaoTrainerLLM(SICApplication):
    """
    NAO robot trainer using:
      - NAO mic (speech input)
      - Whisper STT (speech-to-text)
      - OpenAI GPT (intent understanding + response generation)
      - NAO TTS + motion (spoken + physical feedback)
    """

    def __init__(self):
        super().__init__()
        self.set_log_level(sic_logging.INFO)
        self.nao_ip = os.getenv("NAO_IP", "10.0.0.241")
        self.language = "en"
        self.nao = None
        self.whisper = None
        self.gpt = None
        self.running = True
        self.setup()

    def setup(self):
        """Initialize NAO, Whisper STT, and GPT."""
        self.logger.info("Initializing NAO Trainer LLM...")
        self.nao = Nao(ip=self.nao_ip)

        # Connect to NAO microphone
        self.mic = self.nao.mic
        self.logger.info("NAO microphone connected.")

        # Whisper Speech-to-Text
        openai_key = os.getenv("")
        w_conf = WhisperConf(openai_key=openai_key)
        self.whisper = SICWhisper(input_source=self.mic, conf=w_conf)

        # GPT for conversational reasoning
        g_conf = GPTConf(openai_key=openai_key)
        self.gpt = GPT(conf=g_conf)

        self.logger.info("Services initialized.")

    # --- Helpers ---
    def speak(self, text, animated=True):
        """Make NAO speak with optional gesture animation."""
        self.logger.info(f"NAO says: {text}")
        self.nao.tts.request(NaoqiTextToSpeechRequest(text, animated=animated))

    def gesture(self, name):
        try:
            self.nao.motion.request(NaoqiAnimationRequest(name))
        except Exception as e:
            self.logger.warning(f"Gesture failed: {e}")

    # --- Main trainer logic ---
    def listen_once(self, prompt=None):
        """Prompt user and return transcript using Whisper STT."""
        if prompt:
            self.speak(prompt)
        transcript = self.whisper.request(GetTranscript(timeout=10, phrase_time_limit=6))
        text = transcript.transcript.strip()
        if text:
            self.logger.info(f"[User said]: {text}")
        else:
            self.logger.info("No speech detected.")
        return text

    def get_response(self, user_input):
        """Ask GPT to act as a motivational trainer."""
        system_prompt = (
            "You are NAO, a friendly personal trainer robot. "
            "Respond concisely and enthusiastically in under 15 words. "
            "Encourage the user or guide them through exercises."
        )
        user_prompt = f"The user said: '{user_input}'"
        reply = self.gpt.request(GPTRequest(system_prompt=system_prompt, user_prompt=user_prompt))
        return reply.response.strip()

    # --- Flow ---
    def run(self):
        self.logger.info("Starting NAO Trainer LLM...")
        try:
            self.nao.autonomous.request(NaoWakeUpRequest())
            self.speak("Hello! I’m your personal trainer NAO. Say start when ready.", animated=True)

            # Continuous loop; only stop if user says stop or we get KeyboardInterrupt
            while not self.shutdown_event.is_set():
                # Keep Whisper listening continuously
                self.logger.info("[Loop] Waiting for user speech...")
                user_text = ""
                try:
                    reply = self.whisper.request(GetTranscript(timeout=15, phrase_time_limit=6))
                    user_text = (reply.transcript or "").strip()
                except Exception as e:
                    self.logger.error(f"[ASR error] {e}")
                    continue

                if not user_text:
                    continue

                self.logger.info(f"[User said]: {user_text}")

                # Exit on explicit stop words only
                if any(w in user_text.lower() for w in ("stop", "exit", "bye", "finish")):
                    self.speak("Okay, ending session. Great work today!")
                    break

                # Get GPT response with guard
                try:
                    resp = self.gpt.request(GPTRequest(
                        system_prompt=(
                            "You are NAO, a concise, upbeat personal trainer. "
                            "Reply in <= 15 words, be specific and motivational."
                        ),
                        user_prompt=f"User said: '{user_text}'. Respond as the trainer."
                    ))
                    bot = (resp.response or "").strip()
                    self.logger.info(f"[GPT reply]: {bot}")
                except Exception as e:
                    self.logger.error(f"[GPT error] {e}")
                    bot = "Let’s keep going. Ready for the next set?"

                # Speak reply; never crash on TTS
                try:
                    self.speak(bot, animated=True)
                except Exception as e:
                    self.logger.error(f"[TTS error] {e}")

            self.nao.autonomous.request(NaoRestRequest())

        except KeyboardInterrupt:
            self.speak("Stopping training session. Goodbye!")
            self.nao.autonomous.request(NaoRestRequest())
        finally:
            self.shutdown()



if __name__ == "__main__":
    app = NaoTrainerLLM()
    app.run()
