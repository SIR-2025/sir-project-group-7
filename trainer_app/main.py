import asyncio
import numpy as np

from audio.microphone import MicrophoneStream
from audio.stt_whisper import WhisperSTT
from audio.tts_openai import OpenAITTS
from vision.camera import Camera
from llm.trainer_agent import TrainerAgent
from core.session_manager import SessionManager
from core.state_machine import TrainerStateMachine
from ui.desktop_window import DesktopWindow


async def main():
    # Initialize components
    camera = Camera()
    mic = MicrophoneStream()
    stt = WhisperSTT()
    tts = OpenAITTS()
    agent = TrainerAgent()
    sm = TrainerStateMachine()
    ui = DesktopWindow(camera)

    session = SessionManager(sm, agent)

    await ui.start()

    print("Starting trainer...")

    # important: start microphone stream
    mic.start()

    # Main loop
    while True:
        frame = camera.get_frame()
        ui.update_frame(frame)

        # Check for microphone audio
        if mic.has_speech():
            audio = mic.get_audio()

            # Skip very short chunks (< 0.7 seconds)
            if len(audio) < int(0.7 * 16000):
                continue

            # Transcribe to text
            text = stt.transcribe(audio)

            # Ignore silence / empty text
            if not text or not text.strip():
                continue

            print("USER SAID:", text)

            # Generate trainer reply
            reply, actions = await session.process(text)

            print("TRAINER:", reply)

            # Speak out loud
            tts.speak(reply)

            # If workout needs pose detection
            if "vision_required" in actions:
                pose = camera.detect_pose()
                session.update_pose(pose)

            # Show UI feedback
            if "motivate" in actions:
                ui.flash_green()

        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
