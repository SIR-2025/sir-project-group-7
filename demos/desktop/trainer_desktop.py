import asyncio
import base64
import json
import pyaudio
import websockets
import ssl
import threading
from collections import deque


OPENAI_API_KEY = ""

# FULLY VALID REALTIME MODEL 
REALTIME_URL = (
    "wss://api.openai.com/v1/realtime?"
    "model=gpt-4o-mini-realtime-preview-2024-12-17"
)

# Audio config (24k mono PCM for realtime API)
RATE = 24000
CHUNK = 4096



# Non-blocking TTS Playback
class AudioPlayer:
    def __init__(self):
        self.audio_queue = deque()
        self.playing = False
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
    def add_chunk(self, audio_bytes):
        self.audio_queue.append(audio_bytes)
        if not self.playing:
            threading.Thread(target=self._play_queue, daemon=True).start()
    
    def _play_queue(self):
        self.playing = True
        if self.stream is None:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                output=True
            )
        
        while self.audio_queue:
            chunk = self.audio_queue.popleft()
            self.stream.write(chunk)
        
        self.playing = False
    
    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()



# Trainer State Machine
class TrainerState:
    """
    Simple state machine controlling trainer behavior.
    States:
        IDLE → WARMUP → WORKOUT → COOLDOWN
    """

    def __init__(self):
        self.state = "IDLE"
        self.exercise_counter = 0

    def next_state(self):
        if self.state == "IDLE":
            self.state = "WARMUP"
        elif self.state == "WARMUP":
            self.state = "WORKOUT"
        elif self.state == "WORKOUT":
            self.state = "COOLDOWN"
        elif self.state == "COOLDOWN":
            self.state = "IDLE"
        return self.state

    def describe(self):
        if self.state == "IDLE":
            return "waiting for the user to start"
        if self.state == "WARMUP":
            return "guiding light warm-up movements"
        if self.state == "WORKOUT":
            return "leading the main workout exercises"
        if self.state == "COOLDOWN":
            return "helping the user cool down and stretch"



# Realtime Trainer Class
class RealtimeTrainer:
    def __init__(self):
        self.ws = None
        self.state = TrainerState()
        self.active = True
        self.audio_player = AudioPlayer()
        self.current_transcript = ""
        self.assistant_response = ""
        self.user_speaking = False
        self.last_speech_end = None

    # REALTIME API
    async def connect(self):
        print("Connecting to OpenAI Realtime API...")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        ssl_context = ssl.create_default_context()
        self.ws = await websockets.connect(
            REALTIME_URL,
            extra_headers=headers,
            ssl=ssl_context
        )

        print("Connected to OpenAI Realtime API!")

        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are NAO, a motivating personal trainer. "
                    "You guide warm-ups, exercises, and cooldowns. "
                    "Provide encouragement and keep a friendly, energetic tone. "
                    "Always respond verbally to the user's speech. "
                    "Keep responses SHORT - 1-2 sentences maximum. "
                    f"Current workout phase: {self.state.describe()}. "
                    "Respond to what you hear and provide appropriate coaching. "
                    "If the user says 'start', begin the workout. "
                    "If they say 'next', move to the next phase. "
                    "Be proactive and guide the user through the workout."
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {  
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.8
            }
        }))
        
        print("System ready! The trainer will respond to your voice.")
        print("Say 'start' to begin the workout!")
        print(f"Current state: {self.state.state}\n")


    # Microphone Audio Streaming to OpenAI
    async def stream_microphone(self):
        pa = pyaudio.PyAudio()
        mic = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("Microphone active - start speaking!\n")

        try:
            while self.active:
                data = mic.read(CHUNK, exception_on_overflow=False)
                audio_b64 = base64.b64encode(data).decode()

                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

                await asyncio.sleep(0.01)
        finally:
            mic.stop_stream()
            mic.close()
            pa.terminate()


    # Trigger a response from the assistant
    async def trigger_response(self):
        """Explicitly trigger the assistant to respond"""
        print("Triggering assistant response...")
        await self.ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": f"Current workout phase: {self.state.describe()}. Provide appropriate guidance."
            }
        }))

    # Handle state transitions based on user input
    async def process_user_input(self, text):
        """Process user speech and update state accordingly"""
        text_lower = text.lower()
        
        # State transitions
        if "start" in text_lower and self.state.state == "IDLE":
            self.state.next_state()
            print(f" State changed to: {self.state.state}")
            await self.update_instructions()
            await self.trigger_response()
            
        elif "next" in text_lower:
            new_state = self.state.next_state()
            print(f" State changed to: {new_state}")
            await self.update_instructions()
            
            # Only trigger response if not going back to IDLE
            if new_state != "IDLE":
                await self.trigger_response()
            else:
                print(" Workout complete! Ready for next session.")
                

        elif self.state.state != "IDLE":
            # Auto-trigger response during active workout
            await asyncio.sleep(1.0)  # Brief pause
            await self.trigger_response()


    # Handle Realtime Responses
    async def handle_messages(self):
        while self.active:
            try:
                message = await self.ws.recv()
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

            msg = json.loads(message)
            msg_type = msg.get("type")
            
            
            if msg_type not in ["input_audio_buffer.append", "response.audio.delta"]:
                print(f"[EVENT] {msg_type}")

            # Track when user starts speaking
            if msg_type == "input_audio_buffer.speech_started":
                print("User started speaking...")
                self.user_speaking = True
                self.current_transcript = ""

            # process what they said
            elif msg_type == "input_audio_buffer.speech_stopped":
                print("User stopped speaking")
                self.user_speaking = False
                self.last_speech_end = asyncio.get_event_loop().time()

            # user transcript
            elif msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = msg.get("transcript", "")
                if transcript:
                    self.current_transcript = transcript
                    print(f"User: {transcript}")
                    # Process user input for state management
                    await self.process_user_input(transcript)

            # assistant's text response
            elif msg_type == "response.text.delta":
                delta = msg.get("delta", "")
                self.assistant_response += delta
                print(delta, end="", flush=True)

            elif msg_type == "response.text.done":
                if self.assistant_response:
                    print(f"\n NAO: {self.assistant_response}\n")
                    self.assistant_response = ""

            
            elif msg_type == "response.audio.delta":
                audio_chunk = base64.b64decode(msg["delta"])
                self.audio_player.add_chunk(audio_chunk)

            
            elif msg_type == "response.created":
                print("💭 NAO is preparing response...")

            elif msg_type == "response.done":
                print(" Response complete\n")

            # Error handling
            elif msg_type == "error":
                error = msg.get("error", {})
                print(f" Error: {error}")


    # Update session instructions based on state
    async def update_instructions(self):
        """Update the system instructions based on current state"""
        instructions = (
            "You are NAO, a motivating personal trainer. "
            "Provide encouragement and keep a friendly, energetic tone. "
            "Always respond verbally to the user's speech. "
            "Keep responses SHORT - 1-2 sentences maximum. "
        )
        
        if self.state.state == "IDLE":
            instructions += "The user hasn't started yet. Encourage them to say 'start' when ready!"
        elif self.state.state == "WARMUP":
            instructions += "Guide warm-up exercises like arm circles, leg swings, and light stretches. Demonstrate proper form."
        elif self.state.state == "WORKOUT":
            instructions += "Lead intense workout exercises like squats, push-ups, and burpees. Count reps and provide motivation!"
        elif self.state.state == "COOLDOWN":
            instructions += "Guide cooldown stretches and breathing exercises. Be calming and focus on recovery."
        
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": instructions
            }
        }))
        print(f"Updated instructions for {self.state.state} phase")


    # Run trainer system
    async def run(self):
        try:
            await asyncio.gather(
                self.stream_microphone(),
                self.handle_messages()
            )
        finally:
            self.audio_player.close()



# MAIN ENTRY POINT
async def main():
    if not OPENAI_API_KEY:
        print(" Error: Please add your OpenAI API key to the OPENAI_API_KEY variable")
        return
    
    trainer = RealtimeTrainer()
    await trainer.connect()
    await trainer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Shutting down trainer...")