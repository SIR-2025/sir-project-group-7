import pyaudio
import numpy as np

class MicrophoneStream:
    def __init__(self, rate=16000, chunk=2048):
        self.rate = rate
        self.chunk = chunk
        self.buffer = []

        self.audio_interface = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        # Open the stream here, not in __init__
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._callback
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.buffer.append(audio_data)
        return (in_data, pyaudio.paContinue)

    def has_speech(self):
        return len(self.buffer) > 0

    def get_audio(self):
        if not self.buffer:
            return np.zeros(0, dtype=np.int16)
        audio = np.concatenate(self.buffer)
        self.buffer.clear()
        return audio
