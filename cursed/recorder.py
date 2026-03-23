import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

class Recorder:
    def __init__(self, samplerate=44100, channels=1):
        self.audio_blocks: list = []
        self.samplerate = samplerate
        self.channels = channels

    def _callback(self, indata: NDArray, frames, time, status):
        if status:
            print(status)
        self.audio_blocks.append(indata.copy())

    def record(self) -> NDArray:
        with sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, callback=self._callback
        ):
            print("recording... press enter to stop")
            input()
        final = np.concatenate(self.audio_blocks, axis=0)
        return final