import time
from cursed.cursed import EffectPipeline, Backwards, AmpSetting, Distortion, Compression
from cursed.recorder import Recorder
import numpy as np
import sounddevice as sd

sr = 48000
pipeline = EffectPipeline(sr = sr, effects=[
    # Backwards(),
    Compression(threshold=0.1, ratio = 4, makeup_gain=2, attack=0.01, release=0.1, sr=sr),
    Distortion(amp_setting=AmpSetting(gain_knob=7)),
])


rec = Recorder(samplerate=sr, channels=1)
print("start recording in 3 secs!")
time.sleep(3)
print("recording starts")
audio = rec.record()

print(audio.shape)



# audio_final = pipeline.process(audio)

# sd.play(audio_final, samplerate=sr)
# sd.wait()
