from cursed.cursed import EffectPipeline, Backwards, AmpSetting, Distortion, Recorder
import numpy as np
pipeline = EffectPipeline(effects=[
    Backwards(),
    Distortion(amp_setting=AmpSetting(gain_knob=5), clip=lambda x: np.clip(x, -1.0, 1.0))
])

audio

