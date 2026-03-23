# from pedalboard import Pedalboard, Distortion, Compressor, Delay
# from pedalboard.io import AudioFile
import numpy as np
from numpy.typing import NDArray

from typing import Protocol, Callable
from pydantic import BaseModel, Field

from numba import njit


# # board = Pedalboard(
# #     [
# #         Delay(delay_seconds=0.5, feedback=1, mix=0.3)
# #     ]
# # )
# # audio_final = board.process(audio_original, sr)


# audio_final = np.clip(audio_original*100, -1.0, 1.0)

# audio_final = np.flip(audio_final)

# sd.play(audio_final, samplerate=sr)
# sd.wait()


class _Effect(Protocol):
    def apply_effect(self, audio: NDArray) -> NDArray:
        """
        A protocol for the effects.
        they should all implement the apply_effect method which is a map from ndarray to ndarray
        """
        ...

class EffectPipeline:
    def __init__(self, sr=44100, effects: list[_Effect] = []):
        self.samplerate = sr
        self.effects = effects

    def process(self, audio: NDArray) -> NDArray:
        for effect in self.effects:
            audio = effect.apply_effect(audio)
        return audio


class Backwards:
    """
    Reverts the audio
    """

    def apply_effect(self, audio: NDArray) -> NDArray:
        """
        reverts the audio along axis 0 which is time
        """
        return np.flip(audio, axis=0)


class AmpSetting(BaseModel):
    """
    Amp settings.

    input: gain_knob. a float number that controls the amp level.

    the gain knob is a number between 0.0 and 10.0 and defaults to 5.0.

    but what if you push it beyond?

    >>> amp_setting = AmpSetting(gain_knob=7)
    >>> amp_setting.gain_factor
    25.118864315095795
    """

    gain_knob: float = Field(default=5.0, ge=0.0, le=11.0)

    @property
    def gain_factor(self) -> float:
        """
        returns the gain factor of the class.

        it will calculate the amped db and return the amplitude of the amp

        >>> amp_setting = AmpSetting(gain_knob=7)
        >>> amp_setting.gain_factor
        25.118864315095795
        """
        db = (self.gain_knob / 10) * 40  # between 0 and +40dB amp
        return 10 ** (db / 20)


class Distortion:
    def __init__(
        self,
        amp_setting: AmpSetting,
        clip: Callable[[NDArray], NDArray] = lambda x: np.tanh(x),
    ):
        """
        initiates the class

        inputs:

        - amp_setting: an amp setting of course
        - clip: a function for clipping the wave. it should input an ndarray and output an ndarray. \
            the default clip function is tanh for soft clipping. if you want, try np.clip(x, -1.0, 1.0)
        
        """
        self.amp = amp_setting.gain_factor
        self.clip = clip

    def apply_effect(self, audio: NDArray) -> NDArray:
        return self.clip(audio * self.amp)


@njit
def _compressor_mono(
    audio: NDArray,
    threshold: float,
    ratio: float,
    makeup_gain: float,
    attack_coeff: float,
    release_coeff: float,
) -> NDArray:
    n = len(audio)
    # envelope
    env = 0.0
    # gain
    gain = 1.0
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = audio[i]
        x_abs = np.abs(x)

        if x_abs > env:
            env = attack_coeff * env + (1 - attack_coeff) * x_abs
        else:
            env = release_coeff * env + (1 - release_coeff) * x_abs
        # envelope is for getting an idea of how loud.
        # it is often rounded up

        if env > threshold:
            excess = env - threshold
            compressed = threshold + excess / ratio
            target_gain = compressed / (env + 1e-6)
        else:
            target_gain = 1.0

        if target_gain < gain:
            gain = attack_coeff * gain + (1 - attack_coeff) * target_gain
        else:
            gain = release_coeff * gain + (1 - release_coeff) * target_gain

        out[i] = x * gain * makeup_gain
    return out

@njit
def _compressor_linked(
    audio: NDArray,
    threshold: float,
    ratio: float,
    makeup_gain: float,
    attack_coeff: float,
    release_coeff: float,
) -> NDArray:
    n, c = audio.shape
    # envelope
    env = 0.0
    # gain
    gain = 1.0
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = audio[i]
        x_abs = np.abs(x)

        if x_abs > env:
            env = attack_coeff * env + (1 - attack_coeff) * x_abs
        else:
            env = release_coeff * env + (1 - release_coeff) * x_abs
        # envelope is for getting an idea of how loud.
        # it is often rounded up

        if env > threshold:
            excess = env - threshold
            compressed = threshold + excess / ratio
            target_gain = compressed / (env + 1e-6)
        else:
            target_gain = 1.0

        if target_gain < gain:
            gain = attack_coeff * gain + (1 - attack_coeff) * target_gain
        else:
            gain = release_coeff * gain + (1 - release_coeff) * target_gain

        out[i] = x * gain * makeup_gain
    return out


def _compressor_raw(
    audio: NDArray,
    threshold: float,
    ratio: float,
    makeup_gain: float,
    tol: float = 1e-6,
):
    # without envelope it reacts in zero secs
    abs_audio = np.abs(audio)

    compressed = np.where(
        abs_audio > threshold, threshold + (abs_audio - threshold) / ratio, abs_audio
    )

    gain = compressed / (abs_audio + tol)

    out = audio * gain * makeup_gain
    return out


class Compression:
    """
    A compression pedal.
    makes quiet sounds louder and loud sounds quieter
    """

    def __init__(
        self,
        threshold=0.1,
        ratio=4.0,
        makeup_gain=2.0,
        attack=0.01,
        release=0.1,
        sr=44100,
    ):
        self.attack_coeff: float = np.exp(-1 / (attack * sr))
        self.release_coeff: float = np.exp(-1 / (release * sr))
        self.threshold = threshold
        self.ratio = ratio
        self.makeup_gain = makeup_gain

    def apply_effect(self, audio: NDArray) -> NDArray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            return _compressor_mono(
                audio,
                self.threshold,
                self.ratio,
                self.makeup_gain,
                self.attack_coeff,
                self.release_coeff,
            )
        elif audio.ndim >= 2:
            n, c = audio.shape
            out = np.zeros_like(audio)
            for ch in range(c):
                out[:, ch] = _compressor_mono(audio[:, ch], self.threshold,
                self.ratio,
                self.makeup_gain,
                self.attack_coeff,
                self.release_coeff,
            )
            return out
        else:
            raise ValueError("Not enough channels")
