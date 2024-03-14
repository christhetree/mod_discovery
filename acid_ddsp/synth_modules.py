import logging
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict

import torch as tr
from torch import Tensor as T, nn

from torchsynth.config import SynthConfig
from torchsynth.module import SquareSawVCO, LFO, ADSR
from torchsynth.signal import Signal

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


@dataclass
class ADSRValues:
    attack: float
    decay: float
    sustain: float
    release: float
    alpha: float


class CustomADSR(ADSR):
    def __init__(self,
                 synthconfig: SynthConfig,
                 min_adsr_vals: ADSRValues,
                 max_adsr_vals: ADSRValues,
                 **kwargs: Dict[str, T]) -> None:
        for dr in self.default_parameter_ranges:
            if dr.name == "attack":
                dr.minimum = min_adsr_vals.attack
                dr.maximum = max_adsr_vals.attack
            if dr.name == "decay":
                dr.minimum = min_adsr_vals.decay
                dr.maximum = max_adsr_vals.decay
            if dr.name == "sustain":
                dr.minimum = min_adsr_vals.sustain
                dr.maximum = max_adsr_vals.sustain
            if dr.name == "release":
                dr.minimum = min_adsr_vals.release
                dr.maximum = max_adsr_vals.release
            if dr.name == "alpha":
                dr.minimum = min_adsr_vals.alpha
                dr.maximum = max_adsr_vals.alpha
        super().__init__(synthconfig, **kwargs)


class CustomLFO(LFO):
    def __init__(self,
                 synthconfig: SynthConfig,
                 lfo_type: Optional[str] = None,
                 min_freq: float = 0.5,
                 max_freq: float = 12.0,
                 **kwargs: Dict[str, T]) -> None:
        for dr in self.default_ranges:
            if dr.name == "frequency":
                dr.minimum = min_freq
                dr.maximum = max_freq
                dr.curve = 0.25
        super().__init__(synthconfig, **kwargs)
        if lfo_type is None:
            lfo_type = random.choice(self.lfo_types)
        else:
            assert lfo_type in self.lfo_types
        self.lfo_type = lfo_type
        for curr_lfo_type in self.lfo_types:
            if curr_lfo_type == self.lfo_type:
                self.set_parameter(curr_lfo_type, tr.ones(self.batch_size))
            else:
                self.set_parameter(curr_lfo_type, tr.zeros(self.batch_size))


class SquareSawVCOLite(nn.Module):
    # Based off TorchSynth's SquareSawVCO
    def __init__(self, sr: float):
        super().__init__()
        self.sr = sr

    def calc_n_partials(self, freq_hz: T) -> T:
        assert freq_hz.ndim == 2
        max_f0_hz = tr.max(freq_hz, dim=1, keepdim=True).values
        # TODO(cm): check this calculation
        n_partials = 12000 / (max_f0_hz * tr.log10(max_f0_hz))
        return n_partials

    def forward(self, f0_hz: T, osc_shape: T, n_samples: Optional[int] = None) -> T:
        assert 1 <= f0_hz.ndim <= 2
        assert 1 <= osc_shape.ndim <= 2
        bs = f0_hz.size(0)

        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)
        if osc_shape.ndim == 1:
            assert n_samples is not None
            osc_shape = osc_shape.unsqueeze(1)
            osc_shape = osc_shape.expand(-1, n_samples)

        phase = (tr.rand((bs, 1)) * 2 * tr.pi) - tr.pi
        arg = tr.cumsum(2 * tr.pi * f0_hz / self.sr, dim=1)
        arg += phase

        # TODO(cm): check how this works
        n_partials = self.calc_n_partials(f0_hz)
        square_wave = tr.tanh(tr.pi * n_partials * tr.sin(arg) / 2)
        out_wave = (1 - (osc_shape / 2)) * square_wave * (1 + (osc_shape * tr.cos(arg)))
        return out_wave


class CustomSquareSawVCO(SquareSawVCO):
    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[tr.device] = None,
        **kwargs: Dict[str, T],
    ):
        super().__init__(synthconfig, device, **kwargs)
        self.set_parameter("tuning", tr.zeros((self.batch_size,)))
        self.set_parameter("mod_depth", tr.zeros((self.batch_size,)))

    def oscillator(self, argument: Signal, midi_f0: T, shape: T) -> Signal:
        partials = self.partials_constant(midi_f0).unsqueeze(1)
        square = tr.tanh(tr.pi * partials * tr.sin(argument) / 2)
        return (1 - shape / 2) * square * (1 + shape * tr.cos(argument))

    def output(self, midi_f0: T, shape_mod_signal: Signal, pitch_mod_signal: Optional[Signal] = None) -> Signal:
        assert midi_f0.shape == (self.batch_size,)
        assert shape_mod_signal.shape == (self.batch_size, self.buffer_size)

        if pitch_mod_signal is not None and pitch_mod_signal.shape != (
            self.batch_size,
            self.buffer_size,
        ):
            raise ValueError(
                "mod_signal has incorrect shape. Expected "
                f"{tr.Size([self.batch_size, self.buffer_size])}, "
                f"and received {pitch_mod_signal.shape}. Make sure the mod_signal "
                "being passed in is at full audio sampling rate."
            )

        control_as_frequency = self.make_control_as_frequency(midi_f0, pitch_mod_signal)

        if self.synthconfig.debug:
            assert (control_as_frequency >= 0).all() and (
                control_as_frequency <= self.nyquist
            ).all()

        cosine_argument = self.make_argument(control_as_frequency)
        cosine_argument += self.p("initial_phase").unsqueeze(1)
        output = self.oscillator(cosine_argument, midi_f0, shape_mod_signal)
        return output.as_subclass(Signal)


if __name__ == "__main__":
    import torchaudio
    freq = tr.tensor([220.0, 220.0])
    osc_shape = tr.tensor([1.0, 0.5])

    sr = 48000
    n_samples = 48000
    vco = SquareSawVCOLite(sr)
    out_wave = vco(freq, osc_shape, n_samples)

    for idx, audio in enumerate(out_wave):
        audio = audio.unsqueeze(0)
        torchaudio.save(f"../out/out_wave_{idx}.wav", audio, sr)
