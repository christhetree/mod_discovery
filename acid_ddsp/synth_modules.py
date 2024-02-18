import logging
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict

import torch as tr
from torch import Tensor as T
from torchsynth.config import SynthConfig
from torchsynth.module import SquareSawVCO, LFO, ADSR
from torchsynth.signal import Signal

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


@dataclass
class ADSRValues:
    attack: Optional[float] = None
    decay: Optional[float] = None
    sustain: Optional[float] = None
    release: Optional[float] = None
    alpha: Optional[float] = None


class CustomADSR(ADSR):
    # TODO(cm): refactor?
    def __init__(self,
                 synthconfig: SynthConfig,
                 adsr_vals: Optional[ADSRValues] = None,
                 **kwargs: Dict[str, T]) -> None:
        dur = synthconfig.buffer_size_seconds
        for dr in self.default_parameter_ranges:
            if dr.name == "attack":
                if adsr_vals is not None and adsr_vals.attack is not None:
                    dr.minimum = adsr_vals.attack
                    dr.maximum = adsr_vals.attack
                else:
                    dr.minimum = 0.0
                    dr.maximum = min(1.0, dur)
            if dr.name == "decay":
                if adsr_vals is not None and adsr_vals.decay is not None:
                    dr.minimum = adsr_vals.decay
                    dr.maximum = adsr_vals.decay
                else:
                    dr.minimum = 0.0
                    dr.maximum = min(1.0, dur)
            if dr.name == "sustain":
                if adsr_vals is not None and adsr_vals.sustain is not None:
                    dr.minimum = adsr_vals.sustain
                    dr.maximum = adsr_vals.sustain
                else:
                    dr.minimum = 0.5
                    dr.maximum = 1.0
            if dr.name == "release":
                if adsr_vals is not None and adsr_vals.release is not None:
                    dr.minimum = adsr_vals.release
                    dr.maximum = adsr_vals.release
                else:
                    dr.minimum = 1.0
                    dr.maximum = 1.0
            if dr.name == "alpha":
                if adsr_vals is not None and adsr_vals.alpha is not None:
                    dr.minimum = adsr_vals.alpha
                    dr.maximum = adsr_vals.alpha
                else:
                    dr.minimum = 0.1
                    dr.maximum = 2.0
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
