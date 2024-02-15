import logging
import os
from typing import Optional
import torch as tr
from torch import Tensor as T
from torchsynth.config import SynthConfig
from torchsynth.module import ControlRateUpsample, VCA, SquareSawVCO
from torchsynth.synth import AbstractSynth

from synth_modules import CustomADSR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class CustomSynth(AbstractSynth):
    def __init__(self, synthconfig: SynthConfig, vco_shape: Optional[float] = 1.0):
        super().__init__(synthconfig=synthconfig)
        self.vco_shape = vco_shape
        self.add_synth_modules(
            [
                ("vco", SquareSawVCO),
                ("adsr", CustomADSR),
                ("upsample", ControlRateUpsample),
                ("vca", VCA),
            ]
        )
        if vco_shape is not None:
            self.vco.set_parameter("tuning", tr.zeros((self.batch_size,)))
            self.vco.set_parameter("mod_depth", tr.zeros((self.batch_size,)))
            self.vco.set_parameter("shape", tr.full((self.batch_size,), vco_shape))
        self.freeze_parameters([
            ("vco", "tuning"),
            ("vco", "mod_depth"),
            ("vco", "shape"),
        ])

    def output(self, midi_f0: T, note_on_duration: T) -> (T, T):
        envelope = self.adsr(note_on_duration)
        envelope = self.upsample(envelope)
        audio = self.vco(midi_f0)
        # audio = self.vca(audio, envelope)
        return audio, envelope
