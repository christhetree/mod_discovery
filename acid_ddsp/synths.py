import logging
import os
from typing import Optional

import torch as tr
from torch import Tensor as T
from torchsynth.config import SynthConfig
from torchsynth.module import ControlRateUpsample, MonophonicKeyboard, VCA
from torchsynth.synth import AbstractSynth

from synth_modules import CustomLFO, CustomSquareSawVCO, CustomADSR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class CustomSynth(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None):
        super().__init__(synthconfig=synthconfig)
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard, {"midi_f0": tr.full((self.batch_size,), 57)}),
                ("adsr", CustomADSR),
                ("lfo", CustomLFO, {"lfo_type": "sin"}),
                ("upsample", ControlRateUpsample),
                ("vco", CustomSquareSawVCO),
                ("vca", VCA),
            ]
        )
        self.freeze_parameters([
            # ("keyboard", "midi_f0"),
            ("lfo", "sin"),
            ("lfo", "tri"),
            ("lfo", "saw"),
            ("lfo", "rsaw"),
            ("lfo", "sqr"),
            ("vco", "tuning"),
            ("vco", "mod_depth"),
        ])
        for p in self.parameters():
            p.requires_grad = False

    def output(self) -> (T, T):
        with tr.no_grad():
            midi_f0, note_on_duration = self.keyboard()
            mod_sig = self.lfo()
            mod_sig_upsampled = self.upsample(mod_sig)

            # envelope = self.adsr(note_on_duration)
            # from matplotlib import pyplot as plt
            # plt.plot(envelope[0].detach().numpy())
            # plt.show()
            # envelope = self.upsample(envelope)

            mult = (tr.rand((self.synthconfig.batch_size, 1)).to(self.device) * 0.8) + 0.2
            mod_sig_new = mult * mod_sig_upsampled
            # mod_sig_upsampled += ((1 - mult) / 2.0)
            mod_sig_new += ((1 - mult) * tr.rand((self.synthconfig.batch_size, 1)).to(self.device))
            out = self.vco(midi_f0, mod_sig_new)
            # out = self.vca(out, envelope)
            return out, mod_sig_upsampled
