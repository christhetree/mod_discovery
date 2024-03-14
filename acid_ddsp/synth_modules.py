import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch as tr
from torch import Tensor as T, nn

from torchsynth.config import SynthConfig
from torchsynth.module import ADSR

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
    def __init__(
        self,
        sr: int,
        n_samples: int,
        batch_size: int,
        min_adsr_vals: ADSRValues,
        max_adsr_vals: ADSRValues,
        **kwargs: Dict[str, T],
    ) -> None:
        assert int(sr) == sr
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
        sc = SynthConfig(
            batch_size=batch_size,
            sample_rate=sr,
            buffer_size_seconds=n_samples / sr,
            control_rate=int(sr),
            reproducible=False,
            no_grad=True,
            debug=False,
        )
        super().__init__(sc, **kwargs)


class SquareSawVCOLite(nn.Module):
    # Based off TorchSynth's SquareSawVCO
    def __init__(self, sr: int):
        super().__init__()
        self.sr = sr

    def calc_n_partials(self, f0_hz: T) -> T:
        assert f0_hz.ndim == 2
        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
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
