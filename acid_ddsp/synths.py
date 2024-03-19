import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

import torch as tr
from torch import Tensor as T, nn

from acid_ddsp.synth_modules import CustomADSR, ADSRValues, SquareSawVCOLite
from audio_config import AudioConfig
from filters import (
    TimeVaryingLPBiquad,
    calc_logits_to_biquad_a_coeff_triangle,
    time_varying_fir,
    calc_logits_to_biquad_coeff_pole_zero,
)
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthBase(ABC, nn.Module):
    # TODO(cm): switch to ADSRLite
    def __init__(self, ac: AudioConfig, batch_size: int):
        super().__init__()
        self.ac = ac
        self.batch_size = batch_size
        min_adsr_vals = ADSRValues(
            attack=ac.min_attack,
            decay=ac.min_decay,
            sustain=ac.min_sustain,
            release=ac.min_release,
            alpha=ac.min_alpha,
        )
        max_adsr_vals = ADSRValues(
            attack=ac.max_attack,
            decay=ac.max_decay,
            sustain=ac.max_sustain,
            release=ac.max_release,
            alpha=ac.max_alpha,
        )
        self.vco = SquareSawVCOLite(ac.sr, batch_size)
        self.adsr = CustomADSR(
            ac.sr, ac.n_samples, batch_size, min_adsr_vals, max_adsr_vals
        )

    @abstractmethod
    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        pass

    def forward(
        self,
        f0_hz: T,
        osc_shape: T,
        note_on_duration: T,
        filter_args: Dict[str, T],
        dist_gain: T,
    ) -> (T, T, T):
        assert (
            f0_hz.shape
            == osc_shape.shape
            == note_on_duration.shape
            == dist_gain.shape
            == (self.batch_size,)
        )
        dry_audio = self.vco(f0_hz, osc_shape, n_samples=self.ac.n_samples)
        dry_audio *= self.ac.osc_audio_gain
        envelope = self.adsr(note_on_duration)
        dry_audio *= envelope
        wet_audio = self.filter_dry_audio(dry_audio, filter_args)
        wet_audio = wet_audio * dist_gain.unsqueeze(-1)
        wet_audio = tr.tanh(wet_audio)
        return dry_audio, wet_audio, envelope


class AcidSynth(AcidSynthBase):
    def __init__(self, ac: AudioConfig, batch_size: int):
        super().__init__(ac, batch_size)
        self.tvlpb = TimeVaryingLPBiquad(
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
        )

    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        w_mod_sig = filter_args["w_mod_sig"]
        q_mod_sig = filter_args["q_mod_sig"]
        assert w_mod_sig.shape == q_mod_sig.shape
        y = self.tvlpb(x, w_mod_sig, q_mod_sig)
        return y


class AcidSynthLearnedBiquadCoeff(AcidSynthBase):
    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        a_logits = filter_args["a_logits"]
        a = calc_logits_to_biquad_a_coeff_triangle(a_logits)
        b = filter_args["b_logits"]
        y_a = sample_wise_lpc(x, a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b)
        return y_ab


class AcidSynthLearnedBiquadPoleZero(AcidSynthBase):
    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        q_real = filter_args["q_real"]
        q_imag = filter_args["q_imag"]
        p_real = filter_args["p_real"]
        p_imag = filter_args["p_imag"]
        a, b = calc_logits_to_biquad_coeff_pole_zero(q_real, q_imag, p_real, p_imag)
        y_a = sample_wise_lpc(x, a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b)
        return y_ab


class AcidSynthLSTM(AcidSynthBase):
    def __init__(
        self, ac: AudioConfig, batch_size: int, n_ch: int = 1, n_hidden: int = 3
    ):
        super().__init__(ac, batch_size)
        self.n_ch = n_ch
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_ch + 2,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out_lstm = nn.Linear(n_hidden, n_ch)

    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        w_mod_sig = filter_args["w_mod_sig"]
        q_mod_sig = filter_args["q_mod_sig"]
        x_orig = x
        x = tr.stack([x, w_mod_sig, q_mod_sig], dim=2)
        x, _ = self.lstm(x)
        x = self.out_lstm(x)
        x = x.squeeze(-1)
        x = x + x_orig
        x = tr.tanh(x)
        return x
