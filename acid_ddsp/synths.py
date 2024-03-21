import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch as tr
from torch import Tensor as T, nn

import util
from acid_ddsp.synth_modules import CustomADSR, ADSRValues, SquareSawVCOLite
from audio_config import AudioConfig
from filters import (
    TimeVaryingLPBiquad,
    calc_logits_to_biquad_a_coeff_triangle,
    time_varying_fir,
    calc_logits_to_biquad_coeff_pole_zero,
    TimeVaryingLPBiquadFSM,
)
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def make_synth(
    synth_type: str, ac: AudioConfig, batch_size: int, **kwargs: Dict[str, Any]
) -> "AcidSynthBase":
    if synth_type == "AcidSynth":
        synth_class = AcidSynth
    elif synth_type == "AcidSynthLPBiquadFSM":
        synth_class = AcidSynthLPBiquadFSM
    elif synth_type == "AcidSynthLearnedBiquadCoeff":
        synth_class = AcidSynthLearnedBiquadCoeff
    elif synth_type == "AcidSynthLearnedBiquadPoleZero":
        synth_class = AcidSynthLearnedBiquadPoleZero
    elif synth_type == "AcidSynthLSTM":
        synth_class = AcidSynthLSTM
    else:
        raise ValueError(f"Unknown synth_type: {synth_type}")
    return synth_class(ac, batch_size, **kwargs)


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

    def resize_mod_sig(self, x: T, w_mod_sig: T, q_mod_sig: T) -> (T, T):
        assert x.ndim == 2
        if w_mod_sig.shape != x.shape:
            assert w_mod_sig.ndim == x.ndim
            w_mod_sig = util.linear_interpolate_last_dim(
                w_mod_sig, x.size(1), align_corners=True
            )
        if q_mod_sig.shape != x.shape:
            assert q_mod_sig.ndim == x.ndim
            q_mod_sig = util.linear_interpolate_last_dim(
                q_mod_sig, x.size(1), align_corners=True
            )
        return w_mod_sig, q_mod_sig

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
        self.filter = TimeVaryingLPBiquad(
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
        )

    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        w_mod_sig = filter_args["w_mod_sig"]
        q_mod_sig = filter_args["q_mod_sig"]
        w_mod_sig, q_mod_sig = self.resize_mod_sig(x, w_mod_sig, q_mod_sig)
        y = self.filter(x, w_mod_sig, q_mod_sig)
        return y


class AcidSynthLPBiquadFSM(AcidSynthBase):
    def __init__(
        self,
        ac: AudioConfig,
        batch_size: int,
        win_len: Optional[int] = None,
        win_len_sec: Optional[float] = None,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
    ):
        super().__init__(ac, batch_size)
        self.filter = TimeVaryingLPBiquadFSM(
            win_len=win_len,
            win_len_sec=win_len_sec,
            sr=ac.sr,
            overlap=overlap,
            oversampling_factor=oversampling_factor,
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
        )

    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        w_mod_sig = filter_args["w_mod_sig"]
        q_mod_sig = filter_args["q_mod_sig"]
        n_samples = x.size(1)
        n_frames = self.filter.filter.calc_n_frames(n_samples)
        if w_mod_sig.size(1) != n_frames:
            assert w_mod_sig.ndim == x.ndim
            w_mod_sig = util.linear_interpolate_last_dim(
                w_mod_sig, n_frames, align_corners=True
            )
        if q_mod_sig.size(1) != n_frames:
            assert q_mod_sig.ndim == x.ndim
            q_mod_sig = util.linear_interpolate_last_dim(
                q_mod_sig, n_frames, align_corners=True
            )
        y = self.filter(x, w_mod_sig, q_mod_sig)
        return y


class AcidSynthLearnedBiquadCoeff(AcidSynthBase):
    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        logits = filter_args["logits"]
        assert logits.ndim == 3
        if logits.size(1) != x.size(1):
            logits = logits.swapaxes(1, 2)
            logits = util.linear_interpolate_last_dim(
                logits, x.size(1), align_corners=True
            )
            logits = logits.swapaxes(1, 2)
        assert logits.shape == (x.size(0), x.size(1), 5)
        a_logits = logits[..., :2]
        a = calc_logits_to_biquad_a_coeff_triangle(a_logits)
        b = logits[..., 2:]

        y_a = sample_wise_lpc(x, a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b)
        return y_ab


class AcidSynthLearnedBiquadPoleZero(AcidSynthBase):
    def filter_dry_audio(self, x: T, filter_args: Dict[str, T]) -> T:
        logits = filter_args["logits"]
        assert logits.ndim == 3
        if logits.size(1) != x.size(1):
            logits = logits.swapaxes(1, 2)
            logits = util.linear_interpolate_last_dim(
                logits, x.size(1), align_corners=True
            )
            logits = logits.swapaxes(1, 2)
        assert logits.shape == (x.size(0), x.size(1), 4)
        q_real = logits[..., 0]
        q_imag = logits[..., 1]
        p_real = logits[..., 2]
        p_imag = logits[..., 3]
        a, b = calc_logits_to_biquad_coeff_pole_zero(q_real, q_imag, p_real, p_imag)

        y_a = sample_wise_lpc(x, a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b)
        return y_ab


class AcidSynthLSTM(AcidSynthBase):
    def __init__(self, ac: AudioConfig, batch_size: int, n_hidden: int, n_ch: int = 1):
        super().__init__(ac, batch_size)
        self.n_hidden = n_hidden
        self.n_ch = n_ch
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
        w_mod_sig, q_mod_sig = self.resize_mod_sig(x, w_mod_sig, q_mod_sig)
        x_orig = x
        x = tr.stack([x, w_mod_sig, q_mod_sig], dim=2)
        x, _ = self.lstm(x)
        x = self.out_lstm(x)
        x = x.squeeze(-1)
        x = x + x_orig
        x = tr.tanh(x)
        return x
