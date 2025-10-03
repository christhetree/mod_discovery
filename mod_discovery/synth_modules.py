import logging
import os
from typing import Optional, List, Literal, Tuple

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T, nn

import util
from curves import PiecewiseBezier
from filters import (
    TimeVaryingBiquad,
    calc_logits_to_biquad_a_coeff_triangle,
    time_varying_fir,
    sample_wise_lpc_scriptable,
)
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SynthModule(nn.Module):
    forward_param_names: List[str] = []
    lfo_name: Optional[str] = None

    def forward(self, *args, **kwargs) -> T:
        pass


class SquareSawVCOLite(SynthModule):
    forward_param_names = [
        "f0_hz",
        "osc_shape",
        "n_samples",
        "phase",
    ]
    lfo_name = "osc_shape"

    # Based off TorchSynth's SquareSawVCO
    def __init__(self, sr: int):
        super().__init__()
        self.sr = sr

    @staticmethod
    def calc_n_partials(f0_hz: T) -> T:
        assert f0_hz.ndim == 2
        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
        # TODO(cm): the constant 12000 is only valid for sr of 44100 Hz
        n_partials = 12000 / (max_f0_hz * tr.log10(max_f0_hz))
        return n_partials

    def forward(
        self,
        f0_hz: T,
        osc_shape: T,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        assert 1 <= osc_shape.ndim <= 2

        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)
        if osc_shape.ndim == 1:
            assert n_samples is not None
            osc_shape = osc_shape.unsqueeze(1)
            osc_shape = osc_shape.expand(-1, n_samples)

        arg = WavetableOsc.calc_osc_arg(self.sr, f0_hz, n_samples, phase)
        # TODO(cm): check how this works
        n_partials = self.calc_n_partials(f0_hz)
        square_wave = tr.tanh(tr.pi * n_partials * tr.sin(arg) / 2)
        out_wave = (1 - (osc_shape / 2)) * square_wave * (1 + (osc_shape * tr.cos(arg)))
        bs = f0_hz.size(0)
        assert out_wave.shape == (bs, n_samples)
        return out_wave


class WavetableOsc(SynthModule):
    forward_param_names = [
        "f0_hz",
        "wt_pos_0to1",
        "n_samples",
        "phase",
        "wt",
    ]
    lfo_name = "wt_pos_0to1"

    def __init__(
        self,
        sr: int,
        n_pos: Optional[int] = None,
        n_wt_samples: Optional[int] = None,
        wt: Optional[T] = None,
        aa_filter_n: Optional[int] = None,
        is_trainable: bool = True,
        init_noise_std: float = 0.01,
        cf_factor: float = 0.9,
        use_tanh: bool = False,
        use_aa: bool = True,
    ):
        super().__init__()
        self.sr = sr
        if wt is None:
            assert n_wt_samples is not None
            assert n_pos is not None
            wt = tr.empty(n_pos, n_wt_samples).normal_(mean=0.0, std=init_noise_std)
            is_wt_provided = False
        else:
            assert wt.ndim == 2
            n_pos, n_wt_samples = wt.size()
            is_wt_provided = True
        self.n_pos = n_pos
        self.n_wt_samples = n_wt_samples
        if aa_filter_n is None:
            assert n_wt_samples % 8 == 0
            # This is purely a heuristic
            aa_filter_n = n_wt_samples // 8 + 1
            log.info(
                f"Setting aa_filter_n = {aa_filter_n} "
                f"since n_wt_samples = {n_wt_samples}, n_pos = {n_pos}"
            )
        assert aa_filter_n % 2 == 1
        self.aa_filter_n = aa_filter_n
        self.is_trainable = is_trainable
        self.init_noise_std = init_noise_std
        self.use_tanh = use_tanh
        self.use_aa = use_aa
        self.cf_factor = cf_factor

        if is_trainable:
            self.wt = nn.Parameter(wt)
        else:
            self.register_buffer("wt", wt)
        if is_wt_provided:
            # We do this to run the corresponding checks
            self.set_wt(wt)

        aa_filter_support = 2 * (tr.arange(aa_filter_n) - (aa_filter_n - 1) / 2) / sr
        self.register_buffer(
            "aa_filter_support", aa_filter_support.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "window",
            tr.blackman_window(aa_filter_n, periodic=False).unsqueeze(0),
            persistent=False,
        )
        # TODO(cm): check whether this is correct or not
        self.wt_pitch_hz = sr / n_wt_samples

    def get_wt(self) -> T:
        return self.wt

    def set_wt(self, new_wt: T, strict: bool = True) -> None:
        assert not self.is_trainable, "Cannot set wt if is_trainable is True"
        assert new_wt.ndim == 2
        assert new_wt.size(1) == self.n_wt_samples
        new_n_pos = new_wt.size(0)
        if strict:
            assert new_n_pos == self.n_pos, f"new_n_pos = {new_n_pos} != {self.n_pos}"
        assert (
            new_n_pos <= self.n_pos
        ), f"new_n_pos = {new_n_pos} must be <= {self.n_pos}"
        if new_n_pos != self.n_pos:
            log.debug(
                f"Linearly interpolating new_wt.n_pos from {new_n_pos} "
                f"to {self.n_pos}"
            )
            new_wt = tr.swapaxes(new_wt, 0, 1)  # TODO(cm): tmp
            new_wt = util.interpolate_dim(new_wt, n=self.n_pos, dim=1)
            new_wt = tr.swapaxes(new_wt, 0, 1)  # TODO(cm): tmp
        assert new_wt.min() >= -1.0, f"new_wt.min() = {new_wt.min()}"
        assert new_wt.max() <= 1.0, f"new_wt.max() = {new_wt.max()}"
        self.wt[:, :] = new_wt[:, :]

    def calc_lp_sinc_blackman_coeff(self, cf_hz: T) -> T:
        assert cf_hz.ndim == 2
        # Compute sinc filter
        bs = cf_hz.size(0)
        support = self.aa_filter_support.expand(bs, -1)
        h = tr.sinc(cf_hz * support)
        # Apply window
        window = self.window.expand(bs, -1)
        h *= window
        # Normalize to get unity gain
        summed = tr.sum(h, dim=1, keepdim=True)
        h /= summed
        return h

    def get_maybe_aa_maybe_bounded_wt(self, max_f0_hz: T) -> T:
        wt = self.get_wt()
        # Maybe bound wavetable
        if self.use_tanh:
            assert self.is_trainable, "Using tanh on non-trainable wavetable"
            maybe_bounded_wt = tr.tanh(wt)
        else:
            maybe_bounded_wt = wt

        # Maybe apply anti-aliasing
        if not self.use_aa:
            return maybe_bounded_wt.unsqueeze(0)

        # if self.is_trainable:
        #     wt = maybe_bounded_wt.unsqueeze(0)
        #     wt = tr.swapaxes(wt, 1, 2)
        #     n_pad = self.pos_filter_n // 2
        #     wt = F.pad(wt, (n_pad, n_pad, 0, 0), mode="circular")
        #     wt = F.conv1d(wt, self.pos_filter, padding="valid", groups=self.n_wt_samples)
        #     wt = tr.swapaxes(wt, 1, 2)
        #     maybe_bounded_wt = wt.squeeze(0)

        pitch_ratio = max_f0_hz / self.wt_pitch_hz
        # Make the cutoff frequency a bit lower than the new nyquist
        cf_hz = (self.sr / pitch_ratio / 2.0) * self.cf_factor
        aa_filters = self.calc_lp_sinc_blackman_coeff(cf_hz)
        bs = aa_filters.size(0)
        aa_filters = aa_filters.unsqueeze(1).unsqueeze(1)
        # Put batch in channel dim to apply different kernel to each item in batch
        maybe_bounded_wt = maybe_bounded_wt.unsqueeze(0).unsqueeze(0)
        maybe_bounded_wt = maybe_bounded_wt.expand(1, bs, -1, -1)
        n_pad = self.aa_filter_n // 2
        padded_wt = F.pad(maybe_bounded_wt, (n_pad, n_pad, 0, 0), mode="circular")
        filtered_wt = F.conv2d(padded_wt, aa_filters, padding="valid", groups=bs)
        filtered_wt = tr.swapaxes(filtered_wt, 0, 1)
        filtered_wt = filtered_wt.squeeze(1)
        return filtered_wt

    def forward(
        self,
        f0_hz: T,
        wt_pos_0to1: T,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
        wt: Optional[T] = None,
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        assert 1 <= wt_pos_0to1.ndim <= 2
        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)
        if wt_pos_0to1.ndim == 1:
            assert n_samples is not None
            wt_pos_0to1 = wt_pos_0to1.unsqueeze(1)
            wt_pos_0to1 = wt_pos_0to1.expand(-1, n_samples)
        # This needs to be commented out to prevent rounding errors in TorchScript
        # from crashing the VST
        # assert wt_pos_0to1.min() >= 0.0, f"wt_pos_0to1.min() = {wt_pos_0to1.min()}"
        # assert wt_pos_0to1.max() <= 1.0, f"wt_pos_0to1.max() = {wt_pos_0to1.max()}"
        wt_pos = wt_pos_0to1 * 2.0 - 1.0

        arg = self.calc_osc_arg(self.sr, f0_hz, n_samples, phase)
        arg = arg % (2 * tr.pi)
        temp_coords = arg / tr.pi - 1.0  # Normalize to [-1, 1] for grid_sample
        # assert temp_coords.min() >= -1.0
        # assert temp_coords.max() <= 1.0
        flow_field = tr.stack([temp_coords, wt_pos], dim=2).unsqueeze(1)

        if wt is None:
            max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
            wt = self.get_maybe_aa_maybe_bounded_wt(max_f0_hz)
        wt = wt.unsqueeze(1)

        audio = F.grid_sample(
            wt,
            flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        audio = audio.squeeze(1).squeeze(1)
        # Commented out for TorchScript
        # bs = f0_hz.size(0)
        # assert audio.shape == (bs, n_samples)
        return audio

    @staticmethod
    def calc_osc_arg(
        sr: int, f0_hz: T, n_samples: Optional[int] = None, phase: Optional[T] = None
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        bs = f0_hz.size(0)

        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)

        if phase is None:
            # assert False  # TODO(cm): tmp
            phase = (
                tr.rand((bs,), dtype=f0_hz.dtype, device=f0_hz.device) * 2 * tr.pi
            ) - tr.pi
        phase = phase.view(bs, 1)
        arg = tr.cumsum(2 * tr.pi * f0_hz / sr, dim=1)
        arg += phase
        return arg


class BezierWavetableOsc(WavetableOsc):
    def __init__(
        self,
        sr: int,
        n_pos: int,
        n_wt_samples: int,
        n_segments: int,
        degree: int,
        aa_filter_n: Optional[int] = None,
        is_trainable: bool = True,
    ):
        super().__init__(
            sr,
            n_pos,
            n_wt_samples,
            wt=None,
            aa_filter_n=aa_filter_n,
            is_trainable=is_trainable,
        )
        self.n_segments = n_segments
        self.degree = degree

        self.bezier = PiecewiseBezier(n_pos, n_segments, degree)
        cp = tr.empty(n_wt_samples, n_segments * degree + 1).normal_(mean=0.0, std=0.01)

        if is_trainable:
            self.cp = nn.Parameter(cp)
        else:
            self.register_buffer("cp", cp)
        self.wt = None

    def get_wt(self) -> T:
        cp = self.cp.unfold(dimension=1, size=self.degree + 1, step=self.degree)
        wt = self.bezier.make_bezier(cp, cp_are_logits=False)
        wt = tr.swapaxes(wt, 0, 1)
        return wt


class WavetableOscShan(WavetableOsc):
    """
    Wavetable Oscillator from Differentiable Wavetable Synthesis, Shan et al.
    that uses weighted sum instead of grid_sample.
    """

    @staticmethod
    def _wavetable_osc_shan(
        wavetable: T,
        freq: T,
        sr: int,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
    ):
        """
        Wavetable synthesis oscilator with batch linear interpolation
        Input:
            wavetable: (batch_size, n_wavetable, wavetable_len,)
            freq: (batch_size, n_samples)
            sr: int
        Output:
            signal: (batch_size, n_wavetable, n_samples)

        """
        arg = WavetableOsc.calc_osc_arg(sr, freq, n_samples, phase)
        arg = arg % (2 * tr.pi)
        temp_coords = arg / tr.pi - 1.0  # Normalize to [-1, 1] for grid_sample
        assert temp_coords.min() >= -1.0
        assert temp_coords.max() <= 1.0
        flow_field = tr.stack(
            [temp_coords, tr.zeros_like(temp_coords)], dim=2
        ).unsqueeze(1)
        wt = wavetable.unsqueeze(2)
        signal = F.grid_sample(
            wt,
            flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)
        return signal

    def forward(
        self,
        f0_hz: T,
        wt_pos_0to1: Optional[T] = None,  # TODO(cm): consolidate naming
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
    ) -> T:
        attention_matrix = wt_pos_0to1
        assert 1 <= f0_hz.ndim <= 2
        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)

        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
        wt = self.get_maybe_aa_maybe_bounded_wt(max_f0_hz)

        audio = WavetableOscShan._wavetable_osc_shan(
            wt, f0_hz, self.sr, n_samples, phase
        )
        if attention_matrix is not None:
            # If attention matrix is provided, it must have shape (bs, n_samples, n_pos)
            assert attention_matrix.ndim == 3
            attention_matrix = tr.swapaxes(attention_matrix, 1, 2)
            assert attention_matrix.shape == audio.shape, (
                f"Attention matrix must be the same as audio shape, given attention "
                f"matrix: {attention_matrix.shape}, audio: {audio.shape}"
            )
            audio = tr.einsum("bns,bns->bs", audio, attention_matrix)
        else:
            # If no attention matrix is provided, we simply take the mean
            audio = audio.mean(dim=1)

        return audio


class DDSPHarmonicOsc(SynthModule):
    """
    A harmonic oscillator from DDSP, largely following:
    https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py#L135
    """

    forward_param_names = [
        "f0_hz",
        "harmonic_amplitudes",
        "n_samples",
        "phase",
    ]
    lfo_name = "harmonic_amplitudes"

    def __init__(
        self,
        sr: int,
        n_harmonics: int,
    ):
        super().__init__()
        self.sr = sr
        self.n_harmonics = n_harmonics
        self.is_trainable = False

    def forward(
        self,
        f0_hz: T,
        harmonic_amplitudes: T,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
        eps: float = 1e-5,
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)

        if harmonic_amplitudes is not None:
            # if harmonic amplitudes are provided, it must be of shape (bs, n_samples, n_harmonics)
            assert (
                harmonic_amplitudes.ndim == 3
            ), f"Harmonic amplitudes must be of shape (bs, n_samples, n_harmonics), given shape: {harmonic_amplitudes.shape}"
            assert harmonic_amplitudes.shape == (
                f0_hz.size(0),
                f0_hz.size(1),
                self.n_harmonics + 1,
            ), f"Harmonic amplitudes shape {harmonic_amplitudes.shape} must be the same as f0_hz shape {f0_hz.shape}"
        else:
            # or else, we assign the same amplitude for all harmonics
            harmonic_amplitudes = tr.ones(
                f0_hz.size(0), f0_hz.size(1), self.n_harmonics + 1
            ).to(f0_hz.device)

        harmonic_amplitudes = util.scale_function(harmonic_amplitudes)
        f0_hz = f0_hz.unsqueeze(-1)

        total_amplitude = harmonic_amplitudes[..., :1]
        harmonic_amplitudes = harmonic_amplitudes[..., 1:]

        # anti-aliasing by zero-ing out the amplitudes for harmonics that are above nyquist
        harmonic_amplitudes_aa = self.remove_above_nyquist(harmonic_amplitudes, f0_hz)

        # normalize the amplitudes of each harmonic to sum to `total_amplitude`
        harmonic_amplitudes_aa /= harmonic_amplitudes_aa.sum(dim=-1, keepdim=True) + eps
        harmonic_amplitudes_aa *= total_amplitude

        omega = tr.cumsum(2 * tr.pi * f0_hz / self.sr, dim=1)
        if phase is not None:
            phase = phase.view(-1, 1, 1)
            assert len(phase.shape) == len(
                omega.shape
            ), f"Size mismatch, phase: {phase.shape}, omega: {omega.shape}"
            omega += phase

        omegas = omega * tr.arange(1, self.n_harmonics + 1, device=omega.device)

        signal = tr.sin(omegas) * harmonic_amplitudes_aa
        signal = signal.sum(dim=-1)

        return signal

    def remove_above_nyquist(self, harmonic_amplitudes: T, f0_hz: T) -> T:
        f0_hz_harmonics = f0_hz * tr.arange(
            1, self.n_harmonics + 1, device=f0_hz.device
        )
        aa = (f0_hz_harmonics < self.sr / 2).float()
        return harmonic_amplitudes * aa


class DDSPNoiseModule(SynthModule):
    """
    Filtered noise module in DDSP, largely following:
    https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/model.py#L88
    """

    forward_param_names = [
        "noise_amplitudes",
        "n_samples",
    ]
    lfo_name = "noise_amplitudes"

    def __init__(
        self,
        sr: int,
        n_bands: int,
    ):
        super().__init__()
        self.sr = sr
        self.n_bands = n_bands

    def forward(
        self,
        x: T,
        noise_amplitudes: T,
        n_samples: int,
    ):
        assert (
            noise_amplitudes.size(2) == self.n_bands
        ), f"Noise amplitudes size mismatch, expected {self.n_bands,} but got {noise_amplitudes.size(2)}"

        # NOTE: there is a mysterious `- 5` in the original code for the noise part
        # we follow as-is, but it is not clear why this exists
        noise_amplitudes = util.scale_function(noise_amplitudes - 5)

        block_size = int(round(n_samples / noise_amplitudes.size(1)))
        impulse = DDSPNoiseModule.amp_to_impulse_response(noise_amplitudes, block_size)
        noise = (
            tr.rand(
                impulse.shape[0],
                impulse.shape[1],
                block_size,
            ).to(impulse)
            * 2
            - 1
        )

        noise = DDSPNoiseModule.fft_convolve(noise, impulse)
        noise = noise.reshape(noise.shape[0], -1)
        noise = noise[:, :n_samples]

        signal = x + noise
        return signal

    @staticmethod
    def amp_to_impulse_response(amp: T, target_size: int) -> T:
        amp = tr.stack([amp, tr.zeros_like(amp)], -1)
        amp = tr.view_as_complex(amp)
        amp = tr.fft.irfft(amp)

        filter_size = amp.shape[-1]

        amp = tr.roll(amp, filter_size // 2, -1)
        win = tr.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

        amp = amp * win

        amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
        amp = tr.roll(amp, -filter_size // 2, -1)

        return amp

    @staticmethod
    def fft_convolve(signal: T, kernel: T) -> T:
        signal = nn.functional.pad(signal, (0, signal.shape[-1]))
        kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

        output = tr.fft.irfft(tr.fft.rfft(signal) * tr.fft.rfft(kernel))
        output = output[..., output.shape[-1] // 2 :]

        return output


class BiquadWQFilter(SynthModule):
    forward_param_names = [
        "w_mod_sig",
        "q_mod_sig",
        "filter_type",
    ]
    lfo_name = "w_mod_sig"

    def __init__(
        self,
        sr: int,
        min_w_hz: float,
        max_w_hz: float,
        min_q: float,
        max_q: float,
        eps: float = 1e-3,
        modulate_log_w: bool = True,
        modulate_log_q: bool = True,
        interp_coeff: bool = False,
        filter_type: Literal["lp", "hp", "bp", "no"] = "lp",
    ):
        super().__init__()
        self.sr = sr
        self.min_w_hz = min_w_hz
        self.max_w_hz = max_w_hz
        self.min_q = min_q
        self.max_q = max_q
        self.eps = eps
        self.modulate_log_w = modulate_log_w
        self.modulate_log_q = modulate_log_q
        self.interp_coeff = interp_coeff
        self.filter_type = filter_type

        self.min_w = 2 * tr.pi * min_w_hz / sr
        self.max_w = 2 * tr.pi * max_w_hz / sr
        self.filter = TimeVaryingBiquad(
            self.min_w, self.max_w, min_q, max_q, eps, modulate_log_w, modulate_log_q
        )
        self.next_zi = tr.zeros((1, 2)).float()

    def toggle_scriptable(self, is_scriptable: bool) -> None:
        self.filter.toggle_scriptable(is_scriptable)

    def forward(
        self,
        x: T,
        w_mod_sig: Optional[T] = None,
        q_mod_sig: Optional[T] = None,
        zi: Optional[T] = None,
    ) -> T:
        if w_mod_sig is not None:
            assert x.shape == w_mod_sig.shape
        if q_mod_sig is not None:
            if q_mod_sig.shape != x.shape:
                assert q_mod_sig.shape == (x.size(0),)
                q_mod_sig = q_mod_sig.unsqueeze(1).expand(-1, x.size(1))
        y_ab, a_coeff, b_coeff, y_a = self.filter(
            x,
            self.filter_type,
            w_mod_sig,
            q_mod_sig,
            interp_coeff=self.interp_coeff,
            zi=zi,
        )
        self.next_zi = y_a[:, -2:]
        return y_ab


class BiquadCoeffFilter(SynthModule):
    forward_param_names = [
        "coeff_logits",
    ]
    lfo_name = "coeff_logits"

    def __init__(self, interp_coeff: bool = False, eps: float = 1e-3):
        super().__init__()
        self.interp_coeff = interp_coeff
        self.eps = eps

        self.lpc_func = sample_wise_lpc
        self.next_zi = tr.zeros((1, 2)).float()

    def _calc_coeff(self, logits: T, n_frames: int) -> Tuple[T, T]:
        bs = logits.size(0)
        if not self.interp_coeff:
            logits = util.interpolate_dim(logits, n_frames, dim=1)
            assert logits.shape == (bs, n_frames, 5)
        a_logits = logits[..., :2]
        a_coeff = calc_logits_to_biquad_a_coeff_triangle(a_logits, self.eps)
        b_coeff = logits[..., 2:]
        if self.interp_coeff:
            a_coeff = util.interpolate_dim(a_coeff, n_frames, dim=1)
            assert a_coeff.shape == (bs, n_frames, 2)
            b_coeff = util.interpolate_dim(b_coeff, n_frames, dim=1)
            assert b_coeff.shape == (bs, n_frames, 3)
        return a_coeff, b_coeff

    def toggle_scriptable(self, is_scriptable: bool) -> None:
        if is_scriptable:
            self.lpc_func = sample_wise_lpc_scriptable
        else:
            self.lpc_func = sample_wise_lpc

    def forward(self, x: T, coeff_logits: T, zi: Optional[T] = None) -> T:
        assert coeff_logits.ndim == 3
        n_samples = x.size(1)
        a_coeff, b_coeff = self._calc_coeff(coeff_logits, n_samples)
        zi_a = zi
        if zi_a is not None:
            zi_a = tr.flip(zi_a, dims=[1])  # Match scipy's convention for torchlpc
        y_a = self.lpc_func(x, a_coeff, zi=zi_a)
        # assert not tr.isinf(y_a).any()
        # assert not tr.isnan(y_a).any()
        self.next_zi = y_a[:, -2:]
        y_ab = time_varying_fir(y_a, b_coeff, zi=zi)
        return y_ab
