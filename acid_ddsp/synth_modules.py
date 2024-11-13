import logging
import os
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T, nn

from adsr_naotake import ADSREnvelope

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ADSRLite(nn.Module):
    # Based off TorchSynth's ADSR
    def __init__(self, sr: int, eps: float = 1e-7):
        super().__init__()
        self.sr = sr
        self.eps = eps
        self.register_buffer("zero", tr.tensor(0.0))
        self.register_buffer("one", tr.tensor(1.0))

    def seconds_to_samples(self, seconds: T) -> T:
        return seconds * self.sr

    def ramp(
        self,
        duration_sec: T,
        alpha: T,
        n_samples: int,
        start_sec: Optional[T] = None,
        inverse: bool = False,
    ) -> T:
        assert duration_sec.ndim == 1
        assert alpha.shape == duration_sec.shape
        bs = duration_sec.size(0)
        duration = self.seconds_to_samples(duration_sec).unsqueeze(1)
        assert duration.min() >= 1.0
        assert duration.max() <= n_samples

        # Convert to number of samples.
        start = self.zero
        if start_sec is not None:
            assert start_sec.shape == duration_sec.shape
            start = self.seconds_to_samples(start_sec).unsqueeze(1)
        assert start.min() >= 0.0
        assert start.max() < n_samples

        # Build ramps template.
        range_ = tr.arange(n_samples, dtype=duration.dtype, device=duration.device)
        ramp = range_.expand((bs, range_.size(0)))

        # Shape ramps.
        ramp = ramp - start
        ramp = tr.maximum(ramp, self.zero)
        ramp = (ramp + self.eps) / duration + self.eps
        ramp = tr.minimum(ramp, self.one)

        # The following is a workaround. In inverse mode, a ramp with 0 duration
        # (that is all 1's) becomes all 0's, which is a problem for the
        # ultimate calculation of the ADSR signal (a * d * r => 0's). So this
        # replaces only rows who sum to 0 (i.e., all components are zero).
        if inverse:
            ramp = tr.where(duration > 0.0, 1.0 - ramp, ramp)

        # Apply scaling factor.
        ramp = tr.pow(ramp, alpha.unsqueeze(1))
        return ramp

    def make_attack(self, attack: T, alpha: T, n_samples: int) -> T:
        return self.ramp(attack, alpha, n_samples)

    def make_decay(
        self, attack: T, decay: T, sustain: T, alpha: T, n_samples: int
    ) -> T:
        assert attack.ndim == 1
        assert attack.shape == decay.shape == sustain.shape == alpha.shape
        sustain = sustain.unsqueeze(1)
        a = 1.0 - sustain
        b = self.ramp(decay, alpha, n_samples, start_sec=attack, inverse=True)
        out = a * b + sustain
        out = out.squeeze(1)
        return out

    def make_release(
        self, release: T, alpha: T, note_on_duration: T, n_samples: int
    ) -> T:
        return self.ramp(
            release, alpha, n_samples, start_sec=note_on_duration, inverse=True
        )

    def forward(
        self,
        attack: T,
        decay: T,
        sustain: T,
        release: T,
        alpha: T,
        note_on_duration: T,
        n_samples: int,
    ) -> T:
        assert attack.ndim == 1
        assert (
            attack.shape
            == decay.shape
            == sustain.shape
            == release.shape
            == alpha.shape
            == note_on_duration.shape
        )
        assert alpha.min() >= 0.0

        new_attack = tr.minimum(attack, note_on_duration)
        new_decay = tr.maximum(note_on_duration - attack, self.zero)
        new_decay = tr.minimum(new_decay, decay)

        attack_signal = self.make_attack(new_attack, alpha, n_samples)
        decay_signal = self.make_decay(new_attack, new_decay, sustain, alpha, n_samples)
        release_signal = self.make_release(release, alpha, note_on_duration, n_samples)

        envelope = attack_signal * decay_signal * release_signal
        return envelope


class ExpDecayEnv(ADSRLite):
    def forward(
        self,
        alpha: T,
        note_on_duration: T,
        n_samples: int,
    ) -> T:
        assert alpha.ndim == 1
        assert alpha.shape == note_on_duration.shape
        assert alpha.min() >= 0.0
        envelope = self.ramp(note_on_duration, alpha, n_samples, inverse=True)
        return envelope


class ADSR(nn.Module):
    def __init__(self, n_frames: int):
        super().__init__()
        self.n_frames = n_frames
        self.register_buffer("ramp", tr.linspace(0, 1.0, n_frames))

    def forward(
        self,
        note_off: T,
        attack: T,
        decay: T,
        sustain: T,
        release: T,
        floor: Optional[T] = None,
        peak: Optional[T] = None,
        pow: float = 1.0,
    ):
        env = self.make_env(
            self.ramp,
            note_off,
            attack,
            decay,
            sustain,
            release,
            floor,
            peak,
            pow,
        )
        return env

    @staticmethod
    def soft_clamp_min(x: T, min_val: float, t: float = 100.0):
        return tr.sigmoid((min_val - x) * t) * (min_val - x) + x

    @staticmethod
    def power_function(x: T, pow: float = 2.0) -> T:
        assert x.ndim == 3
        if pow > 0:  # convex
            # transpose
            if x.squeeze()[0] > x.squeeze()[-1]:
                y_intercept = x.squeeze()[-1]
                y = x - x[:, -1, :]
                max_val = y.squeeze()[0]
                y = y / max_val
            else:
                y_intercept = x.squeeze()[0]
                y = x - x[:, 0, :]
                max_val = y.squeeze()[-1]
                y = y / max_val
            y = y**pow
            # transpose back
            y = y * max_val + y_intercept
        else:
            # transpose
            if x.squeeze()[0] > x.squeeze()[-1]:
                max_val = x.squeeze()[0]
                y = x - x[:, 0, :]
                y_intercept = y.squeeze()[-1]
                y = y / -y_intercept
            else:
                max_val = x.squeeze()[-1]
                y = x - x[:, -1, :]
                y_intercept = y.squeeze()[0]
                y = y / -y_intercept

            y = -(y**-pow)

            # transpose back
            y = y * -y_intercept + max_val

        return y

    @staticmethod
    def make_env(
        ramp: T,
        note_off: T,
        attack: T,
        decay: T,
        sustain: T,
        release: T,
        floor: Optional[T] = None,
        peak: Optional[T] = None,
        # pow: float = 1.0,
        soft_clip_t: float = 100.0,
        eps: float = 1e-6,
    ):
        bs = attack.size(0)
        if floor is None:
            floor = tr.zeros_like(attack)
        if peak is None:
            peak = tr.ones_like(attack)
        note_off = tr.clip(note_off, min=0.0, max=1.0).view(-1, 1, 1)
        attack = tr.clip(attack, min=0.0, max=1.0).view(-1, 1, 1)
        decay = tr.clip(decay, min=0.0, max=1.0).view(-1, 1, 1)
        sustain = tr.clip(sustain, min=0.0, max=1.0).view(-1, 1, 1)
        release = tr.clip(release, min=0.0).view(-1, 1, 1)
        peak = tr.clip(peak, min=0.0, max=1.0).view(-1, 1, 1)
        floor = tr.clip(floor, min=0.0, max=1.0).view(-1, 1, 1)

        x = ramp.view(1, -1, 1).repeat(bs, 1, 1)
        # Offset 0 to epsilon value, so when attack = 0, first ADSR value is not 0 but 1
        x[:, 0, :] = eps

        A = x / (attack + eps)
        # A = self.power_function(A, pow=pow)
        A = tr.clip(A, max=1.0)

        D = (x - attack) * (sustain - 1.0) / (decay + eps)
        # D = self.power_function(D, pow=pow)
        D = tr.clip(D, max=0.0)
        D = ADSR.soft_clamp_min(D, sustain - 1.0, soft_clip_t)

        S = (x - note_off) * (-sustain / (release + eps))
        S = tr.clip(S, max=0.0)
        S = ADSR.soft_clamp_min(S, -sustain, soft_clip_t)

        env = (A + D + S) * (peak - floor) + floor
        env = tr.clip(env, min=0.0, max=1.0)
        env = env.squeeze(2)
        return env


class SquareSawVCOLite(nn.Module):
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
                tr.rand((bs, 1), dtype=f0_hz.dtype, device=f0_hz.device) * 2 * tr.pi
            ) - tr.pi
        assert phase.shape == (bs, 1)
        arg = tr.cumsum(2 * tr.pi * f0_hz / sr, dim=1)
        arg += phase
        return arg

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

        arg = self.calc_osc_arg(self.sr, f0_hz, n_samples, phase)
        # TODO(cm): check how this works
        n_partials = self.calc_n_partials(f0_hz)
        square_wave = tr.tanh(tr.pi * n_partials * tr.sin(arg) / 2)
        out_wave = (1 - (osc_shape / 2)) * square_wave * (1 + (osc_shape * tr.cos(arg)))
        return out_wave


class WavetableOsc(nn.Module):
    def __init__(
        self,
        sr: int,
        n_pos: Optional[int] = None,
        n_wt_samples: Optional[int] = None,
        wt: Optional[T] = None,
        aa_filter_n: Optional[int] = None,
        is_trainable: bool = True,
    ):
        super().__init__()
        self.sr = sr
        if wt is None:
            assert n_wt_samples is not None
            assert n_pos is not None
            wt = tr.empty(n_pos, n_wt_samples).normal_(mean=0.0, std=0.01)
            # wt = tr.empty(n_pos, n_wt_samples).uniform_() * 2.0 - 1.0
        else:
            assert wt.ndim == 2
            n_pos, n_wt_samples = wt.size()
        self.n_pos = n_pos
        self.n_wt_samples = n_wt_samples
        if aa_filter_n is None:
            assert n_wt_samples % 8 == 0
            # This is purely a heuristic
            aa_filter_n = n_wt_samples // 8 + 1
            log.info(
                f"Setting aa_filter_n = {aa_filter_n} "
                f"since n_wt_samples = {n_wt_samples}"
            )
        assert aa_filter_n % 2 == 1
        self.aa_filter_n = aa_filter_n
        self.is_trainable = is_trainable

        if is_trainable:
            self.wt = nn.Parameter(wt)
        else:
            self.register_buffer("wt", wt)
        aa_filter_support = 2 * (tr.arange(aa_filter_n) - (aa_filter_n - 1) / 2) / sr
        self.register_buffer("aa_filter_support", aa_filter_support.unsqueeze(0))
        self.register_buffer(
            "window", tr.blackman_window(aa_filter_n, periodic=False).unsqueeze(0)
        )
        # TODO(cm): check whether this is correct or not
        self.wt_pitch_hz = sr / n_wt_samples

    def get_wt(self) -> T:
        return self.wt

    def calc_lp_sinc_blackman_coeff(self, cf_hz: T) -> T:
        assert cf_hz.ndim == 2
        # Compute sinc filter.
        bs = cf_hz.size(0)
        support = self.aa_filter_support.expand(bs, -1)
        h = tr.sinc(cf_hz * support)
        # Apply window.
        window = self.window.expand(bs, -1)
        h *= window
        # Normalize to get unity gain.
        summed = tr.sum(h, dim=1, keepdim=True)
        h /= summed
        return h

    def get_anti_aliased_bounded_wt(self, max_f0_hz: T) -> T:
        wt = self.get_wt()
        bounded_wt = tr.tanh(wt)
        pitch_ratio = max_f0_hz / self.wt_pitch_hz
        # Make the center frequency a bit lower than the new nyquist
        # TODO(cm): add roll-off factor to config
        cf_hz = (self.sr / pitch_ratio / 2.0) * 0.9
        aa_filters = self.calc_lp_sinc_blackman_coeff(cf_hz)
        bs = aa_filters.size(0)
        aa_filters = aa_filters.unsqueeze(1).unsqueeze(1)
        # Put batch in channel dim to apply different kernel to each item in batch
        bounded_wt = bounded_wt.unsqueeze(0).unsqueeze(0)
        bounded_wt = bounded_wt.expand(1, bs, -1, -1)
        n_pad = self.aa_filter_n // 2
        padded_wt = F.pad(bounded_wt, (n_pad, n_pad, 0, 0), mode="circular")
        filtered_wt = F.conv2d(padded_wt, aa_filters, padding="valid", groups=bs)
        filtered_wt = tr.swapaxes(filtered_wt, 0, 1)
        filtered_wt = filtered_wt.squeeze(1)
        return filtered_wt

    def forward(
        self,
        f0_hz: T,
        wt_pos: T,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        assert 1 <= wt_pos.ndim <= 2
        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)
        if wt_pos.ndim == 1:
            assert n_samples is not None
            wt_pos = wt_pos.unsqueeze(1)
            wt_pos = wt_pos.expand(-1, n_samples)
        assert wt_pos.min() >= -1.0, f"wt_pos.min() = {wt_pos.min()}"
        assert wt_pos.max() <= 1.0, f"wt_pos.max() = {wt_pos.max()}"

        arg = SquareSawVCOLite.calc_osc_arg(self.sr, f0_hz, n_samples, phase)
        arg = arg % (2 * tr.pi)
        temp_coords = arg / tr.pi - 1.0  # Normalize to [-1, 1] for grid_sample
        assert temp_coords.min() >= -1.0
        assert temp_coords.max() <= 1.0
        flow_field = tr.stack([temp_coords, wt_pos], dim=2).unsqueeze(1)

        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
        wt = self.get_anti_aliased_bounded_wt(max_f0_hz)
        wt = wt.unsqueeze(1)

        audio = F.grid_sample(
            wt,
            flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        audio = audio.squeeze(1).squeeze(1)
        return audio


class WavetableOscShan(WavetableOsc):
    """
    Wavetable Oscillator from Differentiable Wavetable Synthesis, Shan et al.
    that uses weighted sum instead of grid_sample.
    """
    def __init__(
        self,
        sr: int,
        n_pos: Optional[int] = None,
        n_wt_samples: Optional[int] = None,
        wt: Optional[T] = None,
        aa_filter_n: Optional[int] = None,
        is_trainable: bool = True,
    ):
        super().__init__(sr, n_pos, n_wt_samples, wt, aa_filter_n, is_trainable)
    
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
        bs, n_wavetable, wt_len = wavetable.shape
        arg = SquareSawVCOLite.calc_osc_arg(sr, freq, n_samples, phase)
        arg = arg % (2 * tr.pi)
        index = arg / (2 * tr.pi) * wt_len 

        # batch linear interpolation implementation
        index_low = tr.floor(index.clone())                         # (bs, n_samples)
        index_high = tr.ceil(index.clone())                         # (bs, n_samples)
        alpha = index - index_low                                   # (bs, n_samples)
        index_low = index_low.long()
        index_high = index_high.long()

        index_low = index_low.unsqueeze(1).expand(-1, n_wavetable, -1)        # (bs, n_wavetable, n_samples)
        index_high = index_high.unsqueeze(1).expand(-1, n_wavetable, -1)      # (bs, n_wavetable, n_samples)
        index_high = index_high % wt_len
        alpha = alpha.unsqueeze(1).expand(-1, n_wavetable, -1)                # (bs, n_wavetable, n_samples)

        indexed_wavetables_low = tr.gather(wavetable, 2, index_low)           # (bs, n_wavetable, n_samples)
        indexed_wavetables_high = tr.gather(wavetable, 2, index_high)         # (bs, n_wavetable, n_samples)

        signal = indexed_wavetables_low + alpha * (indexed_wavetables_high - indexed_wavetables_low)
        return signal
    
    def forward(
        self,
        f0_hz: T,
        attention_matrix: Optional[T] = None,
        n_samples: Optional[int] = None,
        phase: Optional[T] = None,
    ) -> T:
        assert 1 <= f0_hz.ndim <= 2
        if f0_hz.ndim == 1:
            assert n_samples is not None
            f0_hz = f0_hz.unsqueeze(1)
            f0_hz = f0_hz.expand(-1, n_samples)
        
        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
        wt = self.get_anti_aliased_bounded_wt(max_f0_hz)

        audio = WavetableOscShan._wavetable_osc_shan(wt, f0_hz, self.sr, n_samples, phase)
        if attention_matrix is not None:
            # if attention matrix is provided, it must be of shape (bs, n_wavetables, n_samples)
            assert attention_matrix.ndim == 3
            assert attention_matrix.shape == audio.shape, f"Attention matrix must be the same as audio shape, given attention matrix: {attention_matrix.shape}, audio: {audio.shape}"
            audio = tr.einsum("bns,bns->bs", audio, attention_matrix)
        else:
            # if not attention matrix is provided, we simply take the mean of the wavetables
            audio = audio.mean(dim=1)

        return audio


class FourierWavetableOsc(WavetableOsc):
    def __init__(
        self,
        sr: int,
        n_pos: Optional[int] = None,
        n_wt_samples: Optional[int] = None,
        wt: Optional[T] = None,
        aa_filter_n: Optional[int] = None,
        is_trainable: bool = True,
        n_bins: Optional[int] = None,
    ):
        super().__init__(sr, n_pos, n_wt_samples, wt, aa_filter_n, is_trainable)
        if n_bins is None:
            n_bins = self.n_wt_samples // 2 + 1
            # n_bins = 32
        self.n_bins = n_bins
        # TODO(cm): check if normalization would be beneficial or not
        fourier_wt = tr.fft.rfft(self.wt, dim=1)
        fourier_wt = fourier_wt[:, :n_bins]
        # wt_mag = tr.abs(fourier_wt)
        # wt_phase = tr.angle(fourier_wt)
        self.wt = None  # Get rid of superclass param or buffer since we don't need it
        if is_trainable:
            self.wt = nn.Parameter(fourier_wt)
            # self.wt_mag = nn.Parameter(wt_mag)
            # self.wt_phase = nn.Parameter(wt_phase)
        else:
            self.register_buffer("wt", fourier_wt)
            # self.register_buffer("wt_mag", wt_mag)
            # self.register_buffer("wt_phase", wt_phase)

    def get_wt(self) -> T:
        # fourier_wt = self.wt_mag * tr.exp(1j * self.wt_phase)
        # wt = tr.fft.irfft(fourier_wt, n=self.n_wt_samples, dim=1)
        # TODO(cm): check if normalization would be beneficial or not
        wt = tr.fft.irfft(self.wt, n=self.n_wt_samples, dim=1)
        return wt


if __name__ == "__main__":
    # ============ Test WavetableOsc ============
    # bs = 2
    # sr = 48000
    # n_sec = 4.0
    # n_samples = int(sr * n_sec)
    # n_wt_samples = 1024
    # f0_hz = tr.tensor([220.0])
    # f0_hz = f0_hz.repeat(bs)
    #
    # # wt_pos = tr.linspace(-1.0, -1.0, n_samples).unsqueeze(0)
    # # wt_pos = tr.linspace(1.0, 1.0, n_samples).unsqueeze(0)
    # wt_pos = tr.linspace(-1.0, 1.0, n_samples).unsqueeze(0)
    # wt_pos = wt_pos.repeat(bs, 1)
    #
    # osc = WavetableOsc(sr, n_pos=2, n_wt_samples=n_wt_samples)
    # audio = osc(f0_hz, wt_pos, n_samples=n_samples)
    #
    # # import matplotlib.pyplot as plt
    # # plt.plot(audio[0, :1000].detach().numpy())
    # # plt.show()
    # # plt.plot(audio[0, -1000:].detach().numpy())
    # # plt.show()
    #
    # import torchaudio
    #
    # torchaudio.save("../out/audio.wav", audio[0:1, :], sr)
    # exit()

    # ============ Test WavetableOscShan ============
    # bs = 3
    # sr = 48000
    # n_sec = 4.0
    # n_samples = int(sr * n_sec)
    # n_wt_samples = 1024
    # f0_hz = tr.tensor([220.0, 440.0, 880.0])
    
    # # prepare some wavetables for test
    # wt_sin = tr.sin(tr.linspace(0, 2 * tr.pi, n_wt_samples))
    # wt_square = tr.sign(tr.sin(tr.linspace(0, 2 * tr.pi, n_wt_samples)))
    # wt = tr.stack([wt_sin, wt_square], dim=0)
    # osc = WavetableOscShan(sr, n_pos=2, n_wt_samples=n_wt_samples, wt=wt)
    # audio = osc(f0_hz=f0_hz, wt_pos=None, n_samples=n_samples)
    # audio = audio.reshape(-1, audio.size(-1))
    # import soundfile as sf
    # for idx, audio in enumerate(audio):
    #     import matplotlib.pyplot as plt
    #     plt.plot(audio.detach().numpy().squeeze()[:1000])
    #     sf.write(f"audio_{idx}.wav", audio.squeeze().detach().numpy(), sr)
    # plt.savefig("audio.png")
    # plt.close()

    note_off = tr.tensor([0.75, 0.75])
    attack = tr.tensor([0.1, 0.2])
    decay = tr.tensor([0.0, 0.2])
    sustain = tr.tensor([0.5, 0.3])
    release = tr.tensor([2.0, 0.2])
    # floor = tr.tensor([0.1, 0.2])
    # peak = tr.tensor([0.3, 0.5])
    pow = tr.tensor([1.0, 2.5])
    note_on_duration = tr.tensor([0.1, 1.0])
    n_samples = 10

    adsr = ADSR(100)
    envelope = adsr(note_off, attack, decay, sustain, release, pow=pow)

    # adsr = ADSREnvelope()
    # envelope = adsr(
    #     floor=tr.zeros_like(attack).view(-1, 1, 1),
    #     peak=tr.ones_like(attack).view(-1, 1, 1),
    #     attack=attack.view(-1, 1, 1),
    #     decay=decay.view(-1, 1, 1),
    #     sus_level=sustain.view(-1, 1, 1),
    #     release=release.view(-1, 1, 1),
    #     note_off=0.8,
    #     n_frames=100,
    # )

    # exp_decay = ExpDecayEnv(10, eps=1e-3)
    # envelope = exp_decay(alpha, note_on_duration, n_samples)

    import matplotlib.pyplot as plt

    plt.plot(envelope[0].numpy())
    plt.plot(envelope[1].numpy())
    plt.show()
    log.info(
        f"envelope.shape = {envelope.shape}, envelope.max(): {envelope.max()}, envelope.min(): {envelope.min()}"
    )
    exit()

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
