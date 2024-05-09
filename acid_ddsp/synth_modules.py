import logging
import os
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T, nn

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
    ):
        super().__init__()
        self.sr = sr
        if wt is None:
            assert n_wt_samples is not None
            assert n_pos is not None
            wt = tr.empty(n_pos, n_wt_samples).normal_(mean=0.0, std=0.01)
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

        # wt = tr.sin(tr.linspace(0.0, 2 * tr.pi, n_wt_samples)).view(1, 1, 1, -1)
        # import matplotlib.pyplot as plt
        # plt.plot(wt.squeeze().numpy())
        # plt.show()
        # wt_2 = tr.sin(tr.linspace(0.0, 4 * tr.pi, n_wt_samples)).view(1, 1, 1, -1)
        # wt_2 = tr.linspace(-1.0, 1.0, n_wt_samples).view(1, 1, 1, -1)
        # wt_2 = tr.linspace(-1.0, 1.0, n_wt_samples // 4).view(1, 1, 1, -1).repeat(1, 1, 1, 4)
        # plt.plot(wt_2.squeeze().numpy())
        # plt.show()
        # wt = tr.cat([wt, wt_2], dim=2)

        self.wt = nn.Parameter(wt.view(1, 1, n_pos, n_wt_samples))
        aa_filter_support = 2 * (tr.arange(aa_filter_n) - (aa_filter_n - 1) / 2) / sr
        self.register_buffer("aa_filter_support", aa_filter_support.unsqueeze(0))
        self.register_buffer(
            "window", tr.blackman_window(aa_filter_n, periodic=False).unsqueeze(0)
        )
        self.wt_pitch_hz = sr / n_wt_samples

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
        bounded_wt = tr.tanh(self.wt)
        pitch_ratio = max_f0_hz / self.wt_pitch_hz
        # Make the center frequency a bit lower than the new nyquist
        cf_hz = (self.sr / pitch_ratio / 2.0) * 0.9
        aa_filters = self.calc_lp_sinc_blackman_coeff(cf_hz)
        bs = aa_filters.size(0)
        aa_filters = aa_filters.unsqueeze(1).unsqueeze(1)
        # Put batch in channel dim to apply different kernel to each item in batch
        bounded_wt = bounded_wt.expand(1, bs, -1, -1)
        n_pad = self.aa_filter_n // 2
        padded_wt = F.pad(bounded_wt, (n_pad, n_pad, 0, 0), mode="circular")
        filtered_wt = F.conv2d(padded_wt, aa_filters, padding="valid", groups=bs)
        filtered_wt = tr.swapaxes(filtered_wt, 0, 1)
        return filtered_wt
        # return bounded_wt.swapaxes(0, 1)
        # return self.wt.expand(bs, -1, -1, -1)

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
        assert wt_pos.min() >= -1.0
        assert wt_pos.max() <= 1.0

        arg = SquareSawVCOLite.calc_osc_arg(self.sr, f0_hz, n_samples, phase)
        arg = arg % (2 * tr.pi)
        temp_coords = arg / tr.pi - 1.0  # Normalize to [-1, 1] for grid_sample
        assert temp_coords.min() >= -1.0
        assert temp_coords.max() <= 1.0
        flow_field = tr.stack([temp_coords, wt_pos], dim=2).unsqueeze(1)

        max_f0_hz = tr.max(f0_hz, dim=1, keepdim=True).values
        wt = self.get_anti_aliased_bounded_wt(max_f0_hz)

        # import matplotlib.pyplot as plt
        # plt.plot(self.wt[0, 0, 0, :].detach().numpy())
        # plt.plot(wt[0, 0, 0, :].detach().numpy())
        # plt.show()
        # plt.plot(self.wt[0, 0, 1, :].detach().numpy())
        # plt.plot(wt[0, 0, 1, :].detach().numpy())
        # plt.show()

        audio = F.grid_sample(
            wt,
            flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        audio = audio.squeeze(1).squeeze(1)
        return audio


if __name__ == "__main__":
    bs = 2
    sr = 48000
    n_sec = 4.0
    n_samples = int(sr * n_sec)
    n_wt_samples = 1024
    f0_hz = tr.tensor([220.0])
    f0_hz = f0_hz.repeat(bs)

    # wt_pos = tr.linspace(-1.0, -1.0, n_samples).unsqueeze(0)
    # wt_pos = tr.linspace(1.0, 1.0, n_samples).unsqueeze(0)
    wt_pos = tr.linspace(-1.0, 1.0, n_samples).unsqueeze(0)
    wt_pos = wt_pos.repeat(bs, 1)

    osc = WavetableOsc(sr, n_pos=2, n_wt_samples=n_wt_samples)
    audio = osc(f0_hz, wt_pos, n_samples=n_samples)

    # import matplotlib.pyplot as plt
    # plt.plot(audio[0, :1000].detach().numpy())
    # plt.show()
    # plt.plot(audio[0, -1000:].detach().numpy())
    # plt.show()

    import torchaudio

    torchaudio.save("../out/audio.wav", audio[0:1, :], sr)
    exit()

    adsr = ADSRLite(10, eps=1e-3)
    exp_decay = ExpDecayEnv(10, eps=1e-3)
    attack = tr.tensor([0.1, 0.2])
    decay = tr.tensor([0.1, 0.2])
    sustain = tr.tensor([0.2, 0.3])
    release = tr.tensor([0.1, 0.2])
    alpha = tr.tensor([1.0, 2.5])
    note_on_duration = tr.tensor([0.1, 1.0])
    n_samples = 10
    # envelope = adsr(attack, decay, sustain, release, alpha, note_on_duration, n_samples)
    envelope = exp_decay(alpha, note_on_duration, n_samples)
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
