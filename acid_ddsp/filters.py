import logging
import math
import os
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

import util
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def time_varying_fir(x: T, b: T) -> T:
    assert x.ndim == 2
    assert b.ndim == 3
    assert x.size(0) == b.size(0)
    assert x.size(1) == b.size(1)
    order = b.size(2) - 1
    x_padded = F.pad(x, (order, 0))
    x_unfolded = x_padded.unfold(dimension=1, size=order + 1, step=1)
    x_unfolded = x_unfolded.unsqueeze(3)
    b = tr.flip(b, dims=[2])  # Go from correlation to convolution
    b = b.unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y


def calc_logits_to_biquad_a_coeff_triangle(a_logits: T, eps: float = 1e-3) -> T:
    assert a_logits.size(-1) == 2
    assert not tr.isnan(a_logits).any()
    stability_factor = 1.0 - eps
    a1_logits = a_logits[..., 0]
    a2_logits = a_logits[..., 1]
    a1 = 2 * tr.tanh(a1_logits) * stability_factor
    a1_abs = a1.abs()
    a2 = (((2 - a1_abs) * tr.tanh(a2_logits) * stability_factor) + a1_abs) / 2
    assert (a1.abs() < 2.0).all(), f"a1.abs().max() = {a1.abs().max()}"
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)
    return a


def calc_logits_to_biquad_coeff_pole_zero(
    q_real: T, q_imag: T, p_real: T, p_imag: T, eps: float = 1e-3
) -> (T, T):
    assert q_real.ndim == 2
    assert q_real.shape == q_imag.shape == p_real.shape == p_imag.shape
    stability_factor = 1.0 - eps
    p_abs = tr.sqrt(p_real**2 + p_imag**2)
    p_scaling_factor = tr.tanh(p_abs) * stability_factor / p_abs
    p_real = p_real * p_scaling_factor
    p_imag = p_imag * p_scaling_factor

    a1 = -2.0 * p_real
    a2 = p_real**2 + p_imag**2
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)

    b0 = tr.ones_like(q_real)
    b1 = -2.0 * q_real
    b2 = q_real**2 + q_imag**2
    b = tr.stack([b0, b1, b2], dim=2)

    return a, b


def calc_lp_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T):
    assert w.ndim == 2
    assert q.ndim == 2
    assert 0.0 <= w.min()
    assert tr.pi >= w.max()
    assert 0.0 < q.min()

    stability_factor = 1.0 - eps
    alpha_q = tr.sin(w) / (2 * q)
    a0 = 1.0 + alpha_q
    a1 = -2.0 * tr.cos(w) * stability_factor
    a1 = a1 / a0
    a2 = (1.0 - alpha_q) * stability_factor
    a2 = a2 / a0
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)

    b0 = (1.0 - tr.cos(w)) / 2.0
    b0 = b0 / a0
    b1 = 1.0 - tr.cos(w)
    b1 = b1 / a0
    b2 = (1.0 - tr.cos(w)) / 2.0
    b2 = b2 / a0
    b = tr.stack([b0, b1, b2], dim=2)

    return a, b


class TimeVaryingIIRFSM(nn.Module):
    def __init__(
        self,
        win_len: Optional[int] = None,
        win_len_sec: Optional[float] = None,
        sr: Optional[float] = None,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
    ):
        super().__init__()
        assert 0.0 < overlap < 1.0
        self.sr = sr
        self.overlap = overlap
        self.oversampling_factor = oversampling_factor

        hops_per_frame = int(1.0 / (1.0 - overlap))
        if win_len is None:
            assert win_len_sec is not None
            assert sr is not None
            self.hop_len = int(win_len_sec * sr / hops_per_frame)
            self.win_len = hops_per_frame * self.hop_len
        else:
            assert win_len_sec is None
            assert win_len % hops_per_frame == 0
            self.win_len = win_len
            self.hop_len = win_len // hops_per_frame

        assert self.hop_len == 32  # TODO(cm): tmp
        self.n_fft = 2 ** (math.ceil(math.log2(self.win_len)) + oversampling_factor)
        self.register_buffer("hann", tr.hann_window(self.win_len, periodic=True))
        log.info(
            f"win_len: {self.win_len}, hop_len: {self.hop_len}, n_fft: {self.n_fft}"
        )

    def calc_n_frames(self, n_samples: int) -> int:
        n_frames = (n_samples // self.hop_len) + 1
        return n_frames

    def forward(self, x: T, a: T, b: T) -> T:
        assert x.ndim == 2
        assert a.ndim == 3
        assert b.ndim == 3
        x_n_samples = x.size(1)
        X = tr.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.hann,
            center=True,
            pad_mode="constant",
            onesided=True,
            return_complex=True,
        )

        n_frames = X.size(2)
        assert a.size(1) == n_frames
        assert b.size(1) == n_frames

        A = tr.fft.rfft(a, self.n_fft)
        B = tr.fft.rfft(b, self.n_fft)
        H = B / A  # TODO(cm): Make more stable
        H = H.swapaxes(1, 2)
        Y = X * H

        y = tr.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.hann,
            center=True,
            onesided=True,
            length=x_n_samples,
        )
        return y


class TimeVaryingLPBiquad(nn.Module):
    def __init__(
        self,
        min_w: float = 0.0,
        max_w: float = tr.pi,
        min_q: float = 0.7071,
        max_q: float = 4.0,
        eps: float = 1e-3,
        modulate_log_w: bool = True,
        modulate_log_q: bool = True,
    ):
        super().__init__()
        assert 0.0 <= min_w <= max_w <= tr.pi
        assert 0.0 < min_q <= max_q
        self.min_w = tr.tensor(min_w)
        self.max_w = tr.tensor(max_w)
        self.min_q = tr.tensor(min_q)
        self.max_q = tr.tensor(max_q)
        self.log_min_w = tr.log(self.min_w)
        self.log_max_w = tr.log(self.max_w)
        self.log_min_q = tr.log(self.min_q)
        self.log_max_q = tr.log(self.max_q)
        self.eps = eps
        self.modulate_log_w = modulate_log_w
        self.modulate_log_q = modulate_log_q

    def calc_w_and_q(
        self, x: T, w_mod_sig: Optional[T] = None, q_mod_sig: Optional[T] = None
    ) -> (T, T):
        if w_mod_sig is None:
            w_mod_sig = tr.zeros_like(x)
        if q_mod_sig is None:
            q_mod_sig = tr.zeros_like(x)

        assert x.ndim == 2
        assert w_mod_sig.ndim == 2
        assert w_mod_sig.min() >= 0.0
        assert w_mod_sig.max() <= 1.0
        assert q_mod_sig.ndim == 2
        assert q_mod_sig.min() >= 0.0
        assert q_mod_sig.max() <= 1.0

        if self.modulate_log_w:
            log_w = self.log_min_w + (self.log_max_w - self.log_min_w) * w_mod_sig
            w = tr.exp(log_w)
        else:
            w = self.min_w + (self.max_w - self.min_w) * w_mod_sig

        if self.modulate_log_q:
            log_q = self.log_min_q + (self.log_max_q - self.log_min_q) * q_mod_sig
            q = tr.exp(log_q)
        else:
            q = self.min_q + (self.max_q - self.min_q) * q_mod_sig

        return w, q

    def forward(
        self,
        x: T,
        w_mod_sig: Optional[T] = None,
        q_mod_sig: Optional[T] = None,
        interp_coeff: bool = False,
    ) -> T:
        w, q = self.calc_w_and_q(x, w_mod_sig, q_mod_sig)
        n_samples = x.size(1)
        if not interp_coeff:
            assert False  # TODO(cm): tmp
            w = util.linear_interpolate_dim(w, n_samples, dim=1, align_corners=True)
            q = util.linear_interpolate_dim(q, n_samples, dim=1, align_corners=True)
            assert x.shape == w.shape == q.shape
        a_coeffs, b_coeffs = calc_lp_biquad_coeff(w, q, eps=self.eps)
        if interp_coeff:
            a_coeffs = util.linear_interpolate_dim(
                a_coeffs, n_samples, dim=1, align_corners=True
            )
            b_coeffs = util.linear_interpolate_dim(
                b_coeffs, n_samples, dim=1, align_corners=True
            )
        y_a = sample_wise_lpc(x, a_coeffs)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeffs)
        return y_ab


class TimeVaryingLPBiquadFSM(TimeVaryingLPBiquad):
    def __init__(
        self,
        win_len: Optional[int] = None,
        win_len_sec: Optional[float] = None,
        sr: Optional[float] = None,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
        min_w: float = 0.0,
        max_w: float = tr.pi,
        min_q: float = 0.7071,
        max_q: float = 4.0,
        eps: float = 1e-3,
        modulate_log_w: bool = True,
        modulate_log_q: bool = True,
    ):
        super().__init__(
            min_w=min_w,
            max_w=max_w,
            min_q=min_q,
            max_q=max_q,
            eps=eps,
            modulate_log_w=modulate_log_w,
            modulate_log_q=modulate_log_q,
        )
        self.filter = TimeVaryingIIRFSM(
            win_len=win_len,
            win_len_sec=win_len_sec,
            sr=sr,
            overlap=overlap,
            oversampling_factor=oversampling_factor,
        )

    def forward(
        self,
        x: T,
        w_mod_sig: Optional[T] = None,
        q_mod_sig: Optional[T] = None,
        interp_coeff: bool = False,
    ) -> T:
        w, q = self.calc_w_and_q(x, w_mod_sig, q_mod_sig)
        n_samples = x.size(1)
        n_frames = self.filter.calc_n_frames(n_samples)
        if not interp_coeff:
            assert False  # TODO(cm): tmp
            w = util.linear_interpolate_dim(w, n_frames, dim=1, align_corners=True)
            q = util.linear_interpolate_dim(q, n_frames, dim=1, align_corners=True)
        a_coeffs, b_coeffs = calc_lp_biquad_coeff(w, q, eps=self.eps)
        if interp_coeff:
            a_coeffs = util.linear_interpolate_dim(
                a_coeffs, n_frames, dim=1, align_corners=True
            )
            b_coeffs = util.linear_interpolate_dim(
                b_coeffs, n_frames, dim=1, align_corners=True
            )
        a1 = a_coeffs[:, :, 0]
        a2 = a_coeffs[:, :, 1]
        a0 = tr.ones_like(a1)
        a_coeffs = tr.stack([a0, a1, a2], dim=2)
        y = self.filter(x, a_coeffs, b_coeffs)
        return y


if __name__ == "__main__":
    # sr = 48000
    # win_len_sec = 0.1
    # filter = TimeVaryingIIRFSM(sr, win_len_sec)
    # audio = tr.rand(1, sr)
    # a = tr.rand(1, 41, 3)
    # b = tr.rand(1, 41, 3)
    # y = filter(audio, a, b)
    # exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    if tr.cuda.is_available():
        log.info("Using GPU")
        device = tr.device("cuda")
    else:
        log.info("Using CPU")
        device = tr.device("cpu")

    tr.manual_seed(42)
    sr = 16000
    bs = 1
    # min_f = 100.0
    # max_f = 500.0
    # min_q = 4.7071
    # max_q = 4.7071

    # min_w = 2 * tr.pi * min_f / sr
    # max_w = 2 * tr.pi * max_f / sr
    # tvb = TimeVaryingBiquad(min_w, max_w, min_q, max_q)
    # tvb.to(device)

    n_samples = sr
    # n_samples = 2 ** 14
    white_noise = tr.rand((bs, n_samples)) * 2.0 - 1.0
    sin_wave_400hz = tr.sin(
        2 * tr.pi * tr.linspace(0.0, 1.0, n_samples).unsqueeze(0).repeat(bs, 1) * 400
    )
    # lfo = tr.linspace(0.0, 1.0, n_samples).unsqueeze(0).repeat(bs, 1)
    # lfo = tr.linspace(1.0, 0.0, n_samples).unsqueeze(0).repeat(bs, 1)
    # lfo = None

    x = white_noise
    # log.info("start")
    # x.to(device)
    # y = tvb(x, cutoff_mod_sig=lfo, resonance_mod_sig=lfo)
    # log.info(f"y.shape: {y.shape}")
    # log.info(f"y[0, :4] = {y[0, :4]}")
    # torchaudio.save("../out/tmp.wav", y, sr)

    # spec_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512)
    # spec = spec_transform(y)
    # log_spec = tr.log10(spec[0] + 1e-9)
    # plt.imshow(
    #     log_spec,
    #     aspect="auto",
    #     origin="lower",
    #     cmap="viridis",
    # )
    # plt.show()

    # mag_response = tr.fft.rfft(y[0]).abs()
    # log.info(f"mag_response.mean() = {mag_response.mean()}")
    # log.info(f"mag_response.std() = {mag_response.std()}")
    # mag_response = mag_response / mag_response.max()
    # mag_response_db = 20 * tr.log10(mag_response)
    # plt.plot(mag_response_db.numpy())
    # plt.show()

    # b_coeff_raw = [
    #     -0.15301418463641955,
    #     0.09979048504546387,
    #     0.35259515472734737,
    #     0.4839024312338306,
    #     0.35259515472734737,
    #     0.09979048504546387,
    #     -0.15301418463641955,
    # ]
    a_coeff_raw = list([1.0, 0.5, 0.1])
    a_coeff_raw = [float(x) for x in a_coeff_raw]
    a_coeff_raw[0] = 1.0
    print(f"a_coeff_raw = {a_coeff_raw}")
    b_coeff_raw = list([1.0, 0.3, 0.4])
    # b_coeff_raw = list(range(3))
    # b_coeff_raw = [float(x) for x in b_coeff_raw]
    # b_coeff_raw = b_coeff_raw[::-1]
    # print(f"b_coeff_raw = {b_coeff_raw}")
    a_coeff = tr.tensor(a_coeff_raw).view(1, 1, -1).repeat(bs, n_samples, 1)
    b_coeff = tr.tensor(b_coeff_raw).view(1, 1, -1).repeat(bs, n_samples, 1)
    y_a = sample_wise_lpc(white_noise, a_coeff[:, :, 1:])
    y = time_varying_fir(y_a, b_coeff)

    fsm = TimeVaryingIIRFSM(
        win_len=n_samples, sr=sr, oversampling_factor=1, overlap=0.90
    )
    n_frames = fsm.calc_n_frames(n_samples)
    print(f"n_frames = {n_frames}")
    a_coeff = tr.tensor(a_coeff_raw).view(1, 1, -1).repeat(bs, n_frames, 1)
    # b_coeff_raw = list([1.0, 0.0, 0.0])
    # b_coeff_raw = b_coeff_raw[::-1]
    b_coeff = tr.tensor(b_coeff_raw).view(1, 1, -1).repeat(bs, n_frames, 1)
    y_fsm = fsm(white_noise, a_coeff, b_coeff)
    assert y.shape == y_fsm.shape
    assert tr.allclose(y, y_fsm, atol=1e-3)
    # print(f"y.avg = {y.mean():.4f}")
    exit()

    import torchaudio
    import matplotlib.pyplot as plt

    spec_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512)
    spec = spec_transform(y)
    log_spec = tr.log10(spec[0] + 1e-9)
    plt.imshow(
        log_spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.show()

    mag_response = tr.fft.rfft(y[0]).abs()
    log.info(f"mag_response.mean() = {mag_response.mean()}")
    log.info(f"mag_response.std() = {mag_response.std()}")
    mag_response = mag_response / mag_response.max()
    # plt.plot(mag_response.numpy())
    mag_response_db = 20 * tr.log10(mag_response)
    plt.plot(mag_response_db.numpy())
    plt.ylim(-60, 0)
    plt.show()
