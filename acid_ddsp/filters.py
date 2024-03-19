import logging
import os
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

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
    b = b.unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y


def calc_logits_to_biquad_a_coeff_triangle(a_logits: T, eps: float = 1e-3) -> T:
    assert a_logits.size(-1) == 2
    stability_factor = 1.0 - eps
    a1_logits = a_logits[..., 0]
    a2_logits = a_logits[..., 1]
    a1 = 2 * tr.tanh(a1_logits) * stability_factor
    a1_abs = a1.abs()
    a2 = (((2 - a1_abs) * tr.tanh(a2_logits) * stability_factor) + a1_abs) / 2
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)
    return a


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
    a1 /= a0
    a2 = (1.0 - alpha_q) * stability_factor
    a2 /= a0
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)

    b0 = (1.0 - tr.cos(w)) / 2.0
    b0 /= a0
    b1 = 1.0 - tr.cos(w)
    b1 /= a0
    b2 = (1.0 - tr.cos(w)) / 2.0
    b2 /= a0
    b = tr.stack([b0, b1, b2], dim=2)

    return a, b


class TimeVaryingBiquad(nn.Module):
    def __init__(
        self,
        min_w: float = 0.0,
        max_w: float = tr.pi,
        min_q: float = 0.7071,
        max_q: float = 4.0,
        stability_eps: float = 1e-3,
        modulate_log_w: bool = True,
        modulate_log_q: bool = True,
    ):
        super().__init__()
        assert 0.0 <= min_w <= max_w <= tr.pi
        assert 0.0 < min_q <= max_q
        self.min_w = tr.tensor(min_w).clamp(stability_eps, tr.pi - stability_eps)
        self.max_w = tr.tensor(max_w).clamp(stability_eps, tr.pi - stability_eps)
        self.min_q = tr.tensor(min_q).clamp_min(stability_eps)
        self.max_q = tr.tensor(max_q).clamp_min(stability_eps)
        self.log_min_w = tr.log(self.min_w)
        self.log_max_w = tr.log(self.max_w)
        self.log_min_q = tr.log(self.min_q)
        self.log_max_q = tr.log(self.max_q)
        self.stability_eps = stability_eps
        self.modulate_log_w = modulate_log_w
        self.modulate_log_q = modulate_log_q
        log.info(f"modulate_log_w = {self.modulate_log_w}")
        log.info(f"modulate_log_q = {self.modulate_log_q}")

    def _calc_coeffs(self, mod_sig_w: T, mod_sig_q: T) -> (T, T):
        if self.modulate_log_w:
            log_w = self.log_min_w + (self.log_max_w - self.log_min_w) * mod_sig_w
            w = tr.exp(log_w)
        else:
            w = self.min_w + (self.max_w - self.min_w) * mod_sig_w

        if self.modulate_log_q:
            log_q = self.log_min_q + (self.log_max_q - self.log_min_q) * mod_sig_q
            q = tr.exp(log_q)
        else:
            q = self.min_q + (self.max_q - self.min_q) * mod_sig_q

        alpha_q = tr.sin(w) / (2 * q)

        a0 = 1.0 + alpha_q
        a1 = -2.0 * tr.cos(w)
        a1 /= a0
        # a1 = (1.0 - self.stability_eps) * a1
        a2 = 1.0 - alpha_q
        a2 /= a0
        # a2 = (1.0 - self.stability_eps) * a2
        assert (a1.abs() < 2.0).all()
        assert (a2 < 1.0).all()
        assert (a1 < a2 + 1.0).all()
        assert (a1 > -(a2 + 1.0)).all()
        # log.info(
        #     f"a0[0,0] = {a0[0, 0]:.4f}, a1[0, 0] = {a1[0, 0]:.4f}, a2[0, 0] = {a2[0, 0]:.4f}"
        # )
        # log.info(f"\n{a0[0, 0]:.4f}\n{a1[0, 0]:.4f}\n{a2[0, 0]:.4f}")
        a = tr.stack([a1, a2], dim=2)

        b0 = (1.0 - tr.cos(w)) / 2.0
        b0 /= a0
        b1 = 1.0 - tr.cos(w)
        b1 /= a0
        b2 = (1.0 - tr.cos(w)) / 2.0
        b2 /= a0
        # log.info(
        #     f"b0[0,0] = {b0[0, 0]:.4f}, b1[0, 0] = {b1[0, 0]:.4f}, b2[0, 0] = {b2[0, 0]:.4f}"
        # )
        # log.info(f"\n{b0[0, 0]:.4f}\n{b1[0, 0]:.4f}\n{b2[0, 0]:.4f}")
        b = tr.stack([b0, b1, b2], dim=2)

        return a, b

    def forward(
        self,
        x: T,
        cutoff_mod_sig: Optional[T] = None,
        resonance_mod_sig: Optional[T] = None,
    ) -> T:
        if cutoff_mod_sig is None:
            cutoff_mod_sig = tr.zeros_like(x)
        if resonance_mod_sig is None:
            resonance_mod_sig = tr.zeros_like(x)

        assert x.ndim == 2
        assert cutoff_mod_sig.shape == x.shape
        assert cutoff_mod_sig.size(1) == x.size(1)
        assert cutoff_mod_sig.min() >= 0.0
        assert cutoff_mod_sig.max() <= 1.0
        assert resonance_mod_sig.shape == x.shape
        assert resonance_mod_sig.size(1) == x.size(1)
        assert resonance_mod_sig.min() >= 0.0
        assert resonance_mod_sig.max() <= 1.0
        a_coeffs, b_coeffs = self._calc_coeffs(cutoff_mod_sig, resonance_mod_sig)
        y_a = sample_wise_lpc(x, a_coeffs)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeffs)
        return y_ab


if __name__ == "__main__":
    sr = 48000
    win_len_sec = 0.1
    filter = TimeVaryingIIRFSM(sr, win_len_sec)
    audio = tr.rand(1, sr)
    a = tr.rand(1, 41, 3)
    b = tr.rand(1, 41, 3)
    y = filter(audio, a, b)
    exit()

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
    min_f = 100.0
    max_f = 500.0
    min_q = 4.7071
    max_q = 4.7071

    min_w = 2 * tr.pi * min_f / sr
    max_w = 2 * tr.pi * max_f / sr
    tvb = TimeVaryingBiquad(min_w, max_w, min_q, max_q)
    tvb.to(device)

    n_samples = sr
    white_noise = tr.rand((bs, n_samples)) * 2.0 - 1.0
    lfo = tr.linspace(0.0, 1.0, n_samples).unsqueeze(0).repeat(bs, 1)
    # lfo = tr.linspace(1.0, 0.0, n_samples).unsqueeze(0).repeat(bs, 1)
    # lfo = None

    x = white_noise
    log.info("start")
    x.to(device)
    y = tvb(x, cutoff_mod_sig=lfo, resonance_mod_sig=lfo)
    log.info(f"y.shape: {y.shape}")
    log.info(f"y[0, :4] = {y[0, :4]}")
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
    # b_coeff = tr.tensor(b_coeff_raw).view(1, 1, -1).repeat(bs, n_samples, 1)
    # y = time_varying_fir(white_noise, b_coeff)
