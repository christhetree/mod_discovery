import logging
import os
from typing import Optional

import torch as tr
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
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
    order = b.size(2)
    x_padded = F.pad(x, (order - 1, 0))
    x_unfolded = x_padded.unfold(dimension=1, size=order, step=1)
    x_unfolded = x_unfolded.unsqueeze(3)
    b = b.unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y


class TimeVaryingBiquad(nn.Module):
    def __init__(
        self,
        min_w: float = 0.001,
        max_w: float = tr.pi,
        min_q: float = 0.7071,
        max_q: float = 4.0,
        stability_eps: float = 1e-3,
    ):
        super().__init__()
        assert 0.0 < min_w
        assert 0.0 < max_w <= 2 * tr.pi
        self.min_w = tr.tensor(min_w)
        self.max_w = tr.tensor(max_w)
        self.min_q = tr.tensor(min_q)
        self.max_q = tr.tensor(max_q)
        self.log_min_w = tr.log(self.min_w)
        self.log_max_w = tr.log(self.max_w)
        self.log_min_q = tr.log(self.min_q)
        self.log_max_q = tr.log(self.max_q)
        self.stability_eps = stability_eps

    def _calc_coeffs(self, mod_sig_w: T, mod_sig_q: T) -> (T, T):
        log_w = self.log_min_w + (self.log_max_w - self.log_min_w) * mod_sig_w
        w = tr.exp(log_w)
        log_q = self.log_min_q + (self.log_max_q - self.log_min_q) * mod_sig_q
        q = tr.exp(log_q)

        alpha_q = tr.sin(w) / (2 * q)

        a0 = 1.0 + alpha_q
        a1 = -2.0 * tr.cos(w)
        a1 /= a0
        # a1 = (1.0 - self.stability_eps) * a1
        a2 = 1.0 - alpha_q
        a2 /= a0
        # a2 = (1.0 - self.stability_eps) * a2
        a0.fill_(1.0)
        assert (a1.abs() < 2.0).all()
        assert (a2 < 1.0).all()
        assert (a1 < a2 + 1.0).all()
        assert (a1 > -(a2 + 1.0)).all()

        a = tr.stack([a0, a1, a2], dim=2)

        b0 = (1.0 - tr.cos(w)) / 2.0
        b1 = 1.0 - tr.cos(w)
        b2 = (1.0 - tr.cos(w)) / 2.0
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
        n_inf = tr.isinf(y_a).sum()
        log.info(f"n_inf: {n_inf}")
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        # y_ab = time_varying_fir(x, b_coeffs)
        # return y_ab
        return y_a


if __name__ == "__main__":
    # q = 4.071
    # for w in tr.linspace(1e-3, tr.pi - 1e-3, 100):
    #     alpha_q = tr.sin(w) / (2 * q)
    #     a0 = 1.0 + alpha_q
    #     a1 = -2.0 * tr.cos(w)
    #     a2 = 1.0 - alpha_q
    #     a1 /= a0
    #     a2 /= a0
    #     # k = tr.tan(theta / 2)
    #     # w = k**2
    #     # alpha_q = 1.0 + (k / q) + w
    #     # a0 = 1.0
    #     # a1 = 2.0 * (w - 1.0) / alpha_q
    #     # a2 = (1.0 - (k / q) + w) / alpha_q
    #     a1p2 = a1.abs() + a2
    #     # w = theta
    #     print(
    #         f"w: {w:.3f}, "
    #         f"alpha_q: {alpha_q:.3f}, "
    #         f"a1: {a1:.3f}, "
    #         f"a2: {a2:.3f}, "
    #         f"a1p2: {a1p2:.3f}, "
    #         f"{a2 < 1.0}, "
    #         f"{a1 < a2 + 1.0}, "
    #         f"{a1 > -(a2 + 1.0)}"
    #     )

    tr.manual_seed(42)
    sr = 4000
    bs = 1
    min_f = 1000.0
    max_f = 1000.0
    min_q = 0.7071
    max_q = 0.7071

    min_w = 2 * tr.pi * min_f / sr
    max_w = 2 * tr.pi * max_f / sr
    tvb = TimeVaryingBiquad(min_w, max_w, min_q, max_q)
    lfo = tr.linspace(0.0, 1.0, sr).unsqueeze(0).repeat(bs, 1).double()

    n_samples = sr
    white_noise = tr.randn((bs, n_samples)).double()

    x = white_noise
    log.info("start")
    y = tvb(x, cutoff_mod_sig=lfo)
    log.info(f"y.shape: {y.shape}")

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

    # mag_response = tr.fft.rfft(y[0]).abs()
    # mag_response = mag_response / mag_response.max()
    # mag_response_db = 20 * tr.log10(mag_response)
    #
    # plt.plot(mag_response_db.detach().numpy())
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
