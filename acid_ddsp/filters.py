import logging
import os
from typing import Optional

import torch as tr
from matplotlib import pyplot as plt
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F

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
        self.min_q_db = 20 * tr.log10(self.min_q)
        self.max_q_db = 20 * tr.log10(self.max_q)
        self.tv_fir = TimeVaryingFIR(order=3)

    def _calc_coeffs(self, mod_sig_w: T, mod_sig_q: T) -> (T, T):
        log_w = self.log_min_w + (self.log_max_w - self.log_min_w) * mod_sig_w
        w = tr.exp(log_w)
        q_db = self.min_q_db + (self.max_q_db - self.min_q_db) * mod_sig_q
        # TODO(cm): check which q to use
        alpha_q = tr.sin(w) / (2 * q_db)

        a0 = 1.0 + alpha_q
        a1 = -2.0 * tr.cos(w)
        a2 = 1.0 - alpha_q
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
        # y_a = sample_wise_lpc(x, a_coeffs)
        # y_ab = self.tv_fir(y_a, b_coeffs)
        y_ab = self.tv_fir(x, b_coeffs)
        return y_ab


if __name__ == "__main__":
    tr.manual_seed(42)
    sr = 2000
    bs = 1
    # min_f = 100.0
    # max_f = 4000.0

    # min_w = 2 * tr.pi * min_f / sr
    # max_w = 2 * tr.pi * max_f / sr
    # tvb = TimeVaryingBiquad(min_w, max_w)
    # lfo = tr.linspace(0.0, 1.0, sr).unsqueeze(0).repeat(bs, 1)

    n_samples = sr
    white_noise = tr.randn((bs, n_samples))

    # x = white_noise
    # log.info("start")
    # y = tvb(x, cutoff_mod_sig=lfo)
    # log.info(f"y.shape: {y.shape}")

    b_coeff_raw = [
        -0.15301418463641955,
        0.09979048504546387,
        0.35259515472734737,
        0.4839024312338306,
        0.35259515472734737,
        0.09979048504546387,
        -0.15301418463641955,
    ]
    b_coeff = tr.tensor(b_coeff_raw).view(1, 1, -1).repeat(bs, n_samples, 1)
    y = time_varying_fir(white_noise, b_coeff)

    mag_response = tr.fft.rfft(y[0]).abs()
    # mag_response = tr.fft.fft(y[0]).abs() ** 2
    mag_response = mag_response / mag_response.max()
    mag_response_db = 20 * tr.log10(mag_response)

    plt.plot(mag_response_db.detach().numpy())
    plt.show()

    derp = 1

