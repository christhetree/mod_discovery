import logging
import os
from typing import Optional, Tuple, Literal

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

import util
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def time_varying_fir(x: T, b: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert b.ndim == 3
    assert x.size(0) == b.size(0)
    assert x.size(1) == b.size(1)
    order = b.size(2) - 1
    x_padded = F.pad(x, (order, 0))
    if zi is not None:
        assert zi.shape == (x.size(0), order)
        x_padded[:, :order] = zi
    x_unfolded = x_padded.unfold(dimension=1, size=order + 1, step=1)
    x_unfolded = x_unfolded.unsqueeze(3)
    b = tr.flip(b, dims=[2])  # Go from correlation to convolution
    b = b.unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y


def sample_wise_lpc_scriptable(x: T, a: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert a.ndim == 3
    assert x.size(0) == a.size(0)
    assert x.size(1) == a.size(1)

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    padded_y = tr.empty((B, T + order), dtype=x.dtype)
    zi = tr.flip(zi, dims=[1])
    padded_y[:, :order] = zi
    padded_y[:, order:] = x
    a_flip = tr.flip(a, dims=[2])

    for t in range(T):
        padded_y[:, t + order] -= (
            a_flip[:, t : t + 1] @ padded_y[:, t : t + order, None]
        )[:, 0, 0]

    return padded_y[:, order:]


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
) -> Tuple[T, T]:
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


def _calc_a_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T, T, T, T, T):
    stability_factor = 1.0 - eps
    sin_w = tr.sin(w)
    cos_w = tr.cos(w)
    alpha_q = sin_w / (2 * q)
    a0 = 1.0 + alpha_q
    a1 = -2.0 * cos_w * stability_factor
    a2 = (1.0 - alpha_q) * stability_factor
    return a0, a1, a2, sin_w, cos_w, alpha_q


def _calc_lp_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T, T, T, T, T):
    a0, a1, a2, sin_w, cos_w, alpha_q = _calc_a_coeff(w, q, eps)
    b0 = (1.0 - cos_w) / 2.0
    b1 = 1.0 - cos_w
    b2 = (1.0 - cos_w) / 2.0
    return a0, a1, a2, b0, b1, b2


def _calc_hp_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T, T, T, T, T):
    a0, a1, a2, sin_w, cos_w, alpha_q = _calc_a_coeff(w, q, eps)
    b0 = (1.0 + cos_w) / 2.0
    b1 = -1.0 - cos_w
    b2 = (1.0 + cos_w) / 2.0
    return a0, a1, a2, b0, b1, b2


def _calc_bp_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T, T, T, T, T):
    a0, a1, a2, sin_w, cos_w, alpha_q = _calc_a_coeff(w, q, eps)
    b0 = alpha_q
    b1 = tr.zeros_like(w)
    b2 = -alpha_q
    # b0 = sin_w / 2.0
    # b1 = tr.zeros_like(w)
    # b2 = -sin_w / 2.0
    return a0, a1, a2, b0, b1, b2


def _calc_no_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T, T, T, T, T):
    a0, a1, a2, sin_w, cos_w, alpha_q = _calc_a_coeff(w, q, eps)
    b0 = tr.ones_like(w)
    b1 = -2.0 * cos_w
    b2 = tr.ones_like(w)
    return a0, a1, a2, b0, b1, b2


def calc_biquad_coeff(
    filter_type: Literal["lp", "hp", "bp", "no"], w: T, q: T, eps: float = 1e-3
) -> Tuple[T, T]:
    assert w.ndim == 2
    assert q.ndim == 2
    assert 0.0 <= w.min()
    assert tr.pi >= w.max()
    assert 0.0 < q.min()
    if filter_type == "lp":
        coeff_fn = _calc_lp_biquad_coeff
    elif filter_type == "hp":
        coeff_fn = _calc_hp_biquad_coeff
    elif filter_type == "bp":
        coeff_fn = _calc_bp_biquad_coeff
    elif filter_type == "no":
        coeff_fn = _calc_no_biquad_coeff
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    a0, a1, a2, b0, b1, b2 = coeff_fn(w, q, eps)
    assert a0.abs().min() > 0.0
    a1 = a1 / a0
    a2 = a2 / a0
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = tr.stack([a1, a2], dim=2)
    b0 = b0 / a0
    b1 = b1 / a0
    b2 = b2 / a0
    b = tr.stack([b0, b1, b2], dim=2)
    return a, b


class TimeVaryingBiquad(nn.Module):
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
        self.is_scriptable = False
        self.lpc_func = sample_wise_lpc

    def toggle_scriptable(self, is_scriptable: bool) -> None:
        self.is_scriptable = is_scriptable
        if is_scriptable:
            self.lpc_func = sample_wise_lpc_scriptable
        else:
            self.lpc_func = sample_wise_lpc

    def calc_w_and_q(
        self, x: T, w_mod_sig: Optional[T] = None, q_mod_sig: Optional[T] = None
    ) -> Tuple[T, T]:
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
        filter_type: Literal["lp", "hp", "bp", "no"],
        w_mod_sig: Optional[T] = None,
        q_mod_sig: Optional[T] = None,
        interp_coeff: bool = False,
        zi: Optional[T] = None,
    ) -> Tuple[T, T, T, Optional[T]]:
        w, q = self.calc_w_and_q(x, w_mod_sig, q_mod_sig)
        n_samples = x.size(1)
        if not interp_coeff:
            w = util.interpolate_dim(w, n_samples, dim=1)
            q = util.interpolate_dim(q, n_samples, dim=1)
            assert x.shape == w.shape == q.shape
        a_coeff, b_coeff = calc_biquad_coeff(filter_type, w, q, eps=self.eps)

        # a1 = a_coeff[0, :, 0].detach().numpy()
        # a1 = (a1 - a1.min()) / (a1.max() - a1.min())
        # a2 = a_coeff[0, :, 1].detach().numpy()
        # a2 = (a2 - a2.min()) / (a2.max() - a2.min())
        # b0 = b_coeff[0, :, 0].detach().numpy()
        # b0 = (b0 - b0.min()) / (b0.max() - b0.min())
        # b1 = b_coeff[0, :, 1].detach().numpy()
        # b1 = (b1 - b1.min()) / (b1.max() - b1.min())
        # b2 = b_coeff[0, :, 2].detach().numpy()
        # b2 = (b2 - b2.min()) / (b2.max() - b2.min())
        # mog_sig = w_mod_sig[0].detach().numpy()
        # mog_sig = (mog_sig - mog_sig.min()) / (mog_sig.max() - mog_sig.min())
        # from matplotlib import pyplot as plt
        # plt.plot(a1, label="a1")
        # plt.plot(a2, label="a2")
        # plt.plot(b0, label="b0")
        # plt.plot(b1, label="b1")
        # plt.plot(b2, label="b2")
        # plt.plot(mog_sig, label="mog_sig", linestyle="--")
        # plt.legend()
        # plt.show()

        if interp_coeff:
            a_coeff = util.interpolate_dim(a_coeff, n_samples, dim=1)
            b_coeff = util.interpolate_dim(b_coeff, n_samples, dim=1)
        zi_a = zi
        if zi_a is not None:
            zi_a = tr.flip(zi_a, dims=[1])  # Match scipy's convention for torchlpc
        y_a = self.lpc_func(x, a_coeff, zi_a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeff, zi)
        a1 = a_coeff[:, :, 0]
        a2 = a_coeff[:, :, 1]
        a0 = tr.ones_like(a1)
        a_coeff = tr.stack([a0, a1, a2], dim=2)
        return y_ab, a_coeff, b_coeff, y_a


if __name__ == "__main__":
    w = tr.full((1, 1), 0.125 * tr.pi)
    # q = tr.full_like(w, 0.7071)
    q = tr.full_like(w, 4.0)
    # a, b = calc_biquad_coeff("lp", w, q)
    # a, b = calc_biquad_coeff("hp", w, q)
    a, b = calc_biquad_coeff("bp", w, q)
    # a, b = calc_biquad_coeff("no", w, q)
    a = a.squeeze()
    b = b.squeeze()
    print(b[0].item())
    print(b[1].item())
    print(b[2].item())
    print(f"")
    print(1.0)
    print(a[0].item())
    print(a[1].item())
