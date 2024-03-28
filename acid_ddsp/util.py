import logging
import os
from typing import Union

import torch as tr
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def linear_interpolate_dim(
    x: T, n: int, dim: int = -1, align_corners: bool = True
) -> T:
    n_dim = x.ndim
    assert 0 < n_dim <= 3
    if dim < 0:
        dim = n_dim + dim
    assert 0 <= dim < n_dim
    if x.size(dim) == n:
        return x

    swapped_dims = False
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        assert dim != 0  # TODO(cm)
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        assert dim != 0  # TODO(cm)
        if dim == 1:
            x = x.swapaxes(1, 2)
            swapped_dims = True

    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    elif swapped_dims:
        x = x.swapaxes(1, 2)
    return x


def sample_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    x = (tr.rand(n) * (high - low)) + low
    if n == 1:
        return x.item()
    return x


def sample_log_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    # TODO(cm): replace with torch
    if low == high:
        if n == 1:
            return low
        else:
            return tr.full(size=(n,), fill_value=low)
    x = loguniform.rvs(low, high, size=n)
    if n == 1:
        return float(x)
    return tr.from_numpy(x).float()


def calc_h(a: T, b: T, n_frames: int = 50, n_fft: int = 1024) -> T:
    assert a.ndim == 3
    assert a.shape == b.shape
    a = linear_interpolate_dim(a, n_frames, dim=1, align_corners=True)
    b = linear_interpolate_dim(b, n_frames, dim=1, align_corners=True)
    A = tr.fft.rfft(a, n_fft)
    B = tr.fft.rfft(b, n_fft)
    H = B / A  # TODO(cm): Make more stable
    return H
