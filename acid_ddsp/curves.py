import logging
import math
import os
from typing import Optional

import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F

import util

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class PiecewiseSplines(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
    ):
        super().__init__()
        assert n_frames >= 2
        assert n_segments >= 1
        assert degree >= 1
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree
        self.min_t = 0.0
        self.max_t = 1.0

        support = tr.linspace(self.min_t, self.max_t, n_frames).view(1, -1, 1, 1)
        support = support.repeat(1, 1, n_segments, degree)

        segment_offsets = tr.linspace(0.0, 1.0, n_segments + 1)[:-1].view(1, 1, -1, 1)
        support = support - segment_offsets
        support = tr.clamp(support, min=self.min_t, max=self.max_t / n_segments)
        exponent = tr.arange(start=1, end=degree + 1).int().view(1, 1, 1, -1)
        support = support.pow(exponent)

        self.register_buffer("support", support)
        # self.register_buffer("exponent", tr.arange(start=1, end=degree + 1).int())

    def forward(self, coeff: T, bias: Optional[T] = None) -> T:
        bs = coeff.size(0)
        n_dim = coeff.ndim
        if n_dim == 4:
            coeff = tr.flatten(coeff, start_dim=0, end_dim=1)
        assert coeff.ndim == 3
        assert coeff.size(1) == self.n_segments
        assert coeff.size(2) == self.degree
        coeff = coeff.unsqueeze(1)
        x = coeff * self.support
        x = x.sum(dim=[2, 3])
        if bias is not None:
            bias = bias.view(x.size(0), 1)
            x = x + bias
        if n_dim == 4:
            x = x.view(bs, -1, x.size(1))
        return x

    # def forward(self, segment_intervals: T, coeff: T, bias: Optional[T] = None) -> T:
    #     bs = coeff.size(0)
    #     n_dim = coeff.ndim
    #     if n_dim == 4:
    #         coeff = tr.flatten(coeff, start_dim=0, end_dim=1)
    #     support = self.support.repeat(bs, 1, 1, 1)
    #     x = self.create_splines(support, segment_intervals, coeff, self.exponent, bias)
    #     if n_dim == 4:
    #         x = x.view(bs, -1, x.size(1))
    #     return x
    #
    # @staticmethod
    # def create_splines(
    #     support: T,
    #     segment_intervals: T,
    #     coeff: T,
    #     exponent: T,
    #     bias: Optional[T] = None,
    # ) -> T:
    #     assert support.ndim == 4
    #     assert segment_intervals.ndim == 2
    #     assert coeff.ndim == 3
    #     bs, n_frames, n_segments, degree = support.size()
    #     assert segment_intervals.size(0) == bs
    #     assert segment_intervals.size(1) == n_segments
    #     assert coeff.size() == (bs, n_segments, degree)
    #     assert exponent.size() == (degree,)
    #     segment_offsets = tr.cumsum(segment_intervals, dim=1)
    #     segment_offsets = tr.roll(segment_offsets, shifts=1, dims=1)
    #     segment_offsets[:, 0] = 0.0
    #     # This is done to avoid floating point errors, but is prob not necessary
    #     # segment_offsets = magic_clamp(segment_offsets, min_value=0.0, max_value=1.0)
    #     segment_offsets = segment_offsets.view(bs, 1, n_segments, 1)
    #     support = support - segment_offsets
    #     clamp_max = segment_intervals[:, None, :, None]
    #     clamp_min = tr.zeros_like(clamp_max)
    #     support = magic_clamp(support, min_value=clamp_min, max_value=clamp_max)
    #     exponent = exponent.view(1, 1, 1, -1)
    #     support = support.pow(exponent)
    #     coeff = coeff.unsqueeze(1)
    #     x = coeff * support
    #     x = x.sum(dim=[2, 3])
    #     if bias is not None:
    #         bias = bias.view(bs, 1)
    #         x = x + bias
    #     return x


def create_support(seg_intervals: T, knots: T) -> (T, T):
    assert seg_intervals.ndim == 2
    n_frames = seg_intervals.size(1) + 1
    assert knots.ndim == 2
    n_segments = knots.size(1) + 1
    assert n_segments < n_frames
    support = tr.cumsum(seg_intervals, dim=1)
    support = F.pad(support, (1, 0), value=0.0).unsqueeze(1)
    max_t = tr.max(support, dim=2, keepdim=True).values
    support = support.repeat(1, n_segments, 1)
    seg_starts = F.pad(knots, (1, 0), value=0.0).unsqueeze(2)
    seg_ends = tr.roll(seg_starts, shifts=-1, dims=1)
    seg_ends[:, -1, :] = max_t
    seg_ranges = seg_ends - seg_starts
    mask = (support >= seg_starts) & (support < seg_ends)
    mask[:, -1, -1] = True
    support = support - seg_starts
    support = support / seg_ranges
    support = support * mask
    return support, mask


def create_bezier(support: T, control_points: T, bin_coeff: T, exp: T) -> T:
    assert support.ndim == 3
    bs, n_segments, n_frames = support.size()
    assert control_points.ndim == 4
    assert control_points.size(0) == bs
    assert control_points.size(1) == n_segments
    degree = control_points.size(2) - 1
    dim = control_points.size(3)
    assert bin_coeff.size() == (degree + 1,)
    assert exp.size() == (degree + 1,)
    t = support.unsqueeze(3)
    bin_coeff = bin_coeff.view(1, 1, 1, -1)
    exp = exp.view(1, 1, 1, -1)
    bernstein_basis = bin_coeff * (t**exp) * ((1.0 - t) ** (degree - exp))
    control_points = control_points.unsqueeze(2)
    bernstein_basis = bernstein_basis.unsqueeze(-1)
    bezier = control_points * bernstein_basis
    bezier = bezier.sum(dim=-2)
    return bezier


class FourierSignal(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_bins: Optional[int] = None,
    ):
        super().__init__()
        self.n_frames = n_frames
        if n_bins is None:
            n_bins = n_frames // 2 + 1
        self.n_bins = n_bins

    def forward(self, mag: T, phase: T) -> T:
        assert mag.ndim == 2
        assert mag.size(1) == self.n_bins
        assert phase.ndim == 2
        assert phase.size(1) == self.n_bins
        mag = mag * self.n_frames
        fourier_x = mag * tr.exp(1j * phase)
        x = tr.fft.irfft(fourier_x, n=self.n_frames, dim=1)
        return x


if __name__ == "__main__":
    # n_frames = 200
    # n_bins = 50
    # mag = tr.rand(1, n_bins)
    # # phase = tr.zeros(1, n_bins)
    # phase = tr.rand(1, n_bins) * 2 * tr.pi
    #
    # fs = FourierSignal(n_frames, n_bins=n_bins)
    # # mag = tr.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    # # phase = tr.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
    # x = fs(mag, phase)
    # x = tr.sigmoid(x)
    #
    # log.info(f"x.shape: {x.shape}")
    # import matplotlib.pyplot as plt
    #
    # plt.plot(x[0].numpy())
    # plt.show()
    # exit()

    # n_frames = 2001
    # # n_frames = 10
    # # coeff = tr.tensor([[1.0, 0.0], [2.0, 0.0], [-1.0, -10.0]]).unsqueeze(0)
    # coeff = tr.tensor([[1.0], [-1.0], [1.0]]).unsqueeze(0)
    # log.info(f"coeff.shape: {coeff.shape}")
    # n_segments = coeff.size(1)
    # degree = coeff.size(2)
    # support = tr.linspace(0.0, 1.0, n_frames).view(1, -1, 1, 1)
    # support = support.repeat(1, 1, n_segments, degree)
    # exponent = tr.arange(start=1, end=degree + 1).int()
    # # segment_intervals = tr.tensor([0.333, 0.333, 0.333]).view(1, -1)
    # segment_intervals = tr.tensor([0.2, 0.4, 0.4]).view(1, -1)
    #
    # bias = None
    # # bias = tr.tensor(0.5)
    #
    # curves = PiecewiseSplines(
    #     n_frames,
    #     n_segments,
    #     degree,
    # )
    # # x = curves(coeff, bias)
    # x = curves.create_splines(support, segment_intervals, coeff, exponent, bias)
    # # x = tr.sigmoid(x)

    control_points = tr.tensor(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 3.0, 3.0, -0.6], [-0.2, 0.0, 0.0, 0.7]]
        # [[0.0, 1.0, 1.0, 0.0], [0.5, 3.0, 3.0, 0.6]]
    )
    control_points = control_points.unsqueeze(0).unsqueeze(-1)
    bs = control_points.size(0)
    n_segments = control_points.size(1)
    degree = control_points.size(2) - 1

    n_frames = 1000
    seg_intervals = tr.full((bs, n_frames - 1), 1.0 / (n_frames - 1))
    knots = tr.linspace(0.0, 1.0, n_segments + 1)[1:-1].view(1, -1)

    support, mask = create_support(seg_intervals, knots)
    # exit()


    # support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1).repeat(bs, n_segments, 1)
    bin_coeff = []
    for i in range(degree + 1):
        bin_coeff.append(math.comb(degree, i))
    bin_coeff = tr.tensor(bin_coeff)
    exp = tr.arange(start=0, end=degree + 1).float()
    x = create_bezier(support, control_points, bin_coeff, exp)
    # x = x.view(1, -1)
    x = x.squeeze(-1)
    x = x * mask
    x = x.sum(dim=1)

    log.info(f"x.shape: {x.shape}")

    x = x[0].numpy()

    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.show()
