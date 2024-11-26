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


class PiecewiseBezier(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert n_frames >= 2
        assert n_segments >= 1
        assert n_segments < n_frames
        assert degree >= 1
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree
        self.eps = eps

        seg_intervals = tr.full((1, n_frames - 1), 1.0 / (n_frames - 1))
        knots = tr.linspace(0.0, 1.0, n_segments + 1)[1:-1].view(1, -1)
        support, mask = self.create_support(seg_intervals, knots)
        assert support.min() == 0.0
        assert support.max() == 1.0
        assert mask.sum() == n_frames
        self.register_buffer("knots", knots)
        self.register_buffer("support", support)
        self.register_buffer("mask", mask)

        bin_coeff = []
        for i in range(degree + 1):
            bin_coeff.append(math.comb(degree, i))
        bin_coeff = tr.tensor(bin_coeff).float()
        exp = tr.arange(start=0, end=degree + 1).float()
        self.register_buffer("bin_coeff", bin_coeff)
        self.register_buffer("exp", exp)

    def logits_to_control_points(self, logits: T) -> T:
        p = tr.tanh(logits) * (1.0 - self.eps)
        p = p * 0.5 + 0.5
        return p

    def logits_to_seg_intervals(self, logits: T) -> T:
        assert logits.ndim == 2
        # assert logits.size(1) == self.n_frames - 1
        n_intervals = logits.size(1)
        min_seg_interval = 1.0 / (2.0 * n_intervals)
        scaling_factor = 1.0 - (n_intervals * min_seg_interval)
        si = util.stable_softmax(logits)
        si = si * scaling_factor + min_seg_interval
        si = si.unsqueeze(1)
        si = F.interpolate(si, size=self.n_frames - 1, mode="nearest")
        si = si.squeeze(1)
        si = si / si.sum(dim=1, keepdim=True)
        return si

    def make_bezier_from_control_points(
        self, control_points: T, support: T, mask: Optional[T] = None
    ) -> T:
        assert control_points.ndim == 3
        assert control_points.size(1) == self.n_segments
        assert control_points.size(2) == self.degree + 1
        control_points = control_points.unsqueeze(-1)
        bezier = self.create_bezier(support, control_points, self.bin_coeff, self.exp)
        bezier = bezier.squeeze(-1)
        if mask is not None:
            bezier = bezier * mask
        bezier = bezier.sum(dim=1)
        return bezier

    def forward(
        self,
        coeff_logits: T,
        last_seg_last_p_logits: T,
        support_logits: Optional[T] = None,
    ) -> T:
        bs = coeff_logits.size(0)
        n_dim = coeff_logits.ndim
        n_ch = 1
        if n_dim == 4:
            n_ch = coeff_logits.size(1)
            coeff_logits = tr.flatten(coeff_logits, start_dim=0, end_dim=1)
            last_seg_last_p_logits = last_seg_last_p_logits.view(bs * n_ch, 1)
        assert coeff_logits.ndim == 3
        assert coeff_logits.size(1) == self.n_segments
        assert coeff_logits.size(2) == self.degree
        assert last_seg_last_p_logits.size() == (bs * n_ch, 1)
        coeff = self.logits_to_control_points(coeff_logits)
        last_seg_last_p = self.logits_to_control_points(last_seg_last_p_logits)
        p0 = coeff[:, :, 0:1]
        last_p = tr.roll(p0, shifts=-1, dims=1)
        last_p[:, -1, :] = last_seg_last_p
        control_points = tr.cat((coeff, last_p), dim=2)

        if support_logits is None:
            # assert False
            support = self.support.repeat(bs * n_ch, 1, 1)
            mask = self.mask
        else:
            seg_intervals = self.logits_to_seg_intervals(support_logits)
            support, mask = self.create_support(seg_intervals, self.knots)

        x = self.make_bezier_from_control_points(control_points, support, mask)
        assert x.min() >= 0.0, f"x.min(): {x.min()}"
        assert x.max() <= 1.0, f"x.max(): {x.max()}"
        if n_dim == 4:
            x = x.view(bs, n_ch, x.size(1))
        return x

    @staticmethod
    def create_support(seg_intervals: T, knots: T) -> (T, T):
        assert seg_intervals.ndim == 2
        bs = seg_intervals.size(0)
        n_frames = seg_intervals.size(1) + 1
        assert knots.ndim == 2
        n_segments = knots.size(1) + 1
        assert n_segments < n_frames
        support = tr.cumsum(seg_intervals, dim=1)
        support = F.pad(support, (1, 0), value=0.0).unsqueeze(1)
        max_t = tr.max(support, dim=2, keepdim=False).values
        support = support.repeat(1, n_segments, 1)
        seg_starts = F.pad(knots, (1, 0), value=0.0).repeat(bs, 1).unsqueeze(2)
        seg_ends = tr.roll(seg_starts, shifts=-1, dims=1)
        seg_ends[:, -1, :] = max_t
        seg_ranges = seg_ends - seg_starts
        mask = (support >= seg_starts) & (support < seg_ends)
        mask[:, -1, -1] = True
        support = support - seg_starts
        support = support / seg_ranges
        support = support * mask
        return support, mask

    @staticmethod
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


class PiecewiseBezierDiffSeg(PiecewiseBezier):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        tsp_window_size: Optional[float] = None,
        n: int = 2,
        eps: float = 1e-3,
    ):
        super().__init__(n_frames, n_segments, degree, eps)
        self.tsp_window_size = tsp_window_size
        self.n = n

        support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1)
        support = support.repeat(1, n_segments, 1)
        self.register_buffer("support", support)

        seg_fn_support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1)
        seg_fn_support = seg_fn_support.repeat(1, n_segments - 1, 1)
        self.register_buffer("seg_fn_support", seg_fn_support)

        seg_indices = tr.arange(0, n_segments).float()
        self.register_buffer("seg_indices", seg_indices)

    def logits_to_seg_intervals(
        self, logits: T, min_seg_interval: Optional[float] = None
    ) -> T:
        assert logits.ndim == 2
        n_intervals = logits.size(1)
        if min_seg_interval is None:
            min_seg_interval = 1.0 / (2.0 * n_intervals)
        else:
            assert 0.0 < min_seg_interval < 1.0 / n_intervals
        scaling_factor = 1.0 - (n_intervals * min_seg_interval)
        si = util.stable_softmax(logits)
        si = si * scaling_factor + min_seg_interval
        return si

    def forward(self, coeff_logits: T, mode_logits: Optional[T] = None) -> T:
        bs = coeff_logits.size(0)
        n_dim = coeff_logits.ndim
        n_ch = 1
        if n_dim == 4:
            n_ch = coeff_logits.size(1)
            coeff_logits = tr.flatten(coeff_logits, start_dim=0, end_dim=1)
        assert coeff_logits.ndim == 3
        assert coeff_logits.size(1) == self.n_segments
        assert coeff_logits.size(2) == self.degree + 1

        if mode_logits is None:
            cont_mask = None
            assert self.n_segments == 1
        else:
            if n_dim == 4:
                mode_logits = tr.flatten(mode_logits, start_dim=0, end_dim=1)
            assert mode_logits.ndim == 2
            assert mode_logits.size(1) == self.n_segments
            seg_intervals = self.logits_to_seg_intervals(mode_logits)
            seg_fn_support = self.seg_fn_support.repeat(bs * n_ch, 1, 1)
            seg_fn = self.create_seg_fn(
                seg_fn_support, seg_intervals, self.tsp_window_size, self.n
            )
            cont_mask = self.create_cont_mask_from_seg_fn(seg_fn, self.seg_indices)

        control_points = self.logits_to_control_points(coeff_logits)
        support = self.support.repeat(bs * n_ch, 1, 1)
        x = self.make_bezier_from_control_points(control_points, support, cont_mask)
        assert x.min() >= 0.0, f"x.min(): {x.min()}"
        assert x.max() <= 1.0, f"x.max(): {x.max()}"
        if n_dim == 4:
            x = x.view(bs, n_ch, x.size(1))
        return x

    @staticmethod
    def create_seg_fn(
        support: T,
        seg_intervals: T,
        tsp_window_size: Optional[float] = None,
        n: int = 2,
    ) -> T:
        """
        Creates a differentiable segmentation function based on the two-sided power distribution.

        Parameters:
        - support (Tensor): Shape (bs, n_segments - 1, n_samples). Values evenly spaced from 0 to 1 for each batch.
        - seg_intervals (Tensor): Shape (bs, n_segments). Widths of segments, summing to 1.0.
        - tsp_window_size (Optional[float]): Support size of the two-sided power distribution.
                                             Default is 1 / n_segments.
        - n (int): Exponent for the two-sided power distribution. Defaults to 2.

        Returns:
        - Tensor: Shape (bs, n_samples), the segmentation function, monotonically increasing
                  from 0 to 1.
        """
        assert support.ndim == 3
        bs = support.size(0)
        n_segments = support.size(1) + 1
        n_samples = support.size(2)
        assert seg_intervals.ndim == 2
        assert seg_intervals.shape == (bs, n_segments)

        # Default window size if not provided
        if tsp_window_size is None:
            tsp_window_size = 1.0 / n_segments
        assert 0.0 < tsp_window_size < 1.0

        m = (
            tr.cumsum(seg_intervals, dim=1)[:, :-1]
            .unsqueeze(2)
            .expand(-1, -1, n_samples)
        )
        a = m - tsp_window_size / 2.0
        b = m + tsp_window_size / 2.0
        u = support

        const_l = (m - a) / (b - a)
        const_r = (b - m) / (b - a)
        mask_ll = u < a
        mask_l = (a <= u) & (u <= m)
        mask_r = (m < u) & (u <= b)
        mask_rr = b < u
        # all_masks = mask_ll.int() + mask_l.int() + mask_r.int() + mask_rr.int()
        # assert all_masks.sum() == all_masks.numel()

        u[mask_ll] = 0.0
        curr_a = a[mask_l]
        curr_m = m[mask_l]
        curr_u = u[mask_l]
        const_l = const_l[mask_l]
        u[mask_l] = const_l * (((curr_u - curr_a) / (curr_m - curr_a)) ** n)
        curr_b = b[mask_r]
        curr_m = m[mask_r]
        curr_u = u[mask_r]
        const_r = const_r[mask_r]
        u[mask_r] = 1.0 - const_r * (((curr_b - curr_u) / (curr_b - curr_m)) ** n)
        u[mask_rr] = 1.0

        seg_fn = u.sum(dim=1)
        # seg_fn = seg_fn / (n_segments - 1)

        return seg_fn

    @staticmethod
    def create_cont_mask_from_seg_fn(seg_fn: T, seg_indices: T) -> T:
        """
        Makes a continuous mask from a segmentation function and segment indices.

        :param seg_fn: segmentation function, shape (bs, n_samples)
        :param seg_indices: segment indices, shape (bs, n_segments)
        :return: continuous mask, shape (bs, n_segments, n_samples)
        """
        assert seg_fn.ndim == 2
        assert seg_indices.ndim == 1
        n_segments = seg_indices.size(0)
        seg_fn = seg_fn.unsqueeze(1).repeat(1, n_segments, 1)
        seg_indices = seg_indices.view(1, -1, 1)
        cont_mask = 1.0 - tr.abs(seg_fn - seg_indices)
        cont_mask = cont_mask.clamp(min=0.0)
        return cont_mask


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
    n_samples = 10
    support = tr.linspace(0.0, 1.0, n_samples).view(1, -1).repeat(2, 1)
    support = support.unsqueeze(0)
    control_points = tr.tensor(
        [[0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [0.3, 0.1, 0.3, 0.7, 0.9, 0.7]]
    )
    control_points = control_points.unsqueeze(0).unsqueeze(-1)
    n_segments = control_points.size(1)
    degree = control_points.size(2) - 1
    splines = PiecewiseBezierDiffSeg(n_samples, n_segments, degree)
    x = splines.create_bezier(support, control_points, splines.bin_coeff, splines.exp)

    import matplotlib.pyplot as plt

    x = x.squeeze()
    plt.plot(x[0].numpy())
    plt.plot(x[1].numpy())
    plt.show()

    exit()

    n_samples = 100
    support = tr.linspace(0.0, 1.0, n_samples).view(1, -1).repeat(2, 1)
    seg_intervals = tr.tensor([[0.1, 0.1, 0.8], [0.2, 0.6, 0.2]])
    # seg_intervals = tr.tensor([[0.1, 0.9], [0.5, 0.5]])
    n_segments = seg_intervals.size(1)
    tsp_window_size = 1 / n_segments
    seg_fn = create_seg_fn(support, seg_intervals, tsp_window_size)

    import matplotlib.pyplot as plt

    plt.plot(seg_fn[0])
    plt.plot(seg_fn[1])
    plt.show()

    seg_vals = tr.arange(0, n_segments).float()
    seg_vals = seg_vals.view(1, -1, 1)
    # seg_bound_l = tr.arange(0, n_segments).float() / n_segments
    # seg_bound_l = seg_bound_l.view(1, -1, 1)
    # seg_bound_r = tr.arange(1, n_segments + 1).float() / n_segments
    # seg_bound_r = seg_bound_r.view(1, -1, 1)

    seg_fn = seg_fn.unsqueeze(1).repeat(1, n_segments, 1)
    cont_mask = 1.0 - tr.abs(seg_fn - seg_vals)
    cont_mask = cont_mask.clamp(min=0.0)

    derp = 1
    exit()

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
        # [[0.0, 1.0, 1.0, 0.0]]
    )
    control_points = control_points.unsqueeze(0)
    bs = control_points.size(0)
    n_segments = control_points.size(1)
    degree = control_points.size(2) - 1

    n_frames = 1000
    bezier_module = PiecewiseBezier(n_frames, n_segments, degree)
    x = bezier_module.make_bezier_from_control_points(control_points)

    # seg_intervals = tr.full((bs, n_frames - 1), 1.0 / (n_frames - 1))
    # knots = tr.linspace(0.0, 1.0, n_segments + 1)[1:-1].view(1, -1)
    # support, mask = PiecewiseBezier.create_support(seg_intervals, knots)
    # exit()

    # support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1).repeat(bs, n_segments, 1)
    # bin_coeff = []
    # for i in range(degree + 1):
    #     bin_coeff.append(math.comb(degree, i))
    # bin_coeff = tr.tensor(bin_coeff)
    # exp = tr.arange(start=0, end=degree + 1).float()
    # x = PiecewiseBezier.create_bezier(support, control_points, bin_coeff, exp)
    # # x = x.view(1, -1)
    # x = x.squeeze(-1)
    # x = x * mask
    # x = x.sum(dim=1)

    log.info(f"x.shape: {x.shape}")

    x = x[0].numpy()

    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.show()
