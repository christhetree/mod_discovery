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


class PiecewiseBezier(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        modes: Optional[T] = None,
        is_c1_cont: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        assert n_frames >= 2
        assert n_segments >= 1
        assert n_segments < n_frames
        assert degree >= 1
        if is_c1_cont:
            assert degree >= 3, f"C1 continuity requires a degree of 3 or higher"
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree
        self.is_c1_cont = is_c1_cont
        self.eps = eps

        # Create the support and mask from evenly spaced modes if none are provided
        if modes is None:
            modes = tr.linspace(0.0, 1.0, n_segments + 1)[1:-1].view(1, -1)
        support, mask = self._create_support_and_mask(modes)
        self.register_buffer("modes", modes, persistent=False)
        self.register_buffer("support", support, persistent=False)
        self.register_buffer("mask", mask, persistent=False)
        if self.n_segments == 1:
            self.mask = None

        # Prepare binomial coefficients and exponents for the Bernstein basis
        bin_coeff = []
        for i in range(degree + 1):
            bin_coeff.append(math.comb(degree, i))
        bin_coeff = tr.tensor(bin_coeff).float()
        exp = tr.arange(start=0, end=degree + 1).float()
        self.register_buffer("bin_coeff", bin_coeff, persistent=False)
        self.register_buffer("exp", exp, persistent=False)

    def _create_support_and_mask(self, modes: T) -> (T, T):
        assert self.n_segments < self.n_frames
        assert modes.ndim == 2
        assert modes.size(1) == self.n_segments - 1
        bs = modes.size(0)
        support = tr.linspace(0.0, 1.0, self.n_frames).view(1, 1, -1)
        support = support.repeat(bs, self.n_segments, 1)
        seg_starts = F.pad(modes, (1, 0), value=0.0).unsqueeze(2)
        seg_ends = tr.roll(seg_starts, shifts=-1, dims=1)
        seg_ends[:, -1, :] = 1.0
        seg_ranges = seg_ends - seg_starts
        mask = (support >= seg_starts) & (support < seg_ends)
        mask[:, -1, -1] = True
        support = support - seg_starts
        support = support / seg_ranges
        support = support * mask
        assert support.min() == 0.0
        assert support.max() == 1.0
        assert (mask.sum(dim=1).sum(dim=1) == self.n_frames).all()
        return support, mask

    def _logits_to_control_points(self, logits: T) -> T:
        p = tr.tanh(logits) * (1.0 - self.eps)
        p = p * 0.5 + 0.5
        return p

    def _make_mask(self, seg_intervals: Optional[T]) -> Optional[T]:
        return self.mask

    def make_bezier(
        self,
        cp: T,
        cp_are_logits: bool = False,
        si: Optional[T] = None,
        si_are_logits: bool = False,
    ) -> T:
        # Process control points and make a Bezier curve for each segment
        assert cp.ndim == 3
        assert cp.size(1) == self.n_segments
        assert cp.size(2) == self.degree + 1
        if cp_are_logits:
            cp = self._logits_to_control_points(cp)
        bs = cp.size(0)
        if self.support.size(0) != bs:
            assert self.support.size(0) == 1
            support = self.support.repeat(bs, 1, 1)
        else:
            support = self.support

        if self.is_c1_cont:
            p_nm1 = cp[:, :-1, -1]
            p_nm2 = cp[:, :-1, -2]
            q1 = 2 * p_nm1 - p_nm2
            cp[:, 1:, 1] = q1

        cp = cp.unsqueeze(-1)
        bezier = self.create_bezier(support, cp, self.bin_coeff, self.exp)
        bezier = bezier.squeeze(-1)
        # Process optional segment intervals and maybe apply a mask
        if si is not None:
            # TODO(cm): debug this
            assert si.ndim == 2
            assert si.size(0) == bs
            assert si.size(1) == self.n_segments
            if si_are_logits:
                si = PiecewiseBezier.logits_to_seg_intervals(si)
            si_sums = si.sum(dim=1)
            assert tr.allclose(si_sums, tr.ones_like(si_sums))
        mask = self._make_mask(si)
        if mask is not None:
            bezier = bezier * mask
        # Sum the Bezier curves into a single curve
        bezier = bezier.sum(dim=1)
        return bezier

    def forward(
        self,
        cp: T,
        cp_are_logits: bool = False,
        si_logits: Optional[T] = None,
    ) -> T:
        bs = cp.size(0)
        n_dim = cp.ndim
        n_ch = 1
        if n_dim == 4:
            n_ch = cp.size(1)
            cp = tr.flatten(cp, start_dim=0, end_dim=1)
            if si_logits is not None:
                assert si_logits.shape == (bs, n_ch, self.n_segments)
                si_logits = tr.flatten(si_logits, start_dim=0, end_dim=1)
        x = self.make_bezier(
            cp, cp_are_logits=cp_are_logits, si=si_logits, si_are_logits=True
        )
        # assert x.min() >= 0.0, f"x.min(): {x.min()}"
        # assert x.max() <= 1.0, f"x.max(): {x.max()}"
        if n_dim == 4:
            x = x.view(bs, n_ch, x.size(1))
        return x

    @staticmethod
    def create_bezier(support: T, cp: T, bin_coeff: T, exp: T) -> T:
        assert support.ndim == 3
        bs, n_segments, n_frames = support.size()
        assert cp.ndim == 4
        assert cp.size(0) == bs
        assert cp.size(1) == n_segments
        degree = cp.size(2) - 1
        dim = cp.size(3)
        assert bin_coeff.size() == (degree + 1,)
        assert exp.size() == (degree + 1,)
        t = support.unsqueeze(3)
        bin_coeff = bin_coeff.view(1, 1, 1, -1)
        exp = exp.view(1, 1, 1, -1)
        bernstein_basis = bin_coeff * (t**exp) * ((1.0 - t) ** (degree - exp))
        cp = cp.unsqueeze(2)
        bernstein_basis = bernstein_basis.unsqueeze(-1)
        bezier = cp * bernstein_basis
        bezier = bezier.sum(dim=-2)
        return bezier

    @staticmethod
    def logits_to_seg_intervals(
        logits: T, min_seg_interval: Optional[float] = None, softmax_tau: float = 1.0
    ) -> T:
        assert logits.ndim == 2
        n_intervals = logits.size(1)
        if min_seg_interval is None:
            min_seg_interval = 1.0 / (2.0 * n_intervals)
        else:
            assert 0.0 < min_seg_interval < 1.0 / n_intervals
        scaling_factor = 1.0 - (n_intervals * min_seg_interval)
        si = util.stable_softmax(logits, tau=softmax_tau)
        si = si * scaling_factor + min_seg_interval
        return si


class PiecewiseBezier2D(PiecewiseBezier):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        is_c1_cont: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__(
            n_frames, n_segments, degree, modes=None, is_c1_cont=is_c1_cont, eps=eps
        )
        if self.mask is not None:
            self.register_buffer(
                "mask", self.mask.unsqueeze(-1).repeat(1, 1, 1, 2), persistent=False
            )

    def _logits_to_control_points(self, logits: T) -> T:
        # logits_x = logits[..., 0]
        # logits_y = logits[..., 1]
        # p = tr.tanh(logits) * (1.0 - self.eps)
        # p = p * 0.5 + 0.5
        # return p
        pass

    def convert_quadratic_cp_logits(self, logits: T, is_bounded: bool) -> T:
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        assert logits.size(2) == self.n_segments
        assert logits.size(3) == 4
        logits_y = logits[..., :3]
        if is_bounded:
            cp_y = tr.tanh(logits_y) * (1.0 - self.eps)
            cp_y = cp_y * 0.5 + 0.5
        else:
            cp_y = logits_y
        logits_x = logits[..., 3]
        cp_x_endpoints = tr.linspace(
            0.0, 1.0, self.n_segments + 1, device=logits.device
        )
        cp_x_endpoints = cp_x_endpoints.view(1, 1, -1).expand(logits.size(0), -1, -1)
        cp_x_left = cp_x_endpoints[..., :-1]
        cp_x_right = cp_x_endpoints[..., 1:]
        cp_x_range = cp_x_right - cp_x_left
        logits_x = tr.tanh(logits_x) * (1.0 - self.eps)
        logits_x = logits_x * 0.5 + 0.5
        cp_x_middle = logits_x * cp_x_range + cp_x_left
        cp_x = tr.stack([cp_x_left, cp_x_middle, cp_x_right], dim=-1)
        cp = tr.stack([cp_x, cp_y], dim=-1)
        return cp

    def make_bezier(
        self,
        cp: T,
        cp_are_logits: bool = False,
    ) -> (T, T, T):
        # Process control points and make a Bezier curve for each segment
        assert cp.ndim == 4
        assert cp.size(1) == self.n_segments
        assert cp.size(2) == self.degree + 1
        assert cp.size(3) == 2
        if cp_are_logits:
            cp = self._logits_to_control_points(cp)
        bs = cp.size(0)
        if self.support.size(0) != bs:
            assert self.support.size(0) == 1
            support = self.support.repeat(bs, 1, 1)
        else:
            support = self.support

        if self.is_c1_cont:
            p_nm1 = cp[:, :-1, -1, :]
            p_nm2 = cp[:, :-1, -2, :]
            q1 = 2 * p_nm1 - p_nm2
            cp[:, 1:, 1, :] = q1

        bezier = self.create_bezier(support, cp, self.bin_coeff, self.exp)
        if self.mask is not None:
            bezier *= self.mask
        bezier = bezier.sum(dim=1, keepdim=True)
        bezier_x = bezier[..., 0]

        grid_x = bezier_x
        grid_y = tr.zeros_like(grid_x)
        grid = tr.stack([grid_x, grid_y], dim=-1)
        grid = (grid * 2.0 - 1.0) * (1.0 - self.eps)
        assert grid.min() >= -1.0, f"grid.min(): {grid.min()}"
        assert grid.max() <= 1.0, f"grid.max(): {grid.max()}"

        bezier_y = bezier[..., 1]
        bez_img = tr.swapaxes(bezier_y.unsqueeze(-1), 2, 3)

        sampled = F.grid_sample(
            bez_img,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.view(bs, self.n_frames)
        bezier_x = bezier_x.view(bs, self.n_frames)
        bezier_y = bezier_y.view(bs, self.n_frames)
        return sampled, bezier_x, bezier_y

    def forward(
        self,
        cp: T,
        cp_are_logits: bool = False,
    ) -> T:
        bs = cp.size(0)
        n_dim = cp.ndim
        n_ch = 1
        if n_dim == 5:
            n_ch = cp.size(1)
            cp = tr.flatten(cp, start_dim=0, end_dim=1)
        x, _, _ = self.make_bezier(cp, cp_are_logits=cp_are_logits)
        # assert x.min() >= 0.0, f"x.min(): {x.min()}"
        # assert x.max() <= 1.0, f"x.max(): {x.max()}"
        if n_dim == 5:
            x = x.view(bs, n_ch, x.size(1))
        return x


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
        super().__init__(n_frames, n_segments, degree, eps=eps)
        self.tsp_window_size = tsp_window_size
        self.n = n

        # Overwrite the superclass support
        support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1)
        support = support.repeat(1, n_segments, 1)
        self.register_buffer("support", support, persistent=False)

        # Create support for the segmentation function
        seg_fn_support = tr.linspace(0.0, 1.0, n_frames).view(1, 1, -1)
        seg_fn_support = seg_fn_support.repeat(1, n_segments - 1, 1)
        self.register_buffer("seg_fn_support", seg_fn_support, persistent=False)

        # Create segment indices for the continuous mask
        seg_indices = tr.arange(0, n_segments).float()
        self.register_buffer("seg_indices", seg_indices, persistent=False)

    def _make_mask(self, seg_intervals: Optional[T]) -> Optional[T]:
        if self.n_segments == 1:
            return None
        assert seg_intervals is not None
        assert seg_intervals.ndim == 2
        assert seg_intervals.size(1) == self.n_segments
        bs = seg_intervals.size(0)
        seg_fn_support = self.seg_fn_support.repeat(bs, 1, 1)
        seg_fn = self.create_seg_fn(
            seg_fn_support, seg_intervals, self.tsp_window_size, self.n
        )
        cont_mask = self.create_cont_mask_from_seg_fn(seg_fn, self.seg_indices)
        return cont_mask

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
