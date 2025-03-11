import logging
import math
import os
from typing import Optional

import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import util
from losses import FirstDerivativeL1Loss, SecondDerivativeL1Loss

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
    from matplotlib import pyplot as plt

    tr.manual_seed(42)

    n_iter = 1000
    lr_acc = 1.01
    lr_brake = 0.9

    l1_loss = nn.L1Loss()
    d1_loss = FirstDerivativeL1Loss()
    d2_loss = SecondDerivativeL1Loss()

    def modex_loss(y_hat: T, y) -> T:
        return l1_loss(y_hat, y) + 5.0 * d1_loss(y_hat, y) + 10.0 * d2_loss(y_hat, y)

    # loss_fn = ESRLoss()
    loss_fn = l1_loss
    # loss_fn = nn.MSELoss()
    # loss_fn = modex_loss
    # loss_fn = d1_loss

    # Larger batch size is worse because there are no batch-granular LR updates
    bs = 1
    # bs = 256
    n_frames = 1501
    n_segments = 4
    degree = 3

    si_logits = tr.rand(bs, n_segments) * 3.0
    si = PiecewiseBezierDiffSeg.logits_to_seg_intervals(
        si_logits,
        # min_seg_interval=1.0 / (2 * n_segments),
    )
    modes = tr.cumsum(si, dim=1)[:, :-1]
    log.info(f"modes = {modes}")
    bezier_module = PiecewiseBezier(
        n_frames, n_segments, degree, modes=modes, is_c1_cont=False
    )
    si = None
    # bezier_module = PiecewiseBezierDiffSeg(n_frames, n_segments, degree)
    # si = tr.rand(bs, n_segments)
    # si = si / si.sum(dim=1, keepdim=True)
    # log.info(f"si = {si}")

    cp = (
        tr.rand(bs, (n_segments * degree) + 1)
        .unfold(dimension=1, size=degree + 1, step=degree)
        .contiguous()
    )
    curves = bezier_module.make_bezier(
        cp, cp_are_logits=False, si=si, si_are_logits=False
    )

    # hz = 30.0
    # amp = 0.5
    # x = tr.linspace(0.0, 1.0, n_frames).view(1, -1)
    # sinusoid = (tr.sin(2 * tr.pi * hz * x) + 1.0) / 2.0 * amp
    # curves = sinusoid

    # ========================= Define hat hyperparams ==============================
    n_segments_hat = 12
    degree_hat = 3
    # n_segments_hat = 1
    # degree_hat = 36

    bezier_module_hat = PiecewiseBezier(
        n_frames, n_segments_hat, degree_hat, is_c1_cont=True
    )
    si_logits = None
    # bezier_module_hat = PiecewiseBezierDiffSeg(n_frames, n_segments_hat, degree_hat)
    # si_logits = tr.rand(bs, n_segments_hat)
    # si_logits.requires_grad_()
    # ========================= Define hat hyperparams ==============================

    cp_logits = tr.rand(bs, (n_segments_hat * degree_hat) + 1)
    cp_logits.requires_grad_()

    loss_hist = []
    curr_lr = 1.0
    eps = 1e-5
    min_lr = eps
    curves_hat = None

    for idx in tqdm(range(n_iter)):
        cp_logits_view = cp_logits.unfold(
            dimension=1, size=degree_hat + 1, step=degree_hat
        )
        curves_hat = bezier_module_hat.make_bezier(
            cp_logits_view, cp_are_logits=True, si=si_logits, si_are_logits=True
        )
        loss = loss_fn(curves_hat, curves)
        loss_hist.append(loss)
        loss.backward()

        if idx > 0 and loss_hist[idx] - loss_hist[idx - 1] > -eps:
            curr_lr *= lr_brake
        else:
            with tr.no_grad():
                cp_logits -= curr_lr * cp_logits.grad
                cp_logits.grad.zero_()
            curr_lr *= lr_acc
        # log.info(f"loss: {loss.item():.6f}, lr: {curr_lr:.6f}")

        if curr_lr < min_lr:
            log.info(f"Reached min_lr: {min_lr}, final loss = {loss.item():.4f}")
            break

        if idx % 5 == 0:
            plt.plot(curves[0].detach().numpy(), label="target")
            plt.plot(curves_hat[0].detach().numpy(), label="hat")
            plt.legend()
            plt.show()

    plt.plot(curves[0].detach().numpy(), label="target 0")
    plt.plot(curves_hat[0].detach().numpy(), label="hat 0")
    plt.title(f"loss = {loss_hist[1].item():.6f}")
    plt.legend()
    plt.show()
    # plt.plot(curves[1].detach().numpy(), label="target 1")
    # plt.plot(curves_hat[1].detach().numpy(), label="hat 1")
    # plt.legend()
    # plt.show()
    exit()

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
