import logging
import os
from abc import abstractmethod, ABC
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

import util
from curves import PiecewiseBezier2D, PiecewiseBezier

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModSignalGenerator(ABC, nn.Module):
    @abstractmethod
    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        pass


class ModSignalGenVitalCurves(ModSignalGenerator):
    def __init__(self, curves_path: str):
        super().__init__()
        assert os.path.exists(curves_path), f"File {curves_path} does not exist"
        self.curves_path = curves_path

        self.curves = tr.load(curves_path)
        log.info(f"Loaded {self.curves.shape} curves from {curves_path}")
        self.n_curves = self.curves.size(0)
        self.n_frames = self.curves.size(1)

    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        assert n_frames == self.n_frames
        idx = tr.randint(0, self.n_curves, (1,), generator=rand_gen).item()
        mod_sig = self.curves[idx]
        return mod_sig


class ModSignalGenRandomUniformFrame(ModSignalGenerator):
    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        mod_sig = tr.rand(n_frames, generator=rand_gen)
        return mod_sig


class ModSignalGenRandomBezier2D(ModSignalGenerator):
    def __init__(
        self,
        min_n_seg: int = 1,
        max_n_seg: int = 8,
        min_degree: int = 1,
        max_degree: int = 3,
        min_seg_interval_frac: float = 0.25,
        is_c1_cont: bool = False,
        normalize: bool = False,
        eps: float = 1e-8,
        spline_eps: float = 1e-3,
    ):
        super().__init__()
        assert 1 <= min_n_seg <= max_n_seg
        self.min_n_seg = min_n_seg
        self.max_n_seg = max_n_seg
        assert 1 <= min_degree <= max_degree
        self.min_degree = min_degree
        self.max_degree = max_degree
        assert 0.0 <= min_seg_interval_frac <= 1.0
        self.min_seg_interval_frac = min_seg_interval_frac
        assert not is_c1_cont, "C1 continuity not supported currently"
        if is_c1_cont:
            assert min_degree >= 3
        self.is_c1_cont = is_c1_cont
        self.normalize = normalize
        self.eps = eps
        self.spline_eps = spline_eps

    def make_bezier(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        rand_gen: Optional[tr.Generator] = None,
    ) -> T:
        bezier = PiecewiseBezier2D(
            n_frames,
            n_segments,
            degree=degree,
            is_c1_cont=self.is_c1_cont,
            eps=self.spline_eps,
        )
        cp_x = self.make_cp_x(
            n_segments * degree, self.min_seg_interval_frac, rand_gen=rand_gen
        )
        cp_x = cp_x.unfold(dimension=1, size=degree + 1, step=degree)
        cp_y = (
            tr.rand((1, (n_segments * degree) + 1), generator=rand_gen)
            .unfold(dimension=1, size=degree + 1, step=degree)
            .contiguous()
        )
        cp = tr.stack([cp_x, cp_y], dim=-1)
        mod_sig, _, _ = bezier.make_bezier(cp=cp, cp_are_logits=False)
        return mod_sig

    @staticmethod
    def make_cp_x(
        n_intervals: int,
        min_interval_frac: float,
        rand_gen: Optional[tr.Generator] = None,
    ) -> T:
        x_intervals = tr.rand(1, n_intervals, generator=rand_gen)
        x_intervals = x_intervals / x_intervals.sum(dim=1)
        x_intervals = x_intervals * (1.0 - min_interval_frac)
        min_interval = min_interval_frac / n_intervals
        x_intervals = x_intervals + min_interval
        cp_x = tr.cumsum(x_intervals, dim=1)
        cp_x = F.pad(cp_x, (1, 0), value=0.0)
        cp_x[:, -1] = 1.0
        return cp_x

    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        if self.min_n_seg == self.max_n_seg:
            n_seg = self.min_n_seg
        else:
            n_seg = tr.randint(
                self.min_n_seg, self.max_n_seg + 1, (1,), generator=rand_gen
            ).item()
        if self.min_degree == self.max_degree:
            degree = self.min_degree
        else:
            degree = tr.randint(
                self.min_degree, self.max_degree + 1, (1,), generator=rand_gen
            ).item()
        mod_sig = self.make_bezier(n_frames, n_seg, degree, rand_gen)
        mod_sig = mod_sig.squeeze(0)
        if self.normalize:
            mod_sig_range = mod_sig.max() - mod_sig.min()
            mod_sig = (mod_sig - mod_sig.min()) / (mod_sig_range + self.eps)

        assert mod_sig.min() >= 0.0
        assert mod_sig.max() <= 1.0
        # from matplotlib import pyplot as plt
        # plt.plot(mod_sig)
        # plt.show()
        return mod_sig


class ModSignalGenRandomBezier1D(ModSignalGenRandomBezier2D):
    def make_bezier(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
        rand_gen: Optional[tr.Generator] = None,
    ) -> T:
        if self.min_seg_interval_frac == 1.0:
            modes = None
        else:
            modes = self.make_cp_x(
                n_segments, self.min_seg_interval_frac, rand_gen=rand_gen
            )
            modes = modes[:, 1:-1]
        bezier = PiecewiseBezier(
            n_frames,
            n_segments,
            degree=degree,
            modes=modes,
            is_c1_cont=self.is_c1_cont,
            eps=self.spline_eps,
        )
        cp = (
            tr.rand((1, (n_segments * degree) + 1), generator=rand_gen)
            .unfold(dimension=1, size=degree + 1, step=degree)
            .contiguous()
        )
        mod_sig = bezier.make_bezier(cp=cp, cp_are_logits=False)
        return mod_sig


class ModSignalGeneratorRandom(ModSignalGenerator):
    def __init__(
        self,
        min_start_val: float = 0.0,
        max_start_val: float = 1.0,
        min_corner_val: float = 0.0,
        max_corner_val: float = 1.0,
        min_end_val: float = 0.0,
        max_end_val: float = 1.0,
        min_attack_frac: float = 0.05,
        max_attack_frac: float = 0.25,
        min_alpha: float = 0.1,
        max_alpha: float = 6.0,
    ):
        super().__init__()
        assert 0.0 <= min_start_val <= max_start_val <= 1.0
        assert 0.0 <= min_corner_val <= max_corner_val <= 1.0
        assert 0.0 <= min_end_val <= max_end_val <= 1.0
        assert 0.0 <= min_attack_frac <= max_attack_frac <= 1.0
        self.min_start_val = min_start_val
        self.max_start_val = max_start_val
        self.min_corner_val = min_corner_val
        self.max_corner_val = max_corner_val
        self.min_end_val = min_end_val
        self.max_end_val = max_end_val
        self.min_attack_frac = min_attack_frac
        self.max_attack_frac = max_attack_frac
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    # TODO(cm): fix alpha application
    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        assert n_frames > 2
        start_val = (
            tr.rand((1,), generator=rand_gen)
            * (self.max_start_val - self.min_start_val)
            + self.min_start_val
        )
        corner_val = (
            tr.rand((1,), generator=rand_gen)
            * (self.max_corner_val - self.min_corner_val)
            + self.min_corner_val
        )
        end_val = (
            tr.rand((1,), generator=rand_gen) * (self.max_end_val - self.min_end_val)
            + self.min_end_val
        )
        corner_frac = (
            tr.rand((1,), generator=rand_gen)
            * (self.max_attack_frac - self.min_attack_frac)
            + self.min_attack_frac
        )
        corner_idx = (n_frames * corner_frac).int().clamp(1, n_frames - 2)

        mod_sig = tr.zeros(n_frames)
        mod_sig[:corner_idx] = tr.linspace(
            start_val.item(), corner_val.item(), corner_idx.item()
        )
        mod_sig[corner_idx:] = tr.linspace(
            corner_val.item(), end_val.item(), n_frames - corner_idx.item()
        )

        alpha = (
            tr.rand((1,), generator=rand_gen) * (self.max_alpha - self.min_alpha)
            + self.min_alpha
        )
        mod_sig = mod_sig.pow(alpha)
        return mod_sig


class ModSignalGeneratorPointy(ModSignalGenerator):
    def __init__(
        self,
        min_diff: float = 0.0,
        max_diff: float = 1.0,
        min_attack_frac: float = 0.0,
        max_attack_frac: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 6.0,
    ):
        super().__init__()
        assert 0.0 <= min_diff <= max_diff <= 1.0
        assert 0.0 <= min_attack_frac <= max_attack_frac <= 1.0
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.min_attack_frac = min_attack_frac
        self.max_attack_frac = max_attack_frac
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        assert n_frames > 2
        corner_frac = util.sample_uniform(
            self.min_attack_frac, self.max_attack_frac, rand_gen=rand_gen
        )
        corner_idx = int(n_frames * corner_frac)
        corner_idx = max(1, corner_idx)
        corner_idx = min(n_frames - 2, corner_idx)
        diff = util.sample_uniform(self.min_diff, self.max_diff, rand_gen=rand_gen)
        if util.sample_uniform(0.0, 1.0, rand_gen=rand_gen) > 0.5:
            start_val = util.sample_uniform(0.0, 1.0 - diff, rand_gen=rand_gen)
            corner_val = start_val + diff
        else:
            start_val = util.sample_uniform(diff, 1.0, rand_gen=rand_gen)
            corner_val = start_val - diff
        end_val = util.sample_uniform(0.0, 1.0, rand_gen=rand_gen)
        mod_sig = tr.zeros(n_frames)
        if util.sample_uniform(0.0, 1.0, rand_gen=rand_gen) > 0.5:
            start_val, end_val = end_val, start_val

        if start_val < corner_val:
            segment_1 = tr.linspace(0.0, 1.0, corner_idx)
        else:
            segment_1 = tr.linspace(1.0, 0.0, corner_idx)
        if corner_val < end_val:
            segment_2 = tr.linspace(0.0, 1.0, n_frames - corner_idx)
        else:
            segment_2 = tr.linspace(1.0, 0.0, n_frames - corner_idx)

        alpha = util.sample_uniform(self.min_alpha, self.max_alpha, rand_gen=rand_gen)
        segment_1 = segment_1.pow(alpha)
        segment_2 = segment_2.pow(alpha)
        segment_1 = segment_1 * abs(corner_val - start_val) + min(start_val, corner_val)
        segment_2 = segment_2 * abs(end_val - corner_val) + min(end_val, corner_val)
        mod_sig[:corner_idx] = segment_1
        mod_sig[corner_idx:] = segment_2

        # import matplotlib.pyplot as plt
        # plt.plot(mod_sig)
        # plt.ylim(0, 1)
        # plt.show()

        return mod_sig
