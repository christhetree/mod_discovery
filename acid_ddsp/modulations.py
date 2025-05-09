import logging
import os
from abc import abstractmethod, ABC
from typing import Optional

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

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
