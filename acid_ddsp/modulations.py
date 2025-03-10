import logging
import os
from abc import abstractmethod, ABC
from typing import Optional

import torch as tr
from torch import Tensor as T
from torch import nn

import util
from curves import PiecewiseBezier

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModSignalGenerator(ABC, nn.Module):
    @abstractmethod
    def forward(self, n_frames: int, rand_gen: Optional[tr.Generator] = None) -> T:
        pass


class ModSignalGenRandomBezier(ModSignalGenerator):
    def __init__(
        self,
        min_n_seg: int = 1,
        max_n_seg: int = 8,
        min_degree: int = 1,
        max_degree: int = 3,
        min_seg_interval_frac: float = 0.25,
        softmax_tau: float = 0.25,
        is_c1_cont: bool = False,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 1 <= min_n_seg <= max_n_seg
        self.min_n_seg = min_n_seg
        self.max_n_seg = max_n_seg
        assert 1 <= min_degree <= max_degree
        self.min_degree = min_degree
        self.max_degree = max_degree
        assert 0.0 < min_seg_interval_frac <= 1.0
        self.min_seg_interval_frac = min_seg_interval_frac
        self.softmax_tau = softmax_tau
        assert not is_c1_cont, "C1 continuity not supported currently"
        if is_c1_cont:
            assert min_degree >= 3
        self.is_c1_cont = is_c1_cont
        self.normalize = normalize
        self.eps = eps

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
        if self.min_seg_interval_frac == 1.0:
            modes = None
        else:
            si_logits = tr.rand((1, n_seg), generator=rand_gen)
            min_seg_interval = (1 / n_seg) * self.min_seg_interval_frac
            si = PiecewiseBezier.logits_to_seg_intervals(
                si_logits, min_seg_interval, self.softmax_tau
            )
            # log.info(f"si_logits = {si_logits}")
            # log.info(f"si = {si}")
            modes = tr.cumsum(si, dim=1)[:, :-1]
        bezier = PiecewiseBezier(
            n_frames,
            n_seg,
            degree=degree,
            modes=modes,
            is_c1_cont=self.is_c1_cont,
        )
        cp = (
            tr.rand((1, (n_seg * degree) + 1), generator=rand_gen)
            .unfold(dimension=1, size=degree + 1, step=degree)
            .contiguous()
        )

        mod_sig = bezier.make_bezier(cp=cp)
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
