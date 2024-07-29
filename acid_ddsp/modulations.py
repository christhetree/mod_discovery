import logging
import os
from abc import abstractmethod, ABC

import torch as tr
from torch import Tensor as T
from torch import nn

import util

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModSignalGenerator(ABC, nn.Module):
    @abstractmethod
    def forward(self, n_frames: int) -> T:
        pass


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
    def forward(self, n_frames: int) -> T:
        assert n_frames > 2
        start_val = (
            tr.rand((1,)) * (self.max_start_val - self.min_start_val)
            + self.min_start_val
        )
        corner_val = (
            tr.rand((1,)) * (self.max_corner_val - self.min_corner_val)
            + self.min_corner_val
        )
        end_val = (
            tr.rand((1,)) * (self.max_end_val - self.min_end_val) + self.min_end_val
        )
        corner_frac = (
            tr.rand((1,)) * (self.max_attack_frac - self.min_attack_frac)
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

        alpha = tr.rand((1,)) * (self.max_alpha - self.min_alpha) + self.min_alpha
        mod_sig = mod_sig.pow(alpha)
        return mod_sig


class ModSignalGeneratorLinear(ModSignalGeneratorRandom):
    """
    This is to generate a simple linear modulation signal
    which can cover more than 50% of wavetable position during the sweep.
    We should assess how this affects the learnt wavetable.
    """
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
        super().__init__(
            min_start_val,
            max_start_val,
            min_corner_val,
            max_corner_val,
            min_end_val,
            max_end_val,
            min_attack_frac,
            max_attack_frac,
            min_alpha,
            max_alpha,
        )
    
    def forward(self, n_frames: int) -> T:
        assert n_frames > 2
        start_val = (
            tr.rand((1,)) * (self.max_start_val - self.min_start_val)
            + self.min_start_val
        )
        corner_val = tr.zeros((1,)) if start_val > 0.5 else tr.ones((1,))
        end_val = start_val

        corner_frac = (
            tr.rand((1,)) * (self.max_attack_frac - self.min_attack_frac)
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

    def forward(self, n_frames: int) -> T:
        assert n_frames > 2
        corner_frac = util.sample_uniform(self.min_attack_frac, self.max_attack_frac)
        corner_idx = int(n_frames * corner_frac)
        corner_idx = max(1, corner_idx)
        corner_idx = min(n_frames - 2, corner_idx)
        diff = util.sample_uniform(self.min_diff, self.max_diff)
        if util.sample_uniform(0.0, 1.0) > 0.5:
            start_val = util.sample_uniform(0.0, 1.0 - diff)
            corner_val = start_val + diff
        else:
            start_val = util.sample_uniform(diff, 1.0)
            corner_val = start_val - diff
        end_val = util.sample_uniform(0.0, 1.0)
        mod_sig = tr.zeros(n_frames)
        if util.sample_uniform(0.0, 1.0) > 0.5:
            start_val, end_val = end_val, start_val

        if start_val < corner_val:
            segment_1 = tr.linspace(0.0, 1.0, corner_idx)
        else:
            segment_1 = tr.linspace(1.0, 0.0, corner_idx)
        if corner_val < end_val:
            segment_2 = tr.linspace(0.0, 1.0, n_frames - corner_idx)
        else:
            segment_2 = tr.linspace(1.0, 0.0, n_frames - corner_idx)

        alpha = util.sample_uniform(self.min_alpha, self.max_alpha)
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


if __name__ == "__main__":
    n_frames = 1000
    mod_sig_gen = ModSignalGeneratorLinear()
    mod_sig = mod_sig_gen(n_frames)
    import matplotlib.pyplot as plt

    plt.plot(mod_sig)
    plt.ylim(0, 1)
    plt.savefig("modulation_pointy.png")