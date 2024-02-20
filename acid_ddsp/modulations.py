import logging
import os

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModSignalGenerator(nn.Module):
    def __init__(
        self,
        n_frames: int,
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
        assert n_frames > 2
        assert 0.0 <= min_start_val <= max_start_val <= 1.0
        assert 0.0 <= min_corner_val <= max_corner_val <= 1.0
        assert 0.0 <= min_end_val <= max_end_val <= 1.0
        assert 0.0 <= min_attack_frac <= max_attack_frac <= 1.0
        self.n_frames = n_frames
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

    def forward(self) -> T:
        start_val = (
            tr.rand() * (self.max_start_val - self.min_start_val) + self.min_start_val
        )
        corner_val = (
            tr.rand() * (self.max_corner_val - self.min_corner_val)
            + self.min_corner_val
        )
        end_val = tr.rand() * (self.max_end_val - self.min_end_val) + self.min_end_val
        corner_frac = (
            tr.rand() * (self.max_attack_frac - self.min_attack_frac)
            + self.min_attack_frac
        )
        corner_idx = (self.n_frames * corner_frac).int().clamp(1, self.n_frames - 2)

        mod_sig = tr.zeros(self.n_frames)
        mod_sig[:corner_idx] = tr.linspace(
            start_val.item(), corner_val.item(), corner_idx.item()
        )
        mod_sig[corner_idx:] = tr.linspace(
            corner_val.item(), end_val.item(), self.n_frames - corner_idx.item()
        )

        alpha = tr.rand() * (self.max_alpha - self.min_alpha) + self.min_alpha
        mod_sig = mod_sig.pow(alpha)
        return mod_sig
