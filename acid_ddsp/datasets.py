import logging
import os
from typing import Dict

import torch as tr
from torch import Tensor as T
from torch.utils.data import Dataset

import util
from acid_ddsp.modulations import ModSignalGenerator
from acid_ddsp.audio_config import AudioConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        n_per_epoch: int,
    ):
        super().__init__()
        self.ac = ac
        self.note_on_duration = tr.tensor(ac.note_on_duration)
        self.mod_sig_gen = mod_sig_gen
        self.num_examples_per_epoch = n_per_epoch

    def __len__(self) -> int:
        return self.num_examples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, T]:
        f0_hz = util.sample_log_uniform(self.ac.min_f0_hz, self.ac.max_f0_hz)
        f0_hz = tr.tensor(f0_hz)
        mod_sig = self.mod_sig_gen(self.ac.n_samples)
        q_norm = tr.rand((1,)).squeeze(0)
        dist_gain_norm = tr.rand((1,)).squeeze(0)
        osc_shape_norm = tr.rand((1,)).squeeze(0)
        return {
            "f0_hz": f0_hz,
            "note_on_duration": self.note_on_duration,
            "mod_sig": mod_sig,
            "q_norm": q_norm,
            "dist_gain_norm": dist_gain_norm,
            "osc_shape_norm": osc_shape_norm,
        }
