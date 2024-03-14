import logging
import os
from typing import Dict

import torch as tr
from torch import Tensor as T
from torch.utils.data import Dataset

from acid_ddsp.modulations import ModSignalGenerator
from acid_ddsp.audio_config import AudioConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MidiF0ModSignalDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        n_frames: int,
        n_per_epoch: int,
    ):
        super().__init__()
        self.ac = ac
        self.note_on_duration = tr.tensor(ac.note_on_duration)
        self.mod_sig_gen = mod_sig_gen
        self.n_frames = n_frames
        self.num_examples_per_epoch = n_per_epoch

    def __len__(self) -> int:
        return self.num_examples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, T]:
        midi_f0 = tr.randint(self.ac.min_midi_f0, self.ac.max_midi_f0 + 1, (1,))
        midi_f0 = midi_f0.squeeze(0)
        osc_shape = tr.rand((1,))
        osc_shape = osc_shape.squeeze(0)
        mod_sig = self.mod_sig_gen(self.n_frames)
        q_norm = tr.rand((1,))
        q_norm = q_norm.squeeze(0)
        dist_gain_norm = tr.rand((1,))
        dist_gain_norm = dist_gain_norm.squeeze(0)
        return {
            "midi_f0": midi_f0,
            "osc_shape": osc_shape,
            "note_on_duration": self.note_on_duration,
            "mod_sig": mod_sig,
            "q_norm": q_norm,
            "dist_gain_norm": dist_gain_norm,
        }
