import logging
import os

import torch as tr
from torch import Tensor as T
from torch.utils.data import Dataset

from acid_ddsp.modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MidiF0ModSignalDataset(Dataset):
    def __init__(
        self,
        min_midi_f0: int,
        max_midi_f0: int,
        note_on_duration: float,
        mod_sig_gen: ModSignalGenerator,
        n_frames: int,
        n_per_epoch: int,
    ):
        super().__init__()
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.note_on_duration = tr.tensor(note_on_duration)
        self.mod_sig_gen = mod_sig_gen
        self.n_frames = n_frames
        self.num_examples_per_epoch = n_per_epoch

    def __len__(self) -> int:
        return self.num_examples_per_epoch

    def __getitem__(self, idx: int) -> (T, T, T):
        midi_f0 = tr.randint(self.min_midi_f0, self.max_midi_f0 + 1, (1,))
        mod_sig = self.mod_sig_gen(self.n_frames)
        return midi_f0, self.note_on_duration, mod_sig
