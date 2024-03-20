import logging
import os
from typing import Dict, List

import librosa
import torch as tr
import torchaudio
from torch import Tensor as T
from torch.utils.data import Dataset

import util
from acid_ddsp.audio_config import AudioConfig
from acid_ddsp.modulations import ModSignalGenerator

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


class PreprocDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        audio_paths: List[str],
    ):
        super().__init__()
        self.ac = ac
        self.audio_paths = audio_paths
        self.note_on_duration = tr.tensor(ac.note_on_duration)

        audio_f0_hz = []
        for audio_path in audio_paths:
            midi_f0 = int(audio_path.split("_")[-1].split(".")[0])
            f0_hz = librosa.midi_to_hz(midi_f0)
            audio_f0_hz.append(f0_hz)
        self.audio_f0_hz = audio_f0_hz

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        audio_path = self.audio_paths[idx]
        f0_hz = self.audio_f0_hz[idx]
        audio, sr = torchaudio.load(audio_path)
        n_samples = audio.size(1)
        assert sr == self.ac.sr
        assert n_samples == self.ac.n_samples

        return {
            "wet": audio,
            "f0_hz": f0_hz,
            "note_on_duration": self.note_on_duration,
        }
