import glob
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


class SynthDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        n_per_epoch: int,
        temp_params_name: str,
    ):
        super().__init__()
        self.ac = ac
        self.note_on_duration = tr.tensor(ac.note_on_duration)
        self.mod_sig_gen = mod_sig_gen
        self.n_per_epoch = n_per_epoch
        self.temp_params_name = temp_params_name

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, T]:
        f0_hz = util.sample_log_uniform(self.ac.min_f0_hz, self.ac.max_f0_hz)
        f0_hz = tr.tensor(f0_hz)
        phase = (tr.rand((1,)) * 2 * tr.pi)
        phase_hat = (tr.rand((1,)) * 2 * tr.pi)
        mod_sig = self.mod_sig_gen(self.ac.n_samples).unsqueeze(-1)
        return {
            "f0_hz": f0_hz,
            "note_on_duration": self.note_on_duration,
            "phase": phase,
            "phase_hat": phase_hat,
            self.temp_params_name: mod_sig,
        }


class AcidSynthDataset(SynthDataset):
    def __getitem__(self, idx: int) -> Dict[str, T]:
        out = super().__getitem__(idx)
        q_0to1 = tr.rand((1,)).squeeze(0)
        dist_gain_0to1 = tr.rand((1,)).squeeze(0)
        osc_shape_0to1 = tr.rand((1,)).squeeze(0)
        osc_gain_0to1 = tr.rand((1,)).squeeze(0)
        learned_alpha_0to1 = tr.rand((1,)).squeeze(0)
        out["q_0to1"] = q_0to1
        out["dist_gain_0to1"] = dist_gain_0to1
        out["osc_shape_0to1"] = osc_shape_0to1
        out["osc_gain_0to1"] = osc_gain_0to1
        out["learned_alpha_0to1"] = learned_alpha_0to1
        return out


class PreprocDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        audio_paths: List[str],
    ):
        super().__init__()
        self.ac = ac
        self.audio_paths = audio_paths

        audio_f0_hz = []
        note_on_durations = []
        for audio_path in audio_paths:
            # TODO(cm): modularize this
            tokens = audio_path.split("__")
            midi_f0 = int(tokens[-3])
            f0_hz = tr.tensor(librosa.midi_to_hz(midi_f0)).float()
            audio_f0_hz.append(f0_hz)
            note_on_duration = int(tokens[-2]) / ac.sr
            note_on_duration = tr.tensor(note_on_duration).float()
            note_on_durations.append(note_on_duration)
        self.audio_f0_hz = audio_f0_hz
        self.note_on_durations = note_on_durations

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        audio_path = self.audio_paths[idx]
        f0_hz = self.audio_f0_hz[idx]
        note_on_duration = self.note_on_durations[idx]
        audio, sr = torchaudio.load(audio_path)
        n_samples = audio.size(1)
        assert sr == self.ac.sr
        assert n_samples == self.ac.n_samples
        audio = audio.squeeze(0)
        phase_hat = (tr.rand((1,)) * 2 * tr.pi) - tr.pi
        # TODO(cm): peak normalize?

        return {
            "wet": audio,
            "f0_hz": f0_hz,
            "note_on_duration": note_on_duration,
            "phase_hat": phase_hat,
            "audio_paths": audio_path,
        }


class NSynthStringsDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        temp_params_name: str,
        nsynth_strings_dir: str,
        ext: str = "flac",
        split: str = "train",
    ):
        super().__init__()
        assert os.path.exists(nsynth_strings_dir)
        self.nsynth_strings_fnames = sorted(glob.glob(f"{nsynth_strings_dir}/*.{ext}"))
        
        # easy train-test split
        if split == "train":
            self.nsynth_strings_fnames = self.nsynth_strings_fnames[:int(0.8 * len(self.nsynth_strings_fnames))]
        elif split == "test":
            self.nsynth_strings_fnames = self.nsynth_strings_fnames[int(0.8 * len(self.nsynth_strings_fnames)):int(0.9 * len(self.nsynth_strings_fnames))]
        else:
            self.nsynth_strings_fnames = self.nsynth_strings_fnames[int(0.9 * len(self.nsynth_strings_fnames)):]
        
        self.ac = ac
        self.note_on_duration = tr.tensor(ac.note_on_duration)
        self.temp_params_name = temp_params_name

    def __len__(self) -> int:
        return len(self.nsynth_strings_fnames)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        fname = self.nsynth_strings_fnames[idx]
        audio, sr = torchaudio.load(fname)
        n_samples = audio.size(1)
        assert sr == self.ac.sr
        assert n_samples == self.ac.n_samples
        audio = audio.squeeze(0)

        # NOTE: NSynth strings filenames are of the form:
        # "<inst_name>_<inst_type>_<inst_str>-<pitch>-<velocity>"
        midi_note = int(fname.split("-")[1])
        f0_hz = tr.tensor(librosa.midi_to_hz(midi_note)).float()

        # gudgud96: I am not too sure why phase_hat is needed yet...
        phase_hat = (tr.rand((1,)) * 2 * tr.pi)
        return {
            "wet": audio,
            "f0_hz": f0_hz,
            "note_on_duration": self.note_on_duration,
            "phase_hat": phase_hat,
        }