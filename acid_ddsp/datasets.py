import glob
import json
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
        phase = tr.rand((1,)) * 2 * tr.pi
        phase_hat = tr.rand((1,)) * 2 * tr.pi
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


class NSynthDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        data_dir: str,
        ext: str = "wav",
        split: str = "train",
        note_on_duration: float = 3.0,
    ):
        super().__init__()
        assert os.path.exists(data_dir)
        self.fnames = sorted(glob.glob(f"{data_dir}/*.{ext}"))
        # self.fnames = self.fnames[:5000]

        # TODO(cm): randomize this more?
        # easy train-test split
        if split == "train":
            self.fnames = self.fnames[: int(0.7 * len(self.fnames))]
        elif split == "val":
            self.fnames = self.fnames[
                int(0.7 * len(self.fnames)) : int(0.9 * len(self.fnames))
            ]
        else:
            self.fnames = self.fnames[int(0.9 * len(self.fnames)) :]

        self.ac = ac
        self.note_on_duration = tr.tensor(note_on_duration)

    def get_pitch(self, fname: str) -> float:
        midi_note = int(os.path.basename(fname).split("-")[-2].split("_")[-1])
        f0_hz = tr.tensor(librosa.midi_to_hz(midi_note)).float()
        return f0_hz

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        fname = self.fnames[idx]
        audio, sr = torchaudio.load(fname)
        assert sr == self.ac.sr
        audio = audio.squeeze(0)

        # Pad all nsynth data to the same length
        if audio.size(0) < self.ac.n_samples:
            audio = tr.nn.functional.pad(audio, (0, self.ac.n_samples - audio.size(0)))
        elif audio.size(0) > self.ac.n_samples:
            audio = audio[: self.ac.n_samples]

        assert sr == self.ac.sr
        assert audio.shape[0] == self.ac.n_samples

        # NOTE: NSynth strings filenames are of the form:
        # "<inst_name>_<inst_type>_<inst_str>-<pitch>-<velocity>"
        # midi_note = int(os.path.basename(fname).split("-")[1])
        f0_hz = self.get_pitch(fname)

        # gudgud96: I am not too sure why phase_hat is needed yet...
        phase_hat = tr.rand((1,)) * 2 * tr.pi
        return {
            "wet": audio,
            "f0_hz": f0_hz,
            "note_on_duration": self.note_on_duration,
            "phase_hat": phase_hat,
        }


class SerumDataset(NSynthDataset):
    def __init__(
        self,
        ac: AudioConfig,
        data_dir: str,
        preset_params_path: str,
        ext: str = "wav",
        split: str = "train",
        note_on_duration: float = 3.0,
    ):
        super().__init__(ac, data_dir, ext, split, note_on_duration)
        with open(preset_params_path, "r") as f:
            self.preset_params = json.load(f)

    def get_pitch(self, fname: str) -> int:
        fname = os.path.basename(fname)
        preset_name_str = fname.split("_")
        if len(preset_name_str) != 2:
            assert len(preset_name_str) == 3, f"Error processing {fname}"
            preset_name = "_".join(preset_name_str[:2])
            pitch, velocity = preset_name_str[2].split("-")
        else:
            preset_name = preset_name_str[0]
            pitch, velocity = preset_name_str[1].split("-")

        pitch = int(pitch)
        pitch_corr_a = self.preset_params[preset_name]["pitch"]["A Osc"][
            "pitch_correction"
        ]
        pitch_fine_a = self.preset_params[preset_name]["pitch"]["A Osc"]["pitch_fine"]
        pitch_corr_b = self.preset_params[preset_name]["pitch"]["B Osc"][
            "pitch_correction"
        ]
        pitch_fine_b = self.preset_params[preset_name]["pitch"]["B Osc"]["pitch_fine"]
        if pitch_fine_a or pitch_fine_b:
            log.info(
                f"Fine tuning detected for {preset_name}: "
                f"A Osc: {pitch_fine_a}, B Osc: {pitch_fine_b}"
            )

        osc_pitch_a = pitch + pitch_corr_a
        f0_hz = tr.tensor(librosa.midi_to_hz(osc_pitch_a)).float()
        return f0_hz
