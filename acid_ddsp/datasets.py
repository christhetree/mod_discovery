import glob
import json
import logging
import os
import random
from typing import Dict, List, Optional

import librosa
import torch as tr
import torchaudio
from pandas import DataFrame
from torch import Tensor as T
from torch.utils.data import Dataset

import acid_ddsp.util as util
from acid_ddsp.audio_config import AudioConfig
from acid_ddsp.modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SeedDataset(Dataset):
    def __init__(
        self,
        ac: AudioConfig,
        df: DataFrame,
        mod_sig_gens: List[ModSignalGenerator],
        global_param_names: List[str],
        temp_param_names: List[str],
        n_frames: Optional[int] = None,
        randomize_seed: bool = False,
    ):
        super().__init__()
        assert "seed" in df.columns
        assert len(mod_sig_gens) == len(temp_param_names) or len(mod_sig_gens) == 1
        if len(mod_sig_gens) == 1:
            mod_sig_gens = mod_sig_gens * len(temp_param_names)
        self.ac = ac
        self.df = df
        self.mod_sig_gens = mod_sig_gens
        self.global_param_names = global_param_names
        self.temp_param_names = temp_param_names
        self.randomize_seed = randomize_seed
        if n_frames is None:
            n_frames = ac.n_samples
        self.n_frames = n_frames

        # To ensure it's of type float32
        self.note_on_duration = tr.tensor(ac.note_on_duration).float()
        self.rand_gen = tr.Generator(device="cpu")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        seed = self.df.iloc[idx]["seed"].item()
        # TODO(cm): this is a bit hacky
        if self.randomize_seed:
            seed = tr.randint(0, 99999999, (1,)).item()

        f0_hz = util.sample_log_uniform(self.ac.min_f0_hz, self.ac.max_f0_hz, seed=seed)
        f0_hz = tr.tensor(f0_hz)
        self.rand_gen.manual_seed(seed)
        phase = tr.rand((1,), generator=self.rand_gen).squeeze() * 2 * tr.pi
        phase_hat = tr.rand((1,), generator=self.rand_gen).squeeze() * 2 * tr.pi

        result = {
            "note_on_duration": self.note_on_duration,
            "f0_hz": f0_hz,
            "phase": phase,
            "phase_hat": phase_hat,
        }
        for name in self.global_param_names:
            assert "_0to1" not in name
            result[f"{name}_0to1"] = tr.rand((1,), generator=self.rand_gen).squeeze()
        for name, mod_sig_gen in zip(self.temp_param_names, self.mod_sig_gens):
            mod_sig = mod_sig_gen(self.n_frames, rand_gen=self.rand_gen)
            result[name] = mod_sig
        return result


class WavetableDataset(SeedDataset):
    def __init__(
        self,
        ac: AudioConfig,
        df: DataFrame,
        mod_sig_gens: List[ModSignalGenerator],
        global_param_names: List[str],
        temp_param_names: List[str],
        randomize_seed: bool = False,
    ):
        super().__init__(
            ac, df, mod_sig_gens, global_param_names, temp_param_names, randomize_seed
        )
        assert "wt_idx" in df.columns

    def __getitem__(self, idx: int) -> Dict[str, T]:
        result = super().__getitem__(idx)
        wt_idx = self.df.iloc[idx]["wt_idx"].item()
        result["wt_idx"] = wt_idx
        return result


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
        max_n_files: Optional[int] = None,
        fname_keywords: Optional[List[str]] = None,
        split_name: str = "train",
        split_train: float = 0.6,
        split_val: float = 0.2,
        shuffle_seed: int = 42,
    ):
        super().__init__()
        assert os.path.exists(data_dir)
        self.ac = ac
        self.shuffle_seed = shuffle_seed

        fnames = sorted(glob.glob(f"{data_dir}/*.{ext}"))
        log.info(f"Found {len(fnames)} files")
        if fname_keywords:
            log.info(f"Filtering filenames with keywords: '{fname_keywords}'")
            fnames = [f for f in fnames if any(k in f for k in fname_keywords)]
            log.info(f"Filtered down to {len(fnames)} files")

        rand = random.Random(shuffle_seed)
        rand.shuffle(fnames)
        if max_n_files is not None:
            log.info(f"Limiting number of files to {max_n_files}")
            fnames = fnames[:max_n_files]
        seeds = list(range(len(fnames)))

        n_files = len(fnames)
        if split_name == "train":
            start_idx = 0
            end_idx = int(split_train * n_files)
        elif split_name == "val":
            start_idx = int(split_train * n_files)
            end_idx = int((split_train + split_val) * n_files)
        else:
            start_idx = int((split_train + split_val) * n_files)
            end_idx = n_files

        self.fnames = fnames[start_idx:end_idx]
        log.info(f"{split_name}: found {n_files} files, using {len(self.fnames)}")
        self.seeds = seeds[start_idx:end_idx]

        # To ensure it's of type float32
        self.note_on_duration = tr.tensor(ac.note_on_duration).float()
        self.rand_gen = tr.Generator(device="cpu")

    def get_f0_hz(self, fname: str) -> float:
        midi_note = int(os.path.basename(fname).split("-")[-2].split("_")[-1])
        f0_hz = float(librosa.midi_to_hz(midi_note))
        return f0_hz

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Dict[str, T]:
        fname = self.fnames[idx]
        seed = self.seeds[idx]

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
        f0_hz = self.get_f0_hz(fname)
        f0_hz = tr.tensor(f0_hz).float()
        self.rand_gen.manual_seed(seed)
        phase_hat = tr.rand((1,), generator=self.rand_gen).squeeze() * 2 * tr.pi
        return {
            "audio": audio,
            "note_on_duration": self.note_on_duration,
            "f0_hz": f0_hz,
            "phase_hat": phase_hat,
        }


class SerumDataset(NSynthDataset):
    def __init__(
        self,
        ac: AudioConfig,
        data_dir: str,
        preset_params_path: str,
        ext: str = "wav",
        max_n_files: Optional[int] = None,
        fname_keyword: Optional[str] = None,
        split_name: str = "train",
        split_train: float = 0.6,
        split_val: float = 0.2,
        shuffle_seed: int = 42,
    ):
        super().__init__(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keyword,
            split_name,
            split_train,
            split_val,
            shuffle_seed,
        )
        with open(preset_params_path, "r") as f:
            self.preset_params = json.load(f)
        # TODO(cm): cleanup
        for k, v in self.preset_params.items():
            on_a_0to1 = float(v["parameters"][212]["text"])
            on_a = round(on_a_0to1)

            oct_a_0to1 = float(v["parameters"][3]["text"])
            oct_a = round(oct_a_0to1 * 8.0) - 4
            semi_a_0to1 = float(v["parameters"][4]["text"])
            semi_a = round(semi_a_0to1 * 24.0) - 12

            pitch_correction_a = oct_a * 12 + semi_a
            pitch_correction_a_2 = v["pitch"]["A Osc"]["pitch_correction"]
            assert pitch_correction_a == pitch_correction_a_2

            cents_a_0to1 = float(v["parameters"][5]["text"])
            cents_a = round(cents_a_0to1 * 200.0) - 100
            cents_a_2 = v["pitch"]["A Osc"]["pitch_fine"]
            assert cents_a == cents_a_2

            coarse_a_0to1 = float(v["parameters"][10]["text"])
            coarse_a = round(coarse_a_0to1 * 128.0 - 64.0, 6)
            # if coarse_a != 0.0:
            #     log.info(f"Coarse tuning detected for {k}: {coarse_a}")
            v["pitch"]["A Osc"]["on"] = on_a
            v["pitch"]["A Osc"]["coarse"] = coarse_a

            on_b_0to1 = float(v["parameters"][213]["text"])
            on_b = round(on_b_0to1)

            oct_b_0to1 = float(v["parameters"][16]["text"])
            oct_b = round(oct_b_0to1 * 8.0) - 4
            semi_b_0to1 = float(v["parameters"][17]["text"])
            semi_b = round(semi_b_0to1 * 24.0) - 12

            pitch_correction_b = oct_b * 12 + semi_b
            pitch_correction_b_2 = v["pitch"]["B Osc"]["pitch_correction"]
            assert pitch_correction_b == pitch_correction_b_2

            cents_b_0to1 = float(v["parameters"][18]["text"])
            cents_b = round(cents_b_0to1 * 200.0) - 100
            cents_b_2 = v["pitch"]["B Osc"]["pitch_fine"]
            assert cents_b == cents_b_2

            coarse_b_0to1 = float(v["parameters"][23]["text"])
            coarse_b = round(coarse_b_0to1 * 128.0 - 64.0, 6)
            v["pitch"]["B Osc"]["on"] = on_b
            v["pitch"]["B Osc"]["coarse"] = coarse_b

            if not on_a and not on_b:
                log.debug(f"Neither A nor B Osc is on for {k}")

    def get_f0_hz(self, fname: str) -> float:
        fname = os.path.basename(fname)
        preset_name_str = fname.split("_")
        if len(preset_name_str) != 2:
            assert len(preset_name_str) == 3, f"Error processing {fname}"
            preset_name = "_".join(preset_name_str[:2])
            keyboard_pitch, velocity = preset_name_str[2].split("-")
        else:
            preset_name = preset_name_str[0]
            keyboard_pitch, velocity = preset_name_str[1].split("-")

        keyboard_pitch = int(keyboard_pitch)

        pitch_data_a = self.preset_params[preset_name]["pitch"]["A Osc"]
        pitch_data_b = self.preset_params[preset_name]["pitch"]["B Osc"]

        if pitch_data_a["on"]:
            f0_hz = self.calc_f0_hz(keyboard_pitch, pitch_data_a)
        elif pitch_data_b["on"]:
            f0_hz = self.calc_f0_hz(keyboard_pitch, pitch_data_b)
        else:
            f0_hz = float(librosa.midi_to_hz(keyboard_pitch))

        return f0_hz

    @staticmethod
    def calc_f0_hz(keyboard_pitch: int, pitch_data: Dict[str, int | float]) -> float:
        pitch_correction = pitch_data["pitch_correction"]
        cents = pitch_data["pitch_fine"]
        coarse = pitch_data["coarse"]
        pitch = keyboard_pitch + pitch_correction + coarse + (cents / 100.0)
        pitch_int = int(pitch)
        pitch_frac = pitch - pitch_int
        pitch_int_f0_hz = float(librosa.midi_to_hz(pitch_int))
        pitch_int_f0_hz_p1 = float(librosa.midi_to_hz(pitch_int + 1))
        delta_hz = pitch_int_f0_hz_p1 - pitch_int_f0_hz
        f0_hz_frac = pitch_frac * delta_hz
        f0_hz = pitch_int_f0_hz + f0_hz_frac
        return f0_hz
