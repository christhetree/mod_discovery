import logging
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as tr
from torch.utils.data import DataLoader

from audio_config import AudioConfig
from datasets import (
    AcidSynthDataset,
    PreprocDataset,
    SynthDataset,
    NSynthDataset,
    SerumDataset,
    WavetableDataset,
)
from modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class WavetableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        wt_dir: str,
        n_seeds_per_wt: int,
        mod_sig_gen: ModSignalGenerator,
        global_param_names: Optional[List[str]] = None,
        temp_param_names: Optional[List[str]] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.ac = ac
        self.wt_dir = wt_dir
        self.n_seeds_per_wt = n_seeds_per_wt
        self.mod_sig_gen = mod_sig_gen
        if global_param_names is None:
            global_param_names = []
        self.global_param_names = global_param_names
        if temp_param_names is None:
            temp_param_names = ["add_lfo", "sub_lfo", "env"]
        self.temp_param_names = temp_param_names
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        wt_paths = [
            os.path.join(wt_dir, f) for f in os.listdir(wt_dir) if f.endswith(".pt")
        ]
        wt_paths = sorted(wt_paths)
        self.wt_paths = wt_paths
        wts = [tr.load(f) for f in wt_paths]
        n_wts = len(wt_paths)
        n_seeds = n_seeds_per_wt * n_wts
        log.info(f"Found {n_wts} wavetables, total of {n_seeds} seeds")

        wt_indices = np.arange(n_wts)
        seeds = np.arange(n_seeds_per_wt)
        wt_indices = np.repeat(wt_indices, n_seeds_per_wt)
        df = pd.DataFrame({"wt_idx": wt_indices, "seed": seeds})
        n = len(df)
        assert n == n_seeds

        # Shuffle such that batches in validation and test contain a variety of
        # different theta values. This makes the visualization callbacks more diverse.
        df = df.sample(frac=1, random_state=tr.random.initial_seed()).reset_index(
            drop=True
        )
        n_val = int(val_split * n)
        n_test = int(test_split * n)
        n_train = n - n_val - n_test
        log.info(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")

        df_train = df.iloc[:n_train]
        df_val = df.iloc[n_train : n_train + n_val]
        df_test = df.iloc[n_train + n_val :]

        self.train_ds = WavetableDataset(
            ac,
            df_train,
            wts,
            mod_sig_gen,
            global_param_names,
            temp_param_names,
        )
        self.val_ds = WavetableDataset(
            ac,
            df_val,
            wts,
            mod_sig_gen,
            global_param_names,
            temp_param_names,
        )
        self.test_ds = WavetableDataset(
            ac,
            df_test,
            wts,
            mod_sig_gen,
            global_param_names,
            temp_param_names,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,  # For more diversity in visualization callbacks
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )


# TODO(cm): refactor these two into one class
class SynthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        temp_params_name: str,
        train_n_per_epoch: int,
        val_n_per_epoch: int,
        test_n_per_epoch: Optional[int] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        if test_n_per_epoch is None:
            test_n_per_epoch = val_n_per_epoch

        self.batch_size = batch_size
        self.ac = ac
        self.mod_sig_gen = mod_sig_gen
        self.temp_params_name = temp_params_name
        self.train_n_per_epoch = train_n_per_epoch
        self.val_n_per_epoch = val_n_per_epoch
        self.test_n_per_epoch = test_n_per_epoch
        self.num_workers = num_workers

        self.train_ds = SynthDataset(
            ac,
            mod_sig_gen,
            train_n_per_epoch,
            temp_params_name,
        )
        self.val_ds = SynthDataset(
            ac,
            mod_sig_gen,
            val_n_per_epoch,
            temp_params_name,
        )
        self.test_ds = SynthDataset(
            ac,
            mod_sig_gen,
            test_n_per_epoch,
            temp_params_name,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,  # To ensure different visualizations
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )


class AcidDDSPDataModule(SynthDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        temp_params_name: str,
        train_n_per_epoch: int,
        val_n_per_epoch: int,
        test_n_per_epoch: Optional[int] = None,
        num_workers: int = 0,
    ):
        super().__init__(
            batch_size,
            ac,
            mod_sig_gen,
            temp_params_name,
            train_n_per_epoch,
            val_n_per_epoch,
            test_n_per_epoch,
            num_workers,
        )
        self.train_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            train_n_per_epoch,
            temp_params_name,
        )
        self.val_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            val_n_per_epoch,
            temp_params_name,
        )
        self.test_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            test_n_per_epoch,
            temp_params_name,
        )


class PreprocDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        data_dir: str,
        ext: str = "wav",
        val_split: float = 0.2,
        test_split: float = 0.2,
        split_seed: int = 42,
        n_phases_per_file: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.isdir(
            data_dir
        ), f"Data directory {os.path.abspath(data_dir)} does not exist."
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.ext = ext
        self.val_split = val_split
        self.test_split = test_split
        self.split_seed = split_seed
        self.n_phases_per_file = n_phases_per_file
        self.num_workers = num_workers
        if n_phases_per_file > 1:
            log.warning(f"n_phases_per_file: {n_phases_per_file}")

        audio_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(ext)
        ]
        audio_paths = sorted(audio_paths)

        n_files = len(audio_paths)
        n_val_files = int(val_split * n_files)
        n_test_files = int(test_split * n_files)
        n_train_files = n_files - n_val_files - n_test_files
        log.info(
            f"n_train_files: {n_train_files}, n_val_files: {n_val_files}, "
            f"n_test_files: {n_test_files}"
        )
        assert n_train_files > 0
        assert n_val_files > 0

        # Shuffle the files
        rng = np.random.default_rng(split_seed)
        rng.shuffle(audio_paths)

        train_paths = audio_paths[:n_train_files]
        train_paths *= n_phases_per_file
        val_paths = audio_paths[n_train_files : n_train_files + n_val_files]
        val_paths *= n_phases_per_file
        test_paths = audio_paths[n_train_files + n_val_files :]
        test_paths *= n_phases_per_file

        self.train_ds = PreprocDataset(ac, train_paths)
        self.val_ds = PreprocDataset(ac, val_paths)
        self.test_ds = PreprocDataset(ac, test_paths)
        log.info(
            f"train n: {len(self.train_ds)}, val n: {len(self.val_ds)}, "
            f"test n: {len(self.test_ds)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,  # To ensure different visualizations
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )


class NSynthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        data_dir: str,
        ext: str = "wav",
        max_n_files: Optional[int] = None,
        fname_keyword: Optional[str] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.exists(data_dir)
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.max_n_files = max_n_files
        self.fname_keyword = fname_keyword
        self.ext = ext
        self.num_workers = num_workers

        self.train_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keyword,
            split="train",
        )
        self.val_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keyword,
            split="val",
        )
        self.test_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keyword,
            split="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # To ensure different visualizations
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )


class SerumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        data_dir: str,
        preset_params_path: str,
        ext: str = "wav",
        max_n_files: Optional[int] = None,
        fname_keyword: Optional[str] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.exists(data_dir)
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.preset_params_path = preset_params_path
        self.ext = ext
        self.max_n_files = max_n_files
        self.fname_keyword = fname_keyword
        self.num_workers = num_workers

        self.train_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            max_n_files,
            fname_keyword,
            split="train",
        )
        self.val_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            max_n_files,
            fname_keyword,
            split="val",
        )
        self.test_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            split="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # To ensure different visualizations
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
