import logging
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as tr
from torch.utils.data import DataLoader

from acid_ddsp.datasets import (
    NSynthDataset,
    SerumDataset,
    SeedDataset,
)
from acid_ddsp.modulations import ModSignalGenerator
from audio_config import AudioConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SeedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        n_seeds: int,
        mod_sig_gens: List[ModSignalGenerator],
        global_param_names: Optional[List[str]] = None,
        temp_param_names: Optional[List[str]] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n_frames: Optional[int] = None,
        randomize_train_seed: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.ac = ac
        self.n_seeds = n_seeds
        self.mod_sig_gens = mod_sig_gens
        if global_param_names is None:
            global_param_names = []
        self.global_param_names = global_param_names
        if temp_param_names is None:
            temp_param_names = ["add_lfo", "sub_lfo", "env"]
        self.temp_param_names = temp_param_names
        self.val_split = val_split
        self.test_split = test_split
        self.n_frames = n_frames
        self.randomize_train_seed = randomize_train_seed
        self.num_workers = num_workers

        seeds = np.arange(n_seeds)
        df = pd.DataFrame({"seed": seeds})
        n = len(df)

        # Shuffle such that batches in validation and test contain a variety of
        # different theta values. This makes the visualization callbacks more diverse.
        log.info(f"Shuffling dataset with seed: {tr.random.initial_seed()}")
        df = df.sample(frac=1, random_state=tr.random.initial_seed()).reset_index(
            drop=True
        )

        n_val = int(val_split * n)
        n_test = int(test_split * n)
        n_train = n - n_val - n_test
        df_train = df.iloc[:n_train]
        df_val = df.iloc[n_train : n_train + n_val]
        df_test = df.iloc[n_train + n_val :]

        log.info(
            f"n_train: {len(df_train)}, n_val: {len(df_val)}, n_test: {len(df_test)}"
        )

        self.train_ds = SeedDataset(
            ac,
            df_train,
            mod_sig_gens,
            global_param_names,
            temp_param_names,
            n_frames=n_frames,
            randomize_seed=randomize_train_seed,
        )
        self.val_ds = SeedDataset(
            ac,
            df_val,
            mod_sig_gens,
            global_param_names,
            temp_param_names,
            n_frames=n_frames,
        )
        self.test_ds = SeedDataset(
            ac,
            df_test,
            mod_sig_gens,
            global_param_names,
            temp_param_names,
            n_frames=n_frames,
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
            shuffle=False,
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
        fname_keywords: Optional[List[str]] = None,
        split_train: float = 0.6,
        split_val: float = 0.2,
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.ext = ext
        self.max_n_files = max_n_files
        self.fname_keywords = fname_keywords
        self.split_train = split_train
        self.split_val = split_val
        self.num_workers = num_workers

        self.train_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keywords,
            split_name="train",
            split_train=split_train,
            split_val=split_val,
        )
        self.val_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keywords,
            split_name="val",
            split_train=split_train,
            split_val=split_val,
        )
        self.test_ds = NSynthDataset(
            ac,
            data_dir,
            ext,
            max_n_files,
            fname_keywords,
            split_name="test",
            split_train=split_train,
            split_val=split_val,
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
            shuffle=False,
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


class SerumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        data_dir: str,
        preset_params_path: str,
        ext: str = "wav",
        max_n_files: Optional[int] = None,
        fname_keywords: Optional[List[str]] = None,
        split_train: float = 0.6,
        split_val: float = 0.2,
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.preset_params_path = preset_params_path
        self.ext = ext
        self.max_n_files = max_n_files
        self.fname_keywords = fname_keywords
        self.split_train = split_train
        self.split_val = split_val
        self.num_workers = num_workers

        self.train_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            max_n_files,
            fname_keywords,
            split_name="train",
            split_train=split_train,
            split_val=split_val,
        )
        self.val_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            max_n_files,
            fname_keywords,
            split_name="val",
            split_train=split_train,
            split_val=split_val,
        )
        self.test_ds = SerumDataset(
            ac,
            data_dir,
            preset_params_path,
            ext,
            max_n_files,
            fname_keywords,
            split_name="test",
            split_train=split_train,
            split_val=split_val,
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
            shuffle=False,
            # shuffle=True,
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
