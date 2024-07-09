import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from audio_config import AudioConfig
from datasets import AcidSynthDataset, PreprocDataset, SynthDataset, NSynthStringsDataset
from modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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


class NSynthStringsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        nsynth_strings_dir: str,
        ext: str = "wav",
        num_workers: int = 0,
    ):
        super().__init__()
        assert os.path.exists(nsynth_strings_dir)
        self.batch_size = batch_size
        self.ac = ac
        self.nsynth_strings_dir = nsynth_strings_dir
        self.ext = ext
        self.num_workers = num_workers

        self.train_ds = NSynthStringsDataset(
            ac,
            nsynth_strings_dir,
            ext,
            "train",
        )
        self.val_ds = NSynthStringsDataset(
            ac,
            nsynth_strings_dir,
            ext,
            "val",
        )
        self.test_ds = NSynthStringsDataset(
            ac,
            nsynth_strings_dir,
            ext,
            "test",
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
            shuffle=False,
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
