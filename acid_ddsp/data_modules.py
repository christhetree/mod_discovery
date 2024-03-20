import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from acid_ddsp.audio_config import AudioConfig
from acid_ddsp.datasets import AcidSynthDataset, PreprocDataset
from acid_ddsp.modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidDDSPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        mod_sig_gen: ModSignalGenerator,
        train_n_per_epoch: int,
        val_n_per_epoch: int,
        test_n_per_epoch: Optional[int] = None,
        num_workers: int = 0,
    ):
        if test_n_per_epoch is None:
            test_n_per_epoch = val_n_per_epoch

        super().__init__()
        self.save_hyperparameters(ignore=["ac", "mod_sig_gen"])
        log.info(f"\n{self.hparams}")

        self.batch_size = batch_size
        self.ac = ac
        self.mod_sig_gen = mod_sig_gen
        self.train_n_per_epoch = train_n_per_epoch
        self.val_n_per_epoch = val_n_per_epoch
        self.test_n_per_epoch = test_n_per_epoch
        self.num_workers = num_workers

        self.train_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            train_n_per_epoch,
        )
        self.val_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            val_n_per_epoch,
        )
        self.test_ds = AcidSynthDataset(
            ac,
            mod_sig_gen,
            test_n_per_epoch,
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
            shuffle=False,
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


class PreprocDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        data_dir: str,
        ext: str = "wav",
        val_split: float = 0.15,
        test_split: float = 0.15,
        split_seed: int = 42,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ac"])
        log.info(f"\n{self.hparams}")

        assert os.path.isdir(data_dir)
        self.batch_size = batch_size
        self.ac = ac
        self.data_dir = data_dir
        self.ext = ext
        self.val_split = val_split
        self.test_split = test_split
        self.split_seed = split_seed
        self.num_workers = num_workers

        audio_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(ext)
        ]

        n_files = len(audio_paths)
        n_val_files = int(val_split * n_files)
        n_test_files = int(test_split * n_files)
        n_train_files = n_files - n_val_files - n_test_files
        assert n_train_files > 0
        assert n_val_files > 0

        # Shuffle the files
        rng = np.random.default_rng(split_seed)
        rng.shuffle(audio_paths)

        train_paths = audio_paths[:n_train_files]
        val_paths = audio_paths[n_train_files : n_train_files + n_val_files]
        test_paths = audio_paths[n_train_files + n_val_files :]

        self.train_ds = PreprocDataset(ac, train_paths)
        self.val_ds = PreprocDataset(ac, val_paths)
        self.test_ds = PreprocDataset(ac, test_paths)

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
