import logging
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from acid_ddsp.datasets import MidiF0ModSignalDataset
from acid_ddsp.modulations import ModSignalGenerator
from audio_config import AudioConfig

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
        n_frames: Optional[int] = None,
        num_workers: int = 0,
    ):
        if test_n_per_epoch is None:
            test_n_per_epoch = val_n_per_epoch
        if n_frames is None:
            n_frames = int(ac.sr * ac.buffer_size_seconds)

        super().__init__()
        self.save_hyperparameters()
        log.info(f"\n{self.hparams}")

        self.batch_size = batch_size
        self.ac = ac
        self.mod_sig_gen = mod_sig_gen
        self.train_n_per_epoch = train_n_per_epoch
        self.val_n_per_epoch = val_n_per_epoch
        self.test_n_per_epoch = test_n_per_epoch
        self.n_frames = n_frames
        self.num_workers = num_workers

        self.train_ds = MidiF0ModSignalDataset(
            ac,
            mod_sig_gen,
            n_frames,
            train_n_per_epoch,
        )
        self.val_ds = MidiF0ModSignalDataset(
            ac,
            mod_sig_gen,
            n_frames,
            val_n_per_epoch,
        )
        self.test_ds = MidiF0ModSignalDataset(
            ac,
            mod_sig_gen,
            n_frames,
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
