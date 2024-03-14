import logging
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from acid_ddsp.datasets import AcidSynthDataset
from acid_ddsp.modulations import ModSignalGenerator
from acid_ddsp.audio_config import AudioConfig

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
