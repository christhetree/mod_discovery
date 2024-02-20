import logging
import os
from typing import Dict

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn

import util
from filters import TimeVaryingBiquad
from synth_modules import ADSRValues
from synths import CustomSynth
from torchsynth.config import SynthConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        sc: SynthConfig,
        adsr_vals: ADSRValues,
        tvb: TimeVaryingBiquad,
        loss_func: nn.Module,
        use_p_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_func", "model"])
        log.info(f"\n{self.hparams}")

        self.model = model
        self.sc = sc
        self.adsr_vals = adsr_vals
        self.tvb = tvb
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss

        self.synth = CustomSynth(synthconfig=sc, adsr_vals=adsr_vals)

        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.global_n = 0

    def on_train_start(self) -> None:
        self.global_n = 0

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        midi_f0, note_on_duration, mod_sig = batch

        # Generate ground truth audio x
        with tr.no_grad():
            (audio, envelope), _, _ = self.synth(
                midi_f0=midi_f0, note_on_duration=note_on_duration
            )
            audio *= 0.5  # TODO(cm): record this somewhere
            if mod_sig.shape != audio.shape:
                assert mod_sig.ndim == audio.ndim
                assert mod_sig.size(-1) < audio.size(-1)
                mod_sig = util.linear_interpolate_last_dim(
                    mod_sig, audio.size(-1), align_corners=True
                )
            x = self.tvb(audio, cutoff_mod_sig=mod_sig)

        # Extract mod_sig_hat
        mod_sig_hat = self.model(x)

        if mod_sig_hat.shape != mod_sig.shape:
            assert mod_sig_hat.ndim == mod_sig.ndim
            assert mod_sig_hat.size(-1) < mod_sig.size(-1)
            mod_sig_hat = util.linear_interpolate_last_dim(
                mod_sig_hat, mod_sig.size(-1), align_corners=True
            )
        mod_sig_mae = self.l1(mod_sig_hat, mod_sig)

        x_hat = None
        if self.use_p_loss:
            loss = self.loss_func(mod_sig_hat, mod_sig)
            self.log(
                f"{stage}/ms_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            x_hat = self.tvb(audio, cutoff_mod_sig=mod_sig_hat)
            assert x_hat.shape == x.shape
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/x_{self.loss_name}", loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/ms_l1", mod_sig_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        out_dict = {
            "loss": loss,
            "mod_sig": mod_sig,
            "mod_sig_hat": mod_sig_hat,
            "x": x,
            "x_hat": x_hat,
            "audio": audio,
            "envelope": envelope,
        }
        return out_dict

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")
