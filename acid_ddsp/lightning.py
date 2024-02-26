import logging
import os
from typing import Dict

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn

import acid_ddsp.util as util
from acid_ddsp.audio_config import AudioConfig
from acid_ddsp.filters import TimeVaryingBiquad
from acid_ddsp.synth_modules import ADSRValues
from acid_ddsp.synths import CustomSynth
from torchsynth.config import SynthConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidDDSPLightingModule(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        ac: AudioConfig,
        model: nn.Module,
        loss_func: nn.Module,
        use_p_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ac", "model", "loss_func"])
        log.info(f"\n{self.hparams}")

        self.batch_size = batch_size
        self.ac = ac
        self.model = model
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss

        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.global_n = 0

        sc = SynthConfig(
            batch_size=batch_size,
            sample_rate=ac.sr,
            buffer_size_seconds=ac.buffer_size_seconds,
            control_rate=ac.control_rate,
            reproducible=False,
            no_grad=True,
            debug=False,
        )
        adsr_vals = ADSRValues(
            attack=ac.attack,
            decay=ac.decay,
            sustain=ac.sustain,
            release=ac.release,
            alpha=ac.alpha,
        )
        self.synth = CustomSynth(synthconfig=sc, adsr_vals=adsr_vals)

        self.tvb = TimeVaryingBiquad(
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
        )

    def on_train_start(self) -> None:
        self.global_n = 0

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        midi_f0, note_on_duration, mod_sig = batch
        batch_size = self.batch_size
        assert midi_f0.shape == (batch_size,)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        # TODO(cm): check if this works for DDP
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        # Generate ground truth audio x
        with tr.no_grad():
            (audio, envelope), _, _ = self.synth(
                midi_f0=midi_f0, note_on_duration=note_on_duration
            )
            assert audio.shape == (self.batch_size, int(self.ac.buffer_size_seconds * self.ac.sr))
            audio *= 0.5  # TODO(cm): record this somewhere
            if mod_sig.shape != audio.shape:
                assert mod_sig.ndim == audio.ndim
                assert mod_sig.size(-1) < audio.size(-1)
                mod_sig = util.linear_interpolate_last_dim(
                    mod_sig, audio.size(-1), align_corners=True
                )
            x = self.tvb(audio, cutoff_mod_sig=mod_sig)
            assert x.shape == audio.shape

        # Extract mod_sig_hat
        model_in = tr.stack([audio, x], dim=1)
        mod_sig_hat, latent, log_spec = self.model(model_in)
        mod_sig_hat = mod_sig_hat.squeeze(1)
        log_spec_dry = log_spec[:, 0, :, :]
        log_spec_wet = log_spec[:, 1, :, :]

        # import torchaudio
        # from matplotlib import pyplot as plt
        # for idx in range(x.size(0)):
        #     torchaudio.save(
        #         f"../out/audio_{idx}.wav", audio[idx].unsqueeze(0), self.ac.sr
        #     )
        #     torchaudio.save(f"../out/x_{idx}.wav", x[idx].unsqueeze(0), self.ac.sr)
        #     # plt.plot(envelope[idx].numpy())
        #     plt.plot(mod_sig[idx].numpy())
        #     plt.ylim(0, 1)
        #     plt.show()
        #     # plt.imshow(log_spec_dry[idx].numpy(), aspect="auto", origin="lower", cmap="magma")
        #     # plt.title("Dry")
        #     # plt.show()
        #     plt.imshow(
        #         log_spec_wet[idx, :, :].numpy(), aspect="auto", origin="lower", cmap="magma"
        #     )
        #     plt.title("Wet")
        #     plt.show()
        # exit()

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
                f"{stage}/ploss_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            x_hat = self.tvb(audio, cutoff_mod_sig=mod_sig_hat)
            assert x_hat.shape == x.shape
            loss = self.loss_func(x_hat.unsqueeze(1), x.unsqueeze(1))
            self.log(f"{stage}/audio_{self.loss_name}", loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/ms_l1", mod_sig_mae, prog_bar=True, sync_dist=True)
        # self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        out_dict = {
            "loss": loss,
            "mod_sig": mod_sig,
            "mod_sig_hat": mod_sig_hat,
            "x": x,
            "x_hat": x_hat,
            "audio": audio,
            "envelope": envelope,
            "log_spec_dry": log_spec_dry,
            "log_spec_wet": log_spec_wet,
        }
        return out_dict

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")
