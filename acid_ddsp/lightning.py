import logging
import os
from typing import Dict

import pytorch_lightning as pl
import torch as tr
from auraloss.time import ESRLoss
from torch import Tensor as T
from torch import nn

import acid_ddsp.util as util
from acid_ddsp.audio_config import AudioConfig
from feature_extraction import LogMelSpecFeatureExtractor
from synths import AcidSynth, AcidSynthLSTM, AcidSynthLearnedBiquad

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
        spectral_visualizer: LogMelSpecFeatureExtractor,
        use_p_loss: bool = False,
        log_envelope: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["ac", "model", "loss_func", "spectral_visualizer"]
        )
        log.info(f"\n{self.hparams}")

        self.batch_size = batch_size
        self.ac = ac
        self.model = model
        self.loss_func = loss_func
        self.spectral_visualizer = spectral_visualizer
        self.use_p_loss = use_p_loss
        self.log_envelope = log_envelope

        self.loss_name = self.loss_func.__class__.__name__
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()
        self.global_n = 0
        self.is_q_learnable = ac.min_q != ac.max_q
        self.is_dist_gain_learnable = ac.min_dist_gain != ac.max_dist_gain
        self.is_osc_shape_learnable = ac.min_osc_shape != ac.max_osc_shape

        self.synth = AcidSynth(ac, batch_size)
        self.synth_hat = AcidSynth(ac, batch_size)
        # self.synth_hat = AcidSynthLearnedBiquad(ac, batch_size)
        # self.synth_hat = AcidSynthLSTM(ac, batch_size, n_hidden=128)

    def on_train_start(self) -> None:
        self.global_n = 0

    def step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        mod_sig = batch["mod_sig"]
        q_norm = batch["q_norm"]
        dist_gain_norm = batch["dist_gain_norm"]
        osc_shape_norm = batch["osc_shape_norm"]

        q = q_norm * (self.ac.max_q - self.ac.min_q) + self.ac.min_q
        dist_gain = (
            dist_gain_norm * (self.ac.max_dist_gain - self.ac.min_dist_gain)
            + self.ac.min_dist_gain
        )
        osc_shape = (
            osc_shape_norm * (self.ac.max_osc_shape - self.ac.min_osc_shape)
            + self.ac.min_osc_shape
        )

        batch_size = self.batch_size
        assert f0_hz.shape == (batch_size,)
        assert osc_shape.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        assert mod_sig.shape == (batch_size, self.ac.n_samples)
        assert q_norm.shape == (batch_size,)
        assert dist_gain_norm.shape == (batch_size,)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        # TODO(cm): check if this works for DDP
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        # Generate ground truth audio x
        with tr.no_grad():
            q_mod_sig = tr.ones_like(mod_sig) * q_norm.unsqueeze(-1)
            dry, wet, envelope = self.synth(
                f0_hz, osc_shape, note_on_duration, mod_sig, q_mod_sig, dist_gain
            )
            assert dry.shape == wet.shape == (self.batch_size, self.ac.n_samples)

        # Extract mod_sig_hat
        model_in = wet.unsqueeze(1)
        model_out = self.model(model_in)
        mod_sig_hat = model_out["mod_sig_hat"]
        q_norm_hat = model_out["q_norm_hat"]
        dist_gain_norm_hat = model_out["dist_gain_norm_hat"]
        osc_shape_norm_hat = model_out["osc_shape_norm_hat"]
        log_spec = model_out["log_spec"]
        # a_coeff_hat = model_out.get("a_coeff_hat", None)
        # b_coeff_hat = model_out.get("b_coeff_hat", None)

        mod_sig_hat = mod_sig_hat.squeeze(1)
        q_hat = q_norm_hat * (self.ac.max_q - self.ac.min_q) + self.ac.min_q
        dist_gain_hat = (
            dist_gain_norm_hat * (self.ac.max_dist_gain - self.ac.min_dist_gain)
            + self.ac.min_dist_gain
        )
        osc_shape_hat = (
            osc_shape_norm_hat * (self.ac.max_osc_shape - self.ac.min_osc_shape)
            + self.ac.min_osc_shape
        )

        log_spec_audio = None
        if log_spec is None:
            log_spec = self.spectral_visualizer(model_in)
        assert log_spec.ndim == 4
        log_spec_x = log_spec[:, 0, :, :]
        if log_spec.size(1) == 2:
            log_spec_audio = log_spec[:, 1, :, :]

        # import torchaudio
        # from matplotlib import pyplot as plt
        # for idx in range(x.size(0)):
        #     # torchaudio.save(
        #     #     f"../out/audio_{idx}.wav", osc_audio[idx].unsqueeze(0), self.ac.sr
        #     # )
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
            # a_coeff_hat = tr.swapaxes(a_coeff_hat, 1, 2)
            # a_coeff_hat = util.linear_interpolate_last_dim(
            #     a_coeff_hat, mod_sig.size(-1), align_corners=True
            # )
            # a_coeff_hat = tr.swapaxes(a_coeff_hat, 1, 2)
            # b_coeff_hat = tr.swapaxes(b_coeff_hat, 1, 2)
            # b_coeff_hat = util.linear_interpolate_last_dim(
            #     b_coeff_hat, mod_sig.size(-1), align_corners=True
            # )
            # b_coeff_hat = tr.swapaxes(b_coeff_hat, 1, 2)

        with tr.no_grad():
            mod_sig_esr = self.esr(mod_sig_hat, mod_sig)
            mod_sig_l1 = self.l1(mod_sig_hat, mod_sig)
            q_norm_l1 = 0.0
            if self.is_q_learnable:
                q_norm_l1 = self.l1(q_norm_hat, q_norm)
            dist_gain_norm_l1 = 0.0
            if self.is_dist_gain_learnable:
                dist_gain_norm_l1 = self.l1(dist_gain_norm_hat, dist_gain_norm)
            osc_shape_norm_l1 = 0.0
            if self.is_osc_shape_learnable:
                osc_shape_norm_l1 = self.l1(osc_shape_norm_hat, osc_shape_norm)

        wet_hat = None
        if self.use_p_loss:
            loss = self.loss_func(mod_sig_hat, mod_sig)
            if self.is_q_learnable:
                q_loss = self.loss_func(q_norm_hat, q_norm)
                self.log(
                    f"{stage}/ploss_{self.loss_name}_q",
                    q_loss,
                    prog_bar=False,
                    sync_dist=True,
                )
                loss += q_loss
            if self.is_dist_gain_learnable:
                dist_gain_loss = self.loss_func(dist_gain_norm_hat, dist_gain_norm)
                self.log(
                    f"{stage}/ploss_{self.loss_name}_dg",
                    dist_gain_loss,
                    prog_bar=False,
                    sync_dist=True,
                )
                loss += dist_gain_loss
            if self.is_osc_shape_learnable:
                osc_shape_loss = self.loss_func(osc_shape_norm_hat, osc_shape_norm)
                self.log(
                    f"{stage}/ploss_{self.loss_name}_os",
                    osc_shape_loss,
                    prog_bar=False,
                    sync_dist=True,
                )
                loss += osc_shape_loss
            self.log(
                f"{stage}/ploss_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            q_mod_sig_hat = tr.ones_like(mod_sig_hat) * q_norm_hat.unsqueeze(-1)
            _, wet_hat, _ = self.synth_hat(
                f0_hz,
                osc_shape_hat,
                note_on_duration,
                mod_sig_hat,
                q_mod_sig_hat,
                dist_gain_hat,
            )
            loss = self.loss_func(wet_hat.unsqueeze(1), wet.unsqueeze(1))
            self.log(
                f"{stage}/audio_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )

        self.log(f"{stage}/ms_esr", mod_sig_esr, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/ms_l1", mod_sig_l1, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/q_l1", q_norm_l1, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/dg_l1", dist_gain_norm_l1, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/os_l1", osc_shape_norm_l1, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        out_dict = {
            "loss": loss,
            "mod_sig": mod_sig,
            "mod_sig_hat": mod_sig_hat,
            "x": wet,
            "x_hat": wet_hat,
            "osc_audio": dry,
            "envelope": envelope,
            "log_spec_audio": log_spec_audio,
            "log_spec_x": log_spec_x,
            "q": q,
            "q_hat": q_hat,
            "dist_gain": dist_gain,
            "dist_gain_hat": dist_gain_hat,
            "osc_shape": osc_shape,
            "osc_shape_hat": osc_shape_hat,
        }
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")
