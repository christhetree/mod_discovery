import logging
import os
from typing import Dict, Any, Optional

import auraloss
import pytorch_lightning as pl
import torch as tr
from auraloss.time import ESRLoss
from torch import Tensor as T
from torch import nn

import acid_ddsp.util as util
from acid_ddsp.audio_config import AudioConfig
from feature_extraction import LogMelSpecFeatureExtractor
from synths import AcidSynth, make_synth

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
        synth_hat_type: str = "AcidSynth",
        synth_hat_kwargs: Optional[Dict[str, Any]] = None,
        synth_eval_type: str = "AcidSynth",
        synth_eval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if synth_hat_kwargs is None:
            synth_hat_kwargs = {}
        if synth_eval_kwargs is None:
            synth_eval_kwargs = {}
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
        self.mss = auraloss.freq.MultiResolutionSTFTLoss()
        self.global_n = 0

        # TODO(cm): refactor this to reduce duplicate code
        self.is_q_learnable = ac.min_q != ac.max_q
        self.is_dist_gain_learnable = ac.min_dist_gain != ac.max_dist_gain
        self.is_osc_shape_learnable = ac.min_osc_shape != ac.max_osc_shape
        self.is_osc_gain_learnable = ac.min_osc_gain != ac.max_osc_gain
        self.is_alpha_learnable = ac.min_learned_alpha != ac.max_learned_alpha

        self.synth = AcidSynth(ac, batch_size)
        self.synth_hat = make_synth(synth_hat_type, ac, batch_size, **synth_hat_kwargs)
        self.synth_eval = make_synth(
            synth_eval_type, ac, batch_size, **synth_eval_kwargs
        )

    def on_train_start(self) -> None:
        self.global_n = 0

    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        osc_arg = batch["osc_arg"]
        mod_sig = batch["mod_sig"]
        q_norm = batch["q_norm"]
        dist_gain_norm = batch["dist_gain_norm"]
        osc_shape_norm = batch["osc_shape_norm"]
        osc_gain_norm = batch["osc_gain_norm"]
        learned_alpha_norm = batch["learned_alpha_norm"]
        batch_size = f0_hz.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        assert osc_arg.shape == (batch_size, self.ac.n_samples)
        assert mod_sig.shape == (batch_size, self.ac.n_samples)
        assert q_norm.shape == (batch_size,)
        assert dist_gain_norm.shape == (batch_size,)
        assert osc_shape_norm.shape == (batch_size,)
        assert osc_gain_norm.shape == (batch_size,)
        assert learned_alpha_norm.shape == (batch_size,)

        q = q_norm * (self.ac.max_q - self.ac.min_q) + self.ac.min_q
        dist_gain = (
            dist_gain_norm * (self.ac.max_dist_gain - self.ac.min_dist_gain)
            + self.ac.min_dist_gain
        )
        osc_shape = (
            osc_shape_norm * (self.ac.max_osc_shape - self.ac.min_osc_shape)
            + self.ac.min_osc_shape
        )
        osc_gain = (
            osc_gain_norm * (self.ac.max_osc_gain - self.ac.min_osc_gain)
            + self.ac.min_osc_gain
        )
        learned_alpha = (
            learned_alpha_norm * (self.ac.max_learned_alpha - self.ac.min_learned_alpha)
            + self.ac.min_learned_alpha
        )

        # Generate ground truth wet audio
        with tr.no_grad():
            q_mod_sig = q_norm.unsqueeze(-1)
            filter_args = {
                "w_mod_sig": mod_sig,
                "q_mod_sig": q_mod_sig,
            }
            dry, wet, envelope = self.synth(
                f0_hz,
                osc_arg,
                osc_shape,
                osc_gain,
                note_on_duration,
                filter_args,
                dist_gain,
                learned_alpha,
            )
            assert dry.shape == wet.shape == (batch_size, self.ac.n_samples)

        batch["q"] = q
        batch["dist_gain"] = dist_gain
        batch["osc_shape"] = osc_shape
        batch["osc_gain"] = osc_gain
        batch["learned_alpha"] = learned_alpha
        batch["dry"] = dry
        batch["wet"] = wet
        batch["envelope"] = envelope
        return batch

    def step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        batch_size = self.batch_size
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        # TODO(cm): check if this works for DDP
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        batch = self.preprocess_batch(batch)
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        wet = batch["wet"]
        osc_arg_hat = batch["osc_arg_hat"]

        mod_sig = batch.get("mod_sig")
        q_norm = batch.get("q_norm")
        q = batch.get("q")
        dist_gain_norm = batch.get("dist_gain_norm")
        dist_gain = batch.get("dist_gain")
        osc_shape_norm = batch.get("osc_shape_norm")
        osc_shape = batch.get("osc_shape")
        osc_gain_norm = batch.get("osc_gain_norm")
        osc_gain = batch.get("osc_gain")
        learned_alpha_norm = batch.get("learned_alpha_norm")
        learned_alpha = batch.get("learned_alpha")
        dry = batch.get("dry")
        envelope = batch.get("envelope")

        # Perform model forward pass
        model_in = wet.unsqueeze(1)
        model_out = self.model(model_in)
        filter_args_hat = {}

        # Postprocess mod_sig_hat
        mod_sig_hat = model_out.get("mod_sig_hat")
        if mod_sig_hat is not None:
            filter_args_hat["w_mod_sig"] = mod_sig_hat

        # Postprocess q_hat
        q_norm_hat = model_out.get("q_norm_hat")
        q_hat = None
        if q_norm_hat is not None:
            if self.is_q_learnable and q_norm is not None:
                with tr.no_grad():
                    q_norm_l1 = self.l1(q_norm_hat, q_norm)
                self.log(f"{stage}/q_l1", q_norm_l1, prog_bar=False, sync_dist=True)
            q_mod_sig_hat = q_norm_hat.unsqueeze(-1)
            filter_args_hat["q_mod_sig"] = q_mod_sig_hat
            q_hat = q_norm_hat * (self.ac.max_q - self.ac.min_q) + self.ac.min_q

        # Postprocess dist_gain_hat
        dist_gain_norm_hat = model_out["dist_gain_norm_hat"]
        dist_gain_hat = (
            dist_gain_norm_hat * (self.ac.max_dist_gain - self.ac.min_dist_gain)
            + self.ac.min_dist_gain
        )
        if self.is_dist_gain_learnable and dist_gain_norm is not None:
            with tr.no_grad():
                dist_gain_norm_l1 = self.l1(dist_gain_norm_hat, dist_gain_norm)
            self.log(
                f"{stage}/dg_l1", dist_gain_norm_l1, prog_bar=False, sync_dist=True
            )

        # Postprocess osc_shape_hat
        osc_shape_norm_hat = model_out["osc_shape_norm_hat"]
        osc_shape_hat = (
            osc_shape_norm_hat * (self.ac.max_osc_shape - self.ac.min_osc_shape)
            + self.ac.min_osc_shape
        )
        if self.is_osc_shape_learnable and osc_shape_norm is not None:
            with tr.no_grad():
                osc_shape_norm_l1 = self.l1(osc_shape_norm_hat, osc_shape_norm)
            self.log(
                f"{stage}/os_l1", osc_shape_norm_l1, prog_bar=False, sync_dist=True
            )

        # Postprocess osc_gain_hat
        osc_gain_norm_hat = model_out["osc_gain_norm_hat"]
        osc_gain_hat = (
            osc_gain_norm_hat * (self.ac.max_osc_gain - self.ac.min_osc_gain)
            + self.ac.min_osc_gain
        )
        if self.is_osc_gain_learnable and osc_gain_norm is not None:
            with tr.no_grad():
                osc_gain_norm_l1 = self.l1(osc_gain_norm_hat, osc_gain_norm)
            self.log(f"{stage}/og_l1", osc_gain_norm_l1, prog_bar=False, sync_dist=True)

        # Postprocess learned_alpha_hat
        learned_alpha_norm_hat = model_out["learned_alpha_norm_hat"]
        learned_alpha_hat = (
            learned_alpha_norm_hat
            * (self.ac.max_learned_alpha - self.ac.min_learned_alpha)
            + self.ac.min_learned_alpha
        )
        if self.is_alpha_learnable and learned_alpha_norm is not None:
            with tr.no_grad():
                learned_alpha_norm_l1 = self.l1(
                    learned_alpha_norm_hat, learned_alpha_norm
                )
            self.log(
                f"{stage}/la_l1", learned_alpha_norm_l1, prog_bar=False, sync_dist=True
            )

        # Postprocess log_spec
        log_spec = model_out.get("log_spec")
        if log_spec is None:
            log_spec = self.spectral_visualizer(model_in)
        assert log_spec.ndim == 4
        log_spec_x = log_spec[:, 0, :, :]

        # Postprocess logits
        logits = model_out.get("logits")
        if logits is not None:
            filter_args_hat["logits"] = logits

        # Generate audio x_hat
        _, wet_hat, envelope_hat = self.synth_hat(
            f0_hz,
            osc_arg_hat,
            osc_shape_hat,
            osc_gain_hat,
            note_on_duration,
            filter_args_hat,
            dist_gain_hat,
            learned_alpha_hat,
        )
        # TODO(cm): refactor
        if envelope is None:
            envelope = envelope_hat

        # Compute loss
        if self.use_p_loss:
            assert mod_sig is not None
            assert mod_sig_hat is not None
            if mod_sig_hat.shape != mod_sig.shape:
                assert mod_sig_hat.ndim == mod_sig.ndim
                mod_sig_hat = util.linear_interpolate_last_dim(
                    mod_sig_hat, mod_sig.size(-1), align_corners=True
                )
            loss = self.loss_func(mod_sig_hat, mod_sig)
            if self.is_q_learnable and q_norm_hat is not None:
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
            if self.is_osc_gain_learnable:
                osc_gain_loss = self.loss_func(osc_gain_norm_hat, osc_gain_norm)
                self.log(
                    f"{stage}/ploss_{self.loss_name}_og",
                    osc_gain_loss,
                    prog_bar=False,
                    sync_dist=True,
                )
                loss += osc_gain_loss
            if self.is_alpha_learnable:
                learned_alpha_loss = self.loss_func(
                    learned_alpha_norm_hat, learned_alpha_norm
                )
                self.log(
                    f"{stage}/ploss_{self.loss_name}_la",
                    learned_alpha_loss,
                    prog_bar=False,
                    sync_dist=True,
                )
                loss += learned_alpha_loss
            self.log(
                f"{stage}/ploss_{self.loss_name}", loss, prog_bar=False, sync_dist=True
            )
        else:
            loss = self.loss_func(wet_hat.unsqueeze(1), wet.unsqueeze(1))
            self.log(
                f"{stage}/audio_{self.loss_name}", loss, prog_bar=False, sync_dist=True
            )

        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        # Log MSS loss
        with tr.no_grad():
            audio_mss = self.mss(wet_hat.unsqueeze(1), wet.unsqueeze(1))
        self.log(f"{stage}/audio_mss", audio_mss, prog_bar=True, sync_dist=True)

        # Log mod_sig_hat metrics
        if mod_sig is not None and mod_sig_hat is not None:
            if mod_sig_hat.shape != mod_sig.shape:
                assert mod_sig_hat.ndim == mod_sig.ndim
                mod_sig_hat = util.linear_interpolate_last_dim(
                    mod_sig_hat, mod_sig.size(-1), align_corners=True
                )
            with tr.no_grad():
                mod_sig_esr = self.esr(mod_sig_hat, mod_sig)
                mod_sig_l1 = self.l1(mod_sig_hat, mod_sig)
            self.log(f"{stage}/ms_esr", mod_sig_esr, prog_bar=False, sync_dist=True)
            self.log(f"{stage}/ms_l1", mod_sig_l1, prog_bar=True, sync_dist=True)

        # Log eval synth metrics if possible
        wet_eval = None
        if stage != "train":
            try:
                _, wet_eval, _ = self.synth_eval(
                    f0_hz,
                    osc_arg_hat,
                    osc_shape_hat,
                    osc_gain_hat,
                    note_on_duration,
                    filter_args_hat,
                    dist_gain_hat,
                    learned_alpha_hat,
                )
            except Exception as e:
                log.error(f"Error in eval synth: {e}")
            if wet_eval is not None:
                audio_mss_eval = self.mss(wet_eval.unsqueeze(1), wet.unsqueeze(1))
                self.log(
                    f"{stage}/audio_mss_{self.synth_eval.__class__.__name__}",
                    audio_mss_eval,
                    prog_bar=False,
                    sync_dist=True,
                )

        out_dict = {
            "loss": loss,
            "mod_sig": mod_sig,
            "mod_sig_hat": mod_sig_hat,
            "x": wet,
            "x_hat": wet_hat,
            "x_eval": wet_eval,
            "osc_audio": dry,
            "envelope": envelope,
            "log_spec_x": log_spec_x,
            "q": q,
            "q_hat": q_hat,
            "dist_gain": dist_gain,
            "dist_gain_hat": dist_gain_hat,
            "osc_shape": osc_shape,
            "osc_shape_hat": osc_shape_hat,
            "osc_gain": osc_gain,
            "osc_gain_hat": osc_gain_hat,
            "learned_alpha": learned_alpha,
            "learned_alpha_hat": learned_alpha_hat,
        }
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")


class PreprocLightningModule(AcidDDSPLightingModule):
    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        wet = batch["wet"]
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        assert wet.ndim == 2
        assert wet.size(1) == self.ac.n_samples
        batch_size = wet.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        batch["wet"] = wet
        return batch
