import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Mapping

import auraloss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch as tr
from auraloss.time import ESRLoss
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MFCC
from tqdm import tqdm

import acid_ddsp.util as util
from acid_ddsp.audio_config import AudioConfig
from fad import save_and_concat_fad_audio, calc_fad
from feature_extraction import LogMelSpecFeatureExtractor
from paths import OUT_DIR
from synths import AcidSynthLPBiquad, make_synth

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
        fad_model_names: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        if synth_hat_kwargs is None:
            synth_hat_kwargs = {}
        if synth_eval_kwargs is None:
            synth_eval_kwargs = {}
        if "interp_logits" in synth_hat_kwargs and "interp_logits" in synth_eval_kwargs:
            assert (
                synth_hat_kwargs["interp_logits"] == synth_eval_kwargs["interp_logits"]
            )
        if fad_model_names is None:
            fad_model_names = []
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        log.info(f"Run name: {self.run_name}")
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
        self.fad_model_names = fad_model_names

        self.loss_name = self.loss_func.__class__.__name__
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()
        self.mss = auraloss.freq.MultiResolutionSTFTLoss()
        self.mel_stft = auraloss.freq.MelSTFTLoss(
            sample_rate=self.ac.sr,
            fft_size=spectral_visualizer.n_fft,
            hop_size=spectral_visualizer.hop_len,
            win_length=spectral_visualizer.n_fft,
            n_mels=spectral_visualizer.n_mels,
        )
        self.mfcc = MFCC(
            sample_rate=self.ac.sr,
            log_mels=True,
            melkwargs={
                "n_fft": spectral_visualizer.n_fft,
                "hop_length": spectral_visualizer.hop_len,
                "n_mels": spectral_visualizer.n_mels,
            },
        )
        self.global_n = 0
        self.test_outs = []

        # TODO(cm): refactor this to reduce duplicate code
        self.is_q_learnable = ac.min_q != ac.max_q
        self.is_dist_gain_learnable = ac.min_dist_gain != ac.max_dist_gain
        self.is_osc_shape_learnable = ac.min_osc_shape != ac.max_osc_shape
        self.is_osc_gain_learnable = ac.min_osc_gain != ac.max_osc_gain
        self.is_alpha_learnable = ac.min_learned_alpha != ac.max_learned_alpha

        self.synth = AcidSynthLPBiquad(ac, batch_size)
        self.synth_hat = make_synth(synth_hat_type, ac, batch_size, **synth_hat_kwargs)
        self.synth_eval = make_synth(
            synth_eval_type, ac, batch_size, **synth_eval_kwargs
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(state_dict, strict=False)

    def on_train_start(self) -> None:
        self.global_n = 0

    def mfcc_l1_loss(self, x: T, y: T) -> T:
        x_mfcc = self.mfcc(x)
        y_mfcc = self.mfcc(y)
        return self.l1(x_mfcc, y_mfcc)

    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase = batch["phase"]
        mod_sig = batch["mod_sig"]
        q_norm = batch["q_norm"]
        dist_gain_norm = batch["dist_gain_norm"]
        osc_shape_norm = batch["osc_shape_norm"]
        osc_gain_norm = batch["osc_gain_norm"]
        learned_alpha_norm = batch["learned_alpha_norm"]
        batch_size = f0_hz.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        assert phase.shape == (batch_size,)
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
                osc_shape,
                osc_gain,
                note_on_duration,
                filter_args,
                dist_gain,
                learned_alpha,
                phase,
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
        phase_hat = batch["phase_hat"]

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
        _, wet_hat, envelope_hat, a_coeff, b_coeff = self.synth_hat(
            f0_hz,
            osc_shape_hat,
            osc_gain_hat,
            note_on_duration,
            filter_args_hat,
            dist_gain_hat,
            learned_alpha_hat,
            phase_hat,
        )
        # TODO(cm): refactor
        if envelope is None:
            envelope = envelope_hat

        # Compute loss
        if self.use_p_loss:
            assert mod_sig is not None
            assert mod_sig_hat is not None
            if mod_sig_hat.shape != mod_sig.shape:
                assert mod_sig_hat.ndim == mod_sig.ndim == 2
                mod_sig_hat = util.linear_interpolate_dim(
                    mod_sig_hat, mod_sig.size(1), dim=1, align_corners=True
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
            audio_mel_stft = self.mel_stft(wet_hat.unsqueeze(1), wet.unsqueeze(1))
            audio_mfcc = self.mfcc_l1_loss(wet_hat.unsqueeze(1), wet.unsqueeze(1))
        self.log(f"{stage}/audio_mss", audio_mss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/audio_mel", audio_mel_stft, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/audio_mfcc", audio_mfcc, prog_bar=False, sync_dist=True)

        # Log mod_sig_hat metrics
        if mod_sig is not None and mod_sig_hat is not None:
            if mod_sig_hat.shape != mod_sig.shape:
                assert mod_sig_hat.ndim == mod_sig.ndim == 2
                mod_sig_hat = util.linear_interpolate_dim(
                    mod_sig_hat, mod_sig.size(1), dim=1, align_corners=True
                )
            with tr.no_grad():
                mod_sig_esr = self.esr(mod_sig_hat, mod_sig)
                mod_sig_l1 = self.l1(mod_sig_hat, mod_sig)
            self.log(f"{stage}/ms_esr", mod_sig_esr, prog_bar=False, sync_dist=True)
            self.log(f"{stage}/ms_l1", mod_sig_l1, prog_bar=True, sync_dist=True)

        # Log eval synth metrics if possible
        wet_eval = None
        audio_mss_eval = None
        audio_mel_stft_eval = None
        audio_mfcc_eval = None
        if stage != "train":
            try:
                _, wet_eval, _, a_coeff_eval, b_coeff_eval = self.synth_eval(
                    f0_hz,
                    osc_shape_hat,
                    osc_gain_hat,
                    note_on_duration,
                    filter_args_hat,
                    dist_gain_hat,
                    learned_alpha_hat,
                    phase_hat,
                )
            except Exception as e:
                log.error(f"Error in eval synth: {e}")
            if wet_eval is not None:
                audio_mss_eval = self.mss(wet_eval.unsqueeze(1), wet.unsqueeze(1))
                audio_mel_stft_eval = self.mel_stft(
                    wet_eval.unsqueeze(1), wet.unsqueeze(1)
                )
                audio_mfcc_eval = self.mfcc_l1_loss(
                    wet_eval.unsqueeze(1), wet.unsqueeze(1)
                )
                self.log(
                    f"{stage}/audio_mss_{self.synth_eval.__class__.__name__}",
                    audio_mss_eval,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/audio_mel_{self.synth_eval.__class__.__name__}",
                    audio_mel_stft_eval,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/audio_mfcc_{self.synth_eval.__class__.__name__}",
                    audio_mfcc_eval,
                    prog_bar=False,
                    sync_dist=True,
                )

        # H = util.calc_h(a_coeff, b_coeff)

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
            "audio_mss": audio_mss,
            "audio_mss_eval": audio_mss_eval,
            "audio_mel_stft": audio_mel_stft,
            "audio_mel_stft_eval": audio_mel_stft_eval,
            "audio_mfcc": audio_mfcc,
            "audio_mfcc_eval": audio_mfcc_eval,
            # "H": H,
        }
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        out = self.step(batch, stage="test")
        self.test_outs.append(out)
        return out

    def on_test_epoch_end(self) -> None:
        loss = tr.stack([out["loss"] for out in self.test_outs], dim=0).mean()
        audio_mss = tr.stack([out["audio_mss"] for out in self.test_outs], dim=0).mean()
        audio_mss_eval = None
        if self.test_outs[0].get("audio_mss_eval") is not None:
            audio_mss_eval = tr.stack(
                [out["audio_mss_eval"] for out in self.test_outs], dim=0
            ).mean()
        audio_mel_stft = tr.stack(
            [out["audio_mel_stft"] for out in self.test_outs], dim=0
        ).mean()
        audio_mel_stft_eval = None
        if self.test_outs[0].get("audio_mel_stft_eval") is not None:
            audio_mel_stft_eval = tr.stack(
                [out["audio_mel_stft_eval"] for out in self.test_outs], dim=0
            ).mean()
        audio_mfcc = tr.stack(
            [out["audio_mfcc"] for out in self.test_outs], dim=0
        ).mean()
        audio_mfcc_eval = None
        if self.test_outs[0].get("audio_mfcc_eval") is not None:
            audio_mfcc_eval = tr.stack(
                [out["audio_mfcc_eval"] for out in self.test_outs], dim=0
            ).mean()

        tsv_data = [
            ["loss", loss.item()],
            ["mss", audio_mss.item()],
            ["mss_eval", audio_mss_eval.item() if audio_mss_eval else None],
            ["mel_stft", audio_mel_stft.item()],
            [
                "mel_stft_eval",
                audio_mel_stft_eval.item() if audio_mel_stft_eval else None,
            ],
            ["mfcc", audio_mfcc.item()],
            ["mfcc_eval", audio_mfcc_eval.item() if audio_mfcc_eval else None],
        ]

        # TODO(cm): refactor
        if self.fad_model_names:
            wet = tr.cat([out["x"] for out in self.test_outs], dim=0).detach().cpu()
            wet_hat = (
                tr.cat([out["x_hat"] for out in self.test_outs], dim=0).detach().cpu()
            )
            wet_eval = None
            if self.test_outs[0].get("x_eval") is not None:
                wet_eval = (
                    tr.cat([out["x_eval"] for out in self.test_outs], dim=0)
                    .detach()
                    .cpu()
                )
            fad_values = []
            fad_eval_values = []
            epoch_n = 68
            for idx in range(0, wet.size(0), epoch_n):
                curr_wet = wet[idx : idx + epoch_n, :]
                curr_wet_hat = wet_hat[idx : idx + epoch_n, :]
                curr_wet_eval = None
                if wet_eval is not None:
                    curr_wet_eval = wet_eval[idx : idx + epoch_n, :]
                fad_wet_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet")
                save_and_concat_fad_audio(
                    self.ac.sr,
                    curr_wet,
                    fad_wet_dir,
                    fade_n_samples=self.spectral_visualizer.hop_len,
                )

                fad_wet_hat_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet_hat")
                save_and_concat_fad_audio(
                    self.ac.sr,
                    curr_wet_hat,
                    fad_wet_hat_dir,
                    fade_n_samples=self.spectral_visualizer.hop_len,
                )
                fad_wet_eval_dir = os.path.join(
                    OUT_DIR, f"{self.run_name}__fad_wet_eval"
                )
                if wet_eval is not None:
                    save_and_concat_fad_audio(
                        self.ac.sr,
                        curr_wet_eval,
                        fad_wet_eval_dir,
                        fade_n_samples=self.spectral_visualizer.hop_len,
                    )

                for fad_model_name in tqdm(self.fad_model_names):
                    fad_hat = calc_fad(
                        fad_model_name,
                        fad_wet_dir,
                        fad_wet_hat_dir,
                        clean_up_baseline=False,
                        clean_up_eval=True,
                    )
                    fad_values.append(fad_hat)
                    # self.log(
                    #     f"test/fad_{fad_model_name}_eval_{idx}", fad_hat, prog_bar=False
                    # )
                    tsv_data.append([f"fad_{fad_model_name}", fad_hat])
                    fad_eval = None
                    if wet_eval is not None:
                        fad_eval = calc_fad(
                            fad_model_name,
                            fad_wet_dir,
                            fad_wet_eval_dir,
                            clean_up_baseline=True,
                            clean_up_eval=True,
                        )
                        fad_eval_values.append(fad_eval)
                        # self.log(
                        #     f"test/fad_{fad_model_name}_{self.synth_eval.__class__.__name__}",
                        #     fad_eval,
                        #     prog_bar=False,
                        # )
                    tsv_data.append([f"fad_{fad_model_name}_eval_{idx}", fad_eval])

            log.info(f"Test FAD hat values: {fad_values}")
            log.info(f"Test FAD eval values: {fad_eval_values}")
            ci_hat = 1.96 * scipy.stats.sem(fad_values)
            ci_eval = 1.96 * scipy.stats.sem(fad_eval_values)
            tsv_data.append([f"fad_{fad_model_name}_mean", np.mean(fad_values)])
            tsv_data.append([f"fad_{fad_model_name}_std", np.std(fad_values)])
            tsv_data.append([f"fad_{fad_model_name}_ci95", ci_hat])
            tsv_data.append([f"fad_{fad_model_name}_eval_mean", np.mean(fad_eval_values)])
            tsv_data.append([f"fad_{fad_model_name}_eval_std", np.std(fad_eval_values)])
            tsv_data.append([f"fad_{fad_model_name}_eval_ci95", ci_eval])

        tsv_path = os.path.join(OUT_DIR, f"{self.run_name}__test.tsv")
        if os.path.exists(tsv_path):
            log.warning(f"Overwriting existing TSV file: {tsv_path}")
        df = pd.DataFrame(tsv_data, columns=["metric", "value"])
        df.to_csv(tsv_path, sep="\t", index=False)

        # H = tr.cat([out["H"] for out in self.test_outs], dim=0)
        # H = H.detach().cpu()
        # H_path = os.path.join(OUT_DIR, f"{self.run_name}__H.pt")
        # tr.save(H, H_path)
        # log.info(f"Saved H to: {H_path}")


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
