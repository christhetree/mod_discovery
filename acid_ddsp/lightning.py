import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Mapping, Tuple

import auraloss
import pytorch_lightning as pl
import torch as tr
from auraloss.time import ESRLoss
from pytorch_lightning.cli import OptimizerCallable
from torch import Tensor as T
from torch import nn

import util
from audio_config import AudioConfig
from fad import save_and_concat_fad_audio, calc_fad
from feature_extraction import LogMelSpecFeatureExtractor
from losses import MFCCL1
from metrics import (
    SpectralCentroidMetric,
    RMSMetric,
    SpectralBandwidthMetric,
    SpectralFlatnessMetric, EntropyMetric, TotalVariationMetric, TurningPointsMetric,
    SpectralEntropyMetric,
)
from modulations import ModSignalGenRandomBezier
from paths import OUT_DIR
from synths import SynthBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidDDSPLightingModule(pl.LightningModule):
    def __init__(
        self,
        ac: AudioConfig,
        loss_func: nn.Module,
        spectral_visualizer: LogMelSpecFeatureExtractor,
        model: Optional[nn.Module] = None,
        synth: Optional[SynthBase] = None,
        synth_hat: Optional[SynthBase] = None,
        temp_param_names: Optional[List[str]] = None,
        temp_param_names_hat: Optional[List[str]] = None,
        interp_temp_param_names_hat: Optional[List[str]] = None,
        global_param_names: Optional[List[str]] = None,
        global_param_names_hat: Optional[List[str]] = None,
        use_p_loss: bool = False,
        fad_model_names: Optional[List[str]] = None,
        run_name: Optional[str] = None,
        use_model: bool = True,
        model_opt: Optional[OptimizerCallable] = None,
        synth_opt: Optional[OptimizerCallable] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert synth_hat is not None
        if temp_param_names is None:
            temp_param_names = []
        if temp_param_names_hat is None:
            temp_param_names_hat = []
        if interp_temp_param_names_hat is None:
            interp_temp_param_names_hat = []
        if global_param_names is None:
            global_param_names = []
        if global_param_names_hat is None:
            global_param_names_hat = []
        if fad_model_names is None:
            fad_model_names = []
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        else:
            self.run_name = run_name
        if use_model:
            assert model is not None
        log.info(f"Run name: {self.run_name}")
        assert ac.sr == spectral_visualizer.sr
        if hasattr(loss_func, "sr"):
            assert loss_func.sr == ac.sr

        self.ac = ac
        self.model = model
        self.loss_func = loss_func
        self.spectral_visualizer = spectral_visualizer
        self.synth = synth
        self.synth_hat = synth_hat
        self.temp_param_names = temp_param_names
        self.temp_param_names_hat = temp_param_names_hat
        self.interp_temp_param_names_hat = interp_temp_param_names_hat
        self.global_param_names = global_param_names
        self.global_param_names_hat = global_param_names_hat
        self.use_p_loss = use_p_loss
        self.fad_model_names = fad_model_names
        self.use_model = use_model
        self.model_opt = model_opt
        self.synth_opt = synth_opt
        self.eps = eps

        self.loss_name = self.loss_func.__class__.__name__
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.audio_metrics = nn.ModuleDict()
        self.audio_metrics["mss"] = auraloss.freq.MultiResolutionSTFTLoss()
        metrics_win_len = int(0.0500 * self.ac.sr)  # 50 milliseconds
        metrics_hop_len = int(0.0125 * self.ac.sr)  # 25% overlap
        metrics_n_mels = 128
        self.audio_metrics["mel_stft"] = auraloss.freq.MelSTFTLoss(
            sample_rate=self.ac.sr,
            fft_size=metrics_win_len,
            hop_size=metrics_hop_len,
            win_length=metrics_win_len,
            n_mels=metrics_n_mels,
        )
        self.audio_metrics["mfcc"] = MFCCL1(
            sr=self.ac.sr,
            log_mels=True,
            n_fft=metrics_win_len,
            hop_len=metrics_hop_len,
            n_mels=metrics_n_mels,
        )
        self.audio_metrics["rms_coss"] = RMSMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="coss",
        )
        self.audio_metrics["rms_pcc"] = RMSMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="pcc",
        )
        self.audio_metrics["sc_coss"] = SpectralCentroidMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="coss",
        )
        self.audio_metrics["sc_pcc"] = SpectralCentroidMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="pcc",
        )
        self.audio_metrics["sb_coss"] = SpectralBandwidthMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="coss",
        )
        self.audio_metrics["sb_pcc"] = SpectralBandwidthMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="pcc",
        )
        self.audio_metrics["sf_coss"] = SpectralFlatnessMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="coss",
        )
        self.audio_metrics["sf_pcc"] = SpectralFlatnessMetric(
            sr=self.ac.sr,
            win_len=metrics_win_len,
            hop_len=metrics_hop_len,
            average_channels=True,
            dist_fn="pcc",
        )

        self.signal_metrics = nn.ModuleDict()
        self.signal_metrics["ent"] = EntropyMetric(eps=eps)
        self.signal_metrics["spec_ent"] = SpectralEntropyMetric(eps=eps)
        self.signal_metrics["tv"] = TotalVariationMetric(eps=eps)
        self.signal_metrics["tp"] = TurningPointsMetric(eps=eps)

        self.global_n = 0
        self.curr_training_step = 0
        self.total_n_training_steps = None

        # TSV logging
        self.wt_name = None
        tsv_cols = [
            "seed",
            "wt_name",
            "stage",
            "step",
            "global_n",
            "loss",
        ]
        for p_name in self.temp_param_names:
            tsv_cols.append(f"l1__{p_name}")
            tsv_cols.append(f"l1_inv__{p_name}")
            tsv_cols.append(f"l1_inv_all__{p_name}")
            tsv_cols.append(f"esr__{p_name}")
            tsv_cols.append(f"esr_inv__{p_name}")
            tsv_cols.append(f"esr_inv_all__{p_name}")
            tsv_cols.append(f"mse__{p_name}")
            tsv_cols.append(f"mse_inv__{p_name}")
            tsv_cols.append(f"mse_inv_all__{p_name}")
            tsv_cols.append(f"fft__{p_name}")
            tsv_cols.append(f"fft_inv__{p_name}")
            tsv_cols.append(f"fft_inv_all__{p_name}")
        for p_name in self.global_param_names:
            tsv_cols.append(f"l1__{p_name}")
        for metric_name in self.audio_metrics:
            tsv_cols.append(f"audio__{metric_name}")
        for metric_name in self.signal_metrics:
            for p_name in self.temp_param_names:
                tsv_cols.append(f"signal__{p_name}__{metric_name}")
            for p_name in self.temp_param_names_hat:
                tsv_cols.append(f"signal__{p_name}_hat__{metric_name}")
        for fad_model_name in self.fad_model_names:
            tsv_cols.append(f"fad__{fad_model_name}")
        tsv_cols.append("add_lfo_mean_range")
        tsv_cols.append("add_lfo_max_range")
        tsv_cols.append("add_lfo_min")
        tsv_cols.append("add_lfo_max")
        self.tsv_cols = tsv_cols
        if run_name:
            self.tsv_path = os.path.join(OUT_DIR, f"{self.run_name}.tsv")
            if not os.path.exists(self.tsv_path):
                with open(self.tsv_path, "w") as f:
                    f.write("\t".join(tsv_cols) + "\n")
            else:
                log.info(f"Appending to existing TSV file: {self.tsv_path}")
        else:
            self.tsv_path = None

    def state_dict(self, *args, **kwargs) -> Dict[str, T]:
        state_dict = super().state_dict(*args, **kwargs)
        excluded_keys = [
            k
            for k in state_dict
            if ".mel_spec." in k or ".mfcc." in k or ".mel_stft." in k
        ]
        for k in excluded_keys:
            del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(state_dict, strict=False)

    def on_fit_start(self) -> None:
        self.global_n = 0
        self.curr_training_step = 0
        n_batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
        n_epochs = self.trainer.max_epochs
        self.total_n_training_steps = n_batches_per_epoch * n_epochs
        # if tr.cuda.is_available():
        #     self.model = tr.compile(self.model)
        #     self.synth = tr.compile(self.synth)
        #     self.synth_hat = tr.compile(self.synth_hat)

    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T | Dict[str, T]]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase = batch["phase"]
        phase_hat = batch["phase_hat"]

        batch_size = f0_hz.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        assert phase.shape == (batch_size,)
        assert phase_hat.shape == (batch_size,)

        temp_params = {}
        for temp_param_name in self.temp_param_names:
            temp_param = batch[temp_param_name]
            assert temp_param.size(0) == batch_size
            assert temp_param.ndim == 2
            assert temp_param.min() >= 0.0
            assert temp_param.max() <= 1.0
            temp_params[temp_param_name] = temp_param

        global_params_0to1 = {p: batch[f"{p}_0to1"] for p in self.global_param_names}
        global_params = {
            k: self.ac.convert_from_0to1(k, v) for k, v in global_params_0to1.items()
        }

        other_params = {}
        # if "add_lfo" in temp_params:
        #     other_params["add_lfo"] = temp_params["add_lfo"]
        # else:
        #     other_mod_sig = list(temp_params.values())[0]
        #     add_mod_sig = tr.zeros_like(other_mod_sig)
        #     # add_mod_sig = tr.ones_like(other_mod_sig)
        #     other_params["add_lfo"] = add_mod_sig
        # if "wt" in batch:
        #     other_params["wt"] = batch["wt"]
        if "q_0to1" in batch:
            other_params["q_mod_sig"] = batch["q_0to1"]
        # filter_types = ["lp", "hp", "bp", "no"]
        # if "filter_type_0to1" in batch:
        #     filter_type_0to1 = batch["filter_type_0to1"][0]
        #     filter_type_idx = int(filter_type_0to1 * len(filter_types))
        #     filter_type = filter_types[filter_type_idx]
        #     other_params["filter_type"] = filter_type

        # Generate ground truth wet audio
        with tr.no_grad():
            synth_out = self.synth(
                self.ac.n_samples,
                f0_hz,
                phase,
                temp_params,
                global_params,
                other_params,
            )
            add_audio = synth_out["add_audio"]
            sub_audio = synth_out["sub_audio"]
            env_audio = synth_out["env_audio"]
            assert add_audio.shape == (batch_size, self.ac.n_samples)
            assert add_audio.shape == sub_audio.shape == env_audio.shape

        batch["add_audio"] = add_audio
        batch["sub_audio"] = sub_audio
        batch["env_audio"] = env_audio
        batch["x"] = env_audio.unsqueeze(1)
        batch["temp_params"] = temp_params
        batch["global_params_0to1"] = global_params_0to1
        batch["global_params"] = global_params
        batch["other_params"] = other_params
        return batch

    def step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        batch_size = batch["f0_hz"].size(0)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        self.log(f"global_n", float(self.global_n))

        batch = self.preprocess_batch(batch)

        # Get mandatory params
        f0_hz = batch["f0_hz"]
        x = batch["x"]
        phase_hat = batch["phase_hat"]
        temp_params = batch.get("temp_params", {})
        global_params_0to1 = batch.get("global_params_0to1", {})
        global_params = batch.get("global_params", {})
        other_params = batch.get("other_params", {})

        # Get optional params
        add_audio = batch.get("add_audio")

        model_in_dict = {
            "audio": x,
            "f0_hz": f0_hz,
        }
        temp_params_hat_raw = {}
        temp_params_hat = {}
        temp_params_hat_inv = {}
        global_params_0to1_hat = {}
        global_params_hat = {}
        log_spec_x = None

        temp_param_metrics = {}
        global_param_metrics = {}

        # Perform model forward pass
        if self.use_model:
            alpha_linear = None
            alpha_noise = None
            # Use alpha_linear and alpha_noise for train and val, but not test
            if stage != "test":
                assert self.total_n_training_steps
                # alpha_noise = self.beta ** self.curr_training_step
                alpha_noise = 1.0 - self.curr_training_step / (
                    self.total_n_training_steps / 1.5
                )
                alpha_noise = max(alpha_noise, 0.0)
                assert alpha_noise >= 0.0
                self.log(f"{stage}/alpha_noise", alpha_noise, prog_bar=False)
                alpha_linear = 1.0 - self.curr_training_step / (
                    self.total_n_training_steps / 1.5
                )
                alpha_linear = max(alpha_linear, 0.0)
                assert alpha_linear >= 0.0
                self.log(f"{stage}/alpha_linear", alpha_linear, prog_bar=False)
                # log.info(f"alpha_noise: {alpha_noise}, alpha_linear: {alpha_linear}")
            model_out = self.model(
                model_in_dict, alpha_noise=alpha_noise, alpha_linear=alpha_linear
            )

            # Postprocess temp_params_hat
            # TODO(cm): clean this up
            for p_name in self.temp_param_names_hat:
                p_hat = model_out[p_name]
                temp_params_hat_raw[p_name] = p_hat
                p_hat_interp = None
                if p_name in temp_params:
                    p = temp_params[p_name]
                    p_hat_interp = util.interpolate_dim(
                        p_hat, p.size(1), dim=1, align_corners=True
                    )
                    # Prevents adapter outputs from being analyzed
                    if p.shape == p_hat_interp.shape:
                        with tr.no_grad():
                            l1 = self.l1(p_hat_interp, p)
                            esr = self.esr(p_hat_interp, p)
                            mse = self.mse(p_hat_interp, p)
                            fft_mag_dist = AcidDDSPLightingModule.calc_fft_mag_dist(
                                p_hat_interp, p, ignore_dc=True
                            )
                        self.log(f"{stage}/{p_name}_l1", l1, prog_bar=False)
                        self.log(f"{stage}/{p_name}_esr", esr, prog_bar=False)
                        self.log(f"{stage}/{p_name}_mse", mse, prog_bar=False)
                        self.log(f"{stage}/{p_name}_fft", fft_mag_dist, prog_bar=False)
                        temp_param_metrics[f"{p_name}_l1"] = l1
                        temp_param_metrics[f"{p_name}_esr"] = esr
                        temp_param_metrics[f"{p_name}_mse"] = mse
                        temp_param_metrics[f"{p_name}_fft"] = fft_mag_dist

                        # Do invariant comparison
                        if stage != "train":
                            with tr.no_grad():
                                p_hat_inv = (
                                    AcidDDSPLightingModule.compute_lstsq_with_bias(
                                        x_hat=p_hat_interp.unsqueeze(1),
                                        x=p.unsqueeze(1),
                                    ).squeeze(1)
                                )
                                l1_inv = self.l1(p_hat_inv, p)
                                esr_inv = self.esr(p_hat_inv, p)
                                mse_inv = self.mse(p_hat_inv, p)
                                fft_mag_dist_inv = (
                                    AcidDDSPLightingModule.calc_fft_mag_dist(
                                        p_hat_inv, p, ignore_dc=True
                                    )
                                )
                            temp_params_hat_inv[f"{p_name}_hat_inv"] = p_hat_inv
                            self.log(f"{stage}/{p_name}_l1_inv", l1_inv, prog_bar=False)
                            self.log(
                                f"{stage}/{p_name}_esr_inv", esr_inv, prog_bar=False
                            )
                            self.log(
                                f"{stage}/{p_name}_mse_inv", mse_inv, prog_bar=False
                            )
                            self.log(
                                f"{stage}/{p_name}_fft_inv",
                                fft_mag_dist_inv,
                                prog_bar=False,
                            )
                            temp_param_metrics[f"{p_name}_l1_inv"] = l1_inv
                            temp_param_metrics[f"{p_name}_esr_inv"] = esr_inv
                            temp_param_metrics[f"{p_name}_mse_inv"] = mse_inv
                            temp_param_metrics[f"{p_name}_fft_inv"] = fft_mag_dist_inv
                        else:
                            temp_param_metrics[f"{p_name}_l1_inv"] = None
                            temp_param_metrics[f"{p_name}_esr_inv"] = None
                            temp_param_metrics[f"{p_name}_mse_inv"] = None
                            temp_param_metrics[f"{p_name}_fft_inv"] = None

                # Config decides which temp params are interpolated for synth_hat
                if p_name in self.interp_temp_param_names_hat:
                    if p_hat_interp is None:
                        p_hat_interp = util.interpolate_dim(
                            p_hat, self.ac.n_samples, dim=1, align_corners=True
                        )
                        temp_params_hat[p_name] = p_hat_interp
                    else:
                        temp_params_hat[p_name] = p_hat_interp
                else:
                    temp_params_hat[p_name] = p_hat
                if f"{p_name}_adapted" in model_out:
                    p_hat = model_out[f"{p_name}_adapted"]
                    # Config decides which temp params are interpolated for synth_hat
                    if p_name in self.interp_temp_param_names_hat:
                        if p_name in temp_params:
                            p_hat = util.interpolate_dim(
                                p_hat,
                                temp_params[p_name].size(1),
                                dim=1,
                                align_corners=True,
                            )
                        else:
                            p_hat = util.interpolate_dim(
                                p_hat, self.ac.n_samples, dim=1, align_corners=True
                            )
                    temp_params_hat[f"{p_name}_adapted"] = p_hat

            # Postprocess global_params_hat
            for p_name in self.global_param_names_hat:
                p_val_0to1_hat = model_out[f"{p_name}_0to1"]
                p_val_hat = self.ac.convert_from_0to1(p_name, p_val_0to1_hat)
                global_params_0to1_hat[p_name] = p_val_0to1_hat
                global_params_hat[p_name] = p_val_hat
                if not self.ac.is_fixed(p_name) and p_name in global_params_0to1:
                    p_val_0to1 = global_params_0to1[p_name]
                    with tr.no_grad():
                        l1 = self.l1(p_val_0to1_hat, p_val_0to1)
                    self.log(f"{stage}/{p_name}_l1", l1, prog_bar=False)
                    global_param_metrics[f"{p_name}_l1"] = l1

            # Postprocess log_spec_x
            log_spec_x = model_out.get("log_spec_x")[:, 0, :, :]

            # Postprocess q_hat TODO(cm): generalize
            # if "q" in self.global_param_names_hat:
            #     q_0to1_hat = global_params_0to1_hat["q"]
            #     q_mod_sig_hat = q_0to1_hat.unsqueeze(-1)
            #     temp_params_hat["q_mod_sig"] = q_mod_sig_hat

        if "add_lfo" in temp_params_hat:
            tp_hat = temp_params_hat["add_lfo"]
            with tr.no_grad():
                lfo_max = tp_hat.max(dim=1).values
                lfo_min = tp_hat.min(dim=1).values
                lfo_range = lfo_max - lfo_min
                add_lfo_mean_range = lfo_range.mean().item()
                add_lfo_max_range = lfo_range.max().item()
                add_lfo_min = lfo_min.min().item()
                add_lfo_max = lfo_max.max().item()
            self.log(f"{stage}/add_lfo_mean_range", add_lfo_mean_range, prog_bar=False)
            self.log(f"{stage}/add_lfo_max_range", add_lfo_max_range, prog_bar=False)
            self.log(f"{stage}/add_lfo_min", add_lfo_min, prog_bar=False)
            self.log(f"{stage}/add_lfo_max", add_lfo_max, prog_bar=False)
        else:
            add_lfo_mean_range = None
            add_lfo_max_range = None
            add_lfo_max = None
            add_lfo_min = None

        # Compute invariant all metrics
        if stage != "train" and self.temp_param_names:
            tp = [temp_params[name] for name in self.temp_param_names]
            tp = tr.stack(tp, dim=1)
            tp_hat_s = []
            for name in self.temp_param_names_hat:
                tp_hat = temp_params_hat[name]
                if tp_hat.ndim != 2:
                    continue
                tp_hat = util.interpolate_dim(
                    tp_hat, tp.size(2), dim=1, align_corners=True
                )
                tp_hat_s.append(tp_hat)

            if tp_hat_s:
                tp_hat = tr.stack(tp_hat_s, dim=1)
                tp_pred = AcidDDSPLightingModule.compute_lstsq_with_bias(tp_hat, tp)
                assert tp_pred.size(1) == len(self.temp_param_names)
                for idx, p_name in enumerate(self.temp_param_names):
                    curr_tp = tp[:, idx, :]
                    curr_tp_pred = tp_pred[:, idx, :]
                    temp_params_hat_inv[f"{p_name}_hat_inv_all"] = curr_tp_pred
                    with tr.no_grad():
                        l1 = self.l1(curr_tp_pred, curr_tp)
                        esr = self.esr(curr_tp_pred, curr_tp)
                        mse = self.mse(curr_tp_pred, curr_tp)
                        fft_mag_dist = AcidDDSPLightingModule.calc_fft_mag_dist(
                            curr_tp_pred, curr_tp
                        )
                    self.log(f"{stage}/{p_name}_l1_inv_all", l1, prog_bar=False)
                    self.log(f"{stage}/{p_name}_esr_inv_all", esr, prog_bar=False)
                    self.log(f"{stage}/{p_name}_mse_inv_all", mse, prog_bar=False)
                    self.log(
                        f"{stage}/{p_name}_fft_inv_all", fft_mag_dist, prog_bar=False
                    )
                    temp_param_metrics[f"{p_name}_l1_inv_all"] = l1
                    temp_param_metrics[f"{p_name}_esr_inv_all"] = esr
                    temp_param_metrics[f"{p_name}_mse_inv_all"] = mse
                    temp_param_metrics[f"{p_name}_fft_inv_all"] = fft_mag_dist
        else:
            for p_name in self.temp_param_names:
                temp_param_metrics[f"{p_name}_l1_inv_all"] = None
                temp_param_metrics[f"{p_name}_esr_inv_all"] = None
                temp_param_metrics[f"{p_name}_mse_inv_all"] = None
                temp_param_metrics[f"{p_name}_fft_inv_all"] = None

        if log_spec_x is None:
            with tr.no_grad():
                log_spec_x = self.spectral_visualizer(x).squeeze(1)

        # Generate audio x_hat
        synth_out_hat = self.synth_hat(
            self.ac.n_samples,
            f0_hz,
            phase_hat,
            temp_params_hat,
            global_params_hat,
            other_params,
        )
        x_hat = synth_out_hat["env_audio"].unsqueeze(1)

        # Compute loss
        if self.use_p_loss:
            loss = 0.0
            for p_name in self.temp_param_names:
                p_val = temp_params[p_name]
                p_val_hat = temp_params_hat[p_name]
                assert p_val.shape == p_val_hat.shape
                p_loss = self.loss_func(p_val_hat, p_val)
                self.log(
                    f"{stage}/ploss_{self.loss_name}_{p_name}",
                    p_loss,
                    prog_bar=False,
                )
                loss += p_loss

            for p_name in self.global_param_names:
                if not self.ac.is_fixed(p_name):
                    p_val = global_params_0to1[p_name]
                    p_val_hat = global_params_0to1_hat[p_name]
                    p_loss = self.loss_func(p_val_hat, p_val)
                    self.log(
                        f"{stage}/ploss_{self.loss_name}_{p_name}",
                        p_loss,
                        prog_bar=False,
                    )
                    loss += p_loss
            self.log(f"{stage}/ploss_{self.loss_name}", loss, prog_bar=False)
        else:
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/audio_{self.loss_name}", loss, prog_bar=False)

        self.log(f"{stage}/loss", loss, prog_bar=False)
        if stage == "train" and not self.automatic_optimization:
            for opt in self.optimizers():
                opt.zero_grad()
            self.manual_backward(loss)
            for opt in self.optimizers():
                self.clip_gradients(opt)
                opt.step()

        # Log audio metrics
        audio_metrics_hat = {}
        if stage != "train":
            for metric_name, metric in self.audio_metrics.items():
                with tr.no_grad():
                    audio_metric = metric(x_hat, x)
                audio_metrics_hat[metric_name] = audio_metric
                self.log(f"{stage}/audio_{metric_name}", audio_metric, prog_bar=False)

        # Calc FAD metrics
        fad_metrics = {}
        if stage == "test":
            fad_metrics = self.calc_fad_metrics(x, x_hat)

        # Interpolate temp_params_hat and temp_params_hat_inv if needed
        temp_params_hat = {f"{k}_hat": v for k, v in temp_params_hat.items()}
        temp_params_hat = {
            k: util.interpolate_dim(v, n=self.ac.n_samples, dim=1, align_corners=True)
            for k, v in temp_params_hat.items()
        }
        temp_params_hat_inv = {
            k: util.interpolate_dim(v, n=self.ac.n_samples, dim=1, align_corners=True)
            for k, v in temp_params_hat_inv.items()
        }

        # Calculate signal metrics
        signal_metrics = {}
        signal_metrics_hat = {}
        n_frames = None
        if stage != "train":
            for metric_name, metric in self.signal_metrics.items():
                for p_name in self.temp_param_names:
                    signal = temp_params[p_name]
                    if n_frames is None:
                        n_frames = signal.size(1)
                    else:
                        assert n_frames == signal.size(1)
                    val = metric(signal)
                    signal_metrics[f"{p_name}_{metric_name}"] = val
                    self.log(
                        f"{stage}/{p_name}_{metric_name}",
                        val,
                        prog_bar=False,
                    )
                for p_name in self.temp_param_names_hat:
                    signal_hat = temp_params_hat[f"{p_name}_hat"]
                    if n_frames is None:
                        n_frames = signal_hat.size(1)
                    else:
                        assert n_frames == signal_hat.size(1)
                    val_hat = metric(signal_hat)
                    signal_metrics_hat[f"{p_name}_{metric_name}"] = val_hat
                    self.log(
                        f"{stage}/{p_name}_hat_{metric_name}",
                        val_hat,
                        prog_bar=False,
                    )

        # TSV logging
        if self.tsv_path:
            with open(self.tsv_path, "a") as f:
                tsv_row = [
                    tr.random.initial_seed(),
                    self.wt_name,
                    stage,
                    self.global_step,
                    self.global_n,
                    loss.item(),
                ]
                # TODO(cm): parameterize temp_param metrics
                for p_name in self.temp_param_names:
                    tsv_row.append(temp_param_metrics[f"{p_name}_l1"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_l1_inv"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_l1_inv_all"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_esr"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_esr_inv"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_esr_inv_all"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_mse"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_mse_inv"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_mse_inv_all"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_fft"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_fft_inv"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_fft_inv_all"].item())
                for p_name in self.global_param_names:
                    tsv_row.append(global_param_metrics[f"{p_name}_l1"].item())
                for metric_name in self.audio_metrics:
                    if metric_name in audio_metrics_hat:
                        val = audio_metrics_hat[metric_name].item()
                    else:
                        val = None
                    tsv_row.append(val)
                for metric_name in self.signal_metrics:
                    for p_name in self.temp_param_names:
                        val = signal_metrics.get(f"{p_name}_{metric_name}")
                        if val is not None:
                            val = val.item()
                        tsv_row.append(val)
                    for p_name in self.temp_param_names_hat:
                        val_hat = signal_metrics_hat.get(f"{p_name}_{metric_name}")
                        if val_hat is not None:
                            val_hat = val_hat.item()
                        tsv_row.append(val_hat)
                for fad_model_name in self.fad_model_names:
                    fad_metric = fad_metrics.get(fad_model_name)
                    tsv_row.append(fad_metric)
                tsv_row.append(add_lfo_mean_range)
                tsv_row.append(add_lfo_max_range)
                tsv_row.append(add_lfo_min)
                tsv_row.append(add_lfo_max)

                assert len(tsv_row) == len(self.tsv_cols)
                f.write("\t".join(str(v) for v in tsv_row) + "\n")

        global_params_hat = {f"{k}_hat": v for k, v in global_params_hat.items()}
        audio_metrics_hat = {f"{k}_hat": v for k, v in audio_metrics_hat.items()}
        out_dict = {
            "loss": loss,
            "add_audio": add_audio,
            "x": x,
            "x_hat": x_hat,
            "log_spec_x": log_spec_x,
            # TODO(cm): tmp
            # "add_lfo_seg_indices_hat": model_out["add_lfo_seg_indices"],
            # "sub_lfo_seg_indices_hat": model_out["sub_lfo_seg_indices"],
        }
        assert all(k not in out_dict for k in temp_params)
        assert all(k not in out_dict for k in temp_params_hat)
        assert all(k not in out_dict for k in temp_param_metrics)
        assert all(k not in out_dict for k in temp_params_hat_inv)
        assert all(k not in out_dict for k in global_params)
        assert all(k not in out_dict for k in global_params_hat)
        assert all(k not in out_dict for k in global_param_metrics)
        assert all(k not in out_dict for k in audio_metrics_hat)
        assert all(k not in out_dict for k in signal_metrics)
        assert all(k not in out_dict for k in signal_metrics_hat)
        assert all(k not in out_dict for k in fad_metrics)
        out_dict.update(temp_params)
        out_dict.update(temp_params_hat)
        out_dict.update(temp_params_hat_inv)
        out_dict.update(temp_param_metrics)
        out_dict.update(global_params)
        out_dict.update(global_params_hat)
        out_dict.update(global_param_metrics)
        out_dict.update(audio_metrics_hat)
        out_dict.update(signal_metrics)
        out_dict.update(signal_metrics_hat)
        out_dict.update(fad_metrics)
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        result = self.step(batch, stage="train")
        self.curr_training_step += 1
        return result

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")

    def configure_optimizers(self) -> Tuple[tr.optim.Optimizer, ...]:
        if self.model_opt is None or self.synth_opt is None:
            log.info(f"Using one optimizer")
            if self.model_opt is not None:
                self.model_opt = self.model_opt(self.model.parameters())
                return self.model_opt
            else:
                self.synth_opt = self.synth_opt(self.synth_hat.parameters())
                return self.synth_opt
        else:
            assert (
                self.trainer.accumulate_grad_batches == 1
            ), "Grad accumulation is not supported with multiple optimizers"
            self.automatic_optimization = False
            self.model_opt = self.model_opt(self.model.parameters())
            self.synth_opt = self.synth_opt(self.synth_hat.parameters())
            log.info(
                f"Using multiple optimizers: "
                f"\n - model_opt initial LR: {self.model_opt.defaults['lr']:.6f}"
                f"\n - synth_opt initial LR: {self.synth_opt.defaults['lr']:.6f}"
            )
            return self.model_opt, self.synth_opt

    @staticmethod
    def normalize_signal(x: T, eps: float = 1e-8) -> T:
        assert x.ndim == 2
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x_range = tr.clamp(x_max - x_min, min=eps)
        x_norm = (x - x_min) / x_range
        return x_norm

    @staticmethod
    def flip_signal_vertically(x: T) -> T:
        assert x.ndim == 2
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x_flipped = x_max + x_min - x
        return x_flipped

    @staticmethod
    def calc_fft_mag_dist(x_hat: T, x: T, ignore_dc: bool = True) -> T:
        assert x.ndim == 2
        assert x.shape == x_hat.shape
        X = tr.fft.rfft(x, dim=1).abs()
        X_hat = tr.fft.rfft(x_hat, dim=1).abs()
        # log.info(f"X = {X.squeeze()}")
        # log.info(f"X_hat = {X_hat.squeeze()}")
        if ignore_dc:
            X = X[:, 1:]
            X_hat = X_hat[:, 1:]
        dist = tr.nn.functional.l1_loss(X_hat, X)
        return dist

    @staticmethod
    def compute_lstsq_with_bias(x_hat: T, x: T) -> T:
        """
        Given x and x_hat of shape (bs, n_signals, n_samples), compute the best
        linear combination matrix W and bias vector b (per batch) such that:
            x ≈ W @ x_hat + b
        using a batched least-squares approach.

        Args:
            x (Tensor): Target tensor of shape (bs, n_signals, n_samples).
            x_hat (Tensor): Basis tensor of shape (bs, n_signals, n_samples).

        Returns:

        """
        bs, n_signals, n_samples = x_hat.shape
        assert x.ndim == 3
        assert x.size(0) == bs
        assert x.size(2) == n_samples

        # Augment x_hat with a row of ones to account for the bias term.
        # ones shape: (bs, 1, n_samples)
        ones = tr.ones(bs, 1, n_samples, device=x_hat.device, dtype=x_hat.dtype)
        x_hat_aug = tr.cat([x_hat, ones], dim=1)  # shape: (bs, n_signals+1, n_samples)

        # Transpose the last two dimensions so that we set up the least-squares problem as:
        # A @ (solution) ≈ B
        A = x_hat_aug.transpose(1, 2)  # shape: (bs, n_samples, n_signals+1)
        B = x.transpose(1, 2)  # shape: (bs, n_samples, n_signals)

        # Solve the least-squares problem for the augmented system.
        lstsq_result = tr.linalg.lstsq(A, B)
        solution = lstsq_result.solution  # shape: (bs, n_signals+1, n_signals)

        # The solution consists of weights and bias:
        # The first n_signals rows correspond to the weight matrix (transposed), and the last row is the bias.
        W_t = solution[:, :-1, :]  # shape: (bs, n_signals, n_signals)
        bias_t = solution[:, -1:, :]  # shape: (bs, 1, n_signals)

        # Transpose to obtain the weight matrix in the right orientation.
        W = W_t.transpose(1, 2)  # shape: (bs, n_signals, n_signals)
        bias = bias_t.transpose(1, 2)  # shape: (bs, n_signals, 1)

        # Compute the predicted x using the estimated weights and bias.
        # Note: bias is added to each sample in the time dimension.
        x_pred = tr.bmm(W, x_hat) + bias  # shape: (bs, n_signals, n_samples)
        return x_pred

    def calc_fad_metrics(self, x: T, x_hat: T, n_workers: int = 0) -> Dict[str, float]:
        x = x.squeeze(1).detach().cpu()
        x_hat = x_hat.squeeze(1).detach().cpu()
        fad_dir_x = os.path.join(OUT_DIR, f"{self.run_name}__fad_x")
        fad_dir_x_hat = os.path.join(OUT_DIR, f"{self.run_name}__fad_x_hat")
        save_and_concat_fad_audio(
            self.ac.sr,
            x,
            fad_dir_x,
            fade_n_samples=None,
        )
        save_and_concat_fad_audio(
            self.ac.sr,
            x_hat,
            fad_dir_x_hat,
            fade_n_samples=None,
        )
        fad_metrics = {}
        for fad_model_name in self.fad_model_names:
            log.info(f"Calculating FAD for {fad_model_name}")
            fad_val = calc_fad(
                fad_model_name,
                baseline_dir=fad_dir_x,
                eval_dir=fad_dir_x_hat,
                workers=n_workers,
            )
            fad_metrics[f"{fad_model_name}"] = fad_val
            self.log(f"test/fad__{fad_model_name}", fad_val, prog_bar=False)
        shutil.rmtree(fad_dir_x)
        shutil.rmtree(fad_dir_x_hat)
        return fad_metrics


class PreprocLightningModule(AcidDDSPLightingModule):
    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        audio = batch["audio"]
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase_hat = batch["phase_hat"]

        assert audio.ndim == 2
        assert (
            audio.size(1) == self.ac.n_samples
        ), f"{audio.size(1)}, {self.ac.n_samples}"
        bs = audio.size(0)
        assert f0_hz.shape == (bs,)
        assert note_on_duration.shape == (bs,)
        assert phase_hat.shape == (bs,)

        batch = {
            "x": audio.unsqueeze(1),
            "f0_hz": f0_hz,
            "note_on_duration": note_on_duration,
            "phase_hat": phase_hat,
        }
        return batch


if __name__ == "__main__":
    tr.manual_seed(71)

    n_signals: int = 3
    n_samples: int = 1000
    mod_gen_x = ModSignalGenRandomBezier()
    mod_gen_x_hat = ModSignalGenRandomBezier(
        min_n_seg=12,
        max_n_seg=12,
        min_degree=3,
        max_degree=3,
        min_seg_interval_frac=1.0,
    )
    x_s = []
    x_hat_s = []
    for _ in range(n_signals):
        x = mod_gen_x(n_samples)
        x_s.append(x)
        # x_vflip = AcidDDSPLightingModule.flip_signal_vertically(x.view(1, -1)).view(-1) + 1.0
        # x_hat_s.append(x_vflip)
        x_hat = mod_gen_x_hat(n_samples)
        x_hat_s.append(x_hat)
    x = tr.stack(x_s, dim=0).unsqueeze(0)
    x_hat = tr.stack(x_hat_s, dim=0).unsqueeze(0)
    # Calc finite difference
    x = x[:, :, 1:] - x[:, :, :-1]
    x_hat = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    x_pred = AcidDDSPLightingModule.compute_lstsq_with_bias(x_hat, x, n_segments=3)
    x = tr.cumsum(x, dim=2)
    x_hat = tr.cumsum(x_hat, dim=2)
    x_pred = tr.cumsum(x_pred, dim=2)

    x_pred_2 = AcidDDSPLightingModule.compute_lstsq_with_bias(x_pred, x, n_segments=1)
    x_pred = x_pred_2

    from matplotlib import pyplot as plt

    for idx in range(n_signals):
        plt.plot(x_hat[0, idx, :])
    plt.title("x_hat")
    # plt.ylim(-0.1, 1.1)
    plt.show()

    for idx in range(n_signals):
        plt.plot(x[0, idx, :], label="x", color="black")
        plt.plot(x_pred[0, idx, :], label="x_pred", color="orange")
        plt.legend()
        plt.title(f"x, x_pred")
        # plt.ylim(-0.1, 1.1)
        plt.show()

    exit()

    n_samples = 7
    x = tr.rand((1, n_samples))
    y = tr.rand((1, n_samples))
    x_scaled = x * 3.0
    x_scaled_shifted = x_scaled + -2.0

    dist = AcidDDSPLightingModule.calc_fft_mag_dist(x, x_scaled)
    log.info(f"dist x, x_scaled = {dist}")
    dist = AcidDDSPLightingModule.calc_fft_mag_dist(x, x_scaled_shifted)
    log.info(f"dist x, x_scaled_shifted = {dist}")

    x_norm = AcidDDSPLightingModule.normalize_signal(x)
    x_scaled_shifted_norm = AcidDDSPLightingModule.normalize_signal(x_scaled_shifted)
    dist = AcidDDSPLightingModule.calc_fft_mag_dist(x_norm, x_scaled_shifted_norm)
    log.info(f"dist x_norm, x_scaled_shifted_norm = {dist}")

    y_norm = AcidDDSPLightingModule.normalize_signal(y)
    xy = x_norm + y_norm
    x_norm_vflip = AcidDDSPLightingModule.flip_signal_vertically(x_norm)
    y_norm_vflip = AcidDDSPLightingModule.flip_signal_vertically(y_norm)
    xy_vflip = x_norm_vflip + y_norm_vflip
    dist = AcidDDSPLightingModule.calc_fft_mag_dist(xy, xy_vflip)
    log.info(f"dist xy, xy_vflip = {dist}")
