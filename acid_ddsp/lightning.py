import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Mapping, Tuple

import auraloss
import pytorch_lightning as pl
import torch as tr
from pytorch_lightning.cli import OptimizerCallable
from torch import Tensor as T
from torch import nn

import util
from audio_config import AudioConfig
from audio_distances import (
    RMSDistance,
    SpectralCentroidDistance,
    SpectralBandwidthDistance,
    SpectralFlatnessDistance,
    MFCCDistance,
)
from fad import save_and_concat_fad_audio, calc_fad
from feature_extraction import LogMelSpecFeatureExtractor
from mod_sig_distances import (
    FirstDerivativeDistance,
    PCCDistance,
    FrechetDistance,
)
from mod_sig_metrics import (
    SpectralEntropyMetric,
    TotalVariationMetric,
    TurningPointsMetric,
    LFORangeMetric,
)
from paths import OUT_DIR
from synths import SynthBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModDiscoveryLightingModule(pl.LightningModule):
    def __init__(
        self,
        ac: AudioConfig,
        spectral_visualizer: LogMelSpecFeatureExtractor,
        model: nn.Module,
        loss_func: nn.Module,
        synth: Optional[SynthBase] = None,
        synth_hat: Optional[SynthBase] = None,
        temp_param_names: Optional[List[str]] = None,
        interp_temp_param_names: Optional[List[str]] = None,
        temp_param_names_hat: Optional[List[str]] = None,
        interp_temp_param_names_hat: Optional[List[str]] = None,
        temp_param_n_frames: Optional[int] = None,
        temp_param_n_samples: Optional[int] = None,
        global_param_names: Optional[List[str]] = None,
        use_p_loss: bool = False,
        fad_model_names: Optional[List[str]] = None,
        run_name: Optional[str] = None,
        model_opt: Optional[OptimizerCallable] = None,
        synth_opt: Optional[OptimizerCallable] = None,
        alpha_divisor: float = 1.5,
        lpf_cf_hz: float = 8.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert synth_hat is not None
        if temp_param_names is None:
            temp_param_names = []
        if interp_temp_param_names is None:
            interp_temp_param_names = []
        if temp_param_names_hat is None:
            temp_param_names_hat = []
        if interp_temp_param_names_hat is None:
            interp_temp_param_names_hat = []
        if temp_param_n_frames is None:
            if hasattr(model, "n_frames"):
                temp_param_n_frames = model.n_frames
            else:
                temp_param_n_frames = ac.n_samples
        if temp_param_n_samples is None:
            temp_param_n_samples = ac.n_samples
        if global_param_names is None:
            global_param_names = []
        if fad_model_names is None:
            fad_model_names = []
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        else:
            self.run_name = run_name
        log.info(f"Run name: {self.run_name}")
        assert ac.sr == spectral_visualizer.sr
        if hasattr(loss_func, "sr"):
            assert loss_func.sr == ac.sr
        if hasattr(model, "n_frames"):
            assert model.n_frames == temp_param_n_frames
        self.ac = ac
        self.spectral_visualizer = spectral_visualizer
        self.model = model
        self.loss_func = loss_func
        self.synth = synth
        self.synth_hat = synth_hat
        self.temp_param_names = temp_param_names
        self.interp_temp_param_names = interp_temp_param_names
        self.temp_param_names_hat = temp_param_names_hat
        self.interp_temp_param_names_hat = interp_temp_param_names_hat
        self.temp_param_n_frames = temp_param_n_frames
        self.temp_param_n_samples = temp_param_n_samples
        self.global_param_names = global_param_names
        self.use_p_loss = use_p_loss
        self.fad_model_names = fad_model_names
        self.model_opt = model_opt
        self.synth_opt = synth_opt
        self.alpha_divisor = alpha_divisor
        self.lpf_cf_hz = lpf_cf_hz
        self.eps = eps

        self.loss_name = self.loss_func.__class__.__name__
        self.compare_temp_param_names = [
            p for p in temp_param_names if p in temp_param_names_hat
        ]
        self.global_n = 0
        self.curr_training_step = 0
        self.total_n_training_steps = None

        # TODO(cm): add to config
        # Audio distances ==============================================================
        metrics_win_len = int(0.0500 * self.ac.sr)
        metrics_hop_len = int(0.0125 * self.ac.sr)
        self.audio_dists = nn.ModuleDict()
        ad_tsv_cols = []
        self.audio_dists["mss"] = auraloss.freq.MultiResolutionSTFTLoss()
        ad_tsv_cols.append("mss")
        self.audio_dists["mel_stft"] = auraloss.freq.MelSTFTLoss(
            sample_rate=self.ac.sr,
            fft_size=metrics_win_len,
            hop_size=metrics_hop_len,
            win_length=metrics_win_len,
            n_mels=128,
        )
        ad_tsv_cols.append("mel_stft")
        self.audio_dists["mfcc"] = MFCCDistance(
            sr=self.ac.sr,
            log_mels=True,
            n_fft=metrics_win_len,
            hop_len=metrics_hop_len,
            n_mels=128,
        )
        ad_tsv_cols.append("mfcc")
        metrics_n_frames = self.ac.n_samples // self.spectral_visualizer.hop_len + 1
        ad_dist_fn_s = {
            "l1": nn.L1Loss(),
            "l1_d1": FirstDerivativeDistance(nn.L1Loss()),
            "pcc": PCCDistance(),
            "fd": FrechetDistance(n_frames=metrics_n_frames),
        }
        for feat_name, feat_cls in [
            ("rms", RMSDistance),
            ("sc", SpectralCentroidDistance),
            ("sb", SpectralBandwidthDistance),
            ("sf", SpectralFlatnessDistance),
        ]:
            self.audio_dists[feat_name] = feat_cls(
                sr=self.ac.sr,
                win_len=self.spectral_visualizer.n_fft,
                hop_len=self.spectral_visualizer.hop_len,
                dist_fn_s=ad_dist_fn_s,
                average_channels=True,
                filter_cf_hz=self.lpf_cf_hz,
            )
            for dist_name in ad_dist_fn_s:
                ad_tsv_cols.append(f"{feat_name}__{dist_name}")
                ad_tsv_cols.append(
                    f"{feat_name}__{dist_name}__cf_{self.lpf_cf_hz:.0f}_hz"
                )
                ad_tsv_cols.append(f"{feat_name}__{dist_name}__inv_all")
                ad_tsv_cols.append(
                    f"{feat_name}__{dist_name}__cf_{self.lpf_cf_hz:.0f}_hz__inv_all"
                )
        ad_tsv_cols = [f"audio__{v}" for v in ad_tsv_cols]

        # LFO distances ================================================================
        self.lfo_dists = nn.ModuleDict()
        self.lfo_dists["l1"] = nn.L1Loss()
        self.lfo_dists["l1_d1"] = FirstDerivativeDistance(dist_fn=nn.L1Loss())
        self.lfo_dists["pcc"] = PCCDistance()
        self.lfo_dists["fd"] = FrechetDistance(n_frames=temp_param_n_frames)

        # LFO metrics ==================================================================
        self.lfo_metrics = nn.ModuleDict()
        self.lfo_metrics["range_mean"] = LFORangeMetric(agg_fn="mean")
        self.lfo_metrics["min_val"] = LFORangeMetric(agg_fn="min_val")
        self.lfo_metrics["max_val"] = LFORangeMetric(agg_fn="max_val")
        self.lfo_metrics["spec_ent"] = SpectralEntropyMetric(eps=eps, normalize=True)
        self.lfo_metrics["tv"] = TotalVariationMetric(eps=eps, normalize=True)
        self.lfo_metrics["tp"] = TurningPointsMetric()

        # TSV logging ==================================================================
        self.wt_name = None
        tsv_cols = [
            "seed",
            "wt_name",
            "stage",
            "step",
            "global_n",
            "loss",
        ]
        tsv_cols.extend(ad_tsv_cols)
        for dist_name in self.lfo_dists:
            for p_name in self.compare_temp_param_names:
                for p_suffix in ["", "_inv", "_inv_all"]:
                    tsv_cols.append(f"{p_name}{p_suffix}__{dist_name}")
        for metric_name in self.lfo_metrics:
            for p_name in self.temp_param_names:
                tsv_cols.append(f"{p_name}__{metric_name}")
            for p_name in self.temp_param_names_hat:
                tsv_cols.append(f"{p_name}_hat__{metric_name}")
            for p_name in self.compare_temp_param_names:
                for p_suffix in ["_inv", "_inv_all"]:
                    tsv_cols.append(f"{p_name}{p_suffix}__{metric_name}")
        for fad_model_name in self.fad_model_names:
            tsv_cols.append(f"fad__{fad_model_name}")
        assert len(tsv_cols) == len(set(tsv_cols)), "Duplicate TSV columns"
        self.tsv_cols = tsv_cols
        # log.info(f"TSV columns: {self.tsv_cols}")
        if run_name:
            self.tsv_path = os.path.join(OUT_DIR, f"{self.run_name}.tsv")
            if not os.path.exists(self.tsv_path):
                with open(self.tsv_path, "w") as f:
                    f.write("\t".join(tsv_cols) + "\n")
            else:
                log.info(f"Appending to existing TSV file: {self.tsv_path}")
                # TODO(cm): check existing header
        else:
            self.tsv_path = None

        # Uncomment for random baselines ===============================================
        # x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__model.yml")
        # x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "serum_2/mod_sig_gen__model.yml")
        # self.x_hat_mod_gen = util.load_class_from_yaml(x_hat_mod_gen_path)

    def state_dict(self, *args, **kwargs) -> Dict[str, T]:
        state_dict = super().state_dict(*args, **kwargs)
        # TODO(cm): exclude more
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

    # This is needed for the ScheduleFree optimizer
    def on_train_epoch_start(self) -> None:
        try:
            self.model_opt.train()
        except AttributeError:
            pass
        try:
            self.synth_opt.train()
        except AttributeError:
            pass

    # This is needed for the ScheduleFree optimizer
    def on_validation_epoch_start(self) -> None:
        try:
            self.model_opt.eval()
        except AttributeError:
            pass
        try:
            self.synth_opt.eval()
        except AttributeError:
            pass

    # This is needed for the ScheduleFree optimizer
    def on_test_epoch_start(self) -> None:
        try:
            self.model_opt.eval()
        except AttributeError:
            pass
        try:
            self.synth_opt.eval()
        except AttributeError:
            pass

    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T | Dict[str, T]]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase = batch["phase"]
        phase_hat = batch["phase_hat"]

        bs = f0_hz.size(0)
        assert f0_hz.shape == (bs,)
        assert note_on_duration.shape == (bs,)
        assert phase.shape == (bs,)
        assert phase_hat.shape == (bs,)

        temp_params_raw = {}
        temp_params = {}
        for temp_param_name in self.temp_param_names:
            temp_param = batch[temp_param_name]
            assert temp_param.shape == (bs, self.temp_param_n_frames)
            assert temp_param.min() >= 0.0
            assert temp_param.max() <= 1.0
            temp_params_raw[temp_param_name] = temp_param
            if temp_param_name in self.interp_temp_param_names:
                temp_param = util.interpolate_dim(
                    temp_param, self.temp_param_n_samples, dim=1, align_corners=True
                )
            temp_params[temp_param_name] = temp_param

        global_params_0to1 = {p: batch[f"{p}_0to1"] for p in self.global_param_names}
        global_params = {
            k: self.ac.convert_from_0to1(k, v) for k, v in global_params_0to1.items()
        }

        other_params = {}
        if "q_0to1" in batch:
            other_params["q_mod_sig"] = batch["q_0to1"]

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
            assert add_audio.shape == (bs, self.ac.n_samples)
            assert add_audio.shape == sub_audio.shape == env_audio.shape
        preproc_batch = {
            "f0_hz": f0_hz,
            "note_on_duration": note_on_duration,
            "phase": phase,
            "phase_hat": phase_hat,
            "add_audio": add_audio,
            "sub_audio": sub_audio,
            "env_audio": env_audio,
            "x": env_audio.unsqueeze(1),
            "temp_params_raw": temp_params_raw,
            "temp_params": temp_params,
            "global_params": global_params,
            "other_params": other_params,
        }
        return preproc_batch

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
        # Get optional params
        temp_params_raw = batch.get("temp_params_raw", {})
        other_params = batch.get("other_params", {})
        add_audio = batch.get("add_audio")

        # Prepare model input
        model_in_dict = {
            "audio": x,
            "f0_hz": f0_hz,
        }
        # Prepare predicted params
        temp_params_hat_raw = {}
        temp_params_hat = {}
        temp_params_hat_inv = {}
        temp_params_hat_inv_all = {}
        # TSV logging
        tsv_row_vals = {}

        # Calculate alpha =========--===================================================
        alpha_linear = None
        alpha_noise = None
        # Use alpha_linear and alpha_noise for train and val, but not test
        if stage != "test":
            assert self.total_n_training_steps
            alpha_noise = 1.0 - self.curr_training_step / (
                self.total_n_training_steps / self.alpha_divisor
            )
            alpha_noise = max(alpha_noise, 0.0)
            self.log(f"{stage}/alpha_noise", alpha_noise, prog_bar=False)
            alpha_linear = 1.0 - self.curr_training_step / (
                self.total_n_training_steps / self.alpha_divisor
            )
            alpha_linear = max(alpha_linear, 0.0)
            self.log(f"{stage}/alpha_linear", alpha_linear, prog_bar=False)

        # Regular inference ============================================================
        model_out = self.model(
            model_in_dict, alpha_noise=alpha_noise, alpha_linear=alpha_linear
        )

        # Oracle inference =============================================================
        # model_out = {}
        # for p_name, tp in self.model.temp_params.items():
        #     mod_sig = temp_params_raw[p_name]
        #     pos_enc = self.model.pos_enc.expand(batch_size, -1, -1)
        #     mod_sig = mod_sig.to(pos_enc.device)
        #     model_out[p_name] = mod_sig
        #     mod_sig = mod_sig.unsqueeze(2)
        #     if tp.adapt_dim:
        #         adapt_in = tr.cat([mod_sig, pos_enc], dim=-1)
        #         if tp.adapt_use_separate:
        #             adapt_outs = []
        #             for dim_idx in range(tp.adapt_dim):
        #                 adapter = self.model.adapters[f"{p_name}_{dim_idx}"]
        #                 adapt_out = adapter(adapt_in)
        #                 adapt_outs.append(adapt_out)
        #             adapt_out = tr.cat(adapt_outs, dim=-1)
        #         else:
        #             adapt_out = self.model.adapters[p_name](adapt_in)
        #         adapt_out = self.model.adapter_acts[p_name](adapt_out)
        #         model_out[f"{p_name}_adapted"] = adapt_out.squeeze(-1)

        # Rand SM inference ============================================================
        # if stage != "test":
        #     model_out = self.model(
        #         model_in_dict, alpha_noise=alpha_noise, alpha_linear=alpha_linear
        #     )
        # else:
        #     model_out = {}
        #     for p_name, tp in self.model.temp_params.items():
        #         mod_sigs = []
        #         for idx in range(batch_size):
        #             mod_sig = self.x_hat_mod_gen(self.temp_param_n_frames)
        #             mod_sigs.append(mod_sig)
        #         mod_sig = tr.stack(mod_sigs, dim=0)
        #         pos_enc = self.model.pos_enc.expand(batch_size, -1, -1)
        #         mod_sig = mod_sig.to(pos_enc.device)
        #         model_out[p_name] = mod_sig
        #         mod_sig = mod_sig.unsqueeze(2)
        #         if tp.adapt_dim:
        #             adapt_in = tr.cat([mod_sig, pos_enc], dim=-1)
        #             if tp.adapt_use_separate:
        #                 adapt_outs = []
        #                 for dim_idx in range(tp.adapt_dim):
        #                     adapter = self.model.adapters[f"{p_name}_{dim_idx}"]
        #                     adapt_out = adapter(adapt_in)
        #                     adapt_outs.append(adapt_out)
        #                 adapt_out = tr.cat(adapt_outs, dim=-1)
        #             else:
        #                 adapt_out = self.model.adapters[p_name](adapt_in)
        #             adapt_out = self.model.adapter_acts[p_name](adapt_out)
        #             model_out[f"{p_name}_adapted"] = adapt_out.squeeze(-1)

        for p_name in self.temp_param_names_hat:
            p_hat = model_out[p_name]
            assert p_hat.size(0) == batch_size
            assert p_hat.size(1) == self.temp_param_n_frames
            temp_params_hat_raw[p_name] = p_hat
            if p_name in self.interp_temp_param_names_hat:
                p_hat = util.interpolate_dim(
                    p_hat, self.temp_param_n_samples, dim=1, align_corners=True
                )
            temp_params_hat[p_name] = p_hat
            if f"{p_name}_adapted" in model_out:
                p_hat_adapted = model_out[f"{p_name}_adapted"]
                if p_name in self.interp_temp_param_names_hat:
                    p_hat_adapted = util.interpolate_dim(
                        p_hat_adapted,
                        self.temp_param_n_samples,
                        dim=1,
                        align_corners=True,
                    )
                temp_params_hat[f"{p_name}_adapted"] = p_hat_adapted

        # Compute LFO distances and metrics ============================================
        lfo_dist_vals = {}
        lfo_metric_vals = {}
        if stage != "train":  # Only compute distances and metrics during val and test
            with tr.no_grad():
                p_hats = [
                    temp_params_hat_raw[p_name]
                    for p_name in self.compare_temp_param_names
                ]
                if p_hats:
                    p_hats = tr.stack(p_hats, dim=1)
                # Calc LFO distances
                for p_name in self.compare_temp_param_names:
                    p = temp_params_raw[p_name]
                    p_hat = temp_params_hat_raw[p_name]
                    if p_hat.ndim != 2:
                        continue
                    assert p.shape == p_hat.shape
                    # Calc p_hat_inv
                    p_hat_inv = util.compute_lstsq_with_bias(
                        x_hat=p_hat.unsqueeze(1), x=p.unsqueeze(1)
                    ).squeeze(1)
                    temp_params_hat_inv[p_name] = p_hat_inv
                    # Calc p_hat_inv_all
                    p_hat_inv_all = util.compute_lstsq_with_bias(
                        x_hat=p_hats, x=p.unsqueeze(1)
                    ).squeeze(1)
                    temp_params_hat_inv_all[p_name] = p_hat_inv_all
                    # Calc LFO distances
                    for dist_name in self.lfo_dists:
                        if stage == "val":
                            # Skip expensive distances during validation
                            if dist_name in ["mse", "fft", "dtw", "fd", "cd"]:
                                continue
                            # Skip derivative distances for now
                            # if "_d1" in dist_name:
                            #     continue
                            if "_d2" in dist_name:
                                continue
                        dist_fn = self.lfo_dists[dist_name]

                        # Calc raw
                        val = dist_fn(p_hat, p)
                        self.log(f"{stage}/{p_name}__{dist_name}", val, prog_bar=False)
                        lfo_dist_vals[f"{p_name}__{dist_name}"] = val.item()

                        # Calc inv
                        val = dist_fn(p_hat_inv, p)
                        self.log(
                            f"{stage}/{p_name}_inv__{dist_name}", val, prog_bar=False
                        )
                        lfo_dist_vals[f"{p_name}_inv__{dist_name}"] = val.item()
                        # Calc inv all
                        val = dist_fn(p_hat_inv_all, p)
                        self.log(
                            f"{stage}/{p_name}_inv_all__{dist_name}",
                            val,
                            prog_bar=False,
                        )
                        lfo_dist_vals[f"{p_name}_inv_all__{dist_name}"] = val.item()
                if stage == "test":  # Only compute LFO metrics during test to save time
                    # Calc LFO metrics raw
                    for p_name in self.temp_param_names:
                        p = temp_params_raw[p_name]
                        if p.ndim != 2:
                            continue
                        for metric_name in self.lfo_metrics:
                            metric_fn = self.lfo_metrics[metric_name]
                            val = metric_fn(p)
                            self.log(
                                f"{stage}/{p_name}__{metric_name}", val, prog_bar=False
                            )
                            lfo_metric_vals[f"{p_name}__{metric_name}"] = val.item()
                if stage != "train":
                    # Calc LFO metrics raw hat
                    for p_name in self.temp_param_names_hat:
                        p_hat = temp_params_hat_raw[p_name]
                        if p_hat.ndim != 2:
                            continue
                        for metric_name in self.lfo_metrics:
                            metric_fn = self.lfo_metrics[metric_name]
                            val = metric_fn(p_hat)
                            self.log(
                                f"{stage}/{p_name}_hat__{metric_name}",
                                val,
                                prog_bar=False,
                            )
                            lfo_metric_vals[f"{p_name}_hat__{metric_name}"] = val.item()
                    # Calc LFO metrics inv and inv all
                    for p_name in self.compare_temp_param_names:
                        p_hat_inv = temp_params_hat_inv[p_name]
                        if p_hat_inv.ndim != 2:
                            continue
                        p_hat_inv_all = temp_params_hat_inv_all[p_name]
                        for metric_name in self.lfo_metrics:
                            metric_fn = self.lfo_metrics[metric_name]
                            val = metric_fn(p_hat_inv)
                            self.log(
                                f"{stage}/{p_name}_inv__{metric_name}",
                                val,
                                prog_bar=False,
                            )
                            lfo_metric_vals[f"{p_name}_inv__{metric_name}"] = val.item()
                            val = metric_fn(p_hat_inv_all)
                            self.log(
                                f"{stage}/{p_name}_inv_all__{metric_name}",
                                val,
                                prog_bar=False,
                            )
                            lfo_metric_vals[f"{p_name}_inv_all__{metric_name}"] = (
                                val.item()
                            )

        assert not any(k in tsv_row_vals for k in lfo_dist_vals)
        assert not any(k in tsv_row_vals for k in lfo_metric_vals)
        tsv_row_vals.update(lfo_dist_vals)
        tsv_row_vals.update(lfo_metric_vals)

        # Postprocess log_spec_x =======================================================
        log_spec_x = model_out.get("log_spec_x")
        if log_spec_x is None:
            with tr.no_grad():
                log_spec_x = self.spectral_visualizer(x).squeeze(1)
        else:
            log_spec_x = log_spec_x[:, 0, :, :]

        # Generate audio x_hat =========================================================
        synth_out_hat = self.synth_hat(
            self.ac.n_samples,
            f0_hz,
            phase_hat,
            temp_params_hat,
            global_params={},
            other_params=other_params,
        )
        x_hat = synth_out_hat["env_audio"].unsqueeze(1)

        # Compute loss =================================================================
        if self.use_p_loss:
            loss = 0.0
            for p_name in self.compare_temp_param_names:
                p = temp_params_raw[p_name]
                p_hat = temp_params_hat_raw[p_name]
                assert p.shape == p_hat.shape
                p_loss = self.loss_func(p_hat, p)
                self.log(
                    f"{stage}/ploss__{p_name}__{self.loss_name}",
                    p_loss,
                    prog_bar=False,
                )
                loss += p_loss
            self.log(f"{stage}/ploss__{self.loss_name}", loss, prog_bar=False)
        else:
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/loss__{self.loss_name}", loss, prog_bar=False)
        self.log(f"{stage}/loss", loss, prog_bar=False)

        # Do manual gradient step if required ==========================================
        if stage == "train" and not self.automatic_optimization:
            for opt in self.optimizers():
                opt.zero_grad()
            self.manual_backward(loss)
            for opt in self.optimizers():
                self.clip_gradients(opt)
                opt.step()

        # Compute audio distances ======================================================
        audio_dist_vals = {}
        p_hats = tr.stack([p for p in temp_params_hat_raw.values()], dim=1)
        # p_hats = None
        if stage != "train":
            with tr.no_grad():
                for feat_name, feat_fn in self.audio_dists.items():
                    # # Skip expensive distances during validation
                    # if stage == "val":
                    #     if feat_name in ["rms", "sc", "sb", "sf"]:
                    #         continue
                    if feat_name in ["rms", "sc", "sb", "sf"]:
                        vals = feat_fn(x_hat, x, p_hats)
                    else:
                        vals = feat_fn(x_hat, x)
                    if isinstance(vals, dict):
                        vals = {f"{feat_name}__{k}": v for k, v in vals.items()}
                    else:
                        vals = {feat_name: vals}
                    for dist_name, val in vals.items():
                        self.log(f"{stage}/audio__{dist_name}", val, prog_bar=False)
                        audio_dist_vals[f"audio__{dist_name}"] = val.item()

        assert not any(k in tsv_row_vals for k in audio_dist_vals)
        tsv_row_vals.update(audio_dist_vals)

        # Compute FAD distances ========================================================
        fad_dists = {}
        if stage == "test":
            fad_dists = self.calc_fad_distances(x_hat, x)

        assert not any(k in tsv_row_vals for k in fad_dists)
        tsv_row_vals.update(fad_dists)

        # TSV logging ==================================================================
        if stage != "train" and self.tsv_path:
            tsv_row = [
                tr.random.initial_seed(),
                self.wt_name,
                stage,
                self.global_step,
                self.global_n,
                loss.item(),
            ]
            curr_tsv_row_len = len(tsv_row)
            for col in self.tsv_cols[curr_tsv_row_len:]:
                if stage == "test" and col not in tsv_row_vals:
                    log.warning(f"Missing TSV column: {col}")
                val = tsv_row_vals.get(col, "")
                tsv_row.append(val)
            assert len(tsv_row) == len(self.tsv_cols)
            with open(self.tsv_path, "a") as f:
                f.write("\t".join(str(v) for v in tsv_row) + "\n")

        # Prepare out_dict =============================================================
        out_dict = {
            "loss": loss,
            "add_audio": add_audio,
            "x": x,
            "x_hat": x_hat,
            "log_spec_x": log_spec_x,
        }
        out_dict.update(temp_params_raw)
        for p_name, p in temp_params_hat_raw.items():
            out_dict[f"{p_name}_hat"] = p
        for p_name, p in temp_params_hat_inv.items():
            out_dict[f"{p_name}_hat_inv"] = p
        for p_name, p in temp_params_hat_inv_all.items():
            out_dict[f"{p_name}_hat_inv_all"] = p
        assert not any(k in out_dict for k in lfo_dist_vals)
        out_dict.update(lfo_dist_vals)

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

    def calc_fad_distances(
        self, x_hat: T, x: T, n_workers: int = 0
    ) -> Dict[str, float]:
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
        fad_dists = {}
        for fad_model_name in self.fad_model_names:
            log.info(f"Calculating FAD for {fad_model_name}")
            try:
                fad_val = calc_fad(
                    fad_model_name,
                    baseline_dir=fad_dir_x,
                    eval_dir=fad_dir_x_hat,
                    workers=n_workers,
                )
                fad_dists[f"fad__{fad_model_name}"] = fad_val
                self.log(f"test/fad__{fad_model_name}", fad_val, prog_bar=False)
            except Exception as e:
                log.error(f"Error calculating FAD: {e}")
        shutil.rmtree(fad_dir_x)
        shutil.rmtree(fad_dir_x_hat)
        return fad_dists


class PreprocLightningModule(ModDiscoveryLightingModule):
    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T | Dict[str, T]]:
        audio = batch["audio"]
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase_hat = batch["phase_hat"]

        assert audio.ndim == 2
        assert (
            audio.size(1) == self.ac.n_samples
        ), f"Expected {self.ac.n_samples}, but got {audio.size(1)}"
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
