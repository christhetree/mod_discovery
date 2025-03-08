import logging
import os
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
from feature_extraction import LogMelSpecFeatureExtractor
from losses import MFCCL1
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
        synth_eval: Optional[SynthBase] = None,
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
        self.synth_eval = synth_eval
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

        self.audio_metrics = nn.ModuleDict()
        self.audio_metrics["mss"] = auraloss.freq.MultiResolutionSTFTLoss()
        self.audio_metrics["mel_stft"] = auraloss.freq.MelSTFTLoss(
            sample_rate=self.ac.sr,
            fft_size=spectral_visualizer.n_fft,
            hop_size=spectral_visualizer.hop_len,
            win_length=spectral_visualizer.n_fft,
            n_mels=spectral_visualizer.n_mels,
        )
        self.audio_metrics["mfcc"] = MFCCL1(
            sr=self.ac.sr,
            log_mels=True,
            n_fft=spectral_visualizer.n_fft,
            hop_len=spectral_visualizer.hop_len,
            n_mels=spectral_visualizer.n_mels,
        )

        self.global_n = 0
        self.test_out_dicts = []

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
            tsv_cols.append(f"esr__{p_name}")
            tsv_cols.append(f"esr_inv__{p_name}")
        for p_name in self.global_param_names:
            tsv_cols.append(f"l1__{p_name}")
        for metric_name in self.audio_metrics:
            tsv_cols.append(f"audio__{metric_name}")
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

    def on_train_start(self) -> None:
        self.global_n = 0

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
        if "add_lfo" in temp_params:
            other_params["add_lfo"] = temp_params["add_lfo"]
        else:
            other_mod_sig = list(temp_params.values())[0]
            add_mod_sig = tr.zeros_like(other_mod_sig)
            # add_mod_sig = tr.ones_like(other_mod_sig)
            other_params["add_lfo"] = add_mod_sig
        if "wt" in batch:
            other_params["wt"] = batch["wt"]
        if "q_0to1" in batch:
            other_params["q_mod_sig"] = batch["q_0to1"]
        filter_types = ["lp", "hp", "bp", "no"]
        if "filter_type_0to1" in batch:
            filter_type_0to1 = batch["filter_type_0to1"][0]
            filter_type_idx = int(filter_type_0to1 * len(filter_types))
            filter_type = filter_types[filter_type_idx]
            other_params["filter_type"] = filter_type

        # Postprocess q_hat TODO(cm): generalize
        # if "q" in self.global_param_names:
        #     q_0to1 = batch["q_0to1"]
        #     q_mod_sig = q_0to1.unsqueeze(-1)
        #     temp_params["q_mod_sig"] = q_mod_sig

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
        temp_params = batch["temp_params"]
        global_params_0to1 = batch["global_params_0to1"]
        global_params = batch["global_params"]
        other_params = batch["other_params"]

        # Get optional params
        add_audio = batch.get("add_audio")

        model_in_dict = {
            "audio": x,
            "f0_hz": f0_hz,
        }
        temp_params_hat = {}
        global_params_0to1_hat = {}
        global_params_hat = {}
        log_spec_x = None

        temp_param_metrics = {}
        global_param_metrics = {}

        temp_params_inv = {}
        temp_params_inv_hat = {}

        # Perform model forward pass
        if self.use_model:
            model_out = self.model(model_in_dict)

            # Postprocess temp_params_hat
            for p_name in self.temp_param_names_hat:
                p_hat = model_out[p_name]
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
                        self.log(f"{stage}/{p_name}_l1", l1, prog_bar=False)
                        temp_param_metrics[f"{p_name}_l1"] = l1
                        self.log(f"{stage}/{p_name}_esr", esr, prog_bar=False)
                        temp_param_metrics[f"{p_name}_esr"] = esr

                        # Do invariant comparison
                        if stage != "train":
                            l1_inv, p_inv, p_inv_hat, inv_name = (
                                self.invariant_compare_individual(
                                    p,
                                    p_hat_interp,
                                    self.l1,
                                    normalize=True,
                                    vflip=True,
                                    eps=self.eps,
                                )
                            )
                            # log.info(f"l1 inv_name = {inv_name}")
                            self.log(f"{stage}/{p_name}_l1_inv", l1_inv, prog_bar=False)
                            temp_param_metrics[f"{p_name}_l1_inv"] = l1_inv
                            temp_params_inv[p_name] = p_inv
                            temp_params_inv_hat[p_name] = p_inv_hat
                            esr_inv, p_inv, p_inv_hat, inv_name = (
                                self.invariant_compare_individual(
                                    p,
                                    p_hat_interp,
                                    self.esr,
                                    normalize=True,
                                    vflip=True,
                                    eps=self.eps,
                                )
                            )
                            # log.info(f"esr inv_name = {inv_name}")
                            self.log(
                                f"{stage}/{p_name}_esr_inv", esr_inv, prog_bar=False
                            )
                            temp_param_metrics[f"{p_name}_esr_inv"] = esr_inv
                            # temp_params_inv[p_name] = p_inv
                            # temp_params_inv_hat[p_name] = p_inv_hat
                        else:
                            temp_param_metrics[f"{p_name}_l1_inv"] = tr.tensor(-1)
                            temp_param_metrics[f"{p_name}_esr_inv"] = tr.tensor(-1)
                            temp_params_inv[p_name] = p
                            temp_params_inv_hat[p_name] = p_hat_interp

                # Config decides which temp params are interpolated for synth_hat
                if (
                    p_name in self.interp_temp_param_names_hat
                    and p_hat_interp is not None
                ):
                    temp_params_hat[p_name] = p_hat_interp
                else:
                    temp_params_hat[p_name] = p_hat
                if f"{p_name}_adapted" in model_out:
                    p_hat = model_out[f"{p_name}_adapted"]
                    # Config decides which temp params are interpolated for synth_hat
                    if (
                        p_name in temp_params
                        and p_name in self.interp_temp_param_names_hat
                    ):
                        p_hat = util.interpolate_dim(
                            p_hat,
                            temp_params[p_name].size(1),
                            dim=1,
                            align_corners=True,
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
        for metric_name, metric in self.audio_metrics.items():
            with tr.no_grad():
                audio_metric = metric(x_hat, x)
            audio_metrics_hat[metric_name] = audio_metric
            self.log(f"{stage}/audio_{metric_name}", audio_metric, prog_bar=False)

        # Log eval synth metrics if applicable
        x_eval = None
        audio_metrics_eval = {}
        if stage != "train" and self.synth_eval is not None:
            # Generate audio x_eval
            try:
                synth_out_eval = self.synth_eval(
                    self.ac.n_samples,
                    f0_hz,
                    phase_hat,
                    temp_params_hat,
                    global_params_hat,
                    other_params,
                )
                x_eval = synth_out_eval["env_audio"].unsqueeze(1)
                for metric_name, metric in self.audio_metrics.items():
                    with tr.no_grad():
                        audio_metric = metric(x_eval, x)
                    audio_metrics_eval[metric_name] = audio_metric
                    self.log(
                        f"{stage}/audio_{metric_name}_eval",
                        audio_metric,
                        prog_bar=False,
                    )
            except Exception as e:
                log.error(f"Error in eval synth: {e}")

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
                for p_name in self.temp_param_names:
                    tsv_row.append(temp_param_metrics[f"{p_name}_l1"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_l1_inv"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_esr"].item())
                    tsv_row.append(temp_param_metrics[f"{p_name}_esr_inv"].item())
                for p_name in self.global_param_names:
                    tsv_row.append(global_param_metrics[f"{p_name}_l1"].item())
                for metric_name in self.audio_metrics:
                    tsv_row.append(audio_metrics_hat[metric_name].item())
                f.write("\t".join(str(v) for v in tsv_row) + "\n")

        temp_params_hat = {f"{k}_hat": v for k, v in temp_params_hat.items()}
        temp_params_hat = {
            k: util.interpolate_dim(v, n=self.ac.n_samples, dim=1, align_corners=True)
            for k, v in temp_params_hat.items()
        }
        temp_params_inv = {f"{k}_inv": v for k, v in temp_params_inv.items()}
        temp_params_inv_hat = {
            f"{k}_inv_hat": v for k, v in temp_params_inv_hat.items()
        }
        temp_params_inv_hat = {
            k: util.interpolate_dim(v, n=self.ac.n_samples, dim=1, align_corners=True)
            for k, v in temp_params_inv_hat.items()
        }
        global_params_hat = {f"{k}_hat": v for k, v in global_params_hat.items()}
        audio_metrics_hat = {f"{k}_hat": v for k, v in audio_metrics_hat.items()}
        audio_metrics_eval = {f"{k}_eval": v for k, v in audio_metrics_eval.items()}
        out_dict = {
            "loss": loss,
            "add_audio": add_audio,
            "x": x,
            "x_hat": x_hat,
            "x_eval": x_eval,
            "log_spec_x": log_spec_x,
            # TODO(cm): tmp
            # "add_lfo_seg_indices_hat": model_out["add_lfo_seg_indices"],
            # "sub_lfo_seg_indices_hat": model_out["sub_lfo_seg_indices"],
        }
        assert all(k not in out_dict for k in temp_params)
        assert all(k not in out_dict for k in temp_params_hat)
        assert all(k not in out_dict for k in temp_param_metrics)
        assert all(k not in out_dict for k in temp_params_inv)
        assert all(k not in out_dict for k in temp_params_inv_hat)
        assert all(k not in out_dict for k in global_params)
        assert all(k not in out_dict for k in global_params_hat)
        assert all(k not in out_dict for k in global_param_metrics)
        assert all(k not in out_dict for k in audio_metrics_hat)
        assert all(k not in out_dict for k in audio_metrics_eval)
        out_dict.update(temp_params)
        out_dict.update(temp_params_hat)
        out_dict.update(temp_params_inv)
        out_dict.update(temp_params_inv_hat)
        out_dict.update(temp_param_metrics)
        out_dict.update(global_params)
        out_dict.update(global_params_hat)
        out_dict.update(global_param_metrics)
        out_dict.update(audio_metrics_hat)
        out_dict.update(audio_metrics_eval)
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        out = self.step(batch, stage="test")
        self.test_out_dicts.append(out)
        return out

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
    def invariant_compare_batch(
        x: T,
        x_hat: T,
        compare_fn: nn.Module,
        normalize: bool = True,
        vflip: bool = True,
        eps: float = 1e-8,
    ) -> (T, T, T, str):
        assert x.ndim == 2
        assert x.shape == x_hat.shape
        comparisons = [(x, x_hat, "default")]
        if vflip:
            x_hat_vflip = AcidDDSPLightingModule.flip_signal_vertically(x_hat)
            comparisons.append((x, x_hat_vflip, "vflip"))
        if normalize:
            x_norm = AcidDDSPLightingModule.normalize_signal(x, eps)
            x_hat_norm = AcidDDSPLightingModule.normalize_signal(x_hat, eps)
            comparisons.append((x_norm, x_hat_norm, "norm"))
            if vflip:
                x_hat_norm_flipped = AcidDDSPLightingModule.flip_signal_vertically(
                    x_hat_norm
                )
                comparisons.append((x_norm, x_hat_norm_flipped, "norm_vflip"))

        min_val = float("inf")
        min_x = None
        min_x_hat = None
        min_name = None
        for x, x_hat, name in comparisons:
            with tr.no_grad():
                val = compare_fn(x_hat, x)
            if val < min_val:
                min_val = val
                min_x = x
                min_x_hat = x_hat
                min_name = name
        return min_val, min_x, min_x_hat, min_name

    @staticmethod
    def invariant_compare_individual(
        x: T,
        x_hat: T,
        compare_fn: nn.Module,
        normalize: bool = True,
        vflip: bool = True,
        eps: float = 1e-8,
    ) -> (T, T, T, List[str]):
        min_vals = []
        min_x_s = []
        min_x_hat_s = []
        min_names = []
        bs = x.size(0)
        for idx in range(bs):
            curr_x = x[idx : idx + 1]
            curr_x_hat = x_hat[idx : idx + 1]
            min_val, min_x, min_x_hat, min_name = (
                AcidDDSPLightingModule.invariant_compare_batch(
                    curr_x, curr_x_hat, compare_fn, normalize, vflip, eps
                )
            )
            min_vals.append(min_val)
            min_x_s.append(min_x)
            min_x_hat_s.append(min_x_hat)
            min_names.append(min_name)
        min_val = tr.stack(min_vals, dim=0).mean()
        min_x = tr.cat(min_x_s, dim=0)
        min_x_hat = tr.cat(min_x_hat_s, dim=0)
        return min_val, min_x, min_x_hat, min_names

    # def on_test_epoch_end(self) -> None:
    #     tsv_rows = []
    #
    #     test_metrics = ["loss"]
    #     for metric_name in self.audio_metrics:
    #         test_metrics.append(f"{metric_name}_hat")
    #         test_metrics.append(f"{metric_name}_eval")
    #
    #     for metric_name in test_metrics:
    #         metric_values = [d.get(metric_name) for d in self.test_out_dicts]
    #         if any(v is None for v in metric_values):
    #             log.warning(f"Skipping test metric: {metric_name}")
    #             continue
    #         metric_values = tr.stack(metric_values, dim=0)
    #         assert metric_values.ndim == 1
    #         metric_mean = metric_values.mean()
    #         metric_std = metric_values.std()
    #         metric_ci95 = 1.96 * scipy.stats.sem(metric_values.numpy())
    #         self.log(f"test/{metric_name}", metric_mean, prog_bar=False)
    #         tsv_rows.append(
    #             [
    #                 metric_name,
    #                 metric_mean.item(),
    #                 metric_std.item(),
    #                 metric_ci95,
    #                 metric_values.size(0),
    #                 metric_values.numpy(),
    #             ]
    #         )
    #
    #     for fad_model_name in self.fad_model_names:
    #         fad_hat_values = []
    #         fad_eval_values = []
    #         for out in self.test_out_dicts:
    #             wet = out["wet"]
    #             wet_hat = out["wet_hat"]
    #             wet_eval = out.get("wet_eval")
    #
    #             fad_wet_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet")
    #             fad_wet_hat_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet_hat")
    #             save_and_concat_fad_audio(
    #                 self.ac.sr,
    #                 wet,
    #                 fad_wet_dir,
    #                 fade_n_samples=self.spectral_visualizer.hop_len,
    #             )
    #             save_and_concat_fad_audio(
    #                 self.ac.sr,
    #                 wet_hat,
    #                 fad_wet_hat_dir,
    #                 fade_n_samples=self.spectral_visualizer.hop_len,
    #             )
    #             clean_up_baseline = True
    #             if wet_eval is not None:
    #                 clean_up_baseline = False
    #             fad_hat = calc_fad(
    #                 fad_model_name,
    #                 baseline_dir=fad_wet_dir,
    #                 eval_dir=fad_wet_hat_dir,
    #                 clean_up_baseline=clean_up_baseline,
    #                 clean_up_eval=True,
    #             )
    #             fad_hat_values.append(fad_hat)
    #             if wet_eval is not None:
    #                 fad_wet_eval_dir = os.path.join(
    #                     OUT_DIR, f"{self.run_name}__fad_wet_eval"
    #                 )
    #                 save_and_concat_fad_audio(
    #                     self.ac.sr,
    #                     wet_eval,
    #                     fad_wet_eval_dir,
    #                     fade_n_samples=self.spectral_visualizer.hop_len,
    #                 )
    #                 fad_eval = calc_fad(
    #                     fad_model_name,
    #                     baseline_dir=fad_wet_dir,
    #                     eval_dir=fad_wet_eval_dir,
    #                     clean_up_baseline=True,
    #                     clean_up_eval=True,
    #                 )
    #                 fad_eval_values.append(fad_eval)
    #
    #         fad_hat_mean = np.mean(fad_hat_values)
    #         fad_hat_std = np.std(fad_hat_values)
    #         fad_hat_ci95 = 1.96 * scipy.stats.sem(fad_hat_values)
    #         self.log(f"test/fad_{fad_model_name}_hat", fad_hat_mean, prog_bar=False)
    #         tsv_rows.append(
    #             [
    #                 f"fad_{fad_model_name}_hat",
    #                 fad_hat_mean,
    #                 fad_hat_std,
    #                 fad_hat_ci95,
    #                 len(fad_hat_values),
    #                 fad_hat_values,
    #             ]
    #         )
    #         if fad_eval_values:
    #             fad_eval_mean = np.mean(fad_eval_values)
    #             fad_eval_std = np.std(fad_eval_values)
    #             fad_eval_ci95 = 1.96 * scipy.stats.sem(fad_eval_values)
    #             self.log(
    #                 f"test/fad_{fad_model_name}_eval", fad_eval_mean, prog_bar=False
    #             )
    #             tsv_rows.append(
    #                 [
    #                     f"fad_{fad_model_name}_eval",
    #                     fad_eval_mean,
    #                     fad_eval_std,
    #                     fad_eval_ci95,
    #                     len(fad_eval_values),
    #                     fad_eval_values,
    #                 ]
    #             )
    #
    #     tsv_path = os.path.join(OUT_DIR, f"{self.run_name}__test.tsv")
    #     if os.path.exists(tsv_path):
    #         log.warning(f"Overwriting existing TSV file: {tsv_path}")
    #     df = pd.DataFrame(
    #         tsv_rows, columns=["metric_name", "mean", "std", "ci95", "n", "values"]
    #     )
    #     df.to_csv(tsv_path, sep="\t", index=False)
    #
    #     # H = tr.cat([out["H"] for out in self.test_outs], dim=0)
    #     # H = H.detach().cpu()
    #     # H_path = os.path.join(OUT_DIR, f"{self.run_name}__H.pt")
    #     # tr.save(H, H_path)
    #     # log.info(f"Saved H to: {H_path}")


class PreprocLightningModule(AcidDDSPLightingModule):
    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        wet = batch["wet"]
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        assert wet.ndim == 2
        assert wet.size(1) == self.ac.n_samples, f"{wet.size(1)}, {self.ac.n_samples}"
        batch_size = wet.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        return batch
