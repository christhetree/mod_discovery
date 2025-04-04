import logging
import math
import os
from collections import defaultdict
from contextlib import suppress
from typing import Any, Dict, List

import numpy as np
import torch as tr
import wandb
from auraloss.time import ESRLoss
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import Tensor as T, nn

from acid_ddsp.plotting import fig2img, plot_waveforms_stacked, plot_wavetable
from lightning import AcidDDSPLightingModule
from synth_modules import WavetableOsc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            latest_stat_str = {k: f"{v:.8f}" for k, v in latest_stat.items()}
            if latest_stat:
                log.info(f"\nCurrent LR: {latest_stat_str}")


class LogModSigAndSpecCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.out_dicts = {}
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AcidDDSPLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:  # TODO(cm): tmp
            return
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                x_hat = out_dict.get("x_hat")
                if x_hat is not None:
                    log_spec_x_hat = pl_module.spectral_visualizer(x_hat).squeeze(1)
                    out_dict["log_spec_x_hat"] = log_spec_x_hat
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                # TODO(cm): tmp
                # self.out_dicts[example_idx] = out_dict
                for idx in range(self.n_examples):
                    idx_out_dict = {
                        k: v[idx : idx + 1] for k, v in out_dict.items() if v.ndim > 0
                    }
                    self.out_dicts[idx] = idx_out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        images = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            log_spec_x = out_dict.get("log_spec_x")
            log_spec_x_hat = out_dict.get("log_spec_x_hat")

            y_coords = pl_module.spectral_visualizer.center_freqs
            y_ticks = [
                (idx, f"{f:.0f}")
                for idx, f in list(enumerate(y_coords))[:: len(y_coords) // 10]
            ]
            y_indices, y_tick_labels = zip(*y_ticks)
            vmin = None
            vmax = None
            if log_spec_x is not None and log_spec_x_hat is not None:
                vmin = min(log_spec_x[0].min(), log_spec_x_hat[0].min())
                vmax = max(log_spec_x[0].max(), log_spec_x_hat[0].max())

            title = f"{trainer.global_step}_idx_{example_idx}"
            fig, ax = plt.subplots(nrows=3, figsize=(6, 18), squeeze=True)
            fig.suptitle(title, fontsize=14)

            if log_spec_x is not None:
                ax[0].imshow(
                    log_spec_x[0].numpy(),
                    extent=[0, log_spec_x.size(2), 0, log_spec_x.size(1)],
                    aspect=log_spec_x.size(2) / log_spec_x.size(1),
                    origin="lower",
                    cmap="magma",
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[0].set_xlabel("n_frames")
                ax[0].set_yticks(y_indices, y_tick_labels)
                ax[0].set_ylabel("Freq (Hz)")
                ax[0].set_title("log_spec_x")

            if log_spec_x_hat is not None:
                ax[1].imshow(
                    log_spec_x_hat[0].numpy(),
                    extent=[0, log_spec_x_hat.size(2), 0, log_spec_x_hat.size(1)],
                    aspect=log_spec_x_hat.size(2) / log_spec_x_hat.size(1),
                    origin="lower",
                    cmap="magma",
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[1].set_xlabel("n_frames")
                ax[1].set_yticks(y_indices, y_tick_labels)
                ax[1].set_ylabel("Freq (Hz)")
                ax[1].set_title("log_spec_x_hat")

            # TODO(cm): tmp
            temp_params_all = [
                (out_dict.get("env"), "orchid", None),
                (out_dict.get("env_hat"), "purple", None),
                # (out_dict.get("env_hat_inv"), "purple", ":"),
                # (out_dict.get("env_hat_inv_all"), "purple", "--"),
                (out_dict.get("add_lfo"), "lightcoral", None),
                (out_dict.get("add_lfo_hat"), "red", None),
                # (out_dict.get("add_lfo_hat_inv"), "red", ":"),
                # (out_dict.get("add_lfo_hat_inv_all"), "red", "--"),
                (out_dict.get("sub_lfo"), "lightblue", None),
                (out_dict.get("sub_lfo_hat"), "blue", None),
                # (out_dict.get("sub_lfo_hat_inv"), "blue", ":"),
                # (out_dict.get("sub_lfo_hat_inv_all"), "blue", "--"),
            ]
            n_frames = None

            for temp_param, color, linestyle in temp_params_all:
                if temp_param is None:
                    continue
                if temp_param.ndim != 2:
                    continue
                if temp_param.min() < 0.0 or temp_param.max() > 1.0:
                    if linestyle is None:  # Only normalize _hat
                        tp_min = temp_param.min()
                        tp_range = temp_param.max() - tp_min
                        temp_param = (temp_param - tp_min) / tp_range
                        linestyle = "-."
                curr_n_frames = temp_param.size(1)
                if n_frames is None:
                    n_frames = curr_n_frames
                else:
                    assert n_frames == curr_n_frames
                temp_param = temp_param[0].numpy()
                ax[-1].plot(temp_param, color=color, linestyle=linestyle)

            ax[-1].set_xlabel("n_samples")
            ax[-1].set_ylabel("Amplitude")
            ax[-1].set_ylim(-0.1, 1.1)
            ax[-1].set_title(
                # f"env (blu), ms (blk), ms_hat (orange), p{degree}s{n_segments} (red)\n"
                # f"env (blue), mod_sig (black), mod_sig_hat (orange)\n"
                f"env (purple), add_lfo (red), sub_lfo (blue)\n"
                # f"ms_l1: {mod_sig_l1:.3f}, ms_esr: {mod_sig_esr:.3f}\n"
                # f"q: {q[0]:.2f}, q': {q_hat[0]:.2f}, "
                # f"dg: {dist_gain[0]:.2f}, dg': {dist_gain_hat[0]:.2f}, "
                # f"la: {learned_alpha[0]:.2f}, la': {learned_alpha_hat[0]:.2f}\n"
                # f"os: {osc_shape[0]:.2f}, os': {osc_shape_hat[0]:.2f}, "
                # f"og: {osc_gain[0]:.2f}, og': {osc_gain_hat[0]:.2f}"
            )

            v_line_x = np.linspace(0, n_frames, 13)
            ax[-1].vlines(
                x=v_line_x,
                ymin=-0.1,
                ymax=1.1,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            fig.tight_layout()
            img = fig2img(fig)
            images.append(img)

        if images and wandb.run:
            wandb.log(
                {
                    "spectrograms": [wandb.Image(i) for i in images],
                    "global_step": trainer.global_step,
                }
            )

        self.out_dicts.clear()


# TODO(cm): make ABC
class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5, min_render_time_sec: float = 1.0) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.min_render_time_sec = min_render_time_sec
        self.out_dicts = {}
        self.columns = ["row_id"] + [f"idx_{idx}" for idx in range(n_examples)]
        # TODO(cm): tmp
        # self.columns = ["row_id"] + [f"idx_{idx}" for idx in range(3)]
        self.rows = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AcidDDSPLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:  # TODO(cm): tmp
            return
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                # TODO(cm): tmp
                # self.out_dicts[example_idx] = out_dict
                for idx in range(self.n_examples):
                    idx_out_dict = {
                        k: v[idx : idx + 1] for k, v in out_dict.items() if v.ndim > 0
                    }
                    self.out_dicts[idx] = idx_out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        sample_time_sec = pl_module.ac.buffer_size_seconds
        n_repeat = math.ceil(self.min_render_time_sec / sample_time_sec)

        images = []
        add_audio_waveforms = []
        x_waveforms = []
        x_hat_waveforms = []
        x_eval_waveforms = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            title = f"{trainer.global_step}_idx_{example_idx}"
            add_audio = out_dict.get("add_audio")
            x = out_dict.get("x")
            if x is not None:
                x = x.squeeze(1)
            x_hat = out_dict.get("x_hat")
            if x_hat is not None:
                x_hat = x_hat.squeeze(1)
            x_eval = out_dict.get("x_eval")
            if x_eval is not None:
                x_eval = x_eval.squeeze(1)
            waveforms = []
            labels = []

            # TODO(cm): tmp
            if x_hat.size(0) == 0:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            if add_audio is not None:
                add_audio = add_audio[0:1]
                waveforms.append(add_audio)
                labels.append("add_audio")
                add_audio = add_audio.repeat(1, n_repeat)
                add_audio_waveforms.append(add_audio.swapaxes(0, 1).numpy())
            if x is not None:
                x = x[0:1]
                waveforms.append(x)
                labels.append("x")
                x = x.repeat(1, n_repeat)
                x_waveforms.append(x.swapaxes(0, 1).numpy())
            if x_hat is not None:
                x_hat = x_hat[0:1]
                waveforms.append(x_hat)
                labels.append("x_hat")
                x_hat = x_hat.repeat(1, n_repeat)
                x_hat_waveforms.append(x_hat.swapaxes(0, 1).numpy())
            if x_eval is not None:
                x_eval = x_eval[0:1]
                waveforms.append(x_eval)
                labels.append("x_eval")
                x_eval = x_eval.repeat(1, n_repeat)
                x_eval_waveforms.append(x_eval.swapaxes(0, 1).numpy())

            fig = plot_waveforms_stacked(waveforms, pl_module.ac.sr, title, labels)
            img = fig2img(fig)
            images.append(img)

        if images and wandb.run:
            wandb.log(
                {
                    "waveforms": [wandb.Image(i) for i in images],
                    "global_step": trainer.global_step,
                }
            )

        data = defaultdict(list)
        if add_audio_waveforms:
            data["add_audio"].append(f"{trainer.global_step}_add_audio")
        for idx, curr_dry in enumerate(add_audio_waveforms):
            data["add_audio"].append(
                wandb.Audio(
                    curr_dry,
                    caption=f"{trainer.global_step}_add_audio_{idx}",
                    sample_rate=int(pl_module.ac.sr),
                )
            )
        if x_waveforms:
            data["x"].append(f"{trainer.global_step}_x")
        for idx, curr_wet in enumerate(x_waveforms):
            data["x"].append(
                wandb.Audio(
                    curr_wet,
                    caption=f"{trainer.global_step}_x_{idx}",
                    sample_rate=int(pl_module.ac.sr),
                )
            )
        if x_hat_waveforms:
            data["x_hat"].append(f"{trainer.global_step}_x_hat")
        for idx, curr_wet_hat in enumerate(x_hat_waveforms):
            data["x_hat"].append(
                wandb.Audio(
                    curr_wet_hat,
                    caption=f"{trainer.global_step}_x_hat_{idx}",
                    sample_rate=int(pl_module.ac.sr),
                )
            )
        if x_eval_waveforms:
            data["x_eval"].append(f"{trainer.global_step}_x_eval")
        for idx, curr_wet_eval in enumerate(x_eval_waveforms):
            data["x_eval"].append(
                wandb.Audio(
                    curr_wet_eval,
                    caption=f"{trainer.global_step}_x_eval_{idx}",
                    sample_rate=int(pl_module.ac.sr),
                )
            )
        data = list(data.values())
        for row in data:
            self.rows.append(row)
        if wandb.run:
            wandb.log(
                {
                    "audio": wandb.Table(columns=self.columns, data=self.rows),
                }
            )
            self.out_dicts.clear()


class LogWavetablesCallback(Callback):
    def create_wt_images(self, osc: WavetableOsc, title: str) -> List[T]:
        images = []
        wt = osc.get_wt().detach().cpu()
        fig = plot_wavetable(wt, title)
        img = fig2img(fig)
        images.append(img)
        wt_pitch_hz = tr.tensor([osc.wt_pitch_hz]).unsqueeze(1).to(osc.window.device)
        wt_bounded = osc.get_maybe_aa_maybe_bounded_wt(wt_pitch_hz).detach().cpu()
        wt_bounded = wt_bounded[0]
        fig = plot_wavetable(wt_bounded, f"{title}__aa_bounded")
        img = fig2img(fig)
        images.append(img)
        return images

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        images = []
        with suppress(Exception):
            title = f"{trainer.global_step}_wt"
            osc = pl_module.synth.add_synth_module
            images += self.create_wt_images(osc, title)

        with suppress(Exception):
            title = f"{trainer.global_step}_wt_hat"
            osc_hat = pl_module.synth_hat.add_synth_module
            images += self.create_wt_images(osc_hat, title)

        with suppress(Exception):
            title = f"{trainer.global_step}_wt_eval"
            osc_eval = pl_module.synth_eval.add_synth_module
            images += self.create_wt_images(osc_eval, title)

        if images and wandb.run:
            wandb.log(
                {
                    "wavetables": [wandb.Image(i) for i in images],
                    "global_step": trainer.global_step,
                }
            )
