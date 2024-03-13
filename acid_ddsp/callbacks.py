import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict

import wandb
from auraloss.time import ESRLoss
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T, nn

from acid_ddsp.plotting import fig2img, plot_waveforms_stacked
from lightning import AcidDDSPLightingModule

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
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                if "x_hat" in out_dict and out_dict["x_hat"] is not None:
                    x_hat = out_dict["x_hat"].unsqueeze(1)
                    log_spec_x_hat = pl_module.spectral_visualizer(x_hat).squeeze(1)
                    out_dict["log_spec_x_hat"] = log_spec_x_hat
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                self.out_dicts[example_idx] = out_dict

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
            envelope = out_dict.get("envelope")
            mod_sig = out_dict.get("mod_sig")
            mod_sig_hat = out_dict.get("mod_sig_hat")
            q = out_dict.get("q", [-1])
            mod_sig_esr = -1
            mod_sig_l1 = -1

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
            if mod_sig is not None and mod_sig_hat is not None:
                mod_sig_esr = self.esr(mod_sig[0], mod_sig_hat[0]).item()
                mod_sig_l1 = self.l1(mod_sig[0], mod_sig_hat[0]).item()

            title = f"idx_{example_idx}"
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

            if pl_module.log_envelope and envelope is not None:
                ax[2].plot(envelope[0].numpy(), label="env", color="blue")
                ax[2].set(aspect=envelope.size(1))

            if mod_sig is not None:
                mod_sig_np = mod_sig[0].numpy()
                ax[2].plot(mod_sig_np, label="ms", color="black")
                ax[2].set(aspect=mod_sig.size(1))
                # mod_sig_fitted = piecewise_fitting_noncontinuous(
                #     mod_sig_np, degree=degree, n_knots=n_segments - 1
                # )
                # ax[2].plot(
                #     mod_sig_fitted,
                #     label=f"poly{degree}s{n_segments}",
                #     color="red"
                # )

            if mod_sig_hat is not None:
                ax[2].plot(mod_sig_hat[0].numpy(), label="ms_hat", color="orange")
                ax[2].set(aspect=mod_sig_hat.size(1))

            ax[2].set_xlabel("n_samples")
            ax[2].set_ylabel("Amplitude")
            ax[2].set_ylim(0, 1)
            ax[2].set_title(
                # f"env (blu), ms (blk), ms_hat (orange), p{degree}s{n_segments} (red)\n"
                f"env (blue), mod_sig (black), mod_sig_hat (orange)\n"
                f"ms_l1: {mod_sig_l1:.3f}, ms_esr: {mod_sig_esr:.3f}, q: {q[0]:.3f}"
            )

            fig.tight_layout()
            img = fig2img(fig)
            images.append(img)

        if images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="spectrograms", images=images, step=trainer.global_step
                    )

        self.out_dicts.clear()


# TODO(cm): make ABC
class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5, render_time_sec: float = 1.0) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.render_time_sec = render_time_sec
        self.out_dicts = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AcidDDSPLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                self.out_dicts[example_idx] = out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        sample_time_sec = pl_module.ac.buffer_size_seconds
        n_repeat = math.ceil(self.render_time_sec / sample_time_sec)

        images = []
        osc_audio_waveforms = []
        x_waveforms = []
        x_hat_waveforms = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            title = f"idx_{example_idx}"
            osc_audio = out_dict.get("osc_audio")
            x = out_dict.get("x")
            x_hat = out_dict.get("x_hat")
            waveforms = []
            labels = []

            if osc_audio is not None:
                osc_audio = osc_audio[0:1]
                waveforms.append(osc_audio)
                labels.append("osc_audio")
                osc_audio = osc_audio.repeat(1, n_repeat)
                osc_audio_waveforms.append(osc_audio.swapaxes(0, 1).numpy())
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

            fig = plot_waveforms_stacked(waveforms, pl_module.ac.sr, title, labels)
            img = fig2img(fig)
            images.append(img)

        for logger in trainer.loggers:
            # TODO(cm): enable for tensorboard as well
            if isinstance(logger, WandbLogger):
                if images:
                    logger.log_image(
                        key="waveforms", images=images, step=trainer.global_step
                    )

                data = defaultdict(list)
                columns = [f"idx_{idx}" for idx in range(len(images))]
                for idx, curr_osc_audio in enumerate(osc_audio_waveforms):
                    data["osc_audio"].append(
                        wandb.Audio(
                            curr_osc_audio,
                            caption=f"osc_audio_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                for idx, curr_x_audio in enumerate(x_waveforms):
                    data["x"].append(
                        wandb.Audio(
                            curr_x_audio,
                            caption=f"x_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                for idx, curr_x_hat_audio in enumerate(x_hat_waveforms):
                    data["x_hat"].append(
                        wandb.Audio(
                            curr_x_hat_audio,
                            caption=f"x_hat_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                data = list(data.values())
                logger.log_table(
                    key="audio", columns=columns, data=data, step=trainer.global_step
                )

        self.out_dicts.clear()
