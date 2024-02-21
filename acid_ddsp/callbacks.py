import logging
import os
from collections import defaultdict
from typing import Any, Dict

import wandb
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T

from acid_ddsp.plotting import fig2img, plot_waveforms_stacked

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self,
                             trainer: Trainer,
                             *args: Any,
                             **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            latest_stat_str = {k: f"{v:.8f}" for k, v in latest_stat.items()}
            if latest_stat:
                log.info(f"\nCurrent LR: {latest_stat_str}")


class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.x_audio = []
        self.x_hat_audio = []
        self.images = []

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightingModule,
                                out_dict: Dict[str, T],
                                batch: (T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        out_dict = {k: v.detach().cpu() for k, v in out_dict.items() if v is not None}

        x = out_dict.get("x")
        x_hat = out_dict.get("x_hat")
        if x is None and x_hat is None:
            log.debug(f"x and x_hat are both None, cannot log audio")
            return

        theta_density = out_dict["theta_density"]
        theta_slope = out_dict["theta_slope"]
        theta_density_hat = out_dict["theta_density_hat"]
        theta_slope_hat = out_dict["theta_slope_hat"]

        n_batches = theta_density.size(0)
        if batch_idx == 0:
            self.images = []
            self.x_audio = []
            self.x_hat_audio = []
            for idx in range(self.n_examples):
                if idx < n_batches:
                    waveforms = []
                    labels = []
                    if x is not None:
                        curr_x = x[idx]
                        waveforms.append(curr_x)
                        labels.append("x")
                        self.x_audio.append(curr_x.swapaxes(0, 1).numpy())
                    if x_hat is not None:
                        curr_x_hat = x_hat[idx]
                        waveforms.append(curr_x_hat)
                        labels.append("x_hat")
                        self.x_hat_audio.append(curr_x_hat.swapaxes(0, 1).numpy())

                    title = (f"batch_idx_{idx}, "
                             f"θd: {theta_density[idx]:.2f} -> "
                             f"{theta_density_hat[idx]:.2f}, "
                             f"θs: {theta_slope[idx]:.2f} -> "
                             f"{theta_slope_hat[idx]:.2f}")

                    fig = plot_waveforms_stacked(waveforms,
                                                 pl_module.synth.sr,
                                                 title,
                                                 labels)
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(self,
                                trainer: Trainer,
                                pl_module: SCRAPLLightingModule) -> None:
        for logger in trainer.loggers:
            # TODO(cm): enable for tensorboard as well
            if isinstance(logger, WandbLogger):
                logger.log_image(key="waveforms",
                                 images=self.images,
                                 step=trainer.global_step)
                data = defaultdict(list)
                columns = [f"idx_{idx}" for idx in range(len(self.images))]
                for idx, curr_x_audio in enumerate(self.x_audio):
                    data["x_audio"].append(
                        wandb.Audio(curr_x_audio,
                                    caption=f"x_{idx}",
                                    sample_rate=int(pl_module.synth.sr))
                    )
                for idx, curr_x_hat_audio in enumerate(self.x_hat_audio):
                    data["x_hat_audio"].append(
                        wandb.Audio(curr_x_hat_audio,
                                    caption=f"x_hat_{idx}",
                                    sample_rate=int(pl_module.synth.sr))
                    )
                data = list(data.values())
                logger.log_table(key="audio",
                                 columns=columns,
                                 data=data,
                                 step=trainer.global_step)
