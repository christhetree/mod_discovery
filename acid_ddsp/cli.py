import logging
import os
from contextlib import suppress
from typing import Optional, Dict, Any

import torch as tr
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.loggers import WandbLogger

from callbacks import LogModSigAndSpecCallback, LogAudioCallback, LogWavetablesCallback
from paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class CustomLightningCLI(LightningCLI):
    def __init__(self, cli_config_path: Optional[str] = None, *args, **kwargs) -> None:
        if cli_config_path is None:
            cli_config_path = os.path.join(CONFIGS_DIR, "cli_config.yml")
        assert os.path.isfile(cli_config_path)
        with open(cli_config_path, "r") as in_f:
            self.cli_config = yaml.safe_load(in_f)
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for add_arg in self.cli_config["additional_arguments"]:
            name = add_arg["name"]
            if not name.startswith("--"):
                name = f"--{name}"
            default = add_arg.get("default")
            if default is not None:
                parser.add_argument(name, default=default)
            else:
                parser.add_argument(name)

        for link_args in self.cli_config["link_arguments"]:
            parser.link_arguments(link_args["src"], link_args["dest"])

    # TODO(cm): refactor
    def link_arguments_if_possible(
        self, src: str, dest: str, config: Dict[str, Any], is_strict: bool = False
    ) -> None:
        src_tokens = src.split(".")
        dest_tokens = dest.split(".")
        assert len(dest_tokens) > 1
        dest_key = dest_tokens[-1]
        dest_tokens = dest_tokens[:-1]
        src_val = config
        for src_token in src_tokens:
            if is_strict:
                assert src_token in src_val, f"Missing src of linked arguments: {src}"
            elif src_token not in src_val:
                log.debug(f"Unable to link src: {src} and dest: {dest}; src not found")
                return
            src_val = src_val[src_token]

        curr_dest = config
        for dest_token in dest_tokens:
            if dest_token not in curr_dest:
                log.debug(f"Unable to link src: {src} and dest: {dest}; dest not found")
                return
            curr_dest = curr_dest[dest_token]
            if curr_dest is None:
                break

        if curr_dest is None:
            log.info(f"Dest {dest} is not reachable")
            return
        if dest_key in curr_dest and curr_dest[dest_key] != src_val:
            log.info(
                f"Dest {dest} already exists: {curr_dest[dest_key]}, overriding it with {src_val}"
            )
        else:
            curr_dest[dest_key] = src_val

    def update_config(self, config: Dict[str, Any]) -> None:
        if "link_arguments_if_possible" in self.cli_config:
            for link_args in self.cli_config["link_arguments_if_possible"]:
                src = link_args["src"]
                dest = link_args["dest"]
                self.link_arguments_if_possible(src, dest, config)

    def before_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            config = self.config[self.subcommand]
            self.update_config(config)
        else:
            config = self.config
            self.update_config(config)

        devices = config.trainer.devices
        if isinstance(devices, list):
            cuda_flag = f'{",".join([str(d) for d in devices])}'
            log.info(f"setting CUDA_VISIBLE_DEVICES = {cuda_flag}")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_flag}"
            config.trainer.devices = len(devices)

        if config.trainer.devices < 2:
            log.debug("Disabling strategy")
            config.trainer.strategy = "auto"

        if config.custom.is_deterministic:
            log.info("Setting torch.use_deterministic_algorithms(True)")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            tr.use_deterministic_algorithms(True, warn_only=True)
            # tr.backends.cudnn.deterministic = True

        if not tr.cuda.is_available():
            config.trainer.accelerator = "auto"
            config.trainer.devices = "auto"
            config.trainer.strategy = "auto"
            config.data.init_args.batch_size = config.custom.cpu_batch_size
            config.data.init_args.num_workers = 0

            if "cpu_link_arguments_if_possible" in self.cli_config:
                for link_args in self.cli_config["cpu_link_arguments_if_possible"]:
                    src = link_args["src"]
                    dest = link_args["dest"]
                    self.link_arguments_if_possible(src, dest, config)

    def before_fit(self, use_wandb: Optional[bool] = None) -> None:
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.filename = (
                    f"{self.config.custom.model_name}__"
                    f"{self.config.custom.dataset_name}__{cb.filename}"
                )
                log.info(f"Setting checkpoint name to: {cb.filename}")

        use_gpu = tr.cuda.is_available()
        if (
            use_wandb
            or (use_gpu and self.config.custom.use_wandb_gpu)
            or (not use_gpu and self.config.custom.use_wandb_cpu)
        ):
            # Used directly by the callbacks
            wandb.init(
                dir="wandb_logs",
                project=self.config.custom.project_name,
                name=f"{self.config.custom.model_name}__"
                f"{self.config.custom.dataset_name}",
            )
            wandb.define_metric("*", step_metric="global_step")

            # Used by lightning to log to wandb
            wandb_logger = WandbLogger(
                save_dir="wandb_logs",
                project=self.config.custom.project_name,
                name=f"{self.config.custom.model_name}__"
                f"{self.config.custom.dataset_name}",
            )
            self.trainer.loggers.append(wandb_logger)
        else:
            log.info("wandb is disabled")

        if not use_gpu:
            log.info("================ Running on CPU ================ ")

        log.info(
            f"================ {self.config.custom.project_name} "
            f"{self.config.custom.model_name} "
            f"{self.config.custom.dataset_name} ================"
        )
        with suppress(Exception):
            log.info(
                f"================ {self.config.optimizer.class_path} "
                f"starting LR = {self.config.optimizer.init_args.lr:.6f} "
                f"================ "
            )

    @staticmethod
    def make_trainer_defaults(save_dir: Optional[str] = None) -> Dict[str, Any]:
        if save_dir is None:
            save_dir = "lightning_logs"

        trainer_defaults = {
            "accelerator": "gpu",
            "callbacks": [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    filename="epoch_{epoch}_step_{step}",  # Name is prepended
                    auto_insert_metric_name=False,
                    monitor="val/loss",
                    mode="min",
                    save_last=False,
                    save_top_k=1,
                    verbose=False,
                ),
                # LogModSigAndSpecCallback(),
                # LogAudioCallback(),
                # LogWavetablesCallback(),
            ],
            "logger": {
                "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": save_dir,
                    "name": None,
                },
            },
            "log_every_n_steps": 1,
            "precision": 32,
        }
        return trainer_defaults
