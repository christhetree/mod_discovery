import itertools
import logging
import os
import warnings

from wavetables import CONTINUOUS_ABLETON_WTS

# Prevents a bug with PyTorch and CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Prevent FADTK from going crazy with CPU usage
os.environ["OMP_NUM_THREADS"] = "4"

import torch as tr

from acid_ddsp.cli import CustomLightningCLI
from acid_ddsp.paths import CONFIGS_DIR, OUT_DIR, WAVETABLES_DIR, MODELS_DIR
from acid_ddsp.synth_modules import WavetableOsc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

tr.set_float32_matmul_precision("high")
# warnings.filterwarnings(
#     "ignore", message="does not have a deterministic", category=UserWarning
# )
warnings.simplefilter("ignore", UserWarning)

os.makedirs("lightning_logs", exist_ok=True)
os.makedirs("wandb_logs", exist_ok=True)


if __name__ == "__main__":
    save_dir = None
    config_basename = None

    # config_name = "synthetic_2/train.yml"

    # config_name = "synthetic_2/train__ase__lfo.yml"
    # config_name = "synthetic_2/train__ase__lfo_frame.yml"
    # config_name = "synthetic_2/train__ase__lfo_frame_8_hz.yml"
    # config_name = "synthetic_2/test_vital__ase__lfo.yml"
    # config_name = "synthetic_2/test_vital__ase__lfo_frame.yml"
    # config_name = "synthetic_2/test_vital__ase__lfo_frame_8_hz.yml"

    # config_basename = os.path.basename(config_name)[:-4]
    # config_basename = config_basename.replace("test_vital", "train")

    # save_dir = config_name.split("/")[-1][:-4]
    # os.makedirs(save_dir, exist_ok=True)

    config_name = "synthetic_2/train__ase__sm.yml"
    # config_name = "synthetic_2/train__ase__sm_frame.yml"
    # config_name = "synthetic_2/train__ase__sm_frame_8_hz.yml"
    # config_name = "synthetic_2/train__ase__sm_rand.yml"
    # config_name = "synthetic_2/train__ase__sm_oracle.yml"
    # config_name = "synthetic_2/test__ase__sm_rand.yml"

    # config_name = "serum_2/train__ase__sm.yml"
    # config_name = "serum_2/train__ase__sm_frame.yml"
    # config_name = "serum_2/train__ase__sm_frame_8_hz.yml"
    # config_name = "serum_2/train__ase__sm_rand.yml"
    # config_name = "serum_2/train__ase__sm_frame_gran.yml"
    # config_name = "serum_2/train__ase__sm_gran.yml"
    # config_name = "serum_2/train__ase__sm_frame_8_hz_gran.yml"

    # config_name = "serum_2/train__ase__sm_shan.yml"
    # config_name = "serum_2/train__ase__sm_shan_frame.yml"
    # config_name = "serum_2/train__ase__sm_shan_frame_8_hz.yml"
    # config_name = "serum_2/train__ase__sm_shan_rand.yml"
    # config_name = "serum_2/train__ase__sm_shan_frame_gran.yml"
    # config_name = "serum_2/train__ase__sm_shan_gran.yml"

    # config_name = "serum_2/train__ase__sm_ddsp.yml"
    # config_name = "serum_2/train__ase__sm_ddsp_frame.yml"
    # config_name = "serum_2/train__ase__sm_ddsp_frame_8_hz.yml"
    # config_name = "serum_2/train__ase__sm_ddsp_rand.yml"
    # config_name = "serum_2/train__ase__sm_ddsp_frame_gran.yml"
    # config_name = "serum_2/train__ase__sm_ddsp_gran.yml"

    seeds = [42]
    # seeds = list(range(10))
    # seeds = list(range(20))
    log.info(f"Running with seeds: {seeds}")

    wt_dir = os.path.join(WAVETABLES_DIR, "ableton")
    wt_names = [f[:-3] for f in os.listdir(wt_dir) if f.endswith(".pt")]
    filtered_wt_names = []
    for wt_name in wt_names:
        if any(wt_name.startswith(n) for n in CONTINUOUS_ABLETON_WTS):
            filtered_wt_names.append(wt_name)
    wt_paths = [os.path.join(wt_dir, f"{wt_name}.pt") for wt_name in filtered_wt_names]
    wt_paths = sorted(wt_paths)
    for wt_path in wt_paths:
        wt_name = os.path.basename(wt_path)
        log.info(wt_name)
    log.info(f"\nWavetable directory: {wt_dir}\nFound {len(wt_paths)} wavetables")
    # wt_paths = [None]

    config_path = os.path.join(CONFIGS_DIR, config_name)

    # # We extract devices for FADTK
    # with open(config_path, "r") as in_f:
    #     config = yaml.safe_load(in_f)
    # devices = config["trainer"]["devices"]

    for idx, (seed, wt_path) in enumerate(itertools.product(seeds, wt_paths)):
        log.info(f"Current seed: {seed} and wavetable: {wt_path}")

        cli = CustomLightningCLI(
            args=["-c", config_path, "--seed_everything", str(seed)],
            trainer_defaults=CustomLightningCLI.make_trainer_defaults(save_dir=save_dir),
            run=False,
        )
        if wt_path is not None:
            wt = tr.load(wt_path, weights_only=True)
            wt_name = os.path.basename(wt_path)[: -len(".pt")]
            # TODO(cm): make this cleaner
            cli.model.wt_name = wt_name
            synth = cli.model.synth
            assert not synth.add_synth_module.is_trainable
            sr = synth.ac.sr
            wt_module = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
            synth.register_module("add_synth_module", wt_module)
            synth_hat = cli.model.synth_hat
            # wt_module_hat = WavetableOsc(sr=sr, wt=basic_shapes_wt, is_trainable=True)
            # synth_hat.register_module("add_synth_module", wt_module_hat)

            if isinstance(synth_hat.add_synth_module, WavetableOsc) and not synth_hat.add_synth_module.is_trainable:
                wt_module_hat = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
                synth_hat.register_module("add_synth_module", wt_module_hat)

        # use_wandb = idx == 0 and tr.cuda.is_available()
        # use_wandb = True
        use_wandb = None
        cli.before_fit(use_wandb=use_wandb)
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

        # Set CUDA_VISIBLE_DEVICES again for FADTK
        # if isinstance(devices, list):
        #     cuda_flag = f'{",".join([str(d) for d in devices])}'
        #     log.info(f"setting CUDA_VISIBLE_DEVICES = {cuda_flag}")
        #     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_flag}"

        cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
        # cli.trainer.test(model=cli.model, datamodule=cli.datamodule)

        # def get_ckpt_path(idx: int, config_basename: str) -> str:
        #     ckpt_dir = os.path.join(
        #         MODELS_DIR,
        #         config_basename,
        #         "acid_ddsp_2",
        #         f"version_{idx}/checkpoints/",
        #     )
        #     ckpt_files = [
        #         f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")
        #     ]
        #     assert len(ckpt_files) == 1, f"Found {len(ckpt_files)} files in {ckpt_dir}"
        #     ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        #     log.info(f"Checkpoint path: {ckpt_path}")
        #     return ckpt_path
        #
        # ckpt_path = get_ckpt_path(idx, config_basename)
        # cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
