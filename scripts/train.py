import itertools
import logging
import os
import warnings

# Prevents a bug with PyTorch and CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Prevent FADTK from going crazy with CPU usage
os.environ["OMP_NUM_THREADS"] = "4"

import torch as tr

from acid_ddsp.cli import CustomLightningCLI
from acid_ddsp.paths import CONFIGS_DIR, WAVETABLES_DIR
from acid_ddsp.synth_modules import WavetableOsc
from acid_ddsp.wavetables import BAD_ABLETON_WTS

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
    # config_name = "synthetic_2/train.yml"

    # config_name = "synthetic_2/train__ase__lfo.yml"

    # config_name = "synthetic_2/train__ase__sm.yml"
    # config_name = "synthetic_2/train__ase__sm_frame_32.yml"
    # config_name = "synthetic_2/train__ase__sm_frame_cf_4.yml"
    # config_name = "synthetic_2/train__ase__sm_frame_32_interp_36.yml"

    # config_name = "synthetic_2/train__a__sm.yml"
    # config_name = "synthetic_2/train__s__sm.yml"

    config_name = "serum_2/train__ase__sm.yml"
    # config_name = "serum_2/train__ase__sm_frame.yml"
    # config_name = "serum_2/train__ase__sm_frame_cf.yml"

    seeds = [42]
    # seeds = [42, 42, 3, 42]
    # seeds = list(range(10))
    log.info(f"Running with seeds: {seeds}")

    # wt_dir = os.path.join(WAVETABLES_DIR, "testing")
    # wt_dir = os.path.join(WAVETABLES_DIR, "ableton_basic_shapes")
    # wt_dir = os.path.join(WAVETABLES_DIR, "ableton")
    # wt_dir = os.path.join(WAVETABLES_DIR, "waveedit")

    # wt_names = [f[:-3] for f in os.listdir(wt_dir) if f.endswith(".pt")]
    # filtered_wt_names = []
    # for wt_name in wt_names:
    #     if any(bad_wt_name in wt_name for bad_wt_name in BAD_ABLETON_WTS):
    #         continue
    #     if not wt_name.startswith("basics__"):
    #         continue
    #     # if not "galactica" in wt_name:
    #     if not "fm_fold" in wt_name:
    #         # if not "basic_shapes" in wt_name:
    #         continue
    #     filtered_wt_names.append(wt_name)
    # wt_paths = [os.path.join(wt_dir, f"{wt_name}.pt") for wt_name in filtered_wt_names]
    # wt_paths = sorted(wt_paths)
    # log.info(f"\nWavetable directory: {wt_dir}\nFound {len(wt_paths)} wavetables")
    wt_paths = [None]

    # basic_shapes_wt_path = os.path.join(WAVETABLES_DIR, "ableton_basic_shapes", "basics__basic_shapes__4_1024.pt")
    # basic_shapes_wt = tr.load(basic_shapes_wt_path, weights_only=True)

    config_path = os.path.join(CONFIGS_DIR, config_name)

    # # We extract devices for FADTK
    # with open(config_path, "r") as in_f:
    #     config = yaml.safe_load(in_f)
    # devices = config["trainer"]["devices"]

    for idx, (seed, wt_path) in enumerate(itertools.product(seeds, wt_paths)):
        log.info(f"Current seed: {seed} and wavetable: {wt_path}")

        cli = CustomLightningCLI(
            args=["fit", "-c", config_path, "--seed_everything", str(seed)],
            trainer_defaults=CustomLightningCLI.make_trainer_defaults(),
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

            if not synth_hat.add_synth_module.is_trainable:
                wt_module_hat = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
                synth_hat.register_module("add_synth_module", wt_module_hat)

        # use_wandb = idx == 0 and tr.cuda.is_available()
        use_wandb = None
        cli.before_fit(use_wandb=use_wandb)
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

        # Set CUDA_VISIBLE_DEVICES again for FADTK
        # if isinstance(devices, list):
        #     cuda_flag = f'{",".join([str(d) for d in devices])}'
        #     log.info(f"setting CUDA_VISIBLE_DEVICES = {cuda_flag}")
        #     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_flag}"

        cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
