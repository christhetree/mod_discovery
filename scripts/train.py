import logging
import os
import warnings

# Prevents a bug with PyTorch and CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from acid_ddsp.cli import CustomLightningCLI
from acid_ddsp.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

torch.set_float32_matmul_precision("high")
# warnings.filterwarnings(
#     "ignore", message="does not have a deterministic", category=UserWarning
# )
warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    # config_name = "nsynth/train.yml"
    # config_name = "serum/train.yml"
    config_name = "synthetic_2/train.yml"

    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(
        args=["fit", "-c", config_path],
        trainer_defaults=CustomLightningCLI.trainer_defaults,
    )
