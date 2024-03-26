import logging
import os

import yaml

from cli import CustomLightningCLI
from paths import MODELS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    model_dir = MODELS_DIR
    # model_dir = OUT_DIR

    model_name = "cnn_mss_coeff__abstract_303_48k__6k__4k_min__epoch_127_step_768"
    # model_name = "cnn_mss_coeff_fsm_128_32_1__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_coeff_fsm_256_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200"
    # model_name = "cnn_mss_coeff_fsm_512_32_1__abstract_303_48k__6k__4k_min__epoch_151_step_912"
    # model_name = "cnn_mss_coeff_fsm_1024_32_1__abstract_303_48k__6k__4k_min__epoch_143_step_864"
    # model_name = "cnn_mss_coeff_fsm_2048_32_1__abstract_303_48k__6k__4k_min__epoch_87_step_528"
    # model_name = "cnn_mss_coeff_fsm_4096_32_1__abstract_303_48k__6k__4k_min__epoch_79_step_480"

    # model_name = "cnn_mss_lp__abstract_303_48k__6k__4k_min__epoch_199_step_1200"
    # model_name = "cnn_mss_lp_fsm_128_32_1__abstract_303_48k__6k__4k_min__epoch_175_step_1056"
    # model_name = "cnn_mss_lp_fsm_256_32_1__abstract_303_48k__6k__4k_min__epoch_175_step_1056"
    # model_name = "cnn_mss_lp_fsm_512_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200"
    # model_name = "cnn_mss_lp_fsm_1024_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200"
    # model_name = "cnn_mss_lp_fsm_2048_32_1__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fsm_4096_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200"

    # model_name = "cnn_mss_lstm64__abstract_303_48k__6k__4k_min__epoch_199_step_1200"

    config_path = os.path.join(model_dir, model_name, f"config.yaml")
    ckpt_path = os.path.join(model_dir, model_name, f"checkpoints/{model_name}.ckpt")
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    if config.get("ckpt_path"):
        assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)

    cli = CustomLightningCLI(
        args=["test", "--config", config_path, "--ckpt_path", ckpt_path],
        trainer_defaults=CustomLightningCLI.trainer_defaults,
    )
