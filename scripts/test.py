import logging
import os

import yaml
from tqdm import tqdm

from cli import CustomLightningCLI
from paths import MODELS_DIR, OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    model_names = [
        "cnn_mss_coeff__abstract_303_48k__6k__4k_min__epoch_127_step_768",
        # "cnn_mss_coeff_fsm_128_32_1__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        # "cnn_mss_coeff_fsm_256_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        # "cnn_mss_coeff_fsm_512_32_1__abstract_303_48k__6k__4k_min__epoch_151_step_912",
        # "cnn_mss_coeff_fsm_1024_32_1__abstract_303_48k__6k__4k_min__epoch_143_step_864",
        # "cnn_mss_coeff_fsm_2048_32_1__abstract_303_48k__6k__4k_min__epoch_87_step_528",
        # "cnn_mss_coeff_fsm_4096_32_1__abstract_303_48k__6k__4k_min__epoch_79_step_480",
        # "cnn_mss_lp__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        # "cnn_mss_lp_fsm_128_32_1__abstract_303_48k__6k__4k_min__epoch_175_step_1056",
        # "cnn_mss_lp_fsm_256_32_1__abstract_303_48k__6k__4k_min__epoch_175_step_1056",
        # "cnn_mss_lp_fsm_512_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        # "cnn_mss_lp_fsm_1024_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        # "cnn_mss_lp_fsm_2048_32_1__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        # "cnn_mss_lp_fsm_4096_32_1__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        # "cnn_mss_lstm64__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
    ]
    fad_model_names = [
        "vggish",
        # "clap-2023",
        # "encodec-emb-48k",
        # "wavlm-base",
        # "clap-laion-audio",
        # "clap-laion-music",
        # "MERT-v1-95M",
        # "dac-44kHz",
        # "cdpam-acoustic",
        # "cdpam-content",
    ]

    model_dir = MODELS_DIR
    for model_name in tqdm(model_names):
        config_path = os.path.join(model_dir, model_name, f"config.yaml")
        ckpt_path = os.path.join(
            model_dir, model_name, f"checkpoints/{model_name}.ckpt"
        )
        with open(config_path, "r") as in_f:
            config = yaml.safe_load(in_f)
        if config.get("ckpt_path"):
            assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)
        # assert config["custom"]["cpu_batch_size"] == 34
        # assert config["model"]["init_args"]["fad_model_names"] is None
        config["custom"]["cpu_batch_size"] = 34
        config["model"]["init_args"]["fad_model_names"] = fad_model_names
        # config["seed_everything"] = 48
        # config["model"]["init_args"]["run_name"] = model_name

        # Save tmp modified config
        tmp_config_path = os.path.join(OUT_DIR, f"config_tmp.yaml")
        with open(tmp_config_path, "w") as out_f:
            yaml.dump(config, out_f)

        cli = CustomLightningCLI(
            args=["test", "--config", tmp_config_path, "--ckpt_path", ckpt_path],
            trainer_defaults=CustomLightningCLI.trainer_defaults,
        )
