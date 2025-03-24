import logging
import os
from collections import defaultdict

import torch as tr
from torch import nn
from tqdm import tqdm

import util
from lightning import AcidDDSPLightingModule
from losses import ESRLoss
from paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    tr.random.manual_seed(42)

    n_iter = 100
    bs = 32
    n_frames = 1501
    x_n_signals = 1
    x_hat_n_signals = 1
    # x_hat_n_signals = 3
    x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__bezier.yml")
    # x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__bezier_norm.yml")
    x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__model.yml")
    # x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__model_norm.yml")
    metric_fn_s = {
        "l1": nn.L1Loss(),
        "esr": ESRLoss(),
        "mse": nn.MSELoss(),
        "fft": AcidDDSPLightingModule.calc_fft_mag_dist,
    }

    x_mod_gen = util.load_class_from_yaml(x_mod_gen_path)
    x_hat_mod_gen = util.load_class_from_yaml(x_hat_mod_gen_path)
    results = defaultdict(list)

    for _ in tqdm(range(n_iter)):
        x_s = []
        x_hat_s = []
        for _ in range(bs * x_n_signals):
            x_s.append(x_mod_gen(n_frames))
        x = tr.stack(x_s, dim=0).view(bs, x_n_signals, n_frames)
        for _ in range(bs * x_hat_n_signals):
            x_hat = x_hat_mod_gen(n_frames)
            # x_hat.fill_(0.5)
            x_hat_s.append(x_hat)
        x_hat = tr.stack(x_hat_s, dim=0).view(bs, x_hat_n_signals, n_frames)

        if x_n_signals == x_hat_n_signals == 1:
            for metric_name, metric_fn in metric_fn_s.items():
                dist = metric_fn(x_hat.squeeze(1), x.squeeze(1))
                results[metric_name].append(dist)

        x_pred = AcidDDSPLightingModule.compute_lstsq_with_bias(x_hat, x)
        for idx in range(x_n_signals):
            curr_x = x[:, idx, :]
            curr_x_pred = x_pred[:, idx, :]
            for metric_name, metric_fn in metric_fn_s.items():
                dist = metric_fn(curr_x_pred, curr_x)
                results[f"{metric_name}_inv_{x_hat_n_signals}"].append(dist)

    log.info(f"n_iter: {n_iter}, bs: {bs}, n_frames: {n_frames}, ")
    log.info(f"x_n_signals: {x_n_signals}, x_hat_n_signals: {x_hat_n_signals}")
    for metric_name, dists in results.items():
        dists = tr.stack(dists, dim=0)
        n = dists.size(0)
        dist = dists.mean().item()
        dist_std = dists.std().item()
        dist_min = dists.min().item()
        dist_max = dists.max().item()
        log.info(f"{metric_name:9}: {dist:6.4f}, std: {dist_std:6.4f}, "
                 f"min: {dist_min:6.4f}, max: {dist_max:6.4f}, n: {n}")



# INFO:__main__:n_iter: 100, bs: 32, n_frames: 1501,
# INFO:__main__:x_n_signals: 1, x_hat_n_signals: 1
# INFO:__main__:l1       : 0.2349, std: 0.0097, min: 0.2120, max: 0.2653, n: 100
# INFO:__main__:esr      : 0.4594, std: 0.4802, min: 0.2699, max: 4.9285, n: 100
# INFO:__main__:mse      : 0.0842, std: 0.0063, min: 0.0700, max: 0.1046, n: 100
# INFO:__main__:fft      : 0.9542, std: 0.0474, min: 0.8613, max: 1.0695, n: 100
# INFO:__main__:l1_inv_1 : 0.1334, std: 0.0088, min: 0.1127, max: 0.1531, n: 100
# INFO:__main__:esr_inv_1: 0.1123, std: 0.0141, min: 0.0754, max: 0.1603, n: 100
# INFO:__main__:mse_inv_1: 0.0289, std: 0.0031, min: 0.0223, max: 0.0354, n: 100
# INFO:__main__:fft_inv_1: 0.8513, std: 0.0748, min: 0.6578, max: 1.0506, n: 100

# INFO:__main__:n_iter: 1000, bs: 32, n_frames: 1501,
# INFO:__main__:x_n_signals: 1, x_hat_n_signals: 1
# INFO:__main__:l1       : 0.2354, std: 0.0092, min: 0.2044, max: 0.2674, n: 1000
# INFO:__main__:esr      : 0.6819, std: 3.9487, min: 0.2431, max: 100.3500, n: 1000
# INFO:__main__:mse      : 0.0844, std: 0.0060, min: 0.0651, max: 0.1046, n: 1000
# INFO:__main__:fft      : 0.9553, std: 0.0469, min: 0.8128, max: 1.1438, n: 1000
# INFO:__main__:l1_inv_1 : 0.1329, std: 0.0083, min: 0.1116, max: 0.1563, n: 1000
# INFO:__main__:esr_inv_1: 0.1121, std: 0.0133, min: 0.0727, max: 0.1612, n: 1000
# INFO:__main__:mse_inv_1: 0.0288, std: 0.0030, min: 0.0211, max: 0.0371, n: 1000
# INFO:__main__:fft_inv_1: 0.8500, std: 0.0695, min: 0.6466, max: 1.0772, n: 1000

# INFO:__main__:n_iter: 100, bs: 32, n_frames: 1501,
# INFO:__main__:x_n_signals: 1, x_hat_n_signals: 3
# INFO:__main__:l1_inv_3 : 0.1229, std: 0.0076, min: 0.1038, max: 0.1405, n: 100
# INFO:__main__:esr_inv_3: 0.0983, std: 0.0120, min: 0.0712, max: 0.1278, n: 100
# INFO:__main__:mse_inv_3: 0.0252, std: 0.0026, min: 0.0195, max: 0.0310, n: 100
# INFO:__main__:fft_inv_3: 0.7240, std: 0.0565, min: 0.5932, max: 0.8666, n: 100
