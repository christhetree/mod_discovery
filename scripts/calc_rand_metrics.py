import logging
import os
from collections import defaultdict

import torch as tr
from torch import nn
from tqdm import tqdm

import util
from lightning import AcidDDSPLightingModule
from mod_sig_distances import (
    ESRLoss,
    FFTMagDist,
    PCCDistance,
    COSSDistance,
    DTWDistance,
    ChamferDistance,
    FrechetDistance,
    FirstDerivativeDistance,
    SecondDerivativeDistance,
)
from paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    tr.random.manual_seed(42)
    eps = 1e-8

    n_iter = 100
    bs = 32
    n_frames = 1501
    x_n_signals = 1
    x_hat_n_signals = 1
    # x_hat_n_signals = 3
    # x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__bezier_1d.yml")
    x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__bezier_2d.yml")
    x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__model.yml")
    metric_fn_s = {
        "l1": nn.L1Loss(),
        "esr": ESRLoss(eps=eps),
        "mse": nn.MSELoss(),
        "pcc": PCCDistance(),
        # "fft": FFTMagDist(ignore_dc=True),
        # "coss": COSSDistance(),
        # "cd": ChamferDistance(n_frames),
        # "fd": FrechetDistance(n_frames),
        # "dtw": DTWDistance(),
    }
    d1_metric_fn_s = {
        # "cd_d1": FirstDerivativeDistance(ChamferDistance(n_frames - 2)),
        # "fd_d1": FirstDerivativeDistance(FrechetDistance(n_frames - 2)),
    }
    d2_metric_fn_s = {
        # "cd_d2": SecondDerivativeDistance(ChamferDistance(n_frames - 4)),
        # "fd_d2": SecondDerivativeDistance(FrechetDistance(n_frames - 4)),
    }
    for metric_name, metric_fn in metric_fn_s.items():
        if metric_name in ["cd", "fd"]:
            continue
        d1_metric_fn_s[f"{metric_name}_d1"] = FirstDerivativeDistance(metric_fn)
        d2_metric_fn_s[f"{metric_name}_d2"] = SecondDerivativeDistance(metric_fn)
    metric_fn_s.update(d1_metric_fn_s)
    metric_fn_s.update(d2_metric_fn_s)

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
        log.info(
            f"{metric_name:12}: {dist:6.4f}, std: {dist_std:6.4f}, "
            f"min: {dist_min:6.4f}, max: {dist_max:6.4f}, n: {n}"
        )


# INFO:__main__:n_iter: 100, bs: 32, n_frames: 1501,
# INFO:__main__:x_n_signals: 1, x_hat_n_signals: 1
# INFO:__main__:l1       : 0.2345, std: 0.0077, min: 0.2102, max: 0.2521, n: 100
# INFO:__main__:esr      : 0.4593, std: 0.4184, min: 0.2354, max: 4.0390, n: 100
# INFO:__main__:mse      : 0.0840, std: 0.0049, min: 0.0680, max: 0.0958, n: 100
# INFO:__main__:fft      : 1.3913, std: 0.0485, min: 1.2520, max: 1.5352, n: 100
# INFO:__main__:pcc      : 0.0006, std: 0.0366, min: -0.0575, max: 0.1172, n: 100
# INFO:__main__:coss     : 0.8738, std: 0.0485, min: 0.5174, max: 0.9641, n: 100
# INFO:__main__:dtw      : 0.0707, std: 0.0045, min: 0.0611, max: 0.0827, n: 100
# INFO:__main__:l1_inv_1 : 0.1385, std: 0.0078, min: 0.1193, max: 0.1541, n: 100
# INFO:__main__:esr_inv_1: 0.1167, std: 0.0138, min: 0.0809, max: 0.1513, n: 100
# INFO:__main__:mse_inv_1: 0.0306, std: 0.0027, min: 0.0238, max: 0.0363, n: 100
# INFO:__main__:fft_inv_1: 0.8544, std: 0.0677, min: 0.7014, max: 1.0669, n: 100
# INFO:__main__:pcc_inv_1: 0.1534, std: 0.0204, min: 0.0974, max: 0.2000, n: 100
# INFO:__main__:coss_inv_1: 0.9389, std: 0.0414, min: 0.6351, max: 1.0000, n: 100
# INFO:__main__:dtw_inv_1: 0.0584, std: 0.0037, min: 0.0501, max: 0.0668, n: 100
