import logging
import os
from collections import defaultdict

import pandas as pd
import torch as tr
from torch import nn
from tqdm import tqdm

import util
from lightning import ModDiscoveryLightingModule
from mod_sig_distances import (
    ESRLoss,
    FFTMagDist,
    PCCDistance,
    DTWDistance,
    ChamferDistance,
    FrechetDistance,
    FirstDerivativeDistance,
    SecondDerivativeDistance,
)
from mod_sig_metrics import (
    LFORangeMetric,
    EntropyMetric,
    TotalVariationMetric,
    SpectralEntropyMetric,
    TurningPointsMetric,
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

    # x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__bezier_1d.yml")
    x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__vital_curves.yml")

    # x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__model.yml")
    x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__rand_uniform_frame.yml")

    dist_fn_s = {
        "l1": nn.L1Loss(),
        "l1_d1": FirstDerivativeDistance(nn.L1Loss()),
        "pcc": PCCDistance(),
        "fd": FrechetDistance(n_frames),
    }
    metric_fn_s = {
        "range_mean": LFORangeMetric(agg_fn="mean"),
        "min_val": LFORangeMetric(agg_fn="min_val"),
        "max_val": LFORangeMetric(agg_fn="max_val"),
        "spec_ent": SpectralEntropyMetric(eps=eps, normalize=True),
        "tv": TotalVariationMetric(eps=eps, normalize=True),
        "tp": TurningPointsMetric(),
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
            x_hat_s.append(x_hat)
        x_hat = tr.stack(x_hat_s, dim=0).view(bs, x_hat_n_signals, n_frames)

        if x_n_signals == x_hat_n_signals == 1:
            for dist_name, dist_fn in dist_fn_s.items():
                dist = dist_fn(x_hat.squeeze(1), x.squeeze(1))
                results[dist_name].append(dist)
            for metric_name, metric_fn in metric_fn_s.items():
                metric = metric_fn(x.squeeze(1))
                results[f"lfo__{metric_name}"].append(metric)
                metric = metric_fn(x_hat.squeeze(1))
                results[f"lfo_hat__{metric_name}"].append(metric)

        x_inv = util.compute_lstsq_with_bias(x_hat, x)
        for idx in range(x_n_signals):
            curr_x = x[:, idx, :]
            curr_x_inv = x_inv[:, idx, :]
            for dist_name, dist_fn in dist_fn_s.items():
                dist = dist_fn(curr_x_inv, curr_x)
                results[f"{dist_name}_inv_{x_hat_n_signals}"].append(dist)

    log.info(f"x_mod_gen: {x_mod_gen_path}")
    log.info(f"x_hat_mod_gen: {x_hat_mod_gen_path}")
    log.info(f"n_iter: {n_iter}, bs: {bs}, n_frames: {n_frames}, ")
    log.info(f"x_n_signals: {x_n_signals}, x_hat_n_signals: {x_hat_n_signals}")
    df_cols = ["name", "mean", "ci95", "std", "min", "max", "n"]
    df_rows = [df_cols]
    for dist_name, dists in results.items():
        dists = tr.stack(dists, dim=0)
        n = dists.size(0)
        dist = dists.mean().item()
        dist_std = dists.std().item()
        ci95 = 1.96 * dist_std / (n**0.5)
        dist_min = dists.min().item()
        dist_max = dists.max().item()
        df_rows.append([dist_name, dist, ci95, dist_std, dist_min, dist_max, n])

    df = pd.DataFrame.from_records(df_rows)
    print(df.to_string())


# x_mod_gen: mod_sig_gen__bezier_1d.yml
# x_hat_mod_gen: mod_sig_gen__model.yml
# n_iter: 100, bs: 32, n_frames: 1501,
#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1                    l1  0.234476  0.001512  0.007714  0.210253  0.252167  100
# 2                 l1_d1  0.008714  0.000026  0.000134  0.008405  0.009002  100
# 3                   pcc    0.0005  0.007182  0.036642 -0.058259  0.117849  100
# 4                    fd  0.564713  0.005613  0.028638  0.507373  0.642912  100
# 5       lfo__range_mean  0.662286  0.007481  0.038166  0.542321  0.769346  100
# 6   lfo_hat__range_mean  0.910702  0.001785  0.009108  0.888036  0.932982  100
# 7          lfo__min_val  0.005598  0.001083  0.005527  0.000099  0.031328  100
# 8      lfo_hat__min_val  0.004444   0.00057  0.002907  0.000105  0.015118  100
# 9          lfo__max_val  0.994186  0.001129  0.005759  0.970512  0.999814  100
# 10     lfo_hat__max_val  0.995508  0.000508  0.002591  0.989073  0.999595  100
# 11        lfo__spec_ent  0.497553  0.004276  0.021817  0.444433  0.561752  100
# 12    lfo_hat__spec_ent  0.560893  0.001562   0.00797  0.538667   0.58087  100
# 13              lfo__tv  0.001875  0.000034  0.000171  0.001431  0.002278  100
# 14          lfo_hat__tv  0.009471  0.000029  0.000147  0.009014  0.009871  100
# 15              lfo__tp  0.003239  0.000098  0.000501  0.001793  0.004586  100
# 16          lfo_hat__tp  0.026874   0.00007  0.000357  0.025976  0.027831  100
# 17             l1_inv_1  0.138495  0.001536  0.007838   0.11929  0.154117  100
# 18          l1_d1_inv_1  0.001908  0.000039  0.000198  0.001328    0.0023  100
# 19            pcc_inv_1  0.153447  0.004002  0.020418  0.097529  0.200308  100
# 20             fd_inv_1  0.364707  0.004458  0.022746  0.291902  0.428538  100
#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1     l1_inv_3  0.133125  0.001602  0.008175  0.112516  0.152996  100
# 2  l1_d1_inv_3  0.002826  0.000044  0.000223  0.002289  0.003438  100
# 3    pcc_inv_3  0.303219  0.003942  0.020112  0.255647  0.342783  100
# 4     fd_inv_3  0.347224  0.004619  0.023565  0.300457   0.40381  100


# x_mod_gen: mod_sig_gen__bezier_1d.yml
# x_hat_mod_gen: mod_sig_gen__rand_uniform_frame.yml
# n_iter: 100, bs: 32, n_frames: 1501,
# x_n_signals: 1, x_hat_n_signals: 3
#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1                    l1  0.296173  0.000909  0.004638  0.284608  0.308059  100
# 2                 l1_d1  0.166666  0.000123  0.000626  0.165574  0.168495  100
# 3                   pcc  0.000231  0.000826  0.004214 -0.011512  0.009016  100
# 4                    fd  0.645772   0.00448  0.022858  0.589951  0.694588  100
# 5       lfo__range_mean  0.670134  0.007501  0.038272  0.574115  0.742187  100
# 6   lfo_hat__range_mean  0.998672   0.00003  0.000154  0.998237  0.999004  100
# 7          lfo__min_val  0.006419  0.001173  0.005984  0.000057  0.036537  100
# 8      lfo_hat__min_val   0.00002  0.000004   0.00002       0.0  0.000114  100
# 9          lfo__max_val  0.993437  0.001113  0.005678  0.972582  0.999804  100
# 10     lfo_hat__max_val  0.999975  0.000004  0.000023  0.999861  0.999999  100
# 11        lfo__spec_ent  0.497351  0.004509  0.023003  0.442918  0.561752  100
# 12    lfo_hat__spec_ent  0.935434  0.000051  0.000259  0.934936  0.936256  100
# 13              lfo__tv  0.001891  0.000034  0.000171  0.001509  0.002384  100
# 14          lfo_hat__tv  0.333613  0.000253  0.001291  0.330349  0.336774  100
# 15              lfo__tp  0.003286  0.000091  0.000466  0.002043  0.004295  100
# 16          lfo_hat__tp  0.666788   0.00036  0.001838  0.662442  0.670489  100
# 17             l1_inv_1  0.142944  0.001872  0.009553  0.116287  0.165195  100
# 18          l1_d1_inv_1  0.002656  0.000058  0.000294  0.001763  0.003328  100
# 19            pcc_inv_1  0.020734  0.000561  0.002864  0.013795  0.026641  100
# 20             fd_inv_1   0.38843  0.004375  0.022323  0.335482  0.440834  100
#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1     l1_inv_3  0.142715  0.001699  0.008667  0.122135   0.16407  100
# 2  l1_d1_inv_3  0.004369  0.000083  0.000425  0.003413  0.005454  100
# 3    pcc_inv_3  0.041552  0.000624  0.003183  0.033662  0.049282  100
# 4     fd_inv_3  0.381848  0.003847  0.019626  0.337635  0.428832  100


# x_mod_gen: mod_sig_gen__vital_curves.yml
# x_hat_mod_gen: mod_sig_gen__model.yml
# n_iter: 100, bs: 32, n_frames: 1501,
# x_n_signals: 1, x_hat_n_signals: 3
#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1                    l1  0.335006  0.001932  0.009855  0.313744  0.367837  100
# 2                 l1_d1  0.008958  0.000037  0.000191   0.00857  0.009527  100
# 3                   pcc  0.000402   0.00667  0.034033 -0.098051  0.073261  100
# 4                    fd  0.710313   0.00532  0.027144  0.645413   0.77754  100
# 5       lfo__range_mean  0.905662  0.005213  0.026595   0.83306  0.963623  100
# 6   lfo_hat__range_mean  0.909797  0.001666  0.008501  0.891698  0.931125  100
# 7          lfo__min_val       0.0       0.0       0.0       0.0       0.0  100
# 8      lfo_hat__min_val  0.005197  0.000533  0.002717  0.000394  0.012243  100
# 9          lfo__max_val       1.0       0.0       0.0       1.0       1.0  100
# 10     lfo_hat__max_val  0.995247  0.000525  0.002679  0.986449  0.998949  100
# 11        lfo__spec_ent  0.463675  0.006448  0.032897  0.375649  0.527341  100
# 12    lfo_hat__spec_ent  0.559849  0.001378  0.007032  0.544765  0.577993  100
# 13              lfo__tv  0.001689  0.000055  0.000282   0.00109   0.00231  100
# 14          lfo_hat__tv  0.009503  0.000025  0.000125  0.009104  0.009795  100
# 15              lfo__tp  0.001203  0.000064  0.000326  0.000459  0.001897  100
# 16          lfo_hat__tp  0.026876  0.000084  0.000428  0.025871  0.028123  100
# 17             l1_inv_1  0.213393  0.002209  0.011272  0.186543  0.243431  100
# 18          l1_d1_inv_1  0.002621  0.000059    0.0003  0.001945  0.003489  100
# 19            pcc_inv_1  0.148891  0.004176  0.021308  0.102264  0.211344  100
# 20             fd_inv_1  0.577221  0.005825  0.029718  0.508701  0.645103  100
#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1     l1_inv_3  0.206655  0.002127  0.010854  0.182921  0.234049  100
# 2  l1_d1_inv_3  0.004294  0.000069   0.00035  0.003533  0.005452  100
# 3    pcc_inv_3  0.300896  0.003834  0.019563  0.255516  0.347917  100
# 4     fd_inv_3  0.563989  0.006811  0.034749  0.469984  0.642974  100


# x_mod_gen: mod_sig_gen__vital_curves.yml
# x_hat_mod_gen: mod_sig_gen__rand_uniform_frame.yml
# n_iter: 100, bs: 32, n_frames: 1501,
# x_n_signals: 1, x_hat_n_signals: 3
#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1                    l1  0.369138  0.001065  0.005433  0.354514  0.384839  100
# 2                 l1_d1  0.166795  0.000131  0.000669  0.164749   0.16853  100
# 3                   pcc  0.000194  0.000792  0.004038 -0.010752  0.011774  100
# 4                    fd  0.776771  0.005031   0.02567  0.697766   0.83272  100
# 5       lfo__range_mean  0.909942  0.005201  0.026537  0.841509  0.961764  100
# 6   lfo_hat__range_mean  0.998657  0.000031   0.00016  0.998206  0.998996  100
# 7          lfo__min_val       0.0       0.0       0.0       0.0       0.0  100
# 8      lfo_hat__min_val  0.000017  0.000003  0.000016       0.0  0.000078  100
# 9          lfo__max_val       1.0       0.0       0.0       1.0       1.0  100
# 10     lfo_hat__max_val  0.999975  0.000005  0.000026  0.999861  0.999999  100
# 11        lfo__spec_ent  0.468449  0.006489  0.033109  0.396999  0.554188  100
# 12    lfo_hat__spec_ent  0.935404  0.000052  0.000267  0.934797  0.936131  100
# 13              lfo__tv  0.001692  0.000054  0.000277  0.001156  0.002401  100
# 14          lfo_hat__tv   0.33361  0.000254  0.001298  0.329906  0.336125  100
# 15              lfo__tp   0.00121  0.000058  0.000298  0.000584  0.002022  100
# 16          lfo_hat__tp  0.666794  0.000362  0.001845  0.661045  0.670051  100
# 17             l1_inv_1  0.218222   0.00258  0.013166  0.187024  0.248792  100
# 18          l1_d1_inv_1  0.003886  0.000092  0.000468  0.002892  0.005528  100
# 19            pcc_inv_1  0.021133   0.00058  0.002959  0.014227  0.030449  100
# 20             fd_inv_1  0.603342  0.004953  0.025273  0.513624  0.656871  100
#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1     l1_inv_3   0.22055  0.002296  0.011714  0.185439  0.248467  100
# 2  l1_d1_inv_3    0.0066   0.00011  0.000563  0.005331  0.007732  100
# 3    pcc_inv_3   0.04084  0.000627  0.003197  0.033469  0.048331  100
# 4     fd_inv_3  0.594808   0.00522   0.02663  0.514638  0.660509  100
