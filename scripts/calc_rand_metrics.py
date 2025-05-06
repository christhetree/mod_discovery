import logging
import os
from collections import defaultdict

import pandas as pd
import torch as tr
from torch import nn
from tqdm import tqdm

import util
from lightning import AcidDDSPLightingModule
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
    # x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__bezier_1d.yml")
    x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__vital_curves.yml")
    # x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__model.yml")
    x_hat_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic_2/mod_sig_gen__rand_uniform_frame.yml")

    dist_fn_s = {
        # "esr": ESRLoss(eps=eps),
        # "l1": nn.L1Loss(),
        # "mse": nn.MSELoss(),
        # "fft": FFTMagDist(ignore_dc=True),
        # "pcc": PCCDistance(),
        # "dtw": DTWDistance(),
        # "cd": ChamferDistance(n_frames),
        # "fd": FrechetDistance(n_frames),
    }
    d1_dist_fn_s = {
        # "cd_d1": FirstDerivativeDistance(ChamferDistance(n_frames - 2)),
        # "fd_d1": FirstDerivativeDistance(FrechetDistance(n_frames - 2)),
    }
    d2_dist_fn_s = {
        # "cd_d2": SecondDerivativeDistance(ChamferDistance(n_frames - 4)),
        # "fd_d2": SecondDerivativeDistance(FrechetDistance(n_frames - 4)),
    }
    metric_fn_s = {
        "range_mean": LFORangeMetric(agg_fn="mean"),
        "min_val": LFORangeMetric(agg_fn="min_val"),
        "max_val": LFORangeMetric(agg_fn="max_val"),
        "ent": EntropyMetric(eps=eps, normalize=True),
        "spec_ent": SpectralEntropyMetric(eps=eps, normalize=True),
        "tv": TotalVariationMetric(eps=eps, normalize=True),
        "tp": TurningPointsMetric(),
    }

    for dist_name, dist_fn in dist_fn_s.items():
        if dist_name in ["dtw", "cd", "fd", "pcc"]:
            continue
        d1_dist_fn_s[f"{dist_name}_d1"] = FirstDerivativeDistance(dist_fn)
        # d2_dist_fn_s[f"{dist_name}_d2"] = SecondDerivativeDistance(dist_fn)
    dist_fn_s.update(d1_dist_fn_s)
    dist_fn_s.update(d2_dist_fn_s)

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

        # x_inv = util.compute_lstsq_with_bias(x_hat, x)
        # for idx in range(x_n_signals):
        #     curr_x = x[:, idx, :]
        #     curr_x_inv = x_inv[:, idx, :]
        #     for dist_name, dist_fn in dist_fn_s.items():
        #         dist = dist_fn(curr_x_inv, curr_x)
        #         results[f"{dist_name}_inv_{x_hat_n_signals}"].append(dist)

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
        # log.info(
        #     f"{dist_name:12}: {dist:6.4f}, std: {dist_std:6.4f}, "
        #     f"min: {dist_min:6.4f}, max: {dist_max:6.4f}, n: {n}"
        # )
        df_rows.append([dist_name, dist, ci95, dist_std, dist_min, dist_max, n])

    df = pd.DataFrame.from_records(df_rows)
    print(df.to_string())

#                       0              1             2            3             4    5
# 0                  name           mean           std          min           max    n
# 1                   esr       0.459046      0.418221     0.235528       4.03744  100
# 2                    l1       0.234476      0.007714     0.210253      0.252167  100
# 3                   mse       0.083897      0.004926     0.068009      0.095731  100
# 4                   fft       1.384295       0.04731     1.246398      1.524272  100
# 5                   pcc         0.0005      0.036642    -0.058259      0.117849  100
# 6                   dtw        0.07067      0.004497     0.061215      0.082738  100
# 7                    cd       0.194168      0.014794     0.156575      0.231637  100
# 8                    fd       0.564713      0.028638     0.507373      0.642912  100
# 9                esr_d1   11915.280273  63206.085938    90.898758  471707.40625  100
# 10                l1_d1       0.008714      0.000134     0.008405      0.009002  100
# 11               mse_d1       0.000143      0.000004     0.000134      0.000152  100
# 12               fft_d1       0.193603      0.004975     0.179324      0.204905  100
# 13               pcc_d1       0.001019      0.010582    -0.023333      0.025518  100
# 14               esr_d2  164597.765625  80905.851562  9714.926758  438981.53125  100
# 15                l1_d2       0.001124      0.000024     0.001069      0.001181  100
# 16               mse_d2       0.000008           0.0     0.000008      0.000009  100
# 17               fft_d2       0.072749      0.002246     0.066947      0.077571  100
# 18               pcc_d2       0.000305      0.005576    -0.018601      0.016429  100
# 19      lfo__range_mean       0.662286      0.038166     0.542321      0.769346  100
# 20  lfo_hat__range_mean       0.910702      0.009108     0.888036      0.932982  100
# 21         lfo__min_val       0.005598      0.005527     0.000099      0.031328  100
# 22     lfo_hat__min_val       0.004444      0.002907     0.000105      0.015118  100
# 23         lfo__max_val       0.994186      0.005759     0.970512      0.999814  100
# 24     lfo_hat__max_val       0.995508      0.002591     0.989073      0.999595  100
# 25             lfo__ent       0.974694      0.002553     0.965276      0.980203  100
# 26         lfo_hat__ent       0.986454      0.000575     0.984393        0.9877  100
# 27        lfo__spec_ent       0.497553      0.021817     0.444433      0.561752  100
# 28    lfo_hat__spec_ent       0.560893       0.00797     0.538667       0.58087  100
# 29              lfo__tv       0.001875      0.000171     0.001431      0.002278  100
# 30          lfo_hat__tv       0.009471      0.000147     0.009014      0.009871  100
# 31              lfo__tp       0.003239      0.000501     0.001793      0.004586  100
# 32          lfo_hat__tp       0.026874      0.000357     0.025976      0.027831  100
# 33            esr_inv_1       0.116712      0.013792     0.080886      0.151314  100
# 34             l1_inv_1       0.138495      0.007838      0.11929      0.154117  100
# 35            mse_inv_1       0.030615      0.002697     0.023804       0.03633  100
# 36            fft_inv_1        0.85502      0.067745     0.702258      1.067568  100
# 37            pcc_inv_1       0.153447      0.020418     0.097529      0.200308  100
# 38            dtw_inv_1       0.058475      0.003705     0.050112      0.066821  100
# 39             cd_inv_1       0.197421      0.010834     0.173198      0.220794  100
# 40             fd_inv_1       0.364707      0.022746     0.291902      0.428538  100
# 41         esr_d1_inv_1       5.647351      1.904902     2.397205     11.719714  100
# 42          l1_d1_inv_1       0.001908      0.000198     0.001328        0.0023  100
# 43         mse_d1_inv_1       0.000008      0.000002     0.000004      0.000014  100
# 44         fft_d1_inv_1       0.027063      0.003557     0.020085      0.034818  100
# 45         pcc_d1_inv_1       0.017894      0.009662    -0.004709      0.046387  100
# 46         esr_d2_inv_1     2204.59668   1929.934326     48.54895   9850.530273  100
# 47          l1_d2_inv_1       0.000168      0.000025     0.000116       0.00023  100
# 48         mse_d2_inv_1            0.0           0.0          0.0      0.000001  100
# 49         fft_d2_inv_1       0.008879      0.001417     0.006129       0.01228  100
# 50         pcc_d2_inv_1       0.000174       0.00621    -0.025318      0.015489  100

#                       0              1             2            3             4    5
# 0                  name           mean           std          min           max    n
# 1             esr_inv_3       0.110096       0.01416     0.079498      0.147648  100
# 2              l1_inv_3       0.133125      0.008175     0.112516      0.152996  100
# 3             mse_inv_3       0.028705      0.002893     0.020737       0.03536  100
# 4             fft_inv_3       0.788416       0.05915     0.665683      0.990864  100
# 5             pcc_inv_3       0.303219      0.020112     0.255647      0.342783  100
# 6             dtw_inv_3       0.046541      0.003629     0.039073      0.055115  100
# 7              cd_inv_3       0.163597       0.01081     0.140283      0.189269  100
# 8              fd_inv_3       0.347224      0.023565     0.300457       0.40381  100
# 9          esr_d1_inv_3       15.09454      4.818378     7.493365     26.752983  100
# 10          l1_d1_inv_3       0.002826      0.000223     0.002289      0.003438  100
# 11         mse_d1_inv_3       0.000016      0.000002     0.000011      0.000021  100
# 12         fft_d1_inv_3        0.04875      0.004324     0.039445      0.059677  100
# 13         pcc_d1_inv_3       0.038402      0.011173     0.010439      0.081244  100
# 14         esr_d2_inv_3    8649.664062   6600.187012   479.087463  30897.933594  100
# 15          l1_d2_inv_3       0.000376      0.000033     0.000292      0.000459  100
# 16         mse_d2_inv_3       0.000001           0.0     0.000001      0.000001  100
# 17         fft_d2_inv_3       0.017821      0.001742     0.014164       0.02191  100
# 18         pcc_d2_inv_3       0.000692      0.006115    -0.010805      0.019657  100




#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1           l1  0.234476  0.001512  0.007714  0.210253  0.252167  100
# 2          pcc    0.0005  0.007182  0.036642 -0.058259  0.117849  100
# 3           fd  0.564713  0.005613  0.028638  0.507373  0.642912  100
# 4        l1_d1  0.008714  0.000026  0.000134  0.008405  0.009002  100
# 5     l1_inv_1  0.138495  0.001536  0.007838   0.11929  0.154117  100
# 6    pcc_inv_1  0.153447  0.004002  0.020418  0.097529  0.200308  100
# 7     fd_inv_1  0.364707  0.004458  0.022746  0.291902  0.428538  100
# 8  l1_d1_inv_1  0.001908  0.000039  0.000198  0.001328    0.0023  100

#              0         1         2         3         4         5    6
# 0         name      mean      ci95       std       min       max    n
# 1     l1_inv_3  0.133125  0.001602  0.008175  0.112516  0.152996  100
# 2    pcc_inv_3  0.303219  0.003942  0.020112  0.255647  0.342783  100
# 3     fd_inv_3  0.347224  0.004619  0.023565  0.300457   0.40381  100
# 4  l1_d1_inv_3  0.002826  0.000044  0.000223  0.002289  0.003438  100




# Vital curves, random spline model
#        0         1         2         3         4         5    6
# 0   name      mean      ci95       std       min       max    n
# 1     l1  0.335006  0.001932  0.009855  0.313744  0.367837  100
# 2    pcc  0.000402   0.00667  0.034033 -0.098051  0.073261  100
# 3     fd  0.710313   0.00532  0.027144  0.645413   0.77754  100
# 4  l1_d1  0.008958  0.000037  0.000191   0.00857  0.009527  100

#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1       lfo__range_mean  0.905662  0.005213  0.026595   0.83306  0.963623  100
# 2   lfo_hat__range_mean  0.909797  0.001666  0.008501  0.891698  0.931125  100
# 3          lfo__min_val       0.0       0.0       0.0       0.0       0.0  100
# 4      lfo_hat__min_val  0.005197  0.000533  0.002717  0.000394  0.012243  100
# 5          lfo__max_val       1.0       0.0       0.0       1.0       1.0  100
# 6      lfo_hat__max_val  0.995247  0.000525  0.002679  0.986449  0.998949  100
# 7              lfo__ent  0.952599  0.001054  0.005377  0.940962  0.965042  100
# 8          lfo_hat__ent  0.986448  0.000109  0.000557  0.985066  0.988123  100
# 9         lfo__spec_ent  0.463675  0.006448  0.032897  0.375649  0.527341  100
# 10    lfo_hat__spec_ent  0.559849  0.001378  0.007032  0.544765  0.577993  100
# 11              lfo__tv  0.001689  0.000055  0.000282   0.00109   0.00231  100
# 12          lfo_hat__tv  0.009503  0.000025  0.000125  0.009104  0.009795  100
# 13              lfo__tp  0.001203  0.000064  0.000326  0.000459  0.001897  100
# 14          lfo_hat__tp  0.026876  0.000084  0.000428  0.025871  0.028123  100

# Vital curves, random uniform frame
#        0         1         2         3         4         5    6
# 0   name      mean      ci95       std       min       max    n
# 1     l1  0.369138  0.001065  0.005433  0.354514  0.384839  100
# 2    pcc  0.000194  0.000792  0.004038 -0.010752  0.011774  100
# 3     fd  0.776771  0.005031   0.02567  0.697766   0.83272  100
# 4  l1_d1  0.166795  0.000131  0.000669  0.164749   0.16853  100

#                       0         1         2         3         4         5    6
# 0                  name      mean      ci95       std       min       max    n
# 1       lfo__range_mean  0.909942  0.005201  0.026537  0.841509  0.961764  100
# 2   lfo_hat__range_mean  0.998657  0.000031   0.00016  0.998206  0.998996  100
# 3          lfo__min_val       0.0       0.0       0.0       0.0       0.0  100
# 4      lfo_hat__min_val  0.000017  0.000003  0.000016       0.0  0.000078  100
# 5          lfo__max_val       1.0       0.0       0.0       1.0       1.0  100
# 6      lfo_hat__max_val  0.999975  0.000005  0.000026  0.999861  0.999999  100
# 7              lfo__ent  0.953454  0.001141  0.005824  0.940052  0.968404  100
# 8          lfo_hat__ent    0.9735  0.000034  0.000175  0.972991  0.974012  100
# 9         lfo__spec_ent  0.468449  0.006489  0.033109  0.396999  0.554188  100
# 10    lfo_hat__spec_ent  0.935404  0.000052  0.000267  0.934797  0.936131  100
# 11              lfo__tv  0.001692  0.000054  0.000277  0.001156  0.002401  100
# 12          lfo_hat__tv   0.33361  0.000254  0.001298  0.329906  0.336125  100
# 13              lfo__tp   0.00121  0.000058  0.000298  0.000584  0.002022  100
# 14          lfo_hat__tp  0.666794  0.000362  0.001845  0.661045  0.670051  100
