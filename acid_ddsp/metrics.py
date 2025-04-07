import abc
import logging
import os
from abc import abstractmethod
from typing import Literal, Optional

import librosa
import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioMetric(nn.Module, abc.ABC):
    def __init__(
        self,
        sr: int,
        win_len: int,
        hop_len: int,
        average_channels: bool = True,
        dist_fn: Literal["coss", "pcc"] = "pcc",
        filter_cf_hz: Optional[float] = 8.0,
    ):
        super().__init__()
        self.sr = sr
        self.win_len = win_len
        self.hop_len = hop_len
        self.average_channels = average_channels
        self.dist_fn = dist_fn
        self.filter_cf_hz = filter_cf_hz

        self.filter = None
        filter_sr = sr / hop_len
        assert filter_sr == 80.0
        n_filter = 11
        filter_support = 2 * (tr.arange(n_filter) - (n_filter - 1) / 2) / filter_sr
        filter_window = tr.blackman_window(n_filter, periodic=False)
        if filter_cf_hz is not None:
            h = tr.sinc(filter_cf_hz * filter_support) * filter_window
            h /= h.sum()
            self.filter = h.view(1, 1, -1)

    @abstractmethod
    def calc_feature(self, x: T) -> T:
        pass

    def maybe_filter_feature(self, x: T) -> T:
        if self.filter_cf_hz is not None:
            x = F.conv1d(x.unsqueeze(1), self.filter, padding="valid").squeeze(1)
        return x

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        with tr.no_grad():
            if self.average_channels:
                x = x.mean(dim=1)
                x_target = x_target.mean(dim=1)
            else:
                x = x.view(-1, x.size(-1))
                x_target = x_target.view(-1, x_target.size(-1))
            x = self.calc_feature(x)
            x_target = self.calc_feature(x_target)
            assert x.ndim == x_target.ndim == 2
            x = self.maybe_filter_feature(x)
            x_target = self.maybe_filter_feature(x_target)
            assert x.shape == x_target.shape
            if self.dist_fn == "coss":
                dist = tr.nn.functional.cosine_similarity(x, x_target, dim=-1)
            elif self.dist_fn == "pcc":
                pcc_s = []
                # TODO(cm): vectorize
                for idx in range(x.size(0)):
                    curr_x = x[idx, :]
                    curr_x_target = x_target[idx, :]
                    data = tr.stack([curr_x, curr_x_target], dim=0)
                    corr_matrix = tr.corrcoef(data)
                    pcc = corr_matrix[0, 1]
                    pcc_s.append(pcc)
                dist = tr.stack(pcc_s, dim=0)
            else:
                raise ValueError(f"Unknown distance function: {self.dist_fn}")
            dist = dist.mean()
            return dist


class RMSMetric(AudioMetric):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.rms(
            y=x, frame_length=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralCentroidMetric(AudioMetric):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_centroid(
            y=x, sr=self.sr, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralBandwidthMetric(AudioMetric):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_bandwidth(
            y=x, sr=self.sr, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralFlatnessMetric(AudioMetric):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_flatness(
            y=x, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class LFOMetric(nn.Module, abc.ABC):
    def __init__(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    @abstractmethod
    def calc_metric(self, x: T) -> T:
        pass

    def forward(self, x: T) -> T:
        assert x.ndim == 2
        with tr.no_grad():
            if self.normalize:
                x_min = x.min(dim=1, keepdim=True).values
                x_max = x.max(dim=1, keepdim=True).values
                x_range = x_max - x_min
                x = (x - x_min) / (x_range + self.eps)
            metric = self.calc_metric(x)
            metric = metric.mean()
            return metric


class EntropyMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        assert x.min() >= 0.0
        x = x / x.sum(dim=1, keepdim=True)
        x_log = tr.log(x)
        x_log = tr.nan_to_num(x_log, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -x * x_log
        entropy = entropy.sum(dim=1)
        assert x.size(1) > 1
        entropy /= tr.log(tr.tensor(x.size(1)))
        return entropy


class TotalVariationMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        assert x.size(1) > 1
        diffs = x[:, 1:] - x[:, :-1]
        diffs = tr.abs(diffs)
        diffs = diffs.mean(dim=1)
        return diffs


class TurningPointsMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        assert x.size(1) > 2
        diffs = x[:, 1:] - x[:, :-1]
        diffs = tr.sign(diffs)
        ddiffs = diffs[:, 1:] * diffs[:, :-1]
        turning_points = (ddiffs < 0).sum(dim=1).float()
        turning_points = turning_points / (x.size(1) - 2)
        return turning_points


# import librosa
# import numpy as np
# from dtw import dtw
#
# def compute_mfcc(target: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
#     window_length = int(0.05 * sample_rate)
#     hop_length = int(0.01 * sample_rate)
#
#     mfcc = librosa.feature.mfcc(
#         y=target,
#         sr=sample_rate,
#         n_mfcc=20,
#         n_fft=window_length,
#         hop_length=hop_length,
#         n_mels=128,
#     )
#
#     return mfcc
#
#
# def compute_wmfcc(target: np.ndarray, pred: np.ndarray) -> float:
#     logger.info("Computing wMFCC...")
#
#     target_mfcc = compute_mfcc(target)
#     pred_mfcc = compute_mfcc(pred)
#
#     target_mfcc = target_mfcc.reshape(-1, target_mfcc.shape[-1])
#     pred_mfcc = pred_mfcc.reshape(-1, pred_mfcc.shape[-1])
#
#     def l1(a, b):
#         return np.mean(np.abs(a - b))
#
#     dist = dtw(target_mfcc.T, pred_mfcc.T, dist_method=l1, distance_only=True)
#     return dist.normalizedDistance


if __name__ == "__main__":
    # x = tr.tensor([1, 2, 3, 2, 1]).float().view(1, -1).repeat(1, 1)
    x = tr.tensor([0, -1, 1, 0]).float().view(1, -1).repeat(1, 1)
    # y = tr.tensor([1, 2, 3, 2, 1]).float().view(1, -1).repeat(1, 1)
    # y[1, :] -= 1

    # metric = EntropyMetric()
    metric = TotalVariationMetric()
    # metric = TurningPointsMetric()
    dist = metric(x)
    print(dist)
    exit()



    cos = tr.nn.functional.cosine_similarity(x, y, dim=-1)
    print(cos)

    pcc_s = []
    for idx in range(x.size(0)):
        curr_x = x[idx, :]
        curr_x_target = y[idx, :]
        data = tr.stack([curr_x, curr_x_target], dim=0)
        corr_matrix = tr.corrcoef(data)
        pcc = corr_matrix[0, 1]
        pcc_s.append(pcc)
    dist = tr.stack(pcc_s, dim=0)
    print(dist)
