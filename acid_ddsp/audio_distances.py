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
from torchaudio.transforms import MFCC

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MFCCDistance(nn.Module):
    def __init__(
        self,
        sr: int,
        log_mels: bool = True,
        n_fft: int = 2048,
        hop_len: int = 512,
        n_mels: int = 128,
        p: int = 1,
    ):
        super().__init__()
        self.p = p
        self.mfcc = MFCC(
            sample_rate=sr,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_len,
                "n_mels": n_mels,
            },
        )
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        if self.p == 1:
            return self.l1(self.mfcc(x), self.mfcc(x_target))
        elif self.p == 2:
            return self.mse(self.mfcc(x), self.mfcc(x_target))
        else:
            raise ValueError(f"Unknown p value: {self.p}")


class OneDimensionalAudioDistance(nn.Module, abc.ABC):
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


class RMSDistance(OneDimensionalAudioDistance):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.rms(
            y=x, frame_length=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralCentroidDistance(OneDimensionalAudioDistance):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_centroid(
            y=x, sr=self.sr, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralBandwidthDistance(OneDimensionalAudioDistance):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_bandwidth(
            y=x, sr=self.sr, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


class SpectralFlatnessDistance(OneDimensionalAudioDistance):
    def calc_feature(self, x: T) -> T:
        x = x.cpu().numpy()
        x = librosa.feature.spectral_flatness(
            y=x, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        x = tr.from_numpy(x).float()
        return x


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
