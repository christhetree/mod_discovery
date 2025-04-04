import logging
import os

import librosa
import torch as tr
from numpy import ndarray
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class RMSCosineSimilarity(nn.Module):
    def __init__(self, win_len: int, hop_len: int, collapse_channels: bool = True):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.collapse_channels = collapse_channels

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        with tr.no_grad():
            if self.collapse_channels:
                x = x.mean(dim=1)
                x_target = x_target.mean(dim=1)
            else:
                x = x.view(-1, x.size(-1))
                x_target = x_target.view(-1, x_target.size(-1))
            x = x.cpu().numpy()
            x_target = x_target.cpu().numpy()
            x_rms = librosa.feature.rms(
                y=x, frame_length=self.win_len, hop_length=self.hop_len
            ).squeeze(1)
            x_target_rms = librosa.feature.rms(
                y=x_target, frame_length=self.win_len, hop_length=self.hop_len
            ).squeeze(1)
            x_rms = tr.from_numpy(x_rms)
            x_target_rms = tr.from_numpy(x_target_rms)
            cosine_sim = tr.nn.functional.cosine_similarity(x_rms, x_target_rms, dim=-1)
            cosine_sim = cosine_sim.mean()
            return cosine_sim


class SpectralCentroidCosineSimilarity(nn.Module):
    def __init__(self, sr: int, win_len: int, hop_len: int, collapse_channels: bool = True):
        super().__init__()
        self.sr = sr
        self.win_len = win_len
        self.hop_len = hop_len
        self.collapse_channels = collapse_channels

    def _calc_feature(self, x: ndarray) -> ndarray:
        spectral_centroid = librosa.feature.spectral_centroid(
            y=x, sr=self.sr, n_fft=self.win_len, hop_length=self.hop_len
        ).squeeze(1)
        return spectral_centroid

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        with tr.no_grad():
            if self.collapse_channels:
                x = x.mean(dim=1)
                x_target = x_target.mean(dim=1)
            else:
                x = x.view(-1, x.size(-1))
                x_target = x_target.view(-1, x_target.size(-1))
            x = x.cpu().numpy()
            x_target = x_target.cpu().numpy()
            sc = self._calc_feature(x)
            sc_target = self._calc_feature(x_target)
            spectral_centroid = tr.from_numpy(sc)
            spectral_centroid_target = tr.from_numpy(sc_target)
            cosine_sim = tr.nn.functional.cosine_similarity(
                spectral_centroid, spectral_centroid_target, dim=-1
            )
            cosine_sim = cosine_sim.mean()
            return cosine_sim


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
