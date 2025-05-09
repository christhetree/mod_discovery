import abc
import logging
import os
from abc import abstractmethod
from typing import Optional, Callable, Dict

import librosa
import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MFCC

import util

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
        n_mfcc: int = 40,
        p: int = 1,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.p = p

        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
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
        dist_fn_s: Dict[str, Callable[[T, T], T]],
        average_channels: bool = True,
        filter_cf_hz: Optional[float] = 8.0,
        n_filter: int = 63,
    ):
        super().__init__()
        self.sr = sr
        self.win_len = win_len
        self.hop_len = hop_len
        self.average_channels = average_channels
        self.dist_fn_s = dist_fn_s
        self.filter_cf_hz = filter_cf_hz
        self.n_filter = n_filter

        self.filter = None
        filter_sr = sr / hop_len
        # assert filter_sr == 500.0
        # assert n_filter == 63
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
        # assert self.filter_cf_hz is not None
        if self.filter_cf_hz is not None:
            x = F.pad(
                x,
                (self.filter.size(-1) // 2, self.filter.size(-1) // 2),
                mode="replicate",
            )
            x = F.conv1d(x.unsqueeze(1), self.filter, padding="valid").squeeze(1)
        return x

    def forward(self, x: T, x_target: T, p_hats: Optional[T] = None) -> Dict[str, T]:
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

        assert self.filter_cf_hz is not None
        x_filtered = self.maybe_filter_feature(x)
        x_target_filtered = self.maybe_filter_feature(x_target)

        # from matplotlib import pyplot as plt
        # # plt.plot(x[0].cpu().numpy(), label="x")
        # plt.plot(x_target[0].cpu().numpy(), label="x_target", color="orange")
        # plt.plot(x_target_filtered[0].cpu().numpy(), label="x_target_f", color="red")

        p_hats_inv_all = None
        p_hats_filtered_inv_all = None
        x_target_frames = None
        x_target_filtered_frames = None
        if p_hats is not None:
            n_frames = p_hats.size(2)
            x_target_frames = util.interpolate_dim(x_target, n_frames, dim=1)
            x_target_filtered_frames = util.interpolate_dim(
                x_target_filtered, n_frames, dim=1
            )
            p_hats = p_hats.to(x_target_frames.device)
            p_hats_inv_all = util.compute_lstsq_with_bias(
                x_hat=p_hats, x=x_target_frames.unsqueeze(1)
            ).squeeze(1)
            p_hats_filtered_inv_all = util.compute_lstsq_with_bias(
                x_hat=p_hats, x=x_target_filtered_frames.unsqueeze(1)
            ).squeeze(1)

        # plt.plot(p_hats_inv_all[0].cpu().numpy(), label="inv_all", color="blue")
        # # plt.plot(p_hats_filtered_inv_all[0].cpu().numpy(), label="inv_all_f", color="purple]")
        # plt.title(f"{self.__class__.__name__} {self.filter_cf_hz} Hz")
        # plt.legend()
        # plt.show()

        assert x.shape == x_target.shape
        dists = {}
        for dist_name, dist_fn in self.dist_fn_s.items():
            dist = dist_fn(x, x_target)
            dist = dist.mean()
            dists[dist_name] = dist
            dist_filtered = dist_fn(x_filtered, x_target_filtered)
            dist_filtered = dist_filtered.mean()
            dists[f"{dist_name}__cf_{self.filter_cf_hz:.0f}_hz"] = dist_filtered
            if p_hats is not None:
                # try:
                dist_inv_all = dist_fn(p_hats_inv_all, x_target_frames)
                dist_filtered_inv_all = dist_fn(
                    p_hats_filtered_inv_all, x_target_filtered_frames
                )
                # except AssertionError:
                #     p_hats_inv_all = util.interpolate_dim(
                #         p_hats_inv_all, dist_fn.n_frames, dim=1
                #     )
                #     p_hats_filtered_inv_all = util.interpolate_dim(
                #         p_hats_filtered_inv_all, dist_fn.n_frames, dim=1
                #     )
                #     dist_inv_all = dist_fn(p_hats_inv_all, x_target)
                #     dist_filtered_inv_all = dist_fn(
                #         p_hats_filtered_inv_all, x_target_filtered
                #     )
                dist_inv_all = dist_inv_all.mean()
                dists[f"{dist_name}__inv_all"] = dist_inv_all
                dist_filtered_inv_all = dist_filtered_inv_all.mean()
                dists[f"{dist_name}__cf_{self.filter_cf_hz:.0f}_hz__inv_all"] = (
                    dist_filtered_inv_all
                )
        return dists


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
