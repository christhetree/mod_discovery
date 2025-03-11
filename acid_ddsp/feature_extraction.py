import logging
import os

import librosa
import torch as tr
import torchaudio
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MelSpectrogram, FrequencyMasking

from losses_freq import DeltaMultiResolutionSTFTLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class LogMelSpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        sr: float = 48000,
        n_fft: int = 1024,
        hop_len: int = 32,
        n_mels: int = 128,
        normalized: bool = False,
        center: bool = True,
        use_delta: bool = False,
        use_delta_delta: bool = False,
        delta_win_ms: int = 50,
        freq_mask_amount: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.normalized = normalized
        self.center = center
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta
        self.delta_win_ms = delta_win_ms
        self.freq_mask_amount = freq_mask_amount
        self.eps = eps

        frame_rate = sr / hop_len
        self.delta_win_len = DeltaMultiResolutionSTFTLoss.calc_delta_win_len(
            delta_win_ms, frame_rate
        )
        if use_delta or use_delta_delta:
            log.info(
                f"use_delta: {use_delta}, use_delta_delta: {use_delta_delta}, "
                f"delta_win_ms: {delta_win_ms}, delta_win_len: {self.delta_win_len}, "
                f"frame_rate: {frame_rate:.2f}, hop_len: {hop_len}"
            )

        self.mel_spec = MelSpectrogram(
            sample_rate=int(sr),
            n_fft=n_fft,
            hop_length=hop_len,
            normalized=normalized,
            n_mels=n_mels,
            center=center,
        )
        self.center_freqs = librosa.mel_frequencies(
            n_mels=self.mel_spec.mel_scale.n_mels,
            fmin=self.mel_spec.mel_scale.f_min,
            fmax=self.mel_spec.mel_scale.f_max,
            htk=self.mel_spec.mel_scale.mel_scale == "htk",
        ).tolist()
        self.n_bins = n_mels
        self.freq_masking = FrequencyMasking(
            freq_mask_param=int(freq_mask_amount * self.n_bins)
        )

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.mel_spec(x)

        if self.training:
            if self.freq_mask_amount > 0:
                x = self.freq_masking(x)

        x = tr.clip(x, min=self.eps)
        x = tr.log(x)

        # time_averaged = x.mean(dim=-1, keepdim=True)
        # x = x - time_averaged

        if self.use_delta or self.use_delta_delta:
            delta = torchaudio.functional.compute_deltas(
                x, win_length=self.delta_win_len
            )
            if self.use_delta:
                x = tr.cat([x, delta], dim=1)
            if self.use_delta_delta:
                delta_delta = torchaudio.functional.compute_deltas(
                    delta, win_length=self.delta_win_len
                )
                x = tr.cat([x, delta_delta], dim=1)

        return x
