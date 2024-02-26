import logging
import os

import torch as tr
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class LogMelSpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_samples: int = 6000,
        sr: float = 48000,
        n_fft: int = 1024,
        hop_len: int = 32,
        n_mels: int = 128,
        normalized: bool = False,
        center: bool = True,
        freq_mask_amount: float = 0.0,
        time_mask_amount: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.freq_mask_amount = freq_mask_amount
        self.time_mask_amount = time_mask_amount
        self.eps = eps
        self.mel_spec = MelSpectrogram(
            sample_rate=int(sr),
            n_fft=n_fft,
            hop_length=hop_len,
            normalized=normalized,
            n_mels=n_mels,
            center=center,
        )
        self.n_bins = n_mels
        self.n_frames = n_samples // hop_len + 1
        self.freq_mask = FrequencyMasking(
            freq_mask_param=int(freq_mask_amount * self.n_bins)
        )
        self.time_mask = TimeMasking(
            time_mask_param=int(time_mask_amount * self.n_frames)
        )

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.mel_spec(x)

        if self.training:
            if self.freq_mask_amount > 0:
                x = self.freq_masking(x)
            if self.time_mask_amount > 0:
                x = self.time_masking(x)

        x = tr.clip(x, min=self.eps)
        x = tr.log(x)
        assert x.shape == (x.size(0), x.size(1), self.n_bins, self.n_frames)
        return x
