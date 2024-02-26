import logging
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T
from torch import nn

from feature_extraction import LogMelSpecFeatureExtractor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_receptive_field(kernel_size: int, dilations: List[int]) -> int:
    """Compute the receptive field in samples."""
    assert dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
    rf = kernel_size
    for dil in dilations[1:]:
        rf = rf + ((kernel_size - 1) * dil)
    return rf


class Spectral2DCNN(nn.Module):
    def __init__(
        self,
        fe: LogMelSpecFeatureExtractor,
        in_ch: int = 2,
        kernel_size: Tuple[int, int] = (5, 7),
        out_channels: Optional[List[int]] = None,
        bin_dilations: Optional[List[int]] = None,
        temp_dilations: Optional[List[int]] = None,
        pool_size: Tuple[int, int] = (2, 1),
        latent_dim: int = 1,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        if out_channels is None:
            out_channels = [64] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [2**idx for idx in range(len(out_channels))]
        self.temp_dilations = temp_dilations
        assert len(out_channels) == len(bin_dilations) == len(temp_dilations)

        temporal_dims = [fe.n_frames] * len(out_channels)

        layers = []
        curr_n_bins = fe.n_bins
        for out_ch, b_dil, t_dil, temp_dim in zip(
            out_channels, bin_dilations, temp_dilations, temporal_dims
        ):
            if use_ln:
                layers.append(
                    nn.LayerNorm([curr_n_bins, temp_dim], elementwise_affine=False)
                )
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=(1, 1),
                    dilation=(b_dil, t_dil),
                    padding="same",
                )
            )
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU(num_parameters=out_ch))
            in_ch = out_ch
            curr_n_bins = curr_n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # TODO(cm): change from regression to classification
        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

    def forward(self, x: T) -> (T, T, T):
        assert x.ndim == 3
        log_spec = self.fe(x)

        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x

        x = self.output(x)
        x = tr.sigmoid(x)
        return x, latent, log_spec


if __name__ == "__main__":
    n_layers = 5
    temp_dilations = [2**idx for idx in range(n_layers)]
    kernel_size = 7
    rf = calc_receptive_field(kernel_size, temp_dilations)
    log.info(
        f"Receptive field: {rf}, "
        f"kernel_size: {kernel_size}, "
        f"temp_dilations: {temp_dilations}"
    )

    model = Spectral2DCNN()
    audio = tr.randn(1, 2, 6000)
    log.info(f"audio.shape: {audio.shape}")
    out, latent, _ = model(audio)
    log.info(f"out.shape: {out.shape}, latent.shape: {latent.shape}")
