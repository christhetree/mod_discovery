import logging
import os
from typing import Optional, List, Tuple
from magic_clamp import magic_clamp
import torch as tr
from torch import Tensor as T
from torch import nn

from feature_extraction import LogMelSpecFeatureExtractor
from tcn import TCN

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
        dropout: float = 0.25,
        # degree: int = 3,
        # n_segments: int = 4,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        # self.degree = degree
        # self.n_segments = n_segments

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

        self.out_ms = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

        self.out_q = nn.Sequential(
            nn.Linear(out_channels[-1], out_channels[-1] // 2),
            nn.Dropout(p=dropout),
            nn.PReLU(num_parameters=out_channels[-1] // 2),
            nn.Linear(out_channels[-1] // 2, 1),
            nn.Sigmoid(),
        )

        self.out_dist_gain = nn.Sequential(
            nn.Linear(out_channels[-1], out_channels[-1] // 2),
            nn.Dropout(p=dropout),
            nn.PReLU(num_parameters=out_channels[-1] // 2),
            nn.Linear(out_channels[-1] // 2, 1),
            nn.Sigmoid(),
        )

        # support = tr.linspace(0.0, 1.0, fe.n_frames).view(1, -1, 1, 1)
        # support = support.repeat(1, 1, n_segments, degree)
        #
        # segment_offsets = tr.linspace(0.0, 1.0, n_segments + 1)[:-1].view(1, 1, -1, 1)
        # support = support - segment_offsets
        # support = tr.clamp(support, min=0.0)
        #
        # exponent = tr.arange(start=1, end=degree + 1).int().view(1, 1, 1, -1)
        # # self.register_buffer("exponent", exponent)
        # support = support.pow(exponent)
        #
        # self.register_buffer("support", support)
        #
        # fc_out_dim = (out_channels[-1] + degree) // 2
        # self.fc = nn.Linear(out_channels[-1], fc_out_dim)
        # self.fc_prelu = nn.PReLU()
        # self.out_poly = nn.Linear(fc_out_dim, degree)
        #
        # self.fc_global = nn.Linear(out_channels[-1], fc_out_dim)
        # self.fc_global_prelu = nn.PReLU(num_parameters=fc_out_dim)
        # self.out_global = nn.Linear(fc_out_dim, 1)

        # self.lstm = nn.LSTM(out_channels[-1], out_channels[-1], batch_first=True)
        # self.lstm = nn.LSTM(fe.n_bins, out_channels[-1], batch_first=True)

    def forward(self, x: T) -> (T, T, Optional[T]):
        assert x.ndim == 3
        log_spec = self.fe(x)

        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x

        # x = tr.swapaxes(x, 1, 2)
        # x, _ = self.lstm(x)
        # x = tr.swapaxes(x, 1, 2)

        # x = tr.mean(latent, dim=-1)
        # x = self.fc_global(x)
        # x = self.fc_global_prelu(x)
        # x = self.out_global(x)
        # bias = x
        # # bias = magic_clamp(x, min_value=0.0, max_value=1.0)

        # x = latent.view(latent.size(0), latent.size(1), self.n_segments, -1)
        # x = tr.mean(x, dim=-1)
        # x = x.permute(0, 2, 1)
        # x = self.fc(x)
        # x = self.fc_prelu(x)
        # x = self.out_poly(x)
        # coeffs = x.unsqueeze(1)

        # support = self.support
        # x = coeffs * support
        # x = tr.sum(x, dim=-1)
        # x = tr.sum(x, dim=-1)
        # x += bias
        # x = tr.sigmoid(x)
        # # x = magic_clamp(x, min_value=0.0, max_value=1.0)

        x = self.out_ms(x)
        ms_hat = tr.sigmoid(x)
        # ms_hat = magic_clamp(x, min_value=0.0, max_value=1.0)

        x = tr.mean(latent, dim=-1)
        q_norm_hat = self.out_q(x).squeeze(-1)
        dist_gain_norm_hat = self.out_dist_gain(x).squeeze(-1)

        return ms_hat, q_norm_hat, dist_gain_norm_hat, latent, log_spec


class AudioTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        dilations: List[int],
        padding: str = "same",
        padding_mode: str = "zeros",
        act_name: str = "prelu",
    ) -> None:
        super().__init__()
        self.tcn = TCN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            causal=False,
            cached=False,
            act_name=act_name,
        )
        log.info(f"TCN receptive field: {self.tcn.calc_receptive_field()}")
        self.output = nn.Conv1d(out_channels[-1], out_channels=1, kernel_size=1)

    def forward(self, x: T) -> (T, T, Optional[T]):
        latent = self.tcn(x)
        x = self.output(latent)
        x = tr.sigmoid(x)
        return x, latent, None


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
