import logging
import os
from typing import Optional, List, Tuple, Dict
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
        use_ln: bool = True,
        dropout: float = 0.25,
        n_logits: int = 0,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.use_ln = use_ln
        self.dropout = dropout
        self.n_logits = n_logits

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

        if n_logits:
            self.out_frame_wise = nn.Linear(out_channels[-1], n_logits)
        else:
            self.out_frame_wise = nn.Linear(out_channels[-1], 1)

        self.out_q = None
        if not n_logits:
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

        self.out_osc_shape = nn.Sequential(
            nn.Linear(out_channels[-1], out_channels[-1] // 2),
            nn.Dropout(p=dropout),
            nn.PReLU(num_parameters=out_channels[-1] // 2),
            nn.Linear(out_channels[-1] // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: T) -> Dict[str, T]:
        assert x.ndim == 3
        log_spec = self.fe(x)

        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)

        x = self.out_frame_wise(latent)

        ms_hat = None
        logits = None
        if self.n_logits:
            logits = x
        else:
            x = tr.sigmoid(x)
            # x = magic_clamp(x, min_value=0.0, max_value=1.0)
            ms_hat = x.squeeze(-1)

        x = tr.mean(latent, dim=-2)
        q_norm_hat = None
        if not self.n_logits:
            q_norm_hat = self.out_q(x).squeeze(-1)
        dist_gain_norm_hat = self.out_dist_gain(x).squeeze(-1)
        osc_shape_norm_hat = self.out_osc_shape(x).squeeze(-1)

        return {
            "mod_sig_hat": ms_hat,
            "q_norm_hat": q_norm_hat,
            "dist_gain_norm_hat": dist_gain_norm_hat,
            "osc_shape_norm_hat": osc_shape_norm_hat,
            "latent": latent,
            "log_spec": log_spec,
            "logits": logits,
        }


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
