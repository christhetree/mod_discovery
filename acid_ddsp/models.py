import logging
import os
from typing import Optional, List, Tuple, Dict

import torch as tr
from magic_clamp import magic_clamp
from torch import Tensor as T
from torch import nn

from feature_extraction import LogMelSpecFeatureExtractor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class Spectral2DCNN(nn.Module):
    def __init__(
        self,
        fe: LogMelSpecFeatureExtractor,
        in_ch: int = 1,
        kernel_size: Tuple[int, int] = (5, 7),
        out_channels: Optional[List[int]] = None,
        bin_dilations: Optional[List[int]] = None,
        temp_dilations: Optional[List[int]] = None,
        pool_size: Tuple[int, int] = (2, 1),
        use_ln: bool = True,
        n_temp_params: int = 1,
        temp_params_name: str = "mod_sig",
        temp_params_act_name: Optional[str] = "sigmoid",
        global_param_names: Optional[List[str]] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.use_ln = use_ln
        self.n_temporal_params = n_temp_params
        self.temporal_params_name = temp_params_name
        self.temp_params_act_name = temp_params_act_name
        self.dropout = dropout

        # Define default params
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
        if global_param_names is None:
            global_param_names = []
        self.global_param_names = global_param_names
        if pool_size[1] == 1:
            log.info(
                f"Temporal receptive field: "
                f"{self.calc_receptive_field(kernel_size[1], temp_dilations)}"
            )

        # Define CNN
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

        # Define temporal params
        self.out_temp = nn.Linear(out_channels[-1], n_temp_params)

        # Define global params
        self.out_global = nn.ModuleDict()
        for param_name in global_param_names:
            self.out_global[param_name] = nn.Sequential(
                nn.Linear(out_channels[-1], out_channels[-1] // 2),
                nn.Dropout(p=dropout),
                nn.PReLU(num_parameters=out_channels[-1] // 2),
                nn.Linear(out_channels[-1] // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: T) -> Dict[str, T]:
        assert x.ndim == 3
        out_dict = {}

        # Extract features
        log_spec = self.fe(x)
        out_dict["log_spec"] = log_spec

        # Calc latent
        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)
        out_dict["latent"] = latent

        # Calc temporal params
        x = self.out_temp(latent)
        if self.temp_params_act_name == "sigmoid":
            out_temp = tr.sigmoid(x)
        elif self.temp_params_act_name == "tanh":
            out_temp = tr.tanh(x)
        elif self.temp_params_act_name == "clamp":
            out_temp = tr.clamp(x, min=0.0, max=1.0)
        elif self.temp_params_act_name == "magic_clamp":
            out_temp = magic_clamp(x, min_value=0.0, max_value=1.0)
        elif self.temp_params_act_name is None:
            out_temp = x
        else:
            raise ValueError(f"Unknown activation: {self.temp_params_act_name}")
        out_dict[self.temporal_params_name] = out_temp

        # Calc global params
        x = tr.mean(latent, dim=-2)
        for param_name in self.global_param_names:
            x = self.out_global[param_name](x).squeeze(-1)
            out_dict[param_name] = x

        return out_dict

    @staticmethod
    def calc_receptive_field(kernel_size: int, dilations: List[int]) -> int:
        """Compute the receptive field in samples."""
        assert dilations
        assert dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = kernel_size
        for dil in dilations[1:]:
            rf = rf + ((kernel_size - 1) * dil)
        return rf


if __name__ == "__main__":
    n_layers = 5
    temp_dilations = [2**idx for idx in range(n_layers)]
    kernel_size = 7
    rf = Spectral2DCNN.calc_receptive_field(kernel_size, temp_dilations)
    log.info(
        f"Receptive field: {rf}, "
        f"kernel_size: {kernel_size}, "
        f"temp_dilations: {temp_dilations}"
    )

    fe = LogMelSpecFeatureExtractor()
    model = Spectral2DCNN(fe)
    audio = tr.randn(3, 1, 6000)
    log.info(f"audio.shape: {audio.shape}")
    out_dict = model(audio)
    mod_sig = out_dict["mod_sig"]
    latent = out_dict["latent"]
    log.info(f"mod_sig.shape: {mod_sig.shape}, latent.shape: {latent.shape}")
