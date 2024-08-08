import logging
import os
from typing import Optional, List, Tuple, Dict

import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F

from curves import PiecewiseSplines
from feature_extraction import LogMelSpecFeatureExtractor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class Spectral2DCNNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        b_dil: int,
        t_dil: int,
        pool_size: Tuple[int, int],
        use_ln: bool,
    ):
        super().__init__()
        self.use_ln = use_ln
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=(1, 1),
            dilation=(b_dil, t_dil),
            padding="same",
        )
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.act = nn.PReLU(num_parameters=out_ch)

    def forward(self, x: T) -> T:
        assert x.ndim == 4
        n_bin = x.size(2)
        n_frame = x.size(3)
        if self.use_ln:
            # TODO(cm): parameterize eps
            x = F.layer_norm(x, [n_bin, n_frame])
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(x)
        return x


class Spectral2DCNN(nn.Module):
    global_param_names: List[str]

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
        n_frames: int = 188,
        n_segments: int = 4,
        degree: int = 3,
        filter_depth: int = 0,
        filter_width: int = 0,
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
        assert (
            n_frames % n_segments == 0
        ), f"n_frames = {n_frames}, n_segments = {n_segments}"
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree
        self.filter_depth = filter_depth
        self.filter_width = filter_width
        self.n_filters = filter_depth * filter_width
        self.n_curves = self.n_filters + 1

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
        # if "spline_bias" not in global_param_names:
        #     global_param_names.append("spline_bias")
        self.global_param_names = global_param_names
        if pool_size[1] == 1:
            log.info(
                f"Temporal receptive field: "
                f"{self.calc_receptive_field(kernel_size[1], temp_dilations)}"
            )

        # Define CNN
        layers = []
        curr_n_bins = fe.n_bins
        for out_ch, b_dil, t_dil in zip(out_channels, bin_dilations, temp_dilations):
            layers.append(
                Spectral2DCNNBlock(
                    in_ch, out_ch, kernel_size, b_dil, t_dil, pool_size, use_ln
                )
            )
            in_ch = out_ch
            curr_n_bins = curr_n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # Define temporal params
        # self.out_temp = nn.Linear(out_channels[-1], n_temp_params)
        n_hidden = (out_channels[-1] + n_temp_params) // 2
        self.out_temp = nn.Sequential(
            nn.Linear(out_channels[-1], n_hidden),
            nn.Dropout(p=dropout),
            nn.PReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=dropout),
            nn.PReLU(),
            nn.Linear(n_hidden, n_temp_params),
        )
        # assert n_temp_params == degree
        assert n_temp_params / degree == self.n_curves

        # Define filter temporal params
        # n_filter_coeff = filter_depth * filter_width * 5
        # n_hidden = (out_channels[-1] + n_filter_coeff) // 2
        # self.filter_temp = nn.Sequential(
        #     nn.Linear(out_channels[-1], n_hidden),
        #     nn.Dropout(p=dropout),
        #     nn.PReLU(),
        #     nn.Linear(n_hidden, n_filter_coeff),
        # )

        # Define temporal params
        # self.out_temp = nn.Linear(out_channels[-1], n_temp_params)
        n_hidden = (out_channels[-1] + self.n_curves) // 2
        self.out_bias = nn.Sequential(
            nn.Linear(out_channels[-1], n_hidden),
            nn.Dropout(p=dropout),
            nn.PReLU(num_parameters=n_hidden),
            nn.Linear(n_hidden, self.n_curves),
        )

        self.curves = PiecewiseSplines(n_frames, n_segments, degree)
        # assert n_temp_params % 2 == 0
        # self.curves = FourierSignal(n_frames, n_bins=n_temp_params // 2)
        n_logits = 5 * self.n_filters
        self.sub_lfo_adapter = TemporalAdapter(
            self.n_filters, n_logits, [(n_logits + self.n_filters) // 2] * 3, dropout
        )

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
        out_dict["log_spec_wet"] = log_spec

        # Calc latent
        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)
        out_dict["latent"] = latent

        # Calc global params
        x = tr.mean(latent, dim=-2)
        for param_name, mlp in self.out_global.items():
            p_val_hat = mlp(x).squeeze(-1)
            out_dict[param_name] = p_val_hat
        spline_bias = self.out_bias(x)

        # Calc temporal params
        # x = self.out_temp(latent)

        # Calc temporal params using piecewise splines
        x = latent.swapaxes(1, 2)
        assert (
            x.size(-1) % self.n_segments == 0
        ), f"x.size(-1) = {x.size(-1)}, self.n_segments = {self.n_segments}"
        x = x.view(x.size(0), x.size(1), self.n_segments, -1)
        x = tr.mean(x, dim=-1)
        x = x.swapaxes(1, 2)
        x = self.out_temp(x)
        # spline_bias = out_dict.get("spline_bias")
        # x = self.curves(x, spline_bias).unsqueeze(-1)

        x = x.view(x.size(0), x.size(1), self.n_curves, self.degree)
        x = tr.swapaxes(x, 1, 2)
        x = tr.flatten(x, start_dim=0, end_dim=1)
        spline_bias = tr.flatten(spline_bias)
        x = self.curves(x, spline_bias)
        x = x.view(-1, self.n_curves, x.size(1))
        x = tr.swapaxes(x, 1, 2)

        # Calc temporal params using Fourier signal
        # x = self.out_temp(x)
        # mag, phase = tr.chunk(x, 2, dim=1)
        # # mag = tr.tanh(raw_mag) * 2.0
        # phase = tr.sigmoid(phase) * 2 * tr.pi
        # x = self.curves(mag, phase).unsqueeze(-1)

        if self.temp_params_act_name is None:
            out_temp = x
        elif self.temp_params_act_name == "sigmoid":
            out_temp = tr.sigmoid(x)
        elif self.temp_params_act_name == "tanh":
            out_temp = tr.tanh(x)
        elif self.temp_params_act_name == "clamp":
            out_temp = tr.clamp(x, min=0.0, max=1.0)
        # elif self.temp_params_act_name == "magic_clamp":
        #     out_temp = magic_clamp(x, min_value=0.0, max_value=1.0)
        else:
            raise ValueError(f"Unknown activation: {self.temp_params_act_name}")
        out_dict[self.temporal_params_name] = out_temp
        add_lfo = out_temp[:, :, :1]
        out_dict["add_lfo"] = add_lfo

        # Calc filter params
        if self.filter_depth and self.filter_width:
            # x = self.filter_temp(latent)
            sub_lfo = out_temp[:, :, 1:]
            x = self.sub_lfo_adapter(sub_lfo)
            x = x.view(
                x.size(0), self.n_frames, self.filter_depth, self.filter_width, -1
            )
            out_dict["sub_lfo"] = x

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


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        n_temp_params: int,
        n_logits: int,
        hidden_sizes: List[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_temp_params = n_temp_params
        self.n_logits = n_logits
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        layers = []
        curr_in_ch = n_temp_params
        hidden_sizes = hidden_sizes + [n_logits]
        for n_hidden in hidden_sizes:
            layers.append(nn.Linear(curr_in_ch, n_hidden))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.PReLU())
            curr_in_ch = n_hidden
        self.adapter = nn.Sequential(*layers)

    def forward(self, x: T) -> Dict[str, T]:
        assert x.ndim == 3
        assert x.size(2) == self.n_temp_params
        x = self.adapter(x)
        return x


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

    tr.jit.script(model)
