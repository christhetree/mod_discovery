import logging
import os
from typing import Optional, List, Tuple, Dict

import torch as tr
import torchaudio
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import AmplitudeToDB

from adsr_haohao import extract_loudness
from curves import PiecewiseSplines
from feature_extraction import LogMelSpecFeatureExtractor
import librosa as li

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def get_activation(act_name: str) -> nn.Module:
    act_name = act_name.lower()
    if not act_name or act_name == "none":
        return nn.Identity()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {act_name}")


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
        temp_params: Optional[Dict[str, Dict[str, str | int]]] = None,
        global_param_names: Optional[List[str]] = None,
        dropout: float = 0.25,
        n_frames: int = 188,
        n_segments: int = 4,
        degree: int = 3,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.use_ln = use_ln
        self.dropout = dropout
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree

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
        if temp_params is None:
            temp_params = {
                "add_lfo": {
                    "dim": 1,
                    "act": "sigmoid",
                    "adapt_dim": 0,
                    "adapt_act": "none",
                },
                "sub_lfo": {
                    "dim": 1,
                    "act": "sigmoid",
                    "adapt_dim": 0,
                    "adapt_act": "none",
                },
            }
        self.temp_params = temp_params
        if global_param_names is None:
            global_param_names = []
        self.global_param_names = global_param_names
        if pool_size[1] == 1:
            log.info(
                f"Temporal receptive field: "
                f"{self.calc_receptive_field(kernel_size[1], temp_dilations)}"
            )

        # Define CNN
        layers = []
        loudness_layers = []
        curr_n_bins = fe.n_bins
        for out_ch, b_dil, t_dil in zip(out_channels, bin_dilations, temp_dilations):
            layers.append(
                Spectral2DCNNBlock(
                    in_ch, out_ch, kernel_size, b_dil, t_dil, pool_size, use_ln
                )
            )
            loudness_layers.append(
                Spectral2DCNNBlock(
                    in_ch, out_ch, (1, kernel_size[1]), 1, t_dil, (1, 1), use_ln
                )
            )
            in_ch = out_ch
            curr_n_bins = curr_n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # ADSR
        self.loudness_extractor = LoudnessExtractor(
            sr=fe.sr, n_fft=fe.n_fft, hop_len=fe.hop_len
        )
        self.loudness_cnn = nn.Sequential(*loudness_layers)
        n_hidden = (out_channels[-1] + 4) // 2
        self.loudness_mlp = nn.Sequential(
            nn.Linear(out_channels[-1], n_hidden),
            nn.Dropout(p=dropout),
            nn.PReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=dropout),
            nn.PReLU(),
            nn.Linear(n_hidden, 4),
            nn.Sigmoid(),
        )

        # Define temporal params
        self.out_temp = nn.ModuleDict()
        self.splines = nn.ModuleDict()
        self.spline_biases = nn.ModuleDict()
        self.spline_acts = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        for name, temp_param in self.temp_params.items():
            dim = temp_param["dim"]
            act = temp_param["act"]
            adapt_dim = temp_param["adapt_dim"]
            adapt_act = temp_param["adapt_act"]
            # Make frame by frame spline params
            actual_dim = dim * self.degree  # Needed for non-linear splines
            n_hidden = (out_channels[-1] + actual_dim) // 2
            self.out_temp[name] = nn.Sequential(
                nn.Linear(out_channels[-1], n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, actual_dim),
            )
            # Make splines
            self.splines[name] = PiecewiseSplines(n_frames, n_segments, degree)
            n_hidden = (out_channels[-1] + dim) // 2
            self.spline_biases[name] = nn.Sequential(
                nn.Linear(out_channels[-1], n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, dim),
            )
            self.spline_acts[name] = get_activation(act)
            # Make adapters (changes dimensions of mod sig from N to M)
            if adapt_dim:
                n_hidden = (dim + adapt_dim) // 2
                self.adapters[name] = nn.Sequential(
                    nn.Linear(dim, n_hidden),
                    nn.Dropout(p=dropout),
                    nn.PReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.Dropout(p=dropout),
                    nn.PReLU(),
                    nn.Linear(n_hidden, adapt_dim),
                    get_activation(adapt_act),
                )

        # n_hidden = (out_channels[-1] + self.degree) // 2
        # self.loudness_mlp = nn.Sequential(
        #     nn.Linear(out_channels[-1], n_hidden),
        #     nn.Dropout(p=dropout),
        #     nn.PReLU(),
        #     nn.Linear(n_hidden, n_hidden),
        #     nn.Dropout(p=dropout),
        #     nn.PReLU(),
        #     nn.Linear(n_hidden, self.degree),
        # )
        # self.loudness_spline = PiecewiseSplines(n_frames - 1, 4, degree)

        # Define global params
        self.out_global = nn.ModuleDict()
        for param_name in global_param_names:
            n_hidden = (out_channels[-1] + 1) // 2
            self.out_global[param_name] = nn.Sequential(
                nn.Linear(out_channels[-1], n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=dropout),
                nn.PReLU(),
                nn.Linear(n_hidden, out_features=1),
                nn.Sigmoid(),
            )

    def forward(self, x: T) -> Dict[str, T]:
        assert x.ndim == 3
        out_dict = {}

        # Extract features
        log_spec = self.fe(x)
        out_dict["log_spec_wet"] = log_spec
        n_frames = log_spec.size(-1)
        assert (
            n_frames == self.n_frames
        ), f"Expected n_frames: {self.n_frames} but got: {n_frames}"

        # loudness_2 = (
        #     extract_loudness(
        #         x.squeeze(1), sampling_rate=self.fe.sr, block_size=self.fe.hop_len, n_fft=self.fe.n_fft
        #     )
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        #     .float()
        # )
        # Extract envelope by conditioning on loudness
        loudness = self.loudness_extractor(x.squeeze(1))
        loudness = loudness.unsqueeze(1).unsqueeze(1)
        x = self.loudness_cnn(loudness)
        x = x.squeeze(2)
        x = tr.mean(x, dim=-1)
        x = self.loudness_mlp(x)
        attack = x[:, 0]
        decay = x[:, 1]
        sustain = x[:, 2]
        release = x[:, 3]
        out_dict["attack"] = attack
        out_dict["decay"] = decay
        out_dict["sustain"] = sustain
        out_dict["release"] = release

        # x = tr.swapaxes(x, 1, 2)
        # x = self.loudness_mlp(x)
        #
        # chunks = tr.tensor_split(x, 4, dim=1)
        # # chunks = tr.tensor_split(x, 36, dim=1)
        # chunks = [tr.mean(c, dim=1) for c in chunks]
        # x = tr.stack(chunks, dim=1)
        # x = tr.swapaxes(x, 1, 2)
        # x = x.view(x.size(0), 1, self.degree, x.size(2))
        # x = tr.swapaxes(x, 2, 3)
        #
        # x = self.loudness_spline(x)
        # x = tr.swapaxes(x, 1, 2)
        # TODO(cm): env should start at 0, not 0.5
        # env = tr.sigmoid(x).squeeze(-1)
        # out_dict["envelope"] = env

        # Calc latent
        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)
        out_dict["latent"] = latent
        global_latent = tr.mean(latent, dim=-2)

        # Calc temporal params
        for name, temp_param in self.temp_params.items():
            dim = temp_param["dim"]
            x = self.out_temp[name](latent)
            chunks = tr.tensor_split(x, self.n_segments, dim=1)
            chunks = [tr.mean(c, dim=1) for c in chunks]
            x = tr.stack(chunks, dim=1)
            # TODO(cm): check whether this is required,
            #  I'm trying to prevent flattening occurring along the temporal axis
            x = tr.swapaxes(x, 1, 2)
            x = x.view(x.size(0), dim, self.degree, x.size(2))
            x = tr.swapaxes(x, 2, 3)

            spline_bias = self.spline_biases[name](global_latent)
            x = self.splines[name](x, spline_bias)
            x = tr.swapaxes(x, 1, 2)
            x = self.spline_acts[name](x)

            out_dict[name] = x
            if name in self.adapters:
                x = self.adapters[name](x)
                out_dict[f"{name}_adapted"] = x

        # Calc global params
        for param_name, mlp in self.out_global.items():
            p_val_hat = mlp(global_latent).squeeze(-1)
            out_dict[param_name] = p_val_hat

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


class LoudnessExtractor(nn.Module):
    def __init__(
        self, sr: float, n_fft: int = 2048, hop_len: int = 512, top_db: float = 80.0
    ):
        super().__init__()
        assert n_fft % hop_len == 0, "n_fft must be divisible by hop_len"
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.top_db = top_db
        self.amp_to_db = AmplitudeToDB(stype="amplitude", top_db=top_db)
        frequencies = li.fft_frequencies(sr=sr, n_fft=n_fft)
        a_weighting = li.A_weighting(frequencies, min_db=-top_db)
        self.register_buffer(
            "a_weighting", tr.from_numpy(a_weighting).view(1, -1, 1).float()
        )

    def forward(self, x: T) -> T:
        assert x.ndim == 2
        x = tr.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        x = self.amp_to_db(x)
        x = x + self.a_weighting
        x = tr.pow(10, x / 20.0)  # TODO(cm): undo dB for taking mean?
        x = tr.mean(x, dim=-2)
        x = self.amp_to_db(x)
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
