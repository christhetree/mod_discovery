import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import librosa as li
import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import AmplitudeToDB

import util
from curves import PiecewiseBezier2D
from feature_extraction import LogMelSpecFeatureExtractor
from modulations import ModSignalGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


@dataclass
class TempParam:
    dim: int = 1
    act: str = "none"
    is_spline: bool = False
    n_segments: int = 12
    degree: int = 3
    is_c1_cont: bool = False
    is_bounded: bool = False
    use_alpha_noise: bool = False
    use_alpha_linear: bool = False
    adapt_dim: int = 0
    adapt_act: str = "none"
    adapt_use_latent: bool = False
    adapt_use_separate: bool = False

    def __post_init__(self):
        if self.is_spline:
            assert (
                self.act == "none"
            ), "Spline control points should not have an activation"
        else:
            assert not self.adapt_use_latent, "Using the latent dim is not supported"
            assert not self.is_bounded, "is_bounded has no meaning for non-splines"
            assert not self.use_alpha_linear, "use_alpha_linear is only for splines"
        if self.is_c1_cont:
            assert self.is_spline, "C1 continuity only makes sense for splines"
            assert not self.is_bounded, "is_bounded has no meaning for C1 splines"
            assert self.degree >= 3, "C1 continuity requires degree >= 3"


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        unroll: bool,
        num_layers: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.unroll = unroll
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
        )

    def forward(self, x: T) -> T:
        output, (h_n, _) = self.lstm(x)
        if self.unroll:
            return output
        else:
            emb = tr.cat((h_n[-2], h_n[-1]), dim=-1)
            return emb


class MeanPooling(nn.Module):
    def __init__(self, n_out_frames: int):
        super().__init__()
        self.n_out_frames = n_out_frames

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        assert x.size(1) >= self.n_out_frames
        chunks = tr.tensor_split(x, self.n_out_frames, dim=1)
        chunks = [tr.mean(c, dim=1) for c in chunks]
        chunks = tr.stack(chunks, dim=1)
        return chunks


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
        act_name: str = "prelu",
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
        if act_name == "prelu":
            self.act = nn.PReLU(num_parameters=out_ch)
        else:
            self.act = util.get_activation(act_name)

    def forward(self, x: T) -> T:
        assert x.ndim == 4
        n_bin = x.size(2)
        n_frame = x.size(3)
        if self.use_ln:
            x = F.layer_norm(x, [n_bin, n_frame])
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(x)
        return x


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
        temp_params: Optional[Dict[str, Dict[str, str | int | bool]]] = None,
        dropout: float = 0.0,
        n_frames: int = 1501,
        cnn_act: str = "prelu",
        fc_act: str = "prelu",
        noise_std: float = 0.33,
        spline_eps: float = 1e-3,
        interp_n_frames: Optional[int] = None,
        filter_cf_hz: Optional[float] = None,
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
        self.cnn_act = cnn_act
        self.fc_act = fc_act
        self.noise_std = noise_std
        self.spline_eps = spline_eps
        self.interp_n_frames = interp_n_frames
        self.filter_cf_hz = filter_cf_hz

        self.n_pad = None
        filter_sr = n_frames // 3
        assert filter_sr == 500
        n_filter = 63
        filter_support = 2 * (tr.arange(n_filter) - (n_filter - 1) / 2) / filter_sr
        filter_window = tr.blackman_window(n_filter, periodic=False)
        if filter_cf_hz is not None:
            h = tr.sinc(filter_cf_hz * filter_support) * filter_window
            h /= h.sum()
            self.register_buffer("filter", h.view(1, 1, -1), persistent=False)
            self.n_pad = n_filter // 2

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
            temp_params = {}
        self.temp_params = {k: TempParam(**v) for k, v in temp_params.items()}
        assert temp_params, "No params to predict"
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
                    in_ch, out_ch, kernel_size, b_dil, t_dil, pool_size, use_ln, cnn_act
                )
            )
            in_ch = out_ch
            curr_n_bins = curr_n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)
        self.latent_dim = out_channels[-1]

        # Define temporal params
        self.out_temp = nn.ModuleDict()
        self.out_temp_acts = nn.ModuleDict()
        self.splines = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        self.adapter_acts = nn.ModuleDict()

        for name, tp in self.temp_params.items():
            # Make frame by frame spline params or features
            in_dim = self.latent_dim
            out_dim = tp.dim
            mean_pooling_layer = nn.Identity()
            if tp.is_spline:
                out_dim = 2  # For PiecewiseBezier2D
                mean_pooling_layer = MeanPooling(tp.n_segments * tp.degree + 1)
            n_hidden = (self.latent_dim + out_dim) // 2
            self.out_temp[name] = nn.Sequential(
                BidirectionalLSTM(in_dim, in_dim // 2, unroll=True),
                mean_pooling_layer,
                nn.Linear(in_dim, n_hidden),
                nn.Dropout(p=dropout),
                util.get_activation(self.fc_act),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=dropout),
                util.get_activation(self.fc_act),
                nn.Linear(n_hidden, out_dim),
            )
            self.out_temp_acts[name] = util.get_activation(tp.act)
            # Make adapters (changes dimensions of mod sig from N to M)
            if tp.adapt_dim:
                # adapt_in_dim = tp.dim
                adapt_in_dim = tp.dim + 1
                if tp.adapt_use_latent:
                    adapt_in_dim += self.latent_dim
                n_hidden = (adapt_in_dim + tp.adapt_dim) // 2
                if tp.adapt_use_separate:
                    for dim_idx in range(tp.adapt_dim):
                        self.adapters[f"{name}_{dim_idx}"] = nn.Sequential(
                            nn.Linear(adapt_in_dim, n_hidden),
                            nn.Dropout(p=dropout),
                            util.get_activation(self.fc_act),
                            nn.Linear(n_hidden, n_hidden),
                            nn.Dropout(p=dropout),
                            util.get_activation(self.fc_act),
                            nn.Linear(n_hidden, 1),
                        )
                else:
                    self.adapters[name] = nn.Sequential(
                        nn.Linear(adapt_in_dim, n_hidden),
                        nn.Dropout(p=dropout),
                        util.get_activation(self.fc_act),
                        nn.Linear(n_hidden, n_hidden),
                        nn.Dropout(p=dropout),
                        util.get_activation(self.fc_act),
                        nn.Linear(n_hidden, tp.adapt_dim),
                    )
                self.adapter_acts[name] = util.get_activation(tp.adapt_act)
            if tp.is_spline:
                # Make splines
                self.splines[name] = PiecewiseBezier2D(
                    n_frames,
                    tp.n_segments,
                    tp.degree,
                    is_c1_cont=tp.is_c1_cont,
                    eps=self.spline_eps,
                )

        if self.interp_n_frames is None:
            self.register_buffer(
                "pos_enc",
                tr.linspace(0, 1, self.n_frames).view(1, -1, 1),
                persistent=False,
            )
        else:
            self.register_buffer(
                "pos_enc",
                tr.linspace(0, 1, self.interp_n_frames).view(1, -1, 1),
                persistent=False,
            )

    def forward(
        self,
        x: Dict[str, T],
        tp_name: Optional[str] = None,
        alpha_noise: Optional[float] = None,
        alpha_linear: Optional[float] = None,
    ) -> Dict[str, T]:
        if alpha_noise is not None:
            assert 0.0 <= alpha_noise <= 1.0
        if alpha_linear is not None:
            assert 0.0 <= alpha_linear <= 1.0

        x = x["audio"]
        assert x.ndim == 3
        out_dict = {}

        # Extract features
        log_spec = self.fe(x)
        out_dict["log_spec_x"] = log_spec
        n_frames = log_spec.size(-1)
        assert (
            n_frames == self.n_frames
        ), f"Expected n_frames: {self.n_frames} but got: {n_frames}"

        # Calc latent
        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)
        out_dict["latent"] = latent

        # Calc temporal params
        for name, tp in self.temp_params.items():
            if tp_name is not None and name != tp_name:
                continue
            if tp.is_spline:
                assert tp.dim == 1
                x = self.out_temp[name](latent)
                logits_x = x[:, :, 0]
                cp_x = self.process_bezier_logits_x(
                    logits_x,
                    tp.n_segments,
                    tp.degree,
                    alpha_noise=None,
                    noise_std=self.noise_std,
                    alpha_linear=alpha_linear if tp.use_alpha_linear else None,
                    softmax_tau=0.25,
                )
                logits_y = x[:, :, 1]
                cp_y = self.process_bezier_logits_y(
                    logits_y,
                    tp.n_segments,
                    tp.degree,
                    alpha_noise=alpha_noise if tp.use_alpha_noise else None,
                    noise_std=self.noise_std,
                    alpha_linear=alpha_linear if tp.use_alpha_linear else None,
                    is_bounded=tp.is_bounded,
                    eps=self.spline_eps,
                )
                cp = tr.stack([cp_x, cp_y], dim=-1)
                cp = cp.unsqueeze(1)
                assert cp.ndim == 5
                x = self.splines[name](cp=cp, cp_are_logits=False)
                x = tr.swapaxes(x, 1, 2)
            else:
                x = self.out_temp[name](latent)

            if not tp.is_spline and tp.use_alpha_noise and alpha_noise is not None:
                sigma = alpha_noise * self.noise_std
                noise = tr.empty(x.shape, device=x.device).normal_(std=sigma)
                x += noise

            x = self.out_temp_acts[name](x)

            if self.filter_cf_hz is not None:
                h = self.filter
                x = tr.swapaxes(x, 1, 2)
                x = F.pad(x, (self.n_pad, self.n_pad, 0, 0), mode="replicate")
                x = F.conv1d(x, h, padding="valid")
                x = tr.swapaxes(x, 1, 2)

            if self.interp_n_frames is not None:
                assert False
                assert self.filter_cf_hz is None
                x = util.interpolate_dim(x, self.interp_n_frames, dim=1)

            out_dict[name] = x.squeeze(-1)

            pos_enc = self.pos_enc.expand(x.size(0), -1, -1)
            if tp.adapt_dim:
                adapt_in = tr.cat([x, pos_enc], dim=-1)
                # if tp.adapt_use_latent:
                #     assert chunks is not None
                #     # TODO(cm): look into this
                #     adapt_latent = util.interpolate_dim(chunks, self.n_frames, dim=1)
                #     # adapt_latent = util.interpolate_dim(chunks, self.n_frames, dim=1, mode="nearest", align_corners=None)
                #     adapt_in = tr.cat([adapt_in, adapt_latent], dim=-1)
                if tp.adapt_use_separate:
                    adapt_outs = []
                    for dim_idx in range(tp.adapt_dim):
                        adapter = self.adapters[f"{name}_{dim_idx}"]
                        adapt_out = adapter(adapt_in)
                        adapt_outs.append(adapt_out)
                    adapt_out = tr.cat(adapt_outs, dim=-1)
                else:
                    adapt_out = self.adapters[name](adapt_in)
                adapt_out = self.adapter_acts[name](adapt_out)
                # if x.shape == adapt_out.shape:
                #     out_dict[name] = adapt_out.squeeze(-1)
                # else:
                #     out_dict[f"{name}_adapted"] = adapt_out.squeeze(-1)
                out_dict[f"{name}_adapted"] = adapt_out.squeeze(-1)

        return out_dict

    @staticmethod
    def process_bezier_logits_x(
        x: T,
        n_segments: int,
        degree: int,
        alpha_noise: Optional[float] = None,
        noise_std: Optional[float] = None,
        alpha_linear: Optional[float] = None,
        softmax_tau: float = 0.25,
    ) -> T:
        assert x.ndim == 2
        assert x.size(1) == n_segments * degree + 1
        x = x[:, 1:]  # Remove first control point since it'll be zero

        if alpha_noise is not None:
            assert noise_std is not None
            assert 0.0 <= alpha_noise <= 1.0
            sigma = alpha_noise * noise_std
            noise = tr.empty_like(x).normal_(std=sigma)
            x += noise

        x = util.stable_softmax(x, softmax_tau)
        x = tr.cumsum(x, dim=1)
        x = F.pad(x, (1, 0), mode="constant", value=0.0)
        x[:, -1] = 1.0

        x = x.unfold(dimension=1, size=degree + 1, step=degree)

        if alpha_linear is not None and degree > 1:
            assert 0.0 <= alpha_linear <= 1.0
            x_linear = x[:, :, [0, -1]]
            x_linear = util.interpolate_dim(
                x_linear, degree + 1, dim=2, align_corners=True
            )
            x = (alpha_linear * x_linear) + ((1.0 - alpha_linear) * x)

        return x

    @staticmethod
    def process_bezier_logits_y(
        x: T,
        n_segments: int,
        degree: int,
        alpha_noise: Optional[float] = None,
        noise_std: Optional[float] = None,
        alpha_linear: Optional[float] = None,
        is_bounded: bool = False,
        eps: float = 1e-3,
    ) -> T:
        assert x.ndim == 2
        assert x.size(1) == n_segments * degree + 1

        if alpha_noise is not None:
            assert noise_std is not None
            assert 0.0 <= alpha_noise <= 1.0
            sigma = alpha_noise * noise_std
            noise = tr.empty_like(x).normal_(std=sigma)
            x += noise

        if is_bounded:
            x = tr.tanh(x) * (1.0 - eps)
            x = x * 0.5 + 0.5

        x = x.unfold(dimension=1, size=degree + 1, step=degree)

        if alpha_linear is not None and degree > 1:
            assert 0.0 <= alpha_linear <= 1.0
            x_linear = x[:, :, [0, -1]]
            x_linear = util.interpolate_dim(
                x_linear, degree + 1, dim=2, align_corners=True
            )
            x = (alpha_linear * x_linear) + ((1.0 - alpha_linear) * x)

        return x

    @staticmethod
    def calc_receptive_field(kernel_size: int, dilations: List[int]) -> int:
        """Compute the receptive field in samples."""
        assert dilations
        assert dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = kernel_size
        for dil in dilations[1:]:
            rf = rf + ((kernel_size - 1) * dil)
        return rf


# Baselines models below here ==========================================================
class RandomModSigModel(nn.Module):
    def __init__(
        self,
        n_frames: int,
        temp_params: Dict[str, TempParam],
        mod_sig_gen: ModSignalGenerator,
        dropout: float = 0.0,
        fc_act: str = "prelu",
    ):
        super().__init__()
        self.n_frames = n_frames
        self.temp_params = temp_params
        self.mod_sig_gen = mod_sig_gen
        self.dropout = dropout
        self.fc_act = fc_act

        self.register_buffer(
            "pos_enc", tr.linspace(0, 1, n_frames).view(1, -1, 1), persistent=False
        )
        self.adapters = nn.ModuleDict()
        self.adapter_acts = nn.ModuleDict()
        for name, tp in temp_params.items():
            if tp.adapt_dim:
                assert not tp.adapt_use_latent
                adapt_in_dim = tp.dim + 1
                n_hidden = (adapt_in_dim + tp.adapt_dim) // 2
                if tp.adapt_use_separate:
                    for dim_idx in range(tp.adapt_dim):
                        self.adapters[f"{name}_{dim_idx}"] = nn.Sequential(
                            nn.Linear(adapt_in_dim, n_hidden),
                            nn.Dropout(p=dropout),
                            util.get_activation(self.fc_act),
                            nn.Linear(n_hidden, n_hidden),
                            nn.Dropout(p=dropout),
                            util.get_activation(self.fc_act),
                            nn.Linear(n_hidden, 1),
                        )
                else:
                    self.adapters[name] = nn.Sequential(
                        nn.Linear(adapt_in_dim, n_hidden),
                        nn.Dropout(p=dropout),
                        util.get_activation(self.fc_act),
                        nn.Linear(n_hidden, n_hidden),
                        nn.Dropout(p=dropout),
                        util.get_activation(self.fc_act),
                        nn.Linear(n_hidden, tp.adapt_dim),
                    )
                self.adapter_acts[name] = util.get_activation(tp.adapt_act)

    def forward(
        self,
        x: Dict[str, T],
        tp_name: Optional[str] = None,
        alpha_noise: Optional[float] = None,
        alpha_linear: Optional[float] = None,
    ) -> Dict[str, T]:
        out_dict = {}
        audio = x["audio"]
        bs = audio.size(0)
        for name, tp in self.temp_params.items():
            if tp_name is not None and name != tp_name:
                continue
            mod_sigs = []
            for idx in range(tp.dim * bs):
                mod_sig = self.mod_sig_gen(self.n_frames)
                mod_sigs.append(mod_sig)
            mod_sig = tr.stack(mod_sigs, dim=0)
            mod_sig = mod_sig.view(bs, -1, tp.dim)
            out_dict[name] = mod_sig.squeeze(-1)

            pos_enc = self.pos_enc.expand(bs, -1, -1)
            if tp.adapt_dim:
                adapt_in = tr.cat([mod_sig, pos_enc], dim=-1)
                if tp.adapt_use_separate:
                    adapt_outs = []
                    for dim_idx in range(tp.adapt_dim):
                        adapter = self.adapters[f"{name}_{dim_idx}"]
                        adapt_out = adapter(adapt_in)
                        adapt_outs.append(adapt_out)
                    adapt_out = tr.cat(adapt_outs, dim=-1)
                else:
                    adapt_out = self.adapters[name](adapt_in)
                adapt_out = self.adapter_acts[name](adapt_out)
                out_dict[f"{name}_adapted"] = adapt_out.squeeze(-1)
        return out_dict


def mlp(in_size: int, hidden_size: int, n_layers: int) -> nn.Sequential:
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input: int, hidden_size: int) -> nn.GRU:
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


class DDSPSimpleModel(nn.Module):
    def __init__(
        self,
        fe: LogMelSpecFeatureExtractor,
        hidden_size: int,
        n_harmonic: int,
        n_bands: int,
        sampling_rate: int,
        block_size: int,
    ):
        super().__init__()
        self.register_buffer("sampling_rate", tr.tensor(sampling_rate))
        self.register_buffer("block_size", tr.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList(
            [
                nn.Linear(hidden_size, n_harmonic + 1),
                nn.Linear(hidden_size, n_bands),
            ]
        )

        self.register_buffer("cache_gru", tr.zeros(1, 1, hidden_size))
        self.register_buffer("phase", tr.zeros(1))

        self.loudness_extractor = LoudnessExtractor(
            sr=fe.sr, n_fft=fe.n_fft, hop_len=fe.hop_len
        )

    def forward(self, x: Dict[str, T]) -> Dict[str, T]:
        audio, pitch = x["audio"], x["f0_hz"]
        pitch = pitch.unsqueeze(-1).unsqueeze(-1)

        num_blocks = audio.shape[-1] // self.block_size
        pitch = pitch.expand(pitch.size(0), num_blocks, 1)
        loudness = self.loudness_extractor(audio.squeeze(1))
        loudness = loudness.unsqueeze(-1)

        loudness = util.interpolate_dim(loudness, num_blocks, dim=1)

        hidden = tr.cat(
            [
                self.in_mlps[0](pitch),
                self.in_mlps[1](loudness),
            ],
            -1,
        )
        hidden = tr.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        harmonic_param = self.proj_matrices[0](hidden)
        harmonic_param = tr.sigmoid(harmonic_param)

        # noise part
        noise_param = self.proj_matrices[1](hidden)
        noise_param = tr.sigmoid(noise_param)

        out_dict = {}
        out_dict["add_lfo"] = tr.cat([harmonic_param, noise_param], dim=-1)

        return out_dict


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
            "a_weighting",
            tr.from_numpy(a_weighting).view(1, -1, 1).float(),
            persistent=False,
        )
        self.register_buffer(
            "hann", tr.hann_window(n_fft, periodic=True), persistent=False
        )

    def forward(self, x: T) -> T:
        assert x.ndim == 2
        x = tr.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=self.hann,
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
