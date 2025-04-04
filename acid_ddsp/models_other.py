import logging
import os

import numpy as np
import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self) -> None:
        with tr.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: T) -> T:
        return tr.sin(self.omega_0 * self.linear(input))


class EncDecLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_out_frames: int,
        num_layers: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size must be even"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_out_frames = n_out_frames
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size,
            hidden_size // 2,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.dec = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False,
        )
        self.start_token = nn.Parameter(tr.zeros(1, hidden_size))

    def forward(self, x: T) -> T:
        enc_out, (enc_h, _) = self.lstm(x)
        emb = tr.cat((enc_h[-2], enc_h[-1]), dim=-1)
        dec_h_init = emb.unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_c_init = tr.zeros_like(dec_h_init)

        dec_in = self.start_token.expand(x.size(0), -1, -1)
        dec_hidden = (dec_h_init, dec_c_init)
        out_frames = []
        for _ in range(self.n_out_frames):
            dec_out, dec_hidden = self.dec(dec_in, dec_hidden)
            out_frames.append(dec_out)
            dec_in = dec_out
        out = tr.cat(out_frames, dim=1)
        return out


class LinearProjection(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = tr.swapaxes(x, 1, 2)
        x = self.linear(x)
        x = tr.swapaxes(x, 1, 2)
        return x
