import logging
import os
from typing import Optional

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class PiecewiseSplines(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_segments: int,
        degree: int,
    ):
        super().__init__()
        assert n_frames >= 2
        assert n_segments >= 1
        assert degree >= 1
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.degree = degree
        self.min_t = 0.0
        self.max_t = 1.0

        support = tr.linspace(self.min_t, self.max_t, n_frames).view(1, -1, 1, 1)
        support = support.repeat(1, 1, n_segments, degree)
        segment_offsets = tr.linspace(0.0, 1.0, n_segments + 1)[:-1].view(
            1, 1, -1, 1
        )
        support = support - segment_offsets
        support = tr.clamp(support, min=self.min_t, max=self.max_t)
        exponent = tr.arange(start=1, end=degree + 1).int().view(1, 1, 1, -1)
        support = support.pow(exponent)
        self.register_buffer("support", support)

    def forward(self, coeff: T, bias: Optional[T] = None) -> T:
        assert coeff.ndim == 3
        assert coeff.size(1) == self.n_segments
        assert coeff.size(2) == self.degree
        coeff = coeff.unsqueeze(1)
        x = coeff * self.support
        x = x.sum(dim=[2, 3])
        if bias is not None:
            bias = bias.view(x.size(0), 1)
            x = x + bias
        return x


if __name__ == "__main__":
    n_frames = 100

    coeff = tr.tensor([[1.0, -1.0], [2.0, 2.0]]).unsqueeze(0)
    log.info(f"coeff.shape: {coeff.shape}")
    n_segments = coeff.size(1)
    degree = coeff.size(2)

    bias = None
    bias = tr.tensor(0.5)

    curves = PiecewiseSplines(
        n_frames,
        n_segments,
        degree,
    )
    x = curves(coeff, bias)
    # x = tr.sigmoid(x)

    log.info(f"x.shape: {x.shape}")

    x = x[0].numpy()

    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.show()
