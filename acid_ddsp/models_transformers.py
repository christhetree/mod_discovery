import logging
import math
import os
from typing import Literal

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class PositionalEncoding(nn.Module):
    def __init__(
        self, size: int, num_pos: int, init: Literal["zeros", "norm0.02"] = "zeros"
    ):
        super().__init__()

        if init == "zeros":
            pe = tr.zeros(1, num_pos, size)
        else:
            pe = tr.randn(1, num_pos, size) * 0.02

        self.pe = nn.Parameter(pe)

    def forward(self, x: T) -> T:
        pe = self.pe[:, : x.size(1), :]
        return x + pe


class PatchEmbed(nn.Module):
    """Convolutional patch encoder like in ViT, with overlap from AST.
    Difference is we zero pad up to next whole patch.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        d_model: int,
        spec_shape: (int, int) = (128, 401),
    ):
        super().__init__()
        assert stride < patch_size, "Overlap must be less than patch size"

        self.patch_size = patch_size

        freq_padding = (stride - (spec_shape[0] - patch_size)) % stride
        time_padding = (stride - (spec_shape[1] - patch_size)) % stride

        self.pad = nn.ZeroPad2d((0, freq_padding, 0, time_padding))
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        self.num_tokens = self._get_num_tokens(in_channels, spec_shape)

    def _get_num_tokens(self, in_channels: int, spec_shape: (int, int)) -> int:
        x = tr.randn(1, in_channels, *spec_shape, device=self.projection.weight.device)
        out_shape = self.projection(self.pad(x)).shape
        return math.prod(out_shape[-2:])

    def forward(self, x: T) -> T:
        x = self.pad(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AudioSpectrogramTransformer(nn.Module):
    """Based on the AST from https://arxiv.org/abs/2104.01778, but adapted to pre-norm
    transformer.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 16,
        n_embed_tokens: int = 1,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 2,
        spec_shape: (int, int) = (128, 401),
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_embed_tokens = n_embed_tokens
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_channels = input_channels
        self.spec_shape = spec_shape

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            stride=patch_stride,
            in_channels=input_channels,
            d_model=d_model,
            spec_shape=spec_shape,
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            self.patch_embed.num_tokens + n_embed_tokens,
            init="norm0.02",
        )
        self.embed_tokens = None
        if n_embed_tokens > 0:
            self.embed_tokens = nn.Parameter(
                tr.empty(1, n_embed_tokens, d_model).normal_(mean=0.0, std=1e-6)
            )

        blocks = [
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: T) -> T:
        # produce input sequence
        x = self.patch_embed(x)

        if self.n_embed_tokens > 0:
            embed_tokens = self.embed_tokens.expand(x.shape[0], -1, -1)
            x = tr.cat((embed_tokens, x), dim=1)

        x = self.positional_encoding(x)

        # apply transformer
        x = self.blocks(x)

        # take just the embed tokens
        if self.n_embed_tokens > 0:
            x = x[:, : self.embed_tokens.size(1)]

        x = self.out_proj(x)
        return x


class ASTWithProjectionHead(AudioSpectrogramTransformer):
    """Extends the AST with an MLP prediction head."""

    def __init__(
        self,
        d_model: int = 768,
        d_out: int = 16,
        n_heads: int = 8,
        n_layers: int = 16,
        n_embed_tokens: int = 1,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 2,
        spec_shape: (int, int) = (128, 401),
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_embed_tokens=n_embed_tokens,
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=input_channels,
            spec_shape=spec_shape,
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_out),
        )

    def forward(self, x: T) -> T:
        return super().forward(x).squeeze(1)


if __name__ == "__main__":
    # Test ASTWithProjectionHead
    model = ASTWithProjectionHead(n_embed_tokens=3)
    x = tr.randn(1, 2, 128, 401)
    y = model(x)
    print(y.shape)
    # torch.Size([1, 16])
