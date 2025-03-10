import logging
import math
import os
from typing import Tuple, Literal

import torch as tr
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

    def forward(self, x: tr.Tensor) -> tr.Tensor:
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
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__()
        assert stride < patch_size, "Overlap must be less than patch size"

        self.patch_size = patch_size

        mel_padding = (stride - (spec_shape[0] - patch_size)) % stride
        time_padding = (stride - (spec_shape[1] - patch_size)) % stride

        self.pad = nn.ZeroPad2d((0, mel_padding, 0, time_padding))
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        self.num_tokens = self._get_num_tokens(in_channels, spec_shape)

    def _get_num_tokens(self, in_channels, spec_shape):
        x = tr.randn(1, in_channels, *spec_shape, device=self.projection.weight.device)
        out_shape = self.projection(self.pad(x)).shape
        return math.prod(out_shape[-2:])

    def forward(self, x: tr.Tensor) -> tr.Tensor:
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
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__()

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
        self.embed_tokens = nn.Parameter(
            tr.empty(1, n_embed_tokens, d_model).normal_(0.0, 1e-6)
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_model,
                    0.0,
                    "gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        # produce input sequence
        x = self.patch_embed(x)

        embed_tokens = self.embed_tokens.expand(x.shape[0], -1, -1)
        x = tr.cat((embed_tokens, x), dim=1)

        x = self.positional_encoding(x)

        # apply transformer
        for block in self.blocks:
            x = block(x)

        # take just the embed tokens
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
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 2,
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_embed_tokens=1,
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

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        return super().forward(x).squeeze(1)


if __name__ == "__main__":
    # Test ASTWithProjectionHead
    model = ASTWithProjectionHead()
    x = tr.randn(1, 2, 128, 401)
    y = model(x)
    print(y.shape)
    # torch.Size([1, 16])
