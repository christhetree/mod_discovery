import logging
import os
from typing import Optional

import numpy as np
import torch as tr
from kymatio.torch import TimeFrequencyScattering, Scattering1D
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MFCC

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MFCCL1(nn.Module):
    def __init__(
        self,
        sr: int,
        log_mels: bool = True,
        n_fft: int = 2048,
        hop_len: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        self.mfcc = MFCC(
            sample_rate=sr,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_len,
                "n_mels": n_mels,
            },
        )
        self.l1 = nn.L1Loss()

    def forward(self, x_hat: T, x: T) -> T:
        return self.l1(self.mfcc(x_hat), self.mfcc(x))


class Scat1DLoss(nn.Module):
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        T: Optional[str | int] = None,
        max_order: int = 2,
        p: int = 2,
        # use_o2_only: bool = False,
    ):
        super().__init__()
        # if use_o2_only:
        #     assert max_order == 2
        self.max_order = max_order
        self.p = p
        # self.use_o2_only = use_o2_only
        self.scat_1d = Scattering1D(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            T=T,
            max_order=max_order,
        )
        meta = self.scat_1d.meta()
        # self.o2_cf_hz = {idx: round(xi[1] * 16000, 6) for idx, xi in enumerate(meta["xi"]) if not np.isnan(xi[1])}
        # self.o2_indices = [idx for idx, order in enumerate(meta["order"]) if order == 2]
        # # self.o2_indices = [idx for idx, cf_hz in self.o2_cf_hz.items() if cf_hz < 5]
        # log.info(f"number of o2_indices = {len(self.o2_indices)}")

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.scat_1d(x)
        Sx_target = self.scat_1d(x_target)
        # if self.use_o2_only:
        #     Sx = Sx[:, :, self.o2_indices, :]
        #     Sx_target = Sx_target[:, :, self.o2_indices, :]
        # else:
        Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
        Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients

        if self.max_order == 1:
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        else:
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=-1)

        dist = tr.mean(dist)
        return dist


class JTFSLoss(nn.Module):
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: Optional[str | int] = None,
        F: Optional[str | int] = None,
        format_: str = "joint",
        p: int = 2,
    ):
        super().__init__()
        assert format_ in ["time", "joint"]
        self.format = format_
        self.p = p
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format_,
        )
        jtfs_meta = self.jtfs.meta()
        jtfs_keys = [key for key in jtfs_meta["key"] if len(key) == 2]
        log.info(f"number of JTFS keys = {len(jtfs_keys)}")

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        Sx_target = self.jtfs(x_target)
        if self.format == "time":
            Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
            Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=-1)
        else:
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist


class ESRLoss(nn.Module):
    """Error-to-signal ratio loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: T, target: T) -> T:
        num = ((target - input) ** 2).sum(dim=-1)
        denom = (target ** 2).sum(dim=-1) + self.eps
        losses = num / denom
        losses = self.apply_reduction(losses, reduction=self.reduction)
        return losses

    @staticmethod
    def apply_reduction(losses, reduction="none"):
        """Apply reduction to collection of losses."""
        if reduction == "mean":
            losses = losses.mean()
        elif reduction == "sum":
            losses = losses.sum()
        return losses


class FirstDerivativeL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input: T, target: T) -> T:
        input_prime = self.calc_first_derivative(input)
        target_prime = self.calc_first_derivative(target)
        loss = self.l1(input_prime, target_prime)
        return loss

    @staticmethod
    def calc_first_derivative(x: T) -> T:
        assert x.size(-1) > 2
        return (x[..., 2:] - x[..., :-2]) / 2.0


class SecondDerivativeL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input: T, target: T) -> T:
        input_prime = self.calc_second_derivative(input)
        target_prime = self.calc_second_derivative(target)
        loss = self.l1(input_prime, target_prime)
        return loss

    @staticmethod
    def calc_second_derivative(x: T) -> T:
        d1 = FirstDerivativeL1Loss.calc_first_derivative(x)
        d2 = FirstDerivativeL1Loss.calc_first_derivative(d1)
        return d2
