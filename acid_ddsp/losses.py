import logging
import os

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

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        return self.l1(self.mfcc(x), self.mfcc(x_target))


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
        denom = (target**2).sum(dim=-1) + self.eps
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
