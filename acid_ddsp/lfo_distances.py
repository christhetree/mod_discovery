import logging
import os
from typing import Callable

import torch as tr
from dtw import dtw
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class COSSDistance(nn.Module):
    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        assert x.shape == x_target.shape
        dist = tr.nn.functional.cosine_similarity(x, x_target, dim=-1)
        return dist


class PCCDistance(nn.Module):
    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        assert x.shape == x_target.shape
        x_mean = x.mean(dim=-1, keepdim=True)
        x_target_mean = x_target.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_target = x_target - x_target_mean
        n_frames = x.size(1)
        cov = (x * x_target).sum(dim=-1) / (n_frames - 1)
        x_std = x.std(dim=1, unbiased=True)
        x_target_std = x_target.std(dim=1, unbiased=True)
        corr = cov / (x_std * x_target_std)
        dist = corr.mean()
        return dist


class DTWDistance(nn.Module):
    def __init__(self, dist_method: str = "cityblock") -> None:
        super().__init__()
        self.dist_method = dist_method

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        assert x.shape == x_target.shape
        bs = x.size(0)
        x = x.numpy()
        x_target = x_target.numpy()
        dists = []
        for idx in range(bs):
            curr_x = x[idx, :]
            curr_x_target = x_target[idx, :]
            dist = dtw(
                curr_x, curr_x_target, dist_method=self.dist_method, distance_only=True
            )
            dist = dist.normalizedDistance
            dists.append(dist)
        dist = tr.tensor(dists, dtype=tr.float)
        dist = dist.mean()
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


class FFTMagDist(nn.Module):
    def __init__(self, ignore_dc: bool = True, p: int = 1) -> None:
        super().__init__()
        self.ignore_dc = ignore_dc
        self.p = p

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        assert x.shape == x_target.shape
        X = tr.fft.rfft(x, dim=1).abs()
        X_target = tr.fft.rfft(x_target, dim=1).abs()
        if self.ignore_dc:
            X = X[:, 1:]
            X_target = X_target[:, 1:]
        if self.p == 1:
            dist = self.l1(X, X_target)
        elif self.p == 2:
            dist = self.mse(X, X_target)
        else:
            raise ValueError(f"Unknown p value: {self.p}")
        return dist


class FirstDerivativeDistance(nn.Module):
    def __init__(self, dist_fn: Callable[[T, T], T]) -> None:
        super().__init__()
        self.dist_fn = dist_fn

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        x_prime = self.calc_first_derivative(x)
        x_target_prime = self.calc_first_derivative(x_target)
        dist = self.dist_fn(x_prime, x_target_prime)
        return dist

    @staticmethod
    def calc_first_derivative(x: T) -> T:
        assert x.size(-1) > 2
        return (x[..., 2:] - x[..., :-2]) / 2.0


class SecondDerivativeDistance(nn.Module):
    def __init__(self, dist_fn: Callable[[T, T], T]) -> None:
        super().__init__()
        self.dist_fn = dist_fn

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 2
        x_pp = self.calc_second_derivative(x)
        x_target_pp = self.calc_second_derivative(x_target)
        dist = self.dist_fn(x_pp, x_target_pp)
        return dist

    @staticmethod
    def calc_second_derivative(x: T) -> T:
        d1 = FirstDerivativeDistance.calc_first_derivative(x)
        d2 = FirstDerivativeDistance.calc_first_derivative(d1)
        return d2


if __name__ == "__main__":
    n_frames = 1501
    t = tr.linspace(0.0, 2 * tr.pi, steps=n_frames)
    x = tr.sin(t)
    x_target = tr.roll(x, 750)
    # x_target = tr.roll(x, 0)
    x = x.view(1, -1).repeat(2, 1)
    x_target = x_target.view(1, -1).repeat(2, 1)
    # x_target[0, :] = x[0, :]
    # x_target[1, :] = x[1, :]
    dtw_dist = DTWDistance()
    # dist = dtw_dist(x, x_target)
    dist = dtw_dist(x_target, x)
    print(f"dist = {dist}")
