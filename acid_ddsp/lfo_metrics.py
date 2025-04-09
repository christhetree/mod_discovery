import abc
import logging
import os
from abc import abstractmethod
from typing import Literal

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class LFORangeMetric(nn.Module):
    def __init__(
        self,
        agg_fn: Literal["mean", "min", "max", "min_val", "max_val"] = "mean",
    ):
        super().__init__()
        self.agg_fn = agg_fn

    def forward(self, x: T) -> T:
        assert x.ndim == 2
        if self.agg_fn == "min_val":
            return x.min()
        elif self.agg_fn == "max_val":
            return x.max()
        x_min = x.min(dim=1).values
        x_max = x.max(dim=1).values
        x_range = x_max - x_min
        if self.agg_fn == "mean":
            return x_range.mean()
        elif self.agg_fn == "max":
            return x_range.max()
        elif self.agg_fn == "min":
            return x_range.min()
        else:
            raise ValueError(f"Unknown agg_fn: {self.agg_fn}")


class LFOMetric(nn.Module, abc.ABC):
    def __init__(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    @abstractmethod
    def calc_metric(self, x: T) -> T:
        pass

    def forward(self, x: T) -> T:
        assert x.ndim == 2
        if self.normalize:
            x_min = x.min(dim=1, keepdim=True).values
            x_max = x.max(dim=1, keepdim=True).values
            x_range = x_max - x_min
            x = (x - x_min) / (x_range + self.eps)
        metric = self.calc_metric(x)
        metric = metric.mean()
        return metric


class EntropyMetric(LFOMetric):
    @staticmethod
    def calc_entropy(x: T, normalize: bool = True) -> T:
        assert x.ndim == 2
        assert x.min() >= 0.0
        x = x / x.sum(dim=1, keepdim=True)
        x_log = tr.log(x)
        x_log = tr.nan_to_num(x_log, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -x * x_log
        entropy = entropy.sum(dim=1)
        if normalize:
            assert x.size(1) > 1
            entropy /= tr.log(tr.tensor(x.size(1)))
        return entropy

    def calc_metric(self, x: T) -> T:
        return self.calc_entropy(x, normalize=True)


class SpectralEntropyMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        mag_spec = tr.fft.rfft(x, dim=1).abs()
        entropy = EntropyMetric.calc_entropy(mag_spec, normalize=True)
        return entropy


class TotalVariationMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        assert x.size(1) > 1
        diffs = x[:, 1:] - x[:, :-1]
        diffs = tr.abs(diffs)
        diffs = diffs.mean(dim=1)
        return diffs


class TurningPointsMetric(LFOMetric):
    def calc_metric(self, x: T) -> T:
        assert x.size(1) > 2
        diffs = x[:, 1:] - x[:, :-1]
        diffs = tr.sign(diffs)
        ddiffs = diffs[:, 1:] * diffs[:, :-1]
        turning_points = (ddiffs < 0).sum(dim=1).float()
        turning_points = turning_points / (x.size(1) - 2)
        return turning_points


if __name__ == "__main__":
    # x = tr.tensor([1, 2, 3, 2, 1]).float().view(1, -1).repeat(1, 1)
    x = tr.tensor([0, -1, 1, 0]).float().view(1, -1).repeat(1, 1)
    # y = tr.tensor([1, 2, 3, 2, 1]).float().view(1, -1).repeat(1, 1)
    # y[1, :] -= 1

    # metric = EntropyMetric()
    metric = TotalVariationMetric()
    # metric = TurningPointsMetric()
    dist = metric(x)
    print(dist)
    exit()

    cos = tr.nn.functional.cosine_similarity(x, y, dim=-1)
    print(cos)

    pcc_s = []
    for idx in range(x.size(0)):
        curr_x = x[idx, :]
        curr_x_target = y[idx, :]
        data = tr.stack([curr_x, curr_x_target], dim=0)
        corr_matrix = tr.corrcoef(data)
        pcc = corr_matrix[0, 1]
        pcc_s.append(pcc)
    dist = tr.stack(pcc_s, dim=0)
    print(dist)
