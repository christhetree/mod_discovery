"""
Render modulation curve from Vital based on given points and power value.
Point values must be between 0 and 1.
"""

import os
from typing import List

import torch
import torch as tr

from paths import DATA_DIR, OUT_DIR


def power_scale(value, power):
    """
    PyTorch implementation of Vital's power_scale function:
    https://github.com/mtytel/vital/blob/main/src/synthesis/framework/futils.h#L360
    """
    kMinPower = 0.01
    # mask to avoid division by zero
    mask = torch.abs(power) < kMinPower

    numerator = torch.exp(power * value) - 1.0
    denominator = torch.exp(power) - 1.0
    result = numerator / denominator

    return torch.where(mask, value, result)


def render(curve, resolution=2048, epsilon=1e-6):
    """
    PyTorch implementation of Vital's render function of LineGenerator:
    https://github.com/mtytel/vital/blob/main/src/common/line_generator.cpp#L165

    Vital's implementation always misses the first sample of a segment.
    Reason is that `point_index` increments only at the end of `x > current_point[0]`,
    after the sample is already written to the buffer.

    Wheres in this torch implementation we use `torch.searchsorted` to assign the segment index
    to each sample in the buffer. This way we can calculate the correct `t` value for each sample.
    """
    buffer = torch.zeros(resolution)
    curve_points = torch.tensor(curve["points"]).view(-1, 2)
    powers = torch.tensor(curve["powers"])

    x_vals = torch.linspace(0, 1, resolution)
    indices = torch.searchsorted(curve_points[:, 0].contiguous(), x_vals) - 1
    indices = torch.clamp(indices, 0, len(curve_points) - 2)

    last_points = curve_points[indices]
    next_points = curve_points[indices + 1]

    t_values = (x_vals - last_points[:, 0]) / (next_points[:, 0] - last_points[:, 0])
    t_values = torch.clamp(t_values, min=epsilon)

    power_indices = indices % len(powers)
    scaled_t = power_scale(t_values, powers[power_indices])
    scaled_t = torch.clamp(scaled_t, 0, 1)

    buffer = last_points[:, 1] + scaled_t * (next_points[:, 1] - last_points[:, 1])
    return buffer


def calc_range_and_pct_flat(points: List[float], eps: float = 0.01) -> (float, float):
    assert len(points) >= 2
    assert len(points) % 2 == 0
    points_x = points[0::2]
    points_y = points[1::2]
    assert points_x[0] == 0.0
    assert points_x[-1] == 1.0, f"points_x[-1]: {points_x[-1]} != 1.0"
    min_y = min(points_y)
    max_y = max(points_y)
    range_y = max_y - min_y
    assert min_y >= 0.0
    assert max_y <= 1.0
    flat_frac = 0.0
    for idx in range(len(points_y) - 1):
        curr_x = points_x[idx]
        next_x = points_x[idx + 1]
        curr_y = points_y[idx]
        next_y = points_y[idx + 1]
        if abs(curr_y - next_y) < eps:
            flat_frac += next_x - curr_x

    return range_y, flat_frac


if __name__ == "__main__":
    import json

    curves_path = os.path.join(DATA_DIR, "vital_curves.json")
    # curves_path = os.path.join(DATA_DIR, "vital_envelopes.json")
    with open(curves_path, "r") as f:
        curves = json.load(f)

    resolution = 1501
    counter = 0

    rendered_curves = []

    for curve in curves:
        n_points = curve["num_points"]
        points = curve["points"]
        y_range, flat_frac = calc_range_and_pct_flat(points)
        if n_points > 25:
            continue
        if y_range < 0.5:
            continue
        if flat_frac > 0.5:
            continue

        plot = render(curve, resolution=resolution)

        x = plot.unsqueeze(0)
        diffs = x[:, 1:] - x[:, :-1]
        diffs = tr.sign(diffs)
        ddiffs = diffs[:, 1:] * diffs[:, :-1]
        turning_points = (ddiffs < 0).sum(dim=1).float()
        turning_points = turning_points.squeeze().long().item()
        print(
            f"n_points: {n_points}, turning_points: {turning_points}, y_range: {y_range:.3f}, flat_frac: {flat_frac:.3f}"
        )

        # plot = plot.numpy()
        # import matplotlib.pyplot as plt
        # plt.plot(plot)
        # plt.title(f"n_points: {n_points}, turning_points: {turning_points}, y_range: {y_range:.3f}, flat_frac: {flat_frac:.3f}")
        # plt.ylim(0.0, 1.0)
        # plt.show()
        # plt.pause(0.20)

        rendered_curves.append(plot)
        counter += 1

    c = torch.stack(rendered_curves, dim=0)
    tr.save(c, os.path.join(OUT_DIR, f"vital_curves__{c.size(0)}_{c.size(1)}.pt"))
    print(f"Finished rendering {counter} curves.")
