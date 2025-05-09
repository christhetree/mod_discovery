import logging
import os

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioConfig:
    def __init__(
        self,
        sr: int = 48000,
        buffer_size_seconds: float = 0.125,
        note_on_duration: float = 0.100,
        min_f0_hz: float = 32.70,
        max_f0_hz: float = 523.25,
    ):
        self.sr = sr
        self.buffer_size_seconds = buffer_size_seconds
        self.note_on_duration = note_on_duration
        self.min_f0_hz = min_f0_hz
        self.max_f0_hz = max_f0_hz

        self.n_samples = int(sr * buffer_size_seconds)
        self.note_off = note_on_duration / buffer_size_seconds

        self.min_vals = {
            "f0_hz": min_f0_hz,
        }
        self.max_vals = {
            "f0_hz": max_f0_hz,
        }
        for param_name in self.min_vals.keys():
            assert self.min_vals[param_name] <= self.max_vals[param_name]

    def calc_w(self, w_hz: float) -> float:
        return 2 * tr.pi * w_hz / self.sr

    def is_fixed(self, param_name: str) -> bool:
        return self.min_vals[param_name] == self.max_vals[param_name]

    def convert_from_0to1(self, param_name: str, val: T) -> T:
        assert val.min() >= 0.0
        assert val.max() <= 1.0
        return (
            val * (self.max_vals[param_name] - self.min_vals[param_name])
        ) + self.min_vals[param_name]

    def convert_to_0to1(self, param_name: str, val: T) -> T:
        assert val.min() >= self.min_vals[param_name]
        assert val.max() <= self.max_vals[param_name]
        if self.is_fixed(param_name):
            return tr.zeros_like(val)
        return (val - self.min_vals[param_name]) / (
            self.max_vals[param_name] - self.min_vals[param_name]
        )
