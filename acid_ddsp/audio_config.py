import logging
import os

import torch as tr

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioConfig:
    def __init__(self,
                 sr: int,
                 control_rate: int,
                 buffer_size_seconds: float,
                 note_on_duration: float,
                 min_attack: float,
                 max_attack: float,
                 min_decay: float,
                 max_decay: float,
                 min_sustain: float,
                 max_sustain: float,
                 min_release: float,
                 max_release: float,
                 min_alpha: float,
                 max_alpha: float,
                 min_midi_f0: int,
                 max_midi_f0: int,
                 min_w_hz: float,
                 max_w_hz: float,
                 min_q: float = 0.7071,
                 max_q: float = 8.0,
                 stability_eps: float = 0.001):
        self.sr = sr
        self.control_rate = control_rate
        self.buffer_size_seconds = buffer_size_seconds
        self.note_on_duration = note_on_duration
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.min_sustain = min_sustain
        self.max_sustain = max_sustain
        self.min_release = min_release
        self.max_release = max_release
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.min_w_hz = min_w_hz
        self.max_w_hz = max_w_hz
        self.min_q = min_q
        self.max_q = max_q
        self.stability_eps = stability_eps

        self.min_w = self.calc_w(min_w_hz)
        self.max_w = self.calc_w(max_w_hz)

    def calc_w(self, w_hz: float) -> float:
        return 2 * tr.pi * w_hz / self.sr
