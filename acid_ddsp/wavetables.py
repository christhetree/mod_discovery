import logging
import os

import torch as tr

from paths import DATA_DIR
from plotting import plot_wavetable

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def create_wavetable(n_wt_samples: int, save_path: str):
    wt_0 = tr.sin(tr.linspace(0.0, 2 * tr.pi, n_wt_samples))    # sine
    wt_1 = tr.cat([
        tr.linspace(0.0, 1.0, n_wt_samples // 4), 
        tr.linspace(1.0, -1.0, n_wt_samples // 2),
        tr.linspace(-1.0, 0.0, n_wt_samples // 4)]
    )  # triangle
    wt_2 = tr.linspace(-1.0, 1.0, n_wt_samples)                 # saw
    wt_3 = tr.cat([
        tr.ones(n_wt_samples // 2),
        -tr.ones(n_wt_samples // 2)
    ]) # square
    wt_4 = tr.cat([
        tr.ones(n_wt_samples // 4),
        -tr.ones(n_wt_samples // 4 * 3)
    ]) # pulse
    wt_5 = tr.cat([
        tr.linspace(1.0, 0.0, n_wt_samples // 2),
        tr.linspace(-1.0, 0.0, n_wt_samples // 2)
    ]) # serum like

    wt = tr.stack([wt_0, wt_1, wt_2, wt_3], dim=0)
    fig = plot_wavetable(wt)
    tr.save(wt, save_path)


if __name__ == "__main__":
    wt_dir = os.path.join(DATA_DIR, "wavetables")
    # wt_name = "test_wt.pt"
    # wt_name = "sine_saw_1024.pt"
    wt_name = "serum_basic_shapes_1024__n_pos_4.pt"
    wt_path = os.path.join(wt_dir, wt_name)
    create_wavetable(1024, wt_path)
