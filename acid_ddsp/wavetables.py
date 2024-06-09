import logging
import os

import torch as tr

from paths import DATA_DIR
from plotting import plot_wavetable

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def create_wavetable(n_wt_samples: int, save_path: str):
    wt_0 = tr.sin(tr.linspace(0.0, 2 * tr.pi, n_wt_samples))
    # wt_1 = tr.sin(tr.linspace(0.0, 4 * tr.pi, n_wt_samples))
    # wt_2 = tr.linspace(-1.0, 1.0, n_wt_samples)
    wt = tr.stack([wt_0], dim=0)
    fig = plot_wavetable(wt)
    tr.save(wt, save_path)


if __name__ == "__main__":
    wt_dir = os.path.join(DATA_DIR, "wavetables")
    # wt_name = "test_wt.pt"
    wt_name = "sines_1_1024.pt"
    wt_path = os.path.join(wt_dir, wt_name)
    create_wavetable(1024, wt_path)
