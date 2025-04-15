import logging
import os

import torch as tr

from paths import DATA_DIR
from plotting import plot_wavetable

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

BAD_ABLETON_WTS = [
    "basics__5th_brutal",
    "basics__beating_1",
    "basics__beating_2",
    "basics__beating_3",
    "basics__beating_4",
    "basics__beating_5",
    "basics__pulse_pw",
    "basics__saw_dual_1",
    "basics__saw_dual_2",
    "basics__saw_dual_3",
    "basics__saw_harmonics",
    "basics__saw_pw_detune",
    # "basics__sub_1",
    "basics__sync_additive",
    "basics__sync_digital",
    "basics__white_noise",

    # "collection__amber",
    # "collection__beige",
    # "collection__charcoal",
    # "collection__cobalt",
    # "collection__copper",
    # "collection__olive",
    # "collection__pearl",
    # "collection__rust",
    # "collection__slate",
    # "collection__violet",
    # "complex__bitkart",
    # "complex__bitten_filter",
    # "complex__menace",
    # "complex__noise_manipulator",
    # "complex__ring_mod",
    # "complex__ripped_sync",
    # "complex__void",
    # "complex__verbed",
    # "complex__xmod_drive",
    # "distortion__brutal_metal",
    # "distortion__clipped_sweep",
    # "distortion__clipped_freak",
    # "distortion__freak",
    # "distortion__jn60_bitter",
    # "distortion__malice",
    # "distortion__westcoast_fold",
    # "filter__dark_throaty",
    # "filter__frequency_fm",
    # "filter__modern_sweep_1",
    # "filter__modern_sweep_2",
    # "filter__super_phased",
    # "formant__aeiou",
    # "noise__",
]

BOTH_LFO_SERUM_WTS = [
    # "BA 8bitfwap",  # harsh, wt lfo
    "BA Access 2 Mthrshp Denied",  # harsh, both lfo
    # "BA Angggeerrrrr 1999",  # harsh, wt lfo
    # "BA Angle Grinder",  # harsh, wt lfo
    # "BA Bandpass 2",  # harsh, filter lfo
    "BA BitterBot",  # both lfo
    # "BA BotWerx",  # nasal, wt lfo
    # "BA Complextro 1",  # only env
    # "BA Complextro 2",  # only env
    # "BA DarkWobble",  # filter lfo
    "BA Deth reece",  # both lfo
    # "BA Digimods",   # filter lfo
    # "BA Downsampler",   # wt lfo
    # "BA Evolving Bass",  # filter lfo
    # "BA FuckButtons",  # harsh, wt lfo
    "BA Gritter",  # both lfo
    # "BA GRWL PNCH!",  # wt lfo
    "BA Hoo",  # harsh, both lfo
    # "BA Inertia Wobbler",  # filter lfo
    # "BA LazorBass",  # wt lfo, filter env
    "BA Le Gigante",  # harsh, both lfo
    # "BA Mean Rawr Bass",  # harsh, wt lfo
    "BA Modulated Chomper",  # both lfo
    # "BA No Effects",  # harsh, wt env, filter lfo
    # "BA Razor",  # harsh, only env
    # "BA Sawtoooooth II",  # harsh, only env
    # "BA Scream 4 Me",  # harsh, wt lfo
    "BA SCREAM Wobble 01",  # both lfo
    # "BA Sever Headz",  # harsh, wt lfo
    "BA Sludgecrank",  # harsh, both lfo
    # "BA Squerbo",  # filter lfo
    # "BA The Standard SC",  # wt lfo, eq lfo
    # "BA Throat Yeah",  # wt lfo
    # "BA Transformer Growl",  # wt lfo, filter env
    "BA Wide Eyed Reese",  # both lfo
    # "BA Y U Mod Wheel",  # wt lfo

    # "BA FM Wob",
]


def create_wavetable(n_wt_samples: int, save_path: str):
    wt_0 = tr.sin(tr.linspace(0.0, 2 * tr.pi, n_wt_samples))
    # wt_1 = tr.sin(tr.linspace(0.0, 4 * tr.pi, n_wt_samples))
    wt_2 = tr.linspace(-1.0, 1.0, n_wt_samples)
    # wt = tr.stack([wt_0], dim=0)
    # wt = tr.stack([wt_0, wt_1], dim=0)
    wt = tr.stack([wt_0, wt_2], dim=0)
    # fig = plot_wavetable(wt)
    tr.save(wt, save_path)


if __name__ == "__main__":
    wt_dir = os.path.join(DATA_DIR, "wavetables")
    # wt_name = "test_wt.pt"
    wt_name = "sine_saw_1024.pt"
    wt_path = os.path.join(wt_dir, wt_name)
    create_wavetable(1024, wt_path)
