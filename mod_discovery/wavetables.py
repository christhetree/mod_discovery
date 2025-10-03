import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

CONTINUOUS_ABLETON_WTS = [
    "basics__fm_fold__78_1024",
    "basics__galactica__4_1024",
    "basics__harmonic_series__7_1024",
    "basics__sub_3__122_1024",
    "collection__aureolin__256_1024",
    "collection__squash__32_1024",
    "complex__bit_ring__256_1024",
    "complex__kicked__4_1024",
    "distortion__dp_fold__230_1024",
    "distortion__phased__178_1024",
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
]
