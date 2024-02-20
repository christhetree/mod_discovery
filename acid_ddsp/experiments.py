import logging
import os

import torch as tr
import torchaudio

from acid_ddsp.filters import TimeVaryingBiquad
from acid_ddsp.synth_modules import ADSRValues
from synths import CustomSynth
from torchsynth.config import SynthConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    sr = 48000
    buffer_size_seconds = 0.125
    control_rate = sr // 100

    vco_shape = 1.0
    midi_f0s = [36, 36, 36, 36, 36, 48, 36, 36, 36, 36, 36, 48, 36, 49, 36, 36] * 4
    # midi_f0s = [m + 4 for m in midi_f0s]
    note_on_duration = 0.100
    adsr_vals = ADSRValues(attack=0.001,
                           decay=0.099,
                           sustain=0.5,
                           release=0.025,
                           alpha=2.0)
    min_f = 100.0
    max_f = 6000.0
    min_q = 6.0
    max_q = 6.0
    dist_gain = 2.0

    batch_size = len(midi_f0s)
    save_name = (
        f"midi_f0s_{'_'.join([str(m) for m in midi_f0s])}"
        f"__q_{max_q:.2f}"
        f"__f_{min_f:.0f}_{max_f:.0f}"
        f"__dist_{dist_gain:.2f}.wav"
    )

    min_w = 2 * tr.pi * min_f / sr
    max_w = 2 * tr.pi * max_f / sr
    tvb = TimeVaryingBiquad(min_w, max_w, min_q, max_q)

    sc = SynthConfig(
        batch_size=batch_size,
        sample_rate=sr,
        buffer_size_seconds=buffer_size_seconds,
        control_rate=control_rate,
        reproducible=False,
        no_grad=True,
        debug=True,
    )
    synth = CustomSynth(sc, vco_shape=vco_shape, adsr_vals=adsr_vals)

    midi_f0_t = tr.tensor(midi_f0s)
    note_on_duration_t = tr.full((batch_size,), note_on_duration)
    (audio, envelope), _, _ = synth(
        midi_f0=midi_f0_t, note_on_duration=note_on_duration_t
    )
    audio = tr.flatten(audio, start_dim=0).unsqueeze(0)
    audio *= 0.5
    envelope = tr.flatten(envelope, start_dim=0).unsqueeze(0)

    log.info(f"before filtering audio.abs().max() = {audio.abs().max()}")
    audio = tvb(audio, cutoff_mod_sig=envelope)
    log.info(f"after filtering audio.abs().max() = {audio.abs().max()}")
    if dist_gain > 0:
        audio = tr.tanh(dist_gain * audio)
    log.info(f"after dist audio.abs().max() = {audio.abs().max()}")

    torchaudio.save(f"../out/{save_name}", audio, sr)
