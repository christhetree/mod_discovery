import logging
import os

import torch as tr
import torchaudio

from acid_ddsp.filters import TimeVaryingBiquad
from synths import CustomSynth
from torchsynth.config import SynthConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    batch_size = 1
    sr = 24000
    buffer_size_seconds = 3.0
    control_rate = sr // 100
    vco_shape = 1.0
    midi_f0 = 60
    note_on_duration = buffer_size_seconds - 1.0

    min_f = 100.0
    max_f = 12000.0
    min_q = 4.7071
    max_q = 4.7071

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
    synth = CustomSynth(sc, vco_shape=vco_shape)

    midi_f0_t = tr.full((batch_size,), midi_f0)
    note_on_duration_t = tr.full((batch_size,), note_on_duration)
    (audio, envelope), _, _ = synth(
        midi_f0=midi_f0_t, note_on_duration=note_on_duration_t
    )

    n_samples = audio.size(1)
    freq_lfo = tr.linspace(0.0, 1.0, n_samples).unsqueeze(0).repeat(batch_size, 1)
    # res_lfo = tr.linspace(0.0, 1.0, n_samples).unsqueeze(0).repeat(batch_size, 1)

    audio = tvb(audio, freq_lfo)

    for idx in range(batch_size):
        log.info(f"idx = {idx}")
        torchaudio.save(f"../out/tmp_{idx}.wav", audio[idx].unsqueeze(0), sr)
        # curr_envelope = envelope[idx]
        # from matplotlib import pyplot as plt
        # plt.plot(curr_envelope.detach().numpy())
        # plt.show()
