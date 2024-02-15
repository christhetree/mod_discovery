import logging
import os
import random

import torchaudio
from matplotlib import pyplot as plt
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

from synths import CustomSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    derp = Voice()

    n_batches = 100
    sr = 44100
    buffer_size_seconds = 2.0

    sc = SynthConfig(batch_size=1,
                     sample_rate=sr,
                     buffer_size_seconds=buffer_size_seconds,
                     reproducible=False)
    synth = CustomSynth(sc)

    idx = random.randint(0, n_batches)
    log.info(f"idx = {idx}")
    (audio, mod_sig), _, _ = synth(idx)

    plt.plot(mod_sig[0].detach().numpy())
    plt.show()

    audio = audio[0].unsqueeze(0)
    torchaudio.save("../out/tmp.wav", audio, sr)
