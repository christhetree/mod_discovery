import logging
import os

from torchsynth.torchsynth.config import SynthConfig

from synths import CustomSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    batch_size = 3
    sr = 44100
    buffer_size_seconds = 4.0
    control_rate = sr // 100
    vco_shape = 1.0
    midi_f0 = 57

    sc = SynthConfig(batch_size=batch_size,
                     sample_rate=sr,
                     buffer_size_seconds=buffer_size_seconds,
                     control_rate=control_rate,
                     reproducible=False,
                     no_grad=True,
                     debug=True)
    synth = CustomSynth(sc, vco_shape=vco_shape)

    # idx = random.randint(0, n_batches)
    # log.info(f"idx = {idx}")
    # (audio, mod_sig), _, _ = synth(idx)
    #
    # plt.plot(mod_sig[0].detach().numpy())
    # plt.show()
    #
    # audio = audio[0].unsqueeze(0)
    # torchaudio.save("../out/tmp.wav", audio, sr)

    # from matplotlib import pyplot as plt
    # plt.plot(envelope[0].detach().numpy())
    # plt.show()
