import logging
import os

import torch as tr
import torchaudio

from paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    dir_path = os.path.join(DATA_DIR, "ableton_wavetables")
    # dir_path = os.path.join(DATA_DIR, "waveedit_wavetables")
    ext = ".wav"
    samples_per_waveform = 1024
    # samples_per_waveform = 256

    for fname in os.listdir(dir_path):
        fname = fname.lower()
        if fname.endswith(ext):
            audio, sr = torchaudio.load(os.path.join(dir_path, fname))
            n_samples = audio.size(1)
            assert n_samples % samples_per_waveform == 0
            n_pos = n_samples // samples_per_waveform
            audio = audio.squeeze(0)
            log.info(f"Loaded {fname}, n_samples: {n_samples}, n_waveforms: {n_pos}")
            wt = tr.zeros(n_pos, samples_per_waveform)

            waveforms = []
            for idx in range(n_pos):
                waveform = audio[idx * samples_per_waveform: (idx + 1) * samples_per_waveform]
                assert waveform.min() >= -1.0
                assert waveform.max() <= 1.0
                waveforms.append(waveform)
                # log.info(
                #     f"{idx}, First val: {waveform[0]}, last val: {waveform[-1]}, Min: {waveform.min()}, Max: {waveform.max()}")
                # plt.plot(waveform)
                # plt.show()
            wt = tr.stack(waveforms, dim=0).float()

            fname = fname.replace(ext, "")
            fname = fname.replace("vector_sprite_", "")
            tokens = fname.split(" ")
            category = tokens[0].lower()
            name = "_".join(tokens[1:]).lower()
            save_name = f"{category}__{name}__{n_pos}_{samples_per_waveform}.pt"

            # name = fname.lower()
            # name = name.replace(" ", "_")
            # name = name.replace("___", "_")
            # name = name.replace("__", "_")
            # if name.endswith("_"):
            #     name = name[:-1]
            # save_name = f"{name}__{n_pos}_{samples_per_waveform}.pt"

            assert "__" not in name
            log.info(f"Saving to {save_name}")
            save_path = os.path.join(DATA_DIR, "wavetables", "ableton", save_name)
            # save_path = os.path.join(DATA_DIR, "wavetables", "waveedit", save_name)
            tr.save(wt, save_path)
