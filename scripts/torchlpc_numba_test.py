import logging
import os

import torch as tr

from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    n_ch = 2
    n_samples = 16000
    audio = tr.rand(n_ch, n_samples)
    a_coeff = tr.rand(n_ch, n_samples, 3)
    if tr.cuda.is_available():
        log.info("CUDA is available. Running on GPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        audio = audio.cuda()
        a_coeff = a_coeff.cuda()
    else:
        log.warning("CUDA is not available. Running on CPU.")
    x = sample_wise_lpc(audio, a_coeff)
    log.info(f"x.shape: {x.shape}")
