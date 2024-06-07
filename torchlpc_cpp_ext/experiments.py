import logging
import os
import time

import torch.utils.cpp_extension

from filters import sample_wise_lpc_scriptable
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    log.info(f"torch number of threads: {torch.get_num_threads()}")
    os.environ["OMP_NUM_THREADS"] = f"{torch.get_num_threads()}"
    # torch.set_num_threads(1)

    torch.utils.cpp_extension.load(
        name="forward",
        sources=["torchlpc.cpp"],
        is_python_module=False,
        verbose=True
    )

    # torch.ops.load_library("build/torchlpc.so")

    # T = 10
    # T = 2000
    T = 100000
    # T = 200000
    bs = 2
    order = 3

    x = torch.randn(bs, T, dtype=torch.double)
    a = torch.randn(bs, T, order, dtype=torch.double)
    zi = torch.randn(bs, order, dtype=torch.double)

    log.info("Testing")
    start = time.time()
    og = sample_wise_lpc(x, a, zi)
    end = time.time()
    log.info(f"Testing sample_wise_lpc: {end - start:.4f}s")
    # start = time.time()
    # scriptable = sample_wise_lpc_scriptable(x, a, zi)
    # end = time.time()
    # log.info(f"Testing sample_wise_lpc_scriptable: {end - start:.4f}s")
    start = time.time()
    cpp = torch.ops.torchlpc.forward(x, a, zi)
    end = time.time()
    log.info(f"Testing sample_wise_lpc_cpp: {end - start:.4f}s")

    eps = 1e-8
    # log.info(torch.allclose(og, scriptable, atol=eps))
    log.info(torch.allclose(og, cpp, atol=eps))
