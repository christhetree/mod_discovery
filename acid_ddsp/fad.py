import logging
import os
import shutil
import time
from typing import Optional, List

import torch as tr
import torchaudio
from torch import Tensor as T

import fadtk
from fadtk import (
    FrechetAudioDistance,
    cache_embedding_files,
)
from paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def save_fad_audio(
    sr: int,
    audio_paths: List[str],
    wet: T,
    wet_hat: T,
    unique_suffix: Optional[str] = None,
) -> (str, str):
    if unique_suffix is None:
        unique_suffix = f"{os.getpid()}_{int(time.time())}"
    assert len(audio_paths) == wet.size(0) == wet_hat.size(0)
    baseline_dir = os.path.join(DATA_DIR, f"fad_baseline_{unique_suffix}")
    eval_dir = os.path.join(DATA_DIR, f"fad_eval_{unique_suffix}")
    os.makedirs(baseline_dir, exist_ok=False)
    os.makedirs(eval_dir, exist_ok=False)
    all_w = []
    all_w_hat = []
    for audio_path, w, w_hat in zip(audio_paths, wet, wet_hat):
        w = w.unsqueeze(0)
        w_hat = w_hat.unsqueeze(0)
        all_w.append(w)
        all_w_hat.append(w_hat)
    all_w = tr.cat(all_w, dim=1)
    all_w_hat = tr.cat(all_w_hat, dim=1)
    torchaudio.save(os.path.join(baseline_dir, "all.wav"), all_w, sr)
    torchaudio.save(os.path.join(eval_dir, "all.wav"), all_w_hat, sr)
    return baseline_dir, eval_dir


def calc_fad(
    fad_model_name: str,
    baseline_dir: str,
    eval_dir: str,
    clean_up: bool = True,
    workers: int = 0,
) -> float:
    assert os.path.isdir(baseline_dir)
    assert os.path.isdir(eval_dir)

    fad_models = {m.name: m for m in fadtk.get_all_models()}
    fad_model = fad_models[fad_model_name]
    cache_embedding_files(baseline_dir, fad_model, workers)
    cache_embedding_files(eval_dir, fad_model, workers)

    # TODO(cm): fix workers here
    fad = FrechetAudioDistance(
        fad_model, audio_load_worker=min(1, workers), load_model=False
    )
    score = fad.score(baseline_dir, eval_dir)
    if clean_up:
        shutil.rmtree(baseline_dir)
        shutil.rmtree(eval_dir)
    return score
