import logging
import os
import shutil
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


def save_fad_audio(sr: int, audio_paths: List[str], wet: T, wet_hat: T) -> None:
    assert len(audio_paths) == wet.size(0) == wet_hat.size(0)
    baseline_dir = os.path.join(DATA_DIR, "fad_baseline")
    eval_dir = os.path.join(DATA_DIR, "fad_eval")
    if os.path.exists(baseline_dir):
        shutil.rmtree(baseline_dir)
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
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


def calc_fad(
    fad_model_name: str,
    baseline_dir: Optional[str] = None,
    eval_dir: Optional[str] = None,
    workers: int = 1,
) -> float:
    if baseline_dir is None:
        baseline_dir = os.path.join(DATA_DIR, "fad_baseline")
    if eval_dir is None:
        eval_dir = os.path.join(DATA_DIR, "fad_eval")
    assert os.path.isdir(baseline_dir)
    assert os.path.isdir(eval_dir)

    fad_models = {m.name: m for m in fadtk.get_all_models()}
    fad_model = fad_models[fad_model_name]
    cache_embedding_files(baseline_dir, fad_model, workers)
    cache_embedding_files(eval_dir, fad_model, workers)

    fad = FrechetAudioDistance(fad_model, audio_load_worker=workers, load_model=False)
    score = fad.score(baseline_dir, eval_dir)
    return score
