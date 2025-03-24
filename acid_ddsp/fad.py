import logging
import os
import shutil
from typing import Optional

import torch as tr
import torchaudio
from torch import Tensor as T

import fadtk

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def get_fad_model(model_name: str) -> fadtk.ModelLoader:
    fad_models = {
        "vggish": fadtk.VGGishModel("2023"),
        "clap-2023": fadtk.CLAPModel("2023"),
        # "clap-laion-audio": fadtk.CLAPLaionModel("audio"),
        # "clap-laion-music": fadtk.CLAPLaionModel("music"),
        "encodec-emb-24k": fadtk.EncodecEmbModel("24k"),
        "encodec-emb-48k": fadtk.EncodecEmbModel("48k"),
        "panns-cnn14-16k": fadtk.PANNsModel("cnn14-16k"),
        "panns-cnn14-32k": fadtk.PANNsModel("cnn14-32k"),
        "panns-wavegram-logmel": fadtk.PANNsModel("wavegram-logmel"),
    }
    assert model_name in fad_models
    return fad_models[model_name]


def save_and_concat_fad_audio(
    sr: int,
    audio: T,
    dir_path: str,
    fade_n_samples: Optional[int] = None,
) -> None:
    assert audio.ndim == 2
    if os.path.exists(dir_path):
        log.warning(f"Directory '{dir_path}' already exists; removing")
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=False)
    if fade_n_samples is not None:
        assert fade_n_samples > 0
        assert fade_n_samples < audio.size(-1)
        audio[:, :fade_n_samples] *= tr.linspace(0.0, 1.0, fade_n_samples)
        audio[:, -fade_n_samples:] *= tr.linspace(1.0, 0.0, fade_n_samples)
    audio = tr.flatten(audio).unsqueeze(0)
    torchaudio.save(os.path.join(dir_path, "audio.wav"), audio, sr)


def calc_fad(
    fad_model_name: str,
    baseline_dir: str,
    eval_dir: str,
    workers: int = 1,
) -> float:
    assert os.path.isdir(baseline_dir)
    assert os.path.isdir(eval_dir)

    fad_model = get_fad_model(fad_model_name)
    fadtk.cache_embedding_files(baseline_dir, fad_model, workers)
    fadtk.cache_embedding_files(eval_dir, fad_model, workers)

    fad = fadtk.FrechetAudioDistance(
        fad_model, audio_load_worker=workers, load_model=False
    )
    score = fad.score(baseline_dir, eval_dir)
    return score
