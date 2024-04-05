import logging
import os
import pathlib
from typing import Dict, List

import torch as tr
import torch.nn as nn
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from torch import Tensor as T

from audio_config import AudioConfig
from feature_extraction import LogMelSpecFeatureExtractor
from models import Spectral2DCNN
from paths import OUT_DIR
from synths import AcidSynthLPBiquad

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Spectral2DCNN(
            fe=LogMelSpecFeatureExtractor(),
            temp_params_name="w_mod_sig",
            global_param_names=[
                "q",
                "osc_shape",
                "osc_gain",
                "dist_gain",
                "learned_alpha",
            ],
        )
        self.ac = AudioConfig(buffer_size_seconds=0.125)
        self.synth = AcidSynthLPBiquad(self.ac, make_scriptable=True)
        # self.synth = AcidSynthLPBiquadFSM(self.ac, win_len=128, overlap=0.75, oversampling_factor=1)
        self.note_on_duration = tr.tensor([self.ac.note_on_duration])
        self.phase = tr.tensor([[0.0]])
        self.f0_hz = tr.tensor([220.0])

    def forward(self, x: T) -> T:
        model_out = self.model(x)
        w_mod_sig = model_out["w_mod_sig"]
        q_mod_sig = model_out["q"].unsqueeze(1)

        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
        }
        global_params = {
            "osc_shape": model_out["osc_shape"],
            "osc_gain": model_out["osc_gain"],
            "dist_gain": model_out["dist_gain"],
            "learned_alpha": model_out["learned_alpha"],
        }
        synth_out = self.synth(
            f0_hz=self.f0_hz,
            note_on_duration=self.note_on_duration,
            phase=self.phase,
            filter_args=filter_args,
            global_params=global_params,
        )
        wet = synth_out["wet"]
        return wet


class AcidSynthModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "testing"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "LFO extraction evaluation model."

    def get_model_long_description(self) -> str:
        return "LFO extraction evaluation model for 'Modulation Extraction for LFO-driven Audio Effects'."

    def get_technical_description(self) -> str:
        return "tbd"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "tbd",
            # "Code": "tbd",
        }

    def get_tags(self) -> List[str]:
        return ["subtractive synth", "acid", "TB-303", "sound matching"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("midi_f0", "Midi f0 pitch [24, 48]", default_value=0.5),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [6000]

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        x = x.unsqueeze(1)
        y = self.model(x)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    model = AcidSynthModel()
    # audio = tr.rand((1, 1, 6000))
    # out = model(audio)
    # exit()
    wrapper = AcidSynthModelWrapper(model)
    root_dir = pathlib.Path(
        os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name())
    )
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=True)
