import logging
import os
import pathlib
from typing import Dict, List

import librosa
import torch as tr
import torch.nn as nn
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from torch import Tensor as T

import util
from models import Spectral2DCNN
from paths import OUT_DIR, MODELS_DIR
from synths import AcidSynthBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthModel(nn.Module):
    def __init__(
        self,
        model: Spectral2DCNN,
        synth: AcidSynthBase,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
        min_note_on_duration: float = 0.125,
        max_note_on_duration: float = 0.125,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        if hasattr(synth, "toggle_scriptable"):
            synth.toggle_scriptable(True)
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.min_note_on_duration = min_note_on_duration
        self.max_note_on_duration = max_note_on_duration
        # TODO(cm): add cached envelope
        assert self.min_note_on_duration == self.max_note_on_duration
        self.ac = synth.ac
        self.register_buffer("phase", tr.zeros((1, 1)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def forward(self, x: T, midi_f0_0to1: T, note_on_duration_0to1: T) -> T:
        midi_f0 = midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]
        note_on_duration = (
            note_on_duration_0to1
            * (self.max_note_on_duration - self.min_note_on_duration)
            + self.min_note_on_duration
        )

        model_out = self.model(x)
        w_mod_sig = model_out["w_mod_sig"]
        q_mod_sig = model_out["q_0to1"].unsqueeze(1)
        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
        }
        global_params = {
            "osc_shape": self.ac.convert_from_0to1(
                "osc_shape", model_out["osc_shape_0to1"]
            ),
            "osc_gain": self.ac.convert_from_0to1(
                "osc_shape", model_out["osc_gain_0to1"]
            ),
            "dist_gain": self.ac.convert_from_0to1(
                "osc_shape", model_out["dist_gain_0to1"]
            ),
            "learned_alpha": self.ac.convert_from_0to1(
                "osc_shape", model_out["learned_alpha_0to1"]
            ),
        }
        self.phase.uniform_()  # TODO(cm)
        n_samples = x.size(-1)
        synth_out = self.synth(
            n_samples=n_samples,
            f0_hz=f0_hz,
            note_on_duration=note_on_duration,
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
            NeutoneParameter("midi_f0", "midi_f0", default_value=0.90),
            NeutoneParameter("note_on_duration", "note_on_duration", default_value=0.5),
        ]

    # def aggregate_params(self, params: Tensor) -> Tensor:
    #     return params

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
        if x.min() == 0.0:
            return x
        midi_f0 = params["midi_f0"]
        note_on_duration = params["note_on_duration"]
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0, note_on_duration)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    model_dir = MODELS_DIR
    # model_name = "cnn_mss_lp_td__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    model_name = "cnn_mss_lp_fs_128__abstract_303_48k__6k__4k_min__epoch_183_step_1104"

    config_path = os.path.join(model_dir, model_name, "config.yaml")
    ckpt_path = os.path.join(model_dir, model_name, "checkpoints", f"{model_name}.ckpt")
    cnn, synth = util.extract_model_and_synth_from_config(config_path, ckpt_path)
    # This isn't actually necessary, doing it just in case
    cnn.eval()
    synth.eval()

    model = AcidSynthModel(cnn, synth)
    wrapper = AcidSynthModelWrapper(model)
    root_dir = pathlib.Path(os.path.join(OUT_DIR, "neutone_models", model_name))
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=False)
