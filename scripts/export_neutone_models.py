import logging
import os
import pathlib
from typing import Dict, List

import torch as tr
import torch.nn as nn
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from torch import Tensor as T

import util
from models import Spectral2DCNN
from paths import CONFIGS_DIR, OUT_DIR
from synths import AcidSynthBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthModel(nn.Module):
    def __init__(
        self,
        model: Spectral2DCNN,
        synth: AcidSynthBase,
        min_f0_hz: float = 32.70,
        max_f0_hz: float = 523.25,
        min_note_on_duration: float = 0.125,
        max_note_on_duration: float = 0.125,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.min_f0_hz = min_f0_hz
        self.max_f0_hz = max_f0_hz
        self.min_note_on_duration = min_note_on_duration
        self.max_note_on_duration = max_note_on_duration
        # TODO(cm): add cached envelope
        assert self.min_note_on_duration == self.max_note_on_duration
        # TODO(cm)
        self.ac = synth.ac
        self.register_buffer("phase", tr.zeros((1, 1)))

    def forward(self, x: T, f0_hz_0to1: T, note_on_duration_0to1: T) -> T:
        f0_hz = f0_hz_0to1 * (self.max_f0_hz - self.min_f0_hz) + self.min_f0_hz
        note_on_duration = note_on_duration_0to1 * (
            self.max_note_on_duration - self.min_note_on_duration
        ) + self.min_note_on_duration

        model_out = self.model(x)
        w_mod_sig = model_out["w_mod_sig"]
        q_mod_sig = model_out["q_0to1"].unsqueeze(1)
        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
        }
        global_params = {
            "osc_shape": self.ac.convert_from_0to1("osc_shape", model_out["osc_shape_0to1"]),
            "osc_gain": self.ac.convert_from_0to1("osc_shape", model_out["osc_gain_0to1"]),
            "dist_gain": self.ac.convert_from_0to1("osc_shape", model_out["dist_gain_0to1"]),
            "learned_alpha": self.ac.convert_from_0to1("osc_shape", model_out["learned_alpha_0to1"]),
        }
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
            NeutoneParameter("f0_hz", "f0_hz", default_value=0.5),
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
        f0_hz = params["f0_hz"]
        note_on_duration = params["note_on_duration"]

        x = x.unsqueeze(1)
        y = self.model(x, f0_hz, note_on_duration)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    # parser = ArgumentParser()
    # cfg = parser.parse_path(ac_path)
    # cfg = parser.instantiate_classes(cfg)
    # exit()

    ac_path = os.path.join(CONFIGS_DIR, "abstract_303", "audio_config.yml")
    ac_class, kwargs = util.load_class_from_config(ac_path)
    ac = ac_class(**kwargs)
    fe_path = os.path.join(CONFIGS_DIR, "abstract_303", "log_mel_spec.yml")
    fe_class, kwargs = util.load_class_from_config(fe_path)
    kwargs["eps"] = float(kwargs["eps"])
    fe = fe_class(**kwargs)
    cnn_path = os.path.join(CONFIGS_DIR, "abstract_303", "spectral_2dcnn_lp.yml")
    cnn_class, kwargs = util.load_class_from_config(cnn_path)
    kwargs["fe"] = fe
    cnn = cnn_class(**kwargs)
    # synth_path = os.path.join(CONFIGS_DIR, "abstract_303", "synth_lp_td.yml")
    synth_path = os.path.join(CONFIGS_DIR, "abstract_303", "synth_lp_fs_128.yml")
    # synth_path = os.path.join(CONFIGS_DIR, "abstract_303", "synth_lp_fs_4096.yml")
    synth_class, kwargs = util.load_class_from_config(synth_path)
    kwargs["ac"] = ac
    # kwargs["make_scriptable"] = True
    synth = synth_class(**kwargs)

    model = AcidSynthModel(cnn, synth)
    wrapper = AcidSynthModelWrapper(model)
    root_dir = pathlib.Path(
        os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name())
    )
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=False)
