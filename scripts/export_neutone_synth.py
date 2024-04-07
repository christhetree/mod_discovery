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
from paths import OUT_DIR, MODELS_DIR
from synths import AcidSynthLPBiquad

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynth(nn.Module):
    def __init__(
        self,
        synth: AcidSynthLPBiquad,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
        note_on_duration: float = 0.125,
        osc_shape: float = 1.0,
        osc_gain: float = 0.5,
        dist_gain: float = 1.0,
        max_proc_duration: float = 10.0,
        cooldown_duration: float = 1.0,
    ):
        super().__init__()
        self.synth = synth
        if hasattr(synth, "toggle_scriptable"):
            synth.toggle_scriptable(True)
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.register_buffer("note_on_duration", tr.full((1,), note_on_duration))
        self.register_buffer("osc_shape", tr.full((1,), osc_shape))
        self.register_buffer("osc_gain", tr.full((1,), osc_gain))
        self.register_buffer("dist_gain", tr.full((1,), dist_gain))
        self.max_proc_duration = max_proc_duration

        self.max_proc_samples = int(max_proc_duration * synth.ac.sr)
        self.cooldown_samples = int(cooldown_duration * synth.ac.sr)
        self.n_proc_samples = 0
        self.can_process = True
        self.register_buffer("phase", tr.zeros((1, 1)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def reset(self) -> None:
        self.n_proc_samples = 0
        self.can_process = True

    def forward(
        self, x: T, midi_f0_0to1: T, w_mod_sig: T, q_mod_sig: T, alpha_0to1: T
    ) -> T:
        n_samples = x.size(-1)
        if x.min() == x.max() == 0.0:
            return x
        # if not self.can_process or x.min() == x.max() == 0.0:
        #     self.n_proc_samples -= n_samples
        #     if self.n_proc_samples <= (self.max_proc_samples - self.cooldown_samples):
        #         self.can_process = True
        #         self.n_proc_samples = 0
        #     return x

        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]
        alpha = self.synth.ac.convert_from_0to1("learned_alpha", alpha_0to1)

        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
        }
        global_params = {
            "osc_shape": self.osc_shape,
            "osc_gain": self.osc_gain,
            "dist_gain": self.dist_gain,
            "learned_alpha": alpha,
        }
        # self.phase.uniform_()
        # tr.mul(self.phase, 2 * tr.pi, out=self.phase)
        synth_out = self.synth(
            n_samples=n_samples,
            f0_hz=f0_hz,
            note_on_duration=self.note_on_duration,
            phase=self.phase,
            filter_args=filter_args,
            global_params=global_params,
        )
        wet = synth_out["wet"]

        # self.n_proc_samples += n_samples
        # if self.n_proc_samples >= self.max_proc_samples:
        #     self.can_process = False
        return wet


class AcidSynthWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "testing"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "tbd"

    def get_model_long_description(self) -> str:
        return "tbd"

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
            NeutoneParameter("midi_f0", "midi_f0", default_value=0.5),
            NeutoneParameter("alpha", "alpha", default_value=0.5),
            NeutoneParameter("w_mod_sig", "w_mod_sig", default_value=1.0),
            NeutoneParameter("q_mod_sig", "q_mod_sig", default_value=0.5),
        ]

    def aggregate_params(self, params: T) -> T:
        return params

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

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        midi_f0 = params["midi_f0"]
        midi_f0 = midi_f0.mean().unsqueeze(0)
        w_mod_sig = params["w_mod_sig"].unsqueeze(0)
        q_mod_sig = params["q_mod_sig"].unsqueeze(0)
        alpha_0to1 = params["alpha"]
        alpha_0to1 = alpha_0to1.mean().unsqueeze(0)
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0, w_mod_sig, q_mod_sig, alpha_0to1)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    model_dir = MODELS_DIR
    # model_name = "cnn_mss_lp_td__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    model_name = "cnn_mss_lp_fs_128__abstract_303_48k__6k__4k_min__epoch_183_step_1104"

    config_path = os.path.join(model_dir, model_name, "config.yaml")
    ckpt_path = os.path.join(model_dir, model_name, "checkpoints", f"{model_name}.ckpt")
    _, synth = util.extract_model_and_synth_from_config(config_path, ckpt_path)
    # This isn't actually necessary, doing it just in case
    synth.eval()

    model = AcidSynth(synth)
    wrapper = AcidSynthWrapper(model)
    root_dir = pathlib.Path(os.path.join(OUT_DIR, "neutone_models", "acid_synth"))
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=False)
