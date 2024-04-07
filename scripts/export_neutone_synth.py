import logging
import os
import pathlib
from typing import Dict, List, Tuple

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
        # self.register_buffer("alpha", tr.full((1,), 1.0))

        self.sr = self.synth.ac.sr
        self.note_on_samples = int(note_on_duration * self.sr)
        self.curr_env_val = 1.0
        self.register_buffer("phase", tr.zeros((1, 1)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def reset(self) -> None:
        self.curr_env_val = 1.0
        self.phase.zero_()

    @staticmethod
    def make_envelope_2(
        x: T, note_on_samples: int, curr_env_val: float
    ) -> Tuple[T, int, float]:
        n_samples = x.size(-1)
        n_env = (2 * n_samples) // note_on_samples
        n_env = max(n_env, 2)
        env = tr.linspace(1.0, 0.0, note_on_samples)
        env = env.repeat(n_env)

        out = tr.zeros((1, n_samples))

        # Find first non-zero index
        non_zero_indices = tr.nonzero(x.squeeze())
        first_nonzero_idx = -1
        if non_zero_indices.size(0) > 0:
            first_nonzero_idx = non_zero_indices[0, 0].item()

        # If all silence and no envelope is being continued, return silence
        if first_nonzero_idx == -1 and curr_env_val == 0.0:
            return out, 0, curr_env_val

        # Calc envelope continuation index
        cont_env_start_idx = 0
        if curr_env_val != 0.0:
            cont_env_start_idx = int(
                round((1.0 - curr_env_val) * (note_on_samples - 1))
            )
            # cont_env_start_idx = int(round(cont_env_start_idx))
            cont_env_start_idx = min(cont_env_start_idx, note_on_samples - 1)
            cont_env_start_idx = max(cont_env_start_idx, 0)

        # If there is no silence, return the envelope
        if first_nonzero_idx == 0:
            out[:, 0:n_samples] = env[
                cont_env_start_idx : cont_env_start_idx + n_samples
            ]
            curr_env_val = out[0, -1].item()
            return out, first_nonzero_idx, curr_env_val

        # Continue the envelope if required
        cont_env_len = 0
        if curr_env_val != 0.0:
            cont_env_len = min(n_samples, note_on_samples - cont_env_start_idx)
            cont_env_end_idx = cont_env_start_idx + cont_env_len
            out[:, 0:cont_env_len] = env[cont_env_start_idx:cont_env_end_idx]

        # If all silence
        if first_nonzero_idx == -1:
            curr_env_val = out[0, -1].item()
            return out, 0, curr_env_val

        first_nonzero_idx = max(first_nonzero_idx, cont_env_len)
        n_non_zero_samples = n_samples - first_nonzero_idx
        out[:, first_nonzero_idx:n_samples] = env[0:n_non_zero_samples]
        curr_env_val = out[0, -1].item()
        return out, first_nonzero_idx, curr_env_val

    def forward(
        self,
        x: T,
        midi_f0_0to1: T,
        # note_on_duration_0to1: T,
        alpha_0to1: T,
        w_mod_sig: T,
        q_mod_sig: T,
    ) -> T:
        n_samples = x.size(-1)
        # note_on_duration = (
        #     note_on_duration_0to1
        #     * (self.max_note_on_duration - self.min_note_on_duration)
        #     + self.min_note_on_duration
        # )
        # note_on_samples = int(note_on_duration.item() * self.sr)
        alpha = self.synth.ac.convert_from_0to1("learned_alpha", alpha_0to1)

        env, env_start_idx, new_env_val = self.make_envelope_2(
            x, self.note_on_samples, self.curr_env_val
        )
        self.curr_env_val = new_env_val
        if alpha != 1.0:
            tr.pow(env, alpha, out=env)

        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]


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
        synth_out = self.synth(
            n_samples=n_samples,
            f0_hz=f0_hz,
            note_on_duration=self.note_on_duration,
            phase=self.phase,
            filter_args=filter_args,
            global_params=global_params,
            envelope=env,
        )
        wet = synth_out["wet"]

        period_completion = (n_samples / (self.sr / f0_hz.item())) % 1.0
        tr.add(self.phase, 2 * tr.pi * period_completion, out=self.phase)
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
            # NeutoneParameter("note_on_duration", "note_on_duration", default_value=0.0),
            NeutoneParameter("alpha", "alpha", default_value=0.25),
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
        return []

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        midi_f0_0to1 = params["midi_f0"]
        midi_f0_0to1 = midi_f0_0to1.mean().unsqueeze(0)
        # note_on_duration_0to1 = params["note_on_duration"]
        # note_on_duration_0to1 = note_on_duration_0to1.mean().unsqueeze(0)
        w_mod_sig = params["w_mod_sig"].unsqueeze(0)
        q_mod_sig = params["q_mod_sig"].unsqueeze(0)
        alpha_0to1 = params["alpha"]
        alpha_0to1 = alpha_0to1.mean().unsqueeze(0)
        x = x.unsqueeze(1)
        # y = self.model(x, midi_f0_0to1, note_on_duration_0to1, w_mod_sig, q_mod_sig)
        y = self.model(x, midi_f0_0to1, alpha_0to1, w_mod_sig, q_mod_sig)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    # buffer_n_samples = 200
    # curr_env_val = 0.0
    # alpha = 1.0
    # note_on_samples_all = [300] * 5
    # envs = []
    # for idx, note_on_samples in enumerate(note_on_samples_all):
    #     if idx == 0:
    #         x = tr.randn((buffer_n_samples,))
    #         x[0:10] = 0.0
    #     elif idx == 1:
    #         x = tr.zeros((buffer_n_samples,))
    #     else:
    #         x = tr.randn((buffer_n_samples,))
    #     # x[6:9] = 0.0
    #
    #     env, start_idx, curr_env_val = AcidSynth.make_envelope_2(
    #         x, note_on_samples, curr_env_val, alpha
    #     )
    #     log.info(
    #         f"env.shape: {env.shape}, start_idx: {start_idx}, curr_env_val: {curr_env_val}"
    #     )
    #     envs.append(env)
    # from matplotlib import pyplot as plt
    #
    # env = tr.cat(envs, dim=-1)
    # plt.plot(env.squeeze().numpy())
    # plt.show()
    # exit()

    model_dir = MODELS_DIR
    model_name = "cnn_mss_lp_td__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fs_128__abstract_303_48k__6k__4k_min__epoch_183_step_1104"

    config_path = os.path.join(model_dir, model_name, "config.yaml")
    ckpt_path = os.path.join(model_dir, model_name, "checkpoints", f"{model_name}.ckpt")
    _, synth = util.extract_model_and_synth_from_config(config_path, ckpt_path)
    # This isn't actually necessary, doing it just in case
    synth.eval()

    model = AcidSynth(synth)
    wrapper = AcidSynthWrapper(model)
    root_dir = pathlib.Path(os.path.join(OUT_DIR, "neutone_models", "acid_synth"))
    save_neutone_model(wrapper, root_dir, dump_samples=False, submission=False)
