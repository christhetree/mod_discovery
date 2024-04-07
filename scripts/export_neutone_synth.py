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
        min_note_on_duration: float = 0.125,
        max_note_on_duration: float = 0.5,
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
        self.min_note_on_duration = min_note_on_duration
        self.max_note_on_duration = max_note_on_duration
        self.register_buffer("note_on_duration", tr.full((1,), min_note_on_duration))
        self.register_buffer("osc_shape", tr.full((1,), osc_shape))
        self.register_buffer("osc_gain", tr.full((1,), osc_gain))
        self.register_buffer("dist_gain", tr.full((1,), dist_gain))
        self.register_buffer("alpha", tr.full((1,), 1.0))

        self.sr = self.synth.ac.sr
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
    def make_envelope(
        x: T, note_on_samples: int, curr_env_val: float, alpha: T
    ) -> Tuple[T, int, float]:
        n_samples = x.size(-1)
        env = tr.zeros((1, n_samples))

        # Find the indices of non-zero audio sections
        audio_idx_pairs: List[Tuple[int, int]] = []
        prev_curr_silent = True
        start_idx = -1
        for idx, val in enumerate(x.squeeze()):
            if val != 0.0:
                curr_silent = False
            else:
                curr_silent = True

            if prev_curr_silent and not curr_silent:
                start_idx = idx
            elif not prev_curr_silent and curr_silent:
                audio_idx_pairs.append((start_idx, idx))
            prev_curr_silent = curr_silent
        audio_idx_pairs.append((start_idx, n_samples))
        assert len(audio_idx_pairs) > 0
        assert all([s < e for s, e in audio_idx_pairs])

        first_start_idx = audio_idx_pairs[0][0]
        if first_start_idx > 0 or curr_env_val == 1.0:
            curr_env_val = 1.0
            first_env_n_samples = note_on_samples
        else:
            first_env_n_samples = int((note_on_samples - 1) * curr_env_val)
        first_env_n_samples = max(first_env_n_samples, 2)

        # Split the audio sections into envelopes if they are too long
        env_idx_pairs: List[Tuple[int, int]] = []
        max_n_samples = first_env_n_samples
        for start_idx, end_idx in audio_idx_pairs:
            if end_idx - start_idx > max_n_samples:
                curr_start_idx = start_idx
                while curr_start_idx < end_idx:
                    curr_end_idx = min(curr_start_idx + max_n_samples, end_idx)
                    env_idx_pairs.append((curr_start_idx, curr_end_idx))
                    curr_start_idx = curr_end_idx
                    max_n_samples = note_on_samples
            else:
                env_idx_pairs.append((start_idx, end_idx))
        assert all([s < e for s, e in env_idx_pairs])
        if len(env_idx_pairs) >= 3:
            filtered_env_idx_pairs = [env_idx_pairs[0]]
            for idx in range(1, len(env_idx_pairs) - 1):
                start_idx, end_idx = env_idx_pairs[idx]
                if end_idx - start_idx >= 16:
                    filtered_env_idx_pairs.append((start_idx, end_idx))
            filtered_env_idx_pairs.append(env_idx_pairs[-1])
            env_idx_pairs = filtered_env_idx_pairs

        env_start_vals = [curr_env_val] + [1.0 for _ in range(len(env_idx_pairs) - 1)]

        last_env_start_idx, last_env_end_idx = env_idx_pairs[-1]
        last_env_n_samples = last_env_end_idx - last_env_start_idx
        slope = 1.0 / (note_on_samples - 1)
        last_env_start_val = env_start_vals[-1]
        last_env_end_val = last_env_start_val - (last_env_n_samples * slope)
        last_env_end_val = max(last_env_end_val, 0.0)
        env_end_vals = [0.0 for _ in range(len(env_idx_pairs) - 1)] + [last_env_end_val]

        for idx in range(len(env_idx_pairs)):
            start_idx, end_idx = env_idx_pairs[idx]
            start_val = env_start_vals[idx]
            end_val = env_end_vals[idx]
            env_n_samples = end_idx - start_idx
            if start_val == 1.0:
                env_section = tr.linspace(start_val, end_val, env_n_samples)
            else:
                env_section = tr.linspace(start_val, end_val, env_n_samples + 1)[1:]
            env[:, start_idx:end_idx] = env_section

        return env, first_start_idx, last_env_end_val

    def forward(
        self,
        x: T,
        midi_f0_0to1: T,
        note_on_duration_0to1: T,
        w_mod_sig: T,
        q_mod_sig: T,
    ) -> T:
        if x.min() == x.max() == 0.0:
            self.curr_env_val = 1.0
            self.phase.zero_()
            return x

        n_samples = x.size(-1)
        note_on_duration = (
            note_on_duration_0to1
            * (self.max_note_on_duration - self.min_note_on_duration)
            + self.min_note_on_duration
        )
        note_on_samples = int(note_on_duration.item() * self.sr)
        env, env_start_idx, new_env_val = self.make_envelope(
            x, note_on_samples, self.curr_env_val, self.alpha
        )
        self.curr_env_val = new_env_val

        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]

        # alpha = self.synth.ac.convert_from_0to1("learned_alpha", alpha_0to1)

        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
        }
        global_params = {
            "osc_shape": self.osc_shape,
            "osc_gain": self.osc_gain,
            "dist_gain": self.dist_gain,
            "learned_alpha": self.alpha,
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
        dry = synth_out["dry"]

        period_completion = (n_samples / (self.sr / f0_hz.item())) % 1.0
        tr.add(self.phase, 2 * tr.pi * period_completion, out=self.phase)
        return dry


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
            NeutoneParameter("note_on_duration", "note_on_duration", default_value=0.0),
            # NeutoneParameter("alpha", "alpha", default_value=0.5),
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
        # return [6000]
        return []

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        midi_f0_0to1 = params["midi_f0"]
        midi_f0_0to1 = midi_f0_0to1.mean().unsqueeze(0)
        note_on_duration_0to1 = params["note_on_duration"]
        note_on_duration_0to1 = note_on_duration_0to1.mean().unsqueeze(0)
        w_mod_sig = params["w_mod_sig"].unsqueeze(0)
        q_mod_sig = params["q_mod_sig"].unsqueeze(0)
        # alpha_0to1 = params["alpha"]
        # alpha_0to1 = alpha_0to1.mean().unsqueeze(0)
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0_0to1, note_on_duration_0to1, w_mod_sig, q_mod_sig)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    # buffer_n_samples = 20
    # curr_env_val = 1.0
    # alpha = 1.0
    # # note_on_samples_all = [30] * 5
    # note_on_samples_all = [5, 5]
    # envs = []
    # for idx, note_on_samples in enumerate(note_on_samples_all):
    #     x = tr.randn((buffer_n_samples,))
    #     if idx == 0:
    #         x[0:4] = 0.0
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
