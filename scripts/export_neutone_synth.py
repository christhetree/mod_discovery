import logging
import os
from contextlib import suppress
from typing import Dict, List

import librosa
import torch as tr
import torch.nn as nn
from neutone_sdk import (
    WaveformToWaveformBase,
    NeutoneParameter,
    ContinuousNeutoneParameter,
)
from torch import Tensor as T

from audio_config import AudioConfig
from cli import CustomLightningCLI
from paths import MODELS_DIR, OUT_DIR, CONFIGS_DIR, WAVETABLES_DIR
from synth_modules import WavetableOsc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynth(nn.Module):
    def __init__(
        self,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
        min_alpha: float = 0.2,
        max_alpha: float = 3.0,
        min_w_hz: float = 100.0,
        max_w_hz: float = 8000.0,
        min_q: float = 0.7071,
        max_q: float = 8.0,
        sr: int = 48000,
        note_on_duration: float = 0.125,
        osc_shape: float = 1.0,
        osc_gain: float = 0.5,
        dist_gain: float = 1.0,
        stability_eps: float = 0.001,
        use_fs: bool = False,
        win_len: int = 128,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
    ):
        super().__init__()
        self.ac = AudioConfig(
            sr=sr,
            min_w_hz=min_w_hz,
            max_w_hz=max_w_hz,
            min_q=min_q,
            max_q=max_q,
            stability_eps=stability_eps,
        )
        # if use_fs:
        #     self.synth = AcidSynthLPBiquadFSM(
        #         self.ac,
        #         win_len=win_len,
        #         overlap=overlap,
        #         oversampling_factor=oversampling_factor,
        #     )
        # else:
        #     self.synth = AcidSynthLPBiquad(self.ac)
        #     self.synth.toggle_scriptable(is_scriptable=True)

        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_w_hz = min_w_hz
        self.max_w_hz = max_w_hz
        self.min_q = min_q
        self.max_q = max_q
        self.sr = sr
        self.register_buffer("note_on_duration", tr.full((1,), note_on_duration))
        self.register_buffer("osc_shape", tr.full((1,), osc_shape))
        self.register_buffer("osc_gain", tr.full((1,), osc_gain))
        self.register_buffer("dist_gain", tr.full((1,), dist_gain))
        self.use_fs = use_fs
        self.win_len = win_len
        self.overlap = overlap
        self.hop_len = int(win_len * (1 - overlap))
        assert win_len % self.hop_len == 0, "Hop length must divide into window length."
        self.oversampling_factor = oversampling_factor

        self.note_on_samples = int(note_on_duration * self.sr)
        self.curr_env_val = 1.0
        self.register_buffer("phase", tr.zeros((1, 1), dtype=tr.double))
        self.register_buffer("zi", tr.zeros((1, 2)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def reset(self) -> None:
        self.curr_env_val = 1.0
        self.phase.zero_()
        self.zi.zero_()

    def forward(
        self,
        x: T,
        midi_f0_0to1: T,
        alpha_0to1: T,
        w_mod_sig: T,
        q_mod_sig: T,
    ) -> T:
        n_samples = x.size(-1)
        alpha = alpha_0to1 * (self.max_alpha - self.min_alpha) + self.min_alpha
        # env, _, new_env_val = make_envelope(x, self.note_on_samples, self.curr_env_val)
        env = None
        new_env_val = None
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
            "zi": self.zi,
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

        period_completion = (n_samples / (self.sr / f0_hz.double())) % 1.0
        tr.add(self.phase, 2 * tr.pi * period_completion, out=self.phase)
        if not self.use_fs:
            y_a = synth_out["y_a"]
            self.zi[:, :] = y_a[:, -2:]
        return wet


class AcidSynthWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        if self.model.use_fs:
            return f"acid_synth_lp_fs_{self.model.win_len}"
        else:
            return "acid_synth_lp_td"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Low-pass biquad TB-303 DDSP implementation."

    def get_model_long_description(self) -> str:
        return "Low-pass biquad TB-303 DDSP implementation for 'Differentiable All-pole Filters for Time-varying Audio Systems'."

    def get_technical_description(self) -> str:
        return "Wrapper for a TB-303 DDSP implementation consisting of a sawtooth or square wave oscillator, time-varying low-pass biquad filter, and hyperbolic tangent distortion."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "tbd",
            "Code": "https://github.com/DiffAPF/TB-303",
        }

    def get_tags(self) -> List[str]:
        return ["subtractive synth", "acid", "TB-303"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(
                "midi_f0",
                f"Oscillator pitch quantized to the nearest midi pitch [f{self.model.min_midi_f0}, f{self.model.max_midi_f0}]",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "alpha",
                f"Decaying envelope generator exponent [f{self.model.min_alpha}, f{self.model.max_alpha}]",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "w_mod_sig",
                f"Filter cutoff frequency [f{self.model.min_w_hz} Hz, f{self.model.max_w_hz} Hz]",
                default_value=1.0,
            ),
            ContinuousNeutoneParameter(
                "q_mod_sig",
                f"Filter resonance Q-factor [f{self.model.min_q}, f{self.model.max_q}]",
                default_value=0.5,
            ),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [self.model.sr]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        if self.model.use_fs:
            return [
                bs
                for bs in range(
                    self.model.win_len,
                    max(self.model.win_len + 1, 10000),
                    self.model.hop_len,
                )
            ]
        else:
            return []

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        n_samples = x.size(-1)
        midi_f0_0to1 = params["midi_f0"]
        w_mod_sig = params["w_mod_sig"].unsqueeze(0)
        q_mod_sig = params["q_mod_sig"].unsqueeze(0)
        w_mod_sig = w_mod_sig.expand(-1, n_samples)
        q_mod_sig = q_mod_sig.expand(-1, n_samples)
        alpha_0to1 = params["alpha"]
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0_0to1, alpha_0to1, w_mod_sig, q_mod_sig)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    # import torch.utils.cpp_extension  # Import is needed first
    # torch.utils.cpp_extension.load(
    #     name="torchlpc",
    #     sources=["../cpp/torchlpc.cpp"],
    #     is_python_module=False,
    #     verbose=True
    # )

    # ("exp_1__frame", f"synthetic/train__mod_extraction__frame.yml", f"train__mod_ex__frame/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__lpf", f"synthetic/train__mod_extraction__lpf.yml", f"train__mod_ex__lpf/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__spline", f"synthetic/train__mod_extraction__spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__rand_spline", f"synthetic/train__mod_extraction__baseline_rand_spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__frame", f"synthetic/test_vital_curves__mod_extraction__frame.yml", f"train__mod_ex__frame/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__lpf", f"synthetic/test_vital_curves__mod_extraction__lpf.yml", f"train__mod_ex__lpf/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__spline", f"synthetic/test_vital_curves__mod_extraction__spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
    # ("exp_1__rand_spline", f"synthetic/test_vital_curves__mod_extraction__baseline_rand_spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),

    # ("exp_2__oracle", f"synthetic/train__mod_discovery__baseline_oracle.yml", f"train__mod_discovery__baseline_oracle/mod_discovery/version_{wt_idx}/checkpoints/mss__oracle__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
    # ("exp_2__frame", f"synthetic/train__mod_discovery__frame.yml", f"train__mod_discovery__frame/mod_discovery/version_{wt_idx}/checkpoints/mss__frame__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
    # # ("exp_2__frame", f"synthetic/train__mod_discovery__frame.yml", f"train__mod_discovery__frame/mod_discovery/version_{wt_idx}/checkpoints/mss__frame__sm_16_1024__ase__ableton_10__epoch_28_step_1160.ckpt"),
    # ("exp_2__lpf", f"synthetic/train__mod_discovery__lpf.yml", f"train__mod_discovery__lpf/mod_discovery/version_{wt_idx}/checkpoints/mss__frame_8_hz__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
    # # ("exp_2__lpf", f"synthetic/train__mod_discovery__lpf.yml", f"train__mod_discovery__lpf/mod_discovery/version_{wt_idx}/checkpoints/mss__frame_8_hz__sm_16_1024__ase__ableton_10__epoch_28_step_1160.ckpt"),
    # ("exp_2__spline", f"synthetic/train__mod_discovery__spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
    # # ("exp_2__spline", f"synthetic/train__mod_discovery__spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_27_step_1120.ckpt"),
    # ("exp_2__rand_spline", f"synthetic/train__mod_discovery__baseline_rand_spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
    # # ("exp_2__rand_spline", f"synthetic/train__mod_discovery__baseline_rand_spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_27_step_1120.ckpt"),

    # ("exp_3__mod_synth__frame", "serum/train__mod_discovery__mod_synth_frame.yml", "mss__frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__mod_synth__lpf", "serum/train__mod_discovery__mod_synth_lpf.yml", "mss__frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__mod_synth__spline", "serum/train__mod_discovery__mod_synth_spline.yml", "mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__mod_synth__rand_spline", "serum/train__mod_discovery__mod_synth_baseline_rand_spline.yml", "mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__shan_et_al__frame", "serum/train__mod_discovery__shan_et_al_frame.yml", "mss__shan_frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__shan_et_al__lpf", "serum/train__mod_discovery__shan_et_al_lpf.yml", "mss__shan_frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_28_step_986.ckpt"),
    # ("exp_3__shan_et_al__spline", "serum/train__mod_discovery__shan_et_al_spline.yml", "mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
    # ("exp_3__shan_et_al__rand_spline", "serum/train__mod_discovery__shan_et_al_baseline_rand_spline.yml", "mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),

    ckpt_to_config = {
        "exp_1__frame": "synthetic/train__mod_extraction__frame.yml",
        "exp_1__lpf": "synthetic/train__mod_extraction__lpf.yml",
        "exp_1__spline": "synthetic/train__mod_extraction__spline.yml",
        "exp_2__oracle": "synthetic/train__mod_discovery__baseline_oracle.yml",
        "exp_2__frame": "synthetic/train__mod_discovery__frame.yml",
        "exp_2__lpf": "synthetic/train__mod_discovery__lpf.yml",
        "exp_2__spline": "synthetic/train__mod_discovery__spline.yml",
        # "exp_3__frame__mod_synth": "synthetic/train__mod_discovery__frame.yml",
        # "exp_3__lpf__mod_synth": "synthetic/train__mod_discovery__lpf.yml",
        # "exp_3__spline__mod_synth": "synthetic/train__mod_discovery__spline.yml",
    }


    exp_name = "exp_1"
    # exp_name = "exp_2"
    # exp_name = "exp_3"
    # method_name = "frame"
    # method_name = "lpf"
    method_name = "spline"
    # method_name = "oracle"
    wt_name = "0__basics__fm_fold__78_1024"
    arch_name = "mod_synth"
    # arch_name = "shan_et_al"
    # arch_name = "engel_et_al"
    seed_name = "seed_0"

    if exp_name == "exp_3":
        ckpt_dir = os.path.join(exp_name, method_name, arch_name, seed_name, "checkpoints")
        config_path = ckpt_to_config[f"{exp_name}__{method_name}__{arch_name}"]
    else:
        ckpt_dir = os.path.join(exp_name, method_name, wt_name, seed_name, "checkpoints")
        config_path = ckpt_to_config[f"{exp_name}__{method_name}"]

    ckpt_dir = os.path.join(MODELS_DIR, ckpt_dir)
    config_path = os.path.join(CONFIGS_DIR, config_path)
    wt_path = os.path.join(WAVETABLES_DIR, "ableton", f"{wt_name[3:]}.pt")
    wt = tr.load(wt_path, weights_only=True)

    def get_ckpt_path(ckpt_dir: str) -> str:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1, f"Expected one checkpoint file in {ckpt_dir}, found {len(ckpt_files)}."
        return os.path.join(ckpt_dir, ckpt_files[0])

    ckpt_path = get_ckpt_path(ckpt_dir)

    log.info(f"Ckpt path: {ckpt_path}")

    cli = CustomLightningCLI(
        args=["-c", config_path],
        trainer_defaults=CustomLightningCLI.make_trainer_defaults(save_dir=OUT_DIR),
        run=False,
    )
    synth = cli.model.synth
    with suppress(Exception):
        if synth.add_synth_module.__class__.__name__ == "WavetableOsc" and not synth.add_synth_module.is_trainable:
            log.info(f"Resizing synth wavetable module to {wt.shape}")
            sr = synth.ac.sr
            wt_module_hat = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
            synth.register_module("add_synth_module", wt_module_hat)
    synth_hat = cli.model.synth_hat
    with suppress(Exception):
        if synth_hat.add_synth_module.__class__.__name__ == "WavetableOsc" and not synth_hat.add_synth_module.is_trainable:
            log.info(f"Resizing synth_hat wavetable module to {wt.shape}")
            sr = synth_hat.ac.sr
            wt_module_hat = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
            synth_hat.register_module("add_synth_module", wt_module_hat)

    state_dict = tr.load(ckpt_path, map_location="cpu")["state_dict"]
    cli.model.load_state_dict(state_dict)
    cli.model.eval()

    scripted = tr.jit.script(synth_hat)


    # model = AcidSynth(use_fs=False)
    # wrapper = AcidSynthWrapper(model)
    # root_dir = pathlib.Path(
    #     os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name())
    # )
    # save_neutone_model(
    #     wrapper,
    #     root_dir,
    #     submission=False,
    #     dump_samples=False,
    #     test_offline_mode=False,
    #     speed_benchmark=False,
    # )
