import logging
import os
import pathlib
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
from neutone_sdk.utils import save_neutone_model
from torch import Tensor as T

from cli import CustomLightningCLI
from paths import MODELS_DIR, OUT_DIR, CONFIGS_DIR, WAVETABLES_DIR
from synth_modules import WavetableOsc, BiquadWQFilter, BiquadCoeffFilter

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ModSynth(nn.Module):
    def __init__(
        self,
        name: str,
        add_synth_module: WavetableOsc,
        sub_synth_module: BiquadWQFilter | BiquadCoeffFilter,
        sr: int = 48000,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
    ):
        super().__init__()
        self.name = name
        assert add_synth_module.use_aa
        self.add_synth_module = add_synth_module
        sub_synth_module.toggle_scriptable(True)
        self.sub_synth_module = sub_synth_module
        self.sr = sr
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0

        self.register_buffer("phase", tr.zeros((1, 1), dtype=tr.double))
        self.register_buffer("zi", tr.zeros((1, 2)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

        # Precompute anti-aliased wavetables for all MIDI pitches
        self.midi_f0_to_wt = {}
        for idx in range(min_midi_f0, max_midi_f0 + 1):
            f0_hz = self.midi_f0_to_hz[idx]
            wt = self.add_synth_module.get_maybe_aa_maybe_bounded_wt(f0_hz.unsqueeze(1))
            self.midi_f0_to_wt[idx] = wt

        # Use env_mod_sig to control the Q of the filter for Experiment 1
        self.use_q_mod_sig = False
        if sub_synth_module.__class__.__name__ == "BiquadWQFilter":
            self.use_q_mod_sig = True
            log.info("Using Neutone control parameter D for Q of BiquadWQFilter")

    def reset(self) -> None:
        self.phase.zero_()
        self.zi.zero_()

    def forward(
        self,
        n_samples: int,
        midi_f0_0to1: T,
        add_mod_sig: T,
        sub_mod_sig: T,
        env_mod_sig: T,
    ) -> T:
        # Additive synthesis
        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0[0].round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]
        wt = self.midi_f0_to_wt[midi_f0]
        add_out = self.add_synth_module(
            f0_hz=f0_hz,
            wt_pos_0to1=add_mod_sig,
            n_samples=n_samples,
            phase=self.phase,
            wt=wt,
        )

        # Subtractive synthesis
        if self.use_q_mod_sig:
            sub_out = self.sub_synth_module(
                x=add_out,
                w_mod_sig=sub_mod_sig,
                q_mod_sig=env_mod_sig,
                zi=self.zi,
            )
            env_out = sub_out
        else:
            env_out = add_out * env_mod_sig

        # Advance phase and store filter state
        period_completion = (n_samples / (self.sr / f0_hz.double())) % 1.0
        tr.add(self.phase, 2 * tr.pi * period_completion, out=self.phase)
        self.zi[:, :] = self.sub_synth_module.next_zi

        return env_out


class ModSynthWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return self.model.name

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "TBD"

    def get_model_long_description(self) -> str:
        return "TBD"

    def get_technical_description(self) -> str:
        return "TBD"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://christhetr.ee/mod_discovery/",
            "Code": "https://github.com/christhetree/mod_discovery/",
        }

    def get_tags(self) -> List[str]:
        return ["TBD"]

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
                "add_mod_sig",
                f"Wavetable position modulation signal",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "sub_mod_sig",
                f"Low-pass filter cutoff frequency" if self.model.use_q_mod_sig else "Filter modulation signal",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "q_mod_sig" if self.model.use_q_mod_sig else "env_mod_sig",
                f"Low-pass filter resonance" if self.model.use_q_mod_sig else "Envelope modulation signal",
                default_value=0.5 if self.model.use_q_mod_sig else 1.0,
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
        return []

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def aggregate_params(self, params: T) -> T:
        return params

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        n_samples = x.size(-1)
        midi_f0_0to1 = params["midi_f0"]
        add_mod_sig = params["add_mod_sig"].unsqueeze(0)
        sub_mod_sig = params["sub_mod_sig"].unsqueeze(0)
        env_mod_sig = params["q_mod_sig"].unsqueeze(0) if self.model.use_q_mod_sig else params["env_mod_sig"].unsqueeze(0)
        y = self.model(
            n_samples=n_samples,
            midi_f0_0to1=midi_f0_0to1,
            add_mod_sig=add_mod_sig,
            sub_mod_sig=sub_mod_sig,
            env_mod_sig=env_mod_sig,
        )
        return y


if __name__ == "__main__":
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

    exp_name = "exp_1"
    # exp_name = "exp_2"
    # exp_name = "exp_3"

    method_name = "frame"
    # method_name = "lpf"
    # method_name = "spline"
    # method_name = "oracle"

    # wt_name = "0__basics__fm_fold__78_1024"
    # wt_name = "1__basics__galactica__4_1024"
    wt_name = "2__basics__harmonic_series__7_1024"
    # wt_name = "3__basics__sub_3__122_1024"
    # wt_name = "4__collection__aureolin__256_1024"
    # wt_name = "5__collection__squash__32_1024"
    # wt_name = "6__complex__bit_ring__256_1024"
    # wt_name = "7__complex__kicked__4_1024"
    # wt_name = "8__distortion__dp_fold__230_1024"
    # wt_name = "9__distortion__phased__178_1024"

    arch_name = "mod_synth"
    # arch_name = "shan_et_al"
    # arch_name = "engel_et_al"

    seed_name = "seed_0"

    # ==================================================================================
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

    # Determine checkpoint path
    if exp_name == "exp_3":
        ckpt_dir = os.path.join(exp_name, method_name, arch_name, seed_name, "checkpoints")
        config_path = ckpt_to_config[f"{exp_name}__{method_name}__{arch_name}"]
        model_name = f"{exp_name}__{method_name}__{arch_name}"
    else:
        ckpt_dir = os.path.join(exp_name, method_name, wt_name, seed_name, "checkpoints")
        config_path = ckpt_to_config[f"{exp_name}__{method_name}"]
        model_name = f"{exp_name}__{method_name}__wt_{wt_name[0]}"

    ckpt_dir = os.path.join(MODELS_DIR, ckpt_dir)
    config_path = os.path.join(CONFIGS_DIR, config_path)

    def get_ckpt_path(ckpt_dir: str) -> str:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1, f"Expected one checkpoint file in {ckpt_dir}, found {len(ckpt_files)}."
        return os.path.join(ckpt_dir, ckpt_files[0])

    ckpt_path = get_ckpt_path(ckpt_dir)
    log.info(f"Ckpt path: {ckpt_path}")

    # Initialize checkpoint classes
    cli = CustomLightningCLI(
        args=["-c", config_path],
        trainer_defaults=CustomLightningCLI.make_trainer_defaults(save_dir=OUT_DIR),
        run=False,
    )

    # Resize wavetables if frozen to match shape of checkpoint weights
    wt_path = os.path.join(WAVETABLES_DIR, "ableton", f"{wt_name[3:]}.pt")
    wt = tr.load(wt_path, weights_only=True)
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

    # Load checkpoint weights
    state_dict = tr.load(ckpt_path, map_location="cpu")["state_dict"]
    cli.model.load_state_dict(state_dict)
    cli.model.eval()

    # Extract modules and sample rate
    add_module = synth_hat.add_synth_module
    sub_module = synth_hat.sub_synth_module
    sr = synth_hat.ac.sr

    # Wrap synth with Neutone SDK and export
    model = ModSynth(
        name=model_name,
        add_synth_module=add_module,
        sub_synth_module=sub_module,
        sr=sr,
    )
    wrapper = ModSynthWrapper(model)
    root_dir = pathlib.Path(
        os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name())
    )
    save_neutone_model(
        wrapper,
        root_dir,
        submission=False,
        # dump_samples=True,
        dump_samples=False,
        test_offline_mode=False,
        speed_benchmark=False,
    )
