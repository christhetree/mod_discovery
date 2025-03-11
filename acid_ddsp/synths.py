import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from torch import Tensor as T, nn

from acid_ddsp.synth_modules import (
    SynthModule,
)
from audio_config import AudioConfig

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SynthBase(ABC, nn.Module):
    def __init__(self, ac: AudioConfig):
        super().__init__()
        self.ac = ac

    @abstractmethod
    def additive_synthesis(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        pass

    def subtractive_synthesis(
        self,
        x: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        # By default, do not apply any subtractive synthesis
        return x, {}

    def forward(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> Dict[str, T]:
        add_audio, add_out = self.additive_synthesis(
            n_samples, f0_hz, phase, temp_params, global_params, other_params
        )
        sub_audio, sub_out = self.subtractive_synthesis(
            add_audio, temp_params, global_params, other_params
        )
        synth_out = {}
        if "env" in temp_params:
            env = temp_params["env"]
            env_audio = sub_audio * env
            synth_out["env"] = env
        else:
            env_audio = sub_audio
        synth_out["add_audio"] = add_audio
        synth_out["sub_audio"] = sub_audio
        synth_out["env_audio"] = env_audio
        assert all(k not in synth_out for k in add_out)
        synth_out.update(add_out)
        assert all(k not in synth_out for k in sub_out)
        synth_out.update(sub_out)
        return synth_out


class ComposableSynth(SynthBase):
    def __init__(
        self,
        ac: AudioConfig,
        add_synth_module: SynthModule,
        sub_synth_module: Optional[SynthModule] = None,
        add_lfo_name: str = "add_lfo",
        sub_lfo_name: str = "sub_lfo",
    ):
        super().__init__(ac)
        self.add_synth_module = add_synth_module
        self.sub_synth_module = sub_synth_module
        self.add_lfo_name = add_lfo_name
        self.sub_lfo_name = sub_lfo_name
        if hasattr(self.add_synth_module, "sr"):
            assert self.add_synth_module.sr == ac.sr
        if hasattr(self.sub_synth_module, "sr"):
            assert self.sub_synth_module.sr == ac.sr

    def _forward_synth_module(
        self,
        synth_module: nn.Module,
        synth_module_kwargs: Dict[str, T],
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> T:
        for param_name in synth_module.forward_param_names:
            if hasattr(self.ac, param_name):
                if param_name in synth_module_kwargs:
                    assert synth_module_kwargs[param_name] == getattr(
                        self.ac, param_name
                    )
                else:
                    synth_module_kwargs[param_name] = getattr(self.ac, param_name)
            if param_name in temp_params:
                assert param_name not in synth_module_kwargs
                synth_module_kwargs[param_name] = temp_params[param_name]
            if param_name in global_params:
                assert param_name not in synth_module_kwargs
                synth_module_kwargs[param_name] = global_params[param_name]
            if param_name in other_params:
                assert param_name not in synth_module_kwargs
                synth_module_kwargs[param_name] = other_params[param_name]
        out = synth_module(**synth_module_kwargs)
        return out

    def additive_synthesis(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        synth_module_kwargs = {
            "n_samples": n_samples,
            "f0_hz": f0_hz,
            "phase": phase,
        }
        module_lfo_name = self.add_synth_module.lfo_name
        if module_lfo_name is not None:
            assert self.add_lfo_name in temp_params or self.add_lfo_name in other_params
            if self.add_lfo_name in temp_params:
                synth_module_kwargs[module_lfo_name] = temp_params[self.add_lfo_name]
            else:
                synth_module_kwargs[module_lfo_name] = other_params[self.add_lfo_name]
        add_audio = self._forward_synth_module(
            self.add_synth_module,
            synth_module_kwargs,
            temp_params=temp_params,
            global_params=global_params,
            other_params=other_params,
        )
        return add_audio, {}

    def subtractive_synthesis(
        self,
        x: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        other_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        if self.sub_synth_module is None:
            return x, {}
        synth_module_kwargs = {
            "x": x,
        }
        module_lfo_name = self.sub_synth_module.lfo_name
        if module_lfo_name is not None:
            assert self.sub_lfo_name in temp_params
            synth_module_kwargs[module_lfo_name] = temp_params[self.sub_lfo_name]
        sub_audio = self._forward_synth_module(
            self.sub_synth_module,
            synth_module_kwargs,
            temp_params=temp_params,
            global_params=global_params,
            other_params=other_params,
        )
        return sub_audio, {}


# class WavetableSynth(SynthBase):
#     def __init__(
#         self,
#         ac: AudioConfig,
#         n_pos: int,
#         n_wt_samples: int,
#         aa_filter_n: int,
#         wt_name: Optional[str] = None,
#         is_trainable: bool = True,
#     ):
#         super().__init__(ac)
#         wt = None
#         if wt_name is not None:
#             wt_path = os.path.join(WAVETABLES_DIR, wt_name)
#             assert os.path.isfile(wt_path)
#             log.info(f"Loading wavetable from {wt_path}")
#             wt = tr.load(wt_path)
#             assert wt.shape == (n_pos, n_wt_samples)
#
#         self.osc = WavetableOsc(
#             ac.sr,
#             n_pos=n_pos,
#             n_wt_samples=n_wt_samples,
#             aa_filter_n=aa_filter_n,
#             wt=wt,
#             is_trainable=is_trainable,
#         )
#         self.lpc_func = sample_wise_lpc
#         self.adsr = ADSR(n_frames=ac.n_samples)
#
#     def additive_synthesis(
#         self,
#         n_samples: int,
#         f0_hz: T,
#         phase: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#     ) -> (T, Dict[str, T]):
#         wt_pos = temp_params["add_lfo"]
#         # if wt_pos is None:
#         #     wt_pos = tr.full_like(f0_hz, -1.0).unsqueeze(1)
#         # else:
#         wt_pos = wt_pos.squeeze(2) * 2.0 - 1.0
#         wt_pos = util.interpolate_dim(wt_pos, self.ac.n_samples)
#         temp_params["wt_pos"] = wt_pos
#         dry_audio = self.osc(f0_hz, wt_pos, n_samples=n_samples, phase=phase)
#         return dry_audio, {}
#
#     def _apply_filter(self, x: T, logits: T) -> T:
#         # TODO(cm): reduce duplicate code
#         assert logits.ndim == 3
#         assert logits.size(2) == 5
#         bs = logits.size(0)
#         n_frames = x.size(1)
#         a_logits = logits[..., :2]
#         a_coeff = calc_logits_to_biquad_a_coeff_triangle(
#             a_logits, self.ac.stability_eps
#         )
#         b_coeff = logits[..., 2:]
#         a_coeff = util.interpolate_dim(a_coeff, n_frames, dim=1)
#         assert a_coeff.shape == (bs, n_frames, 2)
#         b_coeff = util.interpolate_dim(b_coeff, n_frames, dim=1)
#         assert b_coeff.shape == (bs, n_frames, 3)
#         y_a = self.lpc_func(x, a_coeff)
#         assert not tr.isinf(y_a).any()
#         assert not tr.isnan(y_a).any()
#         y_ab = time_varying_fir(y_a, b_coeff)
#         return y_ab
#
#     def subtractive_synthesis(
#         self,
#         x: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#     ) -> (T, Dict[str, T]):
#         # TODO(cm): tmp
#         logits = temp_params.get("logits")
#         if logits is None:
#             return x, {}
#         assert logits.ndim == 5
#         filter_depth = logits.size(2)
#         filter_width = logits.size(3)
#         for depth_idx in range(filter_depth):
#             layer_x_s = []
#             for width_idx in range(filter_width):
#                 curr_logits = logits[:, :, depth_idx, width_idx, :]
#                 curr_x = self._apply_filter(x, curr_logits)
#                 layer_x_s.append(curr_x)
#             # TODO(cm): could add a learnable attention over the filters
#             x = tr.stack(layer_x_s, dim=1)
#             x = tr.mean(x, dim=1)
#         return x, {}
#
#     def forward(
#         self,
#         n_samples: int,
#         f0_hz: T,
#         phase: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#         envelope: Optional[T] = None,
#         note_on_duration: Optional[T] = None,
#     ) -> Dict[str, T]:
#         if envelope is None:
#             attack = global_params["attack"]
#             decay = global_params["decay"]
#             sustain = global_params["sustain"]
#             release = global_params["release"]
#             note_off = tr.full_like(attack, self.ac.note_off)
#             envelope = self.adsr(
#                 note_off,
#                 attack,
#                 decay,
#                 sustain,
#                 release,
#                 note_on_duration,
#             )
#         else:
#             envelope = util.interpolate_dim(envelope, n_samples)
#         # TODO(cm): tmp
#         logits = temp_params.get("sub_lfo")
#         if logits is not None:
#             logits = logits.unsqueeze(2)
#             logits = logits.unsqueeze(3)
#             temp_params["logits"] = logits
#
#         synth_out = super().forward(
#             n_samples,
#             f0_hz,
#             phase,
#             temp_params,
#             global_params,
#             envelope,
#         )
#         synth_out["wet"] = synth_out["filtered_audio"]
#         return synth_out
#
#
# class WavetableSynthShan(WavetableSynth):
#     """
#     From Differentiable Wavetable Synthesis, Shan et al.
#     Main difference with our WavetableSynth is that Shan et al. uses weighted sum
#     (attention) instead of grid_sample to aggregate across wavetable positions.
#     """
#
#     def __init__(
#         self,
#         ac: AudioConfig,
#         n_pos: int,
#         n_wt_samples: int,
#         aa_filter_n: int,
#         wt_name: Optional[str] = None,
#         is_trainable: bool = True,
#     ):
#         super().__init__(ac, n_pos, n_wt_samples, aa_filter_n, wt_name, is_trainable)
#         wt = None
#         if wt_name is not None:
#             wt_path = os.path.join(WAVETABLES_DIR, wt_name)
#             assert os.path.isfile(wt_path)
#             log.info(f"Loading wavetable from {wt_path}")
#             wt = tr.load(wt_path)
#             assert wt.shape == (n_pos, n_wt_samples)
#
#         self.osc = WavetableOscShan(
#             ac.sr,
#             n_pos=n_pos,
#             n_wt_samples=n_wt_samples,
#             aa_filter_n=aa_filter_n,
#             wt=wt,
#             is_trainable=is_trainable,
#         )
#
#     def additive_synthesis(
#         self,
#         n_samples: int,
#         f0_hz: T,
#         phase: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#     ) -> (T, Dict[str, T]):
#         attention_matrix = temp_params["add_lfo"]
#         attention_matrix = tr.swapaxes(attention_matrix, 1, 2)
#         attention_matrix = util.interpolate_dim(attention_matrix, self.ac.n_samples)
#         temp_params["attention_matrix"] = attention_matrix
#         dry_audio = self.osc(f0_hz, attention_matrix, n_samples=n_samples, phase=phase)
#         return dry_audio, {}
#
#
# class DDSPSynth(SynthBase):
#     def __init__(
#         self,
#         ac: AudioConfig,
#         n_harmonics: int = 100,
#         n_bands: int = 65,
#     ):
#         super().__init__(ac)
#         self.osc = DDSPHarmonicOsc(
#             ac.sr,
#             n_harmonics=n_harmonics,
#         )
#         self.n_bands = n_bands
#         self.adsr = ADSR(n_frames=ac.n_samples)
#
#     def additive_synthesis(
#         self,
#         n_samples: int,
#         f0_hz: T,
#         phase: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#     ) -> (T, Dict[str, T]):
#         # apply scaling function to the sigmoid output as per Section B.5, Eq.5 in the DDSP paper
#         temp_params["add_lfo"] = DDSPSynth.scale_function(temp_params["add_lfo"])
#
#         harmonic_amplitudes = temp_params["add_lfo"][..., : self.osc.n_harmonics + 1]
#         harmonic_amplitudes = util.interpolate_dim(
#             harmonic_amplitudes, self.ac.n_samples, dim=1
#         )
#         temp_params["harmonic_amplitudes"] = harmonic_amplitudes
#
#         noise_amplitudes = temp_params["add_lfo"][..., self.osc.n_harmonics + 1 :]
#         assert (
#             noise_amplitudes.size(2) == self.n_bands
#         ), f"Noise amplitudes size mismatch, expected {self.n_bands,} but got {noise_amplitudes.size(2)}"
#         temp_params["noise_amplitudes"] = noise_amplitudes
#
#         # harmonic part
#         harmonic = self.osc(
#             f0_hz,
#             harmonic_amplitudes,
#             n_samples=n_samples,
#             phase=phase,
#         )
#
#         # noise part, get noise filter IRs from noise band amplitudes
#         block_size = round(n_samples / noise_amplitudes.size(1))
#         impulse = DDSPSynth.amp_to_impulse_response(noise_amplitudes, block_size)
#         noise = (
#             tr.rand(
#                 impulse.shape[0],
#                 impulse.shape[1],
#                 block_size,
#             ).to(impulse)
#             * 2
#             - 1
#         )
#
#         noise = DDSPSynth.fft_convolve(noise, impulse)
#         noise = noise.reshape(noise.shape[0], -1)
#
#         # NOTE: because we use 2001 frames, there will be `block_size` residuals, so we trim them
#         noise = noise[:, :n_samples]
#
#         dry_audio = harmonic + noise
#         return dry_audio, {}
#
#     def amp_to_impulse_response(amp: T, target_size: int) -> T:
#         amp = tr.stack([amp, tr.zeros_like(amp)], -1)
#         amp = tr.view_as_complex(amp)
#         amp = fft.irfft(amp)
#
#         filter_size = amp.shape[-1]
#
#         amp = tr.roll(amp, filter_size // 2, -1)
#         win = tr.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
#
#         amp = amp * win
#
#         amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
#         amp = tr.roll(amp, -filter_size // 2, -1)
#
#         return amp
#
#     def fft_convolve(signal: T, kernel: T) -> T:
#         signal = nn.functional.pad(signal, (0, signal.shape[-1]))
#         kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))
#
#         output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
#         output = output[..., output.shape[-1] // 2 :]
#
#         return output
#
#     def scale_function(
#         x_sigmoid: T,
#         eps: float = 1e-7,
#     ):
#         # NOTE: we assume that x_sigmoid already has sigmoid applied by the model
#         assert (
#             x_sigmoid.min() >= 0 and x_sigmoid.max() <= 1
#         ), f"Expected x_sigmoid to be in range [0, 1], but got {x_sigmoid.min(), x_sigmoid.max()}"
#         return 2 * (x_sigmoid ** tr.log(tr.tensor(10).to(x_sigmoid.device))) + eps
#
#     def forward(
#         self,
#         n_samples: int,
#         f0_hz: T,
#         phase: T,
#         temp_params: Dict[str, T],
#         global_params: Dict[str, T],
#         envelope: Optional[T] = None,
#         note_on_duration: Optional[T] = None,
#     ) -> Dict[str, T]:
#         synth_out = super().forward(
#             n_samples,
#             f0_hz,
#             phase,
#             temp_params,
#             global_params,
#             envelope,
#             note_on_duration,
#         )
#         synth_out["wet"] = synth_out["filtered_audio"]
#         return synth_out
