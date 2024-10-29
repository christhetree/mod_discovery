import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch as tr
from torch import Tensor as T, nn

import util
from acid_ddsp.synth_modules import (
    SquareSawVCOLite,
    ExpDecayEnv,
    WavetableOsc,
)
from audio_config import AudioConfig
from filters import (
    TimeVaryingLPBiquad,
    calc_logits_to_biquad_a_coeff_triangle,
    time_varying_fir,
    calc_logits_to_biquad_coeff_pole_zero,
    TimeVaryingLPBiquadFSM,
    TimeVaryingIIRFSM,
    sample_wise_lpc_scriptable,
)
from paths import WAVETABLES_DIR
from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def make_synth(synth_type: str, ac: AudioConfig, **kwargs) -> "SynthBase":
    if synth_type == "AcidSynthLPBiquad":
        synth_class = AcidSynthLPBiquad
    elif synth_type == "AcidSynthLPBiquadFSM":
        synth_class = AcidSynthLPBiquadFSM
    elif synth_type == "AcidSynthLearnedBiquadCoeff":
        synth_class = AcidSynthLearnedBiquadCoeff
    elif synth_type == "AcidSynthLearnedBiquadCoeffFSM":
        synth_class = AcidSynthLearnedBiquadCoeffFSM
    elif synth_type == "AcidSynthLearnedBiquadPoleZero":
        synth_class = AcidSynthLearnedBiquadPoleZero
    elif synth_type == "AcidSynthLSTM":
        synth_class = AcidSynthLSTM
    else:
        raise ValueError(f"Unknown synth_type: {synth_type}")
    return synth_class(ac, **kwargs)


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
    ) -> (T, Dict[str, T]):
        pass

    def subtractive_synthesis(
        self,
        x: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
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
        envelope: Optional[T] = None,
        note_on_duration: Optional[T] = None,
    ) -> Dict[str, T]:
        dry_audio, additive_out = self.additive_synthesis(
            n_samples, f0_hz, phase, temp_params, global_params
        )
        if envelope is not None:
            dry_audio *= envelope
        filtered_audio, subtractive_out = self.subtractive_synthesis(
            dry_audio, temp_params, global_params
        )
        synth_out = {
            "dry": dry_audio,
            "envelope": envelope,
            "filtered_audio": filtered_audio,
        }
        synth_out.update(additive_out)
        synth_out.update(subtractive_out)
        return synth_out


class AcidSynthBase(SynthBase):
    def __init__(self, ac: AudioConfig):
        super().__init__(ac)
        self.osc = SquareSawVCOLite(ac.sr)
        self.env_gen = ExpDecayEnv(ac.sr, ac.stability_eps)

    def additive_synthesis(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        osc_shape = global_params["osc_shape"]
        osc_gain = global_params["osc_gain"]
        dry_audio = self.osc(f0_hz, osc_shape, n_samples=n_samples, phase=phase)
        dry_audio *= osc_gain.unsqueeze(-1)
        return dry_audio, {}

    def resize_mod_sig(self, x: T, w_mod_sig: T, q_mod_sig: T) -> Tuple[T, T]:
        assert w_mod_sig.ndim == q_mod_sig.ndim == x.ndim == 2
        w_mod_sig = util.linear_interpolate_dim(
            w_mod_sig, x.size(1), align_corners=True
        )
        q_mod_sig = util.linear_interpolate_dim(
            q_mod_sig, x.size(1), align_corners=True
        )
        return w_mod_sig, q_mod_sig

    def forward(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        envelope: Optional[T] = None,
        note_on_duration: Optional[T] = None,
    ) -> Dict[str, T]:
        assert False  # TODO(cm): check additive args
        osc_shape = global_params.get("osc_shape")
        if osc_shape is None:
            assert self.ac.is_fixed("osc_shape")
            osc_shape = tr.full_like(f0_hz, self.ac.min_osc_shape)
            additive_args["osc_shape"] = osc_shape
        osc_gain = additive_args.get("osc_gain")
        if osc_gain is None:
            assert self.ac.is_fixed("osc_gain")
            osc_gain = tr.full_like(f0_hz, self.ac.min_osc_gain)
            additive_args["osc_gain"] = osc_gain
        dist_gain = global_params.get("dist_gain")
        if dist_gain is None:
            assert self.ac.is_fixed("dist_gain")
            dist_gain = tr.full_like(f0_hz, self.ac.min_dist_gain)
        learned_alpha = global_params.get("learned_alpha")
        if learned_alpha is None:
            assert self.ac.is_fixed("learned_alpha")
            learned_alpha = tr.full_like(f0_hz, self.ac.min_learned_alpha)
        assert (
            osc_shape.shape
            == osc_gain.shape
            == dist_gain.shape
            == learned_alpha.shape
        )
        if envelope is None:
            assert note_on_duration is not None
            envelope = self.env_gen(learned_alpha, note_on_duration, n_samples)
        synth_out = super().forward(
            n_samples,
            f0_hz,
            phase,
            temp_params,
            global_params,
            envelope,
        )
        filtered_audio = synth_out["filtered_audio"]
        wet_audio = filtered_audio * dist_gain.unsqueeze(-1)
        wet_audio = tr.tanh(wet_audio)
        synth_out["wet"] = wet_audio
        return synth_out


class AcidSynthLPBiquad(AcidSynthBase):
    def __init__(
        self,
        ac: AudioConfig,
        interp_coeff: bool = True,
        make_scriptable=False,  # TODO(cm): remove
    ):
        super().__init__(ac)
        self.interp_coeff = interp_coeff
        self.filter = TimeVaryingLPBiquad(
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
            eps=ac.stability_eps,
        )
        self.is_scriptable = False

    def toggle_scriptable(self, is_scriptable: bool) -> None:
        self.is_scriptable = is_scriptable
        self.filter.toggle_scriptable(is_scriptable)

    def subtractive_synthesis(
        self, x: T, subtractive_args: Dict[str, T]
    ) -> Tuple[T, Dict[str, T]]:
        w_mod_sig = subtractive_args["w_mod_sig"]
        q_mod_sig = subtractive_args["q_mod_sig"]
        zi = subtractive_args.get("zi")
        if w_mod_sig.ndim == 3:
            w_mod_sig = w_mod_sig.squeeze(2)
        if q_mod_sig.ndim == 3:
            q_mod_sig = q_mod_sig.squeeze(2)
        y, a_coeff, b_coeff, y_a = self.filter(
            x,
            w_mod_sig,
            q_mod_sig,
            interp_coeff=self.interp_coeff,
            zi=zi,
        )
        filter_out = {"a_coeff": a_coeff, "b_coeff": b_coeff}
        if y_a is not None:
            filter_out["y_a"] = y_a
        return y, filter_out


class AcidSynthLPBiquadFSM(AcidSynthLPBiquad):
    def __init__(
        self,
        ac: AudioConfig,
        win_len: Optional[int] = None,
        win_len_sec: Optional[float] = None,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
        interp_coeff: bool = True,
    ):
        super().__init__(ac, interp_coeff)
        self.filter = TimeVaryingLPBiquadFSM(
            win_len=win_len,
            win_len_sec=win_len_sec,
            sr=ac.sr,
            overlap=overlap,
            oversampling_factor=oversampling_factor,
            min_w=ac.min_w,
            max_w=ac.max_w,
            min_q=ac.min_q,
            max_q=ac.max_q,
            eps=ac.stability_eps,
        )


class AcidSynthLearnedBiquadCoeff(AcidSynthBase):
    def __init__(
        self,
        ac: AudioConfig,
        interp_logits: bool = False,
        make_scriptable=False,  # TODO(cm): remove
    ):
        super().__init__(ac)
        self.interp_logits = interp_logits
        self.is_scriptable = False
        self.lpc_func = sample_wise_lpc

    def toggle_scriptable(self, is_scriptable: bool) -> None:
        self.is_scriptable = is_scriptable
        if is_scriptable:
            self.lpc_func = sample_wise_lpc_scriptable
        else:
            self.lpc_func = sample_wise_lpc

    def _calc_coeff(self, logits: T, n_frames: int) -> Tuple[T, T]:
        assert logits.ndim == 3
        bs = logits.size(0)
        if self.interp_logits:
            logits = util.linear_interpolate_dim(
                logits, n_frames, dim=1, align_corners=True
            )
            assert logits.shape == (bs, n_frames, 5)
        a_logits = logits[..., :2]
        a_coeff = calc_logits_to_biquad_a_coeff_triangle(
            a_logits, self.ac.stability_eps
        )
        b_coeff = logits[..., 2:]
        if not self.interp_logits:
            a_coeff = util.linear_interpolate_dim(
                a_coeff, n_frames, dim=1, align_corners=True
            )
            assert a_coeff.shape == (bs, n_frames, 2)
            b_coeff = util.linear_interpolate_dim(
                b_coeff, n_frames, dim=1, align_corners=True
            )
            assert b_coeff.shape == (bs, n_frames, 3)
        return a_coeff, b_coeff

    def subtractive_synthesis(
        self, x: T, subtractive_args: Dict[str, T]
    ) -> Tuple[T, Dict[str, T]]:
        logits = subtractive_args["logits"]
        assert logits.ndim == 3
        n_samples = x.size(1)
        a_coeff, b_coeff = self._calc_coeff(logits, n_samples)
        zi = subtractive_args.get("zi")
        zi_a = zi
        if zi_a is not None:
            zi_a = tr.flip(zi_a, dims=[1])  # Match scipy's convention for torchlpc
        y_a = self.lpc_func(x, a_coeff, zi=zi_a)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeff, zi=zi)
        a1 = a_coeff[:, :, 0]
        a2 = a_coeff[:, :, 1]
        a0 = tr.ones_like(a1)
        a_coeff = tr.stack([a0, a1, a2], dim=2)
        filter_out = {"a_coeff": a_coeff, "b_coeff": b_coeff, "y_a": y_a}
        return y_ab, filter_out


class AcidSynthLearnedBiquadCoeffFSM(AcidSynthLearnedBiquadCoeff):
    def __init__(
        self,
        ac: AudioConfig,
        win_len: Optional[int] = None,
        win_len_sec: Optional[float] = None,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
        interp_logits: bool = False,
    ):
        super().__init__(ac, interp_logits)
        self.filter = TimeVaryingIIRFSM(
            win_len=win_len,
            win_len_sec=win_len_sec,
            sr=ac.sr,
            overlap=overlap,
            oversampling_factor=oversampling_factor,
        )

    def subtractive_synthesis(
        self, x: T, subtractive_args: Dict[str, T]
    ) -> Tuple[T, Dict[str, T]]:
        logits = subtractive_args["logits"]
        assert logits.ndim == 3
        n_samples = x.size(1)
        n_frames = self.filter.calc_n_frames(n_samples)
        a_coeff, b_coeff = self._calc_coeff(logits, n_frames)
        a1 = a_coeff[:, :, 0]
        a2 = a_coeff[:, :, 1]
        a0 = tr.ones_like(a1)
        a_coeff = tr.stack([a0, a1, a2], dim=2)
        y = self.filter(x, a_coeff, b_coeff)
        filter_out = {"a_coeff": a_coeff, "b_coeff": b_coeff}
        return y, filter_out


class AcidSynthLearnedBiquadPoleZero(AcidSynthBase):
    def subtractive_synthesis(
        self, x: T, subtractive_args: Dict[str, T]
    ) -> Tuple[T, Dict[str, T]]:
        logits = subtractive_args["logits"]
        assert logits.ndim == 3
        # TODO(cm): enable coeff interpolation
        if logits.size(1) != x.size(1):
            logits = util.linear_interpolate_dim(
                logits, x.size(1), dim=1, align_corners=True
            )
        assert logits.shape == (x.size(0), x.size(1), 4)
        q_real = logits[..., 0]
        q_imag = logits[..., 1]
        p_real = logits[..., 2]
        p_imag = logits[..., 3]
        a_coeff, b_coeff = calc_logits_to_biquad_coeff_pole_zero(
            q_real, q_imag, p_real, p_imag, self.ac.stability_eps
        )
        y_a = sample_wise_lpc(x, a_coeff)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeff)
        a1 = a_coeff[:, :, 0]
        a2 = a_coeff[:, :, 1]
        a0 = tr.ones_like(a1)
        a_coeff = tr.stack([a0, a1, a2], dim=2)
        filter_out = {"a_coeff": a_coeff, "b_coeff": b_coeff}
        return y_ab, filter_out


class AcidSynthLSTM(AcidSynthBase):
    def __init__(self, ac: AudioConfig, n_hidden: int, n_ch: int = 1):
        super().__init__(ac)
        self.n_hidden = n_hidden
        self.n_ch = n_ch
        self.lstm = nn.LSTM(
            input_size=n_ch + 2,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out_lstm = nn.Linear(n_hidden, n_ch)

    def subtractive_synthesis(
        self, x: T, subtractive_args: Dict[str, T]
    ) -> Tuple[T, Dict[str, T]]:
        w_mod_sig = subtractive_args["w_mod_sig"]
        q_mod_sig = subtractive_args["q_mod_sig"]
        if w_mod_sig.ndim == 3:
            w_mod_sig = w_mod_sig.squeeze(2)
        if q_mod_sig.ndim == 3:
            q_mod_sig = q_mod_sig.squeeze(2)
        w_mod_sig, q_mod_sig = self.resize_mod_sig(x, w_mod_sig, q_mod_sig)

        h_n = subtractive_args.get("h_n")
        c_n = subtractive_args.get("c_n")
        hidden: Optional[Tuple[T, T]] = None
        if h_n is not None and c_n is not None:
            # This is a workaround for torchscript typing not working well with Optional
            if h_n.shape == (1, 1, self.n_hidden):
                if c_n.shape == (1, 1, self.n_hidden):
                    hidden = (h_n, c_n)

        x_orig = x
        x = tr.stack([x, w_mod_sig, q_mod_sig], dim=2)
        x, (new_h_n, new_c_n) = self.lstm(x, hidden)
        x = self.out_lstm(x)
        x = x.squeeze(-1)
        x = x + x_orig
        x = tr.tanh(x)
        filter_out = {"h_n": new_h_n, "c_n": new_c_n}
        return x, filter_out


class WavetableSynth(SynthBase):
    def __init__(
        self,
        ac: AudioConfig,
        n_pos: int,
        n_wt_samples: int,
        aa_filter_n: int,
        wt_name: Optional[str] = None,
        is_trainable: bool = True,
    ):
        super().__init__(ac)
        wt = None
        if wt_name is not None:
            wt_path = os.path.join(WAVETABLES_DIR, wt_name)
            assert os.path.isfile(wt_path)
            log.info(f"Loading wavetable from {wt_path}")
            wt = tr.load(wt_path)
            assert wt.shape == (n_pos, n_wt_samples)

        self.osc = WavetableOsc(
            ac.sr,
            n_pos=n_pos,
            n_wt_samples=n_wt_samples,
            aa_filter_n=aa_filter_n,
            wt=wt,
            is_trainable=is_trainable,
        )
        self.lpc_func = sample_wise_lpc

    def additive_synthesis(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        wt_pos = temp_params["wt_pos"]
        dry_audio = self.osc(f0_hz, wt_pos, n_samples=n_samples, phase=phase)
        return dry_audio, {}

    def _apply_filter(self, x: T, logits: T) -> T:
        # TODO(cm): reduce duplicate code
        assert logits.ndim == 3
        assert logits.size(2) == 5
        bs = logits.size(0)
        n_frames = x.size(1)
        a_logits = logits[..., :2]
        a_coeff = calc_logits_to_biquad_a_coeff_triangle(
            a_logits, self.ac.stability_eps
        )
        b_coeff = logits[..., 2:]
        a_coeff = util.linear_interpolate_dim(
            a_coeff, n_frames, dim=1, align_corners=True
        )
        assert a_coeff.shape == (bs, n_frames, 2)
        b_coeff = util.linear_interpolate_dim(
            b_coeff, n_frames, dim=1, align_corners=True
        )
        assert b_coeff.shape == (bs, n_frames, 3)
        y_a = self.lpc_func(x, a_coeff)
        assert not tr.isinf(y_a).any()
        assert not tr.isnan(y_a).any()
        y_ab = time_varying_fir(y_a, b_coeff)
        return y_ab

    def subtractive_synthesis(
        self,
        x: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
    ) -> (T, Dict[str, T]):
        # TODO(cm): tmp
        logits = temp_params.get("logits")
        if logits is None:
            return x, {}
        assert logits.ndim == 5
        filter_depth = logits.size(2)
        filter_width = logits.size(3)
        for depth_idx in range(filter_depth):
            layer_x_s = []
            for width_idx in range(filter_width):
                curr_logits = logits[:, :, depth_idx, width_idx, :]
                curr_x = self._apply_filter(x, curr_logits)
                layer_x_s.append(curr_x)
            # TODO(cm): could add a learnable attention over the filters
            x = tr.stack(layer_x_s, dim=1)
            x = tr.mean(x, dim=1)
        return x, {}

    def forward(
        self,
        n_samples: int,
        f0_hz: T,
        phase: T,
        temp_params: Dict[str, T],
        global_params: Dict[str, T],
        envelope: Optional[T] = None,
        note_on_duration: Optional[T] = None,
    ) -> Dict[str, T]:
        wt_pos = temp_params.get("add_lfo")
        # TODO(cm): tmp
        if wt_pos is None:
            wt_pos = tr.full_like(f0_hz, -1.0).unsqueeze(1)
        else:
            wt_pos = wt_pos.squeeze(2) * 2.0 - 1.0  # TODO(cm): use tanh instead
        wt_pos = util.linear_interpolate_dim(
            wt_pos, self.ac.n_samples, align_corners=True
        )
        assert envelope is not None  # TODO(cm): tmp
        if envelope is not None:
            envelope = util.linear_interpolate_dim(
                envelope, n_samples, align_corners=True
            )
        temp_params["wt_pos"] = wt_pos
        # TODO(cm): tmp
        logits = temp_params.get("sub_lfo_adapted")
        if logits is not None:
            logits = logits.unsqueeze(2)
            logits = logits.unsqueeze(3)
            temp_params["logits"] = logits

        synth_out = super().forward(
            n_samples,
            f0_hz,
            phase,
            temp_params,
            global_params,
            envelope,
        )
        synth_out["wet"] = synth_out["filtered_audio"]
        return synth_out


# if __name__ == "__main__":
#     tr.manual_seed(0)
#     ac = AudioConfig()
#     # synth = AcidSynthLPBiquad(ac, make_scriptable=True)
#     # synth = AcidSynthLPBiquadFSM(ac, win_len=128, overlap=0.75, oversampling_factor=1)
#     synth = AcidSynthLearnedBiquadCoeff(ac, make_scriptable=True)
#     # synth = AcidSynthLearnedBiquadCoeff(ac, make_scriptable=False)
#     # synth = AcidSynthLearnedBiquadCoeffFSM(
#     #     ac, win_len=128, overlap=0.75, oversampling_factor=1
#     # )
#     # synth = AcidSynthLSTM(ac, 64)
#     scripted = tr.jit.script(synth)
#     f0_hz = tr.tensor([220.0])
#     note_on_duration = tr.tensor([0.100])
#     phase = tr.tensor([0.0]).unsqueeze(1)
#     # subtractive_args = {
#     #     "w_mod_sig": tr.tensor([[0.5]]),
#     #     "q_mod_sig": tr.tensor([[0.5]]),
#     # }
#     subtractive_args = {
#         "logits": tr.rand((1, 10, 5)),
#     }
#     global_params = {
#         "osc_shape": tr.tensor([0.5]),
#         "osc_gain": tr.tensor([0.5]),
#         "dist_gain": tr.tensor([0.5]),
#         "learned_alpha": tr.tensor([0.5]),
#     }
#     synth_out = scripted(
#         f0_hz, note_on_duration, phase, subtractive_args, global_params
#     )
#     print(synth_out["wet"])
#     # tr.jit.save(scripted, "synth.ts")
