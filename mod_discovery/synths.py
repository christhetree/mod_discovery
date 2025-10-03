import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from torch import Tensor as T, nn

from mod_discovery.synth_modules import (
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
