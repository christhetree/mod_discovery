from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Sequence

import numpy as np
import torch
from synth.synth_constants import SynthConstants
from utils.gumble_softmax import gumbel_softmax
from utils.types import TensorLike


class SynthModule(ABC):
    """
    Abstract class for a synth module
    """

    def __init__(self, name: str, device: str, synth_structure: SynthConstants):
        self.synth_structure = synth_structure
        self.device = device
        self.name = name

    @abstractmethod
    def process_sound(
        self,
        input_signal: torch.Tensor,
        modulator_signal: torch.Tensor,
        params: dict,
        sample_rate: int,
        signal_duration: float,
        batch_size: int = 1,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def _mix_waveforms(
        waves_tensor: torch.Tensor,
        raw_waveform_selector: Union[str, Sequence[str], TensorLike],
        type_indices: Dict[str, int],
    ) -> torch.Tensor:

        if isinstance(raw_waveform_selector, str):
            idx = type_indices[raw_waveform_selector]
            return waves_tensor[idx]

        if isinstance(raw_waveform_selector[0], str):
            oscillator_tensor = torch.stack(
                [
                    waves_tensor[type_indices[wt]][i]
                    for i, wt in enumerate(raw_waveform_selector)
                ]
            )
            return oscillator_tensor

        oscillator_tensor = 0
        softmax = torch.nn.Softmax(dim=1)
        waveform_probabilities = softmax(raw_waveform_selector)

        for i in range(len(waves_tensor)):
            oscillator_tensor += (
                waveform_probabilities.t()[i].unsqueeze(1) * waves_tensor[i]
            )

        return oscillator_tensor

    def _standardize_input(
        self,
        input_val,
        requested_dtype,
        requested_dims: int,
        batch_size: int,
        value_range: Tuple = None,
    ) -> torch.Tensor:

        # Single scalar input value
        if isinstance(input_val, (float, np.floating, int)):
            assert (
                batch_size == 1
            ), f"Input expected to be of batch size {batch_size} but is scalar"
            input_val = torch.tensor(
                input_val, dtype=requested_dtype, device=self.device
            )

        # List, ndarray or tensor input
        if isinstance(input_val, (np.ndarray, list, np.bool_)):
            output_tensor = torch.tensor(
                input_val, dtype=requested_dtype, device=self.device
            )
        elif torch.is_tensor(input_val):
            output_tensor = input_val.to(dtype=requested_dtype, device=self.device)
        else:
            raise TypeError(
                f"Unsupported input of type {type(input_val)} to synth module"
            )

        # Add batch dim if doesn't exist
        if output_tensor.ndim == 0:
            output_tensor = torch.unsqueeze(output_tensor, dim=0)
        if output_tensor.ndim == 1 and len(output_tensor) == batch_size:
            output_tensor = torch.unsqueeze(output_tensor, dim=1)
        elif output_tensor.ndim == 1 and len(output_tensor) != batch_size:
            assert batch_size == 1, (
                f"Input expected to be of batch size {batch_size} but is of batch size 1, "
                f"shape {output_tensor.shape}"
            )
            output_tensor = torch.unsqueeze(output_tensor, dim=0)

        assert (
            output_tensor.ndim == requested_dims
        ), f"Input has unexpected number of dimensions: {output_tensor.ndim}"

        if value_range is not None:
            assert torch.all(output_tensor >= value_range[0]) and torch.all(
                output_tensor <= value_range[1]
            ), f"Parameter value outside of expected range {value_range}"

        return output_tensor

    def _verify_input_params(self, params_to_test: dict):
        expected_params = self.synth_structure.modular_synth_params[self.name]
        assert set(expected_params).issubset(
            set(params_to_test.keys())
        ), f"Expected input parameters {expected_params} but got {list(params_to_test.keys())}"

    def _process_active_signal(
        self, active_vector: TensorLike, batch_size: int
    ) -> torch.Tensor:

        if active_vector is None:
            ret_active_vector = torch.ones(
                [batch_size, 1], dtype=torch.long, device=self.device
            )
            return ret_active_vector

        if not isinstance(active_vector, torch.Tensor):
            active_vector = torch.tensor(active_vector)

        if active_vector.dtype == torch.bool:
            ret_active_vector = active_vector.long().unsqueeze(1).to(self.device)
            return ret_active_vector

        standartized_active_vector = self._standardize_input(
            active_vector,
            requested_dtype=torch.float32,
            requested_dims=2,
            batch_size=batch_size,
        )
        active_vector_gumble = gumbel_softmax(
            standartized_active_vector, hard=True, device=self.device
        )
        ret_active_vector = active_vector_gumble[:, 1:]

        return ret_active_vector


class ADSR(SynthModule):
    """
    ADSR (attack-decay-sustain-release) envelope module
    """

    def __init__(self, name: str, device: str, synth_structure: SynthConstants):
        super().__init__(name=name, device=device, synth_structure=synth_structure)

    def _build_envelope(
        self,
        params: dict,
        sample_rate: int,
        signal_duration: float,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Build ADSR envelope
        Variable note_off_time - sustain time is passed as parameter

        params:
            self: Self object with ['attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level'] parameters

        Returns:
            A torch with the constructed FM signal

        Raises:
            ValueError: Provided variables are out of range
        """
        n_samples = int(sample_rate * signal_duration)
        x = torch.linspace(0, 1.0, n_samples, device=self.device)[None, :].repeat(
            batch_size, 1
        )

        parsed_params, relative_params = {}, {}
        total_time = 0
        for k in ["attack_t", "decay_t", "sustain_t", "release_t", "sustain_level"]:
            parsed_params[k] = self._standardize_input(
                params[k],
                requested_dtype=torch.float64,
                requested_dims=2,
                batch_size=batch_size,
            )
            if k != "sustain_level":
                relative_params[k] = parsed_params[k] / signal_duration
                total_time += parsed_params[k]

        if torch.any(total_time > signal_duration):
            raise ValueError("Provided ADSR durations exceeds signal duration")

        relative_note_off = (
            relative_params["attack_t"]
            + relative_params["decay_t"]
            + relative_params["sustain_t"]
        )

        eps = 1e-5

        attack = x / (relative_params["attack_t"] + eps)
        attack = torch.clamp(attack, max=1.0)

        decay = (
            (x - relative_params["attack_t"])
            * (parsed_params["sustain_level"] - 1)
            / (relative_params["decay_t"] + eps)
        )
        decay = torch.clamp(
            decay,
            max=torch.tensor(0).to(decay.device),
            min=parsed_params["sustain_level"] - 1,
        )

        sustain = (x - relative_note_off) * (
            -parsed_params["sustain_level"] / (relative_params["release_t"] + eps)
        )
        sustain = torch.clamp(sustain, max=0.0)

        envelope = attack + decay + sustain
        envelope = torch.clamp(envelope, min=0.0, max=1.0)

        return envelope

    def process_sound(
        self,
        input_signal: torch.Tensor,
        modulator_signal: torch.Tensor,
        params: dict,
        sample_rate: int,
        signal_duration: float,
        batch_size: int = 1,
    ) -> torch.Tensor:

        envelope = self._build_envelope(
            params, sample_rate, signal_duration, batch_size
        )
        enveloped_signal = input_signal * envelope

        return enveloped_signal
