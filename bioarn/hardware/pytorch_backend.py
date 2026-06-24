"""PyTorch-backed implementation of the Bio-ARN neuromorphic HAL."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch
import torch.nn.functional as F

from bioarn.config import SpikingConfig
from bioarn.core.spiking import LIFNeuron
from bioarn.hardware.backend import (
    EnergyEstimate,
    HardwareInfo,
    NeuronGroupHandle,
    NeuromorphicBackend,
    StepResult,
    SynapseHandle,
)


@dataclass
class _NeuronGroupState:
    handle: NeuronGroupHandle
    neuron: LIFNeuron
    pending_input: torch.Tensor
    last_spikes: torch.Tensor
    last_potentials: torch.Tensor


@dataclass
class _SynapseState:
    handle: SynapseHandle
    weights: torch.Tensor
    learning_rule: str
    learning_rate: float


class PyTorchBackend(NeuromorphicBackend):
    """PyTorch-based simulation of neuromorphic operations."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._group_counter = itertools.count()
        self._synapse_counter = itertools.count()
        self._timestep = 0
        self._groups: dict[str, _NeuronGroupState] = {}
        self._synapses: dict[str, _SynapseState] = {}
        self._energy_breakdown = {
            "neuron_flops_joules": 0.0,
            "synapse_flops_joules": 0.0,
            "spike_events_joules": 0.0,
        }
        self._operation_counts = {
            "neuron_flops": 0,
            "synapse_flops": 0,
            "spike_events": 0,
        }

    def _flop_energy(self) -> float:
        if self.device.type == "cuda":
            return 10e-12
        return 20e-12

    def _gpu_reference_flop_energy(self) -> float:
        return 1e-12

    def _new_group_handle(self, num_neurons: int, neuron_type: str) -> NeuronGroupHandle:
        return NeuronGroupHandle(
            id=f"group_{next(self._group_counter)}",
            num_neurons=int(num_neurons),
            neuron_type=str(neuron_type).lower(),
            backend="pytorch",
        )

    def _new_synapse_handle(
        self,
        source: NeuronGroupHandle,
        target: NeuronGroupHandle,
        num_synapses: int,
        learning_rule: str,
    ) -> SynapseHandle:
        return SynapseHandle(
            id=f"synapse_{next(self._synapse_counter)}",
            source=source.id,
            target=target.id,
            num_synapses=int(num_synapses),
            learning_rule=learning_rule,
        )

    def _as_batch(self, tensor: torch.Tensor, expected_neurons: int) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError("Spike patterns must be 1D or 2D tensors.")
        if tensor.shape[-1] != expected_neurons:
            raise ValueError(
                f"Expected {expected_neurons} neurons in spike pattern, got {tensor.shape[-1]}."
            )
        return tensor

    def create_neuron_group(
        self,
        num_neurons: int,
        neuron_type: str,
        params: dict | None = None,
    ) -> NeuronGroupHandle:
        params = {} if params is None else dict(params)
        if int(num_neurons) <= 0:
            raise ValueError("num_neurons must be positive.")
        if str(neuron_type).lower() != "lif":
            raise ValueError("PyTorchBackend currently supports only 'lif' neuron groups.")

        config = SpikingConfig(
            beta=float(params.get("beta", 0.9)),
            threshold=float(params.get("threshold", 1.0)),
            reset=float(params.get("reset", 0.0)),
            dt=float(params.get("dt", 1.0)),
            refractory_steps=int(params.get("refractory_steps", 2)),
        )
        handle = self._new_group_handle(num_neurons=num_neurons, neuron_type=neuron_type)
        neuron = LIFNeuron(num_neurons=num_neurons, config=config).to(device=self.device, dtype=self.dtype)
        empty = torch.zeros((1, int(num_neurons)), device=self.device, dtype=self.dtype)
        self._groups[handle.id] = _NeuronGroupState(
            handle=handle,
            neuron=neuron,
            pending_input=empty.clone(),
            last_spikes=empty.clone(),
            last_potentials=empty.clone(),
        )
        return handle

    def create_synapse(
        self,
        source: NeuronGroupHandle,
        target: NeuronGroupHandle,
        weights: torch.Tensor,
        learning_rule: str,
    ) -> SynapseHandle:
        if source.id not in self._groups or target.id not in self._groups:
            raise KeyError("Source and target neuron groups must be created before wiring synapses.")

        learning_rule = str(learning_rule).lower()
        if learning_rule not in {"fixed", "hebbian", "stdp"}:
            raise ValueError("learning_rule must be one of: fixed, hebbian, stdp.")

        weight_tensor = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
        expected_shape = (target.num_neurons, source.num_neurons)
        if tuple(weight_tensor.shape) != expected_shape:
            raise ValueError(
                f"Synapse weights must have shape {expected_shape}, got {tuple(weight_tensor.shape)}."
            )

        handle = self._new_synapse_handle(
            source=source,
            target=target,
            num_synapses=weight_tensor.numel(),
            learning_rule=learning_rule,
        )
        self._synapses[handle.id] = _SynapseState(
            handle=handle,
            weights=weight_tensor.clone(),
            learning_rule=learning_rule,
            learning_rate=0.01 if learning_rule == "hebbian" else 0.005,
        )
        return handle

    def inject_spikes(self, neuron_group: NeuronGroupHandle, spike_pattern: torch.Tensor) -> None:
        state = self._groups[neuron_group.id]
        injected = self._as_batch(spike_pattern, neuron_group.num_neurons)
        if state.pending_input.shape[0] != injected.shape[0]:
            state.pending_input = torch.zeros_like(injected)
            state.last_spikes = torch.zeros_like(injected)
            state.last_potentials = torch.zeros_like(injected)
            state.neuron.reset_state(
                batch_size=injected.shape[0],
                device=self.device,
                dtype=self.dtype,
            )
        state.pending_input = state.pending_input + injected

    def _apply_learning(self, synapse: _SynapseState, source_spikes: torch.Tensor, target_spikes: torch.Tensor) -> None:
        if synapse.learning_rule == "fixed":
            return

        batch_size = max(source_spikes.shape[0], 1)
        potentiation = target_spikes.transpose(0, 1) @ source_spikes
        if synapse.learning_rule == "hebbian":
            delta = synapse.learning_rate * (potentiation / batch_size)
        else:
            depression = (1.0 - target_spikes).transpose(0, 1) @ source_spikes
            delta = synapse.learning_rate * ((potentiation - 0.5 * depression) / batch_size)

        synapse.weights = torch.clamp(synapse.weights + delta, min=-1.0, max=1.0)

    def step(self, num_steps: int = 1) -> StepResult:
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        latest_spikes: dict[str, torch.Tensor] = {}
        latest_potentials: dict[str, torch.Tensor] = {}

        for step_idx in range(num_steps):
            currents = {
                group_id: state.pending_input.clone()
                for group_id, state in self._groups.items()
            }
            for state in self._groups.values():
                state.pending_input.zero_()

            for synapse in self._synapses.values():
                source_state = self._groups[synapse.handle.source]
                target_state = self._groups[synapse.handle.target]
                source_spikes = source_state.last_spikes
                if source_spikes.numel() == 0:
                    continue
                target_current = F.linear(source_spikes, synapse.weights)
                if currents[target_state.handle.id].shape != target_current.shape:
                    currents[target_state.handle.id] = torch.zeros_like(target_current)
                currents[target_state.handle.id] = currents[target_state.handle.id] + target_current

                synapse_flops = source_spikes.shape[0] * synapse.weights.numel() * 2
                self._operation_counts["synapse_flops"] += int(synapse_flops)
                self._energy_breakdown["synapse_flops_joules"] += synapse_flops * self._flop_energy()

            current_step_spikes: dict[str, torch.Tensor] = {}
            for group_id, state in self._groups.items():
                spikes, potentials = state.neuron(currents[group_id])
                state.last_spikes = spikes.detach().clone()
                state.last_potentials = potentials.detach().clone()
                current_step_spikes[group_id] = state.last_spikes
                latest_spikes[group_id] = state.last_spikes
                latest_potentials[group_id] = state.last_potentials

                neuron_flops = currents[group_id].numel() * 6
                spike_events = int((spikes > 0).sum().item())
                self._operation_counts["neuron_flops"] += int(neuron_flops)
                self._operation_counts["spike_events"] += spike_events
                self._energy_breakdown["neuron_flops_joules"] += neuron_flops * self._flop_energy()
                self._energy_breakdown["spike_events_joules"] += spike_events * 23e-12

            for synapse in self._synapses.values():
                source_spikes = self._groups[synapse.handle.source].last_spikes
                target_spikes = current_step_spikes[synapse.handle.target]
                self._apply_learning(synapse, source_spikes, target_spikes)

            self._timestep += 1
            if step_idx < num_steps - 1:
                latest_spikes = {}
                latest_potentials = {}

        return StepResult(spikes=latest_spikes, potentials=latest_potentials, timestep=self._timestep)

    def read_spikes(self, neuron_group: NeuronGroupHandle) -> torch.Tensor:
        return self._groups[neuron_group.id].last_spikes.detach().clone()

    def update_weights(self, synapse: SynapseHandle, new_weights: torch.Tensor) -> None:
        state = self._synapses[synapse.id]
        weights = torch.as_tensor(new_weights, device=self.device, dtype=self.dtype)
        expected_shape = self._synapses[synapse.id].weights.shape
        if tuple(weights.shape) != tuple(expected_shape):
            raise ValueError(
                f"Updated weights must have shape {tuple(expected_shape)}, got {tuple(weights.shape)}."
            )
        state.weights = weights.clone()

    def reset(self) -> None:
        self._timestep = 0
        for state in self._groups.values():
            batch_size = state.last_spikes.shape[0] if state.last_spikes.numel() else 1
            state.neuron.reset_state(batch_size=batch_size, device=self.device, dtype=self.dtype)
            state.pending_input.zero_()
            state.last_spikes.zero_()
            state.last_potentials.zero_()
        for key in self._energy_breakdown:
            self._energy_breakdown[key] = 0.0
        for key in self._operation_counts:
            self._operation_counts[key] = 0

    def get_energy_estimate(self) -> EnergyEstimate:
        total_joules = float(sum(self._energy_breakdown.values()))
        gpu_reference_joules = (
            (self._operation_counts["neuron_flops"] + self._operation_counts["synapse_flops"])
            * self._gpu_reference_flop_energy()
        ) + (self._operation_counts["spike_events"] * 23e-12)
        comparison_gpu = total_joules / max(gpu_reference_joules, 1e-18)
        return EnergyEstimate(
            total_joules=total_joules,
            breakdown=dict(self._energy_breakdown),
            watts_at_freq=float(total_joules * 1000.0),
            comparison_gpu=float(comparison_gpu),
        )

    def get_hardware_info(self) -> HardwareInfo:
        return HardwareInfo(
            backend_name="PyTorch Simulation",
            max_neurons=100_000_000,
            max_synapses=1_000_000_000,
            supports_learning=True,
            supports_stdp=True,
            native_spike=False,
            estimated_power_per_spike=23e-12,
        )


__all__ = ["PyTorchBackend"]
