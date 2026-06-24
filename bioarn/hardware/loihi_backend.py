"""Intel Loihi 2 HAL stub for Bio-ARN neuromorphic execution."""

from __future__ import annotations

import importlib.util

import torch

from bioarn.hardware.backend import (
    EnergyEstimate,
    HardwareInfo,
    NeuronGroupHandle,
    NeuromorphicBackend,
    StepResult,
    SynapseHandle,
)


class LoihiBackend(NeuromorphicBackend):
    """Intel Loihi 2 backend (requires the Lava SDK)."""

    _NOT_IMPLEMENTED = (
        "LoihiBackend is a documented stub. Install Intel Lava, then map Bio-ARN primitives to "
        "Loihi 2 compartments, synapses, and learning engines before using this backend."
    )

    def check_lava_available(self) -> bool:
        """Return whether the Lava SDK can be imported in the current environment."""

        return importlib.util.find_spec("lava") is not None

    def _raise(self, detail: str) -> None:
        raise NotImplementedError(f"{detail} {self._NOT_IMPLEMENTED}")

    def create_neuron_group(
        self,
        num_neurons: int,
        neuron_type: str,
        params: dict | None = None,
    ) -> NeuronGroupHandle:
        """Map Bio-ARN LIF populations to Loihi 2 compartment groups."""

        del num_neurons, neuron_type, params
        self._raise(
            "LIF neurons map to Loihi 2 compartments with per-compartment thresholds, leak, and reset."
        )

    def create_synapse(
        self,
        source: NeuronGroupHandle,
        target: NeuronGroupHandle,
        weights: torch.Tensor,
        learning_rule: str,
    ) -> SynapseHandle:
        """Map Bio-ARN synapses to Loihi 2 connection tables and learning rules."""

        del source, target, weights, learning_rule
        self._raise(
            "Bio-ARN synapses map to Loihi 2 connection memories; Hebbian and STDP rules map to the learning engine."
        )

    def inject_spikes(self, neuron_group: NeuronGroupHandle, spike_pattern: torch.Tensor) -> None:
        """Map sensory spikes to Loihi 2 spike generators and input ports."""

        del neuron_group, spike_pattern
        self._raise(
            "Input spike streams would be emitted by Loihi spike generators and routed onto the on-chip fabric."
        )

    def step(self, num_steps: int = 1) -> StepResult:
        """Map backend stepping to Loihi 2 execution epochs scheduled by Lava."""

        del num_steps
        self._raise(
            "Loihi execution would run in epochs, with CCC margin gates mapped to threshold comparators and GNW inhibition to inhibitory projections."
        )

    def read_spikes(self, neuron_group: NeuronGroupHandle) -> torch.Tensor:
        """Map output reads to Loihi 2 monitor probes."""

        del neuron_group
        self._raise(
            "Spike readout would use Loihi probes; SDM retrieval and GNW broadcast would be observed through monitored ports."
        )

    def update_weights(self, synapse: SynapseHandle, new_weights: torch.Tensor) -> None:
        """Map weight updates to Loihi 2 synaptic memory programming."""

        del synapse, new_weights
        self._raise(
            "Manual weight updates would reprogram Loihi synaptic memory; STDP could instead run in hardware via the learning engine."
        )

    def reset(self) -> None:
        """Map resets to Loihi 2 compartment state clears and epoch restarts."""

        self._raise(
            "Reset would clear compartment voltages, spike traces, fatigue counters, and any predictive-coding error accumulators."
        )

    def get_energy_estimate(self) -> EnergyEstimate:
        """Map Bio-ARN activity accounting to Loihi 2 spike and leakage energy models."""

        self._raise(
            "Energy accounting would sum spike events, synaptic memory accesses, SDM routing, and leakage using published Loihi 2 energy figures."
        )

    def get_hardware_info(self) -> HardwareInfo:
        """Return coarse Loihi 2 capabilities used for estimation and planning."""

        return HardwareInfo(
            backend_name="Intel Loihi 2",
            max_neurons=1_000_000,
            max_synapses=120_000_000,
            supports_learning=True,
            supports_stdp=True,
            native_spike=True,
            estimated_power_per_spike=23e-12,
        )


__all__ = ["LoihiBackend"]
