"""Hardware abstraction primitives for neuromorphic Bio-ARN deployments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, SDMConfig


@dataclass(frozen=True)
class NeuronGroupHandle:
    """Reference to a backend-managed neuron population."""

    id: str
    num_neurons: int
    neuron_type: str
    backend: str


@dataclass(frozen=True)
class SynapseHandle:
    """Reference to a backend-managed synapse projection."""

    id: str
    source: str
    target: str
    num_synapses: int
    learning_rule: str


@dataclass
class StepResult:
    """Result of advancing a backend by one or more timesteps."""

    spikes: dict[str, torch.Tensor]
    potentials: dict[str, torch.Tensor]
    timestep: int


@dataclass
class EnergyEstimate:
    """Backend energy estimate for work executed so far."""

    total_joules: float
    breakdown: dict[str, float]
    watts_at_freq: float
    comparison_gpu: float


@dataclass
class HardwareInfo:
    """Backend capability summary."""

    backend_name: str
    max_neurons: int
    max_synapses: int
    supports_learning: bool
    supports_stdp: bool
    native_spike: bool
    estimated_power_per_spike: float


@dataclass
class MappedComponent:
    """Hardware resource estimate for a Bio-ARN component."""

    name: str
    neuron_count: int
    synapse_count: int
    memory_bytes: int
    learning_rules: list[str]
    notes: str


@dataclass
class SystemMapping:
    """Resource estimate for a full Bio-ARN deployment."""

    components: list[MappedComponent]
    total_neurons: int
    total_synapses: int
    total_memory_bytes: int
    estimated_power_watts: float
    estimated_die_area_mm2: float


@dataclass
class PowerEstimate:
    """Power estimate across inference, training, and idle modes."""

    inference_watts: float
    training_watts: float
    idle_watts: float
    comparison: dict[str, float]


@dataclass
class LatencyEstimate:
    """Latency estimate for a mapped system."""

    inference_ms: float
    pipeline_stages: int
    bottleneck: str


@dataclass
class ComparisonReport:
    """Side-by-side backend comparison for a Bio-ARN configuration."""

    system_mapping: SystemMapping
    power_watts: dict[str, float]
    latency_ms: dict[str, float]
    area_mm2: dict[str, float]
    neuron_utilization: dict[str, float]
    summary: str


class NeuromorphicBackend(ABC):
    """Abstract interface for neuromorphic computation backends."""

    @abstractmethod
    def create_neuron_group(
        self,
        num_neurons: int,
        neuron_type: str,
        params: dict | None = None,
    ) -> NeuronGroupHandle:
        """Create a hardware-managed neuron population."""

    @abstractmethod
    def create_synapse(
        self,
        source: NeuronGroupHandle,
        target: NeuronGroupHandle,
        weights: torch.Tensor,
        learning_rule: str,
    ) -> SynapseHandle:
        """Create a hardware-managed synaptic projection."""

    @abstractmethod
    def inject_spikes(self, neuron_group: NeuronGroupHandle, spike_pattern: torch.Tensor) -> None:
        """Inject a spike tensor into a neuron population."""

    @abstractmethod
    def step(self, num_steps: int = 1) -> StepResult:
        """Advance the backend by one or more timesteps."""

    @abstractmethod
    def read_spikes(self, neuron_group: NeuronGroupHandle) -> torch.Tensor:
        """Read the latest spike state from a neuron population."""

    @abstractmethod
    def update_weights(self, synapse: SynapseHandle, new_weights: torch.Tensor) -> None:
        """Replace the weights of a managed synapse projection."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all backend state."""

    @abstractmethod
    def get_energy_estimate(self) -> EnergyEstimate:
        """Return the backend energy estimate for accumulated work."""

    @abstractmethod
    def get_hardware_info(self) -> HardwareInfo:
        """Return backend capabilities and coarse hardware constraints."""


class ComponentMapper:
    """Maps high-level Bio-ARN components onto backend primitives."""

    def __init__(self, backend: NeuromorphicBackend):
        self.backend = backend

    @staticmethod
    def _memory_from_synapses(synapse_count: int, bytes_per_weight: int = 4) -> int:
        return int(synapse_count * bytes_per_weight)

    @staticmethod
    def _scale_component(
        component: MappedComponent,
        multiplier: int,
        name: str,
        notes_suffix: str,
    ) -> MappedComponent:
        return MappedComponent(
            name=name,
            neuron_count=component.neuron_count * multiplier,
            synapse_count=component.synapse_count * multiplier,
            memory_bytes=component.memory_bytes * multiplier,
            learning_rules=list(component.learning_rules),
            notes=f"{component.notes} {notes_suffix}".strip(),
        )

    def map_ccc(self, ccc_config: CCCConfig) -> MappedComponent:
        """Map a concept cell cluster to spiking groups and local synapses."""

        f1_neurons = int(ccc_config.num_f1_features)
        f2_neurons = int(ccc_config.concept_dim)
        comparator_neurons = 2
        competitive_neurons = max(1, ccc_config.f1_top_k)

        feedforward_synapses = ccc_config.input_dim * ccc_config.num_f1_features
        concept_synapses = ccc_config.num_f1_features * ccc_config.concept_dim
        feedback_synapses = ccc_config.concept_dim * ccc_config.num_f1_features
        inhibitory_synapses = ccc_config.num_f1_features * max(ccc_config.num_f1_features - 1, 0)

        neuron_count = f1_neurons + f2_neurons + comparator_neurons + competitive_neurons
        synapse_count = (
            feedforward_synapses
            + concept_synapses
            + feedback_synapses
            + inhibitory_synapses
        )
        memory_bytes = self._memory_from_synapses(synapse_count) + ((f2_neurons + comparator_neurons) * 4)

        notes = (
            "CCC maps to sparse F1 feature neurons, an F2 concept population, a margin gate "
            "comparator, and feedback synapses for top-down prediction. Learning requires fixed "
            "feedforward weights plus local Hebbian updates on concept and feedback projections."
        )
        return MappedComponent(
            name="ccc",
            neuron_count=int(neuron_count),
            synapse_count=int(synapse_count),
            memory_bytes=int(memory_bytes),
            learning_rules=["fixed", "hebbian"],
            notes=notes,
        )

    def map_sdm(self, sdm_config: SDMConfig) -> MappedComponent:
        """Map Kanerva sparse distributed memory to spiking storage primitives."""

        address_neurons = int(sdm_config.address_dim)
        hard_location_neurons = int(sdm_config.num_hard_locations)
        data_neurons = int(sdm_config.data_dim)
        comparator_neurons = int(sdm_config.num_hard_locations)

        address_synapses = sdm_config.num_hard_locations * sdm_config.address_dim
        storage_synapses = sdm_config.num_hard_locations * sdm_config.data_dim
        inhibitory_synapses = sdm_config.num_hard_locations * max(sdm_config.data_dim // 4, 1)

        address_memory = math.ceil(address_synapses / 8)
        storage_memory = storage_synapses * 4
        counter_memory = sdm_config.num_hard_locations * 4

        notes = (
            "SDM maps to hard-location neurons, sparse address comparators for Hamming distance, "
            "and data-storage synapses for associative recall. Retrieval activates locations within "
            "the configured radius; temporal associations can use STDP-like synaptic updates."
        )
        return MappedComponent(
            name="sdm",
            neuron_count=address_neurons + hard_location_neurons + data_neurons + comparator_neurons,
            synapse_count=address_synapses + storage_synapses + inhibitory_synapses,
            memory_bytes=int(address_memory + storage_memory + counter_memory),
            learning_rules=["fixed", "hebbian", "stdp"],
            notes=notes,
        )

    def map_pe_level(self, input_dim: int, output_dim: int) -> MappedComponent:
        """Map a predictive coding level to prediction, error, and state populations."""

        prediction_neurons = int(input_dim)
        error_neurons = int(input_dim)
        state_neurons = int(output_dim)
        precision_registers = int(input_dim)

        prediction_synapses = int(input_dim * output_dim)
        feedback_synapses = int(input_dim * output_dim)

        notes = (
            "A predictive coding level uses state neurons to generate predictions, error neurons "
            "to encode mismatches, and reciprocal synapses for top-down prediction plus bottom-up "
            "error correction. Local Hebbian learning updates the shared weight matrix."
        )
        return MappedComponent(
            name=f"pc_level_{input_dim}x{output_dim}",
            neuron_count=prediction_neurons + error_neurons + state_neurons,
            synapse_count=prediction_synapses + feedback_synapses,
            memory_bytes=self._memory_from_synapses(prediction_synapses + feedback_synapses)
            + (precision_registers * 4),
            learning_rules=["hebbian"],
            notes=notes,
        )

    def map_gnw(self, gnw_config: GNWConfig) -> MappedComponent:
        """Map the global neuronal workspace to competition and broadcast circuits."""

        slot_neurons = int(gnw_config.capacity * 3)
        competition_neurons = int(gnw_config.capacity * 2)
        fatigue_registers = int(gnw_config.capacity)
        bus_neurons = 16

        competition_synapses = int(gnw_config.capacity * gnw_config.capacity)
        broadcast_synapses = int(gnw_config.capacity * 64)

        notes = (
            "GNW requires a winner-take-most competition circuit, a broadcast bus for selected "
            "concepts, and per-slot fatigue counters that suppress repeated winners over time."
        )
        return MappedComponent(
            name="gnw",
            neuron_count=slot_neurons + competition_neurons + fatigue_registers + bus_neurons,
            synapse_count=competition_synapses + broadcast_synapses,
            memory_bytes=self._memory_from_synapses(competition_synapses + broadcast_synapses)
            + (fatigue_registers * 4),
            learning_rules=["fixed"],
            notes=notes,
        )

    def map_full_system(self, bioarn_config: BioARNConfig) -> SystemMapping:
        """Estimate full-system resources for a Bio-ARN deployment."""

        ccc_single = self.map_ccc(bioarn_config.ccc)
        ccc_pool = self._scale_component(
            ccc_single,
            bioarn_config.ccc.max_pool_size,
            name="ccc_pool",
            notes_suffix=f"Scaled across {bioarn_config.ccc.max_pool_size} pooled columns.",
        )
        sdm_component = self.map_sdm(bioarn_config.sdm)
        gnw_component = self.map_gnw(bioarn_config.gnw)

        predictive_components: list[MappedComponent] = []
        predictive_base = max(int(bioarn_config.ccc.concept_dim), 8)
        predictive_levels = max(int(bioarn_config.predictive.num_levels), 1)
        current_dim = predictive_base
        for level_idx in range(predictive_levels):
            next_dim = max(8, predictive_base // (2 ** (level_idx + 1)))
            predictive_components.append(self.map_pe_level(current_dim, next_dim))
            current_dim = next_dim

        components = [ccc_pool, sdm_component, *predictive_components, gnw_component]
        total_neurons = sum(component.neuron_count for component in components)
        total_synapses = sum(component.synapse_count for component in components)
        total_memory_bytes = sum(component.memory_bytes for component in components)

        info = self.backend.get_hardware_info()
        spike_events_per_inference = (total_neurons * 0.1) + (total_synapses * 0.01)
        synapse_event_energy = max(info.estimated_power_per_spike / 4.0, 1e-13)
        inference_energy = (
            (total_neurons * 0.1 * info.estimated_power_per_spike)
            + (total_synapses * 0.01 * synapse_event_energy)
        )
        estimated_power_watts = float(inference_energy * 1000.0)

        die_area_from_neurons = total_neurons * 1.5e-5
        die_area_from_synapses = total_synapses * 4.0e-7
        die_area_from_memory = total_memory_bytes * 1.0e-7
        estimated_die_area_mm2 = float(
            die_area_from_neurons + die_area_from_synapses + die_area_from_memory
        )

        return SystemMapping(
            components=components,
            total_neurons=int(total_neurons),
            total_synapses=int(total_synapses),
            total_memory_bytes=int(total_memory_bytes),
            estimated_power_watts=estimated_power_watts,
            estimated_die_area_mm2=estimated_die_area_mm2,
        )


__all__ = [
    "ComparisonReport",
    "ComponentMapper",
    "EnergyEstimate",
    "HardwareInfo",
    "LatencyEstimate",
    "MappedComponent",
    "NeuromorphicBackend",
    "NeuronGroupHandle",
    "PowerEstimate",
    "StepResult",
    "SynapseHandle",
    "SystemMapping",
]
