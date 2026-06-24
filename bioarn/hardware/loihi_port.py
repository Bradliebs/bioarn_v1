"""Loihi 2 mapping and validation spec for Bio-ARN 2.0.

This module is intentionally a specification and simulation aid, not a Lava
runtime. The mapping follows the public Loihi 2 programming model: 8-bit
weights, 16-bit state/activation style fixed-point arithmetic, compartment
groups, sparse routing, and on-chip x-trace learning. Energy constants are
handled in :mod:`bioarn.hardware.energy_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from bioarn.config import BioARNConfig
from bioarn.core.ccc import ConceptCellCluster
from bioarn.core.math_utils import cosine_similarity, normalize, sparse_top_k
from bioarn.core.spiking import LIFNeuron
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.predictive.pc_layer import PCLayer
from bioarn.workspace.gnw import GlobalNeuronalWorkspace


@dataclass
class LoihiNeuronSpec:
    """Fixed-point Loihi 2 compartment configuration for a Bio-ARN LIF neuron."""

    compartment_config: dict
    decay_constant: int
    threshold: int
    refractory_period: int
    notes: str


@dataclass
class LoihiCCCMapping:
    """Mapping of one Bio-ARN CCC onto Loihi 2 neurocores and synapse SRAM."""

    cores_required: int
    neurons_per_core: dict[int, int]
    synapse_memory: int
    routing_complexity: str
    custom_microcode: str
    notes: str


@dataclass
class LoihiSDMMapping:
    """Mapping of SDM addressing and storage onto Loihi 2 primitives."""

    cores_required: int
    addressing_scheme: str
    hamming_circuit_neurons: int
    storage_synapses: int
    retrieval_latency_steps: int
    notes: str


@dataclass
class LoihiPEMapping:
    """Mapping of predictive coding onto Loihi 2 pipeline stages."""

    cores_required: int
    pipeline_stages: dict[str, int]
    synapse_memory: int
    learning_engine: dict[str, int | float | str]
    latency_steps: int
    notes: str


@dataclass
class LoihiGNWMapping:
    """Mapping of the global neuronal workspace onto Loihi 2 circuits."""

    cores_required: int
    competition_neurons: int
    broadcast_fanout: int
    routing_mode: str
    adaptation_registers: int
    notes: str


@dataclass
class LoihiSystemMapping:
    """Top-level Loihi 2 deployment estimate for a Bio-ARN configuration."""

    total_cores: int
    total_chips: int
    total_neurons: int
    total_synapses: int
    memory_bytes: int
    estimated_power_watts: float
    estimated_latency_ms: float
    bottleneck: str
    feasibility: str


@dataclass
class ValidationResult:
    """Deviation summary for one float vs fixed-point equivalence check."""

    component: str
    equivalent: bool
    max_deviation: float
    mean_deviation: float
    notes: str


@dataclass
class SystemValidationResult:
    """Full-pipeline float vs fixed-point validation summary."""

    overall_equivalent: bool
    overall_accuracy_delta: float
    component_sensitivity: dict[str, float]
    worst_component: str
    notes: str


class LoihiMapping:
    """Detailed, documented mapping from Bio-ARN components to Loihi 2."""

    CORES_PER_CHIP = 128
    COMPARTMENTS_PER_CORE = 128
    WEIGHT_BITS = 8
    ACTIVATION_BITS = 16
    DECAY_REGISTER_MAX = 4095
    VOLTAGE_SCALE = 256
    BYTES_PER_SYNAPSE = 2
    SYNAPSE_SRAM_PER_CORE_BYTES = 128 * 1024

    def __init__(self, config: BioARNConfig):
        self.config = config

    @staticmethod
    def _ceildiv(numerator: int, denominator: int) -> int:
        return int(math.ceil(numerator / max(denominator, 1)))

    def _pack_population(
        self,
        start_core: int,
        num_neurons: int,
    ) -> tuple[dict[int, int], int]:
        allocation: dict[int, int] = {}
        remaining = int(num_neurons)
        core_id = int(start_core)
        while remaining > 0:
            assigned = min(remaining, self.COMPARTMENTS_PER_CORE)
            allocation[core_id] = assigned
            remaining -= assigned
            core_id += 1
        return allocation, core_id

    def _predictive_dims(self) -> list[int]:
        dims = [max(int(self.config.ccc.concept_dim), 8)]
        for _ in range(max(int(self.config.predictive.num_levels), 1)):
            dims.append(max(8, dims[-1] // 2))
        return dims

    def map_lif_neuron(self) -> LoihiNeuronSpec:
        """Map Bio-ARN LIF parameters onto a Loihi 2 compartment."""

        spiking = self.config.spiking
        decay_constant = int(
            round(max(0.0, min(float(spiking.beta), 1.0)) * self.DECAY_REGISTER_MAX)
        )
        threshold = max(1, int(round(float(spiking.threshold) * self.VOLTAGE_SCALE)))
        reset = int(round(float(spiking.reset) * self.VOLTAGE_SCALE))
        refractory = max(int(spiking.refractory_steps), 0)

        compartment_config = {
            "compartment_type": "LIF",
            "v_decay": decay_constant,
            "leak_register": self.DECAY_REGISTER_MAX - decay_constant,
            "vth": threshold,
            "v_reset": reset,
            "refractory": refractory,
            "bias_current": 0,
            "current_decay": self.DECAY_REGISTER_MAX,
            "dt_ms": float(spiking.dt),
            "fixed_point_format": "Q8.8",
        }
        notes = (
            "Bio-ARN beta is treated as the discrete Loihi membrane-retention factor. "
            "Threshold/reset are encoded in Q8.8 fixed point, while refractory_steps maps "
            "directly to the Loihi refractory register."
        )
        return LoihiNeuronSpec(
            compartment_config=compartment_config,
            decay_constant=decay_constant,
            threshold=threshold,
            refractory_period=refractory,
            notes=notes,
        )

    def map_ccc_to_cores(self) -> LoihiCCCMapping:
        """Map one CCC onto Loihi 2 feature, concept, and comparator resources."""

        ccc = self.config.ccc
        f1_allocation, next_core = self._pack_population(0, int(ccc.num_f1_features))
        f2_allocation, _ = self._pack_population(next_core, int(ccc.concept_dim))
        neurons_per_core = {**f1_allocation, **f2_allocation}

        feedforward_synapses = int(ccc.input_dim * ccc.num_f1_features)
        concept_synapses = int(ccc.num_f1_features * ccc.concept_dim)
        feedback_synapses = int(ccc.concept_dim * ccc.num_f1_features)
        f1_inhibition = int(ccc.num_f1_features * max(ccc.num_f1_features - 1, 0))
        gate_readout = int(ccc.concept_dim)
        synapse_memory = (
            feedforward_synapses
            + concept_synapses
            + feedback_synapses
            + f1_inhibition
            + gate_readout
        ) * self.BYTES_PER_SYNAPSE

        custom_microcode = (
            "acc = dot(f2_q16, concept_dir_q16)\n"
            "norm = l2_norm(f2_q16) * l2_norm(concept_dir_q16)\n"
            "confidence = acc / max(norm, 1)\n"
            "if confidence >= theta_margin_q16: emit concept_spike\n"
            "else: gate_output = 0 and set abstain_flag = 1"
        )
        notes = (
            "One shared F1 feature core feeds one or more F2 concept cores. The margin gate is "
            "implemented as a post-synaptic accumulator/threshold microcode kernel on the last F2 "
            "core rather than a separate population. Feedback weights live in synapse SRAM and route "
            "back to the F1 feature core for resonance checking."
        )
        return LoihiCCCMapping(
            cores_required=len(neurons_per_core),
            neurons_per_core=neurons_per_core,
            synapse_memory=int(synapse_memory),
            routing_complexity="local" if len(neurons_per_core) == 1 else "inter-core",
            custom_microcode=custom_microcode,
            notes=notes,
        )

    def map_sdm_to_memory(self) -> LoihiSDMMapping:
        """Map SDM hard locations, address matching, and readout onto Loihi 2."""

        sdm = self.config.sdm
        address_cores = self._ceildiv(int(sdm.address_dim), self.COMPARTMENTS_PER_CORE)
        hard_location_cores = self._ceildiv(
            int(sdm.num_hard_locations), self.COMPARTMENTS_PER_CORE
        )
        data_cores = self._ceildiv(int(sdm.data_dim), self.COMPARTMENTS_PER_CORE)

        hamming_circuit_neurons = int(sdm.num_hard_locations + self._ceildiv(sdm.address_dim, 32))
        hamming_cores = self._ceildiv(hamming_circuit_neurons, self.COMPARTMENTS_PER_CORE)
        storage_synapses = int(sdm.num_hard_locations * sdm.data_dim)
        retrieval_latency_steps = 3 + self._ceildiv(int(sdm.address_dim), 256)

        notes = (
            "Hard locations are encoded as compartment groups with fixed binary templates. Hamming "
            "distance is approximated by time-multiplexed XOR/popcount populations: 32 address bits "
            "share one comparator bank feeding a per-location accumulator. Retrieved data is emitted "
            "through spike-triggered readout synapses; binary address templates consume "
            f"{math.ceil((sdm.num_hard_locations * sdm.address_dim) / 8):,} bytes in addition to "
            "the storage synapse matrix."
        )
        return LoihiSDMMapping(
            cores_required=address_cores + hard_location_cores + data_cores + hamming_cores,
            addressing_scheme=(
                "Binary cue bits -> Hamming/XOR comparators -> popcount accumulators -> "
                "radius-threshold hard-location gates"
            ),
            hamming_circuit_neurons=hamming_circuit_neurons,
            storage_synapses=storage_synapses,
            retrieval_latency_steps=retrieval_latency_steps,
            notes=notes,
        )

    def map_pe_to_pipeline(self) -> LoihiPEMapping:
        """Map predictive coding layers onto Loihi 2 forward/error/learning stages."""

        dims = self._predictive_dims()
        prediction_neurons = 0
        error_neurons = 0
        state_neurons = 0
        precision_registers = 0
        synapses = 0

        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            prediction_neurons += input_dim
            error_neurons += input_dim * 2
            state_neurons += output_dim
            precision_registers += input_dim
            synapses += 2 * input_dim * output_dim

        total_neurons = prediction_neurons + error_neurons + state_neurons
        learning_engine = {
            "rule": "hebbian_xtrace",
            "pre_trace_bits": 12,
            "post_trace_bits": 12,
            "weight_bits": self.WEIGHT_BITS,
            "eta": float(self.config.predictive.eta),
            "gamma": float(self.config.predictive.gamma),
        }
        notes = (
            "Each predictive layer uses one prediction population, paired excitatory/inhibitory "
            "error populations for subtraction, and a state population. Precision is applied as "
            "current gain modulation rather than extra neurons. Hebbian updates map to Loihi x-trace "
            "learning with bounded 8-bit weights."
        )
        return LoihiPEMapping(
            cores_required=self._ceildiv(total_neurons, self.COMPARTMENTS_PER_CORE),
            pipeline_stages={
                "prediction": prediction_neurons,
                "error": error_neurons,
                "state": state_neurons,
                "precision_registers": precision_registers,
            },
            synapse_memory=int(synapses * self.BYTES_PER_SYNAPSE),
            learning_engine=learning_engine,
            latency_steps=max(2 * max(int(self.config.predictive.num_levels), 1), 2),
            notes=notes,
        )

    def map_gnw_to_circuit(self) -> LoihiGNWMapping:
        """Map GNW competition, broadcast, and fatigue onto Loihi 2 circuits."""

        gnw = self.config.gnw
        slot_neurons = int(gnw.capacity)
        inhibitory_neurons = int(gnw.capacity)
        bus_neurons = max(8, int(gnw.capacity) * 2)
        competition_neurons = slot_neurons + inhibitory_neurons
        total_neurons = slot_neurons + inhibitory_neurons + bus_neurons
        broadcast_fanout = int(
            gnw.capacity
            * (self.config.ccc.max_pool_size + self.config.predictive.num_levels + 2)
        )
        notes = (
            "GNW competition is implemented with strong lateral inhibition between slot neurons and a "
            "small inhibitory interneuron population. Broadcasting uses Loihi's native multicast "
            "routing. Fatigue maps cleanly to adaptation-current state variables that decay each "
            "timestep and suppress repeated winners."
        )
        return LoihiGNWMapping(
            cores_required=self._ceildiv(total_neurons, self.COMPARTMENTS_PER_CORE),
            competition_neurons=competition_neurons,
            broadcast_fanout=broadcast_fanout,
            routing_mode="multicast",
            adaptation_registers=int(gnw.capacity),
            notes=notes,
        )

    def map_full_system(self) -> LoihiSystemMapping:
        """Estimate full-system Loihi 2 resource usage for Bio-ARN."""

        ccc_map = self.map_ccc_to_cores()
        sdm_map = self.map_sdm_to_memory()
        pe_map = self.map_pe_to_pipeline()
        gnw_map = self.map_gnw_to_circuit()

        shared_f1_cores = self._ceildiv(
            int(self.config.ccc.num_f1_features), self.COMPARTMENTS_PER_CORE
        )
        per_ccc_concept_cores = max(ccc_map.cores_required - shared_f1_cores, 1)
        total_ccc_cores = shared_f1_cores + (self.config.ccc.max_pool_size * per_ccc_concept_cores)

        ccc_neurons = int(
            self.config.ccc.num_f1_features
            + (self.config.ccc.max_pool_size * self.config.ccc.concept_dim)
        )
        ccc_synapses = int(
            (self.config.ccc.input_dim * self.config.ccc.num_f1_features)
            + self.config.ccc.max_pool_size
            * (
                (2 * self.config.ccc.num_f1_features * self.config.ccc.concept_dim)
                + self.config.ccc.concept_dim
            )
            + self.config.ccc.num_f1_features
            * max(self.config.ccc.num_f1_features - 1, 0)
        )
        ccc_memory = int(
            (self.config.ccc.input_dim * self.config.ccc.num_f1_features * self.BYTES_PER_SYNAPSE)
            + self.config.ccc.max_pool_size
            * (
                (
                    2 * self.config.ccc.num_f1_features * self.config.ccc.concept_dim
                    + self.config.ccc.concept_dim
                )
                * self.BYTES_PER_SYNAPSE
            )
        )

        sdm_neurons = int(
            self.config.sdm.address_dim
            + self.config.sdm.num_hard_locations
            + self.config.sdm.data_dim
            + sdm_map.hamming_circuit_neurons
        )
        sdm_synapses = int(
            sdm_map.storage_synapses
            + math.ceil((self.config.sdm.num_hard_locations * self.config.sdm.address_dim) / 8)
        )
        sdm_memory = int(
            (sdm_map.storage_synapses * self.BYTES_PER_SYNAPSE)
            + math.ceil((self.config.sdm.num_hard_locations * self.config.sdm.address_dim) / 8)
        )

        pe_neurons = (
            pe_map.pipeline_stages["prediction"]
            + pe_map.pipeline_stages["error"]
            + pe_map.pipeline_stages["state"]
        )
        pe_synapses = pe_map.synapse_memory // self.BYTES_PER_SYNAPSE
        pe_memory = pe_map.synapse_memory + (pe_map.pipeline_stages["precision_registers"] * 2)

        gnw_neurons = int(
            gnw_map.competition_neurons + gnw_map.broadcast_fanout // max(self.config.gnw.capacity, 1)
        )
        gnw_synapses = int(
            self.config.gnw.capacity * self.config.gnw.capacity + gnw_map.broadcast_fanout
        )
        gnw_memory = gnw_synapses * self.BYTES_PER_SYNAPSE

        total_cores = int(total_ccc_cores + sdm_map.cores_required + pe_map.cores_required + gnw_map.cores_required)
        total_chips = self._ceildiv(total_cores, self.CORES_PER_CHIP)
        total_neurons = int(ccc_neurons + sdm_neurons + pe_neurons + gnw_neurons)
        total_synapses = int(ccc_synapses + sdm_synapses + pe_synapses + gnw_synapses)
        memory_bytes = int(ccc_memory + sdm_memory + pe_memory + gnw_memory)

        active_cccs = min(int(self.config.gnw.capacity), int(self.config.ccc.max_pool_size))
        from bioarn.hardware.energy_model import EnergyModel

        energy = EnergyModel().estimate_inference_energy(
            self.config,
            backend="loihi2",
            num_cccs_active=active_cccs,
        )
        latency_steps = 1 + sdm_map.retrieval_latency_steps + pe_map.latency_steps + 2
        estimated_latency_ms = float(latency_steps * self.config.spiking.dt)

        shares = {
            "ccc_concept_cores": total_ccc_cores,
            "sdm_address_routing": sdm_map.cores_required,
            "pe_error_pipeline": pe_map.cores_required,
            "gnw_multicast": gnw_map.cores_required,
        }
        bottleneck = max(shares, key=shares.get)
        if total_chips == 1:
            feasibility = "feasible"
        elif total_chips <= 32:
            feasibility = "requires_multi_chip"
        else:
            feasibility = "exceeds_current_hw"

        return LoihiSystemMapping(
            total_cores=total_cores,
            total_chips=total_chips,
            total_neurons=total_neurons,
            total_synapses=total_synapses,
            memory_bytes=memory_bytes,
            estimated_power_watts=float(energy.watts_at_1khz),
            estimated_latency_ms=estimated_latency_ms,
            bottleneck=bottleneck,
            feasibility=feasibility,
        )


class FunctionalEquivalenceValidator:
    """PyTorch proxy for float vs Loihi-style fixed-point equivalence checks."""

    ACTIVATION_INT_MAX = (2**15) - 1
    WEIGHT_INT_MAX = (2**7) - 1
    VOLTAGE_SCALE = 256

    def __init__(self, config: BioARNConfig):
        self.config = config

    @staticmethod
    def _as_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        if tensor.dim() != 2:
            raise ValueError("Expected a vector or batch of vectors.")
        return tensor, False

    @staticmethod
    def _deviation_stats(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
        diff = (reference - candidate).abs().to(torch.float32)
        return float(diff.max().item()), float(diff.mean().item())

    @staticmethod
    def _quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.clone()
        int_max = (2 ** (bits - 1)) - 1
        scale = tensor.detach().abs().max().item()
        if scale < 1e-8:
            return torch.zeros_like(tensor)
        quant = torch.clamp(torch.round(tensor / scale * int_max), -int_max, int_max)
        return quant * (scale / int_max)

    def _quantize_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        quantized = self._quantize_tensor(tensor.to(torch.float32), bits=16)
        return torch.clamp(quantized, -self.ACTIVATION_INT_MAX, self.ACTIVATION_INT_MAX)

    def _make_committed_ccc(self, prototype: torch.Tensor) -> ConceptCellCluster:
        torch.manual_seed(self.config.seed)
        ccc = ConceptCellCluster(self.config.ccc, self.config.margin_gate)
        with torch.no_grad():
            f1_output = ccc.f1_encode(prototype)
            ccc.learn_fast(prototype, f1_output)
        return ccc

    def _run_quantized_ccc(self, ccc: ConceptCellCluster, test_input: torch.Tensor) -> dict[str, torch.Tensor | bool]:
        x_batch, _ = self._as_batch(test_input.to(torch.float32))
        w1 = self._quantize_tensor(ccc.f1_layer.weight.detach(), bits=8)
        b1 = (
            self._quantize_tensor(ccc.f1_layer.bias.detach(), bits=8)
            if ccc.f1_layer.bias is not None
            else None
        )
        f1 = F.relu(F.linear(x_batch, w1, b1))
        f1 = self._quantize_activation(sparse_top_k(f1, ccc.config.f1_top_k))

        f2_weights = self._quantize_tensor(ccc.f2_weights.detach(), bits=8)
        f2_activation = self._quantize_activation(F.linear(f1, f2_weights))
        concept_direction = normalize(
            self._quantize_activation(ccc.concept_direction.detach()).unsqueeze(0)
        ).squeeze(0)
        confidence = cosine_similarity(f2_activation, concept_direction.unsqueeze(0))
        fired_mask = confidence > float(ccc.margin_gate.theta_margin.item())
        gated = torch.where(fired_mask.unsqueeze(-1), f2_activation, torch.zeros_like(f2_activation))

        feedback = self._quantize_tensor(ccc.feedback_weights.detach(), bits=8)
        prediction = self._quantize_activation(F.linear(gated, feedback))
        return {
            "fired": bool(fired_mask.any().item()),
            "abstained": bool((~fired_mask).all().item()),
            "confidence": confidence,
            "f1_output": f1,
            "f2_activation": f2_activation,
            "prediction": prediction,
        }

    def _simulate_fixed_lif(self, currents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_q = int(round(self.config.spiking.beta * 4096))
        threshold_q = int(round(self.config.spiking.threshold * self.VOLTAGE_SCALE))
        reset_q = int(round(self.config.spiking.reset * self.VOLTAGE_SCALE))
        refractory_steps = int(self.config.spiking.refractory_steps)

        if currents.dim() != 3:
            raise ValueError("currents must have shape (time, batch, neurons)")

        time_steps, batch_size, num_neurons = currents.shape
        v = torch.full((batch_size, num_neurons), reset_q, dtype=torch.int64)
        refractory = torch.zeros((batch_size, num_neurons), dtype=torch.int64)
        spikes = []
        voltages = []

        for step in range(time_steps):
            current_q = torch.round(currents[step].to(torch.float32) * self.VOLTAGE_SCALE).to(torch.int64)
            active = refractory <= 0
            integrated = ((beta_q * v) // 4096) + current_q
            integrated = torch.where(active, integrated, torch.full_like(integrated, reset_q))
            spike = (integrated >= threshold_q) & active
            v = torch.where(spike, torch.full_like(integrated, reset_q), integrated)
            refractory = torch.clamp(refractory - 1, min=0)
            refractory = torch.where(
                spike,
                torch.full_like(refractory, refractory_steps),
                refractory,
            )
            spikes.append(spike.to(torch.float32))
            voltages.append(v.to(torch.float32) / self.VOLTAGE_SCALE)

        return torch.stack(spikes, dim=0), torch.stack(voltages, dim=0)

    def _spike_timing_delta(self, reference: torch.Tensor, candidate: torch.Tensor) -> float:
        reference_times = torch.nonzero(reference > 0, as_tuple=False)
        candidate_times = torch.nonzero(candidate > 0, as_tuple=False)
        if reference_times.numel() == 0 and candidate_times.numel() == 0:
            return 0.0
        if reference_times.numel() == 0 or candidate_times.numel() == 0:
            return float(reference.shape[0])
        paired = min(reference_times.shape[0], candidate_times.shape[0])
        delta = (
            reference_times[:paired, 0].to(torch.float32)
            - candidate_times[:paired, 0].to(torch.float32)
        ).abs()
        tail_penalty = abs(reference_times.shape[0] - candidate_times.shape[0])
        return float(delta.max().item() if delta.numel() else 0.0) + float(tail_penalty)

    def validate_lif_equivalence(self, num_steps: int = 100) -> ValidationResult:
        """Compare Bio-ARN float LIF dynamics to a Loihi-style fixed-point proxy."""

        torch.manual_seed(self.config.seed)
        currents = torch.full((num_steps, 1, 4), 0.28, dtype=torch.float32)
        currents[:, 0, 0] += torch.linspace(0.0, 0.9, steps=num_steps)
        currents[::5, 0, 1] += 0.85
        currents[2::7, 0, 2] += 0.55
        currents[1::9, 0, 3] += 1.10

        lif = LIFNeuron(num_neurons=4, config=self.config.spiking)
        lif.reset_state(batch_size=1, dtype=torch.float32)
        float_spikes, _ = lif(currents)
        fixed_spikes, _ = self._simulate_fixed_lif(currents)

        max_diff, mean_diff = self._deviation_stats(float_spikes, fixed_spikes)
        rate_diff = abs(float(float_spikes.mean().item()) - float(fixed_spikes.mean().item()))
        timing_delta = self._spike_timing_delta(float_spikes, fixed_spikes)
        equivalent = bool(max_diff <= 1.0 and mean_diff <= 0.05 and rate_diff <= 0.05 and timing_delta <= 1.0)
        return ValidationResult(
            component="lif",
            equivalent=equivalent,
            max_deviation=max(max_diff, timing_delta),
            mean_deviation=mean_diff,
            notes=(
                f"Spike-rate delta={rate_diff:.4f}; worst spike-timing delta={timing_delta:.2f} steps. "
                "The proxy uses Q8.8 membrane state and integer leak integration."
            ),
        )

    def validate_ccc_equivalence(self, test_input: torch.Tensor) -> ValidationResult:
        """Compare float CCC firing decisions to 8-bit/16-bit Loihi-style quantization."""

        ccc = self._make_committed_ccc(test_input)
        float_output = ccc(test_input)
        quantized_output = self._run_quantized_ccc(ccc, test_input)

        float_conf = float_output.confidence.reshape(-1).to(torch.float32)
        quant_conf = quantized_output["confidence"].reshape(-1).to(torch.float32)  # type: ignore[union-attr]
        max_diff, mean_diff = self._deviation_stats(float_conf, quant_conf)
        decision_match = (
            float_output.fired == bool(quantized_output["fired"])
            and float_output.abstained == bool(quantized_output["abstained"])
        )
        equivalent = bool(decision_match and max_diff <= 0.15 and mean_diff <= 0.10)
        return ValidationResult(
            component="ccc",
            equivalent=equivalent,
            max_deviation=max_diff,
            mean_deviation=mean_diff,
            notes=(
                f"Decision match={decision_match}; float fired={float_output.fired}; "
                f"quantized fired={bool(quantized_output['fired'])}. Weights were quantized to 8-bit "
                "and activations to 16-bit after F1, F2, and feedback stages."
            ),
        )

    def _split_pattern(self, pattern: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vector = pattern.reshape(-1).to(torch.float32)
        data = torch.zeros(self.config.sdm.data_dim, dtype=torch.float32)
        length = min(vector.numel(), self.config.sdm.data_dim)
        data[:length] = vector[:length]
        return vector, data

    def validate_sdm_equivalence(self, test_patterns: list[torch.Tensor]) -> ValidationResult:
        """Compare float SDM reads to a quantized integer-storage SDM proxy."""

        torch.manual_seed(self.config.seed)
        float_sdm = SparseDistributedMemory(self.config.sdm)
        quant_hard_locations = float_sdm.hard_locations.detach().clone()
        quant_projection: torch.Tensor | None = None
        quant_matrix = torch.zeros(
            (self.config.sdm.num_hard_locations, self.config.sdm.data_dim),
            dtype=torch.int64,
        )
        quant_counts = torch.zeros(self.config.sdm.num_hard_locations, dtype=torch.int64)
        similarities = []

        def quant_compute_address(address: torch.Tensor) -> torch.Tensor:
            nonlocal quant_projection
            address_batch, squeeze = self._as_batch(address.to(torch.float32))
            if address_batch.shape[-1] != self.config.sdm.address_dim:
                if quant_projection is None:
                    projection = float_sdm._get_or_create_projection(  # type: ignore[attr-defined]
                        address_batch.shape[-1],
                        device=address_batch.device,
                        dtype=address_batch.dtype,
                    )
                    quant_projection = projection.detach().clone()
                address_batch = address_batch @ quant_projection.to(address_batch.device)
            binary = (address_batch > 0).to(torch.float32)
            return binary.squeeze(0) if squeeze else binary

        def quant_activated(address: torch.Tensor) -> torch.Tensor:
            address_batch, squeeze = self._as_batch(quant_compute_address(address))
            address_ones = address_batch.sum(dim=-1, keepdim=True)
            location_ones = quant_hard_locations.sum(dim=-1).unsqueeze(0)
            overlaps = address_batch @ quant_hard_locations.T
            distances = address_ones + location_ones - (2.0 * overlaps)
            activated = distances <= float(self.config.sdm.hamming_radius)
            return activated.squeeze(0) if squeeze else activated

        for pattern in test_patterns:
            address, data = self._split_pattern(pattern)
            float_sdm.write(address, data)
            activated = quant_activated(address)
            activated_batch, _ = self._as_batch(activated.to(torch.int64))
            data_q = torch.round(
                self._quantize_tensor(data.unsqueeze(0), bits=8) * self.WEIGHT_INT_MAX
            ).to(torch.int64)
            quant_matrix.add_(activated_batch.T @ data_q)
            quant_counts.add_(activated_batch.sum(dim=0))
            decay_num = int(round(self.config.sdm.decay_rate * 256))
            quant_matrix = (quant_matrix * decay_num) // 256

        for pattern in test_patterns:
            address, _ = self._split_pattern(pattern)
            float_read = float_sdm.read(address).reshape(1, -1)
            activated = quant_activated(address)
            activated_batch, _ = self._as_batch(activated.to(torch.int64))
            quant_read = activated_batch @ quant_matrix
            counts = (activated_batch @ quant_counts).clamp_min(1).unsqueeze(-1)
            quant_read = (quant_read.to(torch.float32) / counts.to(torch.float32)) / self.WEIGHT_INT_MAX
            similarity = cosine_similarity(float_read, quant_read).mean()
            similarities.append(float(similarity.item()))

        similarity_tensor = torch.tensor(similarities, dtype=torch.float32)
        deviation = 1.0 - similarity_tensor
        return ValidationResult(
            component="sdm",
            equivalent=bool(similarity_tensor.mean().item() >= 0.90),
            max_deviation=float(deviation.max().item()),
            mean_deviation=float(deviation.mean().item()),
            notes=(
                f"Mean retrieval cosine similarity={similarity_tensor.mean().item():.4f}. "
                "The proxy uses binary addresses, integer accumulation, and quantized data writes."
            ),
        )

    def _run_quantized_pc_layer(self, layer: PCLayer, actual_input: torch.Tensor) -> dict[str, torch.Tensor]:
        actual_batch, _ = self._as_batch(actual_input.to(torch.float32))
        state = self._quantize_activation(layer.state.detach().unsqueeze(0))
        weights = self._quantize_tensor(layer.W.detach(), bits=8)
        prediction = self._quantize_activation(torch.relu(state @ weights))
        precision = self._quantize_activation(layer.precision.detach()).unsqueeze(0).clamp_min(0.1)
        error = self._quantize_activation((actual_batch - prediction) * precision)
        error = torch.where(
            error.abs() < float(layer.config.error_threshold),
            torch.zeros_like(error),
            error,
        )
        next_state = self._quantize_activation(
            torch.relu(state + float(layer.config.gamma) * (error @ weights.t()))
        )
        return {"prediction": prediction, "error": error, "state": next_state}

    def validate_full_system(self, test_inputs: list[torch.Tensor]) -> SystemValidationResult:
        """Run a small float vs fixed-point proxy pipeline across key Bio-ARN blocks."""

        if not test_inputs:
            raise ValueError("test_inputs must contain at least one tensor.")

        ccc = self._make_committed_ccc(test_inputs[0])
        float_sdm = SparseDistributedMemory(self.config.sdm)
        quant_sdm_result = self.validate_sdm_equivalence(test_inputs)
        pe = PCLayer(self.config.sdm.data_dim, max(self.config.sdm.data_dim // 2, 8), self.config.predictive)
        gnw_float = GlobalNeuronalWorkspace(self.config.gnw)
        gnw_quant = GlobalNeuronalWorkspace(self.config.gnw)

        sensitivity = {"ccc": 0.0, "sdm": 0.0, "pe": 0.0, "gnw": 0.0}

        for timestep, current_input in enumerate(test_inputs):
            float_ccc = ccc(current_input, timestep=timestep)
            quant_ccc = self._run_quantized_ccc(ccc, current_input)

            ccc_delta = abs(
                float(float_ccc.confidence.reshape(-1).mean().item())
                - float(quant_ccc["confidence"].reshape(-1).mean().item())  # type: ignore[union-attr]
            )
            if float_ccc.fired != bool(quant_ccc["fired"]):
                ccc_delta += 1.0
            sensitivity["ccc"] += ccc_delta

            address, data = self._split_pattern(current_input)
            if float_ccc.fired:
                float_sdm.write(address, data)
            float_read = float_sdm.read(address)
            quant_read = self._quantize_activation(float_read.unsqueeze(0)).squeeze(0)
            sensitivity["sdm"] += float((float_read - quant_read).abs().mean().item())

            float_pe = pe(float_read, learn=False)
            quant_pe = self._run_quantized_pc_layer(pe, quant_read)
            sensitivity["pe"] += float(
                (float_pe.prediction.reshape(-1) - quant_pe["prediction"].reshape(-1)).abs().mean().item()
            )

            float_candidates = []
            quant_candidates = []
            if float_ccc.fired:
                confidence = float(float_ccc.confidence.reshape(-1).mean().item())
                float_candidates.append((0, float_ccc.f2_activation.reshape(-1), confidence))
            if bool(quant_ccc["fired"]):
                q_conf = float(quant_ccc["confidence"].reshape(-1).mean().item())  # type: ignore[union-attr]
                quant_candidates.append((0, quant_ccc["f2_activation"].reshape(-1), q_conf))  # type: ignore[index]

            gnw_float.update(float_candidates, timestep=timestep)
            gnw_quant.update(quant_candidates, timestep=timestep)
            sensitivity["gnw"] += abs(len(gnw_float.slots) - len(gnw_quant.slots)) / max(
                self.config.gnw.capacity,
                1,
            )

        num_inputs = float(len(test_inputs))
        sensitivity = {key: value / num_inputs for key, value in sensitivity.items()}
        sensitivity["sdm"] = max(sensitivity["sdm"], quant_sdm_result.mean_deviation)
        worst_component = max(sensitivity, key=sensitivity.get)
        overall_accuracy_delta = float(sum(sensitivity.values()) / len(sensitivity))

        return SystemValidationResult(
            overall_equivalent=bool(overall_accuracy_delta <= 0.25),
            overall_accuracy_delta=overall_accuracy_delta,
            component_sensitivity=sensitivity,
            worst_component=worst_component,
            notes=(
                "The proxy pipeline reuses current PyTorch Bio-ARN modules and inserts Loihi-style "
                "8-bit weight / 16-bit activation quantization between stages."
            ),
        )


__all__ = [
    "FunctionalEquivalenceValidator",
    "LoihiCCCMapping",
    "LoihiGNWMapping",
    "LoihiMapping",
    "LoihiNeuronSpec",
    "LoihiPEMapping",
    "LoihiSDMMapping",
    "LoihiSystemMapping",
    "SystemValidationResult",
    "ValidationResult",
]
