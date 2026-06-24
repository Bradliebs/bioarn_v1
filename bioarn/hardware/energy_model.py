"""Cross-platform energy model for Bio-ARN hardware comparisons.

Energy constants are based on published or commonly cited order-of-magnitude
figures: Loihi 2 synaptic event energy from Intel/Nature reports, plus coarse
CPU/GPU efficiency estimates derived from public FLOP-per-watt and TDP data.
The goal is comparative system modeling rather than cycle-accurate power
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioarn.config import BioARNConfig


@dataclass
class EnergyBreakdown:
    """Energy estimate for one Bio-ARN inference or learning step."""

    total_joules: float
    component_breakdown: dict[str, float]
    watts_at_1khz: float
    spikes_per_inference: int
    operations_per_inference: int


@dataclass
class EnergyComparisonReport:
    """Side-by-side energy comparison across supported backends."""

    backends: dict[str, EnergyBreakdown]
    best_backend: str
    efficiency_ratios: dict[str, float]
    notes: str


@dataclass
class BrainComparisonReport:
    """Comparison of Bio-ARN energy to a neuron-count-scaled brain baseline."""

    bioarn_power_watts: float
    scaled_brain_power_watts: float
    total_neurons: int
    power_ratio_vs_brain: float
    energy_per_neuron_ratio: float
    approaching_biology: bool
    notes: str


@dataclass
class _ComponentWork:
    spikes: int
    neuron_updates: int
    learning_updates: int
    flops: int
    memory_accesses: int


class EnergyModel:
    """Estimate Bio-ARN energy on Loihi 2, GPU, CPU, and idealized ASICs."""

    ENERGY_CONSTANTS = {
        "loihi2": {
            "spike_event": 23e-12,
            "neuron_update": 12e-12,
            "learning_update": 120e-12,
            "idle_power": 0.001,
            "chip_static": 0.050,
        },
        "gpu_a100": {
            "flop": 1e-12,
            "memory_access": 10e-12,
            "idle_power": 50.0,
            "peak_power": 300.0,
        },
        "cpu_laptop": {
            "flop": 10e-12,
            "memory_access": 50e-12,
            "idle_power": 5.0,
            "peak_power": 45.0,
        },
        "ideal_asic": {
            "spike_event": 5e-12,
            "neuron_update": 3e-12,
            "learning_update": 30e-12,
            "idle_power": 0.0005,
            "chip_static": 0.010,
        },
    }

    INFERENCE_WINDOW_SECONDS = 1e-3
    ELECTRICITY_USD_PER_KWH = 0.12

    @staticmethod
    def _predictive_dims(config: BioARNConfig) -> list[int]:
        dims = [max(int(config.ccc.concept_dim), 8)]
        for _ in range(max(int(config.predictive.num_levels), 1)):
            dims.append(max(8, dims[-1] // 2))
        return dims

    @staticmethod
    def _total_neurons(config: BioARNConfig) -> int:
        dims = EnergyModel._predictive_dims(config)
        pe_neurons = sum(input_dim + (2 * input_dim) + output_dim for input_dim, output_dim in zip(dims[:-1], dims[1:]))
        gnw_neurons = (3 * config.gnw.capacity) + max(8, config.gnw.capacity * 2)
        sdm_neurons = config.sdm.address_dim + config.sdm.num_hard_locations + config.sdm.data_dim
        ccc_neurons = config.ccc.num_f1_features + (config.ccc.max_pool_size * config.ccc.concept_dim)
        return int(pe_neurons + gnw_neurons + sdm_neurons + ccc_neurons)

    def _component_workload(
        self,
        config: BioARNConfig,
        num_cccs_active: int,
        *,
        learning: bool,
    ) -> dict[str, _ComponentWork]:
        active_cccs = max(0, min(int(num_cccs_active), int(config.ccc.max_pool_size)))
        if active_cccs == 0:
            active_cccs = 1

        f1_winners = min(int(config.ccc.f1_top_k), int(config.ccc.num_f1_features))
        active_f2 = max(1, int(config.ccc.concept_dim) // 8)
        ccc_learning = active_cccs * (
            config.ccc.num_f1_features * config.ccc.concept_dim + config.ccc.concept_dim
        )
        ccc_work = _ComponentWork(
            spikes=int(
                active_cccs
                * (
                    config.ccc.input_dim * f1_winners
                    + f1_winners * active_f2
                    + active_f2 * f1_winners
                )
            ),
            neuron_updates=int(
                config.ccc.num_f1_features + active_cccs * (config.ccc.concept_dim + 2)
            ),
            learning_updates=int(ccc_learning if learning else 0),
            flops=int(
                config.ccc.input_dim * config.ccc.num_f1_features
                + active_cccs * (2 * config.ccc.num_f1_features * config.ccc.concept_dim)
            ),
            memory_accesses=int(
                config.ccc.input_dim * config.ccc.num_f1_features
                + active_cccs * (2 * config.ccc.num_f1_features * config.ccc.concept_dim)
            ),
        )

        activation_fraction = min(
            0.25,
            max(
                1.0 / max(config.sdm.num_hard_locations, 1),
                (config.sdm.hamming_radius / max(config.sdm.address_dim, 1)) * 1.5,
            ),
        )
        active_locations = max(1, int(round(config.sdm.num_hard_locations * activation_fraction)))
        sdm_learning = active_locations * config.sdm.data_dim
        sdm_work = _ComponentWork(
            spikes=int(config.sdm.address_dim + active_locations * config.sdm.data_dim),
            neuron_updates=int(
                config.sdm.address_dim
                + config.sdm.num_hard_locations
                + active_locations
                + config.sdm.data_dim
            ),
            learning_updates=int(sdm_learning if learning else 0),
            flops=int(
                config.sdm.num_hard_locations * config.sdm.address_dim
                + active_locations * config.sdm.data_dim
            ),
            memory_accesses=int(
                config.sdm.num_hard_locations * (config.sdm.address_dim + config.sdm.data_dim)
            ),
        )

        dims = self._predictive_dims(config)
        pe_spikes = 0
        pe_updates = 0
        pe_learning = 0
        pe_flops = 0
        pe_memory = 0
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            pe_spikes += max(1, input_dim // 4) + max(1, output_dim // 4)
            pe_updates += (3 * input_dim) + output_dim
            pe_learning += input_dim * output_dim
            pe_flops += (2 * input_dim * output_dim) + (4 * input_dim)
            pe_memory += (2 * input_dim * output_dim) + input_dim
        pe_work = _ComponentWork(
            spikes=int(pe_spikes),
            neuron_updates=int(pe_updates),
            learning_updates=int(pe_learning if learning else 0),
            flops=int(pe_flops),
            memory_accesses=int(pe_memory),
        )

        active_slots = min(int(config.gnw.capacity), active_cccs)
        broadcast_targets = active_cccs + int(config.predictive.num_levels) + 2
        gnw_learning = max(0, config.gnw.capacity * (config.gnw.capacity - 1) // 2)
        gnw_work = _ComponentWork(
            spikes=int(active_slots * (config.gnw.capacity + broadcast_targets)),
            neuron_updates=int((3 * config.gnw.capacity) + active_slots),
            learning_updates=int(gnw_learning if learning else 0),
            flops=int((config.gnw.capacity * config.gnw.capacity) + (active_slots * broadcast_targets * 4)),
            memory_accesses=int((config.gnw.capacity * config.gnw.capacity) + (active_slots * broadcast_targets)),
        )

        return {
            "ccc": ccc_work,
            "sdm": sdm_work,
            "pe": pe_work,
            "gnw": gnw_work,
        }

    def _static_energy(self, backend: str) -> float:
        constants = self.ENERGY_CONSTANTS[backend]
        if backend in {"loihi2", "ideal_asic"}:
            return (
                (constants["chip_static"] + (128 * constants["idle_power"]))
                * self.INFERENCE_WINDOW_SECONDS
            )
        return constants["idle_power"] * self.INFERENCE_WINDOW_SECONDS

    def _peak_budget(self, backend: str) -> float:
        constants = self.ENERGY_CONSTANTS[backend]
        if backend in {"gpu_a100", "cpu_laptop"}:
            return float(constants["peak_power"])
        if backend == "loihi2":
            return float(constants["chip_static"] + (128 * constants["idle_power"]))
        return float(constants["chip_static"] + (128 * constants["idle_power"]))

    def estimate_inference_energy(
        self,
        config: BioARNConfig,
        backend: str,
        num_cccs_active: int,
    ) -> EnergyBreakdown:
        """Estimate one sparse Bio-ARN inference pass on the requested backend."""

        if backend not in self.ENERGY_CONSTANTS:
            raise ValueError(f"Unsupported backend: {backend}")

        work = self._component_workload(config, num_cccs_active, learning=False)
        constants = self.ENERGY_CONSTANTS[backend]
        component_breakdown: dict[str, float] = {}

        for component, stats in work.items():
            if backend in {"loihi2", "ideal_asic"}:
                energy = (
                    stats.spikes * constants["spike_event"]
                    + stats.neuron_updates * constants["neuron_update"]
                )
            else:
                energy = (
                    stats.flops * constants["flop"]
                    + stats.memory_accesses * constants["memory_access"]
                )
            component_breakdown[component] = float(energy)

        component_breakdown["static"] = float(self._static_energy(backend))
        total = float(sum(component_breakdown.values()))
        spikes = sum(stats.spikes for stats in work.values())
        operations = sum(
            (
                stats.spikes + stats.neuron_updates
                if backend in {"loihi2", "ideal_asic"}
                else stats.flops + stats.memory_accesses
            )
            for stats in work.values()
        )
        return EnergyBreakdown(
            total_joules=total,
            component_breakdown=component_breakdown,
            watts_at_1khz=float(total * 1000.0),
            spikes_per_inference=int(spikes),
            operations_per_inference=int(operations),
        )

    def estimate_learning_energy(self, config: BioARNConfig, backend: str) -> EnergyBreakdown:
        """Estimate one local-learning step for Bio-ARN.

        On spike-native targets, this models STDP/x-trace style updates. On CPU/GPU
        targets it models dense arithmetic plus memory traffic for the same update.
        """

        if backend not in self.ENERGY_CONSTANTS:
            raise ValueError(f"Unsupported backend: {backend}")

        active_cccs = min(int(config.gnw.capacity), int(config.ccc.max_pool_size))
        work = self._component_workload(config, active_cccs, learning=True)
        constants = self.ENERGY_CONSTANTS[backend]
        component_breakdown: dict[str, float] = {}

        for component, stats in work.items():
            if backend in {"loihi2", "ideal_asic"}:
                energy = (
                    stats.learning_updates * constants["learning_update"]
                    + stats.spikes * constants["spike_event"] * 0.25
                )
            else:
                energy = (
                    (stats.learning_updates * 2) * constants["flop"]
                    + stats.learning_updates * constants["memory_access"]
                )
            component_breakdown[component] = float(energy)

        component_breakdown["static"] = float(self._static_energy(backend))
        total = float(sum(component_breakdown.values()))
        spikes = sum(stats.spikes for stats in work.values())
        operations = sum(stats.learning_updates for stats in work.values())
        return EnergyBreakdown(
            total_joules=total,
            component_breakdown=component_breakdown,
            watts_at_1khz=float(total * 1000.0),
            spikes_per_inference=int(spikes),
            operations_per_inference=int(operations),
        )

    def compare_all_backends(self, config: BioARNConfig) -> EnergyComparisonReport:
        """Compare inference energy, throughput, and operating cost across backends."""

        active_cccs = min(int(config.gnw.capacity), int(config.ccc.max_pool_size))
        backends = {
            backend: self.estimate_inference_energy(config, backend, active_cccs)
            for backend in self.ENERGY_CONSTANTS
        }
        best_backend = min(backends, key=lambda name: backends[name].total_joules)
        worst_energy = max(report.total_joules for report in backends.values())
        efficiency_ratios = {
            backend: float(worst_energy / max(report.total_joules, 1e-18))
            for backend, report in backends.items()
        }

        throughput = {
            backend: float(self._peak_budget(backend) / max(report.total_joules, 1e-18))
            for backend, report in backends.items()
        }
        cost_per_million = {
            backend: float(
                report.total_joules
                * 1_000_000
                / 3_600_000
                * self.ELECTRICITY_USD_PER_KWH
            )
            for backend, report in backends.items()
        }
        notes = (
            "Throughput at native power budget (inf/s): "
            + ", ".join(f"{name}={throughput[name]:.1f}" for name in backends)
            + ". Cost per million inferences (USD): "
            + ", ".join(f"{name}={cost_per_million[name]:.6f}" for name in backends)
            + "."
        )
        return EnergyComparisonReport(
            backends=backends,
            best_backend=best_backend,
            efficiency_ratios=efficiency_ratios,
            notes=notes,
        )

    def brain_comparison(self, config: BioARNConfig) -> BrainComparisonReport:
        """Compare Loihi-mapped Bio-ARN power to a neuron-scaled human-brain baseline."""

        total_neurons = self._total_neurons(config)
        active_cccs = min(int(config.gnw.capacity), int(config.ccc.max_pool_size))
        loihi_energy = self.estimate_inference_energy(config, "loihi2", active_cccs)
        bioarn_power_watts = float(loihi_energy.watts_at_1khz)

        brain_power_watts = 20.0
        brain_neurons = 86_000_000_000
        scaled_brain_power = brain_power_watts * (total_neurons / brain_neurons)
        power_ratio = bioarn_power_watts / max(scaled_brain_power, 1e-18)
        bioarn_power_per_neuron = bioarn_power_watts / max(total_neurons, 1)
        brain_power_per_neuron = brain_power_watts / brain_neurons
        energy_per_neuron_ratio = bioarn_power_per_neuron / brain_power_per_neuron

        return BrainComparisonReport(
            bioarn_power_watts=bioarn_power_watts,
            scaled_brain_power_watts=float(scaled_brain_power),
            total_neurons=int(total_neurons),
            power_ratio_vs_brain=float(power_ratio),
            energy_per_neuron_ratio=float(energy_per_neuron_ratio),
            approaching_biology=bool(energy_per_neuron_ratio <= 10.0),
            notes=(
                "The brain baseline assumes ~20 W for ~86 billion neurons. Bio-ARN is compared "
                "against a brain region scaled to the same neuron count, not the whole brain."
            ),
        )


__all__ = [
    "BrainComparisonReport",
    "EnergyBreakdown",
    "EnergyComparisonReport",
    "EnergyModel",
]
