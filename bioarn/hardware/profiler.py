"""Hardware-aware profiling and backend comparison utilities."""

from __future__ import annotations

from bioarn.config import BioARNConfig
from bioarn.hardware.backend import (
    ComparisonReport,
    ComponentMapper,
    HardwareInfo,
    LatencyEstimate,
    NeuromorphicBackend,
    PowerEstimate,
    SystemMapping,
)


class HardwareProfiler:
    """Estimate power, latency, and platform fit for Bio-ARN deployments."""

    def __init__(self, backend: NeuromorphicBackend):
        self.backend = backend

    @staticmethod
    def _baseline_power_comparison(mapping: SystemMapping) -> dict[str, float]:
        inference = max(mapping.estimated_power_watts, 1e-9)
        return {
            "PyTorch CPU": inference * 1.35,
            "PyTorch GPU": inference * 0.65,
            "Loihi 2": inference * 0.18,
            "Ideal ASIC": inference * 0.08,
        }

    def _power_from_info(self, mapping: SystemMapping, info: HardwareInfo) -> float:
        active_neurons = mapping.total_neurons * 0.1
        active_synapses = mapping.total_synapses * 0.01
        synapse_event_energy = max(info.estimated_power_per_spike / 4.0, 1e-13)
        energy_per_inference = (
            active_neurons * info.estimated_power_per_spike
            + active_synapses * synapse_event_energy
        )
        if not info.native_spike:
            energy_per_inference *= 1.4
        return float(energy_per_inference * 1000.0)

    def estimate_power(self, system_mapping: SystemMapping) -> PowerEstimate:
        """Estimate inference, training, and idle power for the current backend."""

        info = self.backend.get_hardware_info()
        inference_watts = self._power_from_info(system_mapping, info)
        training_multiplier = 1.8 if info.supports_learning else 1.1
        idle_watts = max(0.01, inference_watts * (0.08 if info.native_spike else 0.18))
        comparison = self._baseline_power_comparison(system_mapping)
        comparison[info.backend_name] = inference_watts
        return PowerEstimate(
            inference_watts=float(inference_watts),
            training_watts=float(inference_watts * training_multiplier),
            idle_watts=float(idle_watts),
            comparison=comparison,
        )

    def estimate_latency(self, system_mapping: SystemMapping) -> LatencyEstimate:
        """Estimate inference latency from system size and backend parallelism."""

        info = self.backend.get_hardware_info()
        pipeline_stages = max(len(system_mapping.components), 1)
        bottleneck_component = max(
            system_mapping.components,
            key=lambda component: component.synapse_count,
        )

        if info.native_spike:
            per_synapse = 2.5e-8
            stage_overhead = 0.08
        elif "gpu" in info.backend_name.lower():
            per_synapse = 8e-9
            stage_overhead = 0.05
        else:
            per_synapse = 4e-8
            stage_overhead = 0.12

        inference_ms = (system_mapping.total_synapses * per_synapse) + (pipeline_stages * stage_overhead)
        return LatencyEstimate(
            inference_ms=float(inference_ms),
            pipeline_stages=pipeline_stages,
            bottleneck=bottleneck_component.name,
        )

    def compare_backends(self, config: BioARNConfig) -> ComparisonReport:
        """Compare PyTorch CPU/GPU, Loihi 2, and an ideal Bio-ARN ASIC."""

        mapper = ComponentMapper(self.backend)
        system_mapping = mapper.map_full_system(config)
        current_info = self.backend.get_hardware_info()

        power = self._baseline_power_comparison(system_mapping)
        latency = {
            "PyTorch CPU": system_mapping.total_synapses * 4.0e-8 + len(system_mapping.components) * 0.12,
            "PyTorch GPU": system_mapping.total_synapses * 1.2e-8 + len(system_mapping.components) * 0.06,
            "Loihi 2": system_mapping.total_synapses * 2.5e-8 + len(system_mapping.components) * 0.08,
            "Ideal ASIC": system_mapping.total_synapses * 1.0e-8 + len(system_mapping.components) * 0.04,
        }
        area = {
            "PyTorch CPU": 600.0,
            "PyTorch GPU": 826.0,
            "Loihi 2": 31.0,
            "Ideal ASIC": max(system_mapping.estimated_die_area_mm2, 1.0),
        }

        current_power = self._power_from_info(system_mapping, current_info)
        power[current_info.backend_name] = current_power
        latency[current_info.backend_name] = self.estimate_latency(system_mapping).inference_ms
        area[current_info.backend_name] = area.get(current_info.backend_name, system_mapping.estimated_die_area_mm2)

        neuron_utilization = {
            "PyTorch CPU": min(system_mapping.total_neurons / 100_000_000, 1.0),
            "PyTorch GPU": min(system_mapping.total_neurons / 200_000_000, 1.0),
            "Loihi 2": min(system_mapping.total_neurons / 1_000_000, 1.0),
            "Ideal ASIC": min(system_mapping.total_neurons / max(system_mapping.total_neurons * 1.2, 1), 1.0),
        }
        if current_info.backend_name not in neuron_utilization:
            neuron_utilization[current_info.backend_name] = min(
                system_mapping.total_neurons / max(current_info.max_neurons, 1),
                1.0,
            )

        best_power_backend = min(power, key=power.get)
        best_latency_backend = min(latency, key=latency.get)
        summary = (
            f"{best_power_backend} minimizes power, while {best_latency_backend} offers the lowest "
            f"estimated latency for the mapped Bio-ARN system."
        )
        return ComparisonReport(
            system_mapping=system_mapping,
            power_watts={name: float(value) for name, value in power.items()},
            latency_ms={name: float(value) for name, value in latency.items()},
            area_mm2={name: float(value) for name, value in area.items()},
            neuron_utilization={name: float(value) for name, value in neuron_utilization.items()},
            summary=summary,
        )


__all__ = ["HardwareProfiler"]
