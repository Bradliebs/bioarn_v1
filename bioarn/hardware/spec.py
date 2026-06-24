"""ASIC design-spec generator for a custom Bio-ARN neuromorphic accelerator."""

from __future__ import annotations

from pathlib import Path

from bioarn.config import BioARNConfig
from bioarn.hardware.backend import ComponentMapper
from bioarn.hardware.loihi_backend import LoihiBackend
from bioarn.hardware.profiler import HardwareProfiler


class ASICSpec:
    """Generate a practical research-spec document for a Bio-ARN ASIC."""

    def __init__(self, config: BioARNConfig):
        self.config = config
        self.mapper = ComponentMapper(LoihiBackend())
        self.profiler = HardwareProfiler(LoihiBackend())

    def generate_spec(self) -> str:
        """Produce a human-readable ASIC specification document."""

        mapping = self.mapper.map_full_system(self.config)
        power = self.profiler.estimate_power(mapping)
        latency = self.profiler.estimate_latency(mapping)

        memory_bits = mapping.total_memory_bytes * 8
        transistor_count = int(
            (memory_bits * 6)
            + (mapping.total_neurons * 220)
            + (mapping.total_synapses * 6)
            + (self.config.gnw.capacity * 2_000)
        )
        clock_mhz = 100

        component_lines = "\n".join(
            f"- {component.name}: neurons={component.neuron_count:,}, synapses={component.synapse_count:,}, "
            f"memory={component.memory_bytes:,} B, learning={', '.join(component.learning_rules)}"
            for component in mapping.components
        )

        return (
            "Bio-ARN 2.0 Neuromorphic ASIC Research Specification\n"
            "====================================================\n\n"
            "Target workload\n"
            "---------------\n"
            "A research-oriented accelerator for sparse, event-driven Bio-ARN inference and local learning.\n\n"
            "Functional blocks\n"
            "-----------------\n"
            "1. Native SDM address lookup circuits using Hamming-distance comparators and address/data separation.\n"
            "2. Margin gate comparators implementing cosine similarity plus programmable abstention thresholds.\n"
            "3. STDP and Hebbian learning circuits with spike-timing traces and bounded weight updates.\n"
            "4. Predictive-coding error circuits supporting subtraction, precision scaling, and local feedback.\n"
            "5. GNW broadcast bus with winner-take-all arbitration, recurrent inhibition, and fatigue timers.\n"
            "6. Sparse routing fabric optimized for CCC, SDM, and predictive-coding fan-out.\n\n"
            "Mapped Bio-ARN resources\n"
            "------------------------\n"
            f"{component_lines}\n\n"
            "Top-level estimates\n"
            "-------------------\n"
            f"- Total neurons: {mapping.total_neurons:,}\n"
            f"- Total synapses: {mapping.total_synapses:,}\n"
            f"- Total on-chip memory: {mapping.total_memory_bytes:,} bytes\n"
            f"- Estimated die area: {mapping.estimated_die_area_mm2:.2f} mm^2\n"
            f"- Estimated transistor count: {transistor_count:,}\n"
            f"- Inference power budget: {power.inference_watts:.3f} W\n"
            f"- Training power budget: {power.training_watts:.3f} W\n"
            f"- Idle power budget: {power.idle_watts:.3f} W\n"
            f"- Inference latency target: {latency.inference_ms:.3f} ms\n"
            f"- Recommended clock frequency: {clock_mhz} MHz\n\n"
            "Circuit notes\n"
            "-------------\n"
            "- CCC tiles should co-locate F1/F2 populations and feedback SRAM to minimize routing energy.\n"
            "- SDM address comparators should use binary popcount trees and radius-threshold comparators.\n"
            "- Predictive-coding layers benefit from mixed-signal error accumulation and digital precision registers.\n"
            "- GNW arbitration should expose programmable competition temperature and fatigue decay constants.\n"
            "- Learning datapaths should support fixed-point bounded weights with optional stochastic rounding.\n"
        )

    def save_spec(self, path: str) -> None:
        """Save the generated ASIC specification document to disk."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_spec(), encoding="utf-8")


__all__ = ["ASICSpec"]
