"""Deployment pipeline for moving Bio-ARN models onto Lava/Loihi."""

from __future__ import annotations

import copy
import warnings
from dataclasses import asdict
from typing import Any

import torch

from bioarn.hardware.energy_model import EnergyModel
from bioarn.hardware.lava_bridge import (
    DeploymentPackage,
    EquivalenceReport,
    HardwareRequirements,
    LavaBridge,
    LavaProcessGraph,
)
from bioarn.hardware.loihi_port import LoihiMapping
from bioarn.persistence.model_store import ModelStore
from bioarn.persistence.quantization import ModelQuantizer, QuantizedModel


class LoihiDeploymentPipeline:
    """End-to-end pipeline: train on PyTorch -> deploy on Loihi."""

    def __init__(self, model_store: ModelStore):
        self.model_store = model_store
        self.quantizer: ModelQuantizer = getattr(model_store, "quantizer", ModelQuantizer())
        self.bridge = LavaBridge()
        self.energy_model = EnergyModel()

    @staticmethod
    def _unwrap_system(system: Any) -> Any:
        return getattr(system, "core", system)

    @staticmethod
    def _restore_quantized_model(model: Any, quantized: QuantizedModel) -> Any:
        restored = copy.deepcopy(model)
        restored.load_state_dict(quantized.dequantized_state_dict(), strict=False)
        return restored

    @staticmethod
    def _sample_input_dim(model: Any) -> int:
        if hasattr(model, "config"):
            return int(model.config.ccc.input_dim)
        if hasattr(model, "input_dim"):
            return int(model.input_dim)
        raise ValueError("Unable to infer model input dimensionality for deployment validation.")

    def _build_validation_data(self, model: Any, num_samples: int = 8) -> list[tuple[torch.Tensor, None]]:
        input_dim = self._sample_input_dim(model)
        seed = int(getattr(getattr(model, "config", None), "seed", 7))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        samples: list[tuple[torch.Tensor, None]] = []
        for index in range(max(num_samples, 4)):
            sample = torch.rand(input_dim, generator=generator, dtype=torch.float32)
            if index % 2 == 0:
                sample = sample * 0.5
            else:
                sample = sample.roll(shifts=index % max(input_dim, 1), dims=0)
            samples.append((sample, None))
        return samples

    def prepare_for_deployment(self, model_name: str, version: str) -> DeploymentPackage:
        """Load, quantize, validate, convert, and package a model for Loihi."""

        system, info = self.model_store.load_model(model_name, version)
        model = self._unwrap_system(system)
        validation_data = self._build_validation_data(model)

        quantized = self.quantizer.quantize_weights(model, bits=8)
        quantization_report = self.quantizer.validate_quantization(model, quantized, validation_data)
        if quantization_report.accuracy_drop > 0.05:
            warnings.warn(
                (
                    f"Quantization accuracy drop for {info.name} v{info.version} was "
                    f"{quantization_report.accuracy_drop:.3f}, exceeding the 5% target."
                ),
                stacklevel=2,
            )

        quantized_model = self._restore_quantized_model(model, quantized)
        process_graph = self.bridge.convert_full_model(quantized_model)
        equivalence = self.bridge.validate_equivalence(
            model,
            process_graph,
            [sample for sample, _ in validation_data],
            tolerance=0.05,
        )

        provisional = DeploymentPackage(
            model_name=info.name,
            version=info.version,
            quantization_bits=8,
            process_graph=process_graph,
            equivalence=equivalence,
            hardware_reqs=HardwareRequirements(0, 0, 0, 0, 0.0, 0.0),
            config={},
            ready_for_hardware=False,
        )
        hardware_reqs = self.estimate_hardware_requirements(provisional)
        package = DeploymentPackage(
            model_name=info.name,
            version=info.version,
            quantization_bits=8,
            process_graph=process_graph,
            equivalence=equivalence,
            hardware_reqs=hardware_reqs,
            config={},
            ready_for_hardware=bool(
                equivalence.passed and quantization_report.accuracy_drop <= 0.05
            ),
        )
        package.config = self.generate_deployment_config(package)
        package.config["quantization_report"] = {
            "original_accuracy": float(quantization_report.original_accuracy),
            "quantized_accuracy": float(quantization_report.quantized_accuracy),
            "accuracy_drop": float(quantization_report.accuracy_drop),
            "max_weight_error": float(quantization_report.max_weight_error),
            "mean_weight_error": float(quantization_report.mean_weight_error),
            "compression_ratio": float(quantization_report.compression_ratio),
            "max_output_deviation": float(quantization_report.max_output_deviation),
        }
        return package

    def estimate_hardware_requirements(
        self,
        deployment_package: DeploymentPackage,
    ) -> HardwareRequirements:
        """Estimate Loihi 2 core, memory, power, and latency requirements."""

        graph = deployment_package.process_graph
        config = graph.metadata.get("config")
        num_cores = max(
            int(graph.num_cores_estimate),
            self.bridge._graph_cores(graph.num_neurons, graph.num_synapses),
        )
        total_memory = int((graph.num_synapses * 2) + (graph.num_neurons * 8))
        memory_per_core = int((total_memory + num_cores - 1) // max(num_cores, 1))

        if config is not None:
            active_cccs = min(int(config.gnw.capacity), int(config.ccc.max_pool_size))
            power = self.energy_model.estimate_inference_energy(
                config,
                backend="loihi2",
                num_cccs_active=active_cccs,
            )
            latency_ms = float(LoihiMapping(config).map_full_system().estimated_latency_ms)
            estimated_power_mw = float(power.watts_at_1khz * 1000.0)
        else:
            estimated_power_mw = float(graph.num_synapses * 23e-9)
            latency_ms = float(graph.metadata.get("estimated_latency_ms", 1.0))

        return HardwareRequirements(
            num_cores=int(num_cores),
            total_neurons=int(graph.num_neurons),
            total_synapses=int(graph.num_synapses),
            memory_per_core_bytes=int(memory_per_core),
            estimated_power_mw=estimated_power_mw,
            estimated_latency_ms=float(latency_ms),
        )

    def generate_deployment_config(self, deployment_package: DeploymentPackage) -> dict:
        """Generate a serializable Lava deployment configuration."""

        graph = deployment_package.process_graph
        config = graph.metadata.get("config")
        graph_config = self.bridge.serialize_process_graph(graph)
        return {
            "model": {
                "name": deployment_package.model_name,
                "version": deployment_package.version,
                "quantization_bits": int(deployment_package.quantization_bits),
            },
            "lava": {
                "available": bool(self.bridge.lava_available),
                "run_config": "Loihi2SimCfg" if self.bridge.lava_available else "MockLoihi2SimCfg",
                "run_condition": "RunSteps",
            },
            "process_graph": graph_config,
            "timing": {
                "dt_ms": float(getattr(getattr(config, "spiking", None), "dt", 1.0)),
                "estimated_latency_ms": float(deployment_package.hardware_reqs.estimated_latency_ms),
            },
            "weights": {
                "format": "int8",
                "activation_format": "int16",
                "export_hint": "Use ModelStore.export_for_loihi() for artifact emission.",
            },
            "hardware": asdict(deployment_package.hardware_reqs),
            "equivalence": asdict(deployment_package.equivalence),
            "ready_for_hardware": bool(deployment_package.ready_for_hardware),
        }


__all__ = ["LoihiDeploymentPipeline"]
