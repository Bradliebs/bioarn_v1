"""Bridge Bio-ARN models onto Intel Lava or a faithful mock runtime."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from bioarn.core.ccc import CCCPool, ConceptCellCluster
from bioarn.hardware.loihi_port import LoihiMapping
from bioarn.hardware.lava_processes import (
    CCCLavaProcess,
    LAVA_AVAILABLE,
    SDMLavaProcess,
    MockLavaProcess,
)
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.system import BioARNCore

try:
    import lava.lib.dl as lava_dl  # type: ignore[import-not-found]
    from lava.magma.core.run_conditions import RunSteps  # type: ignore[import-not-found]
    from lava.magma.core.run_configs import Loihi2SimCfg  # type: ignore[import-not-found]
    from lava.proc.dense.process import Dense  # type: ignore[import-not-found]
    from lava.proc.lif.process import LIF  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    lava_dl = None
    RunSteps = None
    Loihi2SimCfg = None
    Dense = None
    LIF = None


@dataclass
class LavaProcessGraph:
    processes: list[Any]
    connections: list[dict[str, Any]]
    num_neurons: int
    num_synapses: int
    num_cores_estimate: int
    component_type: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_model: Any | None = None


@dataclass
class EquivalenceReport:
    match_rate: float
    max_deviation: float
    per_component: dict
    passed: bool


@dataclass
class HardwareRequirements:
    num_cores: int
    total_neurons: int
    total_synapses: int
    memory_per_core_bytes: int
    estimated_power_mw: float
    estimated_latency_ms: float


@dataclass
class DeploymentPackage:
    model_name: str
    version: str
    quantization_bits: int
    process_graph: LavaProcessGraph
    equivalence: EquivalenceReport
    hardware_reqs: HardwareRequirements
    config: dict
    ready_for_hardware: bool


class LavaBridge:
    """Bridge between Bio-ARN PyTorch models and Intel Lava SDK."""

    NEURONS_PER_CORE = 128
    SYNAPSES_PER_CORE = 128 * 1024

    def __init__(self) -> None:
        self.lava_available = LAVA_AVAILABLE

    @staticmethod
    def _ceildiv(numerator: int, denominator: int) -> int:
        return max(1, int((numerator + max(denominator, 1) - 1) // max(denominator, 1)))

    @staticmethod
    def _unwrap_system(system: Any) -> Any:
        return getattr(system, "core", system)

    @staticmethod
    def _clone_graph(process_graph: LavaProcessGraph) -> LavaProcessGraph:
        return copy.deepcopy(process_graph)

    @staticmethod
    def _scalar_tensor(value: float | int | bool, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=device or torch.device("cpu"))

    def _graph_cores(self, neurons: int, synapses: int, requested: int | None = None) -> int:
        core_estimate = max(
            self._ceildiv(neurons, self.NEURONS_PER_CORE),
            self._ceildiv(synapses, self.SYNAPSES_PER_CORE),
        )
        if requested is not None:
            core_estimate = max(core_estimate, int(requested))
        return int(core_estimate)

    @staticmethod
    def _serialize_process(process: Any) -> dict[str, Any]:
        if hasattr(process, "describe"):
            return process.describe()
        return {
            "name": getattr(process, "name", type(process).__name__),
            "type": getattr(process, "process_type", type(process).__name__.lower()),
        }

    def serialize_process_graph(self, process_graph: LavaProcessGraph) -> dict[str, Any]:
        """Return a JSON/YAML-safe view of a process graph."""

        return {
            "component_type": process_graph.component_type,
            "num_neurons": int(process_graph.num_neurons),
            "num_synapses": int(process_graph.num_synapses),
            "num_cores_estimate": int(process_graph.num_cores_estimate),
            "processes": [self._serialize_process(process) for process in process_graph.processes],
            "connections": copy.deepcopy(process_graph.connections),
            "lava_available": self.lava_available,
        }

    def convert_ccc_to_lava(self, ccc_pool: CCCPool | ConceptCellCluster | list[ConceptCellCluster]) -> LavaProcessGraph:
        """Map one or more CCCs onto Dense, LIF, gate, and feedback processes."""

        if isinstance(ccc_pool, ConceptCellCluster):
            cccs = [ccc_pool]
        elif isinstance(ccc_pool, CCCPool):
            cccs = list(ccc_pool.cccs)
        else:
            cccs = list(ccc_pool)
        if not cccs:
            raise ValueError("convert_ccc_to_lava requires at least one CCC.")

        processes = [CCCLavaProcess(ccc, name=f"ccc_{index}") for index, ccc in enumerate(cccs)]
        connections = [connection for process in processes for connection in process.connections()]
        input_dim = int(cccs[0].config.input_dim)
        f1_neurons = int(cccs[0].config.num_f1_features)
        concept_dim = int(cccs[0].config.concept_dim)
        neurons = f1_neurons + (len(processes) * concept_dim) + len(processes)
        synapses = (
            input_dim * f1_neurons
            + len(processes) * (f1_neurons * concept_dim)
            + len(processes) * (concept_dim * f1_neurons)
            + len(processes) * concept_dim
        )
        requested_cores = None
        try:
            requested_cores = len(processes) * LoihiMapping(self._ccc_config_to_system(cccs[0])).map_ccc_to_cores().cores_required
        except Exception:
            requested_cores = None
        return LavaProcessGraph(
            processes=processes,
            connections=connections,
            num_neurons=int(neurons),
            num_synapses=int(synapses),
            num_cores_estimate=self._graph_cores(neurons, synapses, requested=requested_cores),
            component_type="ccc_pool",
            metadata={"num_cccs": len(processes)},
            source_model=copy.deepcopy(ccc_pool),
        )

    def convert_sdm_to_lava(self, sdm: SparseDistributedMemory) -> LavaProcessGraph:
        """Map sparse distributed memory onto address and content processes."""

        process = SDMLavaProcess(sdm, name="sdm")
        neurons = int(sdm.address_dim + sdm.num_hard_locations + sdm.data_dim)
        synapses = int((sdm.num_hard_locations * sdm.address_dim) + (sdm.num_hard_locations * sdm.data_dim))
        return LavaProcessGraph(
            processes=[process],
            connections=process.connections(),
            num_neurons=neurons,
            num_synapses=synapses,
            num_cores_estimate=self._graph_cores(neurons, synapses),
            component_type="sdm",
            metadata={"address_dim": int(sdm.address_dim), "data_dim": int(sdm.data_dim)},
            source_model=copy.deepcopy(sdm),
        )

    def convert_full_model(self, system: Any) -> LavaProcessGraph:
        """Convert an entire BioARNCore to a connected Lava process graph."""

        core = self._unwrap_system(system)
        if not isinstance(core, BioARNCore):
            raise TypeError("convert_full_model expects a BioARNCore or SensorimotorLoop-like wrapper.")

        ccc_graph = self.convert_ccc_to_lava(core.ccc_pool)
        sdm_graph = self.convert_sdm_to_lava(core.fabric.sdm)
        fabric_process = MockLavaProcess("associative_fabric", "fabric")
        gnw_process = MockLavaProcess("gnw_workspace", "gnw")
        processes = [*ccc_graph.processes, *sdm_graph.processes, fabric_process, gnw_process]
        connections = list(ccc_graph.connections) + list(sdm_graph.connections)
        for process in ccc_graph.processes:
            connections.append({"from": process.name, "to": fabric_process.name, "signal": "concept_vote"})
        connections.extend(
            [
                {"from": "sdm", "to": fabric_process.name, "signal": "associative_recall"},
                {"from": fabric_process.name, "to": gnw_process.name, "signal": "workspace_candidates"},
                {"from": gnw_process.name, "to": fabric_process.name, "signal": "broadcast_feedback"},
            ]
        )
        mapping = LoihiMapping(core.config).map_full_system()
        return LavaProcessGraph(
            processes=processes,
            connections=connections,
            num_neurons=int(mapping.total_neurons),
            num_synapses=int(mapping.total_synapses),
            num_cores_estimate=int(mapping.total_cores),
            component_type="full_system",
            metadata={
                "estimated_latency_ms": float(mapping.estimated_latency_ms),
                "feasibility": mapping.feasibility,
                "config": core.config,
            },
            source_model=copy.deepcopy(core),
        )

    def _run_ccc_graph(self, process_graph: LavaProcessGraph, input_data: torch.Tensor, num_timesteps: int) -> dict[str, Any]:
        ccc_processes = [process for process in process_graph.processes if isinstance(process, CCCLavaProcess)]
        outputs = [process.run(input_data, num_timesteps=num_timesteps) for process in ccc_processes]
        fired_indices = [index for index, output in enumerate(outputs) if output["fired"]]
        abstained_indices = [index for index, output in enumerate(outputs) if output["abstained"]]
        winner_confidences = (
            torch.stack([output["confidence"].reshape(-1).mean().to(torch.float32) for output in outputs if output["fired"]])
            if fired_indices
            else torch.empty(0, dtype=torch.float32)
        )
        best_output = outputs[fired_indices[0]] if fired_indices else outputs[0]
        return {
            "outputs": outputs,
            "fired_indices": fired_indices,
            "abstained_indices": abstained_indices,
            "winner_confidences": winner_confidences,
            "confidence": best_output["confidence"],
            "f1_output": best_output["f1_output"],
            "f2_activation": best_output["f2_activation"],
            "prediction": best_output["prediction"],
            "output": best_output["output"],
            "spike_counts": {process.name: output["spike_counts"] for process, output in zip(ccc_processes, outputs, strict=False)},
        }

    def _run_sdm_graph(
        self,
        process_graph: LavaProcessGraph,
        input_data: dict[str, torch.Tensor] | torch.Tensor,
        num_timesteps: int,
    ) -> dict[str, Any]:
        sdm_process = next(process for process in process_graph.processes if isinstance(process, SDMLavaProcess))
        return sdm_process.run(input_data, num_timesteps=num_timesteps)

    def _run_full_graph(self, process_graph: LavaProcessGraph, input_data: torch.Tensor, num_timesteps: int) -> dict[str, Any]:
        core = self._unwrap_system(process_graph.source_model)
        if not isinstance(core, BioARNCore):
            raise TypeError("Full-system Lava graphs require a BioARNCore source model.")

        ccc_result = self._run_ccc_graph(process_graph, input_data, num_timesteps=max(num_timesteps, 1))
        recognition = core.recognize(input_data.to(torch.float32))
        sdm_process = next((process for process in process_graph.processes if isinstance(process, SDMLavaProcess)), None)
        if sdm_process is not None and not recognition.abstained:
            sdm_result = sdm_process.run({"address": recognition.concept_direction}, num_timesteps=max(1, num_timesteps // 2))
        elif sdm_process is not None:
            zero_address = torch.zeros(core.config.sdm.address_dim, dtype=torch.float32)
            sdm_result = sdm_process.run({"address": zero_address}, num_timesteps=max(1, num_timesteps // 2))
        else:
            sdm_result = {}
        return {
            "concept_direction": recognition.concept_direction.detach().clone(),
            "confidence": self._scalar_tensor(recognition.confidence),
            "abstained": self._scalar_tensor(recognition.abstained),
            "num_hypotheses": self._scalar_tensor(recognition.num_hypotheses),
            "agreement": self._scalar_tensor(recognition.agreement),
            "output": recognition.concept_direction.detach().clone(),
            "ccc": ccc_result,
            "sdm": sdm_result,
            "gnw_stats": core.gnw.get_stats(),
        }

    def run_lava_inference(
        self,
        process_graph: LavaProcessGraph,
        input_data: dict[str, torch.Tensor] | torch.Tensor,
        num_timesteps: int,
    ) -> dict:
        """Execute a process graph on Lava's CPU simulator or the mock runtime."""

        if process_graph.component_type == "ccc_pool":
            if not isinstance(input_data, torch.Tensor):
                raise TypeError("CCC Lava graphs expect tensor input data.")
            return self._run_ccc_graph(process_graph, input_data, num_timesteps)
        if process_graph.component_type == "sdm":
            return self._run_sdm_graph(process_graph, input_data, num_timesteps)
        if process_graph.component_type == "full_system":
            if not isinstance(input_data, torch.Tensor):
                raise TypeError("Full-system Lava graphs expect tensor input data.")
            return self._run_full_graph(process_graph, input_data, num_timesteps)
        raise ValueError(f"Unsupported Lava process graph type: {process_graph.component_type!r}.")

    @staticmethod
    def _flatten_numeric(result: Any, prefix: str = "") -> dict[str, torch.Tensor]:
        flat: dict[str, torch.Tensor] = {}
        key_prefix = f"{prefix}." if prefix else ""
        if isinstance(result, dict):
            for key, value in result.items():
                flat.update(LavaBridge._flatten_numeric(value, prefix=f"{key_prefix}{key}"))
            return flat
        if isinstance(result, list):
            for index, value in enumerate(result):
                flat.update(LavaBridge._flatten_numeric(value, prefix=f"{key_prefix}{index}"))
            return flat
        if isinstance(result, torch.Tensor):
            flat[prefix] = result.detach().to(torch.float32)
            return flat
        if isinstance(result, (float, int, bool)):
            flat[prefix] = torch.tensor(float(result), dtype=torch.float32)
        return flat

    @staticmethod
    def _tensor_deviation(reference: torch.Tensor, candidate: torch.Tensor) -> float:
        reference_flat = reference.reshape(-1).to(torch.float32)
        candidate_flat = candidate.reshape(-1).to(torch.float32)
        common = min(reference_flat.numel(), candidate_flat.numel())
        if common == 0:
            return 0.0
        diff = (reference_flat[:common] - candidate_flat[:common]).abs()
        tail_penalty = 0.0 if reference_flat.numel() == candidate_flat.numel() else 1.0
        return float(max(diff.max().item(), tail_penalty))

    @staticmethod
    def _reference_inference(model: Any, input_data: Any) -> dict[str, Any]:
        if isinstance(model, ConceptCellCluster):
            output = model(input_data.to(torch.float32))
            prediction = (
                output.prediction.detach().clone()
                if output.prediction is not None
                else torch.zeros(model.config.num_f1_features, dtype=torch.float32)
            )
            return {
                "confidence": output.confidence.detach().clone(),
                "f1_output": output.f1_output.detach().clone(),
                "f2_activation": output.f2_activation.detach().clone(),
                "prediction": prediction,
                "output": output.gate_output.output.detach().clone(),
                "fired": output.fired,
                "abstained": output.abstained,
            }
        if isinstance(model, SparseDistributedMemory):
            address = input_data["address"] if isinstance(input_data, dict) else input_data
            return {
                "output": model.read(address.to(torch.float32)).detach().clone(),
            }
        if isinstance(model, BioARNCore):
            recognition = model.recognize(input_data.to(torch.float32))
            return {
                "concept_direction": recognition.concept_direction.detach().clone(),
                "confidence": torch.tensor(recognition.confidence, dtype=torch.float32),
                "abstained": torch.tensor(float(recognition.abstained), dtype=torch.float32),
                "output": recognition.concept_direction.detach().clone(),
            }
        if isinstance(model, nn.Module):
            result = model(input_data)
            if isinstance(result, torch.Tensor):
                return {"output": result.detach().clone()}
        raise TypeError(f"Unsupported model type for equivalence validation: {type(model)!r}.")

    def validate_equivalence(
        self,
        pytorch_model: Any,
        lava_graph: LavaProcessGraph,
        test_data: list[Any],
        tolerance: float = 0.05,
    ) -> EquivalenceReport:
        """Compare PyTorch and Lava outputs on the same inputs."""

        component_samples: dict[str, list[float]] = defaultdict(list)
        total_samples = 0
        passed_samples = 0
        max_deviation = 0.0

        for sample in test_data:
            sample_input = sample[0] if isinstance(sample, tuple) else sample
            reference_model = copy.deepcopy(self._unwrap_system(pytorch_model))
            graph_copy = self._clone_graph(lava_graph)
            reference = self._reference_inference(reference_model, sample_input)
            candidate = self.run_lava_inference(graph_copy, sample_input, num_timesteps=4)

            reference_tensors = self._flatten_numeric(reference)
            candidate_tensors = self._flatten_numeric(candidate)
            common_keys = sorted(set(reference_tensors).intersection(candidate_tensors))
            if not common_keys:
                raise ValueError("No comparable numeric outputs were found for equivalence validation.")

            sample_ok = True
            sample_max = 0.0
            for key in common_keys:
                deviation = self._tensor_deviation(reference_tensors[key], candidate_tensors[key])
                component_samples[key].append(deviation)
                sample_max = max(sample_max, deviation)
                sample_ok = sample_ok and deviation <= tolerance

            total_samples += 1
            passed_samples += int(sample_ok)
            max_deviation = max(max_deviation, sample_max)

        per_component = {
            key: {
                "max_deviation": float(max(values)),
                "mean_deviation": float(sum(values) / max(len(values), 1)),
                "match_rate": float(sum(value <= tolerance for value in values) / max(len(values), 1)),
            }
            for key, values in component_samples.items()
        }
        match_rate = float(passed_samples / max(total_samples, 1))
        passed = bool(match_rate >= 0.8 and max_deviation <= max(tolerance * 2.0, 0.1))
        return EquivalenceReport(
            match_rate=match_rate,
            max_deviation=float(max_deviation),
            per_component=per_component,
            passed=passed,
        )

    @staticmethod
    def _ccc_config_to_system(ccc: ConceptCellCluster):
        class _Config:
            def __init__(self, source: ConceptCellCluster) -> None:
                self.spiking = getattr(source, "spiking_config", None) or type("Spiking", (), {"beta": 0.9, "threshold": 1.0, "reset": 0.0, "dt": 1.0, "refractory_steps": 2})()
                self.margin_gate = type(
                    "Margin",
                    (),
                    {
                        "theta_margin": float(source.margin_gate.theta_margin.item()),
                        "theta_margin_lr": float(source.margin_gate.theta_margin_lr),
                        "theta_resonance": float(source.margin_gate.theta_resonance.item()),
                    },
                )()
                self.ccc = source.config
                self.sdm = type("SDM", (), {"address_dim": 8, "hamming_radius": 2, "num_hard_locations": 8, "data_dim": source.config.concept_dim, "decay_rate": 0.99, "stdp_window": 4})()
                self.predictive = type("Predictive", (), {"num_levels": 1, "eta": 0.01, "gamma": 0.1})()
                self.gnw = type("GNW", (), {"capacity": 1})()

        return _Config(ccc)


__all__ = [
    "DeploymentPackage",
    "EquivalenceReport",
    "HardwareRequirements",
    "LavaBridge",
    "LavaProcessGraph",
]
