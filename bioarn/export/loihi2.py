"""Portable Loihi 2 export for CCC pools and visual hierarchies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from bioarn.core.ccc import CCCPool
from bioarn.hierarchy.visual_hierarchy import VisualHierarchy
from bioarn.persistence._common import atomic_write_json, ensure_dir
from bioarn.scaling import BatchedCCCPool


JSONScalar = str | int | float | bool | None


@dataclass
class Loihi2Config:
    """Mapping defaults for a Loihi 2 compatible graph description."""

    cores_per_chip: int = 128
    max_chips: int = 1
    neurons_per_core: int = 128
    synapse_sram_per_core_bytes: int = 128 * 1024
    weight_bits: int = 8
    state_bits: int = 16
    spike_threshold: float = 1.0
    membrane_decay: float = 0.9
    reset_potential: float = 0.0
    refractory_steps: int = 2
    synaptic_delay_steps: int = 1
    lateral_inhibition_weight: float = -0.25
    winner_take_all_inhibition: float = -1.0
    enable_feedback: bool = True

    def allocate_population(self, size: int, start_core: int = 0) -> "CoreAllocation":
        """Allocate one population across Loihi cores."""

        if size < 0:
            raise ValueError("Population size must be non-negative.")

        remaining = int(size)
        current_core = int(start_core)
        core_ids: list[int] = []
        neurons_per_core: list[int] = []
        max_cores = int(self.cores_per_chip * self.max_chips)

        while remaining > 0:
            if current_core >= max_cores:
                raise ValueError("Population exceeds configured Loihi 2 core budget.")
            assigned = min(remaining, int(self.neurons_per_core))
            core_ids.append(current_core)
            neurons_per_core.append(int(assigned))
            remaining -= assigned
            current_core += 1

        return CoreAllocation(core_ids=core_ids, neurons_per_core=neurons_per_core)

    def lif_parameters(self) -> dict[str, JSONScalar]:
        """Return default LIF parameters for exported populations."""

        return {
            "threshold": float(self.spike_threshold),
            "membrane_decay": float(self.membrane_decay),
            "reset_potential": float(self.reset_potential),
            "refractory_steps": int(self.refractory_steps),
            "weight_bits": int(self.weight_bits),
            "state_bits": int(self.state_bits),
        }


@dataclass
class CoreAllocation:
    """Placement of one population across Loihi cores."""

    core_ids: list[int] = field(default_factory=list)
    neurons_per_core: list[int] = field(default_factory=list)

    @property
    def next_core(self) -> int:
        """Return the next available core after this allocation."""

        if not self.core_ids:
            return 0
        return int(self.core_ids[-1] + 1)


@dataclass
class NeuromorphicPopulation:
    """One neuromorphic population in the exported graph."""

    id: str
    size: int
    neuron_model: str
    role: str
    external: bool = False
    allocation: CoreAllocation | None = None
    parameters: dict[str, JSONScalar] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapticProjection:
    """One synaptic projection between exported populations."""

    id: str
    source: str
    target: str
    weight_shape: tuple[int, int]
    learning_rule: str
    delay_steps: int
    weights: list[list[float]] | None = None
    bias: list[float] | None = None
    pattern: str | None = None
    scalar_weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CCCSnapshot:
    """Portable snapshot of one committed CCC."""

    index: int
    input_dim: int
    num_f1_features: int
    concept_dim: int
    f1_weights: list[list[float]]
    f1_bias: list[float]
    f2_weights: list[list[float]]
    feedback_weights: list[list[float]]
    concept_direction: list[float]
    theta_margin: float
    age: int
    last_fired: int


@dataclass
class PoolSnapshot:
    """Portable snapshot of a CCC pool."""

    name: str
    input_dim: int
    concept_dim: int
    num_f1_features: int
    total_slots: int
    committed_count: int
    cccs: list[CCCSnapshot] = field(default_factory=list)


@dataclass
class NeuromorphicGraph:
    """Intermediate graph mapping Bio-ARN concepts to neuromorphic primitives."""

    name: str
    config: Loihi2Config = field(default_factory=Loihi2Config)
    populations: list[NeuromorphicPopulation] = field(default_factory=list)
    projections: list[SynapticProjection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_core: int = field(default=0, init=False, repr=False)

    def add_population(
        self,
        *,
        id: str,
        size: int,
        neuron_model: str,
        role: str,
        external: bool = False,
        parameters: dict[str, JSONScalar] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NeuromorphicPopulation:
        """Add one population and allocate cores when needed."""

        allocation = None
        if not external:
            allocation = self.config.allocate_population(size, start_core=self._next_core)
            self._next_core = allocation.next_core
        population = NeuromorphicPopulation(
            id=id,
            size=int(size),
            neuron_model=neuron_model,
            role=role,
            external=bool(external),
            allocation=allocation,
            parameters=dict(parameters or {}),
            metadata=dict(metadata or {}),
        )
        self.populations.append(population)
        return population

    def add_projection(
        self,
        *,
        id: str,
        source: str,
        target: str,
        weight_shape: tuple[int, int],
        learning_rule: str,
        delay_steps: int | None = None,
        weights: list[list[float]] | None = None,
        bias: list[float] | None = None,
        pattern: str | None = None,
        scalar_weight: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SynapticProjection:
        """Add one synaptic projection."""

        projection = SynapticProjection(
            id=id,
            source=source,
            target=target,
            weight_shape=(int(weight_shape[0]), int(weight_shape[1])),
            learning_rule=learning_rule,
            delay_steps=int(delay_steps if delay_steps is not None else self.config.synaptic_delay_steps),
            weights=weights,
            bias=bias,
            pattern=pattern,
            scalar_weight=None if scalar_weight is None else float(scalar_weight),
            metadata=dict(metadata or {}),
        )
        self.projections.append(projection)
        return projection

    def validate(self) -> None:
        """Validate referential integrity and matrix dimensions."""

        population_ids = [population.id for population in self.populations]
        if len(population_ids) != len(set(population_ids)):
            raise ValueError("Population ids must be unique.")

        known = set(population_ids)
        for projection in self.projections:
            if projection.source not in known or projection.target not in known:
                raise ValueError(
                    f"Projection {projection.id!r} references unknown populations."
                )
            if projection.weights is not None:
                expected_rows, expected_cols = projection.weight_shape
                if len(projection.weights) != expected_rows:
                    raise ValueError(
                        f"Projection {projection.id!r} expected {expected_rows} rows."
                    )
                if any(len(row) != expected_cols for row in projection.weights):
                    raise ValueError(
                        f"Projection {projection.id!r} expected {expected_cols} columns."
                    )
            if projection.bias is not None and len(projection.bias) != projection.weight_shape[0]:
                raise ValueError(
                    f"Projection {projection.id!r} bias must match target width."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe graph payload."""

        self.validate()
        return {
            "name": self.name,
            "backend": "loihi2",
            "config": asdict(self.config),
            "populations": [asdict(population) for population in self.populations],
            "projections": [asdict(projection) for projection in self.projections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuromorphicGraph":
        """Restore a graph from serialized JSON data."""

        graph = cls(
            name=str(payload["name"]),
            config=Loihi2Config(**payload.get("config", {})),
            metadata=dict(payload.get("metadata", {})),
        )

        max_core = -1
        for raw_population in payload.get("populations", []):
            allocation_payload = raw_population.get("allocation")
            allocation = (
                CoreAllocation(**allocation_payload)
                if allocation_payload is not None
                else None
            )
            graph.populations.append(
                NeuromorphicPopulation(
                    id=str(raw_population["id"]),
                    size=int(raw_population["size"]),
                    neuron_model=str(raw_population["neuron_model"]),
                    role=str(raw_population["role"]),
                    external=bool(raw_population.get("external", False)),
                    allocation=allocation,
                    parameters=dict(raw_population.get("parameters", {})),
                    metadata=dict(raw_population.get("metadata", {})),
                )
            )
            if allocation and allocation.core_ids:
                max_core = max(max_core, max(allocation.core_ids))

        for raw_projection in payload.get("projections", []):
            graph.projections.append(
                SynapticProjection(
                    id=str(raw_projection["id"]),
                    source=str(raw_projection["source"]),
                    target=str(raw_projection["target"]),
                    weight_shape=tuple(raw_projection["weight_shape"]),
                    learning_rule=str(raw_projection["learning_rule"]),
                    delay_steps=int(raw_projection["delay_steps"]),
                    weights=raw_projection.get("weights"),
                    bias=raw_projection.get("bias"),
                    pattern=raw_projection.get("pattern"),
                    scalar_weight=raw_projection.get("scalar_weight"),
                    metadata=dict(raw_projection.get("metadata", {})),
                )
            )

        graph._next_core = max_core + 1
        graph.validate()
        return graph


def export_ccc_pool(
    pool: CCCPool,
    path: Path,
    config: Loihi2Config | None = None,
) -> NeuromorphicGraph:
    """Export a trained CCC pool as a portable Loihi 2 graph."""

    graph = NeuromorphicGraph(name="ccc_pool", config=config or Loihi2Config())
    pool_snapshot = _snapshot_ccc_pool(pool, name="ccc_pool")
    input_population = graph.add_population(
        id="ccc_pool_input",
        size=pool_snapshot.input_dim,
        neuron_model="spike_input",
        role="input",
        external=True,
        metadata={"source": "bioarn.core.ccc.CCCPool"},
    )
    _append_pool_subgraph(
        graph,
        pool_snapshot,
        input_population_id=input_population.id,
        pool_prefix="ccc_pool",
    )
    graph.metadata.update(
        {
            "kind": "ccc_pool",
            "committed_count": int(pool_snapshot.committed_count),
            "total_slots": int(pool_snapshot.total_slots),
        }
    )
    _write_export_artifacts(graph, Path(path), default_stem="ccc_pool")
    return graph


def export_hierarchy(
    hierarchy: VisualHierarchy,
    path: Path,
    config: Loihi2Config | None = None,
) -> NeuromorphicGraph:
    """Export a trained visual hierarchy as a portable Loihi 2 graph."""

    graph = NeuromorphicGraph(name="visual_hierarchy", config=config or Loihi2Config())
    layer_order = [layer.name for layer in hierarchy.layers]
    committed_by_layer: dict[str, int] = {}
    previous_output: NeuromorphicPopulation | None = None

    for layer_index, layer in enumerate(hierarchy.layers):
        layer_prefix = layer.name.lower()
        input_population = graph.add_population(
            id=f"{layer_prefix}_input",
            size=int(layer.input_dim),
            neuron_model="spike_input" if layer_index == 0 else "lif",
            role="layer_input",
            external=layer_index == 0,
            parameters={} if layer_index == 0 else graph.config.lif_parameters(),
            metadata={"layer": layer.name, "layer_index": int(layer_index)},
        )
        output_population = graph.add_population(
            id=f"{layer_prefix}_output",
            size=int(layer.concept_dim),
            neuron_model="lif",
            role="layer_output",
            parameters=graph.config.lif_parameters(),
            metadata={
                "layer": layer.name,
                "layer_index": int(layer_index),
                "winner_limit": int(layer.winner_limit),
            },
        )

        if previous_output is not None:
            weights = _repeat_identity_weights(previous_output.size, input_population.size)
            graph.add_projection(
                id=f"{previous_output.id}_to_{input_population.id}",
                source=previous_output.id,
                target=input_population.id,
                weight_shape=(input_population.size, previous_output.size),
                learning_rule="fixed",
                weights=weights,
                metadata={
                    "role": "feedforward",
                    "fan_in": int(input_population.size // max(previous_output.size, 1)),
                    "pooling_mode": "spatial_grouping",
                },
            )

        pool_snapshot = _snapshot_ccc_pool(layer.pool.core, name=layer.name)
        committed_by_layer[layer.name] = int(pool_snapshot.committed_count)
        _append_pool_subgraph(
            graph,
            pool_snapshot,
            input_population_id=input_population.id,
            pool_prefix=layer_prefix,
            output_population_id=output_population.id,
            layer_name=layer.name,
        )
        previous_output = output_population

    graph.metadata.update(
        {
            "kind": "visual_hierarchy",
            "layer_order": layer_order,
            "committed_by_layer": committed_by_layer,
            "label_counts": {str(label): int(count) for label, count in hierarchy.label_counts.items()},
            "label_prototypes": {
                str(label): _tensor_to_vector(prototype)
                for label, prototype in hierarchy.label_prototypes.items()
            },
        }
    )
    _write_export_artifacts(graph, Path(path), default_stem="visual_hierarchy")
    return graph


def _append_pool_subgraph(
    graph: NeuromorphicGraph,
    pool_snapshot: PoolSnapshot,
    *,
    input_population_id: str,
    pool_prefix: str,
    output_population_id: str | None = None,
    layer_name: str | None = None,
) -> None:
    gate_ids: list[str] = []

    for ccc in pool_snapshot.cccs:
        ccc_prefix = f"{pool_prefix}_ccc_{ccc.index}"
        f1 = graph.add_population(
            id=f"{ccc_prefix}_f1",
            size=ccc.num_f1_features,
            neuron_model="lif",
            role="feature_population",
            parameters=graph.config.lif_parameters(),
            metadata={
                "ccc_index": int(ccc.index),
                "layer": layer_name,
            },
        )
        f2 = graph.add_population(
            id=f"{ccc_prefix}_f2",
            size=ccc.concept_dim,
            neuron_model="lif",
            role="concept_population",
            parameters=graph.config.lif_parameters(),
            metadata={
                "ccc_index": int(ccc.index),
                "layer": layer_name,
                "age": int(ccc.age),
                "last_fired": int(ccc.last_fired),
            },
        )
        gate = graph.add_population(
            id=f"{ccc_prefix}_gate",
            size=1,
            neuron_model="lif",
            role="margin_gate",
            parameters={
                **graph.config.lif_parameters(),
                "threshold": float(ccc.theta_margin),
            },
            metadata={
                "ccc_index": int(ccc.index),
                "layer": layer_name,
            },
        )
        gate_ids.append(gate.id)

        graph.add_projection(
            id=f"{input_population_id}_to_{f1.id}",
            source=input_population_id,
            target=f1.id,
            weight_shape=(ccc.num_f1_features, ccc.input_dim),
            learning_rule="fixed",
            weights=ccc.f1_weights,
            bias=ccc.f1_bias,
            metadata={"role": "f1_encoder", "layer": layer_name},
        )
        graph.add_projection(
            id=f"{f1.id}_to_{f2.id}",
            source=f1.id,
            target=f2.id,
            weight_shape=(ccc.concept_dim, ccc.num_f1_features),
            learning_rule="fixed",
            weights=ccc.f2_weights,
            metadata={"role": "concept_projection", "layer": layer_name},
        )
        if graph.config.enable_feedback:
            graph.add_projection(
                id=f"{f2.id}_to_{f1.id}",
                source=f2.id,
                target=f1.id,
                weight_shape=(ccc.num_f1_features, ccc.concept_dim),
                learning_rule="hebbian",
                weights=ccc.feedback_weights,
                metadata={"role": "feedback", "layer": layer_name},
            )
        graph.add_projection(
            id=f"{f1.id}_lateral_inhibition",
            source=f1.id,
            target=f1.id,
            weight_shape=(ccc.num_f1_features, ccc.num_f1_features),
            learning_rule="fixed",
            pattern="all_to_all_inhibitory",
            scalar_weight=float(graph.config.lateral_inhibition_weight),
            metadata={
                "role": "lateral_inhibition",
                "exclude_self": True,
                "layer": layer_name,
            },
        )
        graph.add_projection(
            id=f"{f2.id}_to_{gate.id}",
            source=f2.id,
            target=gate.id,
            weight_shape=(1, ccc.concept_dim),
            learning_rule="fixed",
            weights=[list(ccc.concept_direction)],
            metadata={
                "role": "concept_readout",
                "layer": layer_name,
                "theta_margin": float(ccc.theta_margin),
            },
        )
        if output_population_id is not None:
            graph.add_projection(
                id=f"{gate.id}_to_{output_population_id}",
                source=gate.id,
                target=output_population_id,
                weight_shape=(ccc.concept_dim, 1),
                learning_rule="fixed",
                weights=[[float(value)] for value in ccc.concept_direction],
                metadata={
                    "role": "layer_readout",
                    "layer": layer_name,
                    "ccc_index": int(ccc.index),
                },
            )

    if not gate_ids:
        return

    wta = graph.add_population(
        id=f"{pool_prefix}_wta",
        size=len(gate_ids),
        neuron_model="lif",
        role="winner_take_all",
        parameters=graph.config.lif_parameters(),
        metadata={"layer": layer_name, "members": list(gate_ids)},
    )
    for gate_position, gate_id in enumerate(gate_ids):
        one_hot = [[0.0] for _ in range(len(gate_ids))]
        one_hot[gate_position][0] = 1.0
        graph.add_projection(
            id=f"{gate_id}_to_{wta.id}",
            source=gate_id,
            target=wta.id,
            weight_shape=(len(gate_ids), 1),
            learning_rule="fixed",
            weights=one_hot,
            metadata={
                "role": "winner_vote",
                "layer": layer_name,
                "winner_index": int(gate_position),
            },
        )
    graph.add_projection(
        id=f"{wta.id}_recurrent_inhibition",
        source=wta.id,
        target=wta.id,
        weight_shape=(len(gate_ids), len(gate_ids)),
        learning_rule="fixed",
        pattern="all_to_all_inhibitory",
        scalar_weight=float(graph.config.winner_take_all_inhibition),
        metadata={
            "role": "winner_take_all_competition",
            "exclude_self": True,
            "layer": layer_name,
        },
    )


def _snapshot_ccc_pool(pool: CCCPool | BatchedCCCPool, *, name: str) -> PoolSnapshot:
    if isinstance(pool, BatchedCCCPool):
        committed_indices = (
            pool.committed_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        )
        cccs = [
            CCCSnapshot(
                index=int(index),
                input_dim=int(pool.config.input_dim),
                num_f1_features=int(pool.config.num_f1_features),
                concept_dim=int(pool.config.concept_dim),
                f1_weights=_tensor_to_matrix(pool.f1_weights[index]),
                f1_bias=_tensor_to_vector(pool.f1_bias[index]),
                f2_weights=_tensor_to_matrix(pool.f2_weights[index]),
                feedback_weights=_tensor_to_matrix(pool.feedback_weights[index]),
                concept_direction=_tensor_to_vector(pool.concept_directions[index]),
                theta_margin=float(pool.theta_margin[index].item()),
                age=int(pool.age[index].item()),
                last_fired=int(pool.last_fired[index].item()),
            )
            for index in committed_indices
        ]
        return PoolSnapshot(
            name=name,
            input_dim=int(pool.config.input_dim),
            concept_dim=int(pool.config.concept_dim),
            num_f1_features=int(pool.config.num_f1_features),
            total_slots=int(pool.config.max_pool_size),
            committed_count=len(cccs),
            cccs=cccs,
        )

    cccs = [
        CCCSnapshot(
            index=int(index),
            input_dim=int(ccc.config.input_dim),
            num_f1_features=int(ccc.config.num_f1_features),
            concept_dim=int(ccc.config.concept_dim),
            f1_weights=_tensor_to_matrix(ccc.f1_layer.weight.detach()),
            f1_bias=_tensor_to_vector(ccc.f1_layer.bias.detach()),
            f2_weights=_tensor_to_matrix(ccc.f2_weights.detach()),
            feedback_weights=_tensor_to_matrix(ccc.feedback_weights.detach()),
            concept_direction=_tensor_to_vector(ccc.concept_direction.detach()),
            theta_margin=float(ccc.margin_gate.theta_margin.item()),
            age=int(ccc.age.item()),
            last_fired=int(ccc.last_fired.item()),
        )
        for index, ccc in enumerate(pool.cccs)
        if bool(ccc.is_committed.item())
    ]
    return PoolSnapshot(
        name=name,
        input_dim=int(pool.config.input_dim),
        concept_dim=int(pool.config.concept_dim),
        num_f1_features=int(pool.config.num_f1_features),
        total_slots=int(pool.config.max_pool_size),
        committed_count=len(cccs),
        cccs=cccs,
    )


def _repeat_identity_weights(source_dim: int, target_dim: int) -> list[list[float]]:
    if source_dim <= 0 or target_dim <= 0:
        return []
    if target_dim % source_dim != 0:
        raise ValueError("Hierarchy feedforward mapping expects target_dim to be a multiple of source_dim.")

    identity = torch.eye(source_dim, dtype=torch.float32)
    repeats = target_dim // source_dim
    return _tensor_to_matrix(identity.repeat(repeats, 1))


def _tensor_to_matrix(tensor: torch.Tensor) -> list[list[float]]:
    matrix = tensor.detach().to(torch.float32).cpu()
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
    return [[float(value) for value in row] for row in matrix.tolist()]


def _tensor_to_vector(tensor: torch.Tensor) -> list[float]:
    vector = tensor.detach().to(torch.float32).reshape(-1).cpu()
    return [float(value) for value in vector.tolist()]


def _resolve_export_paths(path: Path, *, default_stem: str) -> tuple[Path, Path]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        loihi_path = target
        nir_path = target.with_name(f"{target.stem}.nir.json")
    else:
        ensure_dir(target)
        loihi_path = target / f"{default_stem}.loihi2.json"
        nir_path = target / f"{default_stem}.nir.json"
    return loihi_path, nir_path


def _write_export_artifacts(graph: NeuromorphicGraph, path: Path, *, default_stem: str) -> None:
    from bioarn.export.nir_format import graph_to_nir

    loihi_path, nir_path = _resolve_export_paths(path, default_stem=default_stem)
    ensure_dir(loihi_path.parent)
    ensure_dir(nir_path.parent)

    nir_document = graph_to_nir(graph)
    atomic_write_json(
        loihi_path,
        {
            "format": "bioarn.loihi2.export",
            "version": 1,
            "graph": graph.to_dict(),
            "nir": nir_document.to_dict(),
        },
    )
    atomic_write_json(nir_path, nir_document.to_dict())


__all__ = [
    "CCCSnapshot",
    "CoreAllocation",
    "Loihi2Config",
    "NeuromorphicGraph",
    "NeuromorphicPopulation",
    "PoolSnapshot",
    "SynapticProjection",
    "export_ccc_pool",
    "export_hierarchy",
]
