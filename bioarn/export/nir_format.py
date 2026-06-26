"""NIR export helpers for Bio-ARN neuromorphic graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bioarn.export.loihi2 import (
    CoreAllocation,
    Loihi2Config,
    NeuromorphicGraph,
    NeuromorphicPopulation,
    SynapticProjection,
)
from bioarn.persistence._common import atomic_write_json


@dataclass
class NIRPopulation:
    """Population node in a Neuromorphic Intermediate Representation graph."""

    id: str
    node_type: str
    size: int
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NIRConnection:
    """Connection edge in NIR."""

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
class NIRDocument:
    """Portable NIR document."""

    version: str = "0.1"
    backend: str = "nir"
    populations: list[NIRPopulation] = field(default_factory=list)
    connections: list[NIRConnection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe NIR payload."""

        return {
            "version": self.version,
            "backend": self.backend,
            "populations": [asdict(population) for population in self.populations],
            "connections": [asdict(connection) for connection in self.connections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NIRDocument":
        """Restore a NIR document from serialized data."""

        return cls(
            version=str(payload.get("version", "0.1")),
            backend=str(payload.get("backend", "nir")),
            populations=[
                NIRPopulation(
                    id=str(population["id"]),
                    node_type=str(population["node_type"]),
                    size=int(population["size"]),
                    parameters=dict(population.get("parameters", {})),
                    metadata=dict(population.get("metadata", {})),
                )
                for population in payload.get("populations", [])
            ],
            connections=[
                NIRConnection(
                    id=str(connection["id"]),
                    source=str(connection["source"]),
                    target=str(connection["target"]),
                    weight_shape=tuple(connection["weight_shape"]),
                    learning_rule=str(connection["learning_rule"]),
                    delay_steps=int(connection["delay_steps"]),
                    weights=connection.get("weights"),
                    bias=connection.get("bias"),
                    pattern=connection.get("pattern"),
                    scalar_weight=connection.get("scalar_weight"),
                    metadata=dict(connection.get("metadata", {})),
                )
                for connection in payload.get("connections", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )


def graph_to_nir(graph: NeuromorphicGraph) -> NIRDocument:
    """Convert a Bio-ARN neuromorphic graph into NIR."""

    graph.validate()
    return NIRDocument(
        populations=[
            NIRPopulation(
                id=population.id,
                node_type="input" if population.external else "lif",
                size=population.size,
                parameters={
                    **population.parameters,
                    "role": population.role,
                    "external": population.external,
                },
                metadata={
                    **population.metadata,
                    "neuron_model": population.neuron_model,
                    "allocation": asdict(population.allocation) if population.allocation else None,
                },
            )
            for population in graph.populations
        ],
        connections=[
            NIRConnection(
                id=projection.id,
                source=projection.source,
                target=projection.target,
                weight_shape=projection.weight_shape,
                learning_rule=projection.learning_rule,
                delay_steps=projection.delay_steps,
                weights=projection.weights,
                bias=projection.bias,
                pattern=projection.pattern,
                scalar_weight=projection.scalar_weight,
                metadata=dict(projection.metadata),
            )
            for projection in graph.projections
        ],
        metadata={
            "source_kind": graph.metadata.get("kind", "graph"),
            "graph_name": graph.name,
            "loihi2_config": asdict(graph.config),
            **graph.metadata,
        },
    )


def nir_to_graph(document: NIRDocument) -> NeuromorphicGraph:
    """Restore a Bio-ARN neuromorphic graph from NIR."""

    metadata = dict(document.metadata)
    graph = NeuromorphicGraph(
        name=str(metadata.pop("graph_name", "nir_graph")),
        config=Loihi2Config(**metadata.pop("loihi2_config", {})),
        metadata=metadata,
    )

    max_core = -1
    for population in document.populations:
        allocation_payload = population.metadata.get("allocation")
        allocation = (
            CoreAllocation(**allocation_payload)
            if allocation_payload is not None
            else None
        )
        graph.populations.append(
            NeuromorphicPopulation(
                id=population.id,
                size=population.size,
                neuron_model=str(population.metadata.get("neuron_model", population.node_type)),
                role=str(population.parameters.get("role", population.node_type)),
                external=bool(population.parameters.get("external", population.node_type == "input")),
                allocation=allocation,
                parameters={
                    key: value
                    for key, value in population.parameters.items()
                    if key not in {"role", "external"}
                },
                metadata={
                    key: value
                    for key, value in population.metadata.items()
                    if key != "allocation"
                },
            )
        )
        if allocation and allocation.core_ids:
            max_core = max(max_core, max(allocation.core_ids))

    for connection in document.connections:
        graph.projections.append(
            SynapticProjection(
                id=connection.id,
                source=connection.source,
                target=connection.target,
                weight_shape=connection.weight_shape,
                learning_rule=connection.learning_rule,
                delay_steps=connection.delay_steps,
                weights=connection.weights,
                bias=connection.bias,
                pattern=connection.pattern,
                scalar_weight=connection.scalar_weight,
                metadata=dict(connection.metadata),
            )
        )

    graph._next_core = max_core + 1
    graph.validate()
    return graph


def write_nir(document: NIRDocument, path: str | Path) -> Path:
    """Write a NIR document to disk."""

    target = Path(path)
    atomic_write_json(target, document.to_dict())
    return target


def load_nir(path: str | Path) -> NIRDocument:
    """Load a NIR document from disk."""

    import json

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return NIRDocument.from_dict(payload)


__all__ = [
    "NIRConnection",
    "NIRDocument",
    "NIRPopulation",
    "graph_to_nir",
    "load_nir",
    "nir_to_graph",
    "write_nir",
]
