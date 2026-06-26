"""Tests for the Bio-ARN neuromorphic export path."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.ccc import CCCPool
from bioarn.export import (
    Loihi2Config,
    NIRDocument,
    NeuromorphicGraph,
    export_ccc_pool,
    export_hierarchy,
    nir_to_graph,
)
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy


def make_export_pool() -> CCCPool:
    pool = CCCPool(
        CCCConfig(
            input_dim=8,
            concept_dim=4,
            num_f1_features=6,
            f1_top_k=3,
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=4,
        ),
        MarginGateConfig(theta_margin=0.2, theta_margin_lr=0.01, theta_resonance=0.45),
    )
    prototypes = [
        torch.tensor([1.0, 0.8, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.1, 0.9, 1.0, 0.0, 0.0, 0.2, 0.1], dtype=torch.float32),
    ]
    with torch.no_grad():
        for index, prototype in enumerate(prototypes):
            f1_output = pool.cccs[index].f1_encode(prototype)
            pool.cccs[index].learn_fast(prototype, f1_output)
    return pool


def make_structured_image(label: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(3, 32, 32, generator=generator) * 0.04
    image[label % 3, 4 + label : 20 + label, :] += 0.5
    image[(label + 1) % 3, :, 6 * (label + 1) : 6 * (label + 1) + 6] += 0.35
    image[(label + 2) % 3, 10:22, 10:22] += 0.08 * (label + 1)
    return image


def make_trained_hierarchy() -> VisualHierarchy:
    hierarchy = VisualHierarchy(
        HierarchyConfig(
            pool_sizes=[12, 10, 10, 6],
            concept_dims=[8, 10, 12, 6],
            thresholds=[0.18, 0.24, 0.28, 0.3],
            learning_rates=[0.05, 0.04, 0.03, 0.02],
        )
    )
    for index in range(12):
        label = index % 3
        hierarchy.learn(make_structured_image(label, 100 + index), label=label)
    return hierarchy


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_ccc_pool_export_produces_valid_structure(tmp_path: Path) -> None:
    graph = export_ccc_pool(make_export_pool(), tmp_path)

    graph.validate()

    payload = read_json(tmp_path / "ccc_pool.loihi2.json")
    assert payload["format"] == "bioarn.loihi2.export"
    assert payload["graph"]["metadata"]["committed_count"] == 2
    assert any(population["id"] == "ccc_pool_wta" for population in payload["graph"]["populations"])
    assert any(
        projection["metadata"].get("role") == "concept_readout"
        for projection in payload["graph"]["projections"]
    )


def test_hierarchy_export_maps_layers_correctly(tmp_path: Path) -> None:
    graph = export_hierarchy(make_trained_hierarchy(), tmp_path)

    graph.validate()

    payload = read_json(tmp_path / "visual_hierarchy.loihi2.json")
    assert payload["graph"]["metadata"]["layer_order"] == ["V1", "V2", "V4", "IT"]
    for layer_name in ("v1", "v2", "v4", "it"):
        assert any(
            population["id"] == f"{layer_name}_output"
            for population in payload["graph"]["populations"]
        )
    feedforward = [
        projection
        for projection in payload["graph"]["projections"]
        if projection["metadata"].get("role") == "feedforward"
    ]
    assert [(projection["source"], projection["target"]) for projection in feedforward] == [
        ("v1_output", "v2_input"),
        ("v2_output", "v4_input"),
        ("v4_output", "it_input"),
    ]


def test_export_round_trip_via_nir_preserves_structure(tmp_path: Path) -> None:
    export_path = tmp_path / "roundtrip.json"
    graph = export_ccc_pool(make_export_pool(), export_path)

    payload = read_json(export_path)
    restored_graph = NeuromorphicGraph.from_dict(payload["graph"])
    nir_document = NIRDocument.from_dict(payload["nir"])
    nir_graph = nir_to_graph(nir_document)

    restored_graph.validate()
    nir_graph.validate()

    assert {population.id for population in restored_graph.populations} == {
        population.id for population in graph.populations
    }
    assert {projection.id for projection in nir_graph.projections} == {
        projection.id for projection in graph.projections
    }
    assert len(nir_graph.populations) == len(graph.populations)


def test_loihi2_config_defaults_are_sensible() -> None:
    config = Loihi2Config()
    allocation = config.allocate_population(300, start_core=2)

    assert config.cores_per_chip == 128
    assert config.neurons_per_core == 128
    assert config.weight_bits == 8
    assert config.spike_threshold > 0.0
    assert config.winner_take_all_inhibition < 0.0
    assert allocation.core_ids == [2, 3, 4]
    assert allocation.neurons_per_core == [128, 128, 44]
