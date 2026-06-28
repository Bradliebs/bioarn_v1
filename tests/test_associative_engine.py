from __future__ import annotations

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.config import AssociativeMemoryConfig
from bioarn.core.math_utils import cosine_similarity, normalize


def make_engine(*, capacity: int = 4) -> AssociativeMemoryEngine:
    engine = AssociativeMemoryEngine(
        AssociativeMemoryConfig(
            capacity=capacity,
            concept_dim=4,
            input_dim=4,
            top_k_retrieval=min(3, capacity),
            auto_consolidate_interval=0,
            importance_threshold=0.8,
            use_workspace=True,
            use_precision=True,
        )
    )
    with torch.no_grad():
        for ccc in engine.ccc_pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(4))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(4)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.locked.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)
            ccc.importance.zero_()
            ccc.protection.zero_()
    engine._rebuild_sdm()
    return engine


def test_store_and_query() -> None:
    engine = make_engine(capacity=4)
    memory_a = torch.tensor([1.0, 0.9, 0.0, 0.0])
    memory_b = torch.tensor([0.0, 1.0, 0.8, 0.0])
    memory_c = torch.tensor([0.0, 0.0, 1.0, 0.9])

    memory_id_a = engine.store(memory_a, metadata={"label": "alpha"}, importance=0.95)
    engine.store(memory_b, metadata={"label": "beta"}, importance=0.4)
    engine.store(memory_c, metadata={"label": "gamma"}, importance=0.3)

    results = engine.query(torch.tensor([0.95, 0.85, 0.0, 0.0]), top_k=2, threshold=0.2)

    assert memory_id_a == "mem_0000"
    assert len(results) >= 1
    assert results[0].memory_id == memory_id_a
    assert results[0].metadata["label"] == "alpha"
    assert engine.stats["workspace_occupancy"] > 0.0


def test_reconstruct() -> None:
    engine = make_engine(capacity=2)
    content = torch.tensor([1.0, 0.8, 0.0, 0.0])

    memory_id = engine.store(content, metadata={"kind": "target"})
    reconstructed = engine.reconstruct(memory_id)

    assert cosine_similarity(reconstructed, content).item() > 0.99


def test_associate() -> None:
    engine = make_engine(capacity=3)
    memory_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    memory_c = torch.tensor([0.8, 0.2, 0.0, 0.0])

    memory_id_a = engine.store(memory_a, metadata={"label": "anchor"})
    memory_id_b = engine.store(memory_b, metadata={"label": "associate"})
    engine.store(memory_c, metadata={"label": "distractor"})

    before = engine.query(memory_a, top_k=3, threshold=0.2)
    assert all(result.metadata["label"] != "associate" for result in before[1:])

    engine.associate(memory_id_a, memory_id_b, strength=1.0)
    after = engine.query(memory_a, top_k=3, threshold=0.2)

    assert any(result.metadata["label"] == "associate" for result in after[1:])


def test_consolidate() -> None:
    engine = make_engine(capacity=4)
    keep_id = engine.store(torch.tensor([1.0, 0.9, 0.0, 0.0]), importance=0.95)
    engine.store(torch.tensor([0.0, 1.0, 0.8, 0.0]), importance=0.2)
    engine.store(torch.tensor([0.0, 0.0, 1.0, 0.8]), importance=0.1)
    engine.store(torch.tensor([0.7, 0.0, 0.7, 0.0]), importance=0.15)

    consolidated = engine.consolidate()

    keep_index = engine._index_from_memory_id(keep_id)
    assert consolidated > 0
    assert engine.ccc_pool.cccs[keep_index].locked.item() is True
    assert engine.reconstruct(keep_id).shape == (4,)


def test_capacity_management() -> None:
    engine = make_engine(capacity=2)
    engine.store(torch.tensor([1.0, 0.0, 0.0, 0.0]), metadata={"label": "weak-a"}, importance=0.1)
    engine.store(torch.tensor([0.0, 1.0, 0.0, 0.0]), metadata={"label": "weak-b"}, importance=0.2)
    engine.store(torch.tensor([0.0, 0.0, 1.0, 0.0]), metadata={"label": "strong-c"}, importance=0.95)

    results = engine.query(torch.tensor([1.0, 0.0, 0.0, 0.0]), top_k=2, threshold=0.2)

    assert engine.stats["active_memories"] == 2
    assert all(result.metadata.get("label") != "weak-a" for result in results)
    assert any(
        result.metadata.get("label") == "strong-c"
        for result in engine.query(torch.tensor([0.0, 0.0, 1.0, 0.0]), top_k=2, threshold=0.2)
    )


def test_partial_cue_retrieval() -> None:
    engine = make_engine(capacity=2)
    content = torch.tensor([1.0, 1.0, 0.0, 0.0])
    partial_cue = torch.tensor([1.0, 0.0, 0.0, 0.0])

    memory_id = engine.store(content, importance=0.9)
    reconstructed = engine.reconstruct(memory_id, partial_cue=partial_cue)

    assert cosine_similarity(reconstructed, content).item() > 0.99
