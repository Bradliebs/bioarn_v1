from __future__ import annotations

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.api.langchain_memory import BioARNMemory
from bioarn.api.memory_server import MemoryAPI
from bioarn.config import AssociativeMemoryConfig
from bioarn.core.math_utils import cosine_similarity, normalize


def make_engine(*, capacity: int = 4, dimension: int = 4) -> AssociativeMemoryEngine:
    engine = AssociativeMemoryEngine(
        AssociativeMemoryConfig(
            capacity=capacity,
            concept_dim=dimension,
            input_dim=dimension,
            top_k_retrieval=min(3, capacity),
            auto_consolidate_interval=0,
            importance_threshold=0.8,
            use_workspace=True,
            use_precision=True,
        )
    )
    with torch.no_grad():
        for ccc in engine.ccc_pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(dimension))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(dimension)))
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


def test_memory_api_store_query_and_recall_roundtrip() -> None:
    api = MemoryAPI(make_engine(capacity=4, dimension=4))
    beach = torch.tensor([1.0, 0.9, 0.0, 0.0])

    stored = api.store(beach.tolist(), metadata={"label": "beach", "text": "sunny beach"}, importance=0.95)
    results = api.query([0.95, 0.85, 0.0, 0.0], top_k=2, threshold=0.2)
    recalled = api.recall(stored["memory_id"])
    stats = api.stats()

    assert stored["memory_id"] == "mem_0000"
    assert results["results"][0]["memory_id"] == stored["memory_id"]
    assert recalled["metadata"]["label"] == "beach"
    assert cosine_similarity(torch.tensor(recalled["content"]), beach).item() > 0.99
    assert stats["stored"] == 1
    assert stats["locked"] == 1


def test_memory_api_associate_forget_and_consolidate() -> None:
    api = MemoryAPI(make_engine(capacity=3, dimension=4))
    anchor = torch.tensor([1.0, 0.0, 0.0, 0.0])
    associate = torch.tensor([0.0, 1.0, 0.0, 0.0])
    distractor = torch.tensor([0.8, 0.2, 0.0, 0.0])

    anchor_id = api.store(anchor.tolist(), metadata={"label": "anchor"})["memory_id"]
    associate_id = api.store(associate.tolist(), metadata={"label": "associate"})["memory_id"]
    distractor_id = api.store(distractor.tolist(), metadata={"label": "distractor"}, importance=0.2)["memory_id"]

    before = api.query(anchor.tolist(), top_k=3, threshold=0.2)["results"]
    api.associate(anchor_id, associate_id, strength=1.0)
    after = api.query(anchor.tolist(), top_k=3, threshold=0.2)["results"]
    forgotten = api.forget(distractor_id)
    consolidated = api.consolidate()

    assert all(result["metadata"]["label"] != "associate" for result in before[1:])
    assert any(result["metadata"]["label"] == "associate" for result in after[1:])
    assert forgotten["forgotten"] is True
    assert api.stats()["stored"] == 2
    assert consolidated["status"] == "consolidated"


def test_langchain_memory_local_engine_roundtrip() -> None:
    memory = BioARNMemory(
        engine=make_engine(capacity=6, dimension=32),
        input_key="input",
        output_key="output",
        top_k=1,
        threshold=0.0,
    )

    memory.save_context({"input": "zebra horizon"}, {"output": "We discussed striped animals."})
    memory.save_context({"input": "quantum lattice"}, {"output": "We discussed condensed matter."})

    loaded = memory.load_memory_variables({"input": "zebra horizon"})

    assert "striped animals" in loaded["history"]
    assert "condensed matter" not in loaded["history"]

    memory.clear()

    assert memory.load_memory_variables({"input": "zebra horizon"}) == {"history": ""}
    assert memory.engine is not None
    assert memory.engine.stats["active_memories"] == 0
