"""Small end-to-end demo for the associative memory engine."""

from __future__ import annotations

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.config import AssociativeMemoryConfig
from bioarn.core.math_utils import cosine_similarity, normalize


def configure_identity_encoder(engine: AssociativeMemoryEngine) -> None:
    with torch.no_grad():
        for ccc in engine.ccc_pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(engine.config.input_dim))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(engine.config.concept_dim)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.locked.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)
            ccc.importance.zero_()
            ccc.protection.zero_()
    engine._rebuild_sdm()


def main() -> None:
    engine = AssociativeMemoryEngine(
        AssociativeMemoryConfig(
            capacity=8,
            concept_dim=8,
            input_dim=8,
            top_k_retrieval=3,
            auto_consolidate_interval=0,
        )
    )
    configure_identity_encoder(engine)

    beach = torch.tensor([1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mountain = torch.tensor([0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    code = torch.tensor([0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0])

    beach_id = engine.store(beach, metadata={"label": "beach"}, importance=0.95)
    mountain_id = engine.store(mountain, metadata={"label": "mountain"}, importance=0.75)
    engine.store(code, metadata={"label": "code"}, importance=0.6)

    print("Stored memories:", engine.stats["active_memory_ids"])

    query = torch.tensor([0.95, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = engine.query(query, top_k=2, threshold=0.2)
    print("\nQuery results")
    for result in results:
        print(
            f"- {result.memory_id}: {result.metadata.get('label')} "
            f"(confidence={result.confidence:.3f}, importance={result.importance:.2f})"
        )

    reconstructed = engine.reconstruct(beach_id)
    print(
        "\nReconstruction similarity:",
        f"{cosine_similarity(reconstructed, beach).item():.3f}",
    )

    engine.associate(beach_id, mountain_id, strength=1.0)
    associated = engine.query(beach, top_k=3, threshold=0.2)
    print("\nAfter linking beach <-> mountain")
    for result in associated:
        print(f"- {result.memory_id}: {result.metadata.get('label')} ({result.confidence:.3f})")

    partial = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    partial_recall = engine.reconstruct(beach_id, partial_cue=partial)
    print(
        "\nPartial-cue similarity:",
        f"{cosine_similarity(partial_recall, beach).item():.3f}",
    )


if __name__ == "__main__":
    main()
