"""End-to-end demo for the Bio-ARN associative memory REST API."""

from __future__ import annotations

import threading

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.api.client import BioARNMemoryClient
from bioarn.api.memory_server import create_memory_server
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

    server = create_memory_server(host="127.0.0.1", port=0, engine=engine)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        host, port = server.server_address[:2]
        client = BioARNMemoryClient(f"http://{host}:{port}")

        beach = [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mountain = [0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        code = [0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0]

        beach_id = client.store(beach, metadata={"label": "beach", "text": "sunny beach"}, importance=0.95)
        mountain_id = client.store(
            mountain,
            metadata={"label": "mountain", "text": "snowy mountain"},
            importance=0.75,
        )
        client.store(code, metadata={"label": "code", "text": "coding session"}, importance=0.6)

        print("Health:", client.health())
        print("Stored memories:", client.stats()["active_memory_ids"])

        results = client.query([0.95, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=2, threshold=0.2)
        print("\nQuery results")
        for result in results:
            print(
                f"- {result['memory_id']}: {result['metadata'].get('label')} "
                f"(confidence={result['confidence']:.3f}, importance={result['importance']:.2f})"
            )

        recalled = client.recall(beach_id)
        similarity = cosine_similarity(torch.tensor(recalled["content"]), torch.tensor(beach)).item()
        print(f"\nRecall similarity for {beach_id}: {similarity:.3f}")

        client.associate(beach_id, mountain_id, strength=1.0)
        associated = client.query(beach, top_k=3, threshold=0.2)
        print("\nAfter linking beach <-> mountain")
        for result in associated:
            print(f"- {result['memory_id']}: {result['metadata'].get('label')} ({result['confidence']:.3f})")

        print("\nFinal stats:", client.stats())
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
