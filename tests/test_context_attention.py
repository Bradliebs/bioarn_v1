from __future__ import annotations

import torch

from bioarn.workspace.context_buffer import ContextBuffer
from bioarn.workspace.recurrent_context import RecurrentContext
from bioarn.workspace.selective_attention import SpikeAttention


def concept(index: int, dim: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=dim).float()


def test_context_buffer_init() -> None:
    buffer = ContextBuffer(buffer_size=32, context_dim=8, decay=0.9)

    assert buffer.buffer_size == 32
    assert buffer.context_dim == 8
    assert buffer.get_context_vector().shape == (8,)


def test_context_buffer_update() -> None:
    buffer = ContextBuffer(buffer_size=8, context_dim=4, decay=0.95)

    buffer.update(concept(0), strength=1.0)
    buffer.update(concept(1), strength=0.8)

    assert len(buffer.items) == 2


def test_context_buffer_decay() -> None:
    buffer = ContextBuffer(buffer_size=8, context_dim=4, decay=0.5, eviction_threshold=0.01)
    buffer.update(concept(0), strength=1.0)

    initial_strength = float(buffer.items[0].strength)
    buffer.update(concept(1), strength=0.2)
    decayed_item = max(
        buffer.items,
        key=lambda item: float(torch.nn.functional.cosine_similarity(item.concept.unsqueeze(0), concept(0).unsqueeze(0)).item()),
    )

    assert decayed_item.strength < initial_strength


def test_context_vector_changes() -> None:
    buffer = ContextBuffer(buffer_size=8, context_dim=4, decay=0.95)
    buffer.update(concept(0), strength=1.0)
    before = buffer.get_context_vector()

    buffer.update(concept(1), strength=1.0)
    after = buffer.get_context_vector()

    assert not torch.allclose(before, after)


def test_context_attend_retrieves_similar() -> None:
    buffer = ContextBuffer(buffer_size=8, context_dim=4, decay=0.95)
    buffer.update(concept(0), strength=1.0)
    buffer.update(concept(1), strength=0.9)

    query = torch.tensor([0.05, 0.95, 0.0, 0.0], dtype=torch.float32)
    attended = buffer.attend(query, top_k=2)

    best_concept, best_score = attended[0]
    assert best_score >= attended[1][1]
    assert torch.argmax(best_concept).item() == 1


def test_recurrent_integration() -> None:
    recurrent = RecurrentContext(context_dim=4, integration_rate=0.25)

    recurrent.integrate(concept(0))
    integrated = recurrent.integrate(concept(1))

    assert integrated.shape == (4,)
    assert integrated[0] > 0.0
    assert integrated[1] > 0.0


def test_context_priming_biases() -> None:
    recurrent = RecurrentContext(context_dim=4, integration_rate=0.2)
    context_vector = concept(0)
    candidates = torch.stack([concept(0), concept(1)], dim=0)

    biases = recurrent.prime_retrieval(context_vector, candidates)

    assert biases[0] > biases[1]


def test_repetition_detection() -> None:
    recurrent = RecurrentContext(context_dim=4, integration_rate=0.2)
    history = [concept(0), concept(1), concept(0), concept(1), concept(0), concept(1)]

    score = recurrent.detect_repetition(history, window=6)

    assert score > 0.8


def test_spike_attention_selects() -> None:
    attention = SpikeAttention(dim=4, num_heads=1)
    query = torch.tensor([0.9, 0.1, 0.0, 0.0], dtype=torch.float32)
    keys = [concept(0), concept(1)]

    attended, weights = attention.attend(query, keys)

    assert weights[0] > weights[1]
    assert torch.argmax(attended).item() == 0


def test_multi_head_diversity() -> None:
    attention = SpikeAttention(dim=4, num_heads=2)
    query = torch.ones(4, dtype=torch.float32)
    keys = [concept(0), concept(1), concept(2), concept(3)]

    _, weights = attention.attend(query, keys)

    assert len(set(attention.last_head_winners)) >= 2
    assert sum(weight > 0.0 for weight in weights) >= 2


def test_topic_drift_detection() -> None:
    buffer = ContextBuffer(buffer_size=8, context_dim=4, decay=0.95)
    buffer.update(concept(0), strength=1.0)
    buffer.update(concept(0), strength=0.9)
    low_drift = buffer.get_topic_drift()

    buffer.update(concept(3), strength=1.0)
    buffer.update(concept(3), strength=0.9)
    high_drift = buffer.get_topic_drift()

    assert high_drift > low_drift


def test_no_backprop_attention() -> None:
    attention = SpikeAttention(dim=4, num_heads=2)
    query = concept(0).clone().requires_grad_(True)
    keys = [concept(0).clone().requires_grad_(True), concept(1).clone().requires_grad_(True)]

    attended, _ = attention.attend(query, keys)

    assert attended.requires_grad is False
    assert all(key.grad is None for key in keys)
