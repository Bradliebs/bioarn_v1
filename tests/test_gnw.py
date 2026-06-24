from __future__ import annotations

import torch

from bioarn.config import GNWConfig
from bioarn.core.math_utils import normalize
from bioarn.workspace.gnw import GlobalNeuronalWorkspace, StreamOfConsciousness


def make_config(**overrides: float | int) -> GNWConfig:
    config = GNWConfig(
        capacity=3,
        broadcast_gain=2.0,
        fatigue_rate=0.2,
        fatigue_threshold=0.35,
        competition_temp=0.5,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def concept(index: int, dim: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=dim).float()


def test_gnw_empty_initial() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())

    broadcast = gnw.broadcast()

    assert gnw.slots == []
    assert gnw.is_full() is False
    assert broadcast.num_occupied == 0
    assert broadcast.total_broadcast_energy == 0.0
    assert gnw.get_stream() == []
    assert gnw.get_stats()["occupancy"] == 0.0


def test_gnw_add_single() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())

    new_entries, evicted = gnw.update([(0, concept(0), 0.9)], timestep=1)

    assert new_entries == [0]
    assert evicted == []
    assert len(gnw.slots) == 1
    assert gnw.slots[0].ccc_index == 0
    assert gnw.slots[0].activation == 1.8
    assert gnw.slots[0].age == 0


def test_gnw_capacity_limit() -> None:
    gnw = GlobalNeuronalWorkspace(make_config(capacity=2))

    gnw.update(
        [
            (0, concept(0), 0.95),
            (1, concept(1), 0.85),
            (2, concept(2), 0.30),
        ],
        timestep=1,
    )

    indices = [slot.ccc_index for slot in gnw.slots]
    assert len(indices) == 2
    assert set(indices) == {0, 1}


def test_gnw_competition() -> None:
    gnw = GlobalNeuronalWorkspace(make_config(capacity=2))

    winners = gnw.compete(
        [
            (0, concept(0), 0.80),
            (1, concept(1), 0.95),
            (2, concept(2), 0.10),
        ]
    )

    assert len(winners) == 2
    assert set(winners) == {0, 1}
    assert 2 not in winners


def test_gnw_fatigue_eviction() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())
    gnw.update([(0, concept(0), 0.5)], timestep=1)

    gnw.update([], timestep=2)
    gnw.update([], timestep=3)
    _, evicted = gnw.update([], timestep=4)

    assert 0 in evicted
    assert gnw.slots == []


def test_gnw_broadcast_amplification() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())
    gnw.update([(0, concept(0), 0.4)], timestep=1)

    broadcast = gnw.broadcast()

    assert broadcast.indices == [0]
    assert broadcast.activations[0] > 0.4
    assert broadcast.total_broadcast_energy == broadcast.activations[0]


def test_gnw_stream_ordering() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())

    for ccc_index in [0, 1, 2]:
        gnw.inject(ccc_index, concept(ccc_index), priority=1.0)
        gnw.broadcast()
        gnw.clear()

    stream = gnw.get_stream(last_n=3)

    assert [slot.ccc_index for slot in stream] == [0, 1, 2]


def test_gnw_inject_priority() -> None:
    gnw = GlobalNeuronalWorkspace(make_config(capacity=2))
    gnw.inject(0, concept(0), priority=0.4)
    gnw.inject(1, concept(1), priority=0.3)

    gnw.inject(2, concept(2), priority=1.0)

    indices = [slot.ccc_index for slot in gnw.slots]
    assert 2 in indices
    assert 1 not in indices
    assert gnw.last_evicted == [1]


def test_gnw_attention_relevance() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())
    gnw.inject(0, concept(0), priority=1.0)
    gnw.inject(1, concept(1), priority=1.0)

    query = normalize(torch.tensor([0.9, 0.1, 0.0, 0.0]).unsqueeze(0)).squeeze(0)
    attention = gnw.attend(query)

    assert gnw.slots[attention.best_match_index].ccc_index == 0
    assert torch.isclose(attention.attention_weights.sum(), torch.tensor(1.0))
    assert attention.relevance_score > 0.9


def test_gnw_thought_chain() -> None:
    gnw = GlobalNeuronalWorkspace(make_config())
    stream = StreamOfConsciousness(gnw, make_config())

    stream.think_step([(0, concept(0), 0.90)], timestep=1)
    stream.think_step([(1, concept(1), 0.95)], timestep=2)
    output = stream.think_step([(2, concept(2), 0.99)], timestep=3)
    chain = stream.get_thought_chain(n=3)

    assert output.thought_chain_length == 3
    assert len(chain) == 3
    assert torch.allclose(chain[0], concept(0))
    assert torch.allclose(chain[1], concept(1))
    assert torch.allclose(chain[2], concept(2))


def test_gnw_rumination_detection() -> None:
    gnw = GlobalNeuronalWorkspace(make_config(capacity=1))
    stream = StreamOfConsciousness(gnw, make_config(capacity=1))

    for timestep in range(1, 5):
        output = stream.think_step([(0, concept(0), 0.95)], timestep=timestep)

    assert output.is_ruminating is True
    assert stream.detect_rumination() is True


def test_gnw_turnover() -> None:
    gnw = GlobalNeuronalWorkspace(make_config(capacity=2))
    gnw.update([(0, concept(0), 0.50), (1, concept(1), 0.45)], timestep=1)
    gnw.update([], timestep=2)
    gnw.update([], timestep=3)

    _, evicted = gnw.update([(2, concept(2), 0.95), (3, concept(3), 0.90)], timestep=4)

    indices = [slot.ccc_index for slot in gnw.slots]
    assert set(indices) == {2, 3}
    assert set(evicted) == {0, 1}
    assert gnw.get_stats()["turnover_rate"] > 0.0
