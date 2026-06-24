import pytest
import torch

from bioarn.config import CCCConfig, MarginGateConfig, SDMConfig
from bioarn.core.ccc import CCCPool, ConceptCellCluster
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.associative_fabric import AssociativeFabric, FabricConnectedPool


def make_fabric(
    *,
    address_dim: int = 4,
    concept_dim: int = 4,
    hamming_radius: int = 0,
    num_hard_locations: int = 16,
    decay_rate: float = 1.0,
    stdp_window: int = 5,
) -> AssociativeFabric:
    sdm_config = SDMConfig(
        address_dim=address_dim,
        hamming_radius=hamming_radius,
        num_hard_locations=num_hard_locations,
        data_dim=concept_dim,
        decay_rate=decay_rate,
        stdp_window=stdp_window,
    )
    ccc_config = CCCConfig(
        input_dim=4,
        concept_dim=concept_dim,
        num_f1_features=4,
        f1_top_k=2,
        max_pool_size=4,
    )
    return AssociativeFabric(sdm_config, ccc_config)


def set_locations_for_directions(
    fabric: AssociativeFabric,
    directions: list[torch.Tensor],
) -> None:
    addresses = fabric.sdm.compute_address(torch.stack(directions))
    unique_addresses = torch.unique(addresses, dim=0)
    hard_locations = torch.zeros(
        fabric.sdm.num_hard_locations,
        fabric.sdm.address_dim,
        dtype=torch.float32,
    )
    hard_locations[: unique_addresses.shape[0]] = unique_addresses.to(torch.float32)
    fabric.sdm.hard_locations.copy_(hard_locations)
    fabric.sdm.data_matrix.zero_()
    fabric.sdm.activation_counts.zero_()


def concept(bits: list[int]) -> torch.Tensor:
    values = torch.tensor([1.0 if bit else -1.0 for bit in bits], dtype=torch.float32)
    return normalize(values.unsqueeze(0)).squeeze(0)


def configure_identity_ccc(ccc: ConceptCellCluster) -> None:
    with torch.no_grad():
        ccc.f1_layer.weight.copy_(torch.eye(4))
        ccc.f1_layer.bias.zero_()
        ccc.f2_weights.copy_(normalize(torch.eye(4)))
        ccc.feedback_weights.zero_()
        ccc.concept_direction.zero_()
        ccc.is_committed.zero_()
        ccc.age.zero_()
        ccc.last_fired.fill_(-1)


def make_pool() -> CCCPool:
    config = CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=2,
        max_pool_size=3,
    )
    margin = MarginGateConfig(theta_margin=0.5, theta_margin_lr=0.01, theta_resonance=0.95)
    pool = CCCPool(config, margin)
    for ccc in pool.cccs:
        configure_identity_ccc(ccc)
    return pool


def test_register_activation() -> None:
    fabric = make_fabric()
    direction = concept([1, 0, 0, 0])
    set_locations_for_directions(fabric, [direction])

    fabric.register_activation(0, direction, confidence=1.0, timestep=0)

    retrieved = fabric.sdm.read(direction)
    assert len(fabric.activation_history) == 1
    assert fabric.sdm.activation_counts.sum().item() == 1.0
    assert torch.allclose(retrieved, direction)


def test_co_activation_forms_association() -> None:
    fabric = make_fabric()
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.register_activation(1, direction_b, confidence=0.8, timestep=0)
    fabric.form_associations(timestep=0)

    assert fabric.association_strength[(0, 1)] == pytest.approx(0.9)
    assert fabric.association_strength[(1, 0)] == pytest.approx(0.9)
    assert fabric.association_temporal[(0, 1)] is False


def test_temporal_stdp() -> None:
    fabric = make_fabric(stdp_window=2)
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.register_activation(1, direction_b, confidence=1.0, timestep=1)
    fabric.form_associations(timestep=1)

    assert fabric.association_strength[(0, 1)] > fabric.association_strength[(1, 0)]
    assert fabric.association_temporal[(0, 1)] is True
    assert fabric.association_temporal[(1, 0)] is False


def test_retrieve_associates() -> None:
    fabric = make_fabric(stdp_window=2)
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.register_activation(1, direction_b, confidence=0.9, timestep=1)
    fabric.form_associations(timestep=1)

    result = fabric.retrieve_associates(direction_a, k=1)
    assert result.indices == [1]
    assert result.temporal_order == [True]
    assert cosine_similarity(result.directions[0], direction_b).item() > 0.99


def test_sequence_retrieval() -> None:
    fabric = make_fabric(stdp_window=1)
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    direction_c = concept([0, 0, 1, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b, direction_c])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.form_associations(timestep=0)
    fabric.register_activation(1, direction_b, confidence=1.0, timestep=1)
    fabric.form_associations(timestep=1)
    fabric.register_activation(2, direction_c, confidence=1.0, timestep=2)
    fabric.form_associations(timestep=2)

    sequence = fabric.retrieve_sequence(direction_a, steps=2)
    assert len(sequence) == 2
    assert cosine_similarity(sequence[0], direction_b).item() > 0.99
    assert cosine_similarity(sequence[1], direction_c).item() > 0.99


def test_lateral_inhibition() -> None:
    fabric = make_fabric(hamming_radius=0)
    strong = concept([1, 1, 0, 0])
    weak = normalize((strong + torch.tensor([0.1, 0.1, 0.0, 0.0])).unsqueeze(0)).squeeze(0)
    distinct = concept([0, 0, 1, 1])

    winners = fabric.lateral_inhibition(
        [
            (0, strong, 0.9),
            (1, weak, 0.6),
            (2, distinct, 0.8),
        ],
        k=2,
    )

    assert winners == [(0, pytest.approx(0.9)), (2, pytest.approx(0.8))]


def test_voting_consensus() -> None:
    fabric = make_fabric()
    prototype = concept([1, 0, 0, 0])
    agreeing = [
        (0, prototype, 0.9),
        (1, normalize((prototype + 0.05 * concept([0, 1, 0, 0])).unsqueeze(0)).squeeze(0), 0.85),
        (2, normalize((prototype + 0.05 * concept([0, 0, 1, 0])).unsqueeze(0)).squeeze(0), 0.8),
    ]

    result = fabric.vote(agreeing)

    assert result.voter_count == 3
    assert result.agreement_score > 0.95
    assert result.confidence > 0.75
    assert cosine_similarity(result.winning_direction, prototype).item() > 0.99


def test_voting_disagreement() -> None:
    fabric = make_fabric()
    disagreeing = [
        (0, concept([1, 0, 0, 0]), 0.9),
        (1, concept([0, 1, 0, 0]), 0.9),
        (2, concept([0, 0, 1, 0]), 0.9),
    ]

    result = fabric.vote(disagreeing)

    assert result.voter_count == 1
    assert result.agreement_score < 0.5
    assert result.confidence < 0.5


def test_fabric_connected_pool() -> None:
    pool = make_pool()
    fabric = make_fabric(hamming_radius=0)
    connected = FabricConnectedPool(pool, fabric)

    direction_a = normalize(torch.tensor([1.0, 0.9, 0.0, 0.0]).unsqueeze(0)).squeeze(0)
    direction_b = normalize(torch.tensor([1.0, 0.7, 0.1, 0.0]).unsqueeze(0)).squeeze(0)
    set_locations_for_directions(fabric, [direction_a, direction_b])

    with torch.no_grad():
        pool.cccs[0].is_committed.fill_(True)
        pool.cccs[0].concept_direction.copy_(direction_a)
        pool.cccs[1].is_committed.fill_(True)
        pool.cccs[1].concept_direction.copy_(direction_b)

    output = connected(torch.tensor([1.0, 0.9, 0.0, 0.0]), timestep=0)

    assert set(output.pool_output.fired_indices) == {0, 1}
    assert len(output.active_cccs) == 2
    assert output.consensus.voter_count == 2
    assert fabric.get_stats()["num_associations"] >= 2


def test_predict_next() -> None:
    pool = make_pool()
    fabric = make_fabric(stdp_window=1)
    connected = FabricConnectedPool(pool, fabric)
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.form_associations(timestep=0)
    fabric.register_activation(1, direction_b, confidence=1.0, timestep=1)
    fabric.form_associations(timestep=1)

    prediction = connected.predict_next(direction_a)
    assert cosine_similarity(prediction, direction_b).item() > 0.99


def test_sparse_storage() -> None:
    fabric = make_fabric(stdp_window=1)
    directions = [concept([int(bit) for bit in f"{value:04b}"]) for value in range(10)]
    set_locations_for_directions(fabric, directions)

    for timestep, direction in enumerate(directions):
        fabric.register_activation(timestep, direction, confidence=1.0, timestep=timestep)
        fabric.form_associations(timestep=timestep)

    assert len(fabric.association_strength) <= 2 * (len(directions) - 1)
    assert len(fabric.association_strength) < len(directions) ** 2 / 2


def test_decay_weakens_old() -> None:
    fabric = make_fabric(decay_rate=0.5, stdp_window=1)
    direction_a = concept([1, 0, 0, 0])
    direction_b = concept([0, 1, 0, 0])
    set_locations_for_directions(fabric, [direction_a, direction_b])

    fabric.register_activation(0, direction_a, confidence=1.0, timestep=0)
    fabric.form_associations(timestep=0)
    fabric.register_activation(1, direction_b, confidence=1.0, timestep=1)
    fabric.form_associations(timestep=1)
    initial = fabric.association_strength[(0, 1)]

    fabric.form_associations(timestep=5)

    assert fabric.association_strength[(0, 1)] == pytest.approx(initial * (0.5 ** 4))
