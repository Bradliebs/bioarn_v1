import pytest
import torch

from bioarn.config import SDMConfig
from bioarn.memory.sdm import SparseDistributedMemory, TemporalAssociator


def make_sdm(
    *,
    address_dim: int = 4,
    data_dim: int = 4,
    hamming_radius: int = 0,
    num_hard_locations: int | None = None,
    decay_rate: float = 1.0,
    stdp_window: int = 5,
) -> SparseDistributedMemory:
    num_hard_locations = num_hard_locations or address_dim
    config = SDMConfig(
        address_dim=address_dim,
        hamming_radius=hamming_radius,
        num_hard_locations=num_hard_locations,
        data_dim=data_dim,
        decay_rate=decay_rate,
        stdp_window=stdp_window,
    )
    return SparseDistributedMemory(config)


def set_hard_locations(sdm: SparseDistributedMemory, locations: torch.Tensor) -> None:
    sdm.hard_locations.copy_(locations.to(torch.float32))
    sdm.data_matrix.zero_()
    sdm.activation_counts.zero_()


def test_address_computation():
    sdm = make_sdm(address_dim=4, data_dim=4)
    concept = torch.tensor([[1.2, -0.3, 0.0, 4.1], [-1.0, 2.0, -5.0, 0.2]])

    address = sdm.compute_address(concept)

    expected = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    assert torch.equal(address, expected)


def test_address_deterministic():
    torch.manual_seed(0)
    sdm = make_sdm(address_dim=8, data_dim=3, num_hard_locations=8)
    concept = torch.tensor([0.5, -1.5, 2.0])

    address_a = sdm.compute_address(concept)
    address_b = sdm.compute_address(concept)

    assert address_a.shape == (8,)
    assert torch.equal(address_a, address_b)


def test_write_read_roundtrip():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=0)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    address = locations[0]
    data = torch.tensor([0.2, 0.8, -0.4])

    sdm.write(address, data)
    retrieved = sdm.read(address)

    assert torch.allclose(retrieved, data)


def test_partial_cue_retrieval():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=1)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    address = locations[0]
    noisy_cue = torch.tensor([1.0, 1.0, 0.0, 0.0])
    data = torch.tensor([1.0, -0.5, 0.25])

    sdm.write(address, data)
    retrieved = sdm.retrieve_associates(noisy_cue)

    assert torch.allclose(retrieved, data)


def test_multiple_patterns():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=0)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    addresses = locations[:3]
    data = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sdm.write(addresses, data)

    for idx in range(3):
        retrieved = sdm.read(addresses[idx])
        assert torch.allclose(retrieved, data[idx])


def test_association_formation():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=0)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    address_a = locations[0]
    address_b = locations[1]
    data_a = torch.tensor([1.0, 0.0, 0.0])
    data_b = torch.tensor([0.0, 1.0, 0.0])

    sdm.associate(address_a, address_b, data_a, data_b)

    retrieved_from_a = sdm.read(address_a)
    retrieved_from_b = sdm.read(address_b)

    assert torch.allclose(retrieved_from_a, data_b * 2.0)
    assert torch.allclose(retrieved_from_b, data_a)


def test_temporal_stdp_ordering():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=0, stdp_window=10)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)
    associator = TemporalAssociator(sdm, sdm.config)

    address_a = locations[0]
    address_b = locations[1]
    data_a = torch.tensor([1.0, 0.0, 0.0])
    data_b = torch.tensor([0.0, 1.0, 0.0])

    associator.record_activation(address_a, data_a, timestamp=0.0)
    associator.record_activation(address_b, data_b, timestamp=2.0)
    associator.form_associations()

    retrieved_from_a = sdm.read(address_a)
    retrieved_from_b = sdm.read(address_b)

    assert retrieved_from_a.norm().item() == pytest.approx(2.0 * retrieved_from_b.norm().item())


def test_lateral_inhibition():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=1)
    activations = torch.tensor([0.9, 0.7, 0.8])
    addresses = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    inhibited = sdm.inhibit(activations, addresses, k=2)

    expected = torch.tensor([0.9, 0.0, 0.8])
    assert torch.allclose(inhibited, expected)


def test_decay():
    sdm = make_sdm(address_dim=2, data_dim=3, hamming_radius=0, decay_rate=0.5)
    locations = torch.eye(2)
    set_hard_locations(sdm, locations)

    address_a = locations[0]
    address_b = locations[1]
    data = torch.tensor([1.0, 2.0, 3.0])

    sdm.write(address_a, data)
    for _ in range(4):
        sdm.write(address_b, torch.zeros(3))

    retrieved = sdm.read(address_a)
    expected = data * (0.5 ** 5)
    assert torch.allclose(retrieved, expected)


def test_capacity():
    sdm = make_sdm(address_dim=8, data_dim=8, hamming_radius=0, num_hard_locations=8)
    locations = torch.eye(8)
    set_hard_locations(sdm, locations)
    data = torch.eye(8)

    sdm.write(locations, data)
    retrieved = sdm.read(locations)

    assert torch.allclose(retrieved, data)


def test_batch_operations():
    sdm = make_sdm(address_dim=4, data_dim=4, hamming_radius=0)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    addresses = locations[:3]
    data = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )

    sdm.write(addresses, data)
    retrieved = sdm.read(addresses)

    assert torch.allclose(retrieved, data)


def test_stats():
    sdm = make_sdm(address_dim=4, data_dim=3, hamming_radius=0)
    locations = torch.eye(4)
    set_hard_locations(sdm, locations)

    sdm.write(locations[0], torch.tensor([1.0, 0.0, 0.0]))
    sdm.write(locations[1], torch.tensor([0.0, 1.0, 0.0]))

    stats = sdm.get_stats()

    assert stats["num_stored"] == 2
    assert stats["mean_activation_count"] == pytest.approx(0.5)
    assert stats["sparsity"] == pytest.approx(0.5)
    assert stats["capacity_used"] == pytest.approx(0.5)
