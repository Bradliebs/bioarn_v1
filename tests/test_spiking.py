"""Tests for Bio-ARN spiking neuron primitives."""

import torch

from bioarn.config import SpikingConfig
from bioarn.core.spiking import (
    LIFLayer,
    LIFNeuron,
    SurrogateSpike,
    delta_encode,
    latency_encode,
    rate_encode,
)


def test_lif_neuron_basic() -> None:
    neuron = LIFNeuron(
        num_neurons=1,
        config=SpikingConfig(beta=1.0, threshold=1.0, reset=0.0, refractory_steps=0),
    )

    spikes = []
    voltages = []
    for _ in range(4):
        spike, voltage = neuron(torch.tensor([[0.6]]))
        spikes.append(spike.item())
        voltages.append(voltage.item())

    assert spikes == [0.0, 1.0, 0.0, 1.0]
    assert voltages[1] == 0.0
    assert voltages[3] == 0.0


def test_lif_refractory() -> None:
    neuron = LIFNeuron(
        num_neurons=1,
        config=SpikingConfig(beta=1.0, threshold=1.0, reset=0.0, refractory_steps=2),
    )

    spikes = []
    for _ in range(5):
        spike, _ = neuron(torch.tensor([[1.1]]))
        spikes.append(spike.item())

    assert spikes == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_lif_decay() -> None:
    neuron = LIFNeuron(
        num_neurons=1,
        config=SpikingConfig(beta=0.5, threshold=10.0, reset=0.0, refractory_steps=0),
    )

    _, voltage = neuron(torch.tensor([[0.8]]))
    assert torch.allclose(voltage, torch.tensor([[0.8]]))

    _, voltage = neuron(torch.zeros(1, 1))
    assert torch.allclose(voltage, torch.tensor([[0.4]]))

    _, voltage = neuron(torch.zeros(1, 1))
    assert torch.allclose(voltage, torch.tensor([[0.2]]))


def test_rate_encode_shape() -> None:
    data = torch.tensor([[0.2, 0.8], [0.5, 0.1]])
    spikes = rate_encode(data, num_steps=7)

    assert spikes.shape == (7, 2, 2)


def test_rate_encode_statistics() -> None:
    torch.manual_seed(0)
    data = torch.tensor([0.1, 0.9])
    spikes = rate_encode(data, num_steps=4000)
    rates = spikes.float().mean(dim=0)

    assert rates[1] > rates[0]
    assert rates[1] - rates[0] > 0.6


def test_latency_encode_ordering() -> None:
    data = torch.tensor([0.2, 0.5, 0.9])
    spikes = latency_encode(data, num_steps=10, tau=1.0)
    first_spike = torch.argmax(spikes, dim=0)

    assert torch.equal(spikes.sum(dim=0), torch.ones_like(data))
    assert first_spike[2] < first_spike[1] < first_spike[0]


def test_delta_encode_changes_only() -> None:
    static = torch.zeros(4, 2)
    assert torch.count_nonzero(delta_encode(static)) == 0

    dynamic = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    spikes = delta_encode(dynamic)

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert torch.equal(spikes, expected)


def test_surrogate_gradient() -> None:
    membrane = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    spikes = SurrogateSpike.apply(membrane)
    spikes.sum().backward()

    assert torch.equal(spikes.detach(), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.all(membrane.grad > 0)


def test_lif_layer_forward() -> None:
    layer = LIFLayer(
        3,
        2,
        config=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        spike_history_steps=3,
    )

    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        layer.linear.bias.zero_()

    inputs = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.4, 0.0]])
    spikes, membrane = layer(inputs)

    assert spikes.shape == (2, 2)
    assert membrane.shape == (2, 2)
    assert torch.equal(spikes, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert layer.spike_history.shape == (1, 2, 2)


def test_batch_processing() -> None:
    neuron = LIFNeuron(
        num_neurons=2,
        config=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
    )
    inputs = torch.tensor(
        [
            [[0.6, 0.4], [0.2, 0.7]],
            [[0.1, 0.8], [0.9, 0.1]],
            [[0.6, 0.6], [0.6, 0.6]],
        ]
    )

    spikes, membrane = neuron(inputs)
    expected_spikes = (inputs > 0.5).float()

    assert spikes.shape == (3, 2, 2)
    assert membrane.shape == (3, 2, 2)
    assert torch.equal(spikes, expected_spikes)
