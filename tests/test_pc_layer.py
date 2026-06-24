"""Tests for predictive coding layers."""

from __future__ import annotations

import torch

from bioarn.config import PredictiveConfig
from bioarn.predictive.pc_layer import PCStack, PCLayer, free_energy


def make_config(**overrides: float) -> PredictiveConfig:
    config = PredictiveConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_pc_layer_predict_shape() -> None:
    layer = PCLayer(input_dim=4, output_dim=3, config=make_config())
    higher_state = torch.randn(2, 3)

    prediction = layer.predict(higher_state)

    assert prediction.shape == (2, 4)


def test_pc_layer_error_computation() -> None:
    layer = PCLayer(input_dim=3, output_dim=2, config=make_config(error_threshold=0.0))
    actual_input = torch.tensor([[0.6, 0.1, -0.2]])
    prediction = torch.tensor([[0.2, 0.4, 0.3]])

    error = layer.compute_error(actual_input, prediction)

    assert torch.allclose(error, actual_input - prediction)


def test_pcl_suppression() -> None:
    layer = PCLayer(input_dim=3, output_dim=2, config=make_config(error_threshold=0.05))
    actual_input = torch.tensor([[0.02, 0.20, 0.01]])
    prediction = torch.zeros_like(actual_input)

    error = layer.compute_error(actual_input, prediction)

    assert torch.allclose(error, torch.tensor([[0.0, 0.20, 0.0]]))


def test_state_update_direction() -> None:
    layer = PCLayer(input_dim=2, output_dim=2, config=make_config(gamma=0.2, error_threshold=0.0))
    with torch.no_grad():
        layer.W.copy_(torch.eye(2))

    actual_input = torch.tensor([1.0, 0.0])
    initial_prediction = layer.predict()
    error = layer.compute_error(actual_input, initial_prediction)
    updated_state = layer.update_state(error)
    updated_prediction = layer.predict(updated_state)

    assert updated_state[0] > 0.0
    assert torch.norm(actual_input - updated_prediction) < torch.norm(actual_input - initial_prediction)


def test_weight_update_hebbian() -> None:
    layer = PCLayer(input_dim=2, output_dim=2, config=make_config(eta=0.05))
    with torch.no_grad():
        layer.W.copy_(torch.eye(2))

    old_weights = layer.W.detach().clone()
    error = torch.tensor([1.0, 2.0])
    state = torch.tensor([1.0, 0.0])

    layer.update_weights(error, state)

    assert layer.W[0, 1] > old_weights[0, 1]
    assert torch.allclose(layer.W[1], old_weights[1], atol=1e-5)


def test_weight_normalization() -> None:
    layer = PCLayer(input_dim=4, output_dim=3, config=make_config())
    error = torch.tensor([0.5, -0.2, 0.4, 0.1])
    state = torch.tensor([1.0, 0.5, 0.2])

    layer.update_weights(error, state)

    row_norms = layer.W.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


def test_precision_increases_on_good_prediction() -> None:
    layer = PCLayer(input_dim=3, output_dim=2, config=make_config())
    initial_precision = layer.precision.clone()
    small_error = torch.full((3,), 0.01)

    layer.update_precision(small_error)

    assert torch.all(layer.precision > initial_precision)


def test_precision_decreases_on_bad_prediction() -> None:
    layer = PCLayer(input_dim=3, output_dim=2, config=make_config())
    initial_precision = layer.precision.clone()
    large_error = torch.full((3,), 2.0)

    layer.update_precision(large_error)

    assert torch.all(layer.precision < initial_precision)


def test_pc_stack_settling() -> None:
    stack = PCStack(layer_dims=[2, 2, 2], config=make_config(gamma=0.2, error_threshold=0.0))
    for layer in stack.layers:
        with torch.no_grad():
            layer.W.copy_(torch.eye(2))

    sensory_input = torch.tensor([1.0, 1.0])

    stack.reset()
    one_step = stack(sensory_input, num_iterations=1, learn=False)
    stack.reset()
    settled = stack(sensory_input, num_iterations=5, learn=False)

    assert settled.free_energy < one_step.free_energy
    assert settled.free_energy_trace[-1] <= settled.free_energy_trace[0]


def test_pc_stack_generation() -> None:
    stack = PCStack(layer_dims=[3, 2, 2], config=make_config())
    with torch.no_grad():
        stack.layers[1].W.copy_(torch.eye(2))
        stack.layers[0].W.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    top_state = torch.tensor([0.6, 0.3])
    generated = stack.generate(top_state)

    assert generated.shape == (3,)
    assert torch.all(generated >= 0.0)
    assert torch.isfinite(generated).all()


def test_free_energy_computation() -> None:
    errors = [torch.tensor([0.5, 0.0]), torch.tensor([0.2, -0.1])]
    precisions = [torch.ones(2), torch.full((2,), 2.0)]

    energy = free_energy(errors, precisions)

    assert isinstance(energy, torch.Tensor)
    assert energy.ndim == 0
    assert energy.item() >= 0.0


def test_reset_clears_state() -> None:
    layer = PCLayer(input_dim=2, output_dim=2, config=make_config(error_threshold=0.0))
    with torch.no_grad():
        layer.W.copy_(torch.eye(2))

    error = torch.tensor([1.0, 0.5])
    layer.update_state(error)
    layer.reset_state()

    assert torch.allclose(layer.state, torch.zeros_like(layer.state))
