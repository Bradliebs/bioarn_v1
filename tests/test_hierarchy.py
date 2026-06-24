"""Tests for the predictive hierarchy and CCC connector."""

from __future__ import annotations

import torch

from bioarn.config import CCCConfig, MarginGateConfig, PredictiveConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.predictive.hierarchy import HierarchyConnector, PredictiveHierarchy


def make_predictive_config(**overrides: float) -> PredictiveConfig:
    config = PredictiveConfig(gamma=0.2, eta=0.01, precision_init=1.0, error_threshold=0.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_ccc_pool(input_dim: int = 4, concept_dim: int = 4, max_pool_size: int = 2) -> CCCPool:
    ccc_config = CCCConfig(
        input_dim=input_dim,
        concept_dim=concept_dim,
        num_f1_features=input_dim,
        f1_top_k=input_dim,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.2,
        max_pool_size=max_pool_size,
    )
    margin_config = MarginGateConfig(theta_margin=0.5, theta_margin_lr=0.01, theta_resonance=0.9)
    pool = CCCPool(ccc_config, margin_config)
    with torch.no_grad():
        for ccc in pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(input_dim))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(concept_dim, input_dim)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
    return pool


def make_identity_hierarchy() -> PredictiveHierarchy:
    hierarchy = PredictiveHierarchy(layer_dims=[4, 4, 4, 4], config=make_predictive_config())
    with torch.no_grad():
        for layer in hierarchy.layers:
            layer.W.copy_(torch.eye(4))
            layer.precision.fill_(1.0)
            layer.state.zero_()
    hierarchy.reset()
    return hierarchy


def test_hierarchy_perceive_shape() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.tensor([1.0, 0.5, 0.0, 0.25]), num_iterations=6)

    assert len(output.states) == 4
    assert len(output.errors) == 4
    assert [state.shape for state in output.states] == [(4,), (4,), (4,), (4,)]
    assert [error.shape for error in output.errors] == [(4,), (4,), (4,), (4,)]


def test_hierarchy_free_energy_decreases() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.tensor([1.0, 0.8, 0.2, 0.4]), num_iterations=8)

    assert output.free_energy_trace[-1] <= output.free_energy_trace[0]


def test_hierarchy_convergence() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.perceive(torch.zeros(4), num_iterations=10)

    assert output.converged is True
    assert output.iterations_used < 10


def test_hierarchy_generation() -> None:
    hierarchy = make_identity_hierarchy()

    output = hierarchy.generate(torch.tensor([0.8, 0.3, 0.1, 0.0]))

    assert output.generated_sensory.shape == (4,)
    assert len(output.level_predictions) == 3
    assert torch.all(output.generated_sensory >= 0.0)


def test_hierarchy_predict_and_compare() -> None:
    hierarchy = make_identity_hierarchy()
    familiar = torch.tensor([0.9, 0.1, 0.4, 0.2])
    novel = torch.tensor([0.0, 1.0, 1.0, 0.0])

    hierarchy.perceive(familiar, num_iterations=12)
    familiar_quality = hierarchy.predict_and_compare(familiar)
    novel_quality = hierarchy.predict_and_compare(novel)

    assert familiar_quality.surprise_score < novel_quality.surprise_score
    assert familiar_quality.novel is False
    assert novel_quality.novel is True


def test_active_inference_signal() -> None:
    hierarchy = make_identity_hierarchy()

    action = hierarchy.active_inference_step(torch.zeros(4), torch.ones(4))

    assert torch.count_nonzero(action.direction).item() > 0
    assert action.urgency > 0.0
    assert action.expected_reduction > 0.0


def test_connector_bottom_up() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    concept_level = connector.bottom_up(torch.tensor([1.0, 0.0, 0.5, 0.25]))

    assert concept_level.shape == (4,)


def test_connector_top_down() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    sensory = connector.top_down(torch.tensor([0.7, 0.2, 0.1, 0.0]))

    assert sensory.shape == (4,)
    assert torch.all(sensory >= 0.0)


def test_resonance_loop_converges() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())
    sensory = torch.tensor([0.8, 0.3, 0.0, 0.4])

    output = connector.resonance_loop(sensory, sensory, max_iters=6)

    assert output.resonated is True
    assert output.final_error < 0.1


def test_resonance_triggers_on_match() -> None:
    hierarchy = make_identity_hierarchy()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())
    sensory = torch.tensor([0.6, 0.2, 0.1, 0.0])

    output = connector.resonance_loop(sensory, sensory, max_iters=4)

    assert output.resonated is True
    assert output.iterations >= 1


def test_resonance_fails_on_mismatch() -> None:
    hierarchy = make_identity_hierarchy()
    with torch.no_grad():
        for layer in hierarchy.layers:
            layer.W.zero_()
    connector = HierarchyConnector(hierarchy, make_ccc_pool(), make_predictive_config())

    output = connector.resonance_loop(
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        max_iters=4,
    )

    assert output.resonated is False
    assert output.final_error > 0.1


def test_generation_different_concepts() -> None:
    hierarchy = make_identity_hierarchy()

    first = hierarchy.generate(torch.tensor([0.9, 0.0, 0.0, 0.0])).generated_sensory
    second = hierarchy.generate(torch.tensor([0.0, 0.9, 0.0, 0.0])).generated_sensory

    assert not torch.allclose(first, second)
