"""Focused lateral-prediction regression tests."""

from __future__ import annotations

import torch

from bioarn.config import CCCConfig, LateralPredictionConfig, MarginGateConfig, PrecisionConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.predictive import LateralPredictionNetwork
from bioarn.predictive.precision_weighting import PrecisionWeightedGate


def _make_config(max_pool_size: int = 4) -> CCCConfig:
    return CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=2,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.3,
        max_pool_size=max_pool_size,
        lock_threshold=0.8,
    )


def _make_margin_config() -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=0.6,
        theta_margin_lr=0.01,
        theta_resonance=0.95,
    )


def _configure_identity_pool(pool: CCCPool) -> None:
    for ccc in pool.cccs:
        with torch.no_grad():
            ccc.f1_layer.weight.copy_(torch.eye(4))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(4)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.locked.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)


def _concept_input() -> torch.Tensor:
    return torch.tensor([1.0, 0.8, 0.2, 0.0])


def _similar_input() -> torch.Tensor:
    return torch.tensor([0.9, 0.7, 0.1, 0.0])


def _commit_ccc(ccc, raw_input: torch.Tensor) -> None:
    f1_output = ccc.f1_encode(raw_input)
    ccc.learn_fast(raw_input, f1_output)


def test_lateral_prediction_network_updates_sparse_weights() -> None:
    config = LateralPredictionConfig(
        enabled=True,
        max_neighbors=2,
        hebbian_lr=0.2,
        anti_hebbian_lr=0.1,
        prediction_threshold=0.0,
    )
    network = LateralPredictionNetwork(pool_size=4, concept_dim=4, config=config)
    concept_directions = normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    predictions = network.predict_lateral([0], concept_directions)
    errors = network.compute_lateral_errors(predictions, [0, 1])
    before = network.lateral_weights.clone()
    network.hebbian_update([0, 1], concept_directions)

    assert 1 in predictions
    assert errors[1] < 0.2
    assert not torch.allclose(before, network.lateral_weights)


def test_precision_gate_uses_lateral_error_for_attention() -> None:
    gate = PrecisionWeightedGate(
        PrecisionConfig(
            enabled=True,
            pool_size=4,
            entropy_window=16,
            lateral_error_weight=0.8,
            hierarchy_error_weight=0.0,
            external_signal_decay=0.0,
        )
    )

    for _ in range(6):
        gate.observe_pool_output([0], lateral_error=0.0)
    low_attention = float(gate.compute_error_attention(0.75))
    gate.observe_pool_output([0], lateral_error=1.0)
    high_attention = float(gate.compute_error_attention(0.75))

    assert high_attention > low_attention
    assert high_attention > 1.0


def test_pool_lateral_prediction_error_updates_precision_state() -> None:
    config = _make_config(max_pool_size=3)
    config.precision = PrecisionConfig(enabled=True, pool_size=3, entropy_window=16)
    config.lateral_prediction = LateralPredictionConfig(
        enabled=True,
        max_neighbors=2,
        prediction_threshold=0.0,
    )
    pool = CCCPool(config, _make_margin_config())
    _configure_identity_pool(pool)
    _commit_ccc(pool.cccs[0], _concept_input())
    _commit_ccc(pool.cccs[1], _similar_input())
    with torch.no_grad():
        pool.cccs[1].margin_gate.theta_margin.fill_(1.1)

    preview_output = pool.preview(_concept_input())
    forward_output = pool(_concept_input(), timestep=3)

    assert preview_output.fired_indices == [0]
    assert forward_output.fired_indices == [0]
    assert pool.get_lateral_prediction_error() > 0.0
    assert float(pool.lateral_network.last_attention_scores[0].item()) > 1.0
