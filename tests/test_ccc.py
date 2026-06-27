import pytest
import torch

from bioarn.config import CCCConfig, MarginGateConfig, PrecisionConfig
from bioarn.core.ccc import CCCPool, ConceptCellCluster
from bioarn.core.margin_gate import ResonanceOutput
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.predictive import PoolEntropyEstimator, PrecisionSignal


def make_config(max_pool_size: int = 4, lock_threshold: float = 0.8) -> CCCConfig:
    return CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=2,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.3,
        max_pool_size=max_pool_size,
        lock_threshold=lock_threshold,
    )


def make_margin_config(
    theta_margin: float = 0.6, theta_resonance: float = 0.95
) -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=theta_margin,
        theta_margin_lr=0.01,
        theta_resonance=theta_resonance,
    )


def configure_identity_ccc(ccc: ConceptCellCluster) -> None:
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


def make_ccc(
    theta_margin: float = 0.6, theta_resonance: float = 0.95
) -> ConceptCellCluster:
    ccc = ConceptCellCluster(make_config(), make_margin_config(theta_margin, theta_resonance))
    configure_identity_ccc(ccc)
    return ccc


def configure_identity_pool(pool: CCCPool) -> None:
    for ccc in pool.cccs:
        configure_identity_ccc(ccc)


def concept_input() -> torch.Tensor:
    return torch.tensor([1.0, 0.8, 0.2, 0.0])


def similar_input() -> torch.Tensor:
    return torch.tensor([0.9, 0.7, 0.1, 0.0])


def shifted_input() -> torch.Tensor:
    return torch.tensor([0.2, 1.0, 0.8, 0.0])


def commit_ccc(ccc: ConceptCellCluster, raw_input: torch.Tensor | None = None) -> None:
    raw_input = concept_input() if raw_input is None else raw_input
    f1_output = ccc.f1_encode(raw_input)
    ccc.learn_fast(raw_input, f1_output)


def test_ccc_f1_encode_sparse() -> None:
    ccc = make_ccc()

    f1_output = ccc.f1_encode(concept_input())

    assert f1_output.shape == (4,)
    assert torch.count_nonzero(f1_output).item() == 2
    assert torch.equal(f1_output, torch.tensor([1.0, 0.8, 0.0, 0.0]))


def test_ccc_uncommitted_abstains() -> None:
    ccc = make_ccc()

    output = ccc(concept_input())

    assert output.fired is False
    assert output.abstained is True
    assert output.prediction is None
    assert output.resonance is None
    assert output.gate_output.fired.item() is False


def test_ccc_fast_learning() -> None:
    ccc = make_ccc()
    f1_output = ccc.f1_encode(concept_input())

    ccc.learn_fast(concept_input(), f1_output)
    output = ccc(similar_input())

    assert ccc.is_committed.item() is True
    assert output.fired is True
    assert output.confidence.item() > 0.99


def test_ccc_slow_learning_shifts_direction() -> None:
    ccc = make_ccc()
    commit_ccc(ccc)
    previous_direction = ccc.concept_direction.clone()
    shifted_f1 = ccc.f1_encode(shifted_input())
    shifted_f2 = ccc.f2_activate(shifted_f1)
    resonance = ResonanceOutput(
        match_score=torch.tensor(1.0),
        resonated=torch.tensor(True),
        learn_signal=torch.tensor(1.0),
    )

    previous_similarity = cosine_similarity(previous_direction, shifted_f2).item()
    ccc.learn_slow(shifted_input(), shifted_f1, resonance)
    new_similarity = cosine_similarity(ccc.concept_direction, shifted_f2).item()

    assert new_similarity > previous_similarity


def test_ccc_locked_skips_fast_learning() -> None:
    ccc = make_ccc()
    ccc.lock()
    f1_output = ccc.f1_encode(concept_input())

    ccc.learn_fast(concept_input(), f1_output)

    assert ccc.is_committed.item() is False
    assert ccc.locked.item() is True
    assert torch.count_nonzero(ccc.concept_direction).item() == 0


def test_ccc_locked_still_fires_but_skips_slow_learning() -> None:
    ccc = make_ccc(theta_resonance=0.95)
    commit_ccc(ccc)
    ccc.lock()
    previous_direction = ccc.concept_direction.clone()
    previous_feedback = ccc.feedback_weights.clone()

    output = ccc(concept_input(), timestep=5)

    assert output.fired is True
    assert output.resonance is not None
    assert ccc.locked.item() is True
    assert torch.allclose(ccc.concept_direction, previous_direction)
    assert torch.allclose(ccc.feedback_weights, previous_feedback)
    assert ccc.last_fired.item() == 5


def test_ccc_feedback_prediction() -> None:
    ccc = make_ccc()
    commit_ccc(ccc)

    output = ccc(concept_input())

    assert output.prediction is not None
    assert output.prediction.shape == output.f1_output.shape
    assert cosine_similarity(output.prediction, output.f1_output).item() > 0.99


def test_ccc_resonance_triggers_learning() -> None:
    ccc = make_ccc(theta_resonance=0.95)
    commit_ccc(ccc)
    original_feedback = ccc.feedback_weights.clone()
    with torch.no_grad():
        ccc.feedback_weights[0].mul_(0.9)
        ccc.feedback_weights[1].mul_(1.1)

    output = ccc(concept_input(), timestep=7)

    assert output.resonance is not None
    assert output.resonance.resonated.item() is True
    assert not torch.allclose(ccc.feedback_weights, original_feedback)
    assert ccc.last_fired.item() == 7


def test_pool_recruitment() -> None:
    pool = CCCPool(make_config(max_pool_size=3), make_margin_config())
    configure_identity_pool(pool)

    output = pool(concept_input(), timestep=1)

    assert output.recruited is True
    assert output.recruited_index == 0
    assert pool.cccs[0].is_committed.item() is True
    assert 0 in output.fired_indices

    next_output = pool(concept_input(), timestep=2)
    assert next_output.recruited is False
    assert 0 in next_output.fired_indices


def test_pool_parallel_firing() -> None:
    pool = CCCPool(make_config(max_pool_size=3), make_margin_config())
    configure_identity_pool(pool)
    commit_ccc(pool.cccs[0])
    commit_ccc(pool.cccs[1])

    output = pool(concept_input())

    assert output.recruited is False
    assert set(output.fired_indices) == {0, 1}


def test_pool_winner_selection() -> None:
    pool = CCCPool(make_config(max_pool_size=3), make_margin_config(theta_margin=0.6))
    configure_identity_pool(pool)
    target_direction = normalize(torch.tensor([1.0, 0.8, 0.0, 0.0]))

    with torch.no_grad():
        for ccc in pool.cccs:
            ccc.is_committed.fill_(True)
        pool.cccs[0].concept_direction.copy_(target_direction)
        pool.cccs[1].concept_direction.copy_(normalize(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        pool.cccs[2].concept_direction.copy_(normalize(torch.tensor([0.0, 1.0, 0.0, 0.0])))

    output = pool(concept_input())
    winners = pool.get_winners(output, k=2)

    assert output.fired_indices == [0, 1, 2]
    assert winners == [0, 1]


def test_pool_no_recruitment_when_someone_fires() -> None:
    pool = CCCPool(make_config(max_pool_size=3), make_margin_config())
    configure_identity_pool(pool)
    commit_ccc(pool.cccs[0])

    output = pool(concept_input())

    assert output.recruited is False
    assert output.recruited_index is None
    assert pool.get_pool_stats()["num_committed"] == 1


def test_ccc_no_backprop() -> None:
    ccc = make_ccc()
    raw_input = concept_input().requires_grad_()

    f1_output = ccc.f1_encode(raw_input)
    ccc.learn_fast(raw_input, f1_output)
    output = ccc(raw_input)

    assert f1_output.grad_fn is None
    assert output.f1_output.grad_fn is None
    assert ccc.concept_direction.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in ccc.f1_layer.parameters())


def test_pool_stats() -> None:
    pool = CCCPool(make_config(max_pool_size=3), make_margin_config())
    configure_identity_pool(pool)

    pool(concept_input(), timestep=1)
    pool(concept_input(), timestep=2)
    stats = pool.get_pool_stats()

    assert stats["num_committed"] == 1
    assert stats["num_uncommitted"] == 2
    assert stats["num_locked"] == 0
    assert stats["total_concepts"] == 3
    assert stats["mean_confidence"] == pytest.approx(1.0)
    assert stats["fire_rate"] == pytest.approx(1.0)


def test_pool_entropy_estimator_normalizes_uncertainty() -> None:
    familiar = PoolEntropyEstimator(pool_size=4, window_size=16)
    uncertain = PoolEntropyEstimator(pool_size=4, window_size=16)

    for _ in range(8):
        familiar.observe([0])
        uncertain.observe([0, 1])

    assert familiar.compute_entropy() == pytest.approx(0.0)
    assert uncertain.compute_entropy() > 0.95


def test_precision_signal_boosts_learning_when_entropy_is_high() -> None:
    signal = PrecisionSignal(
        alpha=5.0,
        threshold=0.5,
        min_precision=0.1,
        max_precision=1.0,
    )

    low_precision = signal.compute(0.0)
    high_precision = signal.compute(1.0)

    assert 0.1 <= low_precision < 0.5
    assert 0.5 < high_precision <= 1.0


def test_pool_precision_updates_from_preview_and_forward() -> None:
    config = make_config(max_pool_size=3)
    config.precision = PrecisionConfig(enabled=True, pool_size=3, entropy_window=16)
    pool = CCCPool(config, make_margin_config())
    configure_identity_pool(pool)
    commit_ccc(pool.cccs[0])
    commit_ccc(pool.cccs[1], shifted_input())

    preview_output = pool.preview(concept_input())
    preview_precision = pool.get_precision()
    forward_output = pool(concept_input(), timestep=3)

    assert preview_output.fired_indices
    assert forward_output.fired_indices
    assert 0.1 <= preview_precision <= 1.0
    assert 0.1 <= pool.get_precision() <= 1.0


def test_pool_auto_lock_and_stats() -> None:
    pool = CCCPool(make_config(max_pool_size=3, lock_threshold=0.8), make_margin_config())
    configure_identity_pool(pool)
    commit_ccc(pool.cccs[0])

    pool.update_importance([0], confidences=[1.0])

    stats = pool.get_pool_stats()

    assert pool.cccs[0].locked.item() is True
    assert stats["num_locked"] == 1
