import pytest
import torch

from bioarn.config import MarginGateConfig
from bioarn.core.margin_gate import MarginGate


def make_gate(
    theta_margin: float = 0.5,
    theta_margin_lr: float = 0.05,
    theta_resonance: float = 0.7,
) -> MarginGate:
    return MarginGate(
        MarginGateConfig(
            theta_margin=theta_margin,
            theta_margin_lr=theta_margin_lr,
            theta_resonance=theta_resonance,
        )
    )


def test_high_confidence_fires():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor([[1.0, 0.0, 0.0]])
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert result.fired.item() is True
    assert result.abstained.item() is False
    assert result.confidence.item() == pytest.approx(1.0)


def test_low_confidence_abstains():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor([[-1.0, 0.0, 0.0]])
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert result.fired.item() is False
    assert result.abstained.item() is True
    assert result.confidence.item() == pytest.approx(-1.0)


def test_orthogonal_abstains():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor([[0.0, 1.0, 0.0]])
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert result.fired.item() is False
    assert result.confidence.item() == pytest.approx(0.0, abs=1e-6)


def test_batch_mixed_decisions():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert torch.equal(result.fired, torch.tensor([True, False, True]))
    assert torch.equal(result.abstained, torch.tensor([False, True, False]))


def test_threshold_adaptation_up():
    gate = make_gate(theta_margin=0.5, theta_margin_lr=0.05)

    gate.adapt_threshold(recent_fire_rate=0.81)

    assert gate.theta_margin.item() == pytest.approx(0.55)


def test_threshold_adaptation_down():
    gate = make_gate(theta_margin=0.5, theta_margin_lr=0.05)

    gate.adapt_threshold(recent_fire_rate=0.04)

    assert gate.theta_margin.item() == pytest.approx(0.45)


def test_threshold_clamping():
    gate_high = make_gate(theta_margin=0.94, theta_margin_lr=0.05)
    gate_low = make_gate(theta_margin=0.11, theta_margin_lr=0.05)

    gate_high.adapt_threshold(recent_fire_rate=0.9)
    gate_low.adapt_threshold(recent_fire_rate=0.0)

    assert gate_high.theta_margin.item() == pytest.approx(0.95)
    assert gate_low.theta_margin.item() == pytest.approx(0.1)


def test_resonance_match():
    gate = make_gate(theta_resonance=0.7)
    prediction = torch.tensor([[1.0, 0.0, 0.0]])
    actual_input = torch.tensor([[1.0, 0.0, 0.0]])

    result = gate.check_resonance(prediction, actual_input)

    assert result.resonated.item() is True
    assert result.match_score.item() == pytest.approx(1.0)
    assert result.learn_signal.item() == pytest.approx(1.0)


def test_resonance_mismatch():
    gate = make_gate(theta_resonance=0.7)
    prediction = torch.tensor([[1.0, 0.0, 0.0]])
    actual_input = torch.tensor([[0.0, 1.0, 0.0]])

    result = gate.check_resonance(prediction, actual_input)

    assert result.resonated.item() is False
    assert result.match_score.item() == pytest.approx(0.0, abs=1e-6)
    assert result.learn_signal.item() == pytest.approx(0.0)


def test_statistics_tracking():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    gate(input_activation, concept_direction)
    stats = gate.get_stats()

    assert stats["total_presentations"] == 4
    assert stats["total_fires"] == 2
    assert stats["total_abstentions"] == 2
    assert stats["fire_rate"] == pytest.approx(0.5)
    assert stats["avg_confidence_when_fired"] == pytest.approx(1.0)
    assert stats["avg_confidence_when_abstained"] == pytest.approx(0.0, abs=1e-6)


def test_abstain_output_is_zeros():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert torch.equal(result.output[1], torch.zeros(3))


def test_fire_output_preserves_activation():
    gate = make_gate(theta_margin=0.5)
    input_activation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )
    concept_direction = torch.tensor([1.0, 0.0, 0.0])

    result = gate(input_activation, concept_direction)

    assert torch.equal(result.output[0], input_activation[0])
    assert torch.equal(result.output[1], input_activation[1])
