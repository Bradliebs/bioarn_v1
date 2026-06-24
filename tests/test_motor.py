"""Tests for the predictive motor cortex."""

from __future__ import annotations

import torch

from bioarn.config import SpikingConfig
from bioarn.sensorimotor.motor import ConceptToLanguage, LanguageMotorStream, PhysicalMotorStream


def make_spiking_config() -> SpikingConfig:
    return SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0)


def make_motor_stream(vocab_size: int = 8, hidden_dim: int = 32) -> LanguageMotorStream:
    return LanguageMotorStream(
        concept_dim=12,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        config=make_spiking_config(),
    )


def concept(index: int, size: int = 12, scale: float = 1.0) -> torch.Tensor:
    value = torch.zeros(size)
    value[index] = scale
    return value


def test_motor_plan_shape() -> None:
    stream = make_motor_stream()

    motor_plan = stream.plan(concept(0))

    assert motor_plan.shape == (1, 32)


def test_motor_execute_step() -> None:
    stream = make_motor_stream()
    motor_plan = stream.plan(concept(1))

    output = stream.execute_step(motor_plan)
    probabilities = torch.softmax(output.logits, dim=-1)

    assert output.logits.shape == (1, 8)
    assert output.spike_state.shape == (1, 32)
    assert 0.0 <= output.confidence <= 1.0
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_generate_sequence_runs() -> None:
    stream = make_motor_stream(vocab_size=10)

    output = stream.generate_sequence(concept(2), max_length=6)

    assert output.token_ids.ndim == 1
    assert output.token_ids.numel() >= 1
    assert output.logits_sequence.shape[0] == output.token_ids.shape[0]
    assert output.logits_sequence.shape[1] == 10
    assert len(output.confidences) == output.token_ids.shape[0]


def test_generation_stops_at_max_length() -> None:
    stream = make_motor_stream(vocab_size=10)
    stream.confidence_threshold = 0.0
    stream.eos_confidence_threshold = 2.0

    output = stream.generate_sequence(concept(3), max_length=3)

    assert output.token_ids.numel() == 3
    assert output.stopped_reason == "max_length"


def test_self_monitoring_detects_error() -> None:
    stream = make_motor_stream(vocab_size=6)
    produced = torch.tensor([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    predicted = torch.tensor([0.01, 0.95, 0.01, 0.01, 0.01, 0.01])

    monitor = stream.self_monitor_step(produced, predicted)

    assert monitor.should_revise
    assert monitor.error_magnitude > stream.monitor_revision_threshold
    assert monitor.correction.shape == (32,)


def test_self_monitoring_passes_match() -> None:
    stream = make_motor_stream(vocab_size=6)
    distribution = torch.tensor([0.05, 0.05, 0.8, 0.05, 0.03, 0.02])

    monitor = stream.self_monitor_step(distribution, distribution.clone())

    assert not monitor.should_revise
    assert monitor.error_magnitude == 0.0
    assert torch.allclose(monitor.correction, torch.zeros_like(monitor.correction))


def test_self_correction() -> None:
    stream = make_motor_stream(vocab_size=8)
    motor_plan = stream.plan(concept(4))
    produced = torch.tensor([0.98, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
    predicted = torch.tensor([0.0, 0.98, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01])

    monitor = stream.self_monitor_step(produced, predicted)
    adjusted_plan = torch.tanh(motor_plan - (stream.correction_gain * monitor.correction.unsqueeze(0)))

    assert monitor.should_revise
    assert not torch.allclose(adjusted_plan, motor_plan)


def test_confidence_stopping() -> None:
    stream = make_motor_stream(vocab_size=8)
    with torch.no_grad():
        stream.output_projection.weight.zero_()
        stream.output_projection.bias.zero_()

    output = stream.generate_sequence(concept(5), max_length=6)

    assert output.stopped_reason == "low_confidence"
    assert output.token_ids.numel() == 1


def test_different_concepts_different_output() -> None:
    stream = make_motor_stream(vocab_size=10)
    stream.confidence_threshold = 0.0
    stream.eos_confidence_threshold = 2.0

    first = stream.generate_sequence(concept(0), max_length=4)
    second = stream.generate_sequence(concept(7), max_length=4)

    assert not torch.allclose(first.logits_sequence, second.logits_sequence)


def test_concept_to_language() -> None:
    model = ConceptToLanguage(concept_dim=12, vocab_size=8, config=make_spiking_config())
    vocab = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "<eos>": 7,
    }

    text = model.speak([concept(1), concept(2)], vocab=vocab)

    assert isinstance(text, str)
    assert len(text) > 0


def test_physical_motor_basic() -> None:
    stream = PhysicalMotorStream(concept_dim=12, action_dim=5, config=make_spiking_config())

    plan = stream.plan_action(concept(6))
    action = stream.execute_action(plan)

    assert plan.shape == (1, 5)
    assert action.action_vector.shape == (1, 5)
    assert 0.0 <= action.confidence <= 1.0


def test_reset_clears_state() -> None:
    stream = make_motor_stream()
    motor_plan = stream.plan(concept(0))
    stream.execute_step(motor_plan)
    stream.predict_next(torch.softmax(torch.ones(8), dim=-1), motor_plan.squeeze(0))

    assert stream.prediction_buffer.numel() > 0
    assert stream.last_motor_state.numel() > 0
    assert stream.motor_planner.spike_history.numel() > 0
    assert stream.motor_executor.spike_history.numel() > 0

    stream.reset()

    assert stream.prediction_buffer.numel() == 0
    assert stream.last_motor_state.numel() == 0
    assert stream.last_output_distribution.numel() == 0
    assert stream.motor_planner.spike_history.numel() == 0
    assert stream.motor_executor.spike_history.numel() == 0
