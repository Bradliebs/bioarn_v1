"""Tests for the full Bio-ARN sensorimotor loop."""

from __future__ import annotations

import torch

from bioarn.config import (
    BioARNConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    PredictiveConfig,
    RewardConfig,
    SDMConfig,
    SpikingConfig,
)
from bioarn.loop import SensorimotorLoop


def make_config() -> BioARNConfig:
    return BioARNConfig(
        spiking=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        ccc=CCCConfig(
            input_dim=16,
            concept_dim=16,
            num_f1_features=16,
            f1_top_k=4,
            fast_lr=1.0,
            slow_lr=0.1,
            feedback_lr=0.1,
            max_pool_size=8,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.0,
            theta_margin_lr=0.01,
            theta_resonance=0.5,
        ),
        sdm=SDMConfig(
            address_dim=16,
            hamming_radius=2,
            num_hard_locations=32,
            data_dim=16,
            decay_rate=0.99,
            stdp_window=4,
        ),
        predictive=PredictiveConfig(
            num_levels=4,
            gamma=0.2,
            eta=0.05,
            precision_init=1.0,
            error_threshold=0.0,
        ),
        gnw=GNWConfig(
            capacity=3,
            broadcast_gain=2.0,
            fatigue_rate=0.05,
            fatigue_threshold=0.1,
            competition_temp=0.5,
        ),
        reward=RewardConfig(
            intrinsic_scale=1.0,
            novelty_threshold=1.5,
            novelty_boost=2.5,
            novelty_decay=0.8,
            curiosity_weight=0.5,
        ),
        seed=5,
    )


def make_loop() -> SensorimotorLoop:
    return SensorimotorLoop(make_config())


def visual_frame(top: int, left: int, size: int = 2) -> torch.Tensor:
    frame = torch.zeros(1, 1, 4, 4)
    frame[:, :, top : top + size, left : left + size] = 1.0
    return frame


def language_tokens(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def concept_vector(index: int, size: int = 16) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=size).float()


def test_loop_initialization() -> None:
    loop = make_loop()

    assert loop.visual_encoder is not None
    assert loop.language_encoder is not None
    assert loop.motor_stream is not None
    assert loop.hierarchy is not None
    assert loop.connector is not None
    assert loop.core is not None
    assert loop.reward is not None
    assert loop.timestep == 0


def test_sense_visual() -> None:
    loop = make_loop()

    output = loop.sense(visual_input=visual_frame(0, 0))

    assert output.visual_output is not None
    assert output.language_output is None
    assert output.features.shape[-1] == 16
    assert output.suppressed_fraction >= 0.0


def test_sense_language() -> None:
    loop = make_loop()

    output = loop.sense(language_input=language_tokens([1, 2, 3]))

    assert output.visual_output is None
    assert output.language_output is not None
    assert output.features.shape[-1] == 16
    assert output.suppressed_fraction >= 0.0


def test_full_step_visual() -> None:
    loop = make_loop()

    output = loop.step(visual_input=visual_frame(1, 1))

    assert output.sensory.visual_output is not None
    assert output.prediction.free_energy >= 0.0
    assert output.recognition.confidence >= 0.0
    assert output.attention.broadcast.num_occupied >= 1
    assert output.plan is not None
    assert output.action is not None
    assert output.action.generated is not None


def test_full_step_language() -> None:
    loop = make_loop()

    output = loop.step(language_input=language_tokens([4, 5, 6]))

    assert output.sensory.language_output is not None
    assert output.prediction.surprise >= 0.0
    assert output.attention.modulation.learning_rate_multiplier >= 0.1
    assert output.plan is not None
    assert output.action is not None
    assert output.action.generated is not None


def test_run_multiple_steps() -> None:
    loop = make_loop()
    inputs = [
        language_tokens([1, 2, 3]),
        language_tokens([1, 2, 3]),
        language_tokens([7, 8, 9]),
    ]

    output = loop.run(inputs)

    assert len(output.steps) == 3
    assert len(output.free_energy_trace) == 3
    assert len(output.reward_trace) == 3
    assert output.final_stats["steps"] == 3


def test_generation_mode() -> None:
    loop = make_loop()
    loop.core.gnw.inject(0, concept_vector(2), priority=1.0)

    output = loop.run(inputs=[], num_steps=5, generate=True)

    assert output.generated_text is not None
    assert len(output.generated_text) > 0
    assert len(output.steps) >= 1


def test_self_monitoring() -> None:
    loop = make_loop()
    seed = concept_vector(3)
    plan = loop.plan(seed)

    probe = loop.motor_stream.execute_step(plan.motor_plan)
    predicted_token = int(torch.argmax(probe.logits, dim=-1).item())
    wrong_prediction = torch.zeros(1, loop.vocab_size)
    wrong_prediction[0, (predicted_token + 1) % loop.vocab_size] = 1.0

    loop.motor_stream.reset()
    plan = loop.plan(seed)
    loop.motor_stream.prediction_buffer = wrong_prediction
    action = loop.act(plan)

    assert action.generated is not None
    assert action.self_correction is True
    assert action.feedback_error >= 0.0


def test_reward_modulates_learning() -> None:
    loop = make_loop()
    baseline_lr = loop.core.config.ccc.slow_lr

    loop.step(language_input=language_tokens([1, 1, 1]))
    loop.step(language_input=language_tokens([1, 1, 1]))
    novel_step = loop.step(language_input=language_tokens([15, 14, 13]))

    assert novel_step.reward.modulation.learning_rate_multiplier >= 1.0
    assert loop.core.config.ccc.slow_lr >= baseline_lr
    if novel_step.reward.novelty.is_novel:
        assert loop.core.config.ccc.slow_lr > baseline_lr


def test_active_inference() -> None:
    loop = make_loop()

    direction = loop.active_inference_step(torch.zeros(16), torch.ones(16))
    step_output = loop.step(language_input=language_tokens([2, 3, 4]), goal=torch.ones(16))

    assert direction.shape == (16,)
    assert torch.count_nonzero(direction).item() > 0
    assert step_output.plan is not None
    assert step_output.plan.action_signal is not None
    assert step_output.plan.action_signal.expected_reduction > 0.0


def test_free_energy_decreases() -> None:
    loop = make_loop()
    repeated = language_tokens([3, 4, 5])

    output = loop.run([repeated, repeated.clone(), repeated.clone(), repeated.clone()])

    assert output.free_energy_trace[-1] <= output.free_energy_trace[0] + 1e-6


def test_loop_stats() -> None:
    loop = make_loop()
    output = loop.run([language_tokens([1, 2, 3]), language_tokens([4, 5, 6])])

    assert {
        "steps",
        "final_free_energy",
        "mean_free_energy",
        "mean_reward",
        "cumulative_reward",
        "novelty_events",
        "concepts_learned",
        "workspace_occupancy",
        "generated_tokens",
        "last_modulation",
    } <= set(output.final_stats)
    assert output.total_learning_events >= 1
    assert output.final_stats["steps"] == len(output.steps)
