from __future__ import annotations

import pytest

from bioarn.config import RewardConfig
from bioarn.reward.novelty import DopamineScheduler, RewardSystem


def make_config(**overrides: float) -> RewardConfig:
    config = RewardConfig(
        intrinsic_scale=1.0,
        novelty_threshold=2.0,
        novelty_boost=3.0,
        novelty_decay=0.5,
        curiosity_weight=0.5,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_system(**overrides: float) -> RewardSystem:
    return RewardSystem(make_config(**overrides))


def test_intrinsic_reward_positive() -> None:
    reward_system = make_system()

    reward = reward_system.compute_intrinsic_reward(current_error=0.25, previous_error=1.0)

    assert reward.reward_type == "intrinsic"
    assert reward.value > 0.0


def test_intrinsic_reward_zero() -> None:
    reward_system = make_system()

    reward = reward_system.compute_intrinsic_reward(current_error=0.5, previous_error=0.5)

    assert reward.value == pytest.approx(0.0)


def test_intrinsic_reward_negative() -> None:
    reward_system = make_system()

    reward = reward_system.compute_intrinsic_reward(current_error=1.0, previous_error=0.25)

    assert reward.value < 0.0


def test_novelty_detection() -> None:
    reward_system = make_system()
    reward_system.detect_novelty(1.0)

    novelty = reward_system.detect_novelty(3.0)

    assert novelty.is_novel is True
    assert novelty.orienting_response is True
    assert novelty.novelty_score > 0.0


def test_novelty_not_triggered() -> None:
    reward_system = make_system()
    reward_system.detect_novelty(1.0)

    novelty = reward_system.detect_novelty(1.5)

    assert novelty.is_novel is False
    assert novelty.orienting_response is False


def test_novelty_learning_boost() -> None:
    reward_system = make_system()
    reward_system.detect_novelty(1.0)

    novelty = reward_system.detect_novelty(3.0)
    modulation = reward_system.get_modulation()

    assert novelty.learning_boost > 1.0
    assert modulation.learning_rate_multiplier > 1.0


def test_novelty_decay() -> None:
    reward_system = make_system()
    reward_system.detect_novelty(1.0)
    reward_system.detect_novelty(3.0)
    boosted = reward_system.get_modulation().learning_rate_multiplier

    for _ in range(6):
        reward_system.step(1.0)
    decayed = reward_system.get_modulation().learning_rate_multiplier

    assert boosted > decayed
    assert decayed == pytest.approx(1.0, abs=0.1)


def test_curiosity_prefers_learnable() -> None:
    reward_system = make_system()

    curiosity = reward_system.compute_curiosity([0.2, 1.0, 4.0])

    assert curiosity.preferred_index == 1
    assert curiosity.drive_strength > 0.0
    assert len(curiosity.expected_learning) == 3
    assert curiosity.expected_learning[1] == max(curiosity.expected_learning)


def test_external_reward_modulates() -> None:
    reward_system = make_system()
    baseline = reward_system.get_modulation().learning_rate_multiplier

    reward_system.apply_external_reward(1.5)
    boosted = reward_system.get_modulation().learning_rate_multiplier

    assert boosted > baseline


def test_modulation_output() -> None:
    reward_system = make_system()
    reward_system.detect_novelty(1.0)
    reward_system.compute_curiosity([0.2, 1.0, 4.0])

    modulation = reward_system.get_modulation()

    assert modulation.learning_rate_multiplier >= 0.1
    assert 0.0 <= modulation.attention_disruption <= 1.0
    assert -0.5 <= modulation.margin_adjustment <= 0.5
    assert 0.0 <= modulation.exploration_drive <= 1.0


def test_dopamine_burst() -> None:
    scheduler = DopamineScheduler(make_config())
    baseline = scheduler.tonic_level()

    scheduler.burst(1.0)

    assert scheduler.tonic_level() > baseline


def test_dopamine_dip() -> None:
    scheduler = DopamineScheduler(make_config())
    baseline = scheduler.tonic_level()

    scheduler.dip(0.5)

    assert scheduler.tonic_level() < baseline


def test_step_full_cycle() -> None:
    reward_system = make_system()
    reward_system.step(1.0)

    output = reward_system.step(0.25, learned=True)

    assert output.reward.value > 0.0
    assert output.cumulative_reward > 0.0
    assert output.modulation.learning_rate_multiplier > 1.0
    assert output.steps_since_novelty >= 1


def test_reset_clears_state() -> None:
    reward_system = make_system()
    reward_system.step(1.0)
    reward_system.apply_external_reward(1.0)
    reward_system.compute_curiosity([0.2, 1.0, 4.0])

    reward_system.reset()
    stats = reward_system.get_stats()

    assert stats["prediction_error_history"] == []
    assert stats["reward_history"] == []
    assert stats["novelty_events"] == 0
    assert stats["novelty_state"] == pytest.approx(0.0)
    assert stats["curiosity_state"] == pytest.approx(0.0)
    assert stats["learning_rate_multiplier"] == pytest.approx(1.0)
