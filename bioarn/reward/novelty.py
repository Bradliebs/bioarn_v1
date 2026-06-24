"""Dopaminergic reward, novelty, and curiosity dynamics for Bio-ARN."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from bioarn.config import RewardConfig


@dataclass
class RewardSignal:
    """Reward value derived from intrinsic or external feedback."""

    value: float
    reward_type: str
    relative_to_baseline: float


@dataclass
class NoveltySignal:
    """Fast novelty readout for orienting and learning-rate control."""

    is_novel: bool
    novelty_score: float
    orienting_response: bool
    learning_boost: float
    attention_disruption: float


@dataclass
class CuriositySignal:
    """Preference over possible actions based on expected learning."""

    preferred_index: int
    drive_strength: float
    expected_learning: list[float]


@dataclass
class ModulationOutput:
    """Modulatory outputs broadcast to the rest of the architecture."""

    learning_rate_multiplier: float
    attention_disruption: float
    margin_adjustment: float
    exploration_drive: float


@dataclass
class RewardStepOutput:
    """Full reward-system update for a single timestep."""

    reward: RewardSignal
    novelty: NoveltySignal
    modulation: ModulationOutput
    cumulative_reward: float
    steps_since_novelty: int


class DopamineScheduler:
    """Simple phasic/tonic dopamine controller with decaying transients."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.decay = float(config.novelty_decay)
        self.max_level = max(2.0, float(config.novelty_boost) * 4.0)
        self.reset()

    @staticmethod
    def _scalar(value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(dtype=torch.float32).reshape(())
        return torch.tensor(float(value), dtype=torch.float32)

    def burst(self, magnitude: float) -> None:
        """Register a positive phasic dopamine burst."""

        magnitude_tensor = self._scalar(max(0.0, magnitude))
        self.phasic.add_(magnitude_tensor)
        self.tonic.copy_(torch.clamp(self.tonic + (0.1 * magnitude_tensor), 0.1, self.max_level))

    def dip(self, magnitude: float) -> None:
        """Register a negative phasic dopamine dip."""

        magnitude_tensor = self._scalar(max(0.0, magnitude))
        self.phasic.sub_(magnitude_tensor)
        self.tonic.copy_(torch.clamp(self.tonic - (0.1 * magnitude_tensor), 0.1, self.max_level))

    def tonic_level(self) -> float:
        """Return the current effective dopamine level."""

        level = torch.clamp(self.tonic + self.phasic, 0.1, self.max_level)
        return float(level.item())

    def update(self) -> None:
        """Decay phasic transients and relax tonic dopamine back to baseline."""

        decay_tensor = self._scalar(self.decay)
        baseline = self._scalar(1.0)
        self.phasic.mul_(decay_tensor)
        self.tonic.copy_(baseline + ((self.tonic - baseline) * decay_tensor))

    def reset(self) -> None:
        """Reset dopamine dynamics to baseline."""

        self.tonic = torch.tensor(1.0, dtype=torch.float32)
        self.phasic = torch.tensor(0.0, dtype=torch.float32)


class RewardSystem(nn.Module):
    """Intrinsic reward, novelty detection, curiosity, and modulation controller."""

    def __init__(self, config: RewardConfig):
        super().__init__()
        self.config = config
        self.history_size = 1000
        self.ema_alpha = 0.9
        self.eps = 1e-6
        self.dopamine = DopamineScheduler(config)

        self.register_buffer("prediction_error_history", torch.zeros(self.history_size, dtype=torch.float32))
        self.register_buffer("reward_history", torch.zeros(self.history_size, dtype=torch.float32))
        self.register_buffer("novelty_state", torch.zeros((), dtype=torch.float32))
        self.register_buffer("curiosity_state", torch.zeros((), dtype=torch.float32))
        self.register_buffer("learning_rate_multiplier", torch.ones((), dtype=torch.float32))
        self.register_buffer("prediction_error_baseline", torch.zeros((), dtype=torch.float32))
        self.register_buffer("reward_baseline", torch.zeros((), dtype=torch.float32))
        self.register_buffer("cumulative_reward", torch.zeros((), dtype=torch.float32))
        self.register_buffer("last_prediction_error", torch.full((), float("nan"), dtype=torch.float32))
        self.register_buffer("steps_since_novelty_tensor", torch.zeros((), dtype=torch.long))
        self.register_buffer("novelty_events", torch.zeros((), dtype=torch.long))
        self.register_buffer("attention_disruption_state", torch.zeros((), dtype=torch.float32))
        self.register_buffer("_prediction_error_cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("_reward_cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("_prediction_error_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_reward_count", torch.zeros((), dtype=torch.long))

    def _scalar(self, value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.learning_rate_multiplier.device, dtype=torch.float32).reshape(())
        return torch.tensor(
            float(value),
            device=self.learning_rate_multiplier.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _safe_float(value: torch.Tensor) -> float:
        return float(value.detach().item())

    def _append_history(
        self,
        history_name: str,
        cursor_name: str,
        count_name: str,
        value: torch.Tensor,
    ) -> None:
        history = getattr(self, history_name)
        cursor = getattr(self, cursor_name)
        count = getattr(self, count_name)
        index = int(cursor.item()) % history.numel()
        history[index] = value
        cursor.fill_((index + 1) % history.numel())
        count.fill_(min(int(count.item()) + 1, history.numel()))

    def _history_values(
        self,
        history_name: str,
        cursor_name: str,
        count_name: str,
    ) -> list[float]:
        history = getattr(self, history_name)
        cursor = int(getattr(self, cursor_name).item())
        count = int(getattr(self, count_name).item())
        if count == 0:
            return []
        start = (cursor - count) % history.numel()
        if start < cursor:
            values = history[start:cursor]
        else:
            values = torch.cat((history[start:], history[:cursor]), dim=0)
        return [float(item) for item in values[:count].tolist()]

    def _update_ema(self, baseline: torch.Tensor, value: torch.Tensor, count: torch.Tensor) -> None:
        if int(count.item()) <= 1 and baseline.abs().item() <= self.eps:
            baseline.copy_(value)
            return
        alpha = self._scalar(self.ema_alpha)
        baseline.copy_((alpha * baseline) + ((1.0 - alpha) * value))

    def _relative_to_baseline(self, value: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if baseline.abs().item() <= self.eps:
            return value
        return (value - baseline) / baseline.abs().clamp_min(self.eps)

    def _refresh_learning_rate_multiplier(self) -> None:
        novelty_component = 1.0 + (self.novelty_state * max(0.0, float(self.config.novelty_boost) - 1.0))
        dopamine_component = self._scalar(self.dopamine.tonic_level())
        combined = torch.clamp(
            novelty_component * dopamine_component,
            min=0.1,
            max=max(2.0, float(self.config.novelty_boost) * 4.0),
        )
        self.learning_rate_multiplier.copy_(combined)
        self.attention_disruption_state.copy_(self.novelty_state.clamp(0.0, 1.0))

    def _decay_novelty(self) -> None:
        self.novelty_state.mul_(float(self.config.novelty_decay))
        if self.novelty_state.abs().item() < 1e-3:
            self.novelty_state.zero_()

    def compute_intrinsic_reward(self, current_error: float, previous_error: float) -> RewardSignal:
        """Reward prediction-error reduction relative to recent experience."""

        current_error_tensor = self._scalar(abs(current_error))
        previous_error_tensor = self._scalar(abs(previous_error))
        reward_value = (previous_error_tensor - current_error_tensor) * float(self.config.intrinsic_scale)
        relative = self._relative_to_baseline(reward_value, self.reward_baseline)

        self._append_history("reward_history", "_reward_cursor", "_reward_count", reward_value)
        self._update_ema(self.reward_baseline, reward_value, self._reward_count)
        self.cumulative_reward.add_(reward_value)

        reward_float = self._safe_float(reward_value)
        if reward_float > 0.0:
            self.dopamine.burst(reward_float)
        elif reward_float < 0.0:
            self.dopamine.dip(abs(reward_float))

        self._refresh_learning_rate_multiplier()
        return RewardSignal(
            value=reward_float,
            reward_type="intrinsic",
            relative_to_baseline=self._safe_float(relative),
        )

    def detect_novelty(self, prediction_error: float) -> NoveltySignal:
        """Detect large deviations from the running prediction-error baseline."""

        prediction_error_tensor = self._scalar(abs(prediction_error))
        if int(self._prediction_error_count.item()) == 0:
            baseline = prediction_error_tensor.clamp_min(self.eps)
        else:
            baseline = self.prediction_error_baseline.clamp_min(self.eps)

        novelty_ratio = prediction_error_tensor / baseline
        novelty_score = torch.clamp(novelty_ratio - 1.0, min=0.0)
        is_novel = bool(
            prediction_error_tensor.item() > 0.0
            and novelty_ratio.item() > float(self.config.novelty_threshold)
        )

        if is_novel:
            self.novelty_state.fill_(1.0)
            self.steps_since_novelty_tensor.zero_()
            self.novelty_events.add_(torch.ones((), dtype=torch.long, device=self.novelty_events.device))
        else:
            self.steps_since_novelty_tensor.add_(torch.ones((), dtype=torch.long, device=self.steps_since_novelty_tensor.device))

        self._append_history(
            "prediction_error_history",
            "_prediction_error_cursor",
            "_prediction_error_count",
            prediction_error_tensor,
        )
        self._update_ema(
            self.prediction_error_baseline,
            prediction_error_tensor,
            self._prediction_error_count,
        )
        self._refresh_learning_rate_multiplier()

        learning_boost = 1.0 + (
            self.novelty_state * max(0.0, float(self.config.novelty_boost) - 1.0)
        )
        attention_disruption = self.novelty_state.clamp(0.0, 1.0)
        return NoveltySignal(
            is_novel=is_novel,
            novelty_score=self._safe_float(novelty_score),
            orienting_response=is_novel,
            learning_boost=self._safe_float(learning_boost),
            attention_disruption=self._safe_float(attention_disruption),
        )

    def compute_curiosity(self, available_options: list[float]) -> CuriositySignal:
        """Prefer options with high expected learning, not merely high confusion."""

        if not available_options:
            return CuriositySignal(preferred_index=-1, drive_strength=0.0, expected_learning=[])

        options = torch.as_tensor(
            available_options,
            device=self.learning_rate_multiplier.device,
            dtype=torch.float32,
        ).abs()
        scale = torch.maximum(
            options.mean(),
            self.prediction_error_baseline.clamp_min(self.eps),
        ).clamp_min(self.eps)
        reducibility = torch.exp(-options / scale)
        expected_learning = options * reducibility

        preferred_index = int(torch.argmax(expected_learning).item())
        best_expected_learning = expected_learning[preferred_index]
        drive_strength = torch.clamp(
            (best_expected_learning / scale) * float(self.config.curiosity_weight),
            min=0.0,
            max=1.0,
        )

        self.curiosity_state.copy_(
            torch.clamp(
                (self.curiosity_state * float(self.config.novelty_decay)) + drive_strength,
                min=0.0,
                max=1.0,
            )
        )

        return CuriositySignal(
            preferred_index=preferred_index,
            drive_strength=self._safe_float(drive_strength),
            expected_learning=[float(value) for value in expected_learning.tolist()],
        )

    def apply_external_reward(self, reward_value: float, source: str = "task") -> None:
        """Inject an external reward signal into the modulation dynamics."""

        del source
        reward_tensor = self._scalar(reward_value)
        relative = self._relative_to_baseline(reward_tensor, self.reward_baseline)

        self._append_history("reward_history", "_reward_cursor", "_reward_count", reward_tensor)
        self._update_ema(self.reward_baseline, reward_tensor, self._reward_count)
        self.cumulative_reward.add_(reward_tensor)

        reward_float = self._safe_float(reward_tensor)
        if reward_float > 0.0:
            self.dopamine.burst(reward_float)
        elif reward_float < 0.0:
            self.dopamine.dip(abs(reward_float))

        curiosity_delta = torch.clamp(relative.abs() * float(self.config.curiosity_weight), 0.0, 1.0)
        self.curiosity_state.copy_(
            torch.clamp(self.curiosity_state + curiosity_delta, min=0.0, max=1.0)
        )
        self._refresh_learning_rate_multiplier()

    def get_modulation(self) -> ModulationOutput:
        """Return the current cross-system modulation values."""

        self._refresh_learning_rate_multiplier()
        attention_disruption = self.attention_disruption_state.clamp(0.0, 1.0)
        margin_adjustment = torch.clamp(
            (-0.2 * attention_disruption) + (0.05 * (self.learning_rate_multiplier - 1.0)),
            min=-0.5,
            max=0.5,
        )
        exploration_drive = torch.clamp(
            self.curiosity_state + (0.5 * attention_disruption),
            min=0.0,
            max=1.0,
        )
        return ModulationOutput(
            learning_rate_multiplier=self._safe_float(self.learning_rate_multiplier),
            attention_disruption=self._safe_float(attention_disruption),
            margin_adjustment=self._safe_float(margin_adjustment),
            exploration_drive=self._safe_float(exploration_drive),
        )

    def step(self, prediction_error: float, learned: bool = False) -> RewardStepOutput:
        """Advance reward, novelty, curiosity, and dopamine state by one timestep."""

        prediction_error_tensor = self._scalar(abs(prediction_error))
        previous_error = prediction_error_tensor
        if not math.isnan(float(self.last_prediction_error.item())):
            previous_error = self.last_prediction_error

        reward = self.compute_intrinsic_reward(
            current_error=self._safe_float(prediction_error_tensor),
            previous_error=self._safe_float(previous_error),
        )
        novelty = self.detect_novelty(self._safe_float(prediction_error_tensor))

        curiosity_delta = torch.clamp(
            self._scalar(max(0.0, novelty.novelty_score))
            * float(self.config.curiosity_weight)
            / max(float(self.config.novelty_threshold), 1.0),
            min=0.0,
            max=1.0,
        )
        if learned and reward.value > 0.0:
            curiosity_delta = torch.clamp(
                curiosity_delta + self._scalar(reward.value * float(self.config.curiosity_weight)),
                min=0.0,
                max=1.0,
            )

        self.curiosity_state.copy_(
            torch.clamp(
                (self.curiosity_state * float(self.config.novelty_decay)) + curiosity_delta,
                min=0.0,
                max=1.0,
            )
        )
        modulation = self.get_modulation()

        if not novelty.is_novel:
            self._decay_novelty()
        self.dopamine.update()
        self._refresh_learning_rate_multiplier()
        self.last_prediction_error.copy_(prediction_error_tensor)

        return RewardStepOutput(
            reward=reward,
            novelty=novelty,
            modulation=modulation,
            cumulative_reward=self._safe_float(self.cumulative_reward),
            steps_since_novelty=int(self.steps_since_novelty_tensor.item()),
        )

    def reset(self) -> None:
        """Reset all reward, novelty, and curiosity state."""

        self.prediction_error_history.zero_()
        self.reward_history.zero_()
        self.novelty_state.zero_()
        self.curiosity_state.zero_()
        self.learning_rate_multiplier.fill_(1.0)
        self.prediction_error_baseline.zero_()
        self.reward_baseline.zero_()
        self.cumulative_reward.zero_()
        self.last_prediction_error.fill_(float("nan"))
        self.steps_since_novelty_tensor.zero_()
        self.novelty_events.zero_()
        self.attention_disruption_state.zero_()
        self._prediction_error_cursor.zero_()
        self._reward_cursor.zero_()
        self._prediction_error_count.zero_()
        self._reward_count.zero_()
        self.dopamine.reset()

    def get_stats(self) -> dict[str, object]:
        """Expose bounded histories and current modulation state."""

        return {
            "prediction_error_history": self._history_values(
                "prediction_error_history",
                "_prediction_error_cursor",
                "_prediction_error_count",
            ),
            "reward_history": self._history_values(
                "reward_history",
                "_reward_cursor",
                "_reward_count",
            ),
            "prediction_error_baseline": self._safe_float(self.prediction_error_baseline),
            "reward_baseline": self._safe_float(self.reward_baseline),
            "novelty_state": self._safe_float(self.novelty_state),
            "curiosity_state": self._safe_float(self.curiosity_state),
            "learning_rate_multiplier": self._safe_float(self.learning_rate_multiplier),
            "cumulative_reward": self._safe_float(self.cumulative_reward),
            "novelty_events": int(self.novelty_events.item()),
            "steps_since_novelty": int(self.steps_since_novelty_tensor.item()),
            "dopamine_tonic_level": float(self.dopamine.tonic_level()),
            "modulation": self.get_modulation(),
        }


__all__ = [
    "CuriositySignal",
    "DopamineScheduler",
    "ModulationOutput",
    "NoveltySignal",
    "RewardSignal",
    "RewardStepOutput",
    "RewardSystem",
]
