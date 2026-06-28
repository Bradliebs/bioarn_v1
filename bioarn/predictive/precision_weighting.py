"""Precision-weighted predictive processing inspired by hippocampal ripples.

Frank et al. (2026) Nature Neuroscience showed that hippocampal ripples
act as precision signals — they increase before stimuli when uncertainty
is high, telling cortex to precision-weight prediction errors.

High precision (high uncertainty) → learn fast from surprise
Low precision (low uncertainty) → protect existing representations
"""

from __future__ import annotations

from collections import Counter, deque
import math

import torch

from bioarn.config import PrecisionConfig


class PoolEntropyEstimator:
    """Estimate contextual uncertainty from recent CCC winner usage."""

    def __init__(self, pool_size: int, window_size: int = 100):
        self.pool_size = int(max(1, pool_size))
        self.window_size = int(max(1, window_size))
        self.fire_history: deque[list[int]] = deque(maxlen=self.window_size)

    def set_pool_size(self, pool_size: int) -> None:
        self.pool_size = int(max(1, pool_size))

    def observe(self, fired_indices: list[int]) -> None:
        self.fire_history.append(list(fired_indices))

    def compute_entropy(self, candidate: list[int] | None = None) -> float:
        """Compute normalized entropy of the recent firing distribution."""

        history_length = len(self.fire_history) + (1 if candidate is not None else 0)
        if history_length < 5:
            return 1.0

        counts: Counter[int] = Counter()
        total = 0
        for indices in self.fire_history:
            for index in indices:
                counts[int(index)] += 1
                total += 1
        if candidate is not None:
            for index in candidate:
                counts[int(index)] += 1
                total += 1
        if total == 0:
            return 1.0

        entropy = 0.0
        for count in counts.values():
            probability = count / total
            if probability > 0.0:
                entropy -= probability * math.log(probability)

        num_active = min(self.pool_size, max(len(counts), 2))
        max_entropy = math.log(num_active)
        if max_entropy <= 0.0:
            return 1.0
        return float(min(1.0, entropy / max_entropy))


class PrecisionSignal:
    """Convert pool entropy into a precision weight."""

    def __init__(
        self,
        alpha: float = 5.0,
        threshold: float = 0.5,
        *,
        min_precision: float = 0.1,
        max_precision: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.min_precision = float(max(0.0, min_precision))
        self.max_precision = float(max(self.min_precision, max_precision))

    def compute(self, pool_entropy: float) -> float:
        """Convert entropy to precision. High entropy increases learning."""

        entropy = float(max(0.0, min(1.0, pool_entropy)))
        sigmoid = 1.0 / (1.0 + math.exp(-self.alpha * (entropy - self.threshold)))
        return float(self.min_precision + ((self.max_precision - self.min_precision) * sigmoid))


class PrecisionWeightedGate:
    """Gate prediction-error learning with a precision signal derived from uncertainty."""

    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.entropy_estimator = PoolEntropyEstimator(
            pool_size=config.pool_size,
            window_size=config.entropy_window,
        )
        self.precision_signal = PrecisionSignal(
            alpha=config.precision_alpha,
            threshold=config.precision_threshold,
            min_precision=config.min_precision,
            max_precision=config.max_precision,
        )
        self.current_precision = float(config.max_precision)
        self.current_entropy = 1.0
        self.current_external_uncertainty = 0.0
        self.current_uncertainty = 1.0

    def set_pool_size(self, pool_size: int) -> None:
        self.entropy_estimator.set_pool_size(pool_size)

    def _external_uncertainty(
        self,
        *,
        lateral_error: float | None = None,
        hierarchy_error: float | None = None,
    ) -> float:
        lateral = 0.0 if lateral_error is None else max(0.0, min(1.0, float(lateral_error)))
        hierarchy = 0.0 if hierarchy_error is None else max(0.0, min(1.0, float(hierarchy_error)))
        combined = (
            float(self.config.lateral_error_weight) * lateral
            + float(self.config.hierarchy_error_weight) * hierarchy
        )
        return float(max(0.0, min(1.0, combined)))

    @staticmethod
    def _blend_uncertainty(pool_entropy: float, external_uncertainty: float) -> float:
        entropy = max(0.0, min(1.0, float(pool_entropy)))
        external = max(0.0, min(1.0, float(external_uncertainty)))
        return float(max(0.0, min(1.0, entropy + external - (entropy * external))))

    def preview_pool_output(
        self,
        fired_indices: list[int],
        *,
        lateral_error: float | None = None,
        hierarchy_error: float | None = None,
    ) -> float:
        """Estimate precision for the next presentation without mutating history."""

        entropy = self.entropy_estimator.compute_entropy(candidate=fired_indices)
        external_uncertainty = max(
            self.current_external_uncertainty,
            self._external_uncertainty(
                lateral_error=lateral_error,
                hierarchy_error=hierarchy_error,
            ),
        )
        uncertainty = self._blend_uncertainty(entropy, external_uncertainty)
        self.current_entropy = entropy
        self.current_uncertainty = uncertainty
        self.current_precision = self.precision_signal.compute(uncertainty)
        return self.current_precision

    def observe_pool_output(
        self,
        fired_indices: list[int],
        *,
        lateral_error: float | None = None,
        hierarchy_error: float | None = None,
    ) -> float:
        """Update entropy estimate from pool output. Returns current precision."""

        self.entropy_estimator.observe(fired_indices)
        entropy = self.entropy_estimator.compute_entropy()
        observed_external = self._external_uncertainty(
            lateral_error=lateral_error,
            hierarchy_error=hierarchy_error,
        )
        decay = float(self.config.external_signal_decay)
        self.current_external_uncertainty = (
            (decay * self.current_external_uncertainty)
            + ((1.0 - decay) * observed_external)
        )
        uncertainty = self._blend_uncertainty(entropy, self.current_external_uncertainty)
        self.current_entropy = entropy
        self.current_uncertainty = uncertainty
        self.current_precision = self.precision_signal.compute(uncertainty)
        return self.current_precision

    def weight_learning_rate(self, base_lr: float | torch.Tensor) -> float | torch.Tensor:
        """Apply precision weighting to a learning rate."""

        if isinstance(base_lr, torch.Tensor):
            return base_lr * self.current_precision
        return float(base_lr) * self.current_precision

    def weight_error_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Apply precision weighting to a prediction error gate tensor."""

        return gate * self.current_precision

    def compute_error_attention(
        self,
        error: float | torch.Tensor,
        *,
        gain: float | None = None,
    ) -> float | torch.Tensor:
        """Convert a surprise magnitude into a precision-modulated attention boost."""

        scale = float(self.config.surprise_gain if gain is None else max(0.0, float(gain)))
        if isinstance(error, torch.Tensor):
            return 1.0 + (error.clamp(min=0.0, max=1.0) * self.current_precision * scale)
        return float(1.0 + (max(0.0, min(1.0, float(error))) * self.current_precision * scale))

    def load_state_from(self, other: "PrecisionWeightedGate") -> None:
        self.entropy_estimator.set_pool_size(other.entropy_estimator.pool_size)
        self.entropy_estimator.fire_history = deque(
            (list(indices) for indices in other.entropy_estimator.fire_history),
            maxlen=other.entropy_estimator.window_size,
        )
        self.current_precision = float(other.current_precision)
        self.current_entropy = float(other.current_entropy)
        self.current_external_uncertainty = float(other.current_external_uncertainty)
        self.current_uncertainty = float(other.current_uncertainty)


__all__ = ["PoolEntropyEstimator", "PrecisionSignal", "PrecisionWeightedGate"]
