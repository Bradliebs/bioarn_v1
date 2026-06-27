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

    def set_pool_size(self, pool_size: int) -> None:
        self.entropy_estimator.set_pool_size(pool_size)

    def preview_pool_output(self, fired_indices: list[int]) -> float:
        """Estimate precision for the next presentation without mutating history."""

        entropy = self.entropy_estimator.compute_entropy(candidate=fired_indices)
        self.current_precision = self.precision_signal.compute(entropy)
        return self.current_precision

    def observe_pool_output(self, fired_indices: list[int]) -> float:
        """Update entropy estimate from pool output. Returns current precision."""

        self.entropy_estimator.observe(fired_indices)
        entropy = self.entropy_estimator.compute_entropy()
        self.current_precision = self.precision_signal.compute(entropy)
        return self.current_precision

    def weight_learning_rate(self, base_lr: float | torch.Tensor) -> float | torch.Tensor:
        """Apply precision weighting to a learning rate."""

        if isinstance(base_lr, torch.Tensor):
            return base_lr * self.current_precision
        return float(base_lr) * self.current_precision

    def weight_error_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Apply precision weighting to a prediction error gate tensor."""

        return gate * self.current_precision

    def load_state_from(self, other: "PrecisionWeightedGate") -> None:
        self.entropy_estimator.set_pool_size(other.entropy_estimator.pool_size)
        self.entropy_estimator.fire_history = deque(
            (list(indices) for indices in other.entropy_estimator.fire_history),
            maxlen=other.entropy_estimator.window_size,
        )
        self.current_precision = float(other.current_precision)


__all__ = ["PoolEntropyEstimator", "PrecisionSignal", "PrecisionWeightedGate"]
