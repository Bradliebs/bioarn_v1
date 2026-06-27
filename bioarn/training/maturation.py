"""Developmental maturation schedule for progressive module activation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaturationConfig:
    enabled: bool = False
    num_phases: int = 3
    stability_threshold: float = 0.15
    min_samples_per_phase: int = 100
    auto_transition: bool = True


class MaturationSchedule:
    """Developmental learning schedule that activates modules progressively.

    Phase 1 (Foundation): Hierarchy + Hebbian + Curiosity only
        - Build stable bottom-up features
        - Curiosity ensures good coverage of data

    Phase 2 (Refinement): + GNW Workspace + STDP
        - Winner-take-most competition refines representations
        - Temporal correlations begin solidifying
        - Learning rate reduced (representations should be stabilizing)

    Phase 3 (Integration): + Prediction Error Gating + Feedback
        - Top-down predictions now have stable features to predict against
        - Feedback connections provide meaningful context
        - Error gating modulates learning on truly novel features
    """

    def __init__(self, config: MaturationConfig):
        self.config = config
        self.phase = 1
        self.samples_in_phase = 0
        self.confidence_history: list[float] = []
        self.last_transition_variance: float | None = None
        self.transition_history: list[dict[str, float | int]] = []

    @staticmethod
    def _variance(values: list[float]) -> float:
        if not values:
            return float("inf")
        if len(values) == 1:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def current_variance(self) -> float:
        return self._variance(self.confidence_history)

    def advance(self, *, variance: float | None = None) -> bool:
        if self.phase >= max(1, int(self.config.num_phases)):
            return False
        previous_phase = self.phase
        self.phase += 1
        recorded_variance = None if variance is None else float(variance)
        self.last_transition_variance = recorded_variance
        self.transition_history.append(
            {
                "from_phase": previous_phase,
                "to_phase": self.phase,
                "variance": 0.0 if recorded_variance is None else recorded_variance,
                "samples_seen": self.samples_in_phase,
            }
        )
        self.samples_in_phase = 0
        self.confidence_history.clear()
        return True

    def check_transition(self, ccc_confidences: list[float]) -> bool:
        """Check if representations are stable enough to advance."""

        if not self.config.enabled:
            return False

        self.samples_in_phase += 1
        if ccc_confidences:
            self.confidence_history.append(
                float(sum(ccc_confidences) / max(len(ccc_confidences), 1))
            )

        if not self.config.auto_transition:
            return False
        if self.phase >= max(1, int(self.config.num_phases)):
            return False
        if self.samples_in_phase < max(1, int(self.config.min_samples_per_phase)):
            return False
        if len(self.confidence_history) < 2:
            return False

        variance = self.current_variance()
        if variance > float(self.config.stability_threshold):
            return False
        return self.advance(variance=variance)

    def get_active_modules(self) -> dict[str, bool]:
        """Return which modules should be active in current phase."""

        return {
            "hierarchy": True,
            "curiosity": True,
            "workspace": self.phase >= 2,
            "stdp": self.phase >= 2,
            "error_gating": self.phase >= 3,
            "feedback": self.phase >= 3,
        }

    def get_learning_rate_scale(self) -> float:
        """Lower LR in later phases (fine-tuning, not learning from scratch)."""

        return {1: 1.0, 2: 0.7, 3: 0.4}.get(self.phase, 0.4)


__all__ = ["MaturationConfig", "MaturationSchedule"]
