"""Configuration primitives for Bio-ARN ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExpertConfig:
    """Concrete expert definition for an ensemble member."""

    name: str
    pool: Any
    preprocessor: Any | None = None


@dataclass
class EnsembleConfig:
    num_experts: int = 4
    voting_method: str = "weighted"  # "majority", "weighted", "confidence"
    abstention_threshold: float = 0.5  # fraction of experts that must agree
    use_boosting: bool = True
    diversity_target: float = 0.3  # minimum disagreement rate
    expert_configs: list[ExpertConfig] | None = None

    def __post_init__(self) -> None:
        self.num_experts = int(max(1, self.num_experts))
        if self.voting_method not in {"majority", "weighted", "confidence"}:
            raise ValueError("voting_method must be 'majority', 'weighted', or 'confidence'.")
        self.abstention_threshold = float(self.abstention_threshold)
        self.diversity_target = float(self.diversity_target)
        if not 0.0 <= self.abstention_threshold <= 1.0:
            raise ValueError("abstention_threshold must be in [0, 1].")
        if not 0.0 <= self.diversity_target <= 1.0:
            raise ValueError("diversity_target must be in [0, 1].")


__all__ = ["EnsembleConfig", "ExpertConfig"]
