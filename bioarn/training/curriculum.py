"""Curriculum scheduling helpers for online vision training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable


class CurriculumScheduler:
    """Track per-sample confidence and replay easy samples before hard ones."""

    def __init__(self, strategy: str = "easy_first"):
        self.strategy = str(strategy)
        self.difficulty_scores: dict[Hashable, float] = {}

    def score_sample(self, sample_id: Hashable, ccc_confidence: float) -> None:
        self.difficulty_scores[sample_id] = float(ccc_confidence)

    def order_samples(self, sample_ids: Iterable[Hashable]) -> list[Hashable]:
        if self.strategy != "easy_first":
            return list(sample_ids)
        return sorted(sample_ids, key=lambda sample_id: -self.difficulty_scores.get(sample_id, 0.0))


__all__ = ["CurriculumScheduler"]
