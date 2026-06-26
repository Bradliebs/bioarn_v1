"""Multimodal stream composition utilities."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator

from bioarn.data.base import DataSample, StreamingDataSource


class MultimodalStream(StreamingDataSource):
    """Interleave vision and language streams."""

    def __init__(
        self,
        vision_source: StreamingDataSource,
        language_source: StreamingDataSource,
        ratio: float = 0.5,
        device=None,
    ) -> None:
        super().__init__(device=device)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be between 0 and 1")
        self.vision_source = vision_source
        self.language_source = language_source
        self.ratio = ratio
        self._schedule = self._build_schedule(ratio)

    @staticmethod
    def _build_schedule(ratio: float) -> tuple[str, ...]:
        if ratio == 0.0:
            return ("language",)
        if ratio == 1.0:
            return ("vision",)
        fraction = Fraction(ratio).limit_denominator(16)
        vision_steps = max(1, fraction.numerator)
        language_steps = max(1, fraction.denominator - fraction.numerator)
        return ("vision",) * vision_steps + ("language",) * language_steps

    def __len__(self) -> int:
        return len(self.vision_source) + len(self.language_source)

    def stream(self) -> Iterator[DataSample]:
        vision_iter = iter(self.vision_source.stream())
        language_iter = iter(self.language_source.stream())
        vision_exhausted = False
        language_exhausted = False
        schedule_index = 0

        while not (vision_exhausted and language_exhausted):
            target = self._schedule[schedule_index % len(self._schedule)]
            schedule_index += 1

            if target == "vision" and not vision_exhausted:
                try:
                    sample = next(vision_iter)
                    sample.metadata = {**sample.metadata, "source_stream": "vision"}
                    yield sample
                    continue
                except StopIteration:
                    vision_exhausted = True

            if target == "language" and not language_exhausted:
                try:
                    sample = next(language_iter)
                    sample.metadata = {**sample.metadata, "source_stream": "language"}
                    yield sample
                    continue
                except StopIteration:
                    language_exhausted = True

            if not vision_exhausted:
                try:
                    sample = next(vision_iter)
                    sample.metadata = {**sample.metadata, "source_stream": "vision"}
                    yield sample
                    continue
                except StopIteration:
                    vision_exhausted = True

            if not language_exhausted:
                try:
                    sample = next(language_iter)
                    sample.metadata = {**sample.metadata, "source_stream": "language"}
                    yield sample
                except StopIteration:
                    language_exhausted = True


__all__ = ["MultimodalStream"]
