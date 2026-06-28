"""Multimodal stream composition and synthetic paired-data utilities."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Iterator

import torch

from bioarn.data.base import DataSample, StreamingDataSource
from bioarn.multimodal.fusion import MultimodalInput


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


class SyntheticMultimodalStream:
    """Generate paired vision, audio, and temporal samples with shared labels."""

    DEFAULT_LABELS = ("horizontal", "vertical", "diagonal", "box")
    BASE_FREQUENCIES = (220.0, 330.0, 440.0, 550.0, 660.0, 770.0)

    def __init__(
        self,
        num_samples: int,
        *,
        num_classes: int = 4,
        image_size: int = 16,
        sample_rate: int = 8000,
        duration_ms: int = 250,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(max(1, num_samples))
        self.num_classes = int(max(2, min(num_classes, len(self.DEFAULT_LABELS))))
        self.image_size = int(max(8, image_size))
        self.sample_rate = int(max(1000, sample_rate))
        self.duration_ms = int(max(50, duration_ms))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.labels = self.DEFAULT_LABELS[: self.num_classes]

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[MultimodalInput]:
        return self.stream()

    def _generator(self, seed_offset: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + seed_offset)

    def _label_index(self, label: str | int) -> int:
        if isinstance(label, int):
            return int(label) % self.num_classes
        if label not in self.labels:
            raise ValueError(f"Unknown multimodal label: {label}")
        return self.labels.index(label)

    def _visual_pattern(self, class_id: int, *, variant: int) -> torch.Tensor:
        size = self.image_size
        image = torch.zeros(size, size, dtype=torch.float32)
        if class_id == 0:
            image[size // 2 - 1 : size // 2 + 1, 2 : size - 2] = 1.0
        elif class_id == 1:
            image[2 : size - 2, size // 2 - 1 : size // 2 + 1] = 1.0
        elif class_id == 2:
            idx = torch.arange(2, size - 2)
            image[idx, idx] = 1.0
        else:
            image[2, 2 : size - 2] = 1.0
            image[size - 3, 2 : size - 2] = 1.0
            image[2 : size - 2, 2] = 1.0
            image[2 : size - 2, size - 3] = 1.0

        noise = 0.02 * torch.randn(image.shape, generator=self._generator(variant + (class_id * 101)))
        return (image + noise).clamp_(0.0, 1.0)

    def _audio_pattern(self, class_id: int, *, variant: int) -> torch.Tensor:
        seconds = self.duration_ms / 1000.0
        total_samples = int(round(self.sample_rate * seconds))
        time = torch.linspace(0.0, seconds, total_samples, dtype=torch.float32)
        base = self.BASE_FREQUENCIES[class_id % len(self.BASE_FREQUENCIES)]
        harmonic = base * (1.0 + (0.25 * ((class_id % 3) + 1)))
        phase = (class_id + 1) * 0.35
        envelope = 0.55 + (0.45 * torch.sin((2.0 * math.pi * (class_id + 1) * time) + phase).abs())
        waveform = (
            torch.sin(2.0 * math.pi * base * time)
            + (0.35 * torch.sin(2.0 * math.pi * harmonic * time + phase))
        )
        waveform = waveform * envelope
        noise = 0.01 * torch.randn(waveform.shape, generator=self._generator(variant + (class_id * 211)))
        waveform = waveform + noise
        waveform = waveform / waveform.abs().max().clamp_min(1e-6)
        return waveform.to(torch.float32)

    def _temporal_context(self, class_id: int, *, variant: int) -> list[int]:
        base = class_id * 3
        shift = variant % 3
        return [base + shift, base + 1 + shift, (base + 2 + shift) % max(self.num_classes * 4, 8)]

    def build_sample(self, label: str | int, *, variant: int = 0) -> MultimodalInput:
        class_id = self._label_index(label)
        label_name = self.labels[class_id]
        return MultimodalInput(
            vision=self._visual_pattern(class_id, variant=variant),
            audio=self._audio_pattern(class_id, variant=variant),
            temporal_context=self._temporal_context(class_id, variant=variant),
            metadata={
                "label": label_name,
                "class_id": class_id,
                "dataset": "synthetic-multimodal",
                "variant": int(variant),
            },
        )

    def sample_for_label(self, label: str | int, *, variant: int = 0) -> MultimodalInput:
        return self.build_sample(label, variant=variant)

    def stream(self) -> Iterator[MultimodalInput]:
        labels = [index % self.num_classes for index in range(self.num_samples)]
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=self._generator()).tolist()
            labels = [labels[index] for index in order]
        for sample_index, class_id in enumerate(labels):
            yield self.build_sample(class_id, variant=sample_index)


__all__ = ["MultimodalStream", "SyntheticMultimodalStream"]
