"""Curriculum scheduling utilities for streamed data."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch

from bioarn.data.base import DataSample, StreamingDataSource


class _OrderedStream(StreamingDataSource):
    def __init__(self, samples: list[DataSample], *, device=None) -> None:
        super().__init__(device=device)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def stream(self) -> Iterator[DataSample]:
        for sample in self._samples:
            cloned = sample.data.clone()
            yield DataSample(
                data=self._move_tensor(cloned),
                label=sample.label,
                modality=sample.modality,
                metadata=dict(sample.metadata),
            )


class CurriculumScheduler:
    """Control the order and difficulty of training samples."""

    @staticmethod
    def easy_first(data_source: StreamingDataSource) -> StreamingDataSource:
        samples = CurriculumScheduler._materialize(data_source)
        samples.sort(key=CurriculumScheduler._difficulty)
        return _OrderedStream(samples, device=data_source.device)

    @staticmethod
    def hard_last(data_source: StreamingDataSource) -> StreamingDataSource:
        return CurriculumScheduler.easy_first(data_source)

    @staticmethod
    def novelty_seeking(data_source: StreamingDataSource, system) -> StreamingDataSource:
        samples = CurriculumScheduler._materialize(data_source)

        def score(sample: DataSample) -> float:
            if "novelty" in sample.metadata:
                return float(sample.metadata["novelty"])
            for attribute in ("prediction_error", "compute_novelty", "estimate_novelty"):
                if hasattr(system, attribute):
                    value = getattr(system, attribute)(sample.data)
                    if isinstance(value, torch.Tensor):
                        return float(value.detach().float().mean().item())
                    return float(value)
            return float(torch.linalg.vector_norm(sample.data.float()).item())

        samples.sort(key=score, reverse=True)
        return _OrderedStream(samples, device=data_source.device)

    @staticmethod
    def class_incremental(data_source: StreamingDataSource, classes_per_stage: int) -> StreamingDataSource:
        if classes_per_stage <= 0:
            raise ValueError("classes_per_stage must be positive")

        grouped: dict[int, list[DataSample]] = defaultdict(list)
        unlabeled: list[DataSample] = []
        for sample in data_source.stream():
            if sample.label is None:
                unlabeled.append(sample)
            else:
                grouped[int(sample.label)].append(sample)

        ordered: list[DataSample] = []
        labels = sorted(grouped)
        for stage_index, offset in enumerate(range(0, len(labels), classes_per_stage)):
            stage_labels = labels[offset : offset + classes_per_stage]
            for label in stage_labels:
                for sample in grouped[label]:
                    sample.metadata = {**sample.metadata, "curriculum_stage": stage_index}
                    ordered.append(sample)
        ordered.extend(unlabeled)
        return _OrderedStream(ordered, device=data_source.device)

    @staticmethod
    def _materialize(data_source: StreamingDataSource) -> list[DataSample]:
        return [
            DataSample(
                data=sample.data.detach().clone(),
                label=sample.label,
                modality=sample.modality,
                metadata=dict(sample.metadata),
            )
            for sample in data_source.stream()
        ]

    @staticmethod
    def _difficulty(sample: DataSample) -> float:
        if "difficulty" in sample.metadata:
            return float(sample.metadata["difficulty"])
        if "confidence" in sample.metadata:
            return 1.0 - float(sample.metadata["confidence"])
        return float(sample.data.float().std().item())


__all__ = ["CurriculumScheduler"]
