"""Base interfaces for streaming online-learning data pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


@dataclass(slots=True)
class DataSample:
    """A single online-learning sample."""

    data: torch.Tensor
    label: int | None
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataBatch:
    """A mini-batch of streamed samples."""

    data: torch.Tensor
    labels: torch.Tensor | None
    modality: str
    batch_size: int


class StreamingDataSource(ABC):
    """Base class for online learning data sources."""

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device) if device is not None else None

    @abstractmethod
    def stream(self) -> Iterator[DataSample]:
        """Yield samples one at a time (online learning)."""

    @abstractmethod
    def __len__(self) -> int:
        """Total samples available."""

    def stream_batched(self, batch_size: int) -> Iterator[DataBatch]:
        """Yield mini-batches for vectorized CCC processing."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        samples: list[DataSample] = []
        for sample in self.stream():
            samples.append(sample)
            if len(samples) == batch_size:
                yield self._pack_batch(samples)
                samples = []

        if samples:
            yield self._pack_batch(samples)

    def _pack_batch(self, samples: list[DataSample]) -> DataBatch:
        modalities = {sample.modality for sample in samples}
        modality = next(iter(modalities)) if len(modalities) == 1 else "mixed"
        data = torch.stack([sample.data for sample in samples], dim=0)
        if self.device is not None:
            data = data.to(self.device)

        if any(sample.label is None for sample in samples):
            labels = None
        else:
            labels = torch.tensor([int(sample.label) for sample in samples], dtype=torch.long)
            if self.device is not None:
                labels = labels.to(self.device)

        return DataBatch(
            data=data,
            labels=labels,
            modality=modality,
            batch_size=len(samples),
        )

    def _move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            return tensor
        return tensor.to(self.device)


__all__ = ["DataBatch", "DataSample", "StreamingDataSource"]
