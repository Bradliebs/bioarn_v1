"""Streaming training utilities for cross-modal fusion."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F

from bioarn.config import MultimodalTrainConfig
from bioarn.data.multimodal import SyntheticMultimodalStream
from bioarn.multimodal.fusion import MultimodalFusionEngine, MultimodalInput


@dataclass(frozen=True)
class MultimodalExample:
    """Lightweight training example compatible with ``MultimodalInput``."""

    vision: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    temporal_context: list[int] | None = None
    label: str | None = None
    metadata: dict[str, Any] | None = None

    def to_input(self) -> MultimodalInput:
        metadata = dict(self.metadata or {})
        if self.label is not None:
            metadata.setdefault("label", self.label)
        return MultimodalInput(
            vision=None if self.vision is None else self.vision.detach().clone(),
            audio=None if self.audio is None else self.audio.detach().clone(),
            temporal_context=None if self.temporal_context is None else list(self.temporal_context),
            metadata=metadata,
        )


@dataclass
class MultimodalTrainingResult:
    """Compact training summary returned by ``train_online``."""

    num_samples: int
    num_passes: int
    mean_agreement: float
    mean_confidence: float
    label_consistency: float
    precision: float
    num_associations: int
    winner_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_samples": int(self.num_samples),
            "num_passes": int(self.num_passes),
            "mean_agreement": float(self.mean_agreement),
            "mean_confidence": float(self.mean_confidence),
            "label_consistency": float(self.label_consistency),
            "precision": float(self.precision),
            "num_associations": int(self.num_associations),
            "winner_histogram": dict(self.winner_histogram),
        }


def _iter_inputs(
    stream: Iterable[MultimodalInput | MultimodalExample] | object,
) -> Iterator[MultimodalInput]:
    if hasattr(stream, "stream"):
        yield from _iter_inputs(stream.stream())  # type: ignore[misc]
        return
    for item in stream:  # type: ignore[operator]
        if isinstance(item, MultimodalInput):
            yield item
        elif isinstance(item, MultimodalExample):
            yield item.to_input()
        else:
            raise ValueError("Multimodal training expects MultimodalInput or MultimodalExample items.")


class MultimodalTrainer:
    """Train the fusion engine on paired multimodal samples."""

    def __init__(self, config: MultimodalTrainConfig):
        self.config = copy.deepcopy(config)
        self.engine = MultimodalFusionEngine(self.config.fusion)
        self._label_prototypes: defaultdict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self._label_counts: Counter[tuple[str, str]] = Counter()

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if float(flattened.norm().item()) <= 1e-8:
            return torch.zeros_like(flattened)
        return F.normalize(flattened.unsqueeze(0), dim=1).squeeze(0)

    def _update_label_prototype(self, label: str, modality: str, concept: torch.Tensor) -> None:
        normalized = self._normalize(concept)
        key = (label, modality)
        count = self._label_counts.get(key, 0)
        existing = self._label_prototypes[label].get(modality)
        if existing is None:
            self._label_prototypes[label][modality] = normalized
        else:
            updated = ((existing * count) + normalized) / float(count + 1)
            self._label_prototypes[label][modality] = self._normalize(updated)
        self._label_counts[key] = count + 1

    def _label_consistency(self) -> float:
        similarities: list[float] = []
        for prototypes in self._label_prototypes.values():
            if len(prototypes) < 2:
                continue
            modalities = list(prototypes.keys())
            for left_index, left_modality in enumerate(modalities[:-1]):
                left = prototypes[left_modality]
                for right_modality in modalities[left_index + 1 :]:
                    right = prototypes[right_modality]
                    similarity = float(
                        F.cosine_similarity(
                            left.unsqueeze(0).to(right),
                            right.unsqueeze(0),
                        ).item()
                    )
                    similarities.append(max(0.0, min(1.0, 0.5 * (similarity + 1.0))))
        return float(sum(similarities) / len(similarities)) if similarities else 0.0

    def train_online(
        self,
        stream: Iterable[MultimodalInput | MultimodalExample] | object | None = None,
    ) -> dict[str, object]:
        """Train on streaming multimodal samples."""

        source = stream or SyntheticMultimodalStream(
            self.config.num_samples,
            num_classes=self.config.num_classes,
            image_size=self.config.vision_size,
            sample_rate=self.config.fusion.audio.sample_rate,
            duration_ms=self.config.fusion.audio.max_duration_ms,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        examples = list(_iter_inputs(source))
        agreements: list[float] = []
        confidences: list[float] = []

        for pass_index in range(self.config.num_passes):
            order = list(range(len(examples)))
            if self.config.shuffle:
                order = torch.randperm(
                    len(examples),
                    generator=torch.Generator().manual_seed(self.config.seed + pass_index),
                ).tolist()
            for sample_index in order:
                sample = examples[sample_index]
                self.engine.learn(
                    sample,
                    learning_rate_multiplier=self.config.learning_rate_multiplier,
                )
                output = self.engine.last_output
                if output is None:
                    continue

                label = None
                if sample.metadata is not None:
                    raw_label = sample.metadata.get("label")
                    if raw_label is not None:
                        label = str(raw_label)
                if label is not None:
                    if sample.vision is not None and "vision" in output.per_modality:
                        result = output.per_modality["vision"]
                        if result.concept_direction is not None:
                            self._update_label_prototype(label, "vision", result.concept_direction)
                    if sample.audio is not None and "audio" in output.per_modality:
                        result = output.per_modality["audio"]
                        if result.concept_direction is not None:
                            self._update_label_prototype(label, "audio", result.concept_direction)
                    if sample.temporal_context and "temporal" in output.per_modality:
                        result = output.per_modality["temporal"]
                        if result.concept_direction is not None:
                            self._update_label_prototype(label, "temporal", result.concept_direction)

                agreements.append(float(output.cross_modal_agreement))
                confidences.append(float(output.confidence))

        stats = self.engine.stats
        result = MultimodalTrainingResult(
            num_samples=len(examples),
            num_passes=int(self.config.num_passes),
            mean_agreement=float(sum(agreements) / len(agreements)) if agreements else 0.0,
            mean_confidence=float(sum(confidences) / len(confidences)) if confidences else 0.0,
            label_consistency=self._label_consistency(),
            precision=float(stats.get("precision", 1.0)),
            num_associations=int(stats.get("associations", {}).get("num_associations", 0)),
            winner_histogram=dict(stats.get("winner_counts", {})),
        )
        return result.to_dict()


__all__ = [
    "MultimodalExample",
    "MultimodalTrainer",
    "MultimodalTrainingResult",
]
