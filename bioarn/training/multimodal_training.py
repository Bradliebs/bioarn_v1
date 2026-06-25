"""Interleaved multimodal training on a shared CCC pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from bioarn.multimodal import ModalityAligner, MultimodalConfig, MultimodalFusion


@dataclass(frozen=True)
class MultimodalExample:
    """Paired multimodal supervision for a single concept."""

    vision: Tensor
    text: str | Tensor | Iterable[int]
    label: str | None = None


@dataclass
class MultimodalTrainingResult:
    """Summary metrics for multimodal training or evaluation."""

    num_pairs: int
    num_steps: int
    epochs: int
    committed_cccs: int
    shared_cccs: int
    concept_sharing_ratio: float
    mean_association_strength: float
    cross_modal_retrieval_accuracy: float
    mean_reciprocal_rank: float
    converged_pairs: int
    modality_sequence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _StepObservation:
    modality: str
    ccc_id: int
    confidence: float


class MultimodalTrainer:
    """Train shared visual and text concepts by alternating paired samples."""

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        fusion: MultimodalFusion | None = None,
    ) -> None:
        self.config = config or MultimodalConfig()
        self.fusion = fusion or MultimodalFusion(self.config)
        self.aligner = ModalityAligner(self.fusion, self.config)
        self.shared_label_to_ccc: dict[str, int] = {}

    def train(
        self,
        examples: Iterable[MultimodalExample | tuple[Tensor, str | Tensor | Iterable[int]] | tuple[Tensor, str | Tensor | Iterable[int], str]],
        *,
        epochs: int = 1,
        start_modality: str = "vision",
    ) -> MultimodalTrainingResult:
        dataset = self._coerce_examples(examples)
        if not dataset:
            return self._build_result([], epochs=0, modality_sequence=(), converged_pairs=0)

        first_modality = self._canonical_modality(start_modality)
        modality_sequence: list[str] = []
        converged_pairs = 0

        for _ in range(max(1, int(epochs))):
            for example in dataset:
                if first_modality == "vision":
                    first = self._process_vision(example)
                    second = self._process_text(example, anchor_ccc_id=first.ccc_id)
                    modality_sequence.extend(("vision", "text"))
                else:
                    first = self._process_text(example)
                    second = self._process_vision(example, anchor_ccc_id=first.ccc_id)
                    modality_sequence.extend(("text", "vision"))

                vision_step, text_step = (
                    (first, second) if first.modality == "vision" else (second, first)
                )
                self._bind_pair(vision_step, text_step)
                converged_pairs += int(vision_step.ccc_id == text_step.ccc_id)

        return self._build_result(
            dataset,
            epochs=max(1, int(epochs)),
            modality_sequence=tuple(modality_sequence),
            converged_pairs=converged_pairs,
        )

    def evaluate(
        self,
        examples: Iterable[MultimodalExample | tuple[Tensor, str | Tensor | Iterable[int]] | tuple[Tensor, str | Tensor | Iterable[int], str]],
    ) -> MultimodalTrainingResult:
        dataset = self._coerce_examples(examples)
        return self._build_result(dataset, epochs=0, modality_sequence=(), converged_pairs=0)

    @staticmethod
    def _canonical_modality(modality: str) -> str:
        lowered = modality.strip().lower()
        if lowered in {"vision", "visual", "image"}:
            return "vision"
        if lowered in {"text", "language", "linguistic"}:
            return "text"
        raise ValueError(f"Unsupported modality: {modality}")

    def _coerce_examples(
        self,
        examples: Iterable[MultimodalExample | tuple[Tensor, str | Tensor | Iterable[int]] | tuple[Tensor, str | Tensor | Iterable[int], str]],
    ) -> list[MultimodalExample]:
        dataset: list[MultimodalExample] = []
        for item in examples:
            if isinstance(item, MultimodalExample):
                dataset.append(item)
                continue
            if len(item) == 2:
                vision, text = item
                label = text.strip() if isinstance(text, str) else None
                dataset.append(MultimodalExample(vision=vision, text=text, label=label))
                continue
            if len(item) == 3:
                vision, text, label = item
                dataset.append(
                    MultimodalExample(
                        vision=vision,
                        text=text,
                        label=None if label is None else str(label).strip(),
                    )
                )
                continue
            raise ValueError("Each multimodal example must contain (vision, text) or (vision, text, label).")
        return dataset

    def _label_for(self, example: MultimodalExample) -> str | None:
        if example.label is not None and example.label.strip():
            return example.label.strip()
        if isinstance(example.text, str) and example.text.strip():
            return example.text.strip()
        return None

    def _recruit_new_ccc(
        self,
        *,
        modality: str,
        feature: Tensor,
        label: str | None,
        visual_input: Tensor | None = None,
    ) -> _StepObservation:
        ccc_id, confidence = self.fusion._recruit_ccc(feature, modality, label)
        if modality == "vision" and visual_input is not None:
            self.fusion._register_visual_pattern(ccc_id, visual_input)
        if label is not None:
            self.shared_label_to_ccc[label] = int(ccc_id)
        return _StepObservation(modality=modality, ccc_id=int(ccc_id), confidence=float(confidence))

    def _attach_to_ccc(
        self,
        ccc_id: int,
        *,
        modality: str,
        feature: Tensor,
        label: str | None,
        visual_input: Tensor | None = None,
    ) -> _StepObservation:
        existing = self.fusion.feature_prototypes.get((int(ccc_id), modality))
        confidence = self.fusion._cosine(feature, existing) if existing is not None else 1.0
        self.fusion.ccc_modalities[int(ccc_id)].add(modality)
        self.fusion._register_feature(int(ccc_id), modality, feature)
        if modality == "vision" and visual_input is not None:
            self.fusion._register_visual_pattern(int(ccc_id), visual_input)
        self.fusion._register_label(int(ccc_id), modality, label)
        if label is not None:
            self.shared_label_to_ccc[label] = int(ccc_id)
        return _StepObservation(modality=modality, ccc_id=int(ccc_id), confidence=float(confidence))

    def _process_vision(
        self,
        example: MultimodalExample,
        *,
        anchor_ccc_id: int | None = None,
    ) -> _StepObservation:
        label = self._label_for(example)
        feature = self.fusion._encode_visual(example.vision)
        if anchor_ccc_id is not None:
            return self._attach_to_ccc(
                anchor_ccc_id,
                modality="vision",
                feature=feature,
                label=label,
                visual_input=example.vision,
            )
        if label is not None and label in self.shared_label_to_ccc:
            return self._attach_to_ccc(
                self.shared_label_to_ccc[label],
                modality="vision",
                feature=feature,
                label=label,
                visual_input=example.vision,
            )
        if label is not None:
            return self._recruit_new_ccc(
                modality="vision",
                feature=feature,
                label=label,
                visual_input=example.vision,
            )
        ccc_id, confidence = self.fusion._resolve_visual_ccc(example.vision, feature, label=None, learn=True)
        if ccc_id is None:
            raise RuntimeError("Failed to resolve a shared visual CCC.")
        return _StepObservation(modality="vision", ccc_id=int(ccc_id), confidence=float(confidence))

    def _process_text(
        self,
        example: MultimodalExample,
        *,
        anchor_ccc_id: int | None = None,
    ) -> _StepObservation:
        feature, decoded_text = self.fusion._encode_text(example.text)
        label = self._label_for(example) or decoded_text or None
        if anchor_ccc_id is not None:
            return self._attach_to_ccc(
                anchor_ccc_id,
                modality="text",
                feature=feature,
                label=label,
            )
        if label is not None and label in self.shared_label_to_ccc:
            return self._attach_to_ccc(
                self.shared_label_to_ccc[label],
                modality="text",
                feature=feature,
                label=label,
            )
        if label is not None:
            return self._recruit_new_ccc(modality="text", feature=feature, label=label)
        ccc_id, confidence = self.fusion._resolve_ccc(feature, "text", label=None, learn=True)
        if ccc_id is None:
            raise RuntimeError("Failed to resolve a shared text CCC.")
        return _StepObservation(modality="text", ccc_id=int(ccc_id), confidence=float(confidence))

    def _bind_pair(self, vision_step: _StepObservation, text_step: _StepObservation) -> float:
        timestep = self.fusion.timestep
        activation_map = {
            int(vision_step.ccc_id): float(vision_step.confidence),
            int(text_step.ccc_id): max(float(text_step.confidence), float(vision_step.confidence))
            if int(text_step.ccc_id) == int(vision_step.ccc_id)
            else float(text_step.confidence),
        }
        self.fusion._activate(list(activation_map.items()), timestep)
        strength = float(self.config.cross_modal_strength) + (0.5 * (vision_step.confidence + text_step.confidence))
        strength = max(0.05, strength)
        if int(vision_step.ccc_id) == int(text_step.ccc_id):
            ccc_id = int(vision_step.ccc_id)
            self.fusion.ccc_modalities[ccc_id].update({"vision", "text"})
            self.fusion.explicit_bindings[(ccc_id, ccc_id)] = self.fusion.explicit_bindings.get((ccc_id, ccc_id), 0.0) + strength
            self.fusion.fabric._add_association(ccc_id, ccc_id, strength, temporal=False)  # noqa: SLF001
        else:
            self.fusion.bind_visual_to_text(int(vision_step.ccc_id), int(text_step.ccc_id), strength=strength)
        self.fusion.timestep += 1
        return strength

    def _build_result(
        self,
        dataset: list[MultimodalExample],
        *,
        epochs: int,
        modality_sequence: tuple[str, ...],
        converged_pairs: int,
    ) -> MultimodalTrainingResult:
        evaluation_pairs = [
            (example.vision, example.text, label)
            if label is not None
            else (example.vision, example.text)
            for example in dataset
            for label in [self._label_for(example)]
        ]
        alignment = self.aligner.measure_alignment(evaluation_pairs)
        pool_stats = self.fusion.ccc_pool.get_pool_stats()
        committed_cccs = int(pool_stats["num_committed"])
        shared_cccs = sum(
            1
            for modalities in self.fusion.ccc_modalities.values()
            if "vision" in modalities and "text" in modalities
        )
        return MultimodalTrainingResult(
            num_pairs=len(dataset),
            num_steps=len(modality_sequence),
            epochs=int(epochs),
            committed_cccs=committed_cccs,
            shared_cccs=shared_cccs,
            concept_sharing_ratio=(shared_cccs / committed_cccs) if committed_cccs else 0.0,
            mean_association_strength=float(alignment.mean_association_strength),
            cross_modal_retrieval_accuracy=float(alignment.retrieval_accuracy),
            mean_reciprocal_rank=float(alignment.mean_reciprocal_rank),
            converged_pairs=int(converged_pairs),
            modality_sequence=modality_sequence,
        )


__all__ = [
    "MultimodalExample",
    "MultimodalTrainer",
    "MultimodalTrainingResult",
]
