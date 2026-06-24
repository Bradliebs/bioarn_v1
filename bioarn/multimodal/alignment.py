"""Cross-modal alignment utilities for Bio-ARN."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from bioarn.data.base import DataSample

from .config import MultimodalConfig
from .fusion import MultimodalFusion


@dataclass
class AlignmentMetrics:
    """Evaluation summary for multimodal alignment."""

    retrieval_accuracy: float
    mean_reciprocal_rank: float
    mean_association_strength: float
    num_pairs: int


class ModalityAligner:
    """Align visual and language representations in shared concept space."""

    def __init__(self, fusion: MultimodalFusion, config: MultimodalConfig | None = None):
        self.fusion = fusion
        self.config = config or fusion.config

    def align_by_label(self, vision_stream, language_stream, labels):
        """Create supervised alignments from paired labels."""

        bindings = []
        for visual_input, text_input, label in zip(vision_stream, language_stream, labels, strict=False):
            bindings.append(self.fusion.learn_cross_modal(visual_input, text_input, label=str(label)))
        return bindings

    def align_by_co_occurrence(self, multimodal_stream):
        """Create unsupervised alignments from near-synchronous co-occurrence."""

        samples = multimodal_stream.stream() if hasattr(multimodal_stream, "stream") else iter(multimodal_stream)
        recent_visual: list[tuple[int, torch.Tensor]] = []
        recent_text: list[tuple[int, str | torch.Tensor]] = []
        bindings = []

        for step, sample in enumerate(samples):
            if isinstance(sample, DataSample):
                modality = self.fusion._canonical_modality(sample.modality)
                payload = sample.data
                if modality == "text":
                    payload = sample.metadata.get("text", payload)
            elif isinstance(sample, tuple) and len(sample) == 2:
                modality, payload = sample
                modality = self.fusion._canonical_modality(str(modality))
            else:
                raise ValueError("Co-occurrence alignment expects DataSample or (modality, payload) pairs.")

            if modality == "vision":
                recent_visual.append((step, payload))
                recent_visual = [
                    (timestamp, value)
                    for timestamp, value in recent_visual
                    if step - timestamp <= int(self.config.temporal_window)
                ]
                for timestamp, text_payload in recent_text:
                    delta = step - timestamp
                    if delta < 0 or delta > int(self.config.temporal_window):
                        continue
                    binding = self.fusion.learn_cross_modal(payload, text_payload, label=None)
                    closeness = math.exp(-delta / max(1.0, float(self.config.temporal_window)))
                    self.fusion.bind_visual_to_text(
                        binding["visual_ccc_id"],
                        binding["text_ccc_id"],
                        strength=float(self.config.cross_modal_strength) * closeness,
                    )
                    bindings.append(binding)
            else:
                recent_text.append((step, payload))
                recent_text = [
                    (timestamp, value)
                    for timestamp, value in recent_text
                    if step - timestamp <= int(self.config.temporal_window)
                ]
                for timestamp, visual_payload in recent_visual:
                    delta = step - timestamp
                    if delta < 0 or delta > int(self.config.temporal_window):
                        continue
                    binding = self.fusion.learn_cross_modal(visual_payload, payload, label=None)
                    closeness = math.exp(-delta / max(1.0, float(self.config.temporal_window)))
                    self.fusion.bind_visual_to_text(
                        binding["visual_ccc_id"],
                        binding["text_ccc_id"],
                        strength=float(self.config.cross_modal_strength) * closeness,
                    )
                    bindings.append(binding)
        return bindings

    def measure_alignment(self, test_pairs) -> AlignmentMetrics:
        """Measure cross-modal retrieval quality on paired image/text examples."""

        pairs = list(test_pairs)
        total = 0
        correct = 0
        reciprocal_ranks: list[float] = []
        strengths: list[float] = []

        for pair in pairs:
            if len(pair) == 3:
                image, text, label = pair
                expected_label = str(label)
            elif len(pair) == 2:
                image, text = pair
                expected_label = text if isinstance(text, str) else ""
            else:
                raise ValueError("Each test pair must contain (image, text) or (image, text, label).")

            vision_matches = self.fusion.cross_modal_retrieval(image, "vision", "text")
            if vision_matches:
                strengths.append(float(vision_matches[0].strength))
            total += 1
            if vision_matches and vision_matches[0].label == expected_label:
                correct += 1
            for rank, match in enumerate(vision_matches, start=1):
                if match.label == expected_label:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

            text_matches = self.fusion.cross_modal_retrieval(text, "text", "vision")
            total += 1
            if text_matches:
                strengths.append(float(text_matches[0].strength))
            visual_ccc = self.fusion.label_to_ccc.get(("vision", expected_label))
            if text_matches and visual_ccc is not None and int(text_matches[0].target_ccc_id) == int(visual_ccc):
                correct += 1
            for rank, match in enumerate(text_matches, start=1):
                if visual_ccc is not None and int(match.target_ccc_id) == int(visual_ccc):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return AlignmentMetrics(
            retrieval_accuracy=float(correct / total) if total else 0.0,
            mean_reciprocal_rank=float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            mean_association_strength=float(sum(strengths) / len(strengths)) if strengths else 0.0,
            num_pairs=len(pairs),
        )


__all__ = ["AlignmentMetrics", "ModalityAligner"]
