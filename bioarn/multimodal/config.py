"""Configuration for Bio-ARN multimodal fusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultimodalConfig:
    """Hyperparameters for cross-modal binding in shared concept space."""

    vision_dim: int = 784
    language_dim: int = 256
    concept_dim: int = 128
    cross_modal_strength: float = 0.5
    temporal_window: int = 5
    max_description_length: int = 20
    alignment_threshold: float = 0.3


__all__ = ["MultimodalConfig"]
