"""Configuration for enhanced sequence memory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SequenceMemoryConfig:
    """Hyperparameters for replay-driven sequence memory."""

    sdm_addresses: int = 20000
    sdm_content_dim: int = 256

    max_concepts: int = 2000
    transition_decay: float = 0.999

    replay_buffer_size: int = 500
    replay_ratio: int = 3
    replay_interval: int = 100
    prioritize_surprising: bool = True

    min_chunk_frequency: int = 5
    max_chunk_length: int = 6
    chunk_vocab_size: int = 500

    sdm_weight: float = 0.3
    transition_weight: float = 0.4
    ngram_weight: float = 0.2
    chunk_weight: float = 0.1


__all__ = ["SequenceMemoryConfig"]
