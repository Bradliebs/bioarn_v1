"""Configuration for dual character+word language processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordLevelConfig:
    """Hyperparameters for word-level language processing."""

    max_vocabulary: int = 1000
    word_ccc_pool_size: int = 500
    word_concept_dim: int = 128
    word_spike_dim: int = 256
    word_transition_decay: float = 0.999
    min_word_frequency: int = 2
    trie_max_completions: int = 10
    word_constraint_strength: float = 0.7
    enable_word_suggestions: bool = True


__all__ = ["WordLevelConfig"]
