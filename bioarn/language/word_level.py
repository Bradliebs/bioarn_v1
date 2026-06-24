"""Word-level language processing built on Bio-ARN CCC primitives."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import torch

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core import CCCPool
from bioarn.memory import TransitionMatrix
from bioarn.tokenization import SpikeTokenEncoder

from .config import WordLevelConfig
from .word_trie import WordTrie

_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[.!?]")
_SEPARATORS = {" ", ".", ",", "!", "?", ";", ":", "\n", "\t"}


class WordLevelProcessor:
    """Process text at word level using dedicated CCC pool."""

    def __init__(self, config: WordLevelConfig):
        self.config = config
        self.trie = WordTrie()
        self.word_counts: Counter[str] = Counter()
        self.vocabulary: list[str] = []
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.word_encoder: SpikeTokenEncoder | None = None
        self.word_ccc_pool = self._build_ccc_pool()
        self.word_transition_matrix = TransitionMatrix(
            max_concepts=max(4, int(self.config.max_vocabulary)),
            decay=float(self.config.word_transition_decay),
        )
        self.word_to_ccc_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self.ccc_to_word_counts: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self.word_concept_sums: dict[str, torch.Tensor] = {}
        self.word_concept_counts: Counter[str] = Counter()
        self.start_word_counts: Counter[str] = Counter()
        self.sentence_end_counts: Counter[str] = Counter()
        self.sentence_occurrences: Counter[str] = Counter()

    def learn_vocabulary(self, text: str) -> None:
        """Extract a frequent vocabulary and train CCCs on whole-word patterns."""

        words = self._extract_words(text)
        self.word_counts = Counter(words)
        ranked = [
            (word, count)
            for word, count in self.word_counts.most_common(int(self.config.max_vocabulary))
            if count >= int(self.config.min_word_frequency)
        ]
        if len(ranked) < int(self.config.max_vocabulary):
            seen = {word for word, _ in ranked}
            for word, count in self.word_counts.most_common(int(self.config.max_vocabulary)):
                if word in seen:
                    continue
                ranked.append((word, count))
                seen.add(word)
                if len(ranked) >= int(self.config.max_vocabulary):
                    break

        self.trie = WordTrie()
        self.vocabulary = [word for word, _ in ranked]
        self.word_to_id = {word: index for index, word in enumerate(self.vocabulary)}
        self.id_to_word = {index: word for word, index in self.word_to_id.items()}
        for word, count in ranked:
            self.trie.insert(word, weight=int(count))

        if not self.vocabulary:
            self.word_encoder = None
            return

        self.word_encoder = SpikeTokenEncoder(
            vocab_size=len(self.vocabulary),
            spike_dim=int(self.config.word_spike_dim),
            num_timesteps=4,
        )
        self.word_ccc_pool = self._build_ccc_pool()
        self.word_to_ccc_counts.clear()
        self.ccc_to_word_counts.clear()
        self.word_concept_sums = {}
        self.word_concept_counts = Counter()
        self.start_word_counts = Counter()
        self.sentence_end_counts = Counter()
        self.sentence_occurrences = Counter()

        previous_word: str | None = None
        start_of_sentence = True
        for token in _TOKEN_PATTERN.findall(text):
            if _WORD_PATTERN.fullmatch(token):
                word = token.lower()
                if word not in self.word_to_id:
                    continue
                if start_of_sentence:
                    self.start_word_counts[word] += 1
                self.sentence_occurrences[word] += 1
                previous_word = word
                start_of_sentence = False
                continue

            if token in {".", "!", "?"}:
                if previous_word is not None:
                    self.sentence_end_counts[previous_word] += 1
                previous_word = None
                start_of_sentence = True

        timestep = 0
        for word, count in ranked:
            pattern = self._word_pattern(word)
            with torch.no_grad():
                pool_output = self.word_ccc_pool(pattern, timestep=timestep)
            ccc_index = self._word_ccc_index(pool_output)
            if ccc_index is not None:
                weight = max(1, int(count))
                self.word_to_ccc_counts[word][ccc_index] += weight
                self.ccc_to_word_counts[ccc_index][word] += weight
                concept = self.word_ccc_pool.cccs[ccc_index].concept_direction.detach().clone()
                self.word_concept_counts[word] += 1
                self.word_concept_sums[word] = concept if word not in self.word_concept_sums else self.word_concept_sums[word] + concept
            timestep += 1

    def learn_word_transitions(self, text: str) -> None:
        """Learn P(next_word | current_word) from a text corpus."""

        size = max(4, len(self.vocabulary) or int(self.config.max_vocabulary))
        self.word_transition_matrix = TransitionMatrix(
            max_concepts=size,
            decay=float(self.config.word_transition_decay),
        )
        words = [word for word in self._extract_words(text) if word in self.word_to_id]
        for current_word, next_word in zip(words[:-1], words[1:], strict=False):
            self.word_transition_matrix.record_transition(
                self.word_to_id[current_word],
                self.word_to_id[next_word],
            )

    def constrain_generation(
        self,
        char_candidates: list[Any],
        partial_word: str,
    ) -> list[Any]:
        """Boost characters that continue a valid vocabulary prefix."""

        partial = self._normalize_partial(partial_word)
        completions = self.trie.get_completions(
            partial,
            top_k=max(4, int(self.config.trie_max_completions)),
        ) if partial or self.vocabulary else []
        next_chars = {
            word[len(partial)]
            for word in completions
            if len(word) > len(partial)
        }
        if self.trie.is_complete_word(partial):
            next_chars |= _SEPARATORS

        tuple_mode = bool(char_candidates and isinstance(char_candidates[0], tuple))
        scored: dict[str, float] = {}
        base_scores: dict[str, float] = {}

        def record(candidate: str, score: float) -> None:
            token = str(candidate)
            if not token:
                return
            if token not in scored or score > scored[token]:
                scored[token] = score
                base_scores[token] = score

        for item in char_candidates:
            candidate, base = item if isinstance(item, tuple) else (item, 1.0)
            char = str(candidate)[:1]
            base_score = float(base)
            boosted = base_score
            if char.lower() in next_chars:
                boosted += float(self.config.word_constraint_strength)
            elif next_chars and char.isalpha():
                boosted *= max(0.05, 1.0 - (0.85 * float(self.config.word_constraint_strength)))
            elif self.trie.is_complete_word(partial) and char in _SEPARATORS:
                boosted += float(self.config.word_constraint_strength) * 1.1
            record(char, boosted)

        if next_chars:
            backfill = 0.01 + (0.1 * float(self.config.word_constraint_strength))
            for char in sorted(next_chars):
                if char not in scored:
                    record(char, backfill)

        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        if tuple_mode:
            return [(char, float(score)) for char, score in ranked]
        return [char for char, _ in ranked]

    def suggest_next_word(self, context_words: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        """Suggest likely next words from the learned transition matrix."""

        if not self.vocabulary:
            return []
        if not self.config.enable_word_suggestions:
            return self._fallback_words(top_k)

        normalized = [self._normalize_partial(word) for word in context_words if self._normalize_partial(word)]
        if normalized:
            current = normalized[-1]
            token_id = self.word_to_id.get(current)
            if token_id is not None:
                predictions = self.word_transition_matrix.predict_next(token_id, top_k=max(1, int(top_k)))
                if predictions:
                    return [
                        (self.id_to_word[word_id], float(probability))
                        for word_id, probability in predictions
                        if word_id in self.id_to_word
                    ]
        if self.start_word_counts:
            total = float(sum(self.start_word_counts.values())) or 1.0
            ranked = self.start_word_counts.most_common(max(1, int(top_k)))
            return [(word, float(count) / total) for word, count in ranked]
        return self._fallback_words(top_k)

    def detect_word_boundary(self, generated_chars: str) -> bool:
        """Return True when the latest generated char ends a word."""

        if not generated_chars:
            return False
        return generated_chars[-1] in _SEPARATORS

    def word_concept(self, word: str) -> torch.Tensor | None:
        token = self._normalize_partial(word)
        count = int(self.word_concept_counts.get(token, 0))
        if count <= 0 or token not in self.word_concept_sums:
            return None
        concept = self.word_concept_sums[token] / float(count)
        norm = float(concept.norm().item())
        if norm <= 1e-8:
            return None
        return (concept / norm).detach().clone()

    def sentence_end_probability(self, word: str) -> float:
        token = self._normalize_partial(word)
        total = float(self.sentence_occurrences.get(token, 0))
        if total <= 0.0:
            return 0.0
        return float(self.sentence_end_counts.get(token, 0) / total)

    def _build_ccc_pool(self) -> CCCPool:
        ccc_config = CCCConfig(
            input_dim=int(self.config.word_spike_dim),
            concept_dim=int(self.config.word_concept_dim),
            num_f1_features=max(32, int(self.config.word_concept_dim // 2)),
            f1_top_k=max(8, int(self.config.word_concept_dim // 8)),
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=int(self.config.word_ccc_pool_size),
        )
        margin_config = MarginGateConfig(
            theta_margin=0.03,
            theta_margin_lr=0.002,
            theta_resonance=0.45,
        )
        return CCCPool(ccc_config, margin_config)

    def _extract_words(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text or "")]

    def _normalize_partial(self, partial_word: str) -> str:
        matches = _WORD_PATTERN.findall(str(partial_word).lower())
        if not matches:
            return ""
        return matches[-1]

    def _word_pattern(self, word: str) -> torch.Tensor:
        if self.word_encoder is None:
            raise ValueError("Word vocabulary has not been learned yet.")
        token_id = self.word_to_id[word]
        return self.word_encoder.encode_token(token_id).to(torch.float32)

    def _word_ccc_index(self, pool_output) -> int | None:
        if pool_output.recruited_index is not None:
            return int(pool_output.recruited_index)
        winners = self.word_ccc_pool.get_winners(pool_output, k=1)
        if winners:
            return int(winners[0])
        if pool_output.fired_indices:
            return int(pool_output.fired_indices[0])
        return None

    def _fallback_words(self, top_k: int) -> list[tuple[str, float]]:
        if not self.word_counts:
            return []
        total = float(sum(self.word_counts[word] for word in self.vocabulary)) or 1.0
        return [
            (word, float(self.word_counts[word]) / total)
            for word in self.vocabulary[: max(1, int(top_k))]
        ]


__all__ = ["WordLevelProcessor"]
