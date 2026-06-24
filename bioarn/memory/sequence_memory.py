"""Enhanced sequence memory with replay, chunking, and predictive retrieval."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn.functional as F

from bioarn.config import SDMConfig
from bioarn.memory.config import SequenceMemoryConfig
from bioarn.memory.sdm import SparseDistributedMemory


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.detach().reshape(-1).to(torch.float32)
    norm = float(flat.norm().item())
    if norm <= 1e-8:
        return torch.zeros_like(flat)
    return flat / norm


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if float(left_norm.norm().item()) == 0.0 or float(right_norm.norm().item()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(left_norm.unsqueeze(0), right_norm.unsqueeze(0)).item())


class TransitionMatrix:
    """Track P(next_concept | current_concept) as a sparse matrix."""

    def __init__(self, max_concepts: int = 2000, decay: float = 1.0):
        if max_concepts <= 0:
            raise ValueError("max_concepts must be positive.")
        self.max_concepts = int(max_concepts)
        self.decay = float(decay)
        self.counts: dict[int, Counter[int]] = {}
        self.totals: Counter[int] = Counter()

    def _valid_id(self, concept_id: int) -> bool:
        return 0 <= int(concept_id) < self.max_concepts

    def _apply_decay(self, from_id: int) -> None:
        if self.decay >= 0.999999:
            return
        row = self.counts.get(int(from_id))
        if not row:
            return
        updated = Counter()
        total = 0.0
        for to_id, count in row.items():
            decayed = float(count) * self.decay
            if decayed >= 1e-6:
                updated[int(to_id)] = decayed
                total += decayed
        if updated:
            self.counts[int(from_id)] = updated
            self.totals[int(from_id)] = total
            return
        self.counts.pop(int(from_id), None)
        self.totals.pop(int(from_id), None)

    def record_transition(self, from_id: int, to_id: int) -> None:
        self.record_weighted_transition(from_id, to_id, weight=1.0)

    def record_weighted_transition(self, from_id: int, to_id: int, weight: float = 1.0) -> None:
        if weight <= 0.0 or not self._valid_id(from_id) or not self._valid_id(to_id):
            return
        self._apply_decay(int(from_id))
        row = self.counts.setdefault(int(from_id), Counter())
        row[int(to_id)] += float(weight)
        self.totals[int(from_id)] = float(self.totals.get(int(from_id), 0.0)) + float(weight)

    def predict_next(self, current_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        row = self.counts.get(int(current_id))
        if not row:
            return []
        total = float(self.totals.get(int(current_id), sum(row.values())))
        ranked = sorted(row.items(), key=lambda item: float(item[1]), reverse=True)
        return [
            (int(token_id), float(count) / max(total, 1e-6))
            for token_id, count in ranked[: max(1, int(top_k))]
        ]

    def get_chain(self, start_id: int, length: int) -> list[int]:
        if length <= 0:
            return []
        chain = [int(start_id)]
        current = int(start_id)
        while len(chain) < int(length):
            predictions = self.predict_next(current, top_k=1)
            if not predictions:
                break
            current = int(predictions[0][0])
            chain.append(current)
        return chain


@dataclass
class _ReplayItem:
    sequence: list[int]
    surprise: float = 0.0


class ReplayBuffer:
    """Store recent sequences and replay them to strengthen memory."""

    def __init__(self, buffer_size: int = 1000, replay_ratio: int = 3):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative.")
        self.buffer_size = int(buffer_size)
        self.replay_ratio = int(replay_ratio)
        self.buffer: deque[_ReplayItem] = deque(maxlen=self.buffer_size)
        self._priority_order: list[_ReplayItem] = []

    def store(self, sequence: list[int]) -> None:
        """Store a sequence for later replay."""

        copied = [int(token_id) for token_id in sequence]
        if len(copied) < 2:
            return
        self.buffer.append(_ReplayItem(sequence=copied))

    def prioritized_replay(self, prediction_errors: list[float]) -> list[list[int]]:
        """Replay sequences where prediction was worst (most surprising)."""

        if not self.buffer:
            self._priority_order = []
            return []
        recent_items = list(self.buffer)
        errors = [float(value) for value in prediction_errors[-len(recent_items) :]]
        if errors:
            if len(errors) < len(recent_items):
                errors = ([0.0] * (len(recent_items) - len(errors))) + errors
            for item, error in zip(recent_items, errors, strict=False):
                item.surprise = max(item.surprise, float(error))
        self._priority_order = sorted(
            recent_items,
            key=lambda item: (float(item.surprise), len(item.sequence)),
            reverse=True,
        )
        return [item.sequence.copy() for item in self._priority_order[: max(1, self.replay_ratio)]]

    def replay(
        self,
        sdm: SparseDistributedMemory,
        transition: TransitionMatrix,
        concept_resolver: Callable[[int], torch.Tensor | None] | None = None,
    ) -> int:
        """Replay stored sequences to strengthen memories."""

        replay_items = self._priority_order or list(self.buffer)
        replays = 0
        for item in replay_items:
            for _ in range(max(0, self.replay_ratio)):
                for from_id, to_id in zip(item.sequence[:-1], item.sequence[1:], strict=False):
                    transition.record_transition(int(from_id), int(to_id))
                    from_concept = self._resolve_concept(int(from_id), sdm, concept_resolver)
                    to_concept = self._resolve_concept(int(to_id), sdm, concept_resolver)
                    if from_concept is None or to_concept is None:
                        continue
                    sdm.associate(
                        from_concept,
                        to_concept,
                        from_concept,
                        to_concept,
                        temporal_order=True,
                    )
                replays += 1
        self._priority_order = []
        return replays

    @staticmethod
    def _resolve_concept(
        token_id: int,
        sdm: SparseDistributedMemory,
        concept_resolver: Callable[[int], torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        if concept_resolver is not None:
            concept = concept_resolver(int(token_id))
            if concept is not None and float(concept.norm().item()) > 0.0:
                concept = _normalize(concept)
                if concept.numel() > sdm.data_dim:
                    return _normalize(concept[: sdm.data_dim])
                if concept.numel() < sdm.data_dim:
                    return _normalize(F.pad(concept, (0, sdm.data_dim - concept.numel())))
                return concept

        size = max(1, int(sdm.data_dim))
        concept = torch.zeros(size, dtype=torch.float32)
        concept[int(token_id) % size] = 1.0
        concept[(int(token_id) * 7 + 3) % size] = 0.5
        return _normalize(concept)


class ChunkLibrary:
    """Discover and store common subsequences as single chunks."""

    def __init__(self, min_frequency: int = 5, max_chunk_length: int = 8, max_chunks: int = 500):
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive.")
        if max_chunk_length < 2:
            raise ValueError("max_chunk_length must be at least 2.")
        self.min_frequency = int(min_frequency)
        self.max_chunk_length = int(max_chunk_length)
        self.max_chunks = int(max_chunks)
        self.chunk_to_id: dict[str, int] = {}
        self.id_to_chunk: dict[int, str] = {}
        self.chunk_frequencies: Counter[str] = Counter()
        self._tokenizer = None

    def bind_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def learn_chunks(self, text: str) -> None:
        """Discover frequent subsequences via byte pair encoding principle."""

        if not text:
            return
        counts: Counter[str] = Counter()
        for length in range(2, self.max_chunk_length + 1):
            if len(text) < length:
                continue
            for index in range(len(text) - length + 1):
                chunk = text[index : index + length]
                if not chunk.strip():
                    continue
                counts[chunk] += 1
        candidates = [
            (chunk, frequency)
            for chunk, frequency in counts.items()
            if frequency >= self.min_frequency
        ]
        candidates.sort(
            key=lambda item: (item[1] * max(1, len(item[0]) - 1), len(item[0]), item[1]),
            reverse=True,
        )
        self.chunk_to_id.clear()
        self.id_to_chunk.clear()
        self.chunk_frequencies.clear()
        for chunk, frequency in candidates[: self.max_chunks]:
            chunk_id = len(self.chunk_to_id)
            self.chunk_to_id[chunk] = chunk_id
            self.id_to_chunk[chunk_id] = chunk
            self.chunk_frequencies[chunk] = int(frequency)

    def encode_with_chunks(self, sequence: list[int]) -> list[int | tuple[int, ...]]:
        """Replace known chunks in sequence with chunk IDs."""

        if not sequence or not self.chunk_to_id:
            return [int(token_id) for token_id in sequence]
        encoded: list[int | tuple[int, ...]] = []
        index = 0
        while index < len(sequence):
            best_match: tuple[int, ...] | None = None
            upper = min(len(sequence), index + self.max_chunk_length)
            for end in range(upper, index + 1, -1):
                window = tuple(int(token_id) for token_id in sequence[index:end])
                if self._decode_sequence(window) in self.chunk_to_id:
                    best_match = window
                    break
            if best_match is None:
                encoded.append(int(sequence[index]))
                index += 1
                continue
            encoded.append(best_match)
            index += len(best_match)
        return encoded

    def decode_chunks(self, chunked_sequence) -> list[int]:
        """Expand chunks back to individual tokens."""

        decoded: list[int] = []
        for item in chunked_sequence:
            if isinstance(item, tuple):
                decoded.extend(int(token_id) for token_id in item)
                continue
            decoded.append(int(item))
        return decoded

    def predict_next(self, context: list[int], top_k: int = 5) -> list[tuple[int, float]]:
        """Predict the next token from learned chunk completions."""

        if not context or not self.chunk_frequencies:
            return []
        context_text = self._decode_sequence(context)
        scores: Counter[int] = Counter()
        for chunk, frequency in self.chunk_frequencies.items():
            max_prefix = min(len(context_text), len(chunk) - 1)
            for prefix_length in range(max_prefix, 0, -1):
                if not context_text.endswith(chunk[:prefix_length]):
                    continue
                next_fragment = chunk[prefix_length]
                token_id = self._encode_token(next_fragment)
                if token_id is not None:
                    scores[int(token_id)] += float(frequency) * (prefix_length / len(chunk))
                break
        if not scores:
            return []
        total = float(sum(scores.values()))
        ranked = scores.most_common(max(1, int(top_k)))
        return [(token_id, float(value) / max(total, 1e-6)) for token_id, value in ranked]

    def learned_chunks(self, top_k: int = 10) -> list[tuple[str, int]]:
        return self.chunk_frequencies.most_common(max(1, int(top_k)))

    def _decode_token(self, token_id: int) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode([int(token_id)])
        if 0 <= int(token_id) < 256:
            return chr(int(token_id))
        return str(int(token_id))

    def _decode_sequence(self, sequence) -> str:
        return "".join(self._decode_token(int(token_id)) for token_id in sequence)

    def _encode_token(self, token: str) -> int | None:
        if self._tokenizer is not None:
            return int(self._tokenizer.vocab.get_id(token))
        if len(token) == 1:
            return ord(token)
        return None


class PredictiveRetrieval:
    """Combine multiple retrieval sources with prediction error weighting."""

    def __init__(self, weights: dict[str, float] | None = None):
        weights = weights or {
            "sdm": 0.3,
            "transition": 0.4,
            "ngram": 0.2,
            "chunk": 0.1,
        }
        self.base_weights = {name: float(value) for name, value in weights.items()}
        self.source_accuracy = {name: 1.0 for name in self.base_weights}
        self.momentum = 0.9

    def update_source_accuracy(self, actual_token_id: int, sources: dict[str, list[tuple[int, float]]]) -> None:
        for name, predictions in sources.items():
            if name not in self.source_accuracy or not predictions:
                continue
            predicted = int(max(predictions, key=lambda item: float(item[1]))[0])
            target = 1.0 if predicted == int(actual_token_id) else 0.0
            current = float(self.source_accuracy[name])
            self.source_accuracy[name] = (self.momentum * current) + ((1.0 - self.momentum) * target)

    def effective_weights(self) -> dict[str, float]:
        effective: dict[str, float] = {}
        for name, base_weight in self.base_weights.items():
            accuracy = float(self.source_accuracy.get(name, 1.0))
            effective[name] = max(0.0, base_weight) * max(0.2, accuracy)
        return effective

    def combine_scores(
        self,
        sources: dict[str, list[tuple[int, float]]],
        candidate_ids: list[int] | None = None,
    ) -> dict[int, float]:
        aggregate: Counter[int] = Counter()
        support = [int(token_id) for token_id in (candidate_ids or [])]
        weights = self.effective_weights()
        for name, predictions in sources.items():
            if not predictions:
                continue
            weight = float(weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            total = float(sum(max(0.0, float(score)) for _, score in predictions))
            if total <= 0.0:
                continue
            for token_id, score in predictions:
                aggregate[int(token_id)] += weight * (max(0.0, float(score)) / total)
        for token_id in support:
            aggregate.setdefault(int(token_id), 0.0)
        total = float(sum(aggregate.values()))
        if total <= 0.0:
            if not support:
                return {}
            uniform = 1.0 / len(support)
            return {int(token_id): uniform for token_id in support}
        return {int(token_id): float(score) / total for token_id, score in aggregate.items()}

    def retrieve_next(self, context: list[int], sources: dict[str, list[tuple[int, float]]]) -> tuple[int, float]:
        del context
        combined = self.combine_scores(sources)
        if not combined:
            return -1, 0.0
        best_token, best_score = max(combined.items(), key=lambda item: float(item[1]))
        return int(best_token), float(best_score)


@dataclass
class SequenceEnsembleResult:
    """Ensemble prediction details for text generation."""

    candidate_ids: list[int]
    probabilities: torch.Tensor
    retrieved_concept: torch.Tensor
    predicted_token_id: int
    confidence: float
    source_confidences: dict[str, float]
    source_predictions: dict[str, list[tuple[int, float]]]


class SequenceMemory:
    """Enhanced sequence memory with hippocampal-like replay and chunking.

    Combines:
    - SDM for raw storage (fast, one-shot)
    - Transition matrix for frequent patterns (slow, consolidated)
    - Chunk library for common subsequences
    - Replay buffer for consolidation
    """

    def __init__(self, config: SequenceMemoryConfig, sdm: SparseDistributedMemory | None = None):
        self.config = config
        if sdm is None:
            sdm_config = SDMConfig(
                address_dim=int(config.sdm_content_dim),
                hamming_radius=max(1, int(config.sdm_content_dim * 0.42)),
                num_hard_locations=int(config.sdm_addresses),
                data_dim=int(config.sdm_content_dim),
                decay_rate=float(config.transition_decay),
                stdp_window=max(4, int(config.max_chunk_length * 2)),
            )
            self.sdm = SparseDistributedMemory(sdm_config)
        else:
            self.sdm = sdm
        self.transition_matrix = TransitionMatrix(
            max_concepts=int(config.max_concepts),
            decay=float(config.transition_decay),
        )
        self.replay_buffer = ReplayBuffer(
            buffer_size=int(config.replay_buffer_size),
            replay_ratio=int(config.replay_ratio),
        )
        self.chunk_library = ChunkLibrary(
            min_frequency=int(config.min_chunk_frequency),
            max_chunk_length=int(config.max_chunk_length),
            max_chunks=int(config.chunk_vocab_size),
        )
        self.predictive_retrieval = PredictiveRetrieval(
            weights={
                "sdm": float(config.sdm_weight),
                "transition": float(config.transition_weight),
                "ngram": float(config.ngram_weight),
                "chunk": float(config.chunk_weight),
            }
        )
        self.token_counts: Counter[int] = Counter()
        self.token_concept_sums: dict[int, torch.Tensor] = {}
        self.token_concept_counts: Counter[int] = Counter()
        self._tokenizer = None
        self.replay_events = 0

    def bind_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.chunk_library.bind_tokenizer(tokenizer)

    def record_token(self, token_id: int, concept: torch.Tensor | None = None) -> None:
        token_id = int(token_id)
        self.token_counts[token_id] += 1
        if concept is None:
            return
        normalized = _normalize(concept)
        self.token_concept_counts[token_id] += 1
        if token_id not in self.token_concept_sums:
            self.token_concept_sums[token_id] = normalized.detach().clone()
            return
        self.token_concept_sums[token_id] = self.token_concept_sums[token_id] + normalized

    def token_prototype(self, token_id: int) -> torch.Tensor | None:
        token_id = int(token_id)
        count = int(self.token_concept_counts.get(token_id, 0))
        if count <= 0 or token_id not in self.token_concept_sums:
            return None
        return _normalize(self.token_concept_sums[token_id] / float(count))

    def record_transition(
        self,
        from_id: int,
        to_id: int,
        from_concept: torch.Tensor | None = None,
        to_concept: torch.Tensor | None = None,
    ) -> None:
        self.transition_matrix.record_transition(int(from_id), int(to_id))
        if from_concept is None:
            from_concept = self.token_prototype(int(from_id))
        if to_concept is None:
            to_concept = self.token_prototype(int(to_id))
        if from_concept is None or to_concept is None:
            return
        if float(from_concept.norm().item()) == 0.0 or float(to_concept.norm().item()) == 0.0:
            return
        from_aligned = self._align_to_sdm_dim(from_concept)
        to_aligned = self._align_to_sdm_dim(to_concept)
        self.sdm.associate(
            from_aligned,
            to_aligned,
            from_aligned,
            to_aligned,
            temporal_order=True,
        )

    def store_sequence(self, sequence: list[int], prediction_error: float = 0.0) -> None:
        self.replay_buffer.store(sequence)
        if self.replay_buffer.buffer:
            self.replay_buffer.buffer[-1].surprise = max(
                float(self.replay_buffer.buffer[-1].surprise),
                float(prediction_error),
            )

    def learn_chunks(self, text: str) -> None:
        self.chunk_library.learn_chunks(text)

    def maybe_replay(self, step_count: int, prediction_errors: list[float]) -> int:
        interval = max(1, int(self.config.replay_interval))
        if step_count <= 0 or step_count % interval != 0:
            return 0
        if self.config.prioritize_surprising:
            self.replay_buffer.prioritized_replay(prediction_errors)
        replayed = self.replay_buffer.replay(
            self.sdm,
            self.transition_matrix,
            concept_resolver=self.token_prototype,
        )
        self.replay_events += int(replayed > 0)
        return replayed

    def score_candidates(
        self,
        context: list[int],
        candidate_ids: list[int],
        token_prototypes: dict[int, torch.Tensor],
        *,
        context_text: str,
        ngram_cache=None,
        temperature: float = 1.0,
        fallback_concept: torch.Tensor | None = None,
    ) -> SequenceEnsembleResult:
        if not candidate_ids:
            empty = torch.empty(0, dtype=torch.float32)
            return SequenceEnsembleResult(
                candidate_ids=[],
                probabilities=empty,
                retrieved_concept=torch.zeros(self.sdm.data_dim, dtype=torch.float32),
                predicted_token_id=-1,
                confidence=0.0,
                source_confidences={"sdm": 0.0, "transition": 0.0, "ngram": 0.0, "chunk": 0.0},
                source_predictions={"sdm": [], "transition": [], "ngram": [], "chunk": []},
            )

        sources: dict[str, list[tuple[int, float]]] = {
            "sdm": [],
            "transition": [],
            "ngram": [],
            "chunk": [],
        }
        retrieved_concept = self._retrieve_from_sdm(context, token_prototypes, fallback_concept)
        if float(retrieved_concept.norm().item()) > 0.0:
            sdm_scores: list[tuple[int, float]] = []
            for token_id in candidate_ids:
                prototype = token_prototypes.get(int(token_id))
                if prototype is None:
                    prototype = self.token_prototype(int(token_id))
                if prototype is None:
                    continue
                score = max(0.0, (1.0 + _cosine(retrieved_concept, self._align_to_sdm_dim(prototype))) * 0.5)
                if score > 0.0:
                    sdm_scores.append((int(token_id), score))
            sdm_scores.sort(key=lambda item: float(item[1]), reverse=True)
            sources["sdm"] = sdm_scores[: min(len(sdm_scores), max(8, len(candidate_ids)))]

        current_id = int(context[-1]) if context else -1
        transition_predictions = self.transition_matrix.predict_next(current_id, top_k=max(8, len(candidate_ids)))
        if not transition_predictions and self.token_counts:
            total = float(sum(self.token_counts.values()))
            transition_predictions = [
                (int(token_id), float(count) / max(total, 1e-6))
                for token_id, count in self.token_counts.most_common(max(1, len(candidate_ids)))
            ]
        sources["transition"] = transition_predictions

        if ngram_cache is not None and context_text:
            char_predictions = ngram_cache.predict_next(context_text, top_k=min(8, max(1, len(candidate_ids))))
            sources["ngram"] = self._char_predictions_to_ids(char_predictions)

        sources["chunk"] = self.chunk_library.predict_next(context, top_k=min(8, max(1, len(candidate_ids))))
        combined = self.predictive_retrieval.combine_scores(sources, candidate_ids=candidate_ids)

        base = torch.tensor(
            [float(combined.get(int(token_id), 0.0)) for token_id in candidate_ids],
            dtype=torch.float32,
        )
        if float(base.sum().item()) <= 0.0:
            base = torch.full((len(candidate_ids),), 1.0 / len(candidate_ids), dtype=torch.float32)
        logits = torch.log(base.clamp_min(1e-6)) / max(0.05, float(temperature))
        probabilities = torch.softmax(logits, dim=0)

        best_index = int(torch.argmax(probabilities).item())
        source_confidences = {
            name: float(max((score for _, score in predictions), default=0.0))
            for name, predictions in sources.items()
        }
        return SequenceEnsembleResult(
            candidate_ids=[int(token_id) for token_id in candidate_ids],
            probabilities=probabilities.detach().clone(),
            retrieved_concept=retrieved_concept.detach().clone(),
            predicted_token_id=int(candidate_ids[best_index]),
            confidence=float(probabilities[best_index].item()),
            source_confidences=source_confidences,
            source_predictions=sources,
        )

    def update_prediction_feedback(
        self,
        actual_token_id: int,
        source_predictions: dict[str, list[tuple[int, float]]],
    ) -> None:
        self.predictive_retrieval.update_source_accuracy(int(actual_token_id), source_predictions)

    def _retrieve_from_sdm(
        self,
        context: list[int],
        token_prototypes: dict[int, torch.Tensor],
        fallback_concept: torch.Tensor | None,
    ) -> torch.Tensor:
        query_parts: list[torch.Tensor] = []
        for token_id in context[-4:]:
            prototype = token_prototypes.get(int(token_id))
            if prototype is None:
                prototype = self.token_prototype(int(token_id))
            if prototype is not None and float(prototype.norm().item()) > 0.0:
                query_parts.append(self._align_to_sdm_dim(prototype))
        if not query_parts and fallback_concept is not None and float(fallback_concept.norm().item()) > 0.0:
            query_parts.append(self._align_to_sdm_dim(fallback_concept))
        if not query_parts:
            return torch.zeros(self.sdm.data_dim, dtype=torch.float32)
        query = _normalize(torch.stack(query_parts).mean(dim=0))
        retrieved = self.sdm.retrieve_associates(query)
        if float(retrieved.norm().item()) == 0.0:
            return query
        return _normalize(retrieved)

    def _char_predictions_to_ids(self, predictions: list[tuple[str, float]]) -> list[tuple[int, float]]:
        converted: list[tuple[int, float]] = []
        if self._tokenizer is None:
            return converted
        for token, probability in predictions:
            token_id = int(self._tokenizer.vocab.get_id(token))
            converted.append((token_id, float(probability)))
        return converted

    def _align_to_sdm_dim(self, vector: torch.Tensor) -> torch.Tensor:
        normalized = _normalize(vector)
        if normalized.numel() == self.sdm.data_dim:
            return normalized
        if normalized.numel() > self.sdm.data_dim:
            return _normalize(normalized[: self.sdm.data_dim])
        return _normalize(F.pad(normalized, (0, self.sdm.data_dim - normalized.numel())))


__all__ = [
    "ChunkLibrary",
    "PredictiveRetrieval",
    "ReplayBuffer",
    "SequenceEnsembleResult",
    "SequenceMemory",
    "TransitionMatrix",
]
