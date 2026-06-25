"""Online text-generation training for Bio-ARN."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import re
import statistics
import time
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F

from bioarn.config import (
    BioARNConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    PredictiveConfig,
    RewardConfig,
    SDMConfig,
    SpikingConfig,
)
from bioarn.data.language import CharacterStream
from bioarn.generation import (
    BeamSearchDecoder,
    GenerationQualityMetrics,
    NGramCache,
    QualityReport,
    RepetitionPenalty,
)
from bioarn.language import DualLevelProcessor, WordLevelConfig, WordLevelProcessor
from bioarn.loop import SensorimotorLoop
from bioarn.memory import SequenceMemory, SequenceMemoryConfig
from bioarn.tokenization import BPETokenizer, CharTokenizer, SpikeTokenEncoder
from bioarn.workspace import RecurrentContext


_BUILTIN_PASSAGES = [
    "The cat sat on the mat. The dog ran in the park. The bird sang in the tree. ",
    "Once upon a time, a small fox found a bright key and wondered what door it might open. ",
    '"Hello there," said Mira. "Hello back," said Jon, and both of them laughed by the warm fire. ',
    "The river moved slowly under the bridge, and the lamps along the road made long gold lines on the water. ",
    "A baker in the town mixed flour and sugar, then sang a quiet song while the bread rose in the night. ",
    "The old clock in the hall ticked and ticked, marking patient time while pages turned and feet crossed the wooden floor. ",
    "In the garden, green leaves bent in the wind. In the kitchen, bright cups waited in a neat row. ",
    "There was a lantern by the gate, and there was a letter by the lantern, and there was a promise in the letter. ",
    "Night came softly. Morning came clearly. The town slept, the town woke, and the town began its songs again. ",
    "Rain on the roof made a gentle drum. Rain on the path made a silver line. Rain on the window made a soft song. ",
    "A little boat rocked in the bay. A little bell rang in the yard. A little lamp glowed in the room. ",
    "The kind baker gave bread to the child. The child gave thanks. The baker smiled and tied the ribbon on the warm loaf. ",
    "The moon was bright and the stars were small. The sky was clear and the road was still. ",
    "Tea on the table, bread on the plate, light at the window, and kind words at the gate. ",
    "Sing a quiet song, walk a little way, turn the page, close the door, and start the day again. ",
]


def build_builtin_corpus(min_chars: int = 12000) -> str:
    """Return a built-in repetitive English corpus for local text training."""

    if min_chars <= 0:
        raise ValueError("min_chars must be positive.")
    repeated = "".join(_BUILTIN_PASSAGES)
    multiplier = max(2, math.ceil(min_chars / max(1, len(repeated))))
    corpus = "".join(repeated for _ in range(multiplier))
    return corpus[: max(min_chars, 10000)]


@dataclass
class TextGenConfig:
    tokenizer_type: str = "char"
    vocab_size: int = 256
    context_length: int = 64
    spike_dim: int = 256
    num_timesteps: int = 8
    max_pool_size: int = 2000
    temperature: float = 1.0
    learning_rate_hebbian: float = 0.01
    sdm_addresses: int = 5000
    generate_max_tokens: int = 200
    num_passes: int = 1
    beam_width: int = 5
    frequency_boost: float = 0.45
    repetition_penalty: float = 1.18
    repetition_window: int = 24
    use_contextual_patterns: bool = True
    enable_ngram_cache: bool = True
    use_word_level: bool = True
    word_level: WordLevelConfig = field(default_factory=WordLevelConfig)
    sequence_memory: SequenceMemoryConfig = field(default_factory=SequenceMemoryConfig)


@dataclass
class TrainingMetrics:
    num_samples: int
    tokens_processed: int
    concepts_learned: int
    concepts_recruited: int
    memory_utilization: float
    mean_learning_rate: float
    mean_prediction_error: float
    unique_tokens_seen: int
    elapsed_seconds: float
    concepts_trace: list[int] = field(default_factory=list)
    memory_trace: list[float] = field(default_factory=list)
    learning_rate_trace: list[float] = field(default_factory=list)
    mean_pattern_frequency: float = 0.0
    num_passes: int = 1


@dataclass
class GenerationMetrics:
    perplexity: float
    prediction_accuracy: float
    diversity: float
    coherence: float
    average_confidence: float
    num_samples: int
    generated_examples: list[str] = field(default_factory=list)
    quality_report: QualityReport | None = None


@dataclass
class _Observation:
    token_id: int
    concept: torch.Tensor
    ccc_index: int | None
    confidence: float
    recruited: bool
    prediction_error: float
    learning_rate: float


@dataclass
class _Prediction:
    token_id: int
    confidence: float
    retrieved_concept: torch.Tensor
    reconstructed_spike: torch.Tensor
    candidate_ids: list[int]
    probabilities: torch.Tensor
    sdm_confidence: float
    margin_confidence: float
    transition_confidence: float
    ngram_confidence: float
    chunk_confidence: float
    ensemble_confidence: float
    source_predictions: dict[str, list[tuple[int, float]]]


class TextGenerationTrainer:
    """Train Bio-ARN for text generation using online Hebbian learning."""

    def __init__(self, config: TextGenConfig, system: SensorimotorLoop | None = None):
        self.config = config
        self.system = system or self._build_system(config)
        self.tokenizer = self._build_tokenizer(config)
        self._tokenizer_ready = config.tokenizer_type == "char"
        self.spike_encoder = SpikeTokenEncoder(
            vocab_size=self._tokenizer_vocab_size(),
            spike_dim=self.config.spike_dim,
            num_timesteps=self.config.num_timesteps,
        )

        self.token_counts: Counter[int] = Counter()
        self.transition_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.token_concept_sums: dict[int, torch.Tensor] = {}
        self.token_concept_counts: Counter[int] = Counter()
        self.token_to_ccc_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.ccc_to_token_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.window_counts: Counter[str] = Counter()
        self.window_to_ccc_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self.training_steps = 0
        self._max_window_frequency = 1
        self._latest_corpus = ""

        self.ngram_cache = NGramCache(max_n=max(4, min(8, int(self.config.context_length))))
        self.quality_metrics = GenerationQualityMetrics()
        self.repetition_penalty = RepetitionPenalty(
            penalty=self.config.repetition_penalty,
            window=self.config.repetition_window,
        )
        self.decoder = BeamSearchDecoder(beam_width=self.config.beam_width, length_penalty=0.6)
        self.sequence_memory = SequenceMemory(
            self.config.sequence_memory,
            sdm=self.system.core.fabric.sdm,
        )
        self.sequence_memory.bind_tokenizer(self.tokenizer)
        self.word_processor = (
            WordLevelProcessor(self.config.word_level)
            if self.config.use_word_level and self.config.tokenizer_type.lower() == "char"
            else None
        )
        self.dual_processor = (
            DualLevelProcessor(self, self.word_processor)
            if self.word_processor is not None
            else None
        )

        self._runtime_token_history: list[int] = []
        self._runtime_concept_history: list[torch.Tensor] = []
        self.recurrent_context = RecurrentContext(
            context_dim=int(self.system.core.config.ccc.concept_dim),
            integration_rate=float(self.system.core.gnw.config.recurrent_integration_rate),
        )
        self.enable_generation_context = True
        self._last_context_utilization = 0.0
        self._last_context_repetition = 0.0
        self._last_topic_drift = 0.0
        self._refresh_generation_helpers()

    @staticmethod
    def _build_tokenizer(config: TextGenConfig) -> CharTokenizer | BPETokenizer:
        tokenizer_type = config.tokenizer_type.lower()
        if tokenizer_type == "char":
            return CharTokenizer()
        if tokenizer_type == "bpe":
            return BPETokenizer(vocab_size=config.vocab_size)
        raise ValueError("tokenizer_type must be 'char' or 'bpe'.")

    @staticmethod
    def _build_system(config: TextGenConfig) -> SensorimotorLoop:
        concept_dim = max(48, int(config.spike_dim))
        ccc_input_dim = max(48, int(config.spike_dim))
        spiking = SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0)
        system_config = BioARNConfig(
            spiking=spiking,
            ccc=CCCConfig(
                input_dim=ccc_input_dim,
                concept_dim=concept_dim,
                num_f1_features=max(32, concept_dim // 2),
                f1_top_k=max(8, concept_dim // 8),
                fast_lr=1.0,
                slow_lr=float(config.learning_rate_hebbian),
                feedback_lr=float(config.learning_rate_hebbian),
                max_pool_size=int(config.max_pool_size),
            ),
            margin_gate=MarginGateConfig(
                theta_margin=0.03,
                theta_margin_lr=0.002,
                theta_resonance=0.45,
            ),
            sdm=SDMConfig(
                address_dim=concept_dim,
                hamming_radius=max(1, int(concept_dim * 0.42)),
                num_hard_locations=int(config.sdm_addresses),
                data_dim=concept_dim,
                decay_rate=0.9995,
                stdp_window=max(6, int(config.context_length)),
            ),
            predictive=PredictiveConfig(
                num_levels=3,
                gamma=0.15,
                eta=0.02,
                precision_init=1.0,
                error_threshold=0.0,
            ),
            gnw=GNWConfig(
                capacity=max(3, min(9, max(1, config.context_length // 4))),
                broadcast_gain=2.0,
                fatigue_rate=0.05,
                fatigue_threshold=0.1,
                competition_temp=0.75,
                concept_dim=concept_dim,
                context_size=max(64, min(256, config.context_length * 4)),
                context_decay=0.97,
                context_eviction_threshold=0.04,
                context_update_rate=0.2,
                attention_heads=4,
                context_top_k=5,
                recurrent_integration_rate=0.12,
                context_bias_gain=0.4,
                repetition_window=max(12, int(config.context_length)),
                repetition_novelty_threshold=0.82,
            ),
            reward=RewardConfig(
                intrinsic_scale=1.0,
                novelty_threshold=1.75,
                novelty_boost=2.0,
                novelty_decay=0.9,
                curiosity_weight=0.35,
            ),
            seed=42,
        )
        return SensorimotorLoop(system_config)

    @property
    def _window_size(self) -> int:
        return max(2, min(8, int(self.config.context_length)))

    def _tokenizer_vocab_size(self) -> int:
        if hasattr(self.tokenizer, "vocab_size"):
            return int(self.tokenizer.vocab_size)
        return len(self.tokenizer.vocab)

    def _ensure_spike_encoder(self) -> None:
        vocab_size = self._tokenizer_vocab_size()
        if self.spike_encoder.vocab_size == vocab_size:
            return
        self.spike_encoder = SpikeTokenEncoder(
            vocab_size=vocab_size,
            spike_dim=self.config.spike_dim,
            num_timesteps=self.config.num_timesteps,
        )

    def _refresh_generation_helpers(self) -> None:
        self.ngram_cache.bind_tokenizer(self.tokenizer)
        self.sequence_memory.bind_tokenizer(self.tokenizer)
        self.decoder = BeamSearchDecoder(beam_width=self.config.beam_width, length_penalty=0.6)
        self.repetition_penalty = RepetitionPenalty(
            penalty=self.config.repetition_penalty,
            window=self.config.repetition_window,
        )

    def _fit_tokenizer_if_needed(self, sample_text: str) -> None:
        if self.config.tokenizer_type.lower() == "bpe" and not self._tokenizer_ready:
            self.tokenizer.train(sample_text, vocab_size=self.config.vocab_size)
            self._tokenizer_ready = True
        self._ensure_spike_encoder()
        self._refresh_generation_helpers()

    @property
    def _special_token_ids(self) -> set[int]:
        vocab = self.tokenizer.vocab
        return {
            vocab.get_id("<PAD>"),
            vocab.get_id("<UNK>"),
            vocab.get_id("<BOS>"),
        }

    @property
    def _eos_token_id(self) -> int:
        return int(self.tokenizer.vocab.get_id("<EOS>"))

    @property
    def generation_stop_token_ids(self) -> set[int]:
        return self._special_token_ids | {self._eos_token_id}

    def _iter_text_chunks(self, text_source: str | Path | CharacterStream | Iterable[str]) -> Iterator[str]:
        if isinstance(text_source, CharacterStream):
            yield from text_source._iter_text_chunks()  # noqa: SLF001
            return
        if isinstance(text_source, Path):
            with text_source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield line
            return
        if isinstance(text_source, str):
            looks_like_path = len(text_source) < 260 and "\n" not in text_source and "\r" not in text_source
            candidate = Path(text_source) if looks_like_path else None
            if candidate is not None and candidate.exists():
                with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        yield line
                return
            yield text_source
            return
        for chunk in text_source:
            yield str(chunk)

    def _materialize_text(self, text_source: str | Path | CharacterStream | Iterable[str], char_limit: int | None = None) -> str:
        parts: list[str] = []
        total = 0
        for chunk in self._iter_text_chunks(text_source):
            if char_limit is None:
                parts.append(chunk)
                continue
            remaining = max(0, int(char_limit) - total)
            if remaining <= 0:
                break
            parts.append(chunk[:remaining])
            total += len(parts[-1])
            if total >= int(char_limit):
                break
        return "".join(parts)

    def _base_token_pattern(self, token_id: int) -> torch.Tensor:
        pattern = self.spike_encoder.encode_token(int(token_id)).to(torch.float32)
        input_dim = int(self.system.core.config.ccc.input_dim)
        if pattern.numel() == input_dim:
            return pattern
        if pattern.numel() > input_dim:
            return pattern[:input_dim]
        return F.pad(pattern, (0, input_dim - pattern.numel()))

    def _contextual_pattern(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            return torch.zeros(self.system.core.config.ccc.input_dim, dtype=torch.float32)

        window_ids = [int(token_id) for token_id in token_ids[-self._window_size :]]
        blended = torch.zeros(self.config.spike_dim, dtype=torch.float32)
        total_weight = 0.0

        for index, token_id in enumerate(window_ids):
            base = self.spike_encoder.encode_token(token_id).to(torch.float32)
            shift = ((index + 1) * max(1, self.config.spike_dim // max(2, self._window_size))) % self.config.spike_dim
            rolled = torch.roll(base, shifts=shift)
            weight = 0.55 + (0.45 * ((index + 1) / len(window_ids)))
            blended += rolled * weight
            total_weight += weight
            if index > 0:
                previous = self.spike_encoder.encode_token(window_ids[index - 1]).to(torch.float32)
                pair = torch.roll(torch.maximum(previous, rolled), shifts=max(1, shift // 2))
                blended += pair * 0.2
                total_weight += 0.2

        contextual = blended / max(total_weight, 1e-6)
        last_base = self.spike_encoder.encode_token(window_ids[-1]).to(torch.float32)
        mixed = (0.65 * contextual) + (0.35 * last_base)

        input_dim = int(self.system.core.config.ccc.input_dim)
        if mixed.numel() == input_dim:
            return mixed
        if mixed.numel() > input_dim:
            return mixed[:input_dim]
        return F.pad(mixed, (0, input_dim - mixed.numel()))

    def _token_pattern(self, token_id: int, context_ids: Sequence[int] | None = None) -> torch.Tensor:
        if not self.config.use_contextual_patterns or not context_ids:
            return self._base_token_pattern(token_id)
        return self._contextual_pattern(list(context_ids)[-(self._window_size - 1) :] + [int(token_id)])

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        norm = float(flattened.norm().item())
        if norm <= 1e-8:
            return torch.zeros_like(flattened)
        return flattened / norm

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = TextGenerationTrainer._normalize(left)
        right_norm = TextGenerationTrainer._normalize(right)
        if float(left_norm.norm().item()) == 0.0 or float(right_norm.norm().item()) == 0.0:
            return 0.0
        return float(F.cosine_similarity(left_norm.unsqueeze(0), right_norm.unsqueeze(0)).item())

    def _reset_runtime_state(self, *, clear_workspace: bool, clear_temporal_buffer: bool) -> None:
        self._runtime_token_history = []
        self._runtime_concept_history = []
        self.recurrent_context.reset()
        self._last_context_utilization = 0.0
        self._last_context_repetition = 0.0
        self._last_topic_drift = 0.0
        self.system.language_encoder.reset_state()
        self.system.motor_stream.reset()
        self.system.hierarchy.reset()
        self.system.reward.reset()
        self.system._feedback_features.zero_()  # noqa: SLF001
        self.system._generated_token_history.clear()  # noqa: SLF001
        self.system._last_sensory = None  # noqa: SLF001
        self.system._last_prediction = None  # noqa: SLF001
        self.system._last_perception = None  # noqa: SLF001
        self.system._last_recognition = None  # noqa: SLF001
        self.system._last_attention = None  # noqa: SLF001
        self.system._last_plan = None  # noqa: SLF001
        self.system._last_action = None  # noqa: SLF001
        self.system._last_reward = None  # noqa: SLF001
        if clear_workspace:
            self.system.core.gnw.clear()
        if clear_temporal_buffer:
            self.system.core.fabric.activation_history = []
            self.system.core.fabric.temporal_associator.clear_buffer()

    def _extract_ccc_index(self, pool_output) -> int | None:
        if pool_output.recruited_index is not None:
            return int(pool_output.recruited_index)
        winners = self.system.core.ccc_pool.get_winners(pool_output, k=1)
        if winners:
            return int(winners[0])
        if pool_output.fired_indices:
            return int(pool_output.fired_indices[0])
        return None

    def _token_prototype(self, token_id: int) -> torch.Tensor | None:
        count = int(self.token_concept_counts.get(int(token_id), 0))
        prototype = self.token_concept_sums.get(int(token_id))
        if count <= 0 or prototype is None:
            return self.sequence_memory.token_prototype(int(token_id))
        return self._normalize(prototype / float(count))

    def _candidate_prototypes(self, candidate_ids: Sequence[int]) -> dict[int, torch.Tensor]:
        prototypes: dict[int, torch.Tensor] = {}
        for token_id in candidate_ids:
            prototype = self._token_prototype(int(token_id))
            if prototype is not None and float(prototype.norm().item()) > 0.0:
                prototypes[int(token_id)] = prototype
        return prototypes

    def _most_common_ccc_index(self, token_id: int) -> int | None:
        counts = self.token_to_ccc_counts.get(int(token_id))
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _most_common_window_ccc(self, window_text: str) -> int | None:
        counts = self.window_to_ccc_counts.get(window_text)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _record_runtime(self, token_id: int, concept: torch.Tensor, ccc_index: int | None, confidence: float) -> None:
        normalized = self._normalize(concept)
        self._runtime_token_history.append(int(token_id))
        self._runtime_concept_history.append(normalized.detach().clone())
        self.recurrent_context.integrate(normalized)
        workspace_index = int(ccc_index) if ccc_index is not None else -(int(token_id) + 1)
        self.system.core.gnw.inject(workspace_index, normalized, priority=max(0.1, float(confidence)))

    def _decode_token(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _window_text(self, token_ids: Sequence[int]) -> str:
        return self._decode_token([int(token_id) for token_id in token_ids if int(token_id) not in self.generation_stop_token_ids])

    def _update_memories(self, token_id: int, concept: torch.Tensor, ccc_index: int | None) -> None:
        token_id = int(token_id)
        normalized = self._normalize(concept)
        self.sequence_memory.record_token(token_id, normalized)
        self.token_counts[token_id] += 1
        self.token_concept_counts[token_id] += 1
        if token_id not in self.token_concept_sums:
            self.token_concept_sums[token_id] = normalized.detach().clone()
        else:
            self.token_concept_sums[token_id] = self.token_concept_sums[token_id] + normalized
        if ccc_index is not None:
            self.token_to_ccc_counts[token_id][int(ccc_index)] += 1
            self.ccc_to_token_counts[int(ccc_index)][token_id] += 1
        if len(self._runtime_token_history) >= 2:
            prev_token = int(self._runtime_token_history[-2])
            self.transition_counts[prev_token][token_id] += 1
            prev_concept = self._runtime_concept_history[-2] if len(self._runtime_concept_history) >= 2 else None
            self.sequence_memory.record_transition(prev_token, token_id, prev_concept, normalized)
            recent_window = self._runtime_token_history[-max(2, min(self.config.context_length, 12)) :]
            self.sequence_memory.store_sequence(recent_window)

    def _adapt_pool_thresholds(self) -> None:
        pool_stats = self.system.core.ccc_pool.get_pool_stats()
        fire_rate = float(pool_stats["fire_rate"])
        for ccc in self.system.core.ccc_pool.cccs:
            if bool(ccc.is_committed.item()):
                ccc.margin_gate.adapt_threshold(fire_rate)

    def _frequency_scale(self, window_text: str) -> float:
        if not window_text or self.config.frequency_boost <= 0.0:
            return 1.0
        frequency = int(self.window_counts.get(window_text, 0))
        if frequency <= 1:
            return 1.0
        relative = math.log1p(frequency) / max(math.log1p(self._max_window_frequency), 1.0)
        return 1.0 + (self.config.frequency_boost * relative)

    def _reinforce_pattern(self, scale: float, ccc_index: int | None, pattern: torch.Tensor, perception, concept: torch.Tensor, confidence: float) -> None:
        if ccc_index is None or scale <= 1.0:
            return
        extra_updates = min(3, int(round((scale - 1.0) * 4.0)))
        if extra_updates <= 0:
            return

        output = perception.pool_output.outputs[ccc_index]
        ccc = self.system.core.ccc_pool.cccs[ccc_index]
        for _ in range(extra_updates):
            if output.resonance is not None and bool(output.resonance.resonated.reshape(-1).any().item()):
                ccc.learn_slow(pattern, output.f1_output, output.resonance)
            self.system.core.fabric.register_activation(
                ccc_index,
                concept,
                confidence=min(1.0, float(confidence) * scale),
                timestep=self.system.core.timestep,
            )
        self.system.core.fabric.form_associations(self.system.core.timestep)

    def _maybe_recruit_window_ccc(
        self,
        window_text: str,
        pattern: torch.Tensor,
        ccc_index: int | None,
        concept: torch.Tensor,
        confidence: float,
    ) -> tuple[int | None, torch.Tensor, float, bool]:
        if not self.config.use_contextual_patterns or not window_text.strip():
            return ccc_index, concept, confidence, False

        preferred = self._most_common_window_ccc(window_text)
        if preferred is not None:
            preferred_concept = self.system.core.ccc_pool.cccs[preferred].concept_direction.detach().clone()
            return preferred, self._normalize(preferred_concept), max(confidence, 0.7), False

        frequency = int(self.window_counts.get(window_text, 0))
        should_recruit = frequency >= 2 and (ccc_index is None or confidence < 0.82 or frequency >= 3)
        if not should_recruit:
            return ccc_index, concept, confidence, False

        recruit_index = self.system.core.ccc_pool._first_uncommitted_index()  # noqa: SLF001
        if recruit_index is None:
            return ccc_index, concept, confidence, False

        recruited_ccc = self.system.core.ccc_pool.cccs[recruit_index]
        f1_output = recruited_ccc.f1_encode(pattern)
        recruited_ccc.learn_fast(pattern, f1_output)
        recruited_concept = self._normalize(recruited_ccc.concept_direction.detach().clone())
        boosted_confidence = max(0.8, float(confidence))
        self.system.core.fabric.register_activation(
            recruit_index,
            recruited_concept,
            confidence=boosted_confidence,
            timestep=self.system.core.timestep,
        )
        self.system.core.fabric.form_associations(self.system.core.timestep)
        self.window_to_ccc_counts[window_text][recruit_index] += 1
        return recruit_index, recruited_concept, boosted_confidence, True

    def _recognize_without_learning(self, token_id: int, context_ids: Sequence[int] | None = None) -> _Observation:
        pattern = self._token_pattern(token_id, context_ids=context_ids)
        window_text = self._window_text(list(context_ids or []) + [int(token_id)])
        pool_output = self.system.core._run_pool(pattern, allow_recruit=False)  # noqa: SLF001
        active_cccs = self.system.core._active_cccs(pool_output)  # noqa: SLF001
        inhibited = (
            self.system.core.fabric.lateral_inhibition(active_cccs, k=max(1, len(active_cccs)))
            if active_cccs
            else []
        )
        surviving = self.system.core._surviving_cccs(active_cccs, inhibited)  # noqa: SLF001
        vote_result = self.system.core.fabric.vote(surviving)
        ccc_index = self._extract_ccc_index(pool_output)
        concept = vote_result.winning_direction.detach().clone()
        if ccc_index is not None:
            concept = self.system.core.ccc_pool.cccs[ccc_index].concept_direction.detach().clone()
        if float(concept.norm().item()) == 0.0:
            prototype = self._token_prototype(token_id)
            if prototype is not None:
                concept = prototype
        confidence = float(vote_result.confidence)
        if ccc_index is not None and pool_output.outputs:
            confidence = float(pool_output.outputs[ccc_index].confidence.reshape(-1).mean().item())
        preferred = self._most_common_window_ccc(window_text)
        if preferred is not None and (ccc_index is None or confidence < 0.65):
            ccc_index = preferred
            concept = self._normalize(self.system.core.ccc_pool.cccs[preferred].concept_direction.detach().clone())
            usage = self.window_to_ccc_counts[window_text][preferred]
            confidence = max(confidence, min(0.98, 0.68 + (0.08 * math.log1p(max(usage, 1)))))
        return _Observation(
            token_id=int(token_id),
            concept=self._normalize(concept),
            ccc_index=ccc_index,
            confidence=confidence,
            recruited=False,
            prediction_error=0.0,
            learning_rate=float(self.system.core.config.ccc.slow_lr),
        )

    def _retrieve_next_concept(self, current_concept: torch.Tensor, temperature: float) -> torch.Tensor:
        normalized = self._normalize(current_concept)
        if float(normalized.norm().item()) == 0.0:
            return normalized
        noise_scale = 0.035 * max(0.0, float(temperature))
        noisy_query = normalized
        if noise_scale > 0.0:
            noisy_query = self._normalize(normalized + (torch.randn_like(normalized) * noise_scale))
        retrieved = self.system.core.fabric.sdm.retrieve_associates(noisy_query)
        if float(retrieved.norm().item()) > 0.0:
            retrieved = self._normalize(retrieved)
        context_vector = self._current_context_vector(noisy_query) if self.enable_generation_context else torch.zeros_like(normalized)
        associates = self.system.core.fabric.retrieve_associates(noisy_query, k=4)
        if associates.directions:
            weights = torch.tensor(
                [max(strength, 1e-6) for strength in associates.strengths],
                dtype=torch.float32,
                device=noisy_query.device,
            )
            weights = weights / weights.sum()
            assoc_vector = torch.stack(
                [self._normalize(direction).to(noisy_query) for direction in associates.directions]
            )
            blended = (weights.unsqueeze(-1) * assoc_vector).sum(dim=0)
            if float(retrieved.norm().item()) > 0.0 and float(context_vector.norm().item()) > 0.0:
                return self._normalize((0.55 * retrieved) + (0.25 * blended) + (0.20 * context_vector))
            if float(retrieved.norm().item()) > 0.0:
                return self._normalize((0.7 * retrieved) + (0.3 * blended))
            if float(context_vector.norm().item()) > 0.0:
                return self._normalize((0.7 * blended) + (0.3 * context_vector))
            return self._normalize(blended)
        if float(retrieved.norm().item()) > 0.0 and float(context_vector.norm().item()) > 0.0:
            return self._normalize((0.7 * retrieved) + (0.3 * context_vector))
        if float(context_vector.norm().item()) > 0.0:
            return context_vector
        return retrieved if float(retrieved.norm().item()) > 0.0 else normalized

    def _word_level_context_vector(self) -> torch.Tensor:
        base = torch.zeros(self.system.core.config.ccc.concept_dim, dtype=torch.float32)
        if self.word_processor is None or not self._runtime_token_history:
            return base

        tail = self._runtime_token_history[-max(48, int(self.config.context_length) * 6) :]
        history_text = self._decode_token(tail)
        if not history_text:
            return base

        trailing_boundary = bool(history_text and history_text[-1] in {" ", ".", ",", "!", "?", ";", ":", "\n", "\t"})
        words = [match.group(0).lower() for match in re.finditer(r"[A-Za-z']+", history_text)]
        if not words:
            return base
        if not trailing_boundary:
            words = words[:-1]
        if not words:
            return base

        vectors: list[tuple[float, torch.Tensor]] = []
        for rank, word in enumerate(reversed(words[-3:]), start=1):
            concept = self.word_processor.word_concept(word)
            if concept is None or float(concept.norm().item()) <= 1e-8:
                continue
            aligned = concept.to(torch.float32)
            if aligned.numel() > base.numel():
                aligned = aligned[: base.numel()]
            elif aligned.numel() < base.numel():
                aligned = F.pad(aligned, (0, base.numel() - aligned.numel()))
            vectors.append((1.0 / float(rank), self._normalize(aligned)))

        if not vectors:
            return base

        combined = torch.zeros_like(base)
        total_weight = 0.0
        for weight, vector in vectors:
            combined = combined + (weight * vector.to(combined))
            total_weight += weight
        return self._normalize(combined / max(total_weight, 1e-6))

    def _current_context_vector(self, query: torch.Tensor | None = None) -> torch.Tensor:
        base = torch.zeros(self.system.core.config.ccc.concept_dim, dtype=torch.float32)
        if hasattr(self.system.core.gnw, "get_context_vector"):
            base = self._normalize(self.system.core.gnw.get_context_vector().detach().clone())
        attended = torch.zeros_like(base)
        if query is not None and hasattr(self.system.core.gnw, "attend_context"):
            attended = self._normalize(self.system.core.gnw.attend_context(query).detach().clone())
        recurrent = self._normalize(self.recurrent_context.state.detach().clone())
        word_level = self._word_level_context_vector()

        vectors: list[tuple[float, torch.Tensor]] = []
        if float(base.norm().item()) > 0.0:
            vectors.append((0.4, base))
        if float(attended.norm().item()) > 0.0:
            vectors.append((0.25, attended))
        if float(recurrent.norm().item()) > 0.0:
            vectors.append((0.2, recurrent))
        if float(word_level.norm().item()) > 0.0:
            vectors.append((0.15, word_level))
        if not vectors:
            return base

        combined = torch.zeros_like(base)
        total_weight = 0.0
        for weight, vector in vectors:
            combined = combined + (weight * vector.to(combined))
            total_weight += weight
        return self._normalize(combined / max(total_weight, 1e-6))

    def _apply_generation_context(self, token_ids: Sequence[int], prediction: _Prediction) -> _Prediction:
        if not self.enable_generation_context or not prediction.candidate_ids:
            return prediction

        query = (
            self._runtime_concept_history[-1]
            if self._runtime_concept_history
            else prediction.retrieved_concept.detach().clone()
        )
        context_vector = self._current_context_vector(query)
        biased_logits = torch.log(prediction.probabilities.clamp_min(1e-6))
        if float(context_vector.norm().item()) > 1e-8:
            prototypes = torch.stack(
                [
                    (
                        prototype
                        if (prototype := self._token_prototype(token_id)) is not None
                        else torch.zeros(self.system.core.config.ccc.concept_dim, dtype=torch.float32)
                    )
                    for token_id in prediction.candidate_ids
                ],
                dim=0,
            )
            context_bias = self.recurrent_context.prime_retrieval(context_vector, prototypes).to(prediction.probabilities)
            biased_logits = biased_logits + (float(self.system.core.gnw.config.context_bias_gain) * context_bias)

            repetition_score = self.recurrent_context.detect_repetition(
                self._runtime_concept_history,
                window=int(self.system.core.gnw.config.repetition_window),
            )
            if repetition_score >= float(self.system.core.gnw.config.repetition_novelty_threshold):
                noise_scale = min(0.15, 0.03 + (0.12 * repetition_score))
                biased_logits = biased_logits + (torch.randn_like(biased_logits) * noise_scale)
        else:
            context_bias = torch.zeros_like(prediction.probabilities)
            repetition_score = 0.0

        if self.dual_processor is not None and self.config.tokenizer_type.lower() == "char":
            prompt_text = self._decode_token([int(token_id) for token_id in token_ids])
            char_candidates = [
                (self._decode_token([int(candidate_id)]), float(probability.item()))
                for candidate_id, probability in zip(prediction.candidate_ids, prediction.probabilities, strict=False)
            ]
            reranked = self.dual_processor.rerank_char_candidates(
                prompt_text,
                char_candidates,
                temperature=max(0.2, float(self.config.temperature)),
            )
            if reranked:
                word_scores = {char: float(score) for char, score in reranked}
                word_logits = torch.tensor(
                    [
                        max(1e-6, word_scores.get(self._decode_token([int(candidate_id)]), 1e-6))
                        for candidate_id in prediction.candidate_ids
                    ],
                    dtype=biased_logits.dtype,
                    device=biased_logits.device,
                )
                word_logits = torch.log(word_logits / word_logits.sum().clamp_min(1e-6))
                blend = min(0.65, 0.2 + (0.45 * float(self.config.word_level.word_constraint_strength)))
                biased_logits = ((1.0 - blend) * biased_logits) + (blend * word_logits)

        probabilities = torch.softmax(biased_logits, dim=0)
        self._last_context_utilization = float(context_bias.abs().mean().item())
        self._last_context_repetition = float(repetition_score)
        self._last_topic_drift = float(
            self.system.core.gnw.context.get_topic_drift() if hasattr(self.system.core.gnw, "context") else 0.0
        )
        return replace(prediction, probabilities=probabilities.detach().clone())

    def _candidate_token_ids(self) -> list[int]:
        observed = [
            token_id
            for token_id in sorted(self.token_counts)
            if token_id not in self._special_token_ids or token_id == self._eos_token_id
        ]
        return observed

    def _reconstruct_spike(self, candidate_ids: list[int], probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
        top_k = min(8, len(candidate_ids))
        if top_k == 0:
            return torch.zeros(self.spike_encoder.spike_dim, dtype=torch.float32)
        top_values, top_indices = torch.topk(probabilities, k=top_k)
        reconstructed = torch.zeros(self.spike_encoder.spike_dim, dtype=torch.float32)
        for weight, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
            reconstructed = reconstructed + (float(weight) * self.spike_encoder.encode_token(candidate_ids[index]))
        if temperature > 1.0:
            reconstructed = reconstructed + (torch.randn_like(reconstructed) * 0.02 * float(temperature))
        return reconstructed.clamp(0.0, 1.0)

    def _predict_next_token(
        self,
        temperature: float,
        *,
        history_ids: Sequence[int] | None = None,
        repetition_penalty: RepetitionPenalty | None = None,
    ) -> _Prediction:
        candidate_ids = self._candidate_token_ids()
        if not candidate_ids:
            return _Prediction(
                token_id=self._eos_token_id,
                confidence=0.0,
                retrieved_concept=torch.zeros(self.system.core.config.ccc.concept_dim, dtype=torch.float32),
                reconstructed_spike=torch.zeros(self.spike_encoder.spike_dim, dtype=torch.float32),
                candidate_ids=[],
                probabilities=torch.empty(0, dtype=torch.float32),
                sdm_confidence=0.0,
                margin_confidence=0.0,
                transition_confidence=0.0,
                ngram_confidence=0.0,
                chunk_confidence=0.0,
                ensemble_confidence=0.0,
                source_predictions={"sdm": [], "transition": [], "ngram": [], "chunk": []},
            )

        history = [int(token_id) for token_id in (history_ids if history_ids is not None else self._runtime_token_history)]
        current_token = history[-1] if history else None
        if self.system.core.gnw.slots:
            current_concept = self.system.core.gnw.slots[0].direction.detach().clone()
        elif self._runtime_concept_history:
            current_concept = self._runtime_concept_history[-1]
        else:
            most_common = self.token_counts.most_common(1)
            if most_common:
                prototype = self._token_prototype(most_common[0][0])
                current_concept = prototype if prototype is not None else torch.zeros(self.system.core.config.ccc.concept_dim)
            else:
                current_concept = torch.zeros(self.system.core.config.ccc.concept_dim)

        candidate_prototypes = self._candidate_prototypes(candidate_ids)
        sequence_result = self.sequence_memory.score_candidates(
            context=history,
            candidate_ids=candidate_ids,
            token_prototypes=candidate_prototypes,
            context_text=self._window_text(history),
            ngram_cache=self.ngram_cache if self.config.enable_ngram_cache else None,
            temperature=temperature,
            fallback_concept=current_concept,
        )
        retrieved = sequence_result.retrieved_concept
        probabilities = sequence_result.probabilities.detach().clone()
        if repetition_penalty is not None:
            probabilities = repetition_penalty.apply(
                probabilities,
                {"history": history, "candidate_ids": candidate_ids},
            )
            probabilities = probabilities / probabilities.sum().clamp_min(1e-6)
        reconstructed_spike = self._reconstruct_spike(candidate_ids, probabilities, temperature)

        sdm_confidence = float(sequence_result.source_confidences.get("sdm", 0.0))
        transition_confidence = float(sequence_result.source_confidences.get("transition", 0.0))
        ngram_confidence = float(sequence_result.source_confidences.get("ngram", 0.0))
        chunk_confidence = float(sequence_result.source_confidences.get("chunk", 0.0))
        sorted_probs = torch.sort(probabilities, descending=True).values
        if sorted_probs.numel() >= 2:
            margin_confidence = float(
                torch.clamp((sorted_probs[0] - sorted_probs[1]) / sorted_probs[0].clamp_min(1e-6), 0.0, 1.0).item()
            )
        else:
            margin_confidence = float(sorted_probs[0].item()) if sorted_probs.numel() == 1 else 0.0

        best_index = int(torch.argmax(probabilities).item())
        decoded = int(candidate_ids[best_index])
        spike_decoded = int(self.spike_encoder.decode_spikes(reconstructed_spike))
        if spike_decoded in candidate_ids and spike_decoded not in self._special_token_ids:
            spike_index = candidate_ids.index(spike_decoded)
            if probabilities[spike_index] >= probabilities[best_index] * 0.9:
                decoded = spike_decoded
                best_index = spike_index

        confidence = math.sqrt(
            max(1e-6, float(probabilities[best_index].item()))
            * max(
                0.05,
                (0.35 * sdm_confidence)
                + (0.35 * transition_confidence)
                + (0.2 * ngram_confidence)
                + (0.1 * chunk_confidence),
            )
            * max(0.05, margin_confidence)
        )
        return _Prediction(
            token_id=decoded,
            confidence=float(confidence),
            retrieved_concept=retrieved.detach().clone(),
            reconstructed_spike=reconstructed_spike.detach().clone(),
            candidate_ids=candidate_ids,
            probabilities=probabilities.detach().clone(),
            sdm_confidence=sdm_confidence,
            margin_confidence=margin_confidence,
            transition_confidence=transition_confidence,
            ngram_confidence=ngram_confidence,
            chunk_confidence=chunk_confidence,
            ensemble_confidence=float(sequence_result.confidence),
            source_predictions=sequence_result.source_predictions,
        )

    def _observe_token(self, token_id: int, *, learn: bool) -> _Observation:
        context_ids = self._runtime_token_history[-(self._window_size - 1) :]
        window_ids = list(context_ids) + [int(token_id)]
        window_text = self._window_text(window_ids)
        if learn and context_ids:
            candidate_ids = self._candidate_token_ids()
            if candidate_ids:
                sequence_preview = self.sequence_memory.score_candidates(
                    context=list(context_ids),
                    candidate_ids=candidate_ids,
                    token_prototypes=self._candidate_prototypes(candidate_ids),
                    context_text=self._window_text(context_ids),
                    ngram_cache=self.ngram_cache if self.config.enable_ngram_cache else None,
                    temperature=1.0,
                    fallback_concept=self._runtime_concept_history[-1] if self._runtime_concept_history else None,
                )
                self.sequence_memory.update_prediction_feedback(
                    int(token_id),
                    sequence_preview.source_predictions,
                )
        if not learn:
            observation = self._recognize_without_learning(token_id, context_ids=context_ids)
            self._record_runtime(token_id, observation.concept, observation.ccc_index, observation.confidence)
            return observation

        preferred_window_ccc = self._most_common_window_ccc(window_text)
        if preferred_window_ccc is not None:
            concept = self._normalize(
                self.system.core.ccc_pool.cccs[preferred_window_ccc].concept_direction.detach().clone()
            )
            predicted = self._runtime_concept_history[-1] if self._runtime_concept_history else None
            prediction_error = 0.0
            if predicted is not None and float(predicted.norm().item()) > 0.0 and float(concept.norm().item()) > 0.0:
                prediction_error = max(0.0, 1.0 - self._cosine(predicted, concept))
            usage = self.window_to_ccc_counts[window_text][preferred_window_ccc]
            confidence = min(0.99, 0.76 + (0.06 * math.log1p(max(usage, 1))))
            self.system.core.fabric.register_activation(
                preferred_window_ccc,
                concept,
                confidence=confidence,
                timestep=self.system.core.timestep,
            )
            self.system.core.fabric.form_associations(self.system.core.timestep)
            self.system.core.timestep += 1
            reward_step = self.system.reward.step(prediction_error, learned=False)
            self.system._apply_modulation(reward_step.modulation)  # noqa: SLF001
            observation = _Observation(
                token_id=int(token_id),
                concept=concept.detach().clone(),
                ccc_index=preferred_window_ccc,
                confidence=confidence,
                recruited=False,
                prediction_error=float(prediction_error),
                learning_rate=float(self.system.core.config.ccc.slow_lr),
            )
            self._record_runtime(token_id, observation.concept, observation.ccc_index, observation.confidence)
            self._update_memories(token_id, observation.concept, observation.ccc_index)
            self.window_to_ccc_counts[window_text][preferred_window_ccc] += 1
            self.training_steps += 1
            return observation

        pattern = self._token_pattern(token_id, context_ids=context_ids)
        predicted = None
        if self._runtime_concept_history:
            predicted = self._retrieve_next_concept(self._runtime_concept_history[-1], temperature=1.0)

        perception = self.system.core.perceive(pattern)
        self.system.core.learn_from_perception(perception, pattern)
        ccc_index = self._extract_ccc_index(perception.pool_output)
        concept = perception.vote_result.winning_direction.detach().clone()
        if ccc_index is not None:
            concept = self.system.core.ccc_pool.cccs[ccc_index].concept_direction.detach().clone()
        concept = self._normalize(concept)
        if float(concept.norm().item()) == 0.0:
            prototype = self._token_prototype(token_id)
            if prototype is not None:
                concept = prototype

        prediction_error = 0.0
        if predicted is not None and float(predicted.norm().item()) > 0.0 and float(concept.norm().item()) > 0.0:
            prediction_error = max(0.0, 1.0 - self._cosine(predicted, concept))
        learned = self.system._learning_occurred(perception)  # noqa: SLF001
        reward_step = self.system.reward.step(prediction_error, learned=learned)
        self.system._apply_modulation(reward_step.modulation)  # noqa: SLF001
        self._adapt_pool_thresholds()

        confidence = float(perception.vote_result.confidence)
        if ccc_index is not None and perception.pool_output.outputs:
            confidence = float(perception.pool_output.outputs[ccc_index].confidence.reshape(-1).mean().item())

        ccc_index, concept, confidence, recruited_window = self._maybe_recruit_window_ccc(
            window_text,
            pattern,
            ccc_index,
            concept,
            confidence,
        )

        scale = self._frequency_scale(window_text)
        self._reinforce_pattern(scale, ccc_index, pattern, perception, concept, confidence)

        observation = _Observation(
            token_id=int(token_id),
            concept=concept.detach().clone(),
            ccc_index=ccc_index,
            confidence=min(1.0, confidence * min(scale, 1.15)),
            recruited=bool(perception.pool_output.recruited or recruited_window),
            prediction_error=float(prediction_error),
            learning_rate=float(self.system.core.config.ccc.slow_lr),
        )
        self._record_runtime(token_id, observation.concept, observation.ccc_index, observation.confidence)
        self._update_memories(token_id, observation.concept, observation.ccc_index)
        if observation.ccc_index is not None and window_text:
            self.window_to_ccc_counts[window_text][int(observation.ccc_index)] += 1
        self.training_steps += 1
        return observation

    def _prime_statistics(self, raw_text: str, token_ids: Sequence[int]) -> None:
        self._latest_corpus = raw_text
        self.ngram_cache.bind_tokenizer(self.tokenizer)
        self.sequence_memory.bind_tokenizer(self.tokenizer)
        if self.config.enable_ngram_cache:
            self.ngram_cache.learn(raw_text)
        self.sequence_memory.learn_chunks(raw_text)
        for index in range(len(token_ids)):
            window = token_ids[max(0, index - self._window_size + 1) : index + 1]
            text = self._window_text(window)
            if text:
                self.window_counts[text] += 1
        self._max_window_frequency = max(self._max_window_frequency, max(self.window_counts.values(), default=1))

    def _train_word_level(self, raw_text: str) -> None:
        if self.word_processor is None:
            return
        self.word_processor.learn_vocabulary(raw_text)
        self.word_processor.learn_word_transitions(raw_text)

    def _compress_training_tokens(self, token_ids: Sequence[int]) -> list[int]:
        if len(token_ids) <= 2000:
            return [int(token_id) for token_id in token_ids]

        stride = max(3, len(token_ids) // 700)
        selected: list[int] = []
        for index, token_id in enumerate(token_ids):
            token_text = self._decode_token([int(token_id)])
            keep = (
                index < 256
                or index % stride == 0
                or token_text in {" ", ".", ",", "!", "?", "\n"}
            )
            if keep:
                selected.append(int(token_id))
        if selected and selected[-1] != int(token_ids[-1]):
            selected.append(int(token_ids[-1]))
        return selected

    def _train_segments(self, token_segments: Sequence[Sequence[int]], *, print_progress: bool) -> TrainingMetrics:
        start_time = time.perf_counter()
        concepts_recruited = 0
        prediction_errors: list[float] = []
        learning_rate_trace: list[float] = []
        memory_trace: list[float] = []
        concepts_trace: list[int] = []
        pattern_frequencies: list[int] = []
        processed = 0

        for segment_index, segment in enumerate(token_segments):
            if segment_index > 0:
                self._reset_runtime_state(clear_workspace=True, clear_temporal_buffer=True)
            for token_id in segment:
                observation = self._observe_token(int(token_id), learn=True)
                processed += 1
                concepts_recruited += int(observation.recruited)
                prediction_errors.append(float(observation.prediction_error))
                learning_rate_trace.append(float(observation.learning_rate))
                window = self._window_text(self._runtime_token_history[-self._window_size :])
                if window:
                    pattern_frequencies.append(int(self.window_counts.get(window, 1)))

                if processed % 10 == 0 or processed == 1:
                    system_stats = self.system.core.get_system_stats()
                    sdm_stats = self.system.core.fabric.sdm.get_stats()
                    concepts_trace.append(int(system_stats["concepts_learned"]))
                    memory_trace.append(float(sdm_stats["capacity_used"]))

                if print_progress and processed % 500 == 0:
                    pool_stats = self.system.core.ccc_pool.get_pool_stats()
                    sdm_stats = self.system.core.fabric.sdm.get_stats()
                    print(
                        "[text-train] "
                        f"samples={processed} "
                        f"concepts={pool_stats['num_committed']} "
                        f"sdm={sdm_stats['capacity_used']:.3f} "
                        f"lr={self.system.core.config.ccc.slow_lr:.4f}",
                        flush=True,
                    )
                self.sequence_memory.maybe_replay(processed, prediction_errors)

        elapsed = time.perf_counter() - start_time
        final_stats = self.system.core.get_system_stats()
        sdm_stats = self.system.core.fabric.sdm.get_stats()
        return TrainingMetrics(
            num_samples=processed,
            tokens_processed=processed,
            concepts_learned=int(final_stats["concepts_learned"]),
            concepts_recruited=concepts_recruited,
            memory_utilization=float(sdm_stats["capacity_used"]),
            mean_learning_rate=float(statistics.fmean(learning_rate_trace) if learning_rate_trace else 0.0),
            mean_prediction_error=float(statistics.fmean(prediction_errors) if prediction_errors else 0.0),
            unique_tokens_seen=len(self.token_counts),
            elapsed_seconds=float(elapsed),
            concepts_trace=concepts_trace,
            memory_trace=memory_trace,
            learning_rate_trace=learning_rate_trace,
            mean_pattern_frequency=float(statistics.fmean(pattern_frequencies) if pattern_frequencies else 0.0),
            num_passes=max(1, int(self.config.num_passes)),
        )

    @torch.no_grad()
    def train_on_corpus(self, text_source, num_samples: int = 10000) -> TrainingMetrics:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        raw_text = self._materialize_text(text_source, char_limit=max(512, int(num_samples)))
        if not raw_text:
            raise ValueError("text_source did not yield any text.")
        self._fit_tokenizer_if_needed(raw_text)
        token_ids = self.tokenizer.encode(raw_text)
        if not token_ids:
            raise ValueError("text_source did not yield any tokens.")
        token_ids = token_ids[:num_samples]

        self._prime_statistics(raw_text[: max(1, len(token_ids))], token_ids)
        self._train_word_level(raw_text[: max(1, len(token_ids))])
        self._reset_runtime_state(clear_workspace=True, clear_temporal_buffer=True)
        observed_token_ids = self._compress_training_tokens(token_ids)
        token_segments = [observed_token_ids for _ in range(max(1, int(self.config.num_passes)))]
        return self._train_segments(token_segments, print_progress=True)

    @torch.no_grad()
    def train_on_text(self, raw_text: str, context_length: int = 64) -> TrainingMetrics:
        if not raw_text:
            raise ValueError("raw_text must be non-empty.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")

        original_context_length = self.config.context_length
        self.config.context_length = int(context_length)
        try:
            self._fit_tokenizer_if_needed(raw_text)
            token_ids = self.tokenizer.encode(raw_text)
            if not token_ids:
                raise ValueError("raw_text did not produce any tokens.")
            self._prime_statistics(raw_text, token_ids)
            self._train_word_level(raw_text)
            self._reset_runtime_state(clear_workspace=True, clear_temporal_buffer=True)
            token_segments: list[list[int]] = []
            for _ in range(max(1, int(self.config.num_passes))):
                for start in range(0, len(token_ids), context_length):
                    window = token_ids[start : start + context_length]
                    if window:
                        token_segments.append(window)
            return self._train_segments(token_segments, print_progress=True)
        finally:
            self.config.context_length = original_context_length
            self._refresh_generation_helpers()

    def _loop_detected(self, token_ids: Sequence[int]) -> bool:
        ids = [int(token_id) for token_id in token_ids]
        if len(ids) >= 8 and ids[-4:] == ids[-8:-4]:
            return True
        if len(ids) >= 6 and ids[-3:] == ids[-6:-3]:
            return True
        if len(ids) >= 5 and len(set(ids[-5:])) == 1:
            return True
        return False

    def _normalize_generation_input(self, prompt_spikes) -> list[int]:
        if isinstance(prompt_spikes, str):
            return [int(token_id) for token_id in self.tokenizer.encode(prompt_spikes)]
        if isinstance(prompt_spikes, torch.Tensor):
            return [int(token_id) for token_id in prompt_spikes.reshape(-1).tolist()]
        return [int(token_id) for token_id in prompt_spikes]

    def _predict_from_tokens(
        self,
        token_ids: Sequence[int],
        *,
        temperature: float,
        repetition_penalty: RepetitionPenalty | None = None,
    ) -> _Prediction:
        self._ensure_spike_encoder()
        self._reset_runtime_state(clear_workspace=True, clear_temporal_buffer=True)
        for token_id in token_ids:
            self._observe_token(int(token_id), learn=False)
        return self._predict_next_token(
            temperature=temperature,
            history_ids=list(token_ids),
            repetition_penalty=repetition_penalty,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        *,
        method: str = "beam",
        beam_width: int | None = None,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self._ensure_spike_encoder()
        self.decoder = BeamSearchDecoder(beam_width=beam_width or self.config.beam_width, length_penalty=0.6)
        prompt_ids = self._normalize_generation_input(prompt)

        if self.dual_processor is not None and method in {"beam", "greedy"}:
            estimated_words = max(1, min(24, math.ceil(max_tokens / 4)))
            constrained = self.dual_processor.generate_sentence(
                prompt,
                max_words=estimated_words,
                temperature=temperature,
            )
            if constrained:
                if len(constrained) <= max_tokens:
                    return constrained
                truncated = constrained[:max_tokens]
                boundary = max(
                    truncated.rfind(" "),
                    truncated.rfind("."),
                    truncated.rfind("!"),
                    truncated.rfind("?"),
                    truncated.rfind(","),
                )
                if boundary >= max(1, max_tokens // 2):
                    return truncated[:boundary].rstrip()
                return truncated.rstrip()

        if method == "beam":
            results = self.decoder.decode(self, prompt_ids, max_tokens=max_tokens)
            if results:
                return results[0].text
            return self.decoder.greedy_decode(self, prompt_ids, max_tokens=max_tokens).text
        if method == "greedy":
            return self.decoder.greedy_decode(self, prompt_ids, max_tokens=max_tokens).text
        if method == "sample":
            return self.decoder.sample_decode(self, prompt_ids, max_tokens=max_tokens, temperature=temperature).text
        if method == "top-k":
            return self.decoder.top_k_decode(self, prompt_ids, max_tokens=max_tokens, k=top_k).text
        if method == "top-p":
            return self.decoder.top_p_decode(self, prompt_ids, max_tokens=max_tokens, p=top_p).text
        raise ValueError("method must be one of: beam, greedy, sample, top-k, top-p.")

    def _sample_positions(self, token_ids: list[int], num_samples: int) -> list[int]:
        if len(token_ids) < 2:
            return []
        stride = max(1, (len(token_ids) - 1) // max(1, num_samples))
        return list(range(1, len(token_ids), stride))[:num_samples]

    def _ngram_set(self, text: str, n: int) -> set[str]:
        if len(text) < n:
            return set()
        return {text[index : index + n] for index in range(len(text) - n + 1)}

    @torch.no_grad()
    def evaluate_generation(self, test_text: str, num_samples: int = 100) -> GenerationMetrics:
        if not test_text:
            raise ValueError("test_text must be non-empty.")
        self._ensure_spike_encoder()
        token_ids = self.tokenizer.encode(test_text)
        positions = self._sample_positions(token_ids, num_samples)
        if not positions:
            return GenerationMetrics(
                perplexity=1.0,
                prediction_accuracy=0.0,
                diversity=0.0,
                coherence=0.0,
                average_confidence=0.0,
                num_samples=0,
                generated_examples=[],
                quality_report=self.quality_metrics.evaluate([], test_text),
            )

        log_probs: list[float] = []
        confidences: list[float] = []
        correct = 0
        effective_context = max(1, int(self.config.context_length))

        for position in positions:
            context = token_ids[max(0, position - effective_context) : position]
            actual = int(token_ids[position])
            prediction = self._predict_from_tokens(
                context,
                temperature=max(0.45, float(self.config.temperature)),
                repetition_penalty=None,
            )
            actual_obs = self._recognize_without_learning(actual, context_ids=context[-(self._window_size - 1) :])
            actual_probability = 1e-6
            if actual in prediction.candidate_ids:
                candidate_index = prediction.candidate_ids.index(actual)
                actual_probability = float(prediction.probabilities[candidate_index].item())
            proxy_probability = max(
                1e-6,
                min(1.0, 0.7 * actual_probability + 0.3 * max(0.0, actual_obs.confidence)),
            )
            log_probs.append(math.log(proxy_probability))
            confidences.append(float(prediction.confidence))
            correct += int(int(prediction.token_id) == actual)

        prompts = [test_text[:4], test_text[:8], test_text[:12]]
        generated_examples = [
            self.generate(
                prompt,
                max_tokens=min(40, self.config.generate_max_tokens),
                temperature=self.config.temperature,
                method="beam",
            )
            for prompt in dict.fromkeys(prompt for prompt in prompts if prompt)
        ]
        generated_text = "".join(generated_examples)
        generated_ngrams = self._ngram_set(generated_text, 3)
        all_generated_ngrams = max(1, len(generated_text) - 2)
        diversity = len(generated_ngrams) / float(all_generated_ngrams)

        reference_bigrams = self._ngram_set(test_text, 2)
        output_bigrams = [generated_text[index : index + 2] for index in range(max(0, len(generated_text) - 1))]
        if output_bigrams:
            coherence = sum(1 for bigram in output_bigrams if bigram in reference_bigrams) / float(len(output_bigrams))
        else:
            coherence = 0.0

        quality_report = self.quality_metrics.evaluate(generated_examples, test_text)
        return GenerationMetrics(
            perplexity=float(math.exp(-statistics.fmean(log_probs))),
            prediction_accuracy=float(correct / len(positions)),
            diversity=float(diversity),
            coherence=float(coherence),
            average_confidence=float(statistics.fmean(confidences) if confidences else 0.0),
            num_samples=len(positions),
            generated_examples=generated_examples,
            quality_report=quality_report,
        )

    def interactive_mode(self) -> None:
        print("Bio-ARN text generation REPL. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("prompt> ")
            except EOFError:
                print()
                break
            if prompt.strip().lower() in {"quit", "exit"}:
                break
            generated = self.generate(
                prompt,
                max_tokens=self.config.generate_max_tokens,
                temperature=self.config.temperature,
                method="beam",
            )
            print(generated)


__all__ = [
    "GenerationMetrics",
    "TextGenConfig",
    "TextGenerationTrainer",
    "TrainingMetrics",
    "build_builtin_corpus",
]
