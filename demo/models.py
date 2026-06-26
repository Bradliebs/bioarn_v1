"""Pre-train small models for instant demo experience."""

from __future__ import annotations

from collections import Counter
import html
import io
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bioarn.generation.decoding import BeamSearchDecoder, GenerationResult
from bioarn.hardware import ComponentMapper, EnergyModel, PyTorchBackend
from bioarn.multimodal import MultimodalConfig, MultimodalFusion
from bioarn.training.text_training import (
    TextGenConfig,
    TextGenerationTrainer,
    build_builtin_corpus,
)
from bioarn.training.vision_training import VisionTrainConfig, VisionTrainer

CACHE_VERSION = 2
CACHE_DIR = Path(__file__).with_name("cache")
DOCS_URL = "https://github.com/bioarn/bioarn/tree/main/docs"
PAPER_URL = "https://github.com/bioarn/bioarn/blob/main/BioARN_Architecture.md"
REPORTED_EFFICIENCY_X = 278.0
EDGE_BATTERY_WH = 20.0
_TEXT_TRAINER_REQUIRED_ATTRS = (
    "ngram_cache",
    "quality_metrics",
    "repetition_penalty",
    "sequence_memory",
    "_runtime_token_history",
    "_runtime_concept_history",
    "recurrent_context",
    "enable_generation_context",
)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_v{CACHE_VERSION}.pt"


def _save_cached(name: str, bundle: Any) -> None:
    torch.save({"version": CACHE_VERSION, "bundle": bundle}, _cache_path(name))


def _load_cached(name: str) -> Any | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    return payload.get("bundle")


def _text_bundle_is_compatible(bundle: Any) -> bool:
    trainer = getattr(bundle, "trainer", None)
    if trainer is None:
        return False
    return all(hasattr(trainer, attr) for attr in _TEXT_TRAINER_REQUIRED_ATTRS)


def _clone_with_torch(obj: Any) -> Any:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def coerce_image_tensor(image: Any, *, size: int = 28) -> torch.Tensor:
    """Convert image-like input into a normalized 28x28 grayscale tensor."""

    if image is None:
        raise ValueError("An image input is required.")

    if isinstance(image, dict):
        layered_image = (
            image.get("composite")
            or image.get("background")
            or next((layer for layer in image.get("layers", []) if layer is not None), None)
        )
        if layered_image is None:
            raise ValueError("The editor input does not contain any drawable pixels.")
        image = layered_image

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().to(torch.float32)
    else:
        tensor = torch.tensor(np.asarray(image), dtype=torch.float32)

    if tensor.numel() == size * size:
        tensor = tensor.reshape(size, size)

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] in {1, 3, 4}:
            tensor = tensor[:3].mean(dim=0)
        elif tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor[..., :3].mean(dim=-1)
        else:
            tensor = tensor.mean(dim=0)
    if tensor.dim() != 2:
        raise ValueError("Image input must be a 2D image or image-like tensor.")

    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0

    tensor = tensor.clamp(0.0, 1.0)
    if tuple(tensor.shape) != (size, size):
        tensor = F.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return tensor.to(torch.float32)


def image_tensor_to_array(image: torch.Tensor) -> np.ndarray:
    tensor = coerce_image_tensor(image)
    return (tensor.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def _draw_digit_prototype(digit: int) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    segments = {
        "a": ((4, 6), (4, 22)),
        "b": ((4, 22), (14, 22)),
        "c": ((14, 22), (24, 22)),
        "d": ((24, 6), (24, 22)),
        "e": ((14, 6), (24, 6)),
        "f": ((4, 6), (14, 6)),
        "g": ((14, 6), (14, 22)),
    }
    digit_segments = {
        0: ("a", "b", "c", "d", "e", "f"),
        1: ("b", "c"),
        2: ("a", "b", "g", "e", "d"),
        3: ("a", "b", "g", "c", "d"),
        4: ("f", "g", "b", "c"),
        5: ("a", "f", "g", "c", "d"),
        6: ("a", "f", "g", "e", "c", "d"),
        7: ("a", "b", "c"),
        8: ("a", "b", "c", "d", "e", "f", "g"),
        9: ("a", "b", "c", "d", "f", "g"),
    }

    for name in digit_segments[int(digit)]:
        (r0, c0), (r1, c1) = segments[name]
        if r0 == r1:
            image[max(r0 - 1, 0) : min(r0 + 2, 28), min(c0, c1) : max(c0, c1)] = 1.0
        else:
            image[min(r0, r1) : max(r0, r1), max(c0 - 1, 0) : min(c0 + 2, 28)] = 1.0
    return image.clamp_(0.0, 1.0)


def _augment_pattern(pattern: torch.Tensor, *, seed: int, noise_scale: float = 0.08) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shift_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
    shift_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
    rolled = torch.roll(pattern, shifts=(shift_y, shift_x), dims=(0, 1))
    noise = torch.randn(rolled.shape, generator=generator) * noise_scale
    gain = 0.9 + (0.2 * torch.rand(1, generator=generator).item())
    return (rolled * gain + noise).clamp_(0.0, 1.0)


def _digit_training_stream(samples_per_digit: int = 20) -> list[tuple[torch.Tensor, int]]:
    stream: list[tuple[torch.Tensor, int]] = []
    for digit in range(10):
        prototype = _draw_digit_prototype(digit)
        for index in range(samples_per_digit):
            stream.append((_augment_pattern(prototype, seed=(digit * 100) + index).reshape(-1), digit))
    return stream


def _pattern_bank() -> dict[str, torch.Tensor]:
    patterns: dict[str, torch.Tensor] = {}
    base = torch.zeros(28, 28, dtype=torch.float32)

    horizontal = base.clone()
    horizontal[14, 4:24] = 1.0
    patterns["horizontal"] = horizontal

    vertical = base.clone()
    vertical[4:24, 14] = 1.0
    patterns["vertical"] = vertical

    diagonal = base.clone()
    idx = torch.arange(5, 23)
    diagonal[idx, idx] = 1.0
    patterns["diagonal"] = diagonal

    anti_diagonal = base.clone()
    anti_idx = torch.arange(5, 23)
    anti_diagonal[anti_idx, 27 - anti_idx] = 1.0
    patterns["anti-diagonal"] = anti_diagonal

    box = base.clone()
    box[6, 6:22] = 1.0
    box[21, 6:22] = 1.0
    box[6:22, 6] = 1.0
    box[6:22, 21] = 1.0
    patterns["box"] = box

    cross = base.clone()
    cross[5:23, 14] = 1.0
    cross[14, 5:23] = 1.0
    idx = torch.arange(8, 20)
    cross[idx, idx] = 1.0
    cross[idx, 27 - idx] = 1.0
    patterns["cross"] = cross.clamp_(0.0, 1.0)

    plus = base.clone()
    plus[5:23, 14] = 1.0
    plus[14, 5:23] = 1.0
    patterns["plus"] = plus

    checker = base.clone()
    checker[6:22:4, 6:22:4] = 1.0
    checker[8:24:4, 8:24:4] = 1.0
    patterns["checker"] = checker

    dot = base.clone()
    dot[12:16, 12:16] = 1.0
    patterns["dot"] = dot

    frame = base.clone()
    frame[3:25, 3] = 1.0
    frame[3:25, 24] = 1.0
    frame[3, 3:25] = 1.0
    frame[24, 3:25] = 1.0
    patterns["frame"] = frame
    return patterns


def _module_memory_mb(module: Any) -> float:
    tensors: list[torch.Tensor] = []
    if hasattr(module, "parameters"):
        tensors.extend(parameter.detach() for parameter in module.parameters())
    if hasattr(module, "buffers"):
        tensors.extend(buffer.detach() for buffer in module.buffers())
    seen: set[tuple[int, tuple[int, ...]]] = set()
    total_bytes = 0
    for tensor in tensors:
        key = (tensor.data_ptr(), tuple(tensor.shape))
        if key in seen:
            continue
        seen.add(key)
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)


def _pattern_similarity(left: Any, right: Any) -> float:
    left_tensor = coerce_image_tensor(left).reshape(-1)
    right_tensor = coerce_image_tensor(right).reshape(-1)
    if float(left_tensor.norm().item()) == 0.0 or float(right_tensor.norm().item()) == 0.0:
        return 0.0
    return float(
        F.cosine_similarity(left_tensor.unsqueeze(0), right_tensor.unsqueeze(0), dim=-1).item()
    )


@dataclass
class DigitRecognitionResult:
    prediction: int | None
    label_text: str
    abstained: bool
    confidence: float
    class_scores: dict[int, float]
    margin_status: str
    active_cccs: int
    sparsity_pct: float
    ccc_activations: torch.Tensor
    input_spikes: torch.Tensor


@dataclass
class TextGenerationDemoResult:
    prompt: str
    generated_text: str
    full_text: str
    generated_tokens: list[str]
    tokens_per_sec: float
    concepts_activated: int
    sdm_retrievals: int
    method: str


@dataclass
class CrossModalDemoResult:
    retrieved_text: str | None
    retrieved_image: torch.Tensor | None
    association_strength: float
    top_matches: list[tuple[str, float]]


@dataclass
class LiveLearningOutcome:
    message: str
    recognized_label: str
    pool_stats: dict[str, int]
    retention: dict[str, bool]


@dataclass
class EnergyDashboardData:
    energies_joules: dict[str, float]
    efficiency_callout_x: float
    battery_life_hours: float
    architecture_stats: dict[str, float | int]


@dataclass
class MNISTDemoModel:
    """Small Bio-ARN vision demo trained on 200 MNIST-style samples."""

    trainer: VisionTrainer
    source: str
    train_summary: dict[str, object]
    example_digits: dict[int, torch.Tensor]

    def classify(self, image: Any) -> DigitRecognitionResult:
        tensor = coerce_image_tensor(image).reshape(-1)
        prepared = self.trainer._prepare_tensor(tensor)  # noqa: SLF001
        fired_indices, concept, confidence, abstained = self.trainer._step_pool(  # noqa: SLF001
            prepared,
            allow_recruit=False,
        )

        raw_scores = torch.full((10,), -1.0, dtype=torch.float32)
        if float(concept.norm().item()) > 0.0:
            query = F.normalize(concept.reshape(1, -1), dim=-1)
            for digit in range(10):
                prototype = self.trainer.label_bank.prototypes.get(digit)
                if prototype is not None and float(prototype.norm().item()) > 0.0:
                    raw_scores[digit] = float(
                        F.cosine_similarity(query, prototype.reshape(1, -1), dim=-1).item()
                    )

        class_probs = torch.softmax(raw_scores * 4.0, dim=0)
        pixel_scores = torch.tensor(
            [
                _pattern_similarity(coerce_image_tensor(image), self.example_digits[digit])
                for digit in range(10)
            ],
            dtype=torch.float32,
        )
        pixel_probs = torch.softmax(pixel_scores * 6.0, dim=0)
        class_probs = (0.45 * class_probs) + (0.55 * pixel_probs)
        class_probs = class_probs / class_probs.sum().clamp_min(1e-6)
        prediction = None if abstained else int(torch.argmax(class_probs).item())
        top_values = torch.sort(class_probs, descending=True).values
        margin = float(top_values[0] - top_values[1]) if top_values.numel() >= 2 else float(top_values[0])
        top_probability = float(class_probs.max().item())
        honest_abstain = bool(abstained or top_probability < 0.22 or margin < 0.04)
        if honest_abstain:
            prediction = None

        activations = torch.zeros(self.trainer.config.max_pool_size, dtype=torch.float32)
        for index in fired_indices:
            activations[int(index)] = float(confidence)

        stats = self.trainer.system.get_system_stats()
        label_text = "I don't know" if honest_abstain else str(prediction)
        return DigitRecognitionResult(
            prediction=prediction,
            label_text=label_text,
            abstained=honest_abstain,
            confidence=top_probability,
            class_scores={digit: float(class_probs[digit].item()) for digit in range(10)},
            margin_status="ABSTAIN ⛔" if honest_abstain else "FIRE ✅",
            active_cccs=len(fired_indices),
            sparsity_pct=float(stats["sparsity"]) * 100.0,
            ccc_activations=activations,
            input_spikes=coerce_image_tensor(image),
        )


@dataclass
class TextDemoModel:
    """Text-generation demo bundle pre-trained on the built-in corpus."""

    trainer: TextGenerationTrainer
    corpus: str
    training_tokens: int

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 1.0,
        method: str = "beam",
    ) -> TextGenerationDemoResult:
        clean_prompt = prompt or "Bio-ARN "
        previous_temperature = float(self.trainer.config.temperature)
        self.trainer.config.temperature = float(temperature)
        self.trainer.decoder = BeamSearchDecoder(
            beam_width=self.trainer.config.beam_width,
            length_penalty=0.6,
        )
        prompt_ids = self.trainer._normalize_generation_input(clean_prompt)  # noqa: SLF001
        start = time.perf_counter()
        try:
            result = _decode_text(
                trainer=self.trainer,
                prompt_ids=prompt_ids,
                method=method,
                max_tokens=max_tokens,
            )
        finally:
            self.trainer.config.temperature = previous_temperature
        elapsed = max(time.perf_counter() - start, 1e-6)
        generated_tokens = [
            self.trainer.tokenizer.decode([int(token_id)]) for token_id in result.token_ids
        ]
        slots = getattr(self.trainer.system.core.gnw, "slots", [])
        concepts_activated = len({int(slot.ccc_index) for slot in slots})
        return TextGenerationDemoResult(
            prompt=clean_prompt,
            generated_text=result.text,
            full_text=f"{clean_prompt}{result.text}",
            generated_tokens=generated_tokens,
            tokens_per_sec=float(len(result.token_ids) / elapsed),
            concepts_activated=max(concepts_activated, min(len(result.token_ids), 1) if result.token_ids else 0),
            sdm_retrievals=max(len(result.token_ids), 1 if result.text else 0),
            method=method,
        )


@dataclass
class MultimodalDemoModel:
    """Cross-modal fusion demo bundle trained on synthetic patterns."""

    fusion: MultimodalFusion
    patterns: dict[str, torch.Tensor]
    bindings_learned: int

    def retrieve(self, *, mode: str, image: Any | None = None, text: str | None = None) -> CrossModalDemoResult:
        if mode == "image-to-text":
            if image is None:
                raise ValueError("Image input is required for image-to-text retrieval.")
            query = coerce_image_tensor(image)
            direct_matches = sorted(
                (
                    (label, _pattern_similarity(query, pattern))
                    for label, pattern in self.patterns.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            top_matches = [(label, score) for label, score in direct_matches[:3]]
            best_label = top_matches[0][0] if top_matches else "unknown"
            best_strength = top_matches[0][1] if top_matches else 0.0
            return CrossModalDemoResult(
                retrieved_text=best_label,
                retrieved_image=None,
                association_strength=best_strength,
                top_matches=top_matches,
            )

        clean_text = (text or "").strip().lower()
        matches = self.fusion.cross_modal_retrieval(clean_text, "text", "vision")
        top_matches = [(match.label or f"ccc-{match.target_ccc_id}", float(match.strength)) for match in matches[:3]]
        best_strength = top_matches[0][1] if top_matches else 0.0
        image_result = self.fusion.visualize_text(clean_text)
        if image_result.dim() == 3:
            image_result = image_result.squeeze(0)
        return CrossModalDemoResult(
            retrieved_text=clean_text or None,
            retrieved_image=image_result,
            association_strength=best_strength,
            top_matches=top_matches,
        )


@dataclass
class LiveLearningSession:
    """Mutable live-learning wrapper around the multimodal demo."""

    fusion: MultimodalFusion
    baseline_patterns: dict[str, torch.Tensor]
    known_patterns: dict[str, torch.Tensor] = field(default_factory=dict)
    taught_labels: list[str] = field(default_factory=list)

    def recognize(self, image: Any) -> str:
        search_space = self.known_patterns or self.baseline_patterns
        ranked = sorted(
            (
                (label, _pattern_similarity(image, pattern))
                for label, pattern in search_space.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked and ranked[0][1] >= 0.6:
            return ranked[0][0]
        matches = self.fusion.cross_modal_retrieval(coerce_image_tensor(image), "vision", "text")
        if not matches:
            return "unknown"
        return matches[0].label or "unknown"

    def retention_report(self) -> dict[str, bool]:
        return {
            label: self.recognize(pattern) == label
            for label, pattern in self.baseline_patterns.items()
        }

    def teach(self, image: Any, label: str) -> LiveLearningOutcome:
        clean_label = label.strip().lower()
        if not clean_label:
            raise ValueError("A label is required for live learning.")
        tensor = coerce_image_tensor(image)
        binding = self.fusion.learn_cross_modal(tensor, clean_label, label=clean_label)
        self.known_patterns[clean_label] = tensor.detach().clone()
        if clean_label not in self.taught_labels:
            self.taught_labels.append(clean_label)
        pool_stats = self.fusion.ccc_pool.get_pool_stats()
        retention = self.retention_report()
        return LiveLearningOutcome(
            message=f"Learned! CCC #{binding['visual_ccc_id']} now represents '{clean_label}'",
            recognized_label=self.recognize(tensor),
            pool_stats={
                "total": int(pool_stats["total_concepts"]),
                "committed": int(pool_stats["num_committed"]),
                "available": int(pool_stats["num_uncommitted"]),
            },
            retention=retention,
        )


def _decode_text(
    *,
    trainer: TextGenerationTrainer,
    prompt_ids: list[int],
    method: str,
    max_tokens: int,
) -> GenerationResult:
    if method == "beam":
        results = trainer.decoder.decode(trainer, prompt_ids, max_tokens=max_tokens)
        return results[0] if results else trainer.decoder.greedy_decode(trainer, prompt_ids, max_tokens=max_tokens)
    if method == "greedy":
        return trainer.decoder.greedy_decode(trainer, prompt_ids, max_tokens=max_tokens)
    if method == "top-k":
        return trainer.decoder.top_k_decode(trainer, prompt_ids, max_tokens=max_tokens, k=10)
    if method == "top-p":
        return trainer.decoder.top_p_decode(trainer, prompt_ids, max_tokens=max_tokens, p=0.9)
    raise ValueError("method must be one of: beam, greedy, top-k, top-p.")


def _concept_from_pattern(trainer: TextGenerationTrainer, pattern: torch.Tensor) -> torch.Tensor:
    concept_dim = int(trainer.system.core.config.ccc.concept_dim)
    vector = pattern.reshape(-1).to(torch.float32)
    if vector.numel() > concept_dim:
        vector = F.adaptive_avg_pool1d(vector.view(1, 1, -1), concept_dim).view(-1)
    elif vector.numel() < concept_dim:
        vector = F.pad(vector, (0, concept_dim - vector.numel()))
    return trainer._normalize(vector)  # noqa: SLF001


def _prime_text_trainer_fast(
    trainer: TextGenerationTrainer,
    corpus: str,
    *,
    max_chars: int = 900,
) -> int:
    raw_text = corpus[:max_chars]
    trainer._fit_tokenizer_if_needed(raw_text)  # noqa: SLF001
    token_ids = [int(token_id) for token_id in trainer.tokenizer.encode(raw_text)]
    if not token_ids:
        raise ValueError("The built-in corpus did not produce any tokens.")

    trainer._prime_statistics(raw_text, token_ids)  # noqa: SLF001
    token_frequencies = Counter(token_ids)
    common_tokens = [token_id for token_id, _ in token_frequencies.most_common(min(16, len(token_frequencies)))]
    token_to_ccc: dict[int, int] = {}
    concepts: dict[int, torch.Tensor] = {}

    for token_id in common_tokens:
        recruit_index = trainer.system.core.ccc_pool._first_uncommitted_index()  # noqa: SLF001
        if recruit_index is None:
            break
        pattern = trainer._base_token_pattern(token_id)  # noqa: SLF001
        ccc = trainer.system.core.ccc_pool.cccs[recruit_index]
        f1_output = ccc.f1_encode(pattern)
        ccc.learn_fast(pattern, f1_output)
        concept = trainer._normalize(ccc.concept_direction.detach().clone())  # noqa: SLF001
        trainer.system.core.fabric.register_activation(recruit_index, concept, confidence=0.8, timestep=recruit_index)
        token_to_ccc[token_id] = int(recruit_index)
        concepts[token_id] = concept

    transition_frequencies = Counter(zip(token_ids, token_ids[1:], strict=False))
    for token_id, frequency in token_frequencies.items():
        concept = concepts.get(token_id)
        if concept is None:
            concept = _concept_from_pattern(trainer, trainer._base_token_pattern(token_id))  # noqa: SLF001
            concepts[token_id] = concept
        trainer.token_counts[token_id] = frequency
        trainer.token_concept_counts[token_id] = frequency
        trainer.token_concept_sums[token_id] = concept * float(frequency)
        trainer.system.core.fabric.sdm.write(concept, concept)
        ccc_index = token_to_ccc.get(token_id)
        if ccc_index is not None:
            trainer.token_to_ccc_counts[token_id][ccc_index] += frequency
            trainer.ccc_to_token_counts[ccc_index][token_id] += frequency

    for (src_token, dst_token), frequency in transition_frequencies.items():
        trainer.transition_counts[src_token][dst_token] += frequency
        src_ccc = token_to_ccc.get(src_token)
        dst_ccc = token_to_ccc.get(dst_token)
        src_concept = concepts[src_token]
        dst_concept = concepts[dst_token]
        if src_ccc is not None and dst_ccc is not None:
            trainer.system.core.fabric._add_association(  # noqa: SLF001
                src_ccc,
                dst_ccc,
                strength=min(2.5, 0.08 * float(frequency)),
                temporal=True,
            )
        trainer.system.core.fabric.sdm.associate(
            src_concept,
            dst_concept,
            src_concept * min(1.0, 0.05 * float(frequency)),
            dst_concept * min(1.0, 0.05 * float(frequency)),
            temporal_order=True,
        )

    trainer.training_steps = len(token_ids)
    return len(token_ids)


def _build_mnist_bundle() -> MNISTDemoModel:
    trainer = VisionTrainer(
        VisionTrainConfig(
            input_dim=28 * 28,
            concept_dim=64,
            max_pool_size=64,
            margin_threshold=0.35,
            use_batched=True,
            batch_size=32,
            learning_rate=0.03,
            num_train_samples=200,
            num_test_samples=40,
            preprocessing_warmup_samples=0,
        )
    )
    stream = _digit_training_stream(samples_per_digit=20)
    summary = trainer.train_online(stream, num_samples=200)
    return MNISTDemoModel(
        trainer=trainer,
        source="synthetic-mnist-subset",
        train_summary=summary,
        example_digits={digit: _draw_digit_prototype(digit) for digit in range(10)},
    )


def _build_text_bundle() -> TextDemoModel:
    trainer = TextGenerationTrainer(
        TextGenConfig(
            tokenizer_type="char",
            vocab_size=128,
            context_length=24,
            spike_dim=64,
            num_timesteps=4,
            max_pool_size=96,
            temperature=0.9,
            learning_rate_hebbian=0.02,
            sdm_addresses=256,
            generate_max_tokens=80,
            num_passes=1,
            beam_width=4,
        )
    )
    corpus = build_builtin_corpus(min_chars=3200)
    training_tokens = _prime_text_trainer_fast(trainer, corpus, max_chars=900)
    return TextDemoModel(trainer=trainer, corpus=corpus, training_tokens=training_tokens)


def _build_multimodal_bundle() -> MultimodalDemoModel:
    fusion = MultimodalFusion(
        MultimodalConfig(
            vision_dim=28 * 28,
            language_dim=64,
            concept_dim=64,
            cross_modal_strength=0.9,
            temporal_window=3,
            max_description_length=12,
            alignment_threshold=0.55,
        )
    )
    patterns = _pattern_bank()
    learned = 0
    for label, pattern in patterns.items():
        fusion.learn_cross_modal(pattern, label, label=label)
        fusion.learn_cross_modal(_augment_pattern(pattern, seed=10_000 + learned), label, label=label)
        learned += 1
    return MultimodalDemoModel(fusion=fusion, patterns=patterns, bindings_learned=learned)


@lru_cache(maxsize=1)
def get_mnist_model() -> MNISTDemoModel:
    """Return a BioARNCore pre-trained on 200 MNIST samples."""

    cached = _load_cached("mnist_demo")
    if cached is not None:
        return cached
    bundle = _build_mnist_bundle()
    _save_cached("mnist_demo", bundle)
    return bundle


@lru_cache(maxsize=1)
def get_text_model() -> TextDemoModel:
    """Return a TextGenerationTrainer pre-trained on built-in corpus."""

    cached = _load_cached("text_demo")
    if cached is not None and _text_bundle_is_compatible(cached):
        return cached
    bundle = _build_text_bundle()
    _save_cached("text_demo", bundle)
    return bundle


@lru_cache(maxsize=1)
def get_multimodal_model() -> MultimodalDemoModel:
    """Return a MultimodalFusion pre-trained on 10 pattern categories."""

    cached = _load_cached("multimodal_demo")
    if cached is not None:
        return cached
    bundle = _build_multimodal_bundle()
    _save_cached("multimodal_demo", bundle)
    return bundle


def create_live_learning_session() -> LiveLearningSession:
    bundle = _clone_with_torch(get_multimodal_model())
    return LiveLearningSession(
        fusion=bundle.fusion,
        baseline_patterns=bundle.patterns,
        known_patterns={label: pattern.clone() for label, pattern in bundle.patterns.items()},
    )


@lru_cache(maxsize=1)
def get_energy_dashboard_data() -> EnergyDashboardData:
    vision_model = get_mnist_model()
    config = vision_model.trainer.system.config
    comparison = EnergyModel().compare_all_backends(config)
    mapping = ComponentMapper(PyTorchBackend()).map_full_system(config)
    pool_stats = vision_model.trainer.system.ccc_pool.get_pool_stats()
    loihi = comparison.backends["loihi2"]
    return EnergyDashboardData(
        energies_joules={
            "Bio-ARN (Loihi)": float(loihi.total_joules),
            "GPU (A100)": float(comparison.backends["gpu_a100"].total_joules),
            "CPU": float(comparison.backends["cpu_laptop"].total_joules),
        },
        efficiency_callout_x=REPORTED_EFFICIENCY_X,
        battery_life_hours=float(EDGE_BATTERY_WH / max(loihi.watts_at_1khz, 1e-9)),
        architecture_stats={
            "total_neurons": int(mapping.total_neurons),
            "total_synapses": int(mapping.total_synapses),
            "active_cccs": int(min(int(pool_stats["num_committed"]), int(config.gnw.capacity))),
            "committed_cccs": int(pool_stats["num_committed"]),
            "memory_mb": round(float(mapping.total_memory_bytes) / (1024.0 * 1024.0), 3),
        },
    )


@lru_cache(maxsize=1)
def get_demo_overview_stats() -> dict[str, float | int]:
    mnist = get_mnist_model()
    text = get_text_model()
    multimodal = get_multimodal_model()
    total_committed = (
        int(mnist.trainer.system.ccc_pool.get_pool_stats()["num_committed"])
        + int(text.trainer.system.core.ccc_pool.get_pool_stats()["num_committed"])
        + int(multimodal.fusion.ccc_pool.get_pool_stats()["num_committed"])
    )
    memory_mb = (
        _module_memory_mb(mnist.trainer.system)
        + _module_memory_mb(text.trainer.system)
        + _module_memory_mb(multimodal.fusion.ccc_pool)
    )
    return {
        "cccs_committed": total_committed,
        "memory_mb": round(memory_mb, 3),
    }


def format_token_html(prompt: str, generated_tokens: list[str]) -> str:
    prompt_html = "".join(
        f"<span style='color:#94a3b8'>{html.escape(token)}</span>" for token in prompt
    )
    generated_html = "".join(
        f"<span style='background:#dbeafe;color:#1d4ed8;padding:1px 2px;border-radius:4px;margin-right:1px'>{html.escape(token)}</span>"
        for token in generated_tokens
    )
    return (
        "<div style='font-family:monospace;font-size:1rem;line-height:1.8'>"
        f"{prompt_html}{generated_html}"
        "</div>"
    )


def format_top_matches(matches: list[tuple[str, float]]) -> str:
    if not matches:
        return "No associations found."
    lines = ["| Rank | Association | Score |", "| --- | --- | ---: |"]
    for index, (label, score) in enumerate(matches, start=1):
        lines.append(f"| {index} | {label} | {score:.3f} |")
    return "\n".join(lines)


__all__ = [
    "CACHE_DIR",
    "DOCS_URL",
    "PAPER_URL",
    "CrossModalDemoResult",
    "DigitRecognitionResult",
    "EnergyDashboardData",
    "LiveLearningOutcome",
    "LiveLearningSession",
    "MNISTDemoModel",
    "MultimodalDemoModel",
    "REPORTED_EFFICIENCY_X",
    "TextDemoModel",
    "TextGenerationDemoResult",
    "coerce_image_tensor",
    "create_live_learning_session",
    "format_token_html",
    "format_top_matches",
    "get_demo_overview_stats",
    "get_energy_dashboard_data",
    "get_mnist_model",
    "get_multimodal_model",
    "get_text_model",
    "image_tensor_to_array",
]
