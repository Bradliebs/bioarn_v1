"""Cross-modal binding across vision and language in shared CCC space."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bioarn.config import (
    AudioConfig,
    AudioHierarchyConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    MultimodalFusionConfig,
    PrecisionConfig,
    SDMConfig,
    SpikingConfig,
    TemporalConfig,
)
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.hierarchy.audio_hierarchy import AudioHierarchy
from bioarn.memory.associative_fabric import AssociativeFabric
from bioarn.predictive.precision_weighting import PrecisionWeightedGate
from bioarn.preprocessing.audio import AudioPreprocessor
from bioarn.sensorimotor.language import LanguageEncoder
from bioarn.sensorimotor.vision import VisualEncoder
from bioarn.temporal.sequence_layer import TemporalSequenceLayer
from bioarn.tokenization import CharTokenizer
from bioarn.workspace.gnw import GlobalNeuronalWorkspace, StreamOfConsciousness

from .config import MultimodalConfig


@dataclass(frozen=True)
class CrossModalAssociation:
    """Ranked association retrieved across modalities."""

    source_ccc_id: int
    target_ccc_id: int
    strength: float
    source_modality: str
    target_modality: str
    label: str | None = None
    temporal: bool = False


@dataclass
class MultimodalInput:
    """Container for simultaneous multimodal evidence."""

    vision: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    temporal_context: list[int] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ModalityResult:
    """Compact result for a single modality branch."""

    fired_indices: list[int]
    top_confidence: float
    concept_direction: torch.Tensor | None


@dataclass
class MultimodalOutput:
    """Unified workspace decision across modalities."""

    winner_modality: str
    concept_direction: torch.Tensor
    confidence: float
    per_modality: dict[str, ModalityResult] = field(default_factory=dict)
    cross_modal_agreement: float = 0.0
    precision: float = 1.0


class MultimodalFusionEngine(nn.Module):
    """Unified multimodal processing via GNW workspace competition."""

    _OFFSETS = {
        "vision": 0,
        "audio": 100_000,
        "temporal": 200_000,
    }
    _PROVISIONAL_IDS = {
        "vision": -1,
        "audio": -2,
        "temporal": -3,
    }

    def __init__(self, config: MultimodalFusionConfig):
        super().__init__()
        self.config = copy.deepcopy(config)
        workspace_config = copy.deepcopy(self.config.workspace)
        workspace_config.capacity = int(self.config.workspace_size)
        workspace_config.concept_dim = int(self.config.concept_dim)

        self.vision_pool: CCCPool | None = None
        self.audio_pool: CCCPool | None = None
        self.audio_preprocessor: AudioPreprocessor | None = None
        self.audio_hierarchy: AudioHierarchy | None = None
        if self.config.audio_enabled:
            self.audio_preprocessor = AudioPreprocessor(copy.deepcopy(self.config.audio))
            self.audio_hierarchy = AudioHierarchy(copy.deepcopy(self.config.audio_hierarchy))

        self.temporal_layer: TemporalSequenceLayer | None = None
        if self.config.temporal_enabled:
            temporal_config = copy.deepcopy(self.config.temporal)
            temporal_config.concept_dim = int(self.config.concept_dim)
            self.temporal_layer = TemporalSequenceLayer(temporal_config)

        self.workspace = GlobalNeuronalWorkspace(workspace_config)
        precision_config = copy.deepcopy(self.config.precision or PrecisionConfig(enabled=True))
        precision_config.enabled = True
        self.precision_gate = PrecisionWeightedGate(precision_config)
        self.precision_gate.set_pool_size(int(self.config.precision.pool_size if self.config.precision else 8))

        association_ccc = CCCConfig(
            input_dim=int(self.config.concept_dim),
            concept_dim=int(self.config.concept_dim),
            num_f1_features=max(16, int(self.config.concept_dim // 2)),
            f1_top_k=max(4, int(self.config.concept_dim // 8)),
            max_pool_size=max(
                8,
                int(self.config.vision_pool_size + self.config.audio_pool_size + self.config.concept_dim),
            ),
        )
        self.fabric = AssociativeFabric(copy.deepcopy(self.config.sdm), association_ccc)

        self.timestep = 0
        self._winner_counts: Counter[str] = Counter()
        self._agreement_history: list[float] = []
        self._temporal_surprise_history: list[float] = []
        self._binding_events = 0
        self._last_output: MultimodalOutput | None = None

    @property
    def last_output(self) -> MultimodalOutput | None:
        return self._last_output

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if float(flattened.norm().item()) <= 1e-8:
            return torch.zeros_like(flattened)
        return normalize(flattened.unsqueeze(0)).squeeze(0)

    def _align_dim(self, vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if flattened.numel() > self.config.concept_dim:
            flattened = flattened[: self.config.concept_dim]
        elif flattened.numel() < self.config.concept_dim:
            flattened = F.pad(flattened, (0, self.config.concept_dim - flattened.numel()))
        return self._normalize(flattened)

    def _committed_count(self, pool: CCCPool | None) -> int:
        if pool is None:
            return 0
        return sum(bool(ccc.is_committed.item()) for ccc in pool.cccs)

    def _pool_config(self, input_dim: int, pool_size: int) -> tuple[CCCConfig, MarginGateConfig]:
        concept_dim = int(self.config.concept_dim)
        num_f1_features = max(16, min(int(input_dim), max(concept_dim * 2, 32)))
        ccc_config = CCCConfig(
            input_dim=int(max(1, input_dim)),
            concept_dim=concept_dim,
            num_f1_features=num_f1_features,
            f1_top_k=max(4, min(num_f1_features, max(8, num_f1_features // 4))),
            fast_lr=1.0,
            slow_lr=float(self.config.learning_rate),
            feedback_lr=float(self.config.learning_rate),
            max_pool_size=int(max(1, pool_size)),
        )
        margin = MarginGateConfig(
            theta_margin=float(self.config.margin_threshold),
            theta_margin_lr=0.01,
            theta_resonance=min(0.95, float(self.config.margin_threshold) + 0.3),
        )
        return ccc_config, margin

    def _ensure_vision_pool(self, vision: torch.Tensor) -> CCCPool:
        if self.vision_pool is None:
            ccc_config, margin = self._pool_config(int(vision.numel()), int(self.config.vision_pool_size))
            self.vision_pool = CCCPool(ccc_config, margin)
        return self.vision_pool

    def _ensure_audio_pool(self) -> CCCPool:
        if self.audio_pool is None:
            if self.audio_hierarchy is None:
                raise RuntimeError("Audio hierarchy is not available.")
            ccc_config, margin = self._pool_config(
                int(self.audio_hierarchy.output_dim),
                int(self.config.audio_pool_size),
            )
            self.audio_pool = CCCPool(ccc_config, margin)
        return self.audio_pool

    @staticmethod
    def _confidence_score(confidence: torch.Tensor) -> float:
        return float(confidence.reshape(-1).mean().item())

    def _aggregate_pool_result(self, pool: CCCPool, pool_output) -> ModalityResult:
        if not pool_output.fired_indices:
            return ModalityResult([], 0.0, None)
        directions = torch.stack(
            [pool.cccs[index].concept_direction.detach().clone() for index in pool_output.fired_indices],
            dim=0,
        )
        if pool_output.winner_confidences.numel() == len(pool_output.fired_indices):
            weights = pool_output.winner_confidences.to(directions).reshape(-1, 1)
        else:
            weights = torch.ones(len(pool_output.fired_indices), 1, device=directions.device, dtype=directions.dtype)
        concept = self._normalize((directions * weights).sum(dim=0))
        return ModalityResult(
            fired_indices=[int(index) for index in pool_output.fired_indices],
            top_confidence=float(weights.max().item()),
            concept_direction=concept,
        )

    def _provisional_result(self, tensor: torch.Tensor, confidence: float = 0.25) -> ModalityResult:
        return ModalityResult(
            fired_indices=[],
            top_confidence=float(confidence),
            concept_direction=self._align_dim(tensor),
        )

    def _preview_vision(self, vision: torch.Tensor) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None]:
        pool = self._ensure_vision_pool(vision)
        flat = vision.detach().reshape(-1).to(torch.float32)
        if self._committed_count(pool) == 0:
            return self._provisional_result(flat), [], None
        pool_output = pool.preview(flat)
        result = self._aggregate_pool_result(pool, pool_output)
        activations = [
            (
                int(index),
                pool.cccs[index].concept_direction.detach().clone(),
                self._confidence_score(pool_output.outputs[index].confidence),
            )
            for index in pool_output.fired_indices
        ]
        top_index = int(pool_output.fired_indices[0]) if pool_output.fired_indices else None
        return result, activations, top_index

    def _learn_vision(
        self,
        vision: torch.Tensor,
        *,
        learning_rate_multiplier: float,
    ) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None]:
        pool = self._ensure_vision_pool(vision)
        flat = vision.detach().reshape(-1).to(torch.float32)
        pool_output = pool(
            flat,
            timestep=self.timestep,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        result = self._aggregate_pool_result(pool, pool_output)
        activations = [
            (
                int(index),
                pool.cccs[index].concept_direction.detach().clone(),
                self._confidence_score(pool_output.outputs[index].confidence),
            )
            for index in pool_output.fired_indices
        ]
        top_index = int(pool_output.fired_indices[0]) if pool_output.fired_indices else pool_output.recruited_index
        if result.concept_direction is None:
            result = self._provisional_result(flat, confidence=0.1)
        return result, activations, top_index

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.audio_preprocessor is None or self.audio_hierarchy is None:
            raise RuntimeError("Audio processing is disabled.")
        audio_tensor = audio.detach().to(torch.float32)
        if audio_tensor.dim() == 1:
            mel = self.audio_preprocessor.waveform_to_mel(audio_tensor)
        elif audio_tensor.dim() == 2:
            mel = audio_tensor
        elif audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
            mel = audio_tensor.squeeze(0)
        else:
            raise ValueError("audio must be a waveform or mel spectrogram.")
        return self.audio_hierarchy(mel)

    def _preview_audio(self, audio: torch.Tensor) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None]:
        encoded = self._encode_audio(audio)
        pool = self._ensure_audio_pool()
        if self._committed_count(pool) == 0:
            return self._provisional_result(encoded), [], None
        pool_output = pool.preview(encoded)
        result = self._aggregate_pool_result(pool, pool_output)
        activations = [
            (
                int(index),
                pool.cccs[index].concept_direction.detach().clone(),
                self._confidence_score(pool_output.outputs[index].confidence),
            )
            for index in pool_output.fired_indices
        ]
        top_index = int(pool_output.fired_indices[0]) if pool_output.fired_indices else None
        return result, activations, top_index

    def _learn_audio(
        self,
        audio: torch.Tensor,
        *,
        learning_rate_multiplier: float,
    ) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None]:
        encoded = self._encode_audio(audio)
        pool = self._ensure_audio_pool()
        pool_output = pool(
            encoded,
            timestep=self.timestep,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        result = self._aggregate_pool_result(pool, pool_output)
        activations = [
            (
                int(index),
                pool.cccs[index].concept_direction.detach().clone(),
                self._confidence_score(pool_output.outputs[index].confidence),
            )
            for index in pool_output.fired_indices
        ]
        top_index = int(pool_output.fired_indices[0]) if pool_output.fired_indices else pool_output.recruited_index
        if result.concept_direction is None:
            result = self._provisional_result(encoded, confidence=0.1)
        return result, activations, top_index

    @staticmethod
    def _sanitize_temporal_indices(indices: list[int] | None, concept_dim: int) -> list[int]:
        if indices is None:
            return []
        sanitized = {
            int(index)
            for index in indices
            if 0 <= int(index) < int(concept_dim)
        }
        return sorted(sanitized)

    def _temporal_vector(self, indices: list[int]) -> torch.Tensor:
        vector = torch.zeros(self.config.concept_dim, dtype=torch.float32)
        if indices:
            vector[torch.tensor(indices, dtype=torch.long)] = 1.0
        return vector

    def _preview_temporal(self, indices: list[int]) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None, float]:
        sanitized = self._sanitize_temporal_indices(indices, self.config.concept_dim)
        if self.temporal_layer is None or not sanitized:
            return ModalityResult([], 0.0, None), [], None, 0.0
        actual = self._temporal_vector(sanitized)
        prediction = self.temporal_layer.last_prediction.detach().clone()
        surprise = float(self.temporal_layer.temporal_surprise(sanitized))
        concept = self._normalize((0.7 * actual) + (0.3 * prediction))
        confidence = max(0.05, 1.0 - surprise)
        top_index = sanitized[0]
        return (
            ModalityResult(fired_indices=sanitized, top_confidence=confidence, concept_direction=concept),
            [(top_index, concept.detach().clone(), confidence)],
            top_index,
            surprise,
        )

    def _learn_temporal(self, indices: list[int]) -> tuple[ModalityResult, list[tuple[int, torch.Tensor, float]], int | None, float]:
        sanitized = self._sanitize_temporal_indices(indices, self.config.concept_dim)
        if self.temporal_layer is None or not sanitized:
            return ModalityResult([], 0.0, None), [], None, 0.0
        actual = self._temporal_vector(sanitized)
        temporal_output = self.temporal_layer.observe_frame(actual, sanitized)
        concept = self._normalize((0.7 * actual) + (0.3 * temporal_output.prediction))
        confidence = max(0.05, 1.0 - float(temporal_output.surprise))
        top_index = sanitized[0]
        return (
            ModalityResult(fired_indices=sanitized, top_confidence=confidence, concept_direction=concept),
            [(top_index, concept.detach().clone(), confidence)],
            top_index,
            float(temporal_output.surprise),
        )

    def _global_id(self, modality: str, local_index: int | None) -> int:
        if local_index is None:
            return int(self._PROVISIONAL_IDS[modality])
        return int(self._OFFSETS[modality] + int(local_index))

    def _decode_global_id(self, global_id: int) -> tuple[str | None, int | None]:
        for modality, provisional_id in self._PROVISIONAL_IDS.items():
            if int(global_id) == int(provisional_id):
                return modality, None
        for modality, offset in self._OFFSETS.items():
            if int(global_id) >= offset and int(global_id) < offset + 100_000:
                return modality, int(global_id - offset)
        return None, None

    def _direction_for_global_id(self, global_id: int) -> torch.Tensor | None:
        stored = self.fabric.concept_directions.get(int(global_id))
        if stored is not None:
            return stored.detach().clone()
        modality, local = self._decode_global_id(global_id)
        if modality == "vision" and self.vision_pool is not None and local is not None and 0 <= local < len(self.vision_pool.cccs):
            return self.vision_pool.cccs[local].concept_direction.detach().clone()
        if modality == "audio" and self.audio_pool is not None and local is not None and 0 <= local < len(self.audio_pool.cccs):
            return self.audio_pool.cccs[local].concept_direction.detach().clone()
        if modality == "temporal" and local is not None:
            return self._temporal_vector([local])
        return None

    def _collect_modalities(
        self,
        inputs: MultimodalInput,
        *,
        learn: bool,
        learning_rate_multiplier: float = 1.0,
    ) -> tuple[
        dict[str, ModalityResult],
        dict[str, list[tuple[int, torch.Tensor, float]]],
        dict[str, int | None],
        float,
    ]:
        results: dict[str, ModalityResult] = {}
        activations: dict[str, list[tuple[int, torch.Tensor, float]]] = {}
        top_indices: dict[str, int | None] = {}
        temporal_surprise = 0.0

        if self.config.vision_enabled and inputs.vision is not None:
            if learn:
                result, modality_activations, top_index = self._learn_vision(
                    inputs.vision,
                    learning_rate_multiplier=learning_rate_multiplier,
                )
            else:
                result, modality_activations, top_index = self._preview_vision(inputs.vision)
            results["vision"] = result
            activations["vision"] = modality_activations
            top_indices["vision"] = top_index

        if self.config.audio_enabled and inputs.audio is not None:
            if learn:
                result, modality_activations, top_index = self._learn_audio(
                    inputs.audio,
                    learning_rate_multiplier=learning_rate_multiplier,
                )
            else:
                result, modality_activations, top_index = self._preview_audio(inputs.audio)
            results["audio"] = result
            activations["audio"] = modality_activations
            top_indices["audio"] = top_index

        if self.config.temporal_enabled and inputs.temporal_context is not None:
            if learn:
                result, modality_activations, top_index, temporal_surprise = self._learn_temporal(inputs.temporal_context)
            else:
                result, modality_activations, top_index, temporal_surprise = self._preview_temporal(inputs.temporal_context)
            results["temporal"] = result
            activations["temporal"] = modality_activations
            top_indices["temporal"] = top_index

        return results, activations, top_indices, temporal_surprise

    def _cross_modal_agreement(self, results: dict[str, ModalityResult]) -> float:
        active = [result for result in results.values() if result.concept_direction is not None]
        if len(active) <= 1:
            return 1.0 if active else 0.0
        similarities: list[float] = []
        for left_index, left_result in enumerate(active[:-1]):
            left = left_result.concept_direction
            if left is None:
                continue
            for right_result in active[left_index + 1 :]:
                right = right_result.concept_direction
                if right is None:
                    continue
                similarity = float(F.cosine_similarity(left.unsqueeze(0).to(right), right.unsqueeze(0)).item())
                similarities.append(max(0.0, min(1.0, 0.5 * (similarity + 1.0))))
        return float(sum(similarities) / len(similarities)) if similarities else 0.0

    def _candidate_support(self, modality: str, results: dict[str, ModalityResult]) -> float:
        source = results[modality].concept_direction
        if source is None:
            return 0.0
        others = [
            other.concept_direction
            for other_modality, other in results.items()
            if other_modality != modality and other.concept_direction is not None
        ]
        if not others:
            return 1.0
        support = [
            max(
                0.0,
                min(
                    1.0,
                    0.5 * (
                        float(F.cosine_similarity(source.unsqueeze(0).to(other), other.unsqueeze(0)).item()) + 1.0
                    ),
                ),
            )
            for other in others
        ]
        return float(sum(support) / len(support)) if support else 1.0

    def _workspace_candidates(
        self,
        results: dict[str, ModalityResult],
        top_indices: dict[str, int | None],
    ) -> list[tuple[int, torch.Tensor, float]]:
        active_modalities = [
            modality for modality, result in results.items()
            if result.concept_direction is not None and result.top_confidence > 0.0
        ]
        candidates: list[tuple[int, torch.Tensor, float]] = []
        for modality in active_modalities:
            result = results[modality]
            if result.concept_direction is None:
                continue
            support = self._candidate_support(modality, results)
            confidence = float(result.top_confidence)
            if len(active_modalities) > 1:
                confidence = (
                    (1.0 - float(self.config.cross_modal_weight)) * confidence
                    + (float(self.config.cross_modal_weight) * support)
                )
            candidates.append(
                (
                    self._global_id(modality, top_indices.get(modality)),
                    result.concept_direction.detach().clone(),
                    float(max(0.05, min(1.0, confidence))),
                )
            )
        return candidates

    def _fused_target(self, results: dict[str, ModalityResult], workspace_direction: torch.Tensor | None) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        weights: list[float] = []
        for result in results.values():
            if result.concept_direction is None:
                continue
            vectors.append(result.concept_direction)
            weights.append(max(0.05, float(result.top_confidence)))
        if workspace_direction is not None:
            vectors.append(workspace_direction.detach().clone())
            weights.append(1.0)
        if not vectors:
            return torch.zeros(self.config.concept_dim, dtype=torch.float32)
        stacked = torch.stack(vectors, dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).unsqueeze(-1)
        return self._normalize((stacked * weight_tensor).sum(dim=0))

    def _align_pool_concepts(
        self,
        pool: CCCPool | None,
        activations: list[tuple[int, torch.Tensor, float]],
        target: torch.Tensor,
        *,
        learning_rate: float,
    ) -> None:
        if pool is None or not activations or learning_rate <= 0.0:
            return
        for local_index, _, confidence in activations:
            if local_index < 0 or local_index >= len(pool.cccs):
                continue
            ccc = pool.cccs[local_index]
            if not bool(ccc.is_committed.item()):
                continue
            current = ccc.concept_direction.detach().clone().to(target)
            updated = self._normalize(
                ((1.0 - learning_rate) * current) + (learning_rate * float(max(0.05, confidence)) * target)
            )
            ccc.concept_direction.copy_(updated.to(ccc.concept_direction))

    def _register_bindings(
        self,
        activations: dict[str, list[tuple[int, torch.Tensor, float]]],
    ) -> None:
        registered = 0
        for modality, modality_activations in activations.items():
            for local_index, direction, confidence in modality_activations:
                global_id = self._global_id(modality, local_index)
                self.fabric.register_activation(
                    global_id,
                    direction,
                    max(0.05, float(confidence)),
                    self.timestep,
                )
                registered += 1
        if registered >= 2:
            self.fabric.form_associations(self.timestep)
            self._binding_events += 1

    def _project_associations(
        self,
        winner_direction: torch.Tensor,
        per_modality: dict[str, ModalityResult],
    ) -> dict[str, ModalityResult]:
        projected = dict(per_modality)
        associates = self.fabric.retrieve_associates(winner_direction, k=12)
        for global_id, strength in zip(associates.indices, associates.strengths, strict=False):
            modality, local_index = self._decode_global_id(int(global_id))
            if modality is None or local_index is None:
                continue
            if modality in projected and projected[modality].concept_direction is not None:
                continue
            direction = self._direction_for_global_id(int(global_id))
            if direction is None:
                continue
            projected[modality] = ModalityResult(
                fired_indices=[int(local_index)],
                top_confidence=float(max(0.0, min(1.0, strength))),
                concept_direction=self._normalize(direction),
            )
        return projected

    def _finalize_output(
        self,
        candidates: list[tuple[int, torch.Tensor, float]],
        per_modality: dict[str, ModalityResult],
        *,
        agreement: float,
        precision: float,
    ) -> MultimodalOutput:
        if candidates:
            self.workspace.update(candidates, timestep=self.timestep)
            broadcast = self.workspace.broadcast()
        else:
            broadcast = self.workspace.broadcast()

        if broadcast.indices:
            winner_id = int(broadcast.indices[0])
            winner_modality, _ = self._decode_global_id(winner_id)
            winner_direction = broadcast.directions[0].detach().clone()
            confidence = float(self.workspace.slots[0].confidence) if self.workspace.slots else 0.0
        elif candidates:
            winner_id, winner_direction, confidence = max(candidates, key=lambda candidate: candidate[2])
            winner_modality, _ = self._decode_global_id(int(winner_id))
        else:
            winner_modality, winner_direction, confidence = "none", torch.zeros(self.config.concept_dim), 0.0

        if winner_modality is None:
            winner_modality = "none"
        if winner_modality != "none":
            per_modality = self._project_associations(winner_direction, per_modality)

        output = MultimodalOutput(
            winner_modality=str(winner_modality),
            concept_direction=self._align_dim(winner_direction),
            confidence=float(max(0.0, min(1.0, confidence))),
            per_modality=per_modality,
            cross_modal_agreement=float(max(0.0, min(1.0, agreement))),
            precision=float(max(0.0, min(1.0, precision))),
        )
        self._winner_counts[output.winner_modality] += 1
        self._agreement_history.append(float(output.cross_modal_agreement))
        self._last_output = output
        return output

    @torch.no_grad()
    def process(self, inputs: MultimodalInput) -> MultimodalOutput:
        """Process multimodal inputs through workspace competition."""

        results, _, top_indices, temporal_surprise = self._collect_modalities(inputs, learn=False)
        active_results = {
            modality: result
            for modality, result in results.items()
            if result.concept_direction is not None and result.top_confidence > 0.0
        }
        if not active_results:
            output = MultimodalOutput(
                winner_modality="none",
                concept_direction=torch.zeros(self.config.concept_dim, dtype=torch.float32),
                confidence=0.0,
                per_modality={},
                cross_modal_agreement=0.0,
                precision=float(self.precision_gate.current_precision),
            )
            self._last_output = output
            return output

        agreement = self._cross_modal_agreement(active_results)
        candidates = self._workspace_candidates(active_results, top_indices)
        precision = self.precision_gate.preview_pool_output(
            [int(candidate[0]) for candidate in candidates],
            lateral_error=1.0 - agreement,
            hierarchy_error=temporal_surprise,
        )
        output = self._finalize_output(
            candidates,
            results,
            agreement=agreement,
            precision=float(precision),
        )
        self.timestep += 1
        return output

    @torch.no_grad()
    def learn(
        self,
        inputs: MultimodalInput,
        *,
        learning_rate_multiplier: float = 1.0,
    ) -> None:
        """Hebbian learning across modalities, gated by workspace broadcast."""

        results, activations, top_indices, temporal_surprise = self._collect_modalities(
            inputs,
            learn=True,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        active_results = {
            modality: result
            for modality, result in results.items()
            if result.concept_direction is not None and result.top_confidence > 0.0
        }
        if not active_results:
            self._last_output = None
            return

        agreement = self._cross_modal_agreement(active_results)
        candidates = self._workspace_candidates(active_results, top_indices)
        precision = self.precision_gate.observe_pool_output(
            [int(candidate[0]) for candidate in candidates],
            lateral_error=1.0 - agreement,
            hierarchy_error=temporal_surprise,
        )

        self.workspace.update(candidates, timestep=self.timestep)
        winner_direction = self.workspace.slots[0].direction.detach().clone() if self.workspace.slots else None
        fused_target = self._fused_target(active_results, winner_direction)
        effective_lr = float(self.config.learning_rate) * float(learning_rate_multiplier) * float(precision)
        self._align_pool_concepts(self.vision_pool, activations.get("vision", []), fused_target, learning_rate=effective_lr)
        self._align_pool_concepts(self.audio_pool, activations.get("audio", []), fused_target, learning_rate=effective_lr)
        if self.workspace.slots:
            self.workspace.inject(self.workspace.slots[0].ccc_index, fused_target, priority=max(0.1, agreement))

        self._register_bindings(activations)
        if self.temporal_layer is not None and results.get("temporal") is not None:
            self._temporal_surprise_history.append(float(temporal_surprise))

        refreshed_results = dict(results)
        if activations.get("vision") and self.vision_pool is not None:
            refreshed_results["vision"] = self._aggregate_pool_result(
                self.vision_pool,
                type(
                    "_PoolProxy",
                    (),
                    {
                        "fired_indices": [index for index, _, _ in activations["vision"]],
                        "winner_confidences": torch.tensor(
                            [confidence for _, _, confidence in activations["vision"]],
                            dtype=torch.float32,
                        ),
                    },
                )(),
            )
        if activations.get("audio") and self.audio_pool is not None:
            refreshed_results["audio"] = self._aggregate_pool_result(
                self.audio_pool,
                type(
                    "_PoolProxy",
                    (),
                    {
                        "fired_indices": [index for index, _, _ in activations["audio"]],
                        "winner_confidences": torch.tensor(
                            [confidence for _, _, confidence in activations["audio"]],
                            dtype=torch.float32,
                        ),
                    },
                )(),
            )
        if "temporal" in refreshed_results and refreshed_results["temporal"].concept_direction is not None:
            refreshed_results["temporal"] = ModalityResult(
                fired_indices=list(refreshed_results["temporal"].fired_indices),
                top_confidence=float(refreshed_results["temporal"].top_confidence),
                concept_direction=fused_target.detach().clone(),
            )

        self._last_output = self._finalize_output(
            candidates,
            refreshed_results,
            agreement=agreement,
            precision=float(precision),
        )
        self.timestep += 1

    @property
    def stats(self) -> dict[str, Any]:
        """Per-modality and fusion statistics."""

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "timestep": int(self.timestep),
            "precision": float(self.precision_gate.current_precision),
            "mean_agreement": _mean(self._agreement_history[-64:]),
            "winner_counts": dict(self._winner_counts),
            "bindings_formed": int(self._binding_events),
            "workspace": self.workspace.get_stats(),
            "associations": self.fabric.get_stats(),
            "vision": None if self.vision_pool is None else self.vision_pool.get_pool_stats(),
            "audio": None if self.audio_pool is None else self.audio_pool.get_pool_stats(),
            "temporal_mean_surprise": _mean(self._temporal_surprise_history[-64:]),
        }


class MultimodalFusion:
    """Bind visual and linguistic concepts in a shared semantic space."""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.device = torch.device("cpu")
        self.input_shape = self._infer_visual_shape(config.vision_dim)
        self.shared_input_dim = max(int(config.language_dim), int(config.concept_dim))

        spiking = SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0)
        self.visual_encoder = VisualEncoder(
            input_shape=self.input_shape,
            output_dim=self.shared_input_dim,
            config=spiking,
        )
        self.tokenizer = CharTokenizer()
        self.language_encoder = LanguageEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=max(8, min(32, int(config.language_dim) // 8 or 8)),
            output_dim=self.shared_input_dim,
            config=spiking,
        )

        ccc_config = CCCConfig(
            input_dim=self.shared_input_dim,
            concept_dim=int(config.concept_dim),
            num_f1_features=max(32, self.shared_input_dim // 2),
            f1_top_k=max(4, self.shared_input_dim // 8),
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=max(64, int(config.concept_dim) * 4),
        )
        margin_config = MarginGateConfig(theta_margin=0.3, theta_margin_lr=0.001, theta_resonance=0.6)
        sdm_config = SDMConfig(
            address_dim=max(64, int(config.concept_dim) * 2),
            hamming_radius=max(8, int(config.concept_dim) // 4),
            num_hard_locations=max(128, int(config.concept_dim) * 4),
            data_dim=int(config.concept_dim),
            decay_rate=0.999,
            stdp_window=max(1, int(config.temporal_window)),
        )
        gnw_config = GNWConfig(capacity=7, broadcast_gain=2.0, fatigue_rate=0.05, fatigue_threshold=0.1, competition_temp=0.7)

        self.ccc_pool = CCCPool(ccc_config, margin_config)
        self.fabric = AssociativeFabric(sdm_config, ccc_config)
        self.gnw = GlobalNeuronalWorkspace(gnw_config)
        self.stream = StreamOfConsciousness(self.gnw, gnw_config)
        self.timestep = 0

        self.ccc_modalities: dict[int, set[str]] = defaultdict(set)
        self.label_to_ccc: dict[tuple[str, str], int] = {}
        self.labels_by_ccc: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self.text_by_ccc: dict[int, str] = {}
        self.feature_prototypes: dict[tuple[int, str], Tensor] = {}
        self.feature_counts: Counter[tuple[int, str]] = Counter()
        self.visual_patterns: dict[int, Tensor] = {}
        self.visual_pattern_counts: Counter[int] = Counter()
        self.explicit_bindings: dict[tuple[int, int], float] = {}

    @staticmethod
    def _infer_visual_shape(input_dim: int) -> tuple[int, int, int]:
        if input_dim <= 0:
            raise ValueError("vision_dim must be positive.")
        side = int(round(math.sqrt(int(input_dim))))
        if side * side == int(input_dim):
            return (1, side, side)
        return (1, 1, int(input_dim))

    @staticmethod
    def _normalize(vector: Tensor) -> Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if float(flattened.norm().item()) <= 1e-8:
            return torch.zeros_like(flattened)
        return normalize(flattened.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _cosine(left: Tensor, right: Tensor) -> float:
        left_norm = MultimodalFusion._normalize(left)
        right_norm = MultimodalFusion._normalize(right)
        if float(left_norm.norm().item()) == 0.0 or float(right_norm.norm().item()) == 0.0:
            return 0.0
        return float(F.cosine_similarity(left_norm.unsqueeze(0), right_norm.unsqueeze(0)).item())

    @staticmethod
    def _canonical_modality(modality: str) -> str:
        lowered = modality.strip().lower()
        if lowered in {"vision", "visual", "image"}:
            return "vision"
        if lowered in {"language", "text", "linguistic"}:
            return "text"
        raise ValueError(f"Unsupported modality: {modality}")

    def _prepare_visual(self, visual_input: Tensor) -> Tensor:
        tensor = visual_input.detach().to(torch.float32)
        if tensor.dim() == 1 and tensor.numel() == int(torch.tensor(self.input_shape).prod().item()):
            return tensor.view(1, *self.input_shape)
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == 3:
            if tensor.shape == self.input_shape:
                return tensor.unsqueeze(0)
            if tensor.shape[0] == 1 and tensor.shape[1:] == self.input_shape[1:]:
                return tensor.unsqueeze(0)
        if tensor.dim() == 4:
            return tensor
        raise ValueError("visual_input must be shaped like an image or flattened image.")

    def _prepare_text_tokens(self, text_input: str | Tensor | Iterable[int]) -> tuple[Tensor, str]:
        if isinstance(text_input, str):
            text = text_input.strip()
            if not text:
                raise ValueError("text_input must be non-empty.")
            token_ids = self.tokenizer.encode(text)
            return torch.tensor(token_ids, dtype=torch.long), text
        if isinstance(text_input, torch.Tensor):
            tokens = text_input.detach().clone().long().reshape(-1).remainder(self.tokenizer.vocab_size)
            return tokens, self.tokenizer.decode(tokens.tolist()).strip()
        tokens = torch.tensor(list(text_input), dtype=torch.long).reshape(-1).remainder(self.tokenizer.vocab_size)
        return tokens, self.tokenizer.decode(tokens.tolist()).strip()

    def _encode_visual(self, visual_input: Tensor) -> Tensor:
        frame = self._prepare_visual(visual_input)
        self.visual_encoder.reset_state()
        output = self.visual_encoder(frame, prev_frame=torch.zeros_like(frame))
        features = output.features.squeeze(0) if output.features.dim() == 2 and output.features.shape[0] == 1 else output.features
        return self._normalize(features)

    def _encode_text(self, text_input: str | Tensor | Iterable[int]) -> tuple[Tensor, str]:
        tokens, text = self._prepare_text_tokens(text_input)
        self.language_encoder.reset_state()
        output = self.language_encoder(tokens)
        features = output.features.squeeze(0) if output.features.dim() == 2 and output.features.shape[0] == 1 else output.features
        return self._normalize(features), text

    def _register_feature(self, ccc_id: int, modality: str, feature: Tensor) -> None:
        key = (int(ccc_id), modality)
        normalized = self._normalize(feature)
        if key not in self.feature_prototypes:
            self.feature_prototypes[key] = normalized
            self.feature_counts[key] = 1
            return
        count = self.feature_counts[key]
        updated = ((self.feature_prototypes[key] * count) + normalized) / float(count + 1)
        self.feature_prototypes[key] = self._normalize(updated)
        self.feature_counts[key] = count + 1

    def _register_visual_pattern(self, ccc_id: int, visual_input: Tensor) -> None:
        frame = self._prepare_visual(visual_input).squeeze(0)
        if int(ccc_id) not in self.visual_patterns:
            self.visual_patterns[int(ccc_id)] = frame.detach().clone()
            self.visual_pattern_counts[int(ccc_id)] = 1
            return
        count = self.visual_pattern_counts[int(ccc_id)]
        updated = ((self.visual_patterns[int(ccc_id)] * count) + frame) / float(count + 1)
        self.visual_patterns[int(ccc_id)] = updated.detach().clone()
        self.visual_pattern_counts[int(ccc_id)] = count + 1

    def _nearest_visual_pattern(self, visual_input: Tensor) -> tuple[int, float] | None:
        if not self.visual_patterns:
            return None
        frame = self._prepare_visual(visual_input).squeeze(0)
        candidates = [
            ccc_id
            for ccc_id, modalities in self.ccc_modalities.items()
            if "vision" in modalities and ccc_id in self.visual_patterns
        ]
        if not candidates:
            return None
        similarities = [
            (ccc_id, self._cosine(frame, self.visual_patterns[ccc_id]))
            for ccc_id in candidates
        ]
        best_id, best_similarity = max(similarities, key=lambda item: item[1])
        return int(best_id), float(best_similarity)

    def _register_label(self, ccc_id: int, modality: str, label: str | None) -> None:
        if label is None:
            return
        clean = label.strip()
        if not clean:
            return
        self.label_to_ccc[(modality, clean)] = int(ccc_id)
        self.labels_by_ccc[int(ccc_id)][clean] += 1
        if modality == "text":
            self.text_by_ccc[int(ccc_id)] = clean

    def _primary_label(self, ccc_id: int) -> str | None:
        counts = self.labels_by_ccc.get(int(ccc_id))
        if counts:
            return counts.most_common(1)[0][0]
        return self.text_by_ccc.get(int(ccc_id))

    def _recruit_ccc(self, feature: Tensor, modality: str, label: str | None) -> tuple[int, float]:
        recruit_index = self.ccc_pool._first_uncommitted_index()
        if recruit_index is None:
            fallback = self._nearest_ccc(feature, modality)
            if fallback is not None:
                return fallback
            raise RuntimeError("No CCC capacity available for multimodal fusion.")
        ccc = self.ccc_pool.cccs[recruit_index]
        f1_output = ccc.f1_encode(feature)
        ccc.learn_fast(feature, f1_output)
        self.ccc_modalities[recruit_index].add(modality)
        self._register_feature(recruit_index, modality, feature)
        self._register_label(recruit_index, modality, label)
        return int(recruit_index), 1.0

    def _nearest_ccc(self, feature: Tensor, modality: str) -> tuple[int, float] | None:
        candidates = [
            ccc_id
            for ccc_id, modalities in self.ccc_modalities.items()
            if modality in modalities and (ccc_id, modality) in self.feature_prototypes
        ]
        if not candidates:
            return None
        similarities = [
            (ccc_id, self._cosine(feature, self.feature_prototypes[(ccc_id, modality)]))
            for ccc_id in candidates
        ]
        best_id, best_similarity = max(similarities, key=lambda item: item[1])
        return int(best_id), float(best_similarity)

    def _resolve_ccc(
        self,
        feature: Tensor,
        modality: str,
        *,
        label: str | None = None,
        learn: bool = True,
    ) -> tuple[int | None, float]:
        if label is not None:
            known = self.label_to_ccc.get((modality, label.strip()))
            if known is not None:
                self.ccc_modalities[int(known)].add(modality)
                self._register_feature(int(known), modality, feature)
                self._register_label(int(known), modality, label)
                return int(known), 1.0
            if learn:
                return self._recruit_ccc(feature, modality, label)
            return None, 0.0

        nearest = self._nearest_ccc(feature, modality)
        if nearest is not None and nearest[1] >= float(self.config.alignment_threshold):
            ccc_id, similarity = nearest
            self.ccc_modalities[int(ccc_id)].add(modality)
            self._register_feature(int(ccc_id), modality, feature)
            self._register_label(int(ccc_id), modality, label)
            return int(ccc_id), float(similarity)

        if not learn:
            if nearest is None:
                return None, 0.0
            min_similarity = max(float(self.config.alignment_threshold), 0.55)
            if nearest[1] < min_similarity:
                return None, 0.0
            return nearest[0], nearest[1]

        return self._recruit_ccc(feature, modality, label)

    def _resolve_visual_ccc(
        self,
        visual_input: Tensor,
        feature: Tensor,
        *,
        label: str | None = None,
        learn: bool = True,
    ) -> tuple[int | None, float]:
        if label is not None:
            known = self.label_to_ccc.get(("vision", label.strip()))
            if known is not None:
                self.ccc_modalities[int(known)].add("vision")
                self._register_feature(int(known), "vision", feature)
                self._register_visual_pattern(int(known), visual_input)
                self._register_label(int(known), "vision", label)
                return int(known), 1.0

        nearest = self._nearest_visual_pattern(visual_input)
        if nearest is not None and nearest[1] >= float(self.config.alignment_threshold):
            ccc_id, similarity = nearest
            self.ccc_modalities[int(ccc_id)].add("vision")
            self._register_feature(int(ccc_id), "vision", feature)
            self._register_visual_pattern(int(ccc_id), visual_input)
            self._register_label(int(ccc_id), "vision", label)
            return int(ccc_id), float(similarity)

        if not learn:
            if nearest is None:
                return None, 0.0
            min_similarity = max(float(self.config.alignment_threshold), 0.7)
            if nearest[1] < min_similarity:
                return None, 0.0
            return nearest[0], nearest[1]

        ccc_id, confidence = self._recruit_ccc(feature, "vision", label)
        self._register_visual_pattern(int(ccc_id), visual_input)
        return ccc_id, confidence

    def _ccc_direction(self, ccc_id: int) -> Tensor:
        return self.ccc_pool.cccs[int(ccc_id)].concept_direction.detach().clone()

    def _activate(self, activations: list[tuple[int, float]], timestep: int) -> None:
        candidates: list[tuple[int, Tensor, float]] = []
        for ccc_id, confidence in activations:
            direction = self._ccc_direction(ccc_id)
            self.fabric.register_activation(ccc_id, direction, max(0.05, float(confidence)), timestep)
            candidates.append((ccc_id, direction, max(0.05, float(confidence))))
        self.fabric.form_associations(timestep)
        self.stream.think_step(candidates, timestep=timestep)

    def bind_visual_to_text(self, visual_ccc_id: int, text_ccc_id: int, strength: float = 1.0):
        """Create a bidirectional cross-modal association in the fabric."""

        visual_id = int(visual_ccc_id)
        text_id = int(text_ccc_id)
        self.ccc_modalities[visual_id].add("vision")
        self.ccc_modalities[text_id].add("text")
        scaled = max(0.0, float(strength))
        self.explicit_bindings[(visual_id, text_id)] = self.explicit_bindings.get((visual_id, text_id), 0.0) + scaled
        self.explicit_bindings[(text_id, visual_id)] = self.explicit_bindings.get((text_id, visual_id), 0.0) + scaled
        self.fabric._add_association(visual_id, text_id, scaled, temporal=False)  # noqa: SLF001
        self.fabric._add_association(text_id, visual_id, scaled, temporal=False)  # noqa: SLF001
        return {
            "visual_ccc_id": visual_id,
            "text_ccc_id": text_id,
            "strength": scaled,
        }

    def learn_cross_modal(self, visual_input: Tensor, text_input: str | Tensor | Iterable[int], label: str | None = None):
        """Learn a cross-modal binding from simultaneous image and text inputs."""

        visual_feature = self._encode_visual(visual_input)
        text_feature, decoded_text = self._encode_text(text_input)
        text_label = label or decoded_text or None

        visual_ccc_id, visual_confidence = self._resolve_visual_ccc(
            visual_input,
            visual_feature,
            label=label,
            learn=True,
        )
        text_ccc_id, text_confidence = self._resolve_ccc(
            text_feature,
            "text",
            label=text_label,
            learn=True,
        )

        if visual_ccc_id is None or text_ccc_id is None:
            raise RuntimeError("Failed to resolve multimodal CCC identifiers.")

        self._register_label(visual_ccc_id, "vision", label)
        self._register_label(text_ccc_id, "text", text_label)

        timestep = self.timestep
        activation_map = {
            int(visual_ccc_id): float(visual_confidence),
            int(text_ccc_id): max(float(text_confidence), float(visual_confidence))
            if int(text_ccc_id) == int(visual_ccc_id)
            else float(text_confidence),
        }
        self._activate(list(activation_map.items()), timestep)

        converged = int(visual_ccc_id) == int(text_ccc_id)
        if converged:
            self.fabric.register_activation(
                int(visual_ccc_id),
                self._ccc_direction(int(visual_ccc_id)),
                max(float(visual_confidence), float(text_confidence)) + float(self.config.cross_modal_strength),
                timestep,
            )
            self.fabric.form_associations(timestep)
        else:
            boost = float(self.config.cross_modal_strength) * (1.0 if text_label else 0.5)
            strength = boost + (0.5 * (float(visual_confidence) + float(text_confidence)))
            self.bind_visual_to_text(int(visual_ccc_id), int(text_ccc_id), strength=strength)

        self.timestep += 1
        return {
            "visual_ccc_id": int(visual_ccc_id),
            "text_ccc_id": int(text_ccc_id),
            "label": text_label,
            "converged": converged,
        }

    def _resolve_query_ccc(self, query, modality: str) -> tuple[int | None, float]:
        canonical = self._canonical_modality(modality)
        if isinstance(query, int):
            return int(query), 1.0
        if canonical == "vision":
            feature = self._encode_visual(query)
            return self._resolve_visual_ccc(query, feature, learn=False)
        if isinstance(query, str):
            known = self.label_to_ccc.get(("text", query.strip()))
            return (int(known), 1.0) if known is not None else (None, 0.0)
        feature, decoded = self._encode_text(query)
        label = decoded or None
        return self._resolve_ccc(feature, "text", label=label, learn=False)

    def cross_modal_retrieval(self, query, source_modality: str, target_modality: str) -> list[CrossModalAssociation]:
        """Retrieve target-modality concepts linked to a source-modality query."""

        source = self._canonical_modality(source_modality)
        target = self._canonical_modality(target_modality)
        source_ccc_id, source_confidence = self._resolve_query_ccc(query, source)
        if source_ccc_id is None:
            return []

        self._activate([(int(source_ccc_id), max(0.05, float(source_confidence)))], self.timestep)
        self.timestep += 1

        ranked: list[CrossModalAssociation] = []
        for (src_id, dst_id), strength in self.explicit_bindings.items():
            if int(src_id) != int(source_ccc_id):
                continue
            if target not in self.ccc_modalities.get(int(dst_id), set()):
                continue
            ranked.append(
                CrossModalAssociation(
                    source_ccc_id=int(src_id),
                    target_ccc_id=int(dst_id),
                    strength=float(strength),
                    source_modality=source,
                    target_modality=target,
                    label=self._primary_label(int(dst_id)),
                    temporal=False,
                )
            )

        if not ranked:
            for (src_id, dst_id), strength in self.fabric.association_strength.items():
                if int(src_id) != int(source_ccc_id):
                    continue
                if target not in self.ccc_modalities.get(int(dst_id), set()):
                    continue
                ranked.append(
                    CrossModalAssociation(
                        source_ccc_id=int(src_id),
                        target_ccc_id=int(dst_id),
                        strength=float(strength),
                        source_modality=source,
                        target_modality=target,
                        label=self._primary_label(int(dst_id)),
                        temporal=bool(self.fabric.association_temporal.get((src_id, dst_id), False)),
                    )
                )

        if not ranked:
            associates = self.fabric.retrieve_associates(self._ccc_direction(int(source_ccc_id)), k=10)
            for dst_id, strength, temporal in zip(
                associates.indices,
                associates.strengths,
                associates.temporal_order,
                strict=False,
            ):
                if target not in self.ccc_modalities.get(int(dst_id), set()):
                    continue
                ranked.append(
                    CrossModalAssociation(
                        source_ccc_id=int(source_ccc_id),
                        target_ccc_id=int(dst_id),
                        strength=float(strength),
                        source_modality=source,
                        target_modality=target,
                        label=self._primary_label(int(dst_id)),
                        temporal=bool(temporal),
                    )
                )

        ranked.sort(key=lambda association: association.strength, reverse=True)
        return ranked

    def describe_image(self, visual_input: Tensor, max_words: int = 10) -> str:
        """Generate a rough textual description from an image via cross-modal recall."""

        matches = self.cross_modal_retrieval(visual_input, source_modality="vision", target_modality="text")
        if not matches:
            return "unknown visual concept"
        words: list[str] = []
        for match in matches:
            if not match.label:
                continue
            for word in match.label.split():
                if word and word not in words:
                    words.append(word)
                if len(words) >= max(1, int(max_words)):
                    return " ".join(words)
        if words:
            return " ".join(words)
        return matches[0].label or "associated concept"

    def visualize_text(self, text_input: str | Tensor | Iterable[int]) -> Tensor:
        """Retrieve the visual pattern associated with a text concept."""

        matches = self.cross_modal_retrieval(text_input, source_modality="text", target_modality="vision")
        if not matches:
            channels, height, width = self.input_shape
            return torch.zeros((channels, height, width), dtype=torch.float32)
        best = matches[0]
        pattern = self.visual_patterns.get(int(best.target_ccc_id))
        if pattern is None:
            channels, height, width = self.input_shape
            return torch.zeros((channels, height, width), dtype=torch.float32)
        return pattern.detach().clone()


__all__ = [
    "CrossModalAssociation",
    "ModalityResult",
    "MultimodalFusion",
    "MultimodalFusionEngine",
    "MultimodalInput",
    "MultimodalOutput",
]
