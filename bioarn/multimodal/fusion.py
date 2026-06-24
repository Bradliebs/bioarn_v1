"""Cross-modal binding across vision and language in shared CCC space."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from bioarn.config import CCCConfig, GNWConfig, MarginGateConfig, SDMConfig, SpikingConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.memory.associative_fabric import AssociativeFabric
from bioarn.sensorimotor.language import LanguageEncoder
from bioarn.sensorimotor.vision import VisualEncoder
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


__all__ = ["CrossModalAssociation", "MultimodalFusion"]
