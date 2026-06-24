"""Hierarchical visual feature learning built from stacked CCC pools."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.hierarchy.config import HierarchyConfig
from bioarn.hierarchy.feature_binding import FeatureBinding
from bioarn.hierarchy.receptive_fields import ReceptiveFieldExtractor
from bioarn.scaling import BatchedCCCPool


@dataclass
class LayerBatchResult:
    """Compact per-layer result over all spatial positions."""

    inputs: torch.Tensor
    activations: torch.Tensor
    fired_indices: list[list[int]]
    confidences: torch.Tensor
    recruited: list[bool]
    recruited_indices: list[int | None]


@dataclass
class HierarchyOutput:
    """Outputs from every layer in the visual hierarchy."""

    patches: list[torch.Tensor]
    patch_grid: tuple[int, int]
    layer_inputs: list[torch.Tensor]
    layer_activations: list[torch.Tensor]
    fired_indices: list[list[list[int]]]
    confidences: list[torch.Tensor]
    recruited: list[list[bool]]
    recruited_indices: list[list[int | None]]
    groupings: list[list[list[int]]] = field(default_factory=list)

    @property
    def final_features(self) -> torch.Tensor:
        return self.layer_activations[-1]


@dataclass
class HierarchyLayer:
    """Single hierarchical stage backed by a CCC pool."""

    name: str
    pool: "HierarchyPool"
    input_dim: int
    concept_dim: int
    threshold: float
    winner_limit: int
    last_inputs: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_features: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_fired_indices: list[list[int]] = field(default_factory=list)


@dataclass
class _LayerSampleResult:
    concept: torch.Tensor
    fired_indices: list[int]
    confidence: float
    recruited: bool
    recruited_index: int | None


class HierarchyPool:
    """Thin wrapper around the vectorized CCC pool with compact helpers."""

    def __init__(
        self,
        config: CCCConfig,
        margin_config: MarginGateConfig,
        *,
        min_input_norm: float = 1e-4,
    ) -> None:
        self.core = BatchedCCCPool(config, margin_config)
        self.min_input_norm = float(min_input_norm)

    @property
    def config(self) -> CCCConfig:
        return self.core.config

    @property
    def committed_count(self) -> int:
        return int(self.core.committed_mask.sum().item())

    @property
    def concept_directions(self) -> torch.Tensor:
        return self.core.concept_directions

    def get_pool_stats(self) -> dict[str, float | int]:
        return self.core.get_pool_stats()

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2:
            raise ValueError("Expected shape (input_dim,) or (batch, input_dim).")
        return x.to(torch.float32), False

    def _aggregate(
        self,
        fired_indices: list[int],
        winner_confidences: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        if not fired_indices:
            return torch.zeros(self.config.concept_dim, device=device, dtype=torch.float32), 0.0
        directions = self.core.concept_directions[fired_indices].to(device=device, dtype=torch.float32)
        weights = winner_confidences.to(device=device, dtype=torch.float32).reshape(-1, 1)
        concept = normalize((directions * weights).sum(dim=0, keepdim=True)).squeeze(0)
        return concept, float(weights.max().item())

    @staticmethod
    def _select_winners(
        indices: list[int],
        confidences: torch.Tensor,
        *,
        limit: int,
    ) -> tuple[list[int], torch.Tensor]:
        if not indices:
            return [], torch.empty(0, dtype=torch.float32, device=confidences.device)
        top_k = min(max(int(limit), 1), len(indices))
        values, order = torch.topk(confidences.to(torch.float32), k=top_k)
        selected_indices = [int(indices[position]) for position in order.tolist()]
        return selected_indices, values

    @torch.no_grad()
    def infer_batch(
        self,
        raw_input: torch.Tensor,
        *,
        winner_limit: int,
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
        """Infer concept activations without mutating the pool."""

        batch, _ = self._ensure_batch(raw_input)
        device = batch.device
        concepts = torch.zeros(
            batch.shape[0],
            self.config.concept_dim,
            device=device,
            dtype=torch.float32,
        )
        confidences = torch.zeros(batch.shape[0], device=device, dtype=torch.float32)
        fired_indices: list[list[int]] = [[] for _ in range(batch.shape[0])]

        if self.committed_count == 0:
            return concepts, fired_indices, confidences

        active_positions = (
            batch.norm(dim=-1) > self.min_input_norm
        ).nonzero(as_tuple=False).reshape(-1)
        if active_positions.numel() == 0:
            return concepts, fired_indices, confidences

        active_batch = batch.index_select(0, active_positions)
        committed_indices = self.core.committed_mask.nonzero(as_tuple=False).reshape(-1)

        shared_projection = active_batch @ self.core.f1_weights[0].transpose(0, 1)
        shared_projection = shared_projection + self.core.f1_bias[0]
        shared_activated = F.relu(shared_projection)
        top_k = min(self.config.f1_top_k, shared_activated.shape[-1])
        top_values, top_indices = torch.topk(shared_activated, k=top_k, dim=-1)
        shared_f1 = torch.zeros_like(shared_activated).scatter(-1, top_indices, top_values)

        committed_weights = self.core.f2_weights.index_select(0, committed_indices)
        committed_f2 = torch.matmul(
            committed_weights,
            shared_f1.transpose(0, 1),
        ).transpose(1, 2)
        directions = normalize(
            self.core.concept_directions.index_select(0, committed_indices)
        ).unsqueeze(1)
        confidence = (normalize(committed_f2).to(directions.dtype) * directions).sum(dim=-1)
        fired = confidence > self.core.theta_margin.index_select(0, committed_indices).unsqueeze(-1)

        for local_index, sample_index in enumerate(active_positions.tolist()):
            sample_mask = fired[:, local_index]
            if not bool(sample_mask.any().item()):
                continue
            sample_indices = committed_indices[sample_mask].tolist()
            sample_confidences = confidence[sample_mask, local_index].to(torch.float32)
            sample_indices, sample_confidences = self._select_winners(
                sample_indices,
                sample_confidences,
                limit=winner_limit,
            )
            concept, aggregate_confidence = self._aggregate(
                sample_indices,
                sample_confidences,
                device=device,
            )
            concepts[sample_index] = concept
            confidences[sample_index] = aggregate_confidence
            fired_indices[sample_index] = [int(index) for index in sample_indices]

        return concepts, fired_indices, confidences

    @torch.no_grad()
    def learn_single(
        self,
        raw_input: torch.Tensor,
        *,
        timestep: int,
        allow_recruit: bool,
        winner_limit: int,
    ) -> _LayerSampleResult:
        """Run one online learning step for a single input vector."""

        batch, _ = self._ensure_batch(raw_input)
        if float(batch.norm().item()) <= self.min_input_norm:
            return _LayerSampleResult(
                concept=torch.zeros(self.config.concept_dim, dtype=torch.float32, device=batch.device),
                fired_indices=[],
                confidence=0.0,
                recruited=False,
                recruited_index=None,
            )

        state = self.core._vectorized_state(batch, timestep=timestep)
        fired_mask = state.fired.squeeze(-1)
        fired_indices = fired_mask.nonzero(as_tuple=False).reshape(-1).tolist()

        recruited = False
        recruited_index: int | None = None
        if allow_recruit and not fired_indices:
            recruited_index, recruited_output = self.core.recruit(batch, timestep=timestep)
            if recruited_index is not None and recruited_output is not None:
                recruited = True
                fired_indices = [int(recruited_index)]
                winner_confidences = torch.tensor(
                    [float(recruited_output.confidence.reshape(-1).mean().item())],
                    dtype=torch.float32,
                    device=batch.device,
                )
            else:
                winner_confidences = torch.empty(0, dtype=torch.float32, device=batch.device)
        else:
            winner_confidences = (
                state.confidence[fired_mask, 0].to(torch.float32)
                if fired_indices
                else torch.empty(0, dtype=torch.float32, device=batch.device)
            )
            fired_indices, winner_confidences = self._select_winners(
                [int(index) for index in fired_indices],
                winner_confidences,
                limit=winner_limit,
            )

        concept, confidence = self._aggregate(
            [int(index) for index in fired_indices],
            winner_confidences,
            device=batch.device,
        )
        return _LayerSampleResult(
            concept=concept,
            fired_indices=[int(index) for index in fired_indices],
            confidence=confidence,
            recruited=recruited,
            recruited_index=recruited_index,
        )

    @torch.no_grad()
    def recruit_single(
        self,
        raw_input: torch.Tensor,
        *,
        timestep: int,
    ) -> _LayerSampleResult | None:
        """Force a new concept recruitment for supervised specialization."""

        batch, _ = self._ensure_batch(raw_input)
        if float(batch.norm().item()) <= self.min_input_norm:
            return None
        recruited_index, recruited_output = self.core.recruit(batch, timestep=timestep)
        if recruited_index is None or recruited_output is None:
            return None
        winner_confidences = torch.tensor(
            [float(recruited_output.confidence.reshape(-1).mean().item())],
            dtype=torch.float32,
            device=batch.device,
        )
        concept, confidence = self._aggregate(
            [int(recruited_index)],
            winner_confidences,
            device=batch.device,
        )
        return _LayerSampleResult(
            concept=concept,
            fired_indices=[int(recruited_index)],
            confidence=confidence,
            recruited=True,
            recruited_index=int(recruited_index),
        )


class VisualHierarchy:
    """Multi-layer CCC hierarchy mimicking the ventral visual stream."""

    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.extractor = ReceptiveFieldExtractor(
            config.image_size,
            include_position=config.include_position,
        )
        self.layers = [
            self._build_layer("V1", config.l1_input_dim, config.concept_dims[0], 0),
            self._build_layer("V2", 4 * config.concept_dims[0], config.concept_dims[1], 1),
            self._build_layer("V4", 4 * config.concept_dims[1], config.concept_dims[2], 2),
            self._build_layer("IT", config.concept_dims[2], config.concept_dims[3], 3),
        ]
        self.binding = (
            FeatureBinding(config.pool_sizes, binding_strength=config.binding_strength)
            if config.enable_binding
            else None
        )
        self.l4_label_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.label_prototypes: dict[int, torch.Tensor] = {}
        self.label_counts: Counter[int] = Counter()
        self.timestep = 0
        self.last_output: HierarchyOutput | None = None
        self._structure_count = 0
        self._structure_mean = 0.0
        self._structure_m2 = 0.0

    def _build_layer(
        self,
        name: str,
        input_dim: int,
        concept_dim: int,
        layer_index: int,
    ) -> HierarchyLayer:
        num_f1_features = max(
            concept_dim,
            min(128, max(24, int(input_dim // 2))),
        )
        ccc_config = CCCConfig(
            input_dim=int(input_dim),
            concept_dim=int(concept_dim),
            num_f1_features=int(num_f1_features),
            f1_top_k=max(4, min(32, int(num_f1_features // 4))),
            fast_lr=1.0,
            slow_lr=float(self.config.learning_rates[layer_index]),
            feedback_lr=float(self.config.learning_rates[layer_index]),
            max_pool_size=int(self.config.pool_sizes[layer_index]),
        )
        margin_config = MarginGateConfig(
            theta_margin=float(self.config.thresholds[layer_index]),
            theta_margin_lr=0.001,
            theta_resonance=min(0.95, float(self.config.thresholds[layer_index]) + 0.2),
        )
        return HierarchyLayer(
            name=name,
            pool=HierarchyPool(
                ccc_config,
                margin_config,
                min_input_norm=self.config.min_input_norm,
            ),
            input_dim=int(input_dim),
            concept_dim=int(concept_dim),
            threshold=float(self.config.thresholds[layer_index]),
            winner_limit=[4, 3, 2, 1][layer_index],
        )

    @staticmethod
    def _stack(vectors: list[torch.Tensor], width: int) -> torch.Tensor:
        if not vectors:
            return torch.zeros(0, width, dtype=torch.float32)
        return torch.stack([vector.to(torch.float32).reshape(-1) for vector in vectors], dim=0)

    @staticmethod
    def _normalize_feature(feature: torch.Tensor) -> torch.Tensor:
        if float(feature.norm().item()) <= 1e-8:
            return torch.zeros_like(feature, dtype=torch.float32)
        return normalize(feature.reshape(1, -1).to(torch.float32)).squeeze(0)

    def _structure_score(self, image: torch.Tensor) -> float:
        frame = self.extractor._ensure_image(image)
        vertical = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean().item()
        horizontal = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean().item()
        return float(vertical + horizontal)

    def _update_structure_stats(self, image: torch.Tensor) -> None:
        score = self._structure_score(image)
        self._structure_count += 1
        delta = score - self._structure_mean
        self._structure_mean += delta / self._structure_count
        delta2 = score - self._structure_mean
        self._structure_m2 += delta * delta2

    def _is_noise_like(self, image: torch.Tensor) -> bool:
        if self._structure_count < 10:
            return False
        variance = self._structure_m2 / max(self._structure_count - 1, 1)
        std = variance ** 0.5
        threshold = self._structure_mean + max(4.0 * std, 0.15)
        return self._structure_score(image) > threshold

    def _run_layer_infer(self, layer: HierarchyLayer, inputs: torch.Tensor) -> LayerBatchResult:
        activations, fired_indices, confidences = layer.pool.infer_batch(
            inputs,
            winner_limit=layer.winner_limit,
        )
        layer.last_inputs = inputs.detach().clone()
        layer.last_features = activations.detach().clone()
        layer.last_fired_indices = [list(indices) for indices in fired_indices]
        return LayerBatchResult(
            inputs=inputs,
            activations=activations,
            fired_indices=fired_indices,
            confidences=confidences,
            recruited=[False] * len(fired_indices),
            recruited_indices=[None] * len(fired_indices),
        )

    def _run_layer_learn(
        self,
        layer: HierarchyLayer,
        inputs: torch.Tensor,
        *,
        allow_recruit: bool,
    ) -> LayerBatchResult:
        activations: list[torch.Tensor] = []
        fired_indices: list[list[int]] = []
        confidences: list[float] = []
        recruited: list[bool] = []
        recruited_indices: list[int | None] = []

        for index in range(inputs.shape[0]):
            sample = layer.pool.learn_single(
                inputs[index],
                timestep=self.timestep,
                allow_recruit=allow_recruit,
                winner_limit=layer.winner_limit,
            )
            self.timestep += 1
            activations.append(sample.concept)
            fired_indices.append(sample.fired_indices)
            confidences.append(sample.confidence)
            recruited.append(sample.recruited)
            recruited_indices.append(sample.recruited_index)

        stacked_activations = self._stack(activations, layer.concept_dim)
        stacked_confidences = torch.tensor(confidences, dtype=torch.float32)
        layer.last_inputs = inputs.detach().clone()
        layer.last_features = stacked_activations.detach().clone()
        layer.last_fired_indices = [list(indices) for indices in fired_indices]
        return LayerBatchResult(
            inputs=inputs,
            activations=stacked_activations,
            fired_indices=fired_indices,
            confidences=stacked_confidences,
            recruited=recruited,
            recruited_indices=recruited_indices,
        )

    def _prepare_l2_inputs(
        self,
        l1_activations: torch.Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, list[list[int]]]:
        grouping = self.extractor.make_grouping(
            patch_grid,
            group_size=int(self.config.patch_sizes[1]),
            stride=int(self.config.patch_sizes[1]),
        )
        pooled = self.extractor.pool_activations(list(l1_activations), grouping)
        return self._stack(pooled, 4 * self.config.concept_dims[0]), grouping

    def _prepare_l3_inputs(self, l2_activations: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
        grouping = [list(range(l2_activations.shape[0]))] if l2_activations.shape[0] else []
        pooled = self.extractor.pool_activations(list(l2_activations), grouping)
        return self._stack(pooled, 4 * self.config.concept_dims[1]), grouping

    def _prepare_l4_inputs(self, l3_activations: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
        grouping = [list(range(l3_activations.shape[0]))] if l3_activations.shape[0] else []
        return l3_activations.to(torch.float32), grouping

    def _update_bindings(self, output: HierarchyOutput) -> None:
        if self.binding is None:
            return

        groupings12 = output.groupings[0]
        for group_index, higher_indices in enumerate(output.fired_indices[1]):
            lower_indices = [output.fired_indices[0][index] for index in groupings12[group_index]]
            self.binding.strengthen(0, lower_indices, higher_indices, delay=1)

        lower_indices_l2 = [output.fired_indices[1][index] for index in output.groupings[1][0]]
        self.binding.strengthen(1, lower_indices_l2, output.fired_indices[2][0], delay=1)
        self.binding.strengthen(2, output.fired_indices[2][0], output.fired_indices[3][0], delay=1)

    def _update_label_memory(
        self,
        label: int,
        evidence_feature: torch.Tensor,
        fired_indices: list[int],
        confidence: float,
    ) -> None:
        del confidence
        if float(evidence_feature.norm().item()) <= 1e-8:
            return
        normalized = self._normalize_feature(evidence_feature)
        count = self.label_counts[int(label)] + 1
        if int(label) not in self.label_prototypes:
            self.label_prototypes[int(label)] = normalized.detach().clone()
        else:
            updated = (
                (self.label_prototypes[int(label)] * self.label_counts[int(label)]) + normalized
            ) / count
            self.label_prototypes[int(label)] = self._normalize_feature(updated)
        self.label_counts[int(label)] = count

        for fired_index in fired_indices:
            self.l4_label_counts[int(fired_index)][int(label)] += 1

    def _has_label_specialist(self, label: int, fired_indices: list[int]) -> bool:
        for fired_index in fired_indices:
            counts = self.l4_label_counts.get(int(fired_index))
            if not counts:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / max(sum(counts.values()), 1)
            if int(dominant_label) == int(label) and purity >= 0.55:
                return True
        return False

    def _prototype_prediction(self, evidence_feature: torch.Tensor) -> tuple[int | None, float, float]:
        if not self.label_prototypes or float(evidence_feature.norm().item()) <= 1e-8:
            return None, 0.0, 0.0
        labels = sorted(self.label_prototypes)
        stacked = torch.stack(
            [self.label_prototypes[label].to(evidence_feature) for label in labels],
            dim=0,
        )
        query = self._normalize_feature(evidence_feature).unsqueeze(0).expand_as(stacked)
        similarities = cosine_similarity(stacked, query)
        best_index = int(torch.argmax(similarities).item())
        sorted_similarities, _ = torch.sort(similarities, descending=True)
        margin = (
            float(sorted_similarities[0].item() - sorted_similarities[1].item())
            if sorted_similarities.numel() > 1
            else float(sorted_similarities[0].item())
        )
        return labels[best_index], float(similarities[best_index].item()), margin

    def _label_signature_from_output(self, output: HierarchyOutput) -> torch.Tensor:
        parts = [
            output.layer_inputs[0].reshape(-1),
            output.layer_activations[1].reshape(-1),
            output.layer_activations[2].reshape(-1),
            output.layer_inputs[-1][0].reshape(-1),
        ]
        return torch.cat([part.to(torch.float32).reshape(-1) for part in parts], dim=0)

    def _predict_label(
        self,
        evidence_feature: torch.Tensor,
        fired_indices: list[int],
        final_confidence: float,
    ) -> tuple[int, float]:
        if not fired_indices:
            return -1, 0.0

        prototype_label, prototype_similarity, prototype_margin = self._prototype_prediction(
            evidence_feature
        )
        if (
            prototype_label is not None
            and prototype_similarity >= 0.35
            and prototype_margin >= 0.01
        ):
            return int(prototype_label), max(float(final_confidence), float(prototype_similarity))

        votes: defaultdict[int, float] = defaultdict(float)
        for fired_index in fired_indices:
            counts = self.l4_label_counts.get(int(fired_index))
            if not counts:
                continue
            total = sum(counts.values())
            for label, count in counts.items():
                votes[int(label)] += count / max(total, 1)

        if votes:
            voted_label, vote_score = max(votes.items(), key=lambda item: item[1])
            vote_confidence = float(vote_score / max(len(fired_indices), 1))
            confidence = max(
                float(final_confidence),
                vote_confidence,
                float(prototype_similarity),
            )
            if vote_confidence < 0.4 and prototype_similarity < 0.25:
                return -1, confidence
            if prototype_label is not None and prototype_label != voted_label and prototype_similarity < 0.15:
                return -1, confidence
            return int(voted_label), confidence

        if prototype_label is None or prototype_similarity < 0.2:
            return -1, max(float(final_confidence), float(prototype_similarity))
        return int(prototype_label), max(float(final_confidence), float(prototype_similarity))

    def _forward(
        self,
        image: torch.Tensor,
        *,
        learn: bool,
        label: int | None = None,
    ) -> HierarchyOutput:
        patches = self.extractor.extract_patches(
            image,
            patch_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        patch_grid = self.extractor.last_grid_shape
        l1_inputs = self._stack(patches, self.config.l1_input_dim)
        l1_result = (
            self._run_layer_learn(self.layers[0], l1_inputs, allow_recruit=True)
            if learn
            else self._run_layer_infer(self.layers[0], l1_inputs)
        )

        l2_inputs, grouping12 = self._prepare_l2_inputs(l1_result.activations, patch_grid)
        l2_result = (
            self._run_layer_learn(self.layers[1], l2_inputs, allow_recruit=True)
            if learn
            else self._run_layer_infer(self.layers[1], l2_inputs)
        )

        l3_inputs, grouping23 = self._prepare_l3_inputs(l2_result.activations)
        l3_result = (
            self._run_layer_learn(self.layers[2], l3_inputs, allow_recruit=True)
            if learn
            else self._run_layer_infer(self.layers[2], l3_inputs)
        )

        l4_inputs, grouping34 = self._prepare_l4_inputs(l3_result.activations)
        allow_l4_recruit = learn and label is not None
        l4_result = (
            self._run_layer_learn(self.layers[3], l4_inputs, allow_recruit=allow_l4_recruit)
            if learn
            else self._run_layer_infer(self.layers[3], l4_inputs)
        )
        if (
            learn
            and label is not None
            and l4_inputs.shape[0] > 0
            and not self._has_label_specialist(int(label), l4_result.fired_indices[0])
        ):
            recruited = self.layers[3].pool.recruit_single(
                l4_inputs[0],
                timestep=self.timestep,
            )
            self.timestep += 1
            if recruited is not None:
                l4_result.activations[0] = recruited.concept
                l4_result.fired_indices[0] = recruited.fired_indices
                l4_result.confidences[0] = recruited.confidence
                l4_result.recruited[0] = True
                l4_result.recruited_indices[0] = recruited.recruited_index

        output = HierarchyOutput(
            patches=patches,
            patch_grid=patch_grid,
            layer_inputs=[l1_inputs, l2_inputs, l3_inputs, l4_inputs],
            layer_activations=[
                l1_result.activations,
                l2_result.activations,
                l3_result.activations,
                l4_result.activations,
            ],
            fired_indices=[
                l1_result.fired_indices,
                l2_result.fired_indices,
                l3_result.fired_indices,
                l4_result.fired_indices,
            ],
            confidences=[
                l1_result.confidences,
                l2_result.confidences,
                l3_result.confidences,
                l4_result.confidences,
            ],
            recruited=[
                l1_result.recruited,
                l2_result.recruited,
                l3_result.recruited,
                l4_result.recruited,
            ],
            recruited_indices=[
                l1_result.recruited_indices,
                l2_result.recruited_indices,
                l3_result.recruited_indices,
                l4_result.recruited_indices,
            ],
            groupings=[grouping12, grouping23, grouping34],
        )

        if learn:
            self._update_bindings(output)
            if label is not None and l4_result.activations.shape[0] > 0:
                self._update_label_memory(
                    int(label),
                    self._label_signature_from_output(output),
                    l4_result.fired_indices[0],
                    float(l4_result.confidences[0].item()),
                )

        self.last_output = output
        return output

    @torch.no_grad()
    def process(self, image: torch.Tensor) -> HierarchyOutput:
        """Process an image bottom-up without recruiting new concepts."""

        return self._forward(image, learn=False)

    @torch.no_grad()
    def learn(self, image: torch.Tensor, label: int | None = None) -> HierarchyOutput:
        """Learn one image through unsupervised lower layers and supervised IT binding."""

        self._update_structure_stats(image)
        return self._forward(image, learn=True, label=label)

    @torch.no_grad()
    def classify(self, image: torch.Tensor) -> tuple[int, float]:
        """Classify an image from the top-level IT layer."""

        if self._is_noise_like(image):
            return -1, 0.0
        output = self.process(image)
        final_fired = output.fired_indices[-1][0] if output.fired_indices[-1] else []
        final_confidence = (
            float(output.confidences[-1][0].item())
            if output.confidences[-1].numel()
            else 0.0
        )
        if not final_fired:
            return -1, 0.0

        predicted_label, confidence = self._predict_label(
            self._label_signature_from_output(output),
            final_fired,
            final_confidence,
        )
        if predicted_label < 0:
            return -1, float(confidence)
        return predicted_label, float(min(1.0, confidence))

    @torch.no_grad()
    def get_layer_features(self, image: torch.Tensor, layer: int) -> torch.Tensor:
        """Extract intermediate features from a 1-indexed layer."""

        if layer < 1 or layer > len(self.layers):
            raise ValueError(f"layer must be between 1 and {len(self.layers)}.")
        output = self.process(image)
        return output.layer_activations[layer - 1]


__all__ = [
    "HierarchyLayer",
    "HierarchyOutput",
    "LayerBatchResult",
    "VisualHierarchy",
]
