"""Hierarchical visual feature learning built from stacked CCC pools."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.hierarchy.attention import SpatialAttention
from bioarn.hierarchy.competition import CompetitiveLateralInhibition
from bioarn.hierarchy.config import HierarchyConfig
from bioarn.hierarchy.feature_binding import FeatureBinding
from bioarn.hierarchy.receptive_fields import ReceptiveFieldExtractor
from bioarn.predictive.error_gating import ErrorGatingOutput, PredictionErrorGate
from bioarn.predictive.hierarchy import PredictiveHierarchy
from bioarn.scaling import AdaptiveCapacity, BatchedCCCPool


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
    predictive_states: list[torch.Tensor] = field(default_factory=list)
    predictive_errors: list[torch.Tensor] = field(default_factory=list)
    prediction_error_gates: list[torch.Tensor] = field(default_factory=list)
    predictive_free_energy_trace: list[float] = field(default_factory=list)
    predictive_converged: bool = False
    predictive_surprise: float = 0.0
    predictive_mode: str = "disabled"
    hierarchy_prediction_errors: dict[str, float] = field(default_factory=dict)
    hierarchy_uncertainty: float = 0.0

    @property
    def final_features(self) -> torch.Tensor:
        if self.predictive_mode == "settling" and self.predictive_states:
            predictive = self.predictive_states[-1]
            raw = self.layer_activations[-1].to(torch.float32)
            if predictive.dim() == 1 and raw.dim() == 2:
                predictive = predictive.unsqueeze(0)
            if predictive.shape == raw.shape:
                return normalize((raw + predictive.to(raw)).reshape(raw.shape[0], -1))
            return predictive
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
        max_capacity: int | None = None,
        growth_factor: float = 1.35,
        abstention_window: int = 24,
        abstention_threshold: float = 0.35,
        prune_interval: int = 256,
        prune_min_presentations: int = 96,
        prune_max_fire_count: int = 0,
        enable_lateral_inhibition: bool = True,
        inhibition_similarity_threshold: float = 0.9,
    ) -> None:
        self.min_input_norm = float(min_input_norm)
        self.max_capacity = int(max_capacity or config.max_pool_size)
        self.capacity_controller: AdaptiveCapacity | None = None
        if self.max_capacity > int(config.max_pool_size):
            self.capacity_controller = AdaptiveCapacity(
                initial_size=int(config.max_pool_size),
                max_size=self.max_capacity,
                config=config,
                margin_config=margin_config,
                growth_factor=growth_factor,
                abstention_window=abstention_window,
                abstention_threshold=abstention_threshold,
            )
            self.core = self.capacity_controller.pool
        else:
            self.core = BatchedCCCPool(config, margin_config)
        self.prune_interval = int(max(0, prune_interval))
        self.prune_min_presentations = int(max(1, prune_min_presentations))
        self.prune_max_fire_count = int(max(0, prune_max_fire_count))
        self.inhibition = (
            CompetitiveLateralInhibition(
                similarity_threshold=inhibition_similarity_threshold,
            )
            if enable_lateral_inhibition
            else None
        )

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
        stats = self.core.get_pool_stats()
        if self.capacity_controller is not None:
            stats["current_capacity"] = int(self.core.config.max_pool_size)
            stats["max_capacity"] = int(self.max_capacity)
        return stats

    def _sync_core(self) -> None:
        if self.capacity_controller is not None and self.core is not self.capacity_controller.pool:
            self.core = self.capacity_controller.pool

    def _observe_abstention(self, abstained: bool) -> None:
        if self.capacity_controller is None:
            return
        if abstained and self.capacity_controller.utilization() < 0.85:
            return
        self.capacity_controller.observe_abstention(abstained)
        self._sync_core()

    def _maybe_grow_or_prune(self) -> None:
        if self.capacity_controller is None:
            return
        if self.core._first_uncommitted_index() is not None:
            return
        previous_size = int(self.core.config.max_pool_size)
        self.capacity_controller.grow()
        self._sync_core()
        if int(self.core.config.max_pool_size) > previous_size:
            return
        self.capacity_controller.prune_dead_cccs(
            min_presentations=self.prune_min_presentations,
            max_fire_count=self.prune_max_fire_count,
        )
        self._sync_core()

    def _maybe_prune(self, timestep: int) -> None:
        if (
            self.capacity_controller is None
            or self.prune_interval <= 0
            or timestep <= 0
            or timestep % self.prune_interval != 0
            or self.capacity_controller.utilization() < 0.85
        ):
            return
        self.capacity_controller.prune_dead_cccs(
            min_presentations=self.prune_min_presentations,
            max_fire_count=self.prune_max_fire_count,
        )
        self._sync_core()

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

    def _select_winners(
        self,
        indices: list[int],
        confidences: torch.Tensor,
        *,
        limit: int,
    ) -> tuple[list[int], torch.Tensor]:
        if not indices:
            return [], torch.empty(0, dtype=torch.float32, device=confidences.device)
        if self.inhibition is not None and len(indices) > 1:
            return self.inhibition.select(
                indices,
                confidences.to(torch.float32),
                self.core.concept_directions,
                limit=limit,
            )
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

        self._sync_core()
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
        state = self.core._vectorized_state(
            active_batch,
            timestep=self.core.last_fired.max().item() if self.committed_count else 0,
            apply_learning=False,
            update_state=False,
        )
        committed_indices = self.core.committed_mask.nonzero(as_tuple=False).reshape(-1)
        confidence = state.confidence.index_select(0, committed_indices)
        fired = state.fired.index_select(0, committed_indices)

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

        self._sync_core()
        batch, _ = self._ensure_batch(raw_input)
        if float(batch.norm().item()) <= self.min_input_norm:
            return _LayerSampleResult(
                concept=torch.zeros(self.config.concept_dim, dtype=torch.float32, device=batch.device),
                fired_indices=[],
                confidence=0.0,
                recruited=False,
                recruited_index=None,
            )

        observer = getattr(self.core, "observe_samples", None)
        if callable(observer):
            observer(batch.shape[0])
        state = self.core._vectorized_state(batch, timestep=timestep)
        fired_mask = state.fired.squeeze(-1)
        fired_indices = fired_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        self._observe_abstention(not fired_indices)

        recruited = False
        recruited_index: int | None = None
        if allow_recruit and not fired_indices:
            self._maybe_grow_or_prune()
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

        update_importance = getattr(self.core, "update_importance", None)
        if callable(update_importance):
            update_importance(fired_indices, confidences=winner_confidences)
        self._maybe_prune(timestep)
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

        self._sync_core()
        batch, _ = self._ensure_batch(raw_input)
        if float(batch.norm().item()) <= self.min_input_norm:
            return None
        observer = getattr(self.core, "observe_samples", None)
        if callable(observer):
            observer(batch.shape[0])
        self._maybe_grow_or_prune()
        recruited_index, recruited_output = self.core.recruit(batch, timestep=timestep)
        if recruited_index is None or recruited_output is None:
            return None
        update_importance = getattr(self.core, "update_importance", None)
        if callable(update_importance):
            update_importance(
                [int(recruited_index)],
                confidences=[float(recruited_output.confidence.reshape(-1).mean().item())],
            )
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

    def freeze_f1(self) -> None:
        freezer = getattr(self.core, "freeze_f1", None)
        if callable(freezer):
            freezer()

    def create_task_adapter(self) -> None:
        creator = getattr(self.core, "create_task_adapter", None)
        if callable(creator):
            creator()


class VisualHierarchy:
    """Multi-layer CCC hierarchy mimicking the ventral visual stream."""

    def __init__(self, config: HierarchyConfig):
        self.config = config
        if self.config.init_seed is not None:
            torch.manual_seed(int(self.config.init_seed))
        self.extractor = ReceptiveFieldExtractor(
            config.image_size,
            include_position=config.include_position,
        )
        self.attention = (
            SpatialAttention(
                config.image_size,
                gain_strength=config.attention_gain_strength,
                center_bias=config.attention_center_bias,
            )
            if config.enable_spatial_attention
            else None
        )
        self.layers = [
            self._build_layer("V1", config.l1_input_dim, config.concept_dims[0], 0),
            self._build_layer("V2", 4 * config.concept_dims[0], config.concept_dims[1], 1),
            self._build_layer("V4", 4 * config.concept_dims[1], config.concept_dims[2], 2),
            self._build_layer("IT", config.concept_dims[2], config.concept_dims[3], 3),
        ]
        binding_pool_sizes = (
            config.max_pool_sizes if config.enable_adaptive_capacity else config.pool_sizes
        )
        self.binding = (
            FeatureBinding(binding_pool_sizes, binding_strength=config.binding_strength)
            if config.enable_binding
            else None
        )
        self.predictive_hierarchy = None
        self.prediction_error_gate = None
        self.predictive_mode = "disabled"
        self.predictive_novelty_threshold = 0.0
        if config.predictive is not None:
            self.predictive_mode = str(config.predictive.mode)
            self.predictive_novelty_threshold = max(
                0.05,
                float(config.predictive.error_threshold) * 5.0,
            )
            if self.predictive_mode == "settling":
                self.predictive_hierarchy = PredictiveHierarchy(
                    list(config.concept_dims),
                    config.predictive,
                )
            else:
                self.prediction_error_gate = PredictionErrorGate(
                    list(config.concept_dims),
                    config.predictive,
                )
            self._initialize_predictive_module()
        self.feedback_v2_to_v1 = torch.zeros(
            config.concept_dims[0],
            config.concept_dims[1],
            dtype=torch.float32,
        )
        self.feedback_v4_to_v2 = torch.zeros(
            config.concept_dims[1],
            config.concept_dims[2],
            dtype=torch.float32,
        )
        self.feedback_it_to_v4 = torch.zeros(
            config.concept_dims[2],
            config.concept_dims[3],
            dtype=torch.float32,
        )
        self.hierarchy_predictors = [
            self._seed_predictive_weights(config.concept_dims[0], config.concept_dims[1]),
            self._seed_predictive_weights(config.concept_dims[1], config.concept_dims[2]),
            self._seed_predictive_weights(config.concept_dims[2], config.concept_dims[3]),
        ]
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
            consolidation_strength=float(self.config.consolidation_strength),
            freeze_f1_after=int(self.config.freeze_f1_after),
            f1_adapter_dim=int(self.config.f1_adapter_dim),
            fast_lr=1.0,
            slow_lr=float(self.config.learning_rates[layer_index]),
            feedback_lr=float(self.config.learning_rates[layer_index]),
            max_pool_size=int(self.config.pool_sizes[layer_index]),
            stdp=self.config.stdp,
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
                max_capacity=(
                    int(self.config.max_pool_sizes[layer_index])
                    if self.config.enable_adaptive_capacity
                    else int(self.config.pool_sizes[layer_index])
                ),
                growth_factor=self.config.capacity_growth_factor,
                abstention_window=self.config.capacity_abstention_window,
                abstention_threshold=self.config.capacity_abstention_threshold,
                prune_interval=self.config.capacity_prune_interval,
                prune_min_presentations=self.config.capacity_prune_min_presentations,
                prune_max_fire_count=self.config.capacity_prune_max_fire_count,
                enable_lateral_inhibition=self.config.enable_lateral_inhibition,
                inhibition_similarity_threshold=self.config.inhibition_similarity_threshold,
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

    @staticmethod
    def _seed_predictive_weights(input_dim: int, output_dim: int) -> torch.Tensor:
        weights = torch.zeros(output_dim, input_dim, dtype=torch.float32)
        if input_dim <= 0 or output_dim <= 0:
            return weights
        lower_positions = torch.linspace(0, input_dim - 1, steps=output_dim).round().to(torch.long)
        weights[torch.arange(output_dim), lower_positions] = 1.0
        weights = weights + (0.01 * torch.randn_like(weights))
        return normalize(weights)

    def _predictive_layers(self):
        if self.predictive_hierarchy is not None:
            return self.predictive_hierarchy.layers
        if self.prediction_error_gate is not None:
            return self.prediction_error_gate.layers
        return None

    def _initialize_predictive_module(self) -> None:
        predictive_layers = self._predictive_layers()
        if predictive_layers is None:
            return
        with torch.no_grad():
            layer_count = max(len(predictive_layers), 1)
            for layer_index, layer in enumerate(predictive_layers):
                layer.W.copy_(
                    self._seed_predictive_weights(
                        input_dim=layer.input_dim,
                        output_dim=layer.output_dim,
                    )
                )
                lower_level_bias = (layer_count - layer_index - 1) / layer_count
                precision = float(layer.config.precision_init) * (1.0 + (0.75 * lower_level_bias))
                layer.precision.fill_(precision)
                layer.state.zero_()

    def _summarize_activations(
        self,
        activations: torch.Tensor,
        confidences: torch.Tensor,
        *,
        target_dim: int,
    ) -> torch.Tensor:
        if activations.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32, device=activations.device)
        batch = activations.to(torch.float32)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        weights = confidences.to(device=batch.device, dtype=torch.float32).reshape(-1)
        if weights.numel() != batch.shape[0]:
            weights = torch.ones(batch.shape[0], device=batch.device, dtype=torch.float32)
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 1e-8:
            weights = torch.ones_like(weights)
        summary = (batch * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        return self._normalize_feature(summary)

    def _predictive_signature(
        self,
        raw_signature: torch.Tensor,
        states: list[torch.Tensor],
        errors: list[torch.Tensor],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = [self._normalize_feature(raw_signature.reshape(-1))]
        if states:
            parts.append(self._normalize_feature(states[-1].reshape(-1)))
            if len(states) > 1:
                parts.append(self._normalize_feature(states[-2].reshape(-1)))
        informative_errors = [error.reshape(-1) for error in errors[:-1] if float(error.norm().item()) > 1e-8]
        if informative_errors:
            parts.append(self._normalize_feature(torch.cat(informative_errors[:2], dim=0)))
        return torch.cat(parts, dim=0)

    def _summarize_layer_activity(self, activations: torch.Tensor, *, target_dim: int) -> torch.Tensor:
        if activations.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32, device=activations.device)
        batch = activations.to(torch.float32)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        return self._normalize_feature(batch.mean(dim=0))

    def _project_feedback(
        self,
        higher_activations: torch.Tensor,
        projection: torch.Tensor,
        *,
        target_dim: int,
    ) -> torch.Tensor:
        if higher_activations.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32, device=projection.device)
        higher_summary = self._summarize_layer_activity(
            higher_activations,
            target_dim=projection.shape[1],
        ).to(device=projection.device, dtype=projection.dtype)
        return torch.tanh(projection @ higher_summary)

    def _project_hierarchy_prediction(
        self,
        lower_activations: torch.Tensor,
        projection: torch.Tensor,
        *,
        target_dim: int,
    ) -> torch.Tensor:
        if lower_activations.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32, device=projection.device)
        lower_summary = self._summarize_layer_activity(
            lower_activations,
            target_dim=projection.shape[1],
        ).to(device=projection.device, dtype=projection.dtype)
        predicted = torch.tanh(projection @ lower_summary)
        return self._normalize_feature(predicted)

    @staticmethod
    def _prediction_error(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        if float(predicted.norm().item()) <= 1e-8 or float(actual.norm().item()) <= 1e-8:
            return 0.0
        similarity = float(
            cosine_similarity(
                predicted.reshape(1, -1).to(actual),
                actual.reshape(1, -1),
            ).item()
        )
        return float(max(0.0, min(1.0, 1.0 - similarity)))

    @staticmethod
    def _mean_prediction_error(errors: dict[str, float]) -> float:
        if not errors:
            return 0.0
        return float(sum(errors.values()) / max(len(errors), 1))

    @torch.no_grad()
    def compute_hierarchy_prediction_errors(
        self,
        layer_activations: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Compute one-shot bottom-up prediction errors between adjacent layers."""

        errors: dict[str, float] = {}
        names = ["V1", "V2", "V4", "IT"]
        for index, (lower_name, higher_name) in enumerate(zip(names[:-1], names[1:], strict=False)):
            lower = layer_activations.get(lower_name)
            higher = layer_activations.get(higher_name)
            if lower is None or higher is None:
                continue
            actual = self._summarize_layer_activity(
                higher,
                target_dim=self.layers[index + 1].concept_dim,
            )
            predicted = self._project_hierarchy_prediction(
                lower,
                self.hierarchy_predictors[index],
                target_dim=self.layers[index + 1].concept_dim,
            )
            errors[f"{lower_name}->{higher_name}"] = self._prediction_error(predicted, actual)
        return errors

    @torch.no_grad()
    def _update_hierarchy_predictors(
        self,
        layer_activations: dict[str, torch.Tensor],
        *,
        learning_rate_scale: float = 1.0,
    ) -> None:
        base_lr = (
            float(self.config.predictive.eta)
            if self.config.predictive is not None
            else float(sum(self.config.learning_rates[:3]) / 3.0)
        )
        effective_lr = max(0.0, base_lr * float(learning_rate_scale))
        if effective_lr <= 0.0:
            return
        names = ["V1", "V2", "V4", "IT"]
        for index, (lower_name, higher_name) in enumerate(zip(names[:-1], names[1:], strict=False)):
            lower = layer_activations.get(lower_name)
            higher = layer_activations.get(higher_name)
            if lower is None or higher is None:
                continue
            lower_summary = self._summarize_layer_activity(
                lower,
                target_dim=self.layers[index].concept_dim,
            )
            higher_summary = self._summarize_layer_activity(
                higher,
                target_dim=self.layers[index + 1].concept_dim,
            )
            if (
                float(lower_summary.norm().item()) <= 1e-8
                or float(higher_summary.norm().item()) <= 1e-8
            ):
                continue
            predictor = self.hierarchy_predictors[index]
            predictor.copy_(
                normalize(
                    predictor
                    + (
                        effective_lr
                        * torch.outer(higher_summary.to(predictor), lower_summary.to(predictor))
                    )
                )
            )

    def _propagate_hierarchy_prediction_error(self, hierarchy_uncertainty: float) -> None:
        for layer in self.layers:
            setter = getattr(layer.pool.core, "set_hierarchy_prediction_error", None)
            if callable(setter):
                setter(hierarchy_uncertainty)

    def _previous_feedback_signals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.feedback_strength <= 0.0 or self.last_output is None:
            return (
                torch.zeros(self.layers[0].concept_dim, dtype=torch.float32),
                torch.zeros(self.layers[1].concept_dim, dtype=torch.float32),
                torch.zeros(self.layers[2].concept_dim, dtype=torch.float32),
            )
        return (
            self._project_feedback(
                self.last_output.layer_activations[1],
                self.feedback_v2_to_v1,
                target_dim=self.layers[0].concept_dim,
            ),
            self._project_feedback(
                self.last_output.layer_activations[2],
                self.feedback_v4_to_v2,
                target_dim=self.layers[1].concept_dim,
            ),
            self._project_feedback(
                self.last_output.layer_activations[3],
                self.feedback_it_to_v4,
                target_dim=self.layers[2].concept_dim,
            ),
        )

    def _apply_feedback_modulation(
        self,
        layer: HierarchyLayer,
        result: LayerBatchResult,
        feedback_signal: torch.Tensor,
    ) -> LayerBatchResult:
        if self.config.feedback_strength <= 0.0 or result.activations.numel() == 0:
            return result
        modulation = 1.0 + (
            self.config.feedback_strength
            * feedback_signal.to(device=result.activations.device, dtype=result.activations.dtype)
        )
        modulation = modulation.clamp_min(0.0)
        result.activations = result.activations * modulation.unsqueeze(0)
        layer.last_features = result.activations.detach().clone()
        return result

    def _update_feedback_connections(self, output: HierarchyOutput) -> None:
        if self.config.feedback_strength <= 0.0:
            return
        updates = (
            (
                self.feedback_v2_to_v1,
                self._summarize_layer_activity(output.layer_activations[0], target_dim=self.layers[0].concept_dim),
                self._summarize_layer_activity(output.layer_activations[1], target_dim=self.layers[1].concept_dim),
                float(self.config.learning_rates[0]),
            ),
            (
                self.feedback_v4_to_v2,
                self._summarize_layer_activity(output.layer_activations[1], target_dim=self.layers[1].concept_dim),
                self._summarize_layer_activity(output.layer_activations[2], target_dim=self.layers[2].concept_dim),
                float(self.config.learning_rates[1]),
            ),
            (
                self.feedback_it_to_v4,
                self._summarize_layer_activity(output.layer_activations[2], target_dim=self.layers[2].concept_dim),
                self._summarize_layer_activity(output.layer_activations[3], target_dim=self.layers[3].concept_dim),
                float(self.config.learning_rates[2]),
            ),
        )
        for matrix, lower_summary, higher_summary, learning_rate in updates:
            if (
                float(lower_summary.norm().item()) <= 1e-8
                or float(higher_summary.norm().item()) <= 1e-8
            ):
                continue
            updated = matrix + (learning_rate * torch.outer(lower_summary, higher_summary).to(matrix))
            matrix.copy_(normalize(updated))

    def _raw_label_signature_from_output(self, output: HierarchyOutput) -> torch.Tensor:
        parts = [
            output.layer_inputs[0].reshape(-1),
            output.layer_activations[1].reshape(-1),
            output.layer_activations[2].reshape(-1),
            output.layer_inputs[-1][0].reshape(-1),
        ]
        return torch.cat([part.to(torch.float32).reshape(-1) for part in parts], dim=0)

    def _run_predictive_refinement(
        self,
        layer_results: list[LayerBatchResult],
        groupings: list[list[list[int]]],
        *,
        learn: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[float], bool, float]:
        if self.predictive_hierarchy is None and self.prediction_error_gate is None:
            return [], [], [], [], False, 0.0

        feedforward_states = [
            self._summarize_activations(
                result.activations,
                result.confidences,
                target_dim=layer.concept_dim,
            )
            for layer, result in zip(self.layers, layer_results, strict=False)
        ]
        if self.predictive_hierarchy is not None:
            predictive_output = self.predictive_hierarchy.settle_states(
                feedforward_states,
                num_iterations=max(1, int(self.config.predictive.settling_steps)),
                learn=learn,
            )
            return (
                predictive_output.states,
                predictive_output.errors,
                [],
                predictive_output.free_energy_trace,
                predictive_output.converged,
                predictive_output.surprise,
            )

        grouping12, grouping23, _ = groupings
        higher_states = [
            self._expand_higher_activations(layer_results[1].activations, grouping12),
            self._expand_higher_activations(layer_results[2].activations, grouping23),
            layer_results[3].activations,
        ]
        predictive_output = self.prediction_error_gate.compute_error_gates(
            [result.activations for result in layer_results],
            higher_activations=higher_states,
            learn=learn,
        )
        return (
            predictive_output.states,
            predictive_output.errors,
            predictive_output.gates,
            predictive_output.free_energy_trace,
            True,
            predictive_output.surprise,
        )

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
        learning_rate_scale: float | torch.Tensor = 1.0,
    ) -> LayerBatchResult:
        activations: list[torch.Tensor] = []
        fired_indices: list[list[int]] = []
        confidences: list[float] = []
        recruited: list[bool] = []
        recruited_indices: list[int | None] = []
        layer.pool._sync_core()
        base_slow_lr = float(layer.pool.core.config.slow_lr)
        base_feedback_lr = float(layer.pool.core.config.feedback_lr)
        scale_vector = (
            learning_rate_scale.reshape(-1).to(torch.float32)
            if isinstance(learning_rate_scale, torch.Tensor)
            else None
        )

        try:
            for index in range(inputs.shape[0]):
                scale = (
                    float(scale_vector[min(index, scale_vector.numel() - 1)].clamp(0.0, 1.0).item())
                    if scale_vector is not None and scale_vector.numel() > 0
                    else max(0.0, float(learning_rate_scale))
                )
                layer.pool._sync_core()
                layer.pool.core.config.slow_lr = base_slow_lr * scale
                layer.pool.core.config.feedback_lr = base_feedback_lr * scale
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
        finally:
            layer.pool._sync_core()
            layer.pool.core.config.slow_lr = base_slow_lr
            layer.pool.core.config.feedback_lr = base_feedback_lr

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
        raw_signature = self._raw_label_signature_from_output(output)
        if (
            output.predictive_mode == "settling"
            and output.predictive_states
            and output.predictive_errors
        ):
            return self._predictive_signature(
                raw_signature,
                output.predictive_states,
                output.predictive_errors,
            )
        return raw_signature

    def _predict_label(
        self,
        evidence_feature: torch.Tensor,
        fired_indices: list[int],
        final_confidence: float,
    ) -> tuple[int, float]:
        if not fired_indices:
            prototype_label, prototype_similarity, prototype_margin = self._prototype_prediction(
                evidence_feature
            )
            if (
                prototype_label is None
                or prototype_similarity < 0.35
                or prototype_margin < 0.005
            ):
                return -1, float(prototype_similarity)
            return int(prototype_label), max(float(final_confidence), float(prototype_similarity))

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
        return int(prototype_label), max(float(final_confidence), float(prototype_similarity)        )

    @staticmethod
    def _mean_gate_value(gate: torch.Tensor | None) -> float:
        if gate is None or gate.numel() == 0:
            return 1.0
        return float(gate.to(torch.float32).mean().clamp(0.0, 1.0).item())

    @staticmethod
    def _expand_higher_activations(
        higher_activations: torch.Tensor,
        grouping: list[list[int]],
    ) -> torch.Tensor:
        if higher_activations.numel() == 0 or not grouping:
            return higher_activations
        batch = higher_activations.to(torch.float32)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        num_lower = max((max(indices) for indices in grouping if indices), default=-1) + 1
        if num_lower <= 0:
            return batch
        expanded = torch.zeros(
            num_lower,
            batch.shape[-1],
            device=batch.device,
            dtype=batch.dtype,
        )
        for higher_index, lower_indices in enumerate(grouping):
            source = batch[min(higher_index, batch.shape[0] - 1)]
            for lower_index in lower_indices:
                expanded[lower_index] = source
        return expanded

    def _run_hierarchy_pass(
        self,
        *,
        patch_grid: tuple[int, int],
        l1_inputs: torch.Tensor,
        feedback_signals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        learn: bool,
        label: int | None,
        learning_rate_scales: list[float] | None = None,
    ) -> tuple[list[torch.Tensor], list[LayerBatchResult], list[list[list[int]]]]:
        feedback_l1, feedback_l2, feedback_l3 = feedback_signals
        scales = learning_rate_scales or [1.0] * len(self.layers)

        l1_result = (
            self._run_layer_learn(
                self.layers[0],
                l1_inputs,
                allow_recruit=True,
                learning_rate_scale=scales[0],
            )
            if learn
            else self._run_layer_infer(self.layers[0], l1_inputs)
        )
        l1_result = self._apply_feedback_modulation(self.layers[0], l1_result, feedback_l1)

        l2_inputs, grouping12 = self._prepare_l2_inputs(l1_result.activations, patch_grid)
        l2_result = (
            self._run_layer_learn(
                self.layers[1],
                l2_inputs,
                allow_recruit=True,
                learning_rate_scale=scales[1],
            )
            if learn
            else self._run_layer_infer(self.layers[1], l2_inputs)
        )
        l2_result = self._apply_feedback_modulation(self.layers[1], l2_result, feedback_l2)

        l3_inputs, grouping23 = self._prepare_l3_inputs(l2_result.activations)
        l3_result = (
            self._run_layer_learn(
                self.layers[2],
                l3_inputs,
                allow_recruit=True,
                learning_rate_scale=scales[2],
            )
            if learn
            else self._run_layer_infer(self.layers[2], l3_inputs)
        )
        l3_result = self._apply_feedback_modulation(self.layers[2], l3_result, feedback_l3)

        l4_inputs, grouping34 = self._prepare_l4_inputs(l3_result.activations)
        if learn and label is not None:
            l4_result = self._run_layer_learn(
                self.layers[3],
                l4_inputs,
                allow_recruit=True,
                learning_rate_scale=scales[3],
            )
        else:
            l4_result = self._run_layer_infer(self.layers[3], l4_inputs)
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

        return (
            [l1_inputs, l2_inputs, l3_inputs, l4_inputs],
            [l1_result, l2_result, l3_result, l4_result],
            [grouping12, grouping23, grouping34],
        )

    def predictive_is_novel(self, output: HierarchyOutput | None = None) -> bool:
        candidate = output or self.last_output
        if candidate is None or candidate.predictive_mode == "disabled":
            return False
        return bool(candidate.predictive_surprise > self.predictive_novelty_threshold)

    def _forward(
        self,
        image: torch.Tensor,
        *,
        learn: bool,
        label: int | None = None,
    ) -> HierarchyOutput:
        feedback_l1, feedback_l2, feedback_l3 = self._previous_feedback_signals()
        patches = self.extractor.extract_patches(
            image,
            patch_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        if self.attention is not None and patches:
            frame = self.extractor._ensure_image(image)
            gains = self.attention.patch_gains(
                frame,
                self.extractor.last_patch_positions,
                patch_size=self.config.patch_size,
            )
            patches = self.attention.apply_to_patches(
                patches,
                gains,
                sensory_dim=self.config.patch_size * self.config.patch_size * self.config.channels,
            )
        patch_grid = self.extractor.last_grid_shape
        l1_inputs = self._stack(patches, self.config.l1_input_dim)
        layer_inputs: list[torch.Tensor]
        layer_results: list[LayerBatchResult]
        groupings: list[list[list[int]]]

        if learn and self.predictive_mode == "error_gating" and self.prediction_error_gate is not None:
            _, preview_results, preview_groupings = self._run_hierarchy_pass(
                patch_grid=patch_grid,
                l1_inputs=l1_inputs,
                feedback_signals=(feedback_l1, feedback_l2, feedback_l3),
                learn=False,
                label=label,
            )
            (
                _preview_states,
                _preview_errors,
                preview_gates,
                _preview_trace,
                _preview_converged,
                _preview_surprise,
            ) = self._run_predictive_refinement(preview_results, preview_groupings, learn=True)
            learning_rate_scales = [
            (preview_gates[index] if index < len(preview_gates) else 1.0)
                for index in range(len(self.layers))
            ]
            layer_inputs, layer_results, groupings = self._run_hierarchy_pass(
                patch_grid=patch_grid,
                l1_inputs=l1_inputs,
                feedback_signals=(feedback_l1, feedback_l2, feedback_l3),
                learn=True,
                label=label,
                learning_rate_scales=learning_rate_scales,
            )
        else:
            layer_inputs, layer_results, groupings = self._run_hierarchy_pass(
                patch_grid=patch_grid,
                l1_inputs=l1_inputs,
                feedback_signals=(feedback_l1, feedback_l2, feedback_l3),
                learn=learn,
                label=label,
            )

        (
            predictive_states,
            predictive_errors,
            predictive_gates,
            predictive_free_energy_trace,
            predictive_converged,
            predictive_surprise,
        ) = self._run_predictive_refinement(
            layer_results,
            groupings,
            learn=(learn and self.predictive_mode == "settling"),
        )
        l1_result, l2_result, l3_result, l4_result = layer_results
        grouping12, grouping23, grouping34 = groupings
        l1_inputs, l2_inputs, l3_inputs, l4_inputs = layer_inputs
        hierarchy_activations = {
            "V1": l1_result.activations,
            "V2": l2_result.activations,
            "V4": l3_result.activations,
            "IT": l4_result.activations,
        }
        hierarchy_prediction_errors = self.compute_hierarchy_prediction_errors(hierarchy_activations)
        hierarchy_uncertainty = self._mean_prediction_error(hierarchy_prediction_errors)
        self._propagate_hierarchy_prediction_error(hierarchy_uncertainty)

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
            predictive_states=predictive_states,
            predictive_errors=predictive_errors,
            prediction_error_gates=predictive_gates,
            predictive_free_energy_trace=predictive_free_energy_trace,
            predictive_converged=predictive_converged,
            predictive_surprise=predictive_surprise,
            predictive_mode=self.predictive_mode,
            hierarchy_prediction_errors=hierarchy_prediction_errors,
            hierarchy_uncertainty=hierarchy_uncertainty,
        )

        if learn:
            self._update_hierarchy_predictors(hierarchy_activations)
            self._update_bindings(output)
            self._update_feedback_connections(output)
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
    def start_new_task(self) -> None:
        if int(self.config.freeze_f1_after) <= 0:
            return
        for layer in self.layers:
            layer.pool.freeze_f1()
            layer.pool.create_task_adapter()

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
