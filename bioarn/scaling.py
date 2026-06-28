"""Scaling-oriented Bio-ARN components for larger CCC and SDM deployments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import BioARNConfig, CCCConfig, MarginGateConfig, SDMConfig
from bioarn.core.ccc import CCCOutput, CCCPool, CCCPoolOutput, F1Adapter
from bioarn.core.consolidation import SynapticConsolidation
from bioarn.core.margin_gate import MarginGateOutput, ResonanceOutput
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.associative_fabric import AssociativeFabric
from bioarn.memory.sdm import SparseDistributedMemory, TemporalAssociator
from bioarn.predictive.lateral_prediction import LateralPredictionNetwork
from bioarn.predictive.precision_weighting import PrecisionWeightedGate
from bioarn.system import BioARNCore


@dataclass
class ProfileResult:
    operation: str
    scale_points: list[int]
    times_ms: list[float]
    memory_mb: list[float]
    scaling_order: str


@dataclass
class ComparisonResult:
    original_times: list[float]
    optimized_times: list[float]
    speedup_factors: list[float]
    correctness_verified: bool


@dataclass
class PoolInferenceSummary:
    """Compact pool inference result that avoids materializing per-CCC Python objects."""

    fired_indices: list[int]
    winner_confidences: torch.Tensor
    recruited: bool
    recruited_index: int | None
    num_fired: int
    num_abstained: int
    sparsity: float
    mean_confidence: float


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        return (
            tensor.indices().numel() * tensor.indices().element_size()
            + tensor.values().numel() * tensor.values().element_size()
        )
    return tensor.numel() * tensor.element_size()


def _iter_extra_tensors(module: Any) -> Iterable[torch.Tensor]:
    extra = getattr(module, "storage_tensors", None)
    if callable(extra):
        yield from (tensor for tensor in extra() if isinstance(tensor, torch.Tensor))

    sparse_cache = getattr(module, "sparse_data_matrix", None)
    if isinstance(sparse_cache, torch.Tensor):
        yield sparse_cache


def estimate_module_memory_mb(module: nn.Module | Any) -> float:
    """Estimate module memory usage from parameter, buffer, and auxiliary tensor storage."""

    total_bytes = 0
    seen: set[tuple[int, tuple[int, ...]]] = set()

    tensors: list[torch.Tensor] = []
    if isinstance(module, nn.Module):
        tensors.extend(module.parameters())
        tensors.extend(module.buffers())
    tensors.extend(_iter_extra_tensors(module))

    for tensor in tensors:
        if tensor.is_sparse:
            key = (id(tensor), tuple(tensor.shape))
        else:
            key = (tensor.data_ptr(), tuple(tensor.shape))
        if key in seen:
            continue
        seen.add(key)
        total_bytes += _tensor_bytes(tensor)
    return total_bytes / (1024.0 * 1024.0)


def _copy_batched_pool_slice(
    source: BatchedCCCPool,
    target: BatchedCCCPool,
    *,
    source_start: int = 0,
    target_start: int = 0,
    length: int | None = None,
) -> None:
    if length is None:
        length = min(
            source.config.max_pool_size - source_start,
            target.config.max_pool_size - target_start,
        )
    if length <= 0:
        return

    source_slice = slice(source_start, source_start + length)
    target_slice = slice(target_start, target_start + length)
    target.f1_weights[target_slice].copy_(source.f1_weights[source_slice])
    target.f1_bias[target_slice].copy_(source.f1_bias[source_slice])
    target.f2_weights[target_slice].copy_(source.f2_weights[source_slice])
    target.feedback_weights[target_slice].copy_(source.feedback_weights[source_slice])
    target.concept_directions[target_slice].copy_(source.concept_directions[source_slice])
    target.committed_mask[target_slice].copy_(source.committed_mask[source_slice])
    target.age[target_slice].copy_(source.age[source_slice])
    target.last_fired[target_slice].copy_(source.last_fired[source_slice])
    target.theta_margin[target_slice].copy_(source.theta_margin[source_slice])
    target.theta_resonance[target_slice].copy_(source.theta_resonance[source_slice])
    target.total_presentations[target_slice].copy_(source.total_presentations[source_slice])
    target.total_fires[target_slice].copy_(source.total_fires[source_slice])
    target.total_abstentions[target_slice].copy_(source.total_abstentions[source_slice])
    target.avg_confidence_when_fired[target_slice].copy_(
        source.avg_confidence_when_fired[source_slice]
    )
    target.avg_confidence_when_abstained[target_slice].copy_(
        source.avg_confidence_when_abstained[source_slice]
    )
    target.adapter_assignments[target_slice].copy_(source.adapter_assignments[source_slice])
    target.consolidation.copy_slice_from(
        source.consolidation,
        source_start=source_start,
        target_start=target_start,
        length=length,
    )
    target._copy_shared_state_from(source)


@dataclass
class _BatchedForwardState:
    base_f1_output: torch.Tensor
    f1_output: torch.Tensor
    f2_activation: torch.Tensor
    confidence: torch.Tensor
    fired: torch.Tensor
    abstained: torch.Tensor
    gate_output: torch.Tensor
    prediction: torch.Tensor
    match_score: torch.Tensor
    resonated: torch.Tensor
    learn_signal: torch.Tensor
    any_fired: bool


class BatchedCCCPool(nn.Module):
    """CCC pool implemented with batched tensor operations instead of Python loops."""

    def __init__(self, config: CCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config
        self.margin_config = margin_config
        self.initial_capacity = int(config.max_pool_size)
        self.max_capacity = max(
            self.initial_capacity,
            int(math.ceil(self.initial_capacity * float(config.max_growth_factor))),
        )
        self.base_config = replace(config, max_pool_size=self.initial_capacity)

        num_cccs = self.initial_capacity
        shared_f1 = nn.Linear(config.input_dim, config.num_f1_features)

        self.register_buffer(
            "f1_weights",
            shared_f1.weight.detach().clone().unsqueeze(0).repeat(num_cccs, 1, 1),
        )
        self.register_buffer(
            "f1_bias",
            shared_f1.bias.detach().clone().unsqueeze(0).repeat(num_cccs, 1),
        )
        self.register_buffer(
            "f2_weights",
            normalize(
                torch.randn(
                    num_cccs,
                    config.concept_dim,
                    config.num_f1_features,
                    dtype=torch.float32,
                )
            ),
        )
        self.register_buffer(
            "feedback_weights",
            torch.zeros(
                num_cccs,
                config.num_f1_features,
                config.concept_dim,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "concept_directions",
            torch.zeros(num_cccs, config.concept_dim, dtype=torch.float32),
        )
        self.register_buffer("committed_mask", torch.zeros(num_cccs, dtype=torch.bool))
        self.register_buffer("adapter_assignments", torch.full((num_cccs,), -1, dtype=torch.long))
        self.register_buffer("age", torch.zeros(num_cccs, dtype=torch.long))
        self.register_buffer("last_fired", torch.full((num_cccs,), -1, dtype=torch.long))
        self.register_buffer(
            "theta_margin",
            torch.full((num_cccs,), float(margin_config.theta_margin), dtype=torch.float32),
        )
        self.register_buffer(
            "theta_resonance",
            torch.full((num_cccs,), float(margin_config.theta_resonance), dtype=torch.float32),
        )
        self.register_buffer("total_presentations", torch.zeros(num_cccs, dtype=torch.long))
        self.register_buffer("total_fires", torch.zeros(num_cccs, dtype=torch.long))
        self.register_buffer("total_abstentions", torch.zeros(num_cccs, dtype=torch.long))
        self.register_buffer(
            "avg_confidence_when_fired",
            torch.zeros(num_cccs, dtype=torch.float32),
        )
        self.register_buffer(
            "avg_confidence_when_abstained",
            torch.zeros(num_cccs, dtype=torch.float32),
        )
        self.consolidation = SynapticConsolidation(
            num_cccs,
            strength=self.config.consolidation_strength,
            device=self.concept_directions.device,
            dtype=torch.float32,
        )
        self.task_adapters = nn.ModuleList()
        self.active_adapter_id = -1
        self.register_buffer("f1_samples_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("f1_frozen", torch.tensor(False, dtype=torch.bool))
        precision_config = getattr(self.config, "precision", None)
        self.precision_gate = (
            PrecisionWeightedGate(precision_config)
            if precision_config is not None and bool(precision_config.enabled)
            else None
        )
        if self.precision_gate is not None:
            self.precision_gate.set_pool_size(num_cccs)
        lateral_config = getattr(self.config, "lateral_prediction", None)
        self.lateral_network = (
            LateralPredictionNetwork(
                num_cccs,
                self.config.concept_dim,
                lateral_config,
            )
            if lateral_config is not None and bool(lateral_config.enabled)
            else None
        )
        self.last_lateral_prediction_error = 0.0
        self.last_hierarchy_prediction_error = 0.0

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() != 2:
            raise ValueError("CCC inputs must have shape (input_dim,) or (batch, input_dim).")
        return x, False

    @staticmethod
    def _maybe_squeeze(x: torch.Tensor | None, squeeze: bool) -> torch.Tensor | None:
        if x is None:
            return None
        return x.squeeze(0) if squeeze else x

    @staticmethod
    def _confidence_score(confidence: torch.Tensor) -> torch.Tensor:
        return confidence.reshape(-1).mean()

    @property
    def current_adapter(self) -> F1Adapter | None:
        self._ensure_runtime_state()
        if 0 <= self.active_adapter_id < len(self.task_adapters):
            return self.task_adapters[self.active_adapter_id]
        return None

    def _ensure_runtime_buffers(self) -> None:
        device = self.committed_mask.device
        if "f1_samples_seen" not in self._buffers:
            self.register_buffer("f1_samples_seen", torch.tensor(0, dtype=torch.long, device=device))
        if "f1_frozen" not in self._buffers:
            self.register_buffer("f1_frozen", torch.tensor(False, dtype=torch.bool, device=device))

    def _ensure_runtime_state(self) -> None:
        self._ensure_runtime_buffers()
        device = self.committed_mask.device
        if "adapter_assignments" not in self._buffers:
            self.register_buffer(
                "adapter_assignments",
                torch.full(self.committed_mask.shape, -1, dtype=torch.long, device=device),
            )
        if not hasattr(self, "task_adapters"):
            self.task_adapters = nn.ModuleList()
        if not hasattr(self, "active_adapter_id"):
            self.active_adapter_id = -1
        if not hasattr(self, "precision_gate"):
            precision_config = getattr(self.config, "precision", None)
            self.precision_gate = (
                PrecisionWeightedGate(precision_config)
                if precision_config is not None and bool(precision_config.enabled)
                else None
            )
            if self.precision_gate is not None:
                self.precision_gate.set_pool_size(int(self.config.max_pool_size))
        if not hasattr(self, "lateral_network"):
            lateral_config = getattr(self.config, "lateral_prediction", None)
            self.lateral_network = (
                LateralPredictionNetwork(
                    int(self.config.max_pool_size),
                    self.config.concept_dim,
                    lateral_config,
                )
                if lateral_config is not None and bool(lateral_config.enabled)
                else None
            )
        if not hasattr(self, "last_lateral_prediction_error"):
            self.last_lateral_prediction_error = 0.0
        if not hasattr(self, "last_hierarchy_prediction_error"):
            self.last_hierarchy_prediction_error = 0.0

    def freeze_f1(self) -> None:
        self._ensure_runtime_state()
        self.f1_frozen.fill_(True)

    def observe_samples(self, sample_count: int) -> None:
        self._ensure_runtime_state()
        count = int(max(0, sample_count))
        if count <= 0:
            return
        self.f1_samples_seen.add_(count)
        if (
            not bool(self.f1_frozen.item())
            and int(self.config.freeze_f1_after) > 0
            and int(self.f1_samples_seen.item()) >= int(self.config.freeze_f1_after)
        ):
            self.freeze_f1()

    def create_task_adapter(self) -> F1Adapter | None:
        self._ensure_runtime_state()
        if int(self.config.freeze_f1_after) <= 0:
            return None
        self.freeze_f1()
        current = self.current_adapter
        if current is not None:
            current.freeze()
        adapter = F1Adapter(
            self.config.num_f1_features,
            adapter_dim=self.config.f1_adapter_dim,
        )
        adapter.unfreeze()
        self.task_adapters.append(adapter)
        self.active_adapter_id = len(self.task_adapters) - 1
        return adapter

    def _copy_shared_state_from(self, source: "BatchedCCCPool") -> None:
        self._ensure_runtime_state()
        source._ensure_runtime_state()
        cloned_adapters = nn.ModuleList()
        for adapter in source.task_adapters:
            cloned = F1Adapter(source.config.num_f1_features, adapter_dim=source.config.f1_adapter_dim)
            cloned.load_state_dict(adapter.state_dict())
            if bool(adapter.is_frozen.item()):
                cloned.freeze()
            else:
                cloned.unfreeze()
            cloned_adapters.append(cloned)
        self.task_adapters = cloned_adapters
        self.active_adapter_id = int(source.active_adapter_id)
        self.f1_samples_seen.copy_(source.f1_samples_seen)
        self.f1_frozen.copy_(source.f1_frozen)

    def _copy_precision_state_from(self, source: CCCPool | "BatchedCCCPool") -> None:
        if self.precision_gate is None:
            return
        source_precision = getattr(source, "precision_gate", None)
        if source_precision is None:
            return
        self.precision_gate.load_state_from(source_precision)
        self.precision_gate.set_pool_size(int(self.config.max_pool_size))

    def _copy_lateral_state_from(self, source: CCCPool | "BatchedCCCPool") -> None:
        if self.lateral_network is None:
            return
        source_lateral = getattr(source, "lateral_network", None)
        if source_lateral is None:
            return
        self.lateral_network.copy_state_from(source_lateral)
        self.lateral_network.set_pool_size(int(self.config.max_pool_size))
        self.last_lateral_prediction_error = float(
            getattr(source, "last_lateral_prediction_error", 0.0)
        )
        self.last_hierarchy_prediction_error = float(
            getattr(source, "last_hierarchy_prediction_error", 0.0)
        )

    def _adapter_lr(self, base_lr: float) -> float:
        return min(0.05, max(1e-3, float(base_lr)))

    def _shared_f1_encode(self, raw_batch: torch.Tensor) -> torch.Tensor:
        projected = (
            raw_batch.to(self.f1_weights.dtype) @ self.f1_weights[0].transpose(0, 1)
        ) + self.f1_bias[0]
        activated = F.relu(projected)
        top_k = min(self.config.f1_top_k, activated.shape[-1])
        values, indices = torch.topk(activated, k=top_k, dim=-1)
        return torch.zeros_like(activated).scatter(-1, indices, values)

    def _apply_adapter(self, f1_batch: torch.Tensor, adapter_id: int) -> torch.Tensor:
        self._ensure_runtime_state()
        if adapter_id < 0 or adapter_id >= len(self.task_adapters):
            return f1_batch
        adapted = self.task_adapters[adapter_id](f1_batch)
        top_k = min(self.config.f1_top_k, adapted.shape[-1])
        values, indices = torch.topk(F.relu(adapted), k=top_k, dim=-1)
        return torch.zeros_like(adapted).scatter(-1, indices, values)

    def grow(self, new_size: int | None = None) -> "BatchedCCCPool":
        current_size = int(self.config.max_pool_size)
        if current_size >= self.max_capacity:
            return self

        target = new_size or max(current_size + 1, int(math.ceil(current_size * 1.5)))
        target = min(self.max_capacity, int(target))
        if target <= current_size:
            return self

        grown = BatchedCCCPool(
            replace(self.base_config, max_pool_size=target),
            self.margin_config,
        )
        grown.initial_capacity = self.initial_capacity
        grown.max_capacity = self.max_capacity
        grown.base_config = self.base_config
        _copy_batched_pool_slice(self, grown, length=current_size)
        grown._copy_precision_state_from(self)
        grown._copy_lateral_state_from(self)
        if grown.precision_gate is not None:
            grown.precision_gate.set_pool_size(target)
        if grown.lateral_network is not None:
            grown.lateral_network.set_pool_size(target)
        return grown

    def update_importance(
        self,
        fired_indices: list[int] | torch.Tensor,
        *,
        confidences: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        return self.consolidation.update_importance(
            fired_indices,
            current_size=int(self.config.max_pool_size),
            confidences=confidences,
        )

    def set_hierarchy_prediction_error(self, error: float | None) -> None:
        self.last_hierarchy_prediction_error = 0.0 if error is None else max(0.0, min(1.0, float(error)))

    def _lateral_attention_map(self, errors: dict[int, float], fired_indices: list[int]) -> dict[int, float]:
        if not fired_indices:
            return {}
        gain = (
            float(self.lateral_network.config.surprise_gain)
            if self.lateral_network is not None
            else 1.0
        )
        if self.precision_gate is None:
            return {
                int(index): float(1.0 + (max(0.0, min(1.0, errors.get(int(index), 0.0))) * gain))
                for index in fired_indices
            }
        return {
            int(index): float(
                self.precision_gate.compute_error_attention(
                    errors.get(int(index), 0.0),
                    gain=gain,
                )
            )
            for index in fired_indices
        }

    def _update_lateral_prediction_state(self, fired_indices: list[int]) -> float:
        if self.lateral_network is None:
            self.last_lateral_prediction_error = 0.0
            return 0.0
        predictions = self.lateral_network.predict_lateral(fired_indices, self.concept_directions)
        errors = self.lateral_network.compute_lateral_errors(predictions, fired_indices)
        attention = self._lateral_attention_map(errors, fired_indices)
        self.lateral_network.cache_error_state(predictions, errors, attention=attention)
        self.last_lateral_prediction_error = self.lateral_network.summarize_errors(errors)
        return self.last_lateral_prediction_error

    @torch.no_grad()
    def hebbian_update_lateral(self, fired_indices: list[int] | None = None) -> None:
        if self.lateral_network is None:
            return
        active_indices = [] if fired_indices is None else [int(index) for index in fired_indices]
        self.lateral_network.hebbian_update(active_indices, self.concept_directions)

    def _row_f1_encode(self, index: int, raw_batch: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime_state()
        base_f1 = self._shared_f1_encode(raw_batch)
        return self._apply_adapter(base_f1, int(self.adapter_assignments[index].item()))

    def _row_f2_activate(self, index: int, f1_batch: torch.Tensor) -> torch.Tensor:
        return f1_batch.to(self.f2_weights.dtype) @ self.f2_weights[index].transpose(0, 1)

    def _row_predict(self, index: int, f2_batch: torch.Tensor) -> torch.Tensor:
        return f2_batch.to(self.feedback_weights.dtype) @ self.feedback_weights[index].transpose(0, 1)

    def _update_gate_stats(
        self,
        confidence: torch.Tensor,
        fired: torch.Tensor,
        committed: torch.Tensor,
    ) -> None:
        batch_size = confidence.shape[-1]
        committed_long = committed.to(torch.long)
        self.total_presentations.add_(committed_long * batch_size)

        fired_count = fired.sum(dim=-1).to(torch.long)
        abstained_count = committed_long * batch_size - fired_count

        prev_fires = self.total_fires.clone()
        prev_abstentions = self.total_abstentions.clone()

        self.total_fires.add_(fired_count)
        self.total_abstentions.add_(abstained_count)

        fired_conf_sum = (confidence * fired.to(confidence.dtype)).sum(dim=-1)
        fired_total = prev_fires + fired_count
        fired_updates = torch.where(
            fired_count > 0,
            (
                self.avg_confidence_when_fired * prev_fires.to(torch.float32)
                + fired_conf_sum.to(torch.float32)
            )
            / fired_total.clamp_min(1).to(torch.float32),
            self.avg_confidence_when_fired,
        )
        self.avg_confidence_when_fired.copy_(fired_updates)

        abstained_mask = committed.unsqueeze(-1) & ~fired
        abstained_conf_sum = (confidence * abstained_mask.to(confidence.dtype)).sum(dim=-1)
        abstained_total = prev_abstentions + abstained_count
        abstained_updates = torch.where(
            abstained_count > 0,
            (
                self.avg_confidence_when_abstained * prev_abstentions.to(torch.float32)
                + abstained_conf_sum.to(torch.float32)
            )
            / abstained_total.clamp_min(1).to(torch.float32),
            self.avg_confidence_when_abstained,
        )
        self.avg_confidence_when_abstained.copy_(abstained_updates)

    def _apply_slow_learning(
        self,
        base_f1_output: torch.Tensor,
        f1_output: torch.Tensor,
        f2_activation: torch.Tensor,
        resonated: torch.Tensor,
        learn_signal: torch.Tensor,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        active_learning = resonated.any(dim=-1)
        if not bool(active_learning.any().item()):
            return

        learning_mask = learn_signal.to(f2_activation.dtype).unsqueeze(-1)
        concept_delta = (f2_activation * learning_mask).mean(dim=1)
        lr_scale = self.consolidation.learning_rate_scales(
            current_size=int(self.config.max_pool_size),
            device=f2_activation.device,
            dtype=f2_activation.dtype,
        )
        if isinstance(learning_rate_multiplier, torch.Tensor):
            lr_multiplier = learning_rate_multiplier.to(device=f2_activation.device, dtype=f2_activation.dtype).reshape(-1)
            if lr_multiplier.numel() == 1:
                lr_multiplier = lr_multiplier.expand(int(self.config.max_pool_size))
        else:
            lr_multiplier = torch.full(
                (int(self.config.max_pool_size),),
                float(learning_rate_multiplier),
                device=f2_activation.device,
                dtype=f2_activation.dtype,
            )
        lr_multiplier = lr_multiplier.clamp_min(0.0)
        concept_lr = float(self.config.slow_lr) * lr_scale * lr_multiplier
        updated_direction = self.concept_directions + (concept_lr.unsqueeze(-1) * concept_delta)
        normalized_update = normalize(updated_direction)
        self.concept_directions.copy_(
            torch.where(active_learning.unsqueeze(-1), normalized_update, self.concept_directions)
        )

        prediction_full = torch.matmul(self.feedback_weights, f2_activation.transpose(1, 2)).transpose(1, 2)
        residual = (f1_output - prediction_full) * learning_mask.to(f1_output.dtype)
        hebbian = torch.matmul(residual.transpose(1, 2), f2_activation)
        hebbian = hebbian / max(f1_output.shape[1], 1)
        feedback_lr = float(self.config.feedback_lr) * lr_scale * lr_multiplier
        updated_feedback = self.feedback_weights + (feedback_lr.view(-1, 1, 1) * hebbian)
        normalized_feedback = normalize(updated_feedback)
        self.feedback_weights.copy_(
            torch.where(
                active_learning.view(-1, 1, 1),
                normalized_feedback,
                self.feedback_weights,
            )
        )
        if not self.task_adapters:
            return
        for adapter_id in torch.unique(self.adapter_assignments[active_learning]).tolist():
            adapter_id = int(adapter_id)
            if adapter_id < 0 or adapter_id >= len(self.task_adapters):
                continue
            adapter_mask = active_learning & (self.adapter_assignments == adapter_id)
            if not bool(adapter_mask.any().item()):
                continue
            adapter_signal = learn_signal[adapter_mask].amax(dim=0).to(base_f1_output.dtype)
            self.task_adapters[adapter_id].hebbian_update(
                base_f1_output,
                learning_signal=adapter_signal,
                lr=self._adapter_lr(self.config.slow_lr) * float(lr_multiplier[adapter_mask].mean().item()),
            )

    def _vectorized_state(
        self,
        raw_batch: torch.Tensor,
        timestep: int,
        *,
        apply_learning: bool = True,
        update_state: bool = True,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> _BatchedForwardState:
        self._ensure_runtime_state()
        inputs = raw_batch.to(self.f1_weights.dtype)
        num_cccs = int(self.config.max_pool_size)
        batch_size = raw_batch.shape[0]
        committed = self.committed_mask
        if update_state:
            self.age.add_(committed.to(self.age.dtype) * raw_batch.shape[0])
        device = inputs.device
        dtype = inputs.dtype

        base_f1_output = self._shared_f1_encode(raw_batch)
        f1_output = torch.zeros(
            num_cccs,
            batch_size,
            self.config.num_f1_features,
            device=device,
            dtype=dtype,
        )
        f2_activation = torch.zeros(
            num_cccs,
            batch_size,
            self.config.concept_dim,
            device=device,
            dtype=self.f2_weights.dtype,
        )
        confidence = torch.zeros(num_cccs, batch_size, device=device, dtype=torch.float32)
        fired = torch.zeros(num_cccs, batch_size, device=device, dtype=torch.bool)
        abstained = torch.ones(num_cccs, batch_size, device=device, dtype=torch.bool)
        gate_output = torch.zeros_like(f2_activation)
        prediction = torch.zeros(
            num_cccs,
            batch_size,
            self.config.num_f1_features,
            device=device,
            dtype=self.feedback_weights.dtype,
        )
        match_score = torch.zeros(num_cccs, batch_size, device=device, dtype=torch.float32)
        resonated = torch.zeros(num_cccs, batch_size, device=device, dtype=torch.bool)
        learn_signal = torch.zeros(num_cccs, batch_size, device=device, dtype=torch.float32)

        committed_indices = committed.nonzero(as_tuple=False).squeeze(-1)
        if committed_indices.numel() > 0:
            committed_directions = normalize(self.concept_directions.index_select(0, committed_indices)).unsqueeze(1)
            committed_adapters = self.adapter_assignments.index_select(0, committed_indices)
            for adapter_id in torch.unique(committed_adapters).tolist():
                adapter_id = int(adapter_id)
                adapter_mask = committed_adapters == adapter_id
                adapter_indices = committed_indices[adapter_mask]
                adapted_f1 = self._apply_adapter(base_f1_output, adapter_id)
                committed_count = adapter_indices.numel()
                committed_f1 = adapted_f1.unsqueeze(0).expand(committed_count, -1, -1)
                committed_f2_weights = self.f2_weights.index_select(0, adapter_indices)
                committed_f2 = torch.matmul(
                    committed_f2_weights,
                    adapted_f1.transpose(0, 1),
                ).transpose(1, 2)
                adapter_directions = committed_directions[adapter_mask]
                committed_confidence = (
                    normalize(committed_f2).to(adapter_directions.dtype) * adapter_directions
                ).sum(dim=-1)
                committed_fired = (
                    committed_confidence
                    > self.theta_margin.index_select(0, adapter_indices).unsqueeze(-1)
                )
                committed_gate = torch.where(
                    committed_fired.unsqueeze(-1),
                    committed_f2,
                    torch.zeros_like(committed_f2),
                )
                committed_feedback = self.feedback_weights.index_select(0, adapter_indices)
                committed_prediction = torch.matmul(
                    committed_feedback,
                    committed_gate.transpose(1, 2),
                ).transpose(1, 2)
                committed_match = cosine_similarity(committed_prediction, committed_f1)
                resonance_margin = self.theta_resonance.index_select(0, adapter_indices).unsqueeze(-1)
                committed_resonated = committed_fired & (committed_match > resonance_margin)
                committed_learn = torch.where(
                    committed_resonated,
                    (
                        (committed_match - resonance_margin)
                        / (1.0 - resonance_margin).clamp_min(1e-6)
                    ).clamp(min=0.0, max=1.0),
                    torch.zeros_like(committed_match),
                )

                f1_output.index_copy_(0, adapter_indices, committed_f1)
                f2_activation.index_copy_(0, adapter_indices, committed_f2)
                confidence.index_copy_(0, adapter_indices, committed_confidence)
                fired.index_copy_(0, adapter_indices, committed_fired)
                abstained.index_copy_(0, adapter_indices, ~committed_fired)
                gate_output.index_copy_(0, adapter_indices, committed_gate)
                prediction.index_copy_(0, adapter_indices, committed_prediction)
                match_score.index_copy_(0, adapter_indices, committed_match)
                resonated.index_copy_(0, adapter_indices, committed_resonated)
                learn_signal.index_copy_(0, adapter_indices, committed_learn)

        if update_state:
            self._update_gate_stats(confidence, fired, committed)
        if apply_learning:
            self._apply_slow_learning(
                base_f1_output,
                f1_output,
                f2_activation,
                resonated,
                learn_signal,
                learning_rate_multiplier=learning_rate_multiplier,
            )

        fired_any = fired.any(dim=-1)
        if update_state:
            self.last_fired.copy_(
                torch.where(
                    fired_any,
                    torch.full_like(self.last_fired, int(timestep)),
                    self.last_fired,
                )
            )

        return _BatchedForwardState(
            base_f1_output=base_f1_output,
            f1_output=f1_output,
            f2_activation=f2_activation,
            confidence=confidence,
            fired=fired,
            abstained=abstained,
            gate_output=gate_output,
            prediction=prediction,
            match_score=match_score,
            resonated=resonated,
            learn_signal=learn_signal,
            any_fired=bool(fired.any().item()),
        )

    def _single_slow_learn(
        self,
        index: int,
        raw_batch: torch.Tensor,
        f1_batch: torch.Tensor,
        f2_batch: torch.Tensor,
        resonance: ResonanceOutput,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        learn_signal = resonance.learn_signal.reshape(-1).to(f2_batch.dtype)
        concept_delta = (f2_batch * learn_signal.unsqueeze(-1)).mean(dim=0)
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        concept_lr = self.consolidation.effective_lr(self.config.slow_lr, index) * lr_multiplier
        updated_direction = self.concept_directions[index] + (concept_lr * concept_delta)
        self.concept_directions[index].copy_(normalize(updated_direction.unsqueeze(0)).squeeze(0))

        prediction = self._row_predict(index, f2_batch)
        residual = (f1_batch - prediction) * learn_signal.unsqueeze(-1)
        hebbian = residual.transpose(0, 1) @ f2_batch
        hebbian /= max(f1_batch.shape[0], 1)
        feedback_lr = self.consolidation.effective_lr(self.config.feedback_lr, index) * lr_multiplier
        updated_feedback = self.feedback_weights[index] + (feedback_lr * hebbian)
        self.feedback_weights[index].copy_(normalize(updated_feedback))
        adapter_id = int(self.adapter_assignments[index].item())
        if 0 <= adapter_id < len(self.task_adapters):
            self.task_adapters[adapter_id].hebbian_update(
                self._shared_f1_encode(raw_batch),
                learning_signal=learn_signal,
                lr=self._adapter_lr(self.config.slow_lr) * lr_multiplier,
            )

    def _single_forward_index(
        self,
        index: int,
        raw_batch: torch.Tensor,
        squeeze: bool,
        timestep: int,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> CCCOutput:
        f1_batch = self._row_f1_encode(index, raw_batch)
        f2_batch = self._row_f2_activate(index, f1_batch)

        self.age[index].add_(raw_batch.shape[0])
        self.total_presentations[index].add_(raw_batch.shape[0])

        confidence = cosine_similarity(f2_batch, self.concept_directions[index].unsqueeze(0))
        fired = confidence > self.theta_margin[index]
        abstained = ~fired
        gate_tensor = torch.where(fired.unsqueeze(-1), f2_batch, torch.zeros_like(f2_batch))

        fired_count = int(fired.sum().item())
        abstained_count = int(abstained.sum().item())
        prev_fires = int(self.total_fires[index].item())
        prev_abstentions = int(self.total_abstentions[index].item())
        self.total_fires[index].add_(fired_count)
        self.total_abstentions[index].add_(abstained_count)

        if fired_count:
            fired_mean = confidence[fired].mean()
            total = prev_fires + fired_count
            updated = (
                self.avg_confidence_when_fired[index] * prev_fires + fired_mean * fired_count
            ) / max(total, 1)
            self.avg_confidence_when_fired[index].copy_(updated)
        if abstained_count:
            abstained_mean = confidence[abstained].mean()
            total = prev_abstentions + abstained_count
            updated = (
                self.avg_confidence_when_abstained[index] * prev_abstentions
                + abstained_mean * abstained_count
            ) / max(total, 1)
            self.avg_confidence_when_abstained[index].copy_(updated)

        prediction_tensor = self._row_predict(index, gate_tensor)
        prediction_output: torch.Tensor | None = None
        resonance_output: ResonanceOutput | None = None

        if bool(fired.any().item()):
            match_score = cosine_similarity(prediction_tensor, f1_batch)
            resonated = match_score > self.theta_resonance[index]
            learn_signal = torch.where(
                resonated,
                (
                    (match_score - self.theta_resonance[index])
                    / max(1.0 - float(self.theta_resonance[index].item()), 1e-6)
                ).clamp(min=0.0, max=1.0),
                torch.zeros_like(match_score),
            )
            resonance_output = ResonanceOutput(
                match_score=self._maybe_squeeze(match_score, squeeze),
                resonated=self._maybe_squeeze(resonated, squeeze),
                learn_signal=self._maybe_squeeze(learn_signal, squeeze),
            )
            prediction_output = self._maybe_squeeze(prediction_tensor, squeeze)
            if bool(resonated.any().item()):
                self._single_slow_learn(
                    index,
                    raw_batch,
                    f1_batch,
                    f2_batch,
                    ResonanceOutput(match_score=match_score, resonated=resonated, learn_signal=learn_signal),
                    learning_rate_multiplier=learning_rate_multiplier,
                )
            self.last_fired[index].fill_(int(timestep))

        gate_output = MarginGateOutput(
            output=self._maybe_squeeze(gate_tensor, squeeze),
            confidence=self._maybe_squeeze(confidence, squeeze),
            fired=self._maybe_squeeze(fired, squeeze),
            abstained=self._maybe_squeeze(abstained, squeeze),
        )
        return CCCOutput(
            fired=bool(fired.any().item()),
            abstained=bool(abstained.all().item()),
            confidence=self._maybe_squeeze(confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_batch, squeeze),
            f2_activation=self._maybe_squeeze(f2_batch, squeeze),
            gate_output=gate_output,
            prediction=prediction_output,
            resonance=resonance_output,
        )

    def _pack_outputs(self, state: _BatchedForwardState, squeeze: bool) -> list[CCCOutput]:
        outputs: list[CCCOutput] = []
        for index in range(self.config.max_pool_size):
            committed = bool(self.committed_mask[index].item())
            fired_tensor = state.fired[index]
            abstained_tensor = state.abstained[index]
            confidence = state.confidence[index]
            f1_output = state.f1_output[index]
            f2_activation = state.f2_activation[index]

            if not committed:
                gate_output = MarginGateOutput(
                    output=self._maybe_squeeze(torch.zeros_like(f2_activation), squeeze),
                    confidence=self._maybe_squeeze(torch.zeros_like(confidence), squeeze),
                    fired=self._maybe_squeeze(torch.zeros_like(fired_tensor, dtype=torch.bool), squeeze),
                    abstained=self._maybe_squeeze(torch.ones_like(abstained_tensor, dtype=torch.bool), squeeze),
                )
                outputs.append(
                    CCCOutput(
                        fired=False,
                        abstained=True,
                        confidence=self._maybe_squeeze(torch.zeros_like(confidence), squeeze),
                        f1_output=self._maybe_squeeze(torch.zeros_like(f1_output), squeeze),
                        f2_activation=self._maybe_squeeze(torch.zeros_like(f2_activation), squeeze),
                        gate_output=gate_output,
                        prediction=None,
                        resonance=None,
                    )
                )
                continue

            gate_output = MarginGateOutput(
                output=self._maybe_squeeze(state.gate_output[index], squeeze),
                confidence=self._maybe_squeeze(confidence, squeeze),
                fired=self._maybe_squeeze(fired_tensor, squeeze),
                abstained=self._maybe_squeeze(abstained_tensor, squeeze),
            )
            prediction: torch.Tensor | None = None
            resonance: ResonanceOutput | None = None
            if bool(fired_tensor.any().item()):
                prediction = self._maybe_squeeze(state.prediction[index], squeeze)
                resonance = ResonanceOutput(
                    match_score=self._maybe_squeeze(state.match_score[index], squeeze),
                    resonated=self._maybe_squeeze(state.resonated[index], squeeze),
                    learn_signal=self._maybe_squeeze(state.learn_signal[index], squeeze),
                )

            outputs.append(
                CCCOutput(
                    fired=bool(fired_tensor.any().item()),
                    abstained=bool(abstained_tensor.all().item()),
                    confidence=self._maybe_squeeze(confidence, squeeze),
                    f1_output=self._maybe_squeeze(f1_output, squeeze),
                    f2_activation=self._maybe_squeeze(f2_activation, squeeze),
                    gate_output=gate_output,
                    prediction=prediction,
                    resonance=resonance,
                )
            )
        return outputs

    def _first_uncommitted_index(self) -> int | None:
        available = (~self.committed_mask).to(torch.int64)
        if int(available.max().item()) == 0:
            return None
        return int(torch.argmax(available).item())

    @torch.no_grad()
    def fast_infer(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        allow_recruit: bool = False,
    ) -> PoolInferenceSummary:
        """Run compact vectorized inference without constructing per-CCC Python outputs."""

        raw_batch, _ = self._ensure_batch(raw_input)
        state = self._vectorized_state(
            raw_batch,
            timestep=timestep,
            apply_learning=False,
            update_state=False,
        )
        fired_mask = state.fired.any(dim=-1)
        fired_indices = fired_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        winner_confidences = (
            state.confidence[fired_mask].mean(dim=-1)
            if fired_indices
            else torch.empty(0, dtype=torch.float32, device=self.f1_weights.device)
        )
        recruited = False
        recruited_index: int | None = None

        if allow_recruit and not state.any_fired:
            recruited_index, recruited_output = self.recruit(raw_batch, timestep=timestep)
            if recruited_index is not None and recruited_output is not None:
                recruited = True
                fired_indices = [int(recruited_index)]
                confidence = recruited_output.confidence.reshape(-1).mean().to(torch.float32)
                winner_confidences = confidence.unsqueeze(0)

        num_fired = len(fired_indices)
        num_abstained = int(self.config.max_pool_size) - num_fired
        return PoolInferenceSummary(
            fired_indices=fired_indices,
            winner_confidences=winner_confidences.detach().clone(),
            recruited=recruited,
            recruited_index=recruited_index,
            num_fired=num_fired,
            num_abstained=num_abstained,
            sparsity=float(num_fired / max(int(self.config.max_pool_size), 1)),
            mean_confidence=float(winner_confidences.mean().item()) if num_fired else 0.0,
        )

    def _learn_fast_index(
        self,
        index: int,
        raw_batch: torch.Tensor,
        f1_batch: torch.Tensor,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        base_f1 = self._shared_f1_encode(raw_batch)
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        adapter_id = int(self.adapter_assignments[index].item())
        if 0 <= adapter_id < len(self.task_adapters):
            self.task_adapters[adapter_id].hebbian_update(
                base_f1,
                learning_signal=torch.ones(
                    base_f1.shape[0],
                    device=base_f1.device,
                    dtype=base_f1.dtype,
                ),
                lr=self._adapter_lr(self.config.fast_lr) * lr_multiplier,
            )
            f1_batch = self._apply_adapter(base_f1, adapter_id)
        prototype_f2 = self._row_f2_activate(index, f1_batch).mean(dim=0)
        updated_direction = self.concept_directions[index] + (
            (float(self.config.fast_lr) * lr_multiplier) * prototype_f2
        )
        normalized_direction = normalize(updated_direction.unsqueeze(0)).squeeze(0)
        self.concept_directions[index].copy_(normalized_direction)

        prototype_f1 = f1_batch.mean(dim=0)
        feedback_init = prototype_f1.unsqueeze(-1) * normalized_direction.unsqueeze(0)
        self.feedback_weights[index].copy_(feedback_init)
        self.committed_mask[index].fill_(True)

    @torch.no_grad()
    def recruit(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> tuple[int | None, CCCOutput | None]:
        recruit_index = self._first_uncommitted_index()
        if recruit_index is None:
            return None, None

        raw_batch, squeeze = self._ensure_batch(raw_input)
        if self.current_adapter is not None:
            self.adapter_assignments[recruit_index].fill_(int(self.active_adapter_id))
        f1_batch = self._row_f1_encode(recruit_index, raw_batch)
        self._learn_fast_index(
            recruit_index,
            raw_batch,
            f1_batch,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        return recruit_index, self._single_forward_index(
            recruit_index,
            raw_batch,
            squeeze,
            timestep,
            learning_rate_multiplier=learning_rate_multiplier,
        )

    @torch.no_grad()
    def preview(self, raw_input: torch.Tensor) -> CCCPoolOutput:
        """Run the batched pool read-only without recruitment or learning."""

        raw_batch, squeeze = self._ensure_batch(raw_input)
        state = self._vectorized_state(
            raw_batch,
            timestep=0,
            apply_learning=False,
            update_state=False,
        )
        outputs = self._pack_outputs(state, squeeze=squeeze)
        fired_indices = [index for index, output in enumerate(outputs) if output.fired]
        abstained_indices = [index for index, output in enumerate(outputs) if output.abstained]
        winner_confidences = (
            torch.stack(
                [self._confidence_score(outputs[index].confidence) for index in fired_indices]
            )
            if fired_indices
            else torch.empty(0, dtype=torch.float32, device=self.f1_weights.device)
        )
        lateral_error = self._update_lateral_prediction_state(fired_indices)
        if self.precision_gate is not None:
            self.precision_gate.preview_pool_output(
                fired_indices,
                lateral_error=lateral_error,
                hierarchy_error=self.last_hierarchy_prediction_error,
            )
        return CCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=False,
            recruited_index=None,
            winner_confidences=winner_confidences,
        )

    @torch.no_grad()
    def forward(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        allow_recruit: bool = True,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> CCCPoolOutput:
        raw_batch, squeeze = self._ensure_batch(raw_input)
        self.observe_samples(raw_batch.shape[0])
        state = self._vectorized_state(
            raw_batch,
            timestep=timestep,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        outputs = self._pack_outputs(state, squeeze=squeeze)

        recruited = False
        recruited_index: int | None = None
        if allow_recruit and not state.any_fired:
            recruited_index, recruited_output = self.recruit(
                raw_batch,
                timestep=timestep,
                learning_rate_multiplier=learning_rate_multiplier,
            )
            if recruited_index is not None and recruited_output is not None:
                recruited = True
                outputs[recruited_index] = recruited_output

        fired_indices = [index for index, output in enumerate(outputs) if output.fired]
        abstained_indices = [index for index, output in enumerate(outputs) if output.abstained]
        winner_confidences = (
            torch.stack(
                [self._confidence_score(outputs[index].confidence) for index in fired_indices]
            )
            if fired_indices
            else torch.empty(0, dtype=torch.float32, device=self.f1_weights.device)
        )
        lateral_error = self._update_lateral_prediction_state(fired_indices)
        if self.precision_gate is not None:
            self.precision_gate.observe_pool_output(
                fired_indices,
                lateral_error=lateral_error,
                hierarchy_error=self.last_hierarchy_prediction_error,
            )

        return CCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=recruited,
            recruited_index=recruited_index,
            winner_confidences=winner_confidences,
        )

    @torch.no_grad()
    def get_winners(self, pool_output: CCCPoolOutput, k: int = 5) -> list[int]:
        if not pool_output.fired_indices:
            return []
        top_k = min(k, len(pool_output.fired_indices))
        _, top_indices = torch.topk(pool_output.winner_confidences, k=top_k)
        return [pool_output.fired_indices[index] for index in top_indices.tolist()]

    @torch.no_grad()
    def get_pool_stats(self) -> dict[str, float | int]:
        self._ensure_runtime_state()
        num_committed = int(self.committed_mask.sum().item())
        total_presentations = int(self.total_presentations.sum().item())
        total_fires = int(self.total_fires.sum().item())
        total_confidence = float(
            (
                self.avg_confidence_when_fired * self.total_fires.to(torch.float32)
            ).sum().item()
        )
        mean_confidence = total_confidence / total_fires if total_fires else 0.0
        fire_rate = total_fires / total_presentations if total_presentations else 0.0
        return {
            "num_committed": num_committed,
            "num_uncommitted": int(self.config.max_pool_size) - num_committed,
            "mean_confidence": float(mean_confidence),
            "fire_rate": float(fire_rate),
            "total_concepts": int(self.config.max_pool_size),
            "initial_capacity": self.initial_capacity,
            "max_capacity": self.max_capacity,
            "mean_importance": float(
                self.consolidation.importance[: int(self.config.max_pool_size)].mean().item()
            ),
            "f1_frozen": bool(self.f1_frozen.item()),
            "task_adapters": len(self.task_adapters),
        }

    def get_precision(self) -> float:
        self._ensure_runtime_state()
        if getattr(self, "precision_gate", None) is None:
            return 1.0
        return float(self.precision_gate.current_precision)

    def get_lateral_prediction_error(self) -> float:
        return float(getattr(self, "last_lateral_prediction_error", 0.0))

    @torch.no_grad()
    def load_from_pool(self, pool: CCCPool | "BatchedCCCPool") -> "BatchedCCCPool":
        self._ensure_runtime_state()
        if isinstance(pool, BatchedCCCPool):
            self.f1_weights.copy_(pool.f1_weights)
            self.f1_bias.copy_(pool.f1_bias)
            self.f2_weights.copy_(pool.f2_weights)
            self.feedback_weights.copy_(pool.feedback_weights)
            self.concept_directions.copy_(pool.concept_directions)
            self.committed_mask.copy_(pool.committed_mask)
            self.adapter_assignments.copy_(pool.adapter_assignments)
            self.age.copy_(pool.age)
            self.last_fired.copy_(pool.last_fired)
            self.theta_margin.copy_(pool.theta_margin)
            self.theta_resonance.copy_(pool.theta_resonance)
            self.total_presentations.copy_(pool.total_presentations)
            self.total_fires.copy_(pool.total_fires)
            self.total_abstentions.copy_(pool.total_abstentions)
            self.avg_confidence_when_fired.copy_(pool.avg_confidence_when_fired)
            self.avg_confidence_when_abstained.copy_(pool.avg_confidence_when_abstained)
            self.consolidation.copy_from(pool.consolidation)
            self._copy_shared_state_from(pool)
            self._copy_precision_state_from(pool)
            self._copy_lateral_state_from(pool)
            return self

        for index, ccc in enumerate(pool.cccs):
            self.f1_weights[index].copy_(ccc.f1_layer.weight.detach())
            self.f1_bias[index].copy_(ccc.f1_layer.bias.detach())
            self.f2_weights[index].copy_(ccc.f2_weights.detach())
            self.feedback_weights[index].copy_(ccc.feedback_weights.detach())
            self.concept_directions[index].copy_(ccc.concept_direction.detach())
            self.committed_mask[index].copy_(ccc.is_committed.detach())
            self.adapter_assignments[index].fill_(int(getattr(ccc, "adapter_id", -1)))
            self.age[index].copy_(ccc.age.detach())
            self.last_fired[index].copy_(ccc.last_fired.detach())
            self.theta_margin[index].copy_(ccc.margin_gate.theta_margin.detach())
            self.theta_resonance[index].copy_(ccc.margin_gate.theta_resonance.detach())
            self.total_presentations[index].copy_(ccc.margin_gate.total_presentations.detach())
            self.total_fires[index].copy_(ccc.margin_gate.total_fires.detach())
            self.total_abstentions[index].copy_(ccc.margin_gate.total_abstentions.detach())
            self.avg_confidence_when_fired[index].copy_(
                ccc.margin_gate.avg_confidence_when_fired.detach()
            )
            self.avg_confidence_when_abstained[index].copy_(
                ccc.margin_gate.avg_confidence_when_abstained.detach()
            )
            self.consolidation.importance[index].copy_(ccc.importance.detach())
        if hasattr(pool, "consolidation"):
            self.consolidation.copy_from(pool.consolidation)
        if hasattr(pool, "task_adapters"):
            cloned_adapters = nn.ModuleList()
            for adapter in pool.task_adapters:
                cloned = F1Adapter(pool.config.num_f1_features, adapter_dim=pool.config.f1_adapter_dim)
                cloned.load_state_dict(adapter.state_dict())
                if bool(adapter.is_frozen.item()):
                    cloned.freeze()
                else:
                    cloned.unfreeze()
                cloned_adapters.append(cloned)
            self.task_adapters = cloned_adapters
            self.active_adapter_id = int(getattr(pool, "active_adapter_id", -1))
        if hasattr(pool, "f1_samples_seen"):
            self.f1_samples_seen.copy_(pool.f1_samples_seen)
        if hasattr(pool, "f1_frozen"):
            self.f1_frozen.copy_(pool.f1_frozen)
        self._copy_precision_state_from(pool)
        self._copy_lateral_state_from(pool)
        return self


class PoolSharding(nn.Module):
    """Shard a large CCC pool across sub-pools and route inputs with LSH-guided locality."""

    def __init__(
        self,
        total_size: int,
        shard_size: int = 1000,
        config: CCCConfig | None = None,
        margin_config: MarginGateConfig | None = None,
        *,
        hash_bits: int = 8,
        shards_per_query: int = 1,
    ) -> None:
        super().__init__()
        self.total_size = int(max(1, total_size))
        self.shard_size = int(max(1, shard_size))
        self.num_shards = math.ceil(self.total_size / self.shard_size)
        self.config = replace(config or CCCConfig(), max_pool_size=self.total_size)
        self.margin_config = margin_config or MarginGateConfig()
        self.hash_bits = int(max(1, min(hash_bits, self.config.num_f1_features)))
        self.shards_per_query = int(max(1, min(shards_per_query, self.num_shards)))

        self.shard_offsets = [index * self.shard_size for index in range(self.num_shards)]
        self.shards = nn.ModuleList(
            [
                BatchedCCCPool(
                    replace(
                        self.config,
                        max_pool_size=min(self.shard_size, self.total_size - offset),
                    ),
                    self.margin_config,
                )
                for offset in self.shard_offsets
            ]
        )
        planes = normalize(
            torch.randn(self.hash_bits, self.config.num_f1_features, dtype=torch.float32)
        )
        self.register_buffer("lsh_planes", planes, persistent=False)
        self.register_buffer(
            "_hash_weights",
            (2 ** torch.arange(self.hash_bits, dtype=torch.long)),
            persistent=False,
        )
        self.register_buffer(
            "shard_centroids",
            torch.zeros(self.num_shards, self.config.num_f1_features, dtype=torch.float32),
        )
        self._bucket_map: dict[int, list[int]] = {}
        self.rebuild_routing_index()

    def _hash_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        projections = vectors.to(self.lsh_planes.dtype) @ self.lsh_planes.transpose(0, 1)
        bits = (projections > 0).to(torch.long)
        return bits @ self._hash_weights.to(device=bits.device)

    def _shared_f1_encode(self, raw_batch: torch.Tensor) -> torch.Tensor:
        base_shard = self.shards[0]
        projected = (
            raw_batch.to(base_shard.f1_weights.dtype) @ base_shard.f1_weights[0].transpose(0, 1)
        ) + base_shard.f1_bias[0]
        activated = F.relu(projected)
        top_k = min(base_shard.config.f1_top_k, activated.shape[-1])
        values, indices = torch.topk(activated, k=top_k, dim=-1)
        return torch.zeros_like(activated).scatter(-1, indices, values)

    @staticmethod
    def _prototype_f1(pool: BatchedCCCPool) -> torch.Tensor:
        prototypes = torch.einsum("nfc,nc->nf", pool.feedback_weights, pool.concept_directions)
        return normalize(prototypes)

    @torch.no_grad()
    def rebuild_routing_index(self) -> None:
        self._bucket_map = {}
        for shard_index, shard in enumerate(self.shards):
            if bool(shard.committed_mask.any().item()):
                prototypes = self._prototype_f1(shard)
                active = prototypes[shard.committed_mask]
                centroid = normalize(active.mean(dim=0, keepdim=True)).squeeze(0)
            else:
                centroid = torch.zeros(
                    shard.config.num_f1_features,
                    device=shard.f1_weights.device,
                    dtype=shard.f1_weights.dtype,
                )
            self.shard_centroids[shard_index].copy_(centroid.to(self.shard_centroids.dtype))

        codes = self._hash_vectors(self.shard_centroids)
        for shard_index, code in enumerate(codes.tolist()):
            self._bucket_map.setdefault(int(code), []).append(shard_index)

    def route_to_shards(self, raw_input: torch.Tensor) -> list[int]:
        raw_batch = raw_input.unsqueeze(0) if raw_input.dim() == 1 else raw_input
        f1_query = self._shared_f1_encode(raw_batch).mean(dim=0, keepdim=True)
        query = normalize(f1_query).squeeze(0)
        code = int(self._hash_vectors(query.unsqueeze(0))[0].item())
        candidates = list(self._bucket_map.get(code, []))

        centroid_norm = self.shard_centroids.norm(dim=-1)
        valid = centroid_norm > 0
        if not candidates:
            if bool(valid.any().item()):
                similarities = cosine_similarity(
                    self.shard_centroids[valid],
                    query.unsqueeze(0).expand(int(valid.sum().item()), -1),
                )
                valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
                top_k = min(self.shards_per_query, similarities.numel())
                top_idx = torch.topk(similarities, k=top_k).indices
                return [int(valid_indices[index].item()) for index in top_idx]
            return [0]

        if len(candidates) > self.shards_per_query:
            candidate_centroids = self.shard_centroids[candidates]
            similarities = cosine_similarity(
                candidate_centroids,
                query.unsqueeze(0).expand(candidate_centroids.shape[0], -1),
            )
            top_idx = torch.topk(similarities, k=self.shards_per_query).indices.tolist()
            candidates = [candidates[index] for index in top_idx]
        return candidates

    @torch.no_grad()
    def fast_infer(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        allow_recruit: bool = False,
    ) -> PoolInferenceSummary:
        shard_indices = self.route_to_shards(raw_input)
        searched = set(shard_indices)
        fired_indices: list[int] = []
        winner_confidences: list[torch.Tensor] = []
        recruited = False
        recruited_index: int | None = None

        def evaluate(selected_indices: list[int]) -> None:
            for shard_index in selected_indices:
                summary = self.shards[shard_index].fast_infer(
                    raw_input,
                    timestep=timestep,
                    allow_recruit=False,
                )
                if summary.fired_indices:
                    offset = self.shard_offsets[shard_index]
                    fired_indices.extend(offset + index for index in summary.fired_indices)
                    if summary.winner_confidences.numel():
                        winner_confidences.append(summary.winner_confidences)

        evaluate(shard_indices)
        if not fired_indices:
            evaluate([index for index in range(self.num_shards) if index not in searched])

        if allow_recruit and not fired_indices:
            for shard_index in shard_indices:
                local_index, local_output = self.shards[shard_index].recruit(raw_input, timestep=timestep)
                if local_index is None or local_output is None:
                    continue
                recruited = True
                recruited_index = self.shard_offsets[shard_index] + int(local_index)
                fired_indices = [recruited_index]
                winner_confidences = [local_output.confidence.reshape(-1).mean().to(torch.float32).unsqueeze(0)]
                self.rebuild_routing_index()
                break

        winner_tensor = (
            torch.cat(winner_confidences).to(torch.float32)
            if winner_confidences
            else torch.empty(0, dtype=torch.float32, device=self.shard_centroids.device)
        )
        num_fired = len(fired_indices)
        return PoolInferenceSummary(
            fired_indices=fired_indices,
            winner_confidences=winner_tensor.detach().clone(),
            recruited=recruited,
            recruited_index=recruited_index,
            num_fired=num_fired,
            num_abstained=self.total_size - num_fired,
            sparsity=float(num_fired / max(self.total_size, 1)),
            mean_confidence=float(winner_tensor.mean().item()) if num_fired else 0.0,
        )

    @torch.no_grad()
    def load_from_pool(self, pool: CCCPool | BatchedCCCPool) -> "PoolSharding":
        source = (
            BatchedCCCPool(pool.config, self.margin_config).load_from_pool(pool)
            if isinstance(pool, CCCPool)
            else pool
        )
        if int(source.config.max_pool_size) != self.total_size:
            raise ValueError(
                f"PoolSharding expected pool size {self.total_size}, received {source.config.max_pool_size}."
            )

        for shard, offset in zip(self.shards, self.shard_offsets, strict=False):
            _copy_batched_pool_slice(
                source,
                shard,
                source_start=offset,
                target_start=0,
                length=int(shard.config.max_pool_size),
            )
        self.rebuild_routing_index()
        return self


class AdaptiveCapacity(nn.Module):
    """Grow or prune a batched CCC pool in response to demand and utilization."""

    def __init__(
        self,
        initial_size: int = 1000,
        max_size: int = 10000,
        config: CCCConfig | None = None,
        margin_config: MarginGateConfig | None = None,
        *,
        growth_factor: float = 1.5,
        abstention_window: int = 32,
        abstention_threshold: float = 0.25,
        target_utilization: tuple[float, float] = (0.6, 0.8),
    ) -> None:
        super().__init__()
        self.initial_size = int(max(1, initial_size))
        self.max_size = int(max(self.initial_size, max_size))
        self.base_config = replace(config or CCCConfig(), max_pool_size=self.initial_size)
        self.margin_config = margin_config or MarginGateConfig()
        self.growth_factor = float(max(1.1, growth_factor))
        self.abstention_window = int(max(4, abstention_window))
        self.abstention_threshold = float(max(0.0, min(1.0, abstention_threshold)))
        self.target_utilization = target_utilization
        self.pool = BatchedCCCPool(self.base_config, self.margin_config)
        self._recent_abstentions: list[float] = []

    @property
    def current_size(self) -> int:
        return int(self.pool.config.max_pool_size)

    def utilization(self) -> float:
        return float(self.pool.committed_mask.float().mean().item())

    @torch.no_grad()
    def _resize(self, new_size: int) -> None:
        new_size = int(max(self.current_size, min(self.max_size, new_size)))
        if new_size == self.current_size:
            return

        resized = BatchedCCCPool(
            replace(self.base_config, max_pool_size=new_size),
            self.margin_config,
        )
        _copy_batched_pool_slice(self.pool, resized, length=self.current_size)
        self.pool = resized

    @torch.no_grad()
    def grow(self, new_size: int | None = None) -> int:
        if self.current_size >= self.max_size:
            return self.current_size
        target = new_size or int(math.ceil(self.current_size * self.growth_factor))
        target = max(self.current_size + 1, min(self.max_size, target))
        self._resize(target)
        return self.current_size

    def observe_abstention(self, abstained: bool) -> None:
        self._recent_abstentions.append(1.0 if abstained else 0.0)
        if len(self._recent_abstentions) > self.abstention_window:
            self._recent_abstentions.pop(0)
        if len(self._recent_abstentions) < self.abstention_window:
            return
        if sum(self._recent_abstentions) / len(self._recent_abstentions) > self.abstention_threshold:
            self.grow()

    @torch.no_grad()
    def fast_infer(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        allow_recruit: bool = True,
    ) -> PoolInferenceSummary:
        summary = self.pool.fast_infer(raw_input, timestep=timestep, allow_recruit=allow_recruit)
        self.observe_abstention(summary.recruited or summary.num_fired == 0)
        return summary

    @torch.no_grad()
    def prune_dead_cccs(
        self,
        *,
        min_presentations: int = 64,
        max_fire_count: int = 0,
    ) -> list[int]:
        committed = self.pool.committed_mask
        stale = (
            committed
            & (self.pool.total_presentations >= int(min_presentations))
            & (self.pool.total_fires <= int(max_fire_count))
        )
        dead_indices = stale.nonzero(as_tuple=False).squeeze(-1).tolist()
        for index in dead_indices:
            self.pool.feedback_weights[index].zero_()
            self.pool.concept_directions[index].zero_()
            self.pool.committed_mask[index].fill_(False)
            self.pool.adapter_assignments[index].fill_(-1)
            self.pool.age[index].zero_()
            self.pool.last_fired[index].fill_(-1)
            self.pool.total_presentations[index].zero_()
            self.pool.total_fires[index].zero_()
            self.pool.total_abstentions[index].zero_()
            self.pool.avg_confidence_when_fired[index].zero_()
            self.pool.avg_confidence_when_abstained[index].zero_()
        return [int(index) for index in dead_indices]

    def get_stats(self) -> dict[str, float | int]:
        return {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "utilization": self.utilization(),
            "recent_abstention_rate": (
                float(sum(self._recent_abstentions) / len(self._recent_abstentions))
                if self._recent_abstentions
                else 0.0
            ),
        }


class OptimizedSDM(SparseDistributedMemory):
    """SDM with batched Hamming distance, LSH routing, and chunked retrieval."""

    def __init__(self, config: SDMConfig, *, chunk_size: int = 256, hash_bits: int = 8):
        super().__init__(config)
        self.chunk_size = int(max(1, chunk_size))
        self.hash_bits = int(max(1, min(hash_bits, config.address_dim)))

        planes = normalize(torch.randn(self.hash_bits, config.address_dim, dtype=torch.float32))
        self.register_buffer("lsh_planes", planes, persistent=False)
        self.register_buffer(
            "_hash_weights",
            (2 ** torch.arange(self.hash_bits, dtype=torch.long)),
            persistent=False,
        )
        self._bucket_map: dict[int, torch.Tensor] = {}
        self.sparse_data_matrix: torch.Tensor | None = None
        self.rebuild_index()

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self.rebuild_index()
        return self

    def _hash_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        projections = vectors.to(self.lsh_planes.dtype) @ self.lsh_planes.transpose(0, 1)
        bits = (projections > 0).to(torch.long)
        return bits @ self._hash_weights.to(device=bits.device)

    @torch.no_grad()
    def rebuild_index(self) -> None:
        codes = self._hash_vectors(self.hard_locations)
        self._bucket_map = {}
        for code in torch.unique(codes).tolist():
            mask = codes == int(code)
            self._bucket_map[int(code)] = mask.nonzero(as_tuple=False).squeeze(-1)
        self._refresh_sparse_storage()

    def _refresh_sparse_storage(self) -> None:
        occupied_ratio = float((self.data_matrix.abs().sum(dim=-1) > 0).float().mean().item())
        if occupied_ratio <= 0.1:
            indices = self.data_matrix.nonzero(as_tuple=False).transpose(0, 1)
            values = self.data_matrix[tuple(indices)] if indices.numel() else torch.empty(
                0,
                device=self.data_matrix.device,
                dtype=self.data_matrix.dtype,
            )
            prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
            torch.sparse.check_sparse_tensor_invariants.disable()
            try:
                self.sparse_data_matrix = torch.sparse_coo_tensor(
                    indices,
                    values,
                    self.data_matrix.shape,
                    device=self.data_matrix.device,
                    dtype=self.data_matrix.dtype,
                ).coalesce()
            finally:
                if prev_checks_enabled:
                    torch.sparse.check_sparse_tensor_invariants.enable()
            return
        self.sparse_data_matrix = None

    def _candidate_indices(self, address_row: torch.Tensor) -> torch.Tensor:
        code = int(self._hash_vectors(address_row.unsqueeze(0))[0].item())
        bucket = self._bucket_map.get(code)
        if bucket is None or bucket.numel() == 0:
            return torch.arange(self.num_hard_locations, device=self.hard_locations.device)
        return bucket.to(device=self.hard_locations.device)

    def _distances_for_candidates(
        self,
        address_row: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for chunk in candidate_indices.split(self.chunk_size):
            hard_chunk = self.hard_locations.index_select(0, chunk)
            distances = (address_row.unsqueeze(0) != hard_chunk).sum(dim=-1)
            chunks.append(distances)
        if not chunks:
            return torch.empty(0, device=address_row.device, dtype=torch.long)
        return torch.cat(chunks, dim=0)

    def _activated_mask_from_binary(self, address: torch.Tensor) -> torch.Tensor:
        full_mask = torch.zeros(
            address.shape[0],
            self.num_hard_locations,
            device=self.hard_locations.device,
            dtype=torch.bool,
        )
        all_indices = torch.arange(self.num_hard_locations, device=self.hard_locations.device)

        for batch_index, address_row in enumerate(address.to(self.hard_locations.dtype)):
            candidate_indices = self._candidate_indices(address_row)
            candidate_distances = self._distances_for_candidates(address_row, candidate_indices)
            active = candidate_distances <= int(self.hamming_radius)

            if not bool(active.any().item()) and candidate_indices.numel() != self.num_hard_locations:
                candidate_indices = all_indices
                candidate_distances = self._distances_for_candidates(address_row, candidate_indices)
                active = candidate_distances <= int(self.hamming_radius)

            if bool(active.any().item()):
                full_mask[batch_index, candidate_indices[active]] = True
        return full_mask

    @torch.no_grad()
    def _write_impl(
        self,
        address: torch.Tensor,
        data: torch.Tensor,
        apply_decay: bool = True,
    ) -> None:
        address = self.compute_address(address)
        address, _ = self._ensure_2d(address)
        data, _ = self._ensure_2d(data.to(self.data_matrix.dtype))

        if address.shape[0] != data.shape[0]:
            raise ValueError(
                "Address and data batch sizes must match: "
                f"{address.shape[0]} != {data.shape[0]}"
            )
        if data.shape[-1] != self.data_dim:
            raise ValueError(f"Expected data_dim={self.data_dim}, received {data.shape[-1]}.")

        activated = self._activated_mask_from_binary(address).to(self.data_matrix.dtype)
        self.data_matrix.add_(activated.transpose(0, 1) @ data)
        self.activation_counts.add_(activated.sum(dim=0))

        if apply_decay:
            self.data_matrix.mul_(self.decay_rate)
        self._refresh_sparse_storage()

    def read(self, address: torch.Tensor) -> torch.Tensor:
        address = self.compute_address(address)
        address, squeeze = self._ensure_2d(address)
        activated = self._activated_mask_from_binary(address)

        retrieved_rows: list[torch.Tensor] = []
        for row_mask in activated:
            indices = row_mask.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                retrieved_rows.append(torch.zeros(self.data_dim, device=self.data_matrix.device))
                continue

            total = torch.zeros(self.data_dim, device=self.data_matrix.device, dtype=self.data_matrix.dtype)
            for chunk in indices.split(self.chunk_size):
                total.add_(self.data_matrix.index_select(0, chunk).sum(dim=0))
            count = self.activation_counts.index_select(0, indices).sum().clamp_min(1.0)
            retrieved_rows.append(total / count)

        retrieved = torch.stack(retrieved_rows, dim=0)
        return retrieved.squeeze(0) if squeeze else retrieved

    @torch.no_grad()
    def associate(
        self,
        address_a: torch.Tensor,
        address_b: torch.Tensor,
        data_a: torch.Tensor,
        data_b: torch.Tensor,
        temporal_order: bool = True,
    ) -> None:
        super().associate(address_a, address_b, data_a, data_b, temporal_order=temporal_order)
        self._refresh_sparse_storage()

    @torch.no_grad()
    def copy_state_from(self, other: SparseDistributedMemory) -> "OptimizedSDM":
        self.hard_locations.copy_(other.hard_locations.detach())
        self.data_matrix.copy_(other.data_matrix.detach())
        self.activation_counts.copy_(other.activation_counts.detach())
        self.address_projection = other.address_projection.detach().clone()
        self._projection_input_dim = other._projection_input_dim
        self.rebuild_index()
        return self


class HierarchicalSDM(nn.Module):
    """Hierarchical SDM that routes addresses into progressively finer regions."""

    def __init__(self, config: SDMConfig, num_levels: int = 3):
        super().__init__()
        self.config = config
        self.num_levels = int(max(1, num_levels))
        self.bits_per_level = max(1, min(8, config.address_dim // self.num_levels))
        self.route_counts: dict[tuple[int, ...], int] = {}
        self._projection_input_dim: int | None = None
        self.register_buffer("address_projection", torch.empty(0, dtype=torch.float32))
        self.regions = nn.ModuleDict()

    def _ensure_2d(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _get_or_create_projection(
        self,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._projection_input_dim is None:
            projection = torch.randn(
                input_dim,
                self.config.address_dim,
                device=device,
                dtype=dtype,
            ) / math.sqrt(max(input_dim, 1))
            self.address_projection = projection
            self._projection_input_dim = input_dim
        elif self._projection_input_dim != input_dim:
            raise ValueError(
                "HierarchicalSDM projection was initialized for "
                f"{self._projection_input_dim} input dims, but received {input_dim}."
            )
        return self.address_projection.to(device=device, dtype=dtype)

    def compute_address(self, concept_direction: torch.Tensor) -> torch.Tensor:
        concept_direction, squeeze = self._ensure_2d(concept_direction.to(torch.float32))
        if concept_direction.shape[-1] != self.config.address_dim:
            projection = self._get_or_create_projection(
                input_dim=concept_direction.shape[-1],
                device=concept_direction.device,
                dtype=concept_direction.dtype,
            )
            concept_direction = concept_direction @ projection
        address = (concept_direction > 0).to(torch.float32)
        return address.squeeze(0) if squeeze else address

    def _path_from_row(self, address_row: torch.Tensor) -> tuple[int, ...]:
        bits = address_row.to(torch.long)
        path: list[int] = []
        for level in range(self.num_levels):
            start = level * self.bits_per_level
            end = min(start + self.bits_per_level, bits.numel())
            if start >= bits.numel():
                path.append(0)
                continue
            chunk = bits[start:end]
            weights = 2 ** torch.arange(chunk.numel() - 1, -1, -1, device=chunk.device)
            path.append(int((chunk * weights).sum().item()))
        return tuple(path)

    def route_to_region(self, concept_direction: torch.Tensor) -> tuple[int, ...] | list[tuple[int, ...]]:
        address = self.compute_address(concept_direction)
        address, squeeze = self._ensure_2d(address)
        routes = [self._path_from_row(row) for row in address]
        return routes[0] if squeeze else routes

    def _route_key(self, path: tuple[int, ...]) -> str:
        return "::".join(str(part) for part in path)

    def _leaf_config(self) -> SDMConfig:
        leaf_locations = max(
            32,
            min(
                self.config.num_hard_locations,
                self.config.num_hard_locations // max(1, 2 ** (self.num_levels - 1)),
            ),
        )
        return replace(self.config, num_hard_locations=leaf_locations)

    def _get_or_create_region(self, path: tuple[int, ...]) -> OptimizedSDM:
        key = self._route_key(path)
        if key not in self.regions:
            self.regions[key] = OptimizedSDM(self._leaf_config())
        return self.regions[key]  # type: ignore[return-value]

    def _bind_address_to_leaf(self, leaf: OptimizedSDM, address_row: torch.Tensor) -> None:
        exact = (leaf.hard_locations == address_row.unsqueeze(0)).all(dim=-1)
        if bool(exact.any().item()):
            return

        empty = (leaf.data_matrix.abs().sum(dim=-1) == 0)
        if bool(empty.any().item()):
            slot = int(torch.argmax(empty.to(torch.int64)).item())
        else:
            slot = 0
        leaf.hard_locations[slot].copy_(address_row.to(leaf.hard_locations.dtype))
        leaf.rebuild_index()

    @torch.no_grad()
    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        address = self.compute_address(address)
        address, _ = self._ensure_2d(address)
        data, _ = self._ensure_2d(data.to(torch.float32))
        for address_row, data_row in zip(address, data):
            path = self._path_from_row(address_row)
            self.route_counts[path] = self.route_counts.get(path, 0) + 1
            leaf = self._get_or_create_region(path)
            self._bind_address_to_leaf(leaf, address_row)
            leaf.write(address_row, data_row)

    def _nearest_existing_path(self, path: tuple[int, ...]) -> tuple[int, ...] | None:
        if not self.route_counts:
            return None
        return max(
            self.route_counts,
            key=lambda candidate: (
                sum(int(a == b) for a, b in zip(candidate, path, strict=False)),
                self.route_counts[candidate],
            ),
        )

    def read(self, address: torch.Tensor) -> torch.Tensor:
        address = self.compute_address(address)
        address, squeeze = self._ensure_2d(address)
        rows: list[torch.Tensor] = []
        for address_row in address:
            path = self._path_from_row(address_row)
            key = self._route_key(path)
            if key not in self.regions:
                fallback = self._nearest_existing_path(path)
                if fallback is None:
                    rows.append(torch.zeros(self.config.data_dim, device=address.device))
                    continue
                key = self._route_key(fallback)
            rows.append(self.regions[key].read(address_row))
        result = torch.stack(rows, dim=0)
        return result.squeeze(0) if squeeze else result

    @torch.no_grad()
    def associate(
        self,
        address_a: torch.Tensor,
        address_b: torch.Tensor,
        data_a: torch.Tensor,
        data_b: torch.Tensor,
        temporal_order: bool = True,
    ) -> None:
        forward_scale = 2.0 if temporal_order else 1.0
        self.write(address_a, data_b * forward_scale)
        self.write(address_b, data_a)

    def retrieve_associates(self, cue_address: torch.Tensor) -> torch.Tensor:
        return self.read(cue_address)

    def get_stats(self) -> dict[str, float | int]:
        return {
            "num_regions": len(self.regions),
            "num_levels": self.num_levels,
            "mean_region_load": float(sum(self.route_counts.values()) / max(1, len(self.route_counts))),
        }


class MemoryEfficientSDM(nn.Module):
    """SDM optimized for sparse occupancy with chunked, low-precision storage."""

    def __init__(
        self,
        config: SDMConfig,
        *,
        chunk_size: int = 256,
        hash_bits: int = 8,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.config = config
        self.address_dim = config.address_dim
        self.hamming_radius = config.hamming_radius
        self.num_hard_locations = config.num_hard_locations
        self.data_dim = config.data_dim
        self.decay_rate = config.decay_rate
        self.chunk_size = int(max(1, chunk_size))
        self.hash_bits = int(max(1, min(hash_bits, config.address_dim)))
        self.storage_dtype = storage_dtype

        hard_locations = torch.randint(
            0,
            2,
            (self.num_hard_locations, self.address_dim),
            dtype=torch.float32,
        )
        self.register_buffer("hard_locations", hard_locations)
        self.register_buffer("address_projection", torch.empty(0, dtype=torch.float32))
        self._projection_input_dim: int | None = None
        planes = normalize(torch.randn(self.hash_bits, config.address_dim, dtype=torch.float32))
        self.register_buffer("lsh_planes", planes, persistent=False)
        self.register_buffer(
            "_hash_weights",
            (2 ** torch.arange(self.hash_bits, dtype=torch.long)),
            persistent=False,
        )
        self._bucket_map: dict[int, torch.Tensor] = {}
        self._data_chunks: dict[int, torch.Tensor] = {}
        self._count_chunks: dict[int, torch.Tensor] = {}
        self.rebuild_index()

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self._data_chunks = {key: fn(value) for key, value in self._data_chunks.items()}
        self._count_chunks = {key: fn(value) for key, value in self._count_chunks.items()}
        self.rebuild_index()
        return self

    def storage_tensors(self) -> list[torch.Tensor]:
        return list(self._data_chunks.values()) + list(self._count_chunks.values())

    def _ensure_2d(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _get_or_create_projection(
        self,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._projection_input_dim is None:
            projection = torch.randn(input_dim, self.address_dim, device=device, dtype=dtype)
            projection = projection / math.sqrt(max(input_dim, 1))
            self.address_projection = projection
            self._projection_input_dim = input_dim
        elif self._projection_input_dim != input_dim:
            raise ValueError(
                "MemoryEfficientSDM projection was initialized for "
                f"{self._projection_input_dim} input dims, but received {input_dim}."
            )
        return self.address_projection.to(device=device, dtype=dtype)

    def compute_address(self, concept_direction: torch.Tensor) -> torch.Tensor:
        concept_direction, squeeze = self._ensure_2d(concept_direction.to(torch.float32))
        if concept_direction.shape[-1] != self.address_dim:
            projection = self._get_or_create_projection(
                input_dim=concept_direction.shape[-1],
                device=concept_direction.device,
                dtype=concept_direction.dtype,
            )
            concept_direction = concept_direction @ projection
        address = (concept_direction > 0).to(torch.float32)
        return address.squeeze(0) if squeeze else address

    def _hash_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        projections = vectors.to(self.lsh_planes.dtype) @ self.lsh_planes.transpose(0, 1)
        bits = (projections > 0).to(torch.long)
        return bits @ self._hash_weights.to(device=bits.device)

    @torch.no_grad()
    def rebuild_index(self) -> None:
        self._bucket_map = {}
        codes = self._hash_vectors(self.hard_locations)
        for code in torch.unique(codes).tolist():
            mask = codes == int(code)
            self._bucket_map[int(code)] = mask.nonzero(as_tuple=False).squeeze(-1)

    def _candidate_indices(self, address_row: torch.Tensor) -> torch.Tensor:
        code = int(self._hash_vectors(address_row.unsqueeze(0))[0].item())
        bucket = self._bucket_map.get(code)
        if bucket is None or bucket.numel() == 0:
            return torch.arange(self.num_hard_locations, device=self.hard_locations.device)
        return bucket.to(device=self.hard_locations.device)

    def _active_indices(self, address_row: torch.Tensor) -> torch.Tensor:
        candidates = self._candidate_indices(address_row)
        if candidates.numel() == 0:
            return candidates
        hard = self.hard_locations.index_select(0, candidates)
        distances = (hard != address_row.unsqueeze(0)).sum(dim=-1)
        active = candidates[distances <= int(self.hamming_radius)]
        if active.numel() > 0 or candidates.numel() == self.num_hard_locations:
            return active
        full = torch.arange(self.num_hard_locations, device=self.hard_locations.device)
        full_distances = (self.hard_locations != address_row.unsqueeze(0)).sum(dim=-1)
        return full[full_distances <= int(self.hamming_radius)]

    def _chunk_info(self, indices: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        if indices.numel() == 0:
            return []
        chunk_ids = torch.div(indices, self.chunk_size, rounding_mode="floor")
        groups: list[tuple[int, torch.Tensor]] = []
        for chunk_id in torch.unique(chunk_ids).tolist():
            mask = chunk_ids == int(chunk_id)
            groups.append((int(chunk_id), indices[mask] - (int(chunk_id) * self.chunk_size)))
        return groups

    def _get_or_create_chunk(self, chunk_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_chunk = self._data_chunks.get(chunk_id)
        count_chunk = self._count_chunks.get(chunk_id)
        chunk_len = min(self.chunk_size, self.num_hard_locations - (chunk_id * self.chunk_size))
        if data_chunk is None:
            data_chunk = torch.zeros(
                chunk_len,
                self.data_dim,
                device=self.hard_locations.device,
                dtype=self.storage_dtype,
            )
            self._data_chunks[chunk_id] = data_chunk
        if count_chunk is None:
            count_chunk = torch.zeros(
                chunk_len,
                device=self.hard_locations.device,
                dtype=self.storage_dtype,
            )
            self._count_chunks[chunk_id] = count_chunk
        return data_chunk, count_chunk

    @torch.no_grad()
    def _write_impl(self, address: torch.Tensor, data: torch.Tensor, apply_decay: bool = True) -> None:
        address = self.compute_address(address)
        address, _ = self._ensure_2d(address)
        data, _ = self._ensure_2d(data.to(torch.float32))

        if address.shape[0] != data.shape[0]:
            raise ValueError(
                "Address and data batch sizes must match: "
                f"{address.shape[0]} != {data.shape[0]}"
            )
        if data.shape[-1] != self.data_dim:
            raise ValueError(f"Expected data_dim={self.data_dim}, received {data.shape[-1]}.")

        for address_row, data_row in zip(address, data):
            active_indices = self._active_indices(address_row)
            for chunk_id, local_indices in self._chunk_info(active_indices):
                data_chunk, count_chunk = self._get_or_create_chunk(chunk_id)
                data_chunk[local_indices] += data_row.to(self.storage_dtype)
                count_chunk[local_indices] += 1

        if apply_decay and self._data_chunks:
            for chunk_id, data_chunk in self._data_chunks.items():
                self._data_chunks[chunk_id] = data_chunk.mul(self.decay_rate)

    @torch.no_grad()
    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        self._write_impl(address, data, apply_decay=True)

    def read(self, address: torch.Tensor) -> torch.Tensor:
        address = self.compute_address(address)
        address, squeeze = self._ensure_2d(address)
        rows: list[torch.Tensor] = []

        for address_row in address:
            active_indices = self._active_indices(address_row)
            total = torch.zeros(self.data_dim, device=self.hard_locations.device, dtype=torch.float32)
            count = torch.tensor(0.0, device=self.hard_locations.device)
            for chunk_id, local_indices in self._chunk_info(active_indices):
                data_chunk = self._data_chunks.get(chunk_id)
                count_chunk = self._count_chunks.get(chunk_id)
                if data_chunk is None or count_chunk is None:
                    continue
                total += data_chunk[local_indices].to(torch.float32).sum(dim=0)
                count += count_chunk[local_indices].to(torch.float32).sum()
            rows.append(total / count.clamp_min(1.0))

        result = torch.stack(rows, dim=0)
        return result.squeeze(0) if squeeze else result

    @torch.no_grad()
    def associate(
        self,
        address_a: torch.Tensor,
        address_b: torch.Tensor,
        data_a: torch.Tensor,
        data_b: torch.Tensor,
        temporal_order: bool = True,
    ) -> None:
        forward_scale = 2.0 if temporal_order else 1.0
        self._write_impl(address_a, data_b * forward_scale, apply_decay=False)
        self._write_impl(address_b, data_a, apply_decay=False)
        if self._data_chunks:
            for chunk_id, data_chunk in self._data_chunks.items():
                self._data_chunks[chunk_id] = data_chunk.mul(self.decay_rate)

    def retrieve_associates(self, cue_address: torch.Tensor) -> torch.Tensor:
        return self.read(cue_address)

    def estimated_memory_mb(self) -> float:
        return estimate_module_memory_mb(self)

    def get_stats(self) -> dict[str, float | int]:
        occupied = sum(int((chunk.abs().sum(dim=-1) > 0).sum().item()) for chunk in self._data_chunks.values())
        capacity_used = occupied / float(self.num_hard_locations)
        return {
            "num_chunks": len(self._data_chunks),
            "num_stored": occupied,
            "capacity_used": float(capacity_used),
            "sparsity": float(1.0 - capacity_used),
            "estimated_memory_mb": self.estimated_memory_mb(),
        }


class ScalingProfiler:
    """Utility for lightweight empirical profiling of scaling-oriented components."""

    def __init__(self) -> None:
        self.default_pool_scales = [100, 500, 1000, 5000, 10000]
        self.default_sdm_scales = [100, 1000, 5000, 10000]

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return _tensor_bytes(tensor)

    def _estimate_module_memory_mb(self, module: nn.Module) -> float:
        return estimate_module_memory_mb(module)

    def _infer_scaling_order(self, scale_points: list[int], times_ms: list[float]) -> str:
        if len(scale_points) < 2 or len(times_ms) < 2:
            return "O(1)"

        def stable(normalized: list[float]) -> float:
            finite = [value for value in normalized if math.isfinite(value) and value > 0]
            if len(finite) < 2:
                return float("inf")
            return max(finite) / max(min(finite), 1e-9)

        linear = stable([t / max(n, 1) for n, t in zip(scale_points, times_ms, strict=False)])
        nlogn = stable(
            [
                t / max(n * math.log2(max(n, 2)), 1e-9)
                for n, t in zip(scale_points, times_ms, strict=False)
            ]
        )
        quadratic = stable([t / max(n * n, 1) for n, t in zip(scale_points, times_ms, strict=False)])

        scores = {
            "O(n)": linear,
            "O(n log n)": nlogn,
            "O(n²)": quadratic,
        }
        return min(scores, key=scores.get)

    def _margin_config_from_pool(self, pool: CCCPool | BatchedCCCPool) -> MarginGateConfig:
        if isinstance(pool, BatchedCCCPool):
            return pool.margin_config
        gate = pool.cccs[0].margin_gate
        return MarginGateConfig(
            theta_margin=float(gate.theta_margin.item()),
            theta_margin_lr=float(gate.theta_margin_lr),
            theta_resonance=float(gate.theta_resonance.item()),
        )

    def _prime_pool(self, pool: CCCPool | BatchedCCCPool) -> None:
        num_commit = max(1, min(32, pool.config.max_pool_size // 8 or 1))
        if isinstance(pool, BatchedCCCPool):
            with torch.no_grad():
                pool.committed_mask[:num_commit] = True
                directions = normalize(
                    torch.randn(num_commit, pool.config.concept_dim, dtype=pool.concept_directions.dtype)
                )
                pool.concept_directions[:num_commit].copy_(directions)
                proto_f1 = torch.randn(
                    num_commit,
                    pool.config.num_f1_features,
                    dtype=pool.feedback_weights.dtype,
                )
                pool.feedback_weights[:num_commit].copy_(
                    proto_f1.unsqueeze(-1) * directions.unsqueeze(1)
                )
            return

        with torch.no_grad():
            for ccc in pool.cccs[:num_commit]:
                ccc.is_committed.fill_(True)
                direction = normalize(
                    torch.randn(1, pool.config.concept_dim, dtype=ccc.concept_direction.dtype)
                ).squeeze(0)
                ccc.concept_direction.copy_(direction)
                prototype_f1 = torch.randn(pool.config.num_f1_features, dtype=ccc.feedback_weights.dtype)
                ccc.feedback_weights.copy_(prototype_f1.unsqueeze(-1) * direction.unsqueeze(0))

    def profile_ccc_pool(
        self,
        pool: CCCPool | BatchedCCCPool,
        input_dim: int,
        num_inputs: int = 100,
        scale_points: list[int] | None = None,
    ) -> ProfileResult:
        scale_points = scale_points or list(self.default_pool_scales)
        times_ms: list[float] = []
        memory_mb: list[float] = []
        margin_config = self._margin_config_from_pool(pool)

        for scale in scale_points:
            cfg = replace(pool.config, input_dim=input_dim, max_pool_size=int(scale))
            profiled_pool: CCCPool | BatchedCCCPool
            if isinstance(pool, BatchedCCCPool):
                profiled_pool = BatchedCCCPool(cfg, margin_config)
            else:
                profiled_pool = CCCPool(cfg, margin_config)
            self._prime_pool(profiled_pool)

            samples = torch.randn(num_inputs, input_dim, dtype=torch.float32)
            with torch.no_grad():
                profiled_pool(samples[0], timestep=0)
                start = time.perf_counter()
                for idx in range(num_inputs):
                    profiled_pool(samples[idx], timestep=idx)
                elapsed = time.perf_counter() - start

            times_ms.append((elapsed * 1000.0) / max(num_inputs, 1))
            memory_mb.append(self._estimate_module_memory_mb(profiled_pool))

        return ProfileResult(
            operation="ccc_pool.forward",
            scale_points=scale_points,
            times_ms=times_ms,
            memory_mb=memory_mb,
            scaling_order=self._infer_scaling_order(scale_points, times_ms),
        )

    def profile_sdm(
        self,
        sdm: SparseDistributedMemory | OptimizedSDM,
        num_queries: int = 100,
        scale_points: list[int] | None = None,
    ) -> ProfileResult:
        scale_points = scale_points or list(self.default_sdm_scales)
        times_ms: list[float] = []
        memory_mb: list[float] = []

        for scale in scale_points:
            cfg = replace(sdm.config, num_hard_locations=int(scale))
            profiled_sdm = OptimizedSDM(cfg) if isinstance(sdm, OptimizedSDM) else SparseDistributedMemory(cfg)

            addresses = torch.randn(num_queries, cfg.address_dim, dtype=torch.float32)
            payloads = torch.randn(num_queries, cfg.data_dim, dtype=torch.float32)

            with torch.no_grad():
                profiled_sdm.write(addresses[0], payloads[0])
                start = time.perf_counter()
                for idx in range(num_queries):
                    profiled_sdm.write(addresses[idx], payloads[idx])
                    profiled_sdm.read(addresses[idx])
                elapsed = time.perf_counter() - start

            times_ms.append((elapsed * 1000.0) / max(num_queries, 1))
            memory_mb.append(self._estimate_module_memory_mb(profiled_sdm))

        return ProfileResult(
            operation="sdm.read_write",
            scale_points=scale_points,
            times_ms=times_ms,
            memory_mb=memory_mb,
            scaling_order=self._infer_scaling_order(scale_points, times_ms),
        )

    def _prepare_comparable_pools(
        self,
        input_dim: int,
        pool_size: int,
    ) -> tuple[CCCPool, BatchedCCCPool]:
        config = CCCConfig(
            input_dim=input_dim,
            concept_dim=max(4, min(32, input_dim)),
            num_f1_features=max(4, min(32, input_dim)),
            f1_top_k=max(1, min(8, input_dim)),
            fast_lr=1.0,
            slow_lr=0.01,
            feedback_lr=0.01,
            max_pool_size=pool_size,
        )
        margin = MarginGateConfig(theta_margin=0.2, theta_margin_lr=0.0, theta_resonance=1.1)
        original = CCCPool(config, margin)
        optimized = BatchedCCCPool(config, margin).load_from_pool(original)

        torch.manual_seed(pool_size)
        num_commit = max(1, min(16, pool_size // 4 or 1))
        with torch.no_grad():
            for ccc in original.cccs[:num_commit]:
                ccc.is_committed.fill_(True)
                direction = normalize(torch.randn(1, config.concept_dim)).squeeze(0)
                ccc.concept_direction.copy_(direction)
                proto_f1 = torch.randn(config.num_f1_features)
                ccc.feedback_weights.copy_(proto_f1.unsqueeze(-1) * direction.unsqueeze(0))
        optimized.load_from_pool(original)
        return original, optimized

    @staticmethod
    def _pool_outputs_match(original: CCCPoolOutput, optimized: CCCPoolOutput) -> bool:
        if original.recruited != optimized.recruited:
            return False
        if original.recruited_index != optimized.recruited_index:
            return False
        if original.fired_indices != optimized.fired_indices:
            return False
        if original.abstained_indices != optimized.abstained_indices:
            return False
        if not torch.allclose(original.winner_confidences, optimized.winner_confidences, atol=1e-5):
            return False
        for left, right in zip(original.outputs, optimized.outputs, strict=False):
            if left.fired != right.fired or left.abstained != right.abstained:
                return False
            if not torch.allclose(left.confidence, right.confidence, atol=1e-5):
                return False
            if not torch.allclose(left.f1_output, right.f1_output, atol=1e-5):
                return False
            if not torch.allclose(left.f2_activation, right.f2_activation, atol=1e-5):
                return False
        return True

    def compare_original_vs_optimized(
        self,
        input_dim: int,
        pool_sizes: list[int],
    ) -> ComparisonResult:
        original_times: list[float] = []
        optimized_times: list[float] = []
        speedups: list[float] = []
        correctness_verified = True

        for pool_size in pool_sizes:
            original, optimized = self._prepare_comparable_pools(input_dim, pool_size)
            sample = torch.randn(input_dim, dtype=torch.float32)
            with torch.no_grad():
                correctness_verified = correctness_verified and self._pool_outputs_match(
                    original(sample, timestep=0),
                    optimized(sample, timestep=0),
                )

            eval_samples = torch.randn(16, input_dim, dtype=torch.float32)
            with torch.no_grad():
                start = time.perf_counter()
                for idx in range(eval_samples.shape[0]):
                    original(eval_samples[idx], timestep=idx + 1)
                original_elapsed = time.perf_counter() - start

                start = time.perf_counter()
                for idx in range(eval_samples.shape[0]):
                    optimized(eval_samples[idx], timestep=idx + 1)
                optimized_elapsed = time.perf_counter() - start

            original_ms = (original_elapsed * 1000.0) / eval_samples.shape[0]
            optimized_ms = (optimized_elapsed * 1000.0) / eval_samples.shape[0]
            original_times.append(original_ms)
            optimized_times.append(optimized_ms)
            speedups.append(original_ms / max(optimized_ms, 1e-9))

        return ComparisonResult(
            original_times=original_times,
            optimized_times=optimized_times,
            speedup_factors=speedups,
            correctness_verified=correctness_verified,
        )


class ScaledBioARN(BioARNCore):
    """Drop-in BioARNCore replacement that swaps in optimized scaling components."""

    def __init__(self, config: BioARNConfig, use_optimized: bool = True):
        super().__init__(config)
        self.use_optimized = bool(use_optimized)
        if not self.use_optimized:
            return

        self.ccc_pool = BatchedCCCPool(config.ccc, config.margin_gate).load_from_pool(self.ccc_pool)
        optimized_sdm = OptimizedSDM(config.sdm).copy_state_from(self.fabric.sdm)
        self.fabric = AssociativeFabric(config.sdm, config.ccc)
        self.fabric.sdm = optimized_sdm
        self.fabric.temporal_associator = TemporalAssociator(self.fabric.sdm, config.sdm)

    def _run_pool(
        self,
        raw_input: torch.Tensor,
        *,
        allow_recruit: bool,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> CCCPoolOutput:
        if not self.use_optimized:
            return super()._run_pool(
                raw_input,
                allow_recruit=allow_recruit,
                learning_rate_multiplier=learning_rate_multiplier,
            )
        return self.ccc_pool(
            raw_input,
            timestep=self.timestep,
            allow_recruit=allow_recruit,
            learning_rate_multiplier=learning_rate_multiplier,
        )

    def _active_cccs(self, pool_output: CCCPoolOutput) -> list[tuple[int, torch.Tensor, float]]:
        if not self.use_optimized:
            return super()._active_cccs(pool_output)
        active_cccs: list[tuple[int, torch.Tensor, float]] = []
        for index in pool_output.fired_indices:
            direction = self.ccc_pool.concept_directions[index].detach().clone()
            active_cccs.append((index, direction, self._mean_confidence(pool_output.outputs[index])))
        return active_cccs

    def _ensure_growth_capacity(self) -> None:
        if not self.use_optimized or not isinstance(self.ccc_pool, BatchedCCCPool):
            return
        if self.ccc_pool._first_uncommitted_index() is not None:
            return
        grown_pool = self.ccc_pool.grow()
        if grown_pool is self.ccc_pool:
            return
        self.ccc_pool = grown_pool
        self.config.ccc = grown_pool.config

    @torch.no_grad()
    def learn_from_perception(self, perception, raw_input: torch.Tensor) -> None:  # type: ignore[override]
        if not self.use_optimized:
            super().learn_from_perception(perception, raw_input)
            return
        if perception.pool_output.recruited or perception.pool_output.fired_indices:
            return

        self._ensure_growth_capacity()
        recruit_index, _ = self.ccc_pool.recruit(raw_input, timestep=self.timestep)
        if recruit_index is None:
            return

        direction = self.ccc_pool.concept_directions[recruit_index].detach().clone()
        self.fabric.register_activation(recruit_index, direction, confidence=1.0, timestep=self.timestep)
        self.fabric.form_associations(self.timestep)
        self.gnw.inject(recruit_index, direction, priority=1.0)


__all__ = [
    "AdaptiveCapacity",
    "BatchedCCCPool",
    "ComparisonResult",
    "HierarchicalSDM",
    "MemoryEfficientSDM",
    "OptimizedSDM",
    "PoolInferenceSummary",
    "PoolSharding",
    "ProfileResult",
    "ScaledBioARN",
    "ScalingProfiler",
    "estimate_module_memory_mb",
]
