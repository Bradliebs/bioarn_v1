"""Sparse lateral predictive coding between CCCs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import LateralPredictionConfig
from bioarn.core.math_utils import normalize


class LateralPredictionNetwork(nn.Module):
    """Lateral prediction connections between CCCs in a pool."""

    def __init__(self, pool_size: int, concept_dim: int, config: LateralPredictionConfig):
        super().__init__()
        self.config = config
        self.pool_size = int(max(1, pool_size))
        self.concept_dim = int(max(1, concept_dim))
        max_neighbors = min(int(config.max_neighbors), max(self.pool_size - 1, 1))
        self.register_buffer(
            "neighbor_indices",
            torch.full((self.pool_size, max_neighbors), -1, dtype=torch.long),
        )
        self.register_buffer(
            "neighbor_mask",
            torch.zeros(self.pool_size, max_neighbors, dtype=torch.bool),
        )
        self.register_buffer(
            "lateral_weights",
            torch.ones(self.pool_size, max_neighbors, self.concept_dim, dtype=torch.float32),
        )
        self.register_buffer("refresh_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_prediction_scores", torch.zeros(self.pool_size, dtype=torch.float32))
        self.register_buffer("last_error_scores", torch.zeros(self.pool_size, dtype=torch.float32))
        self.register_buffer("last_attention_scores", torch.ones(self.pool_size, dtype=torch.float32))
        self.register_buffer("last_mean_error", torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _active_mask(concept_directions: torch.Tensor) -> torch.Tensor:
        return concept_directions.to(torch.float32).norm(dim=-1) > 1e-6

    @staticmethod
    def _prediction_score(
        source_direction: torch.Tensor,
        weight_vector: torch.Tensor,
        target_direction: torch.Tensor,
    ) -> float:
        projected = weight_vector.to(source_direction) * source_direction
        if float(projected.norm().item()) <= 1e-6 or float(target_direction.norm().item()) <= 1e-6:
            return 0.0
        projected = normalize(projected.unsqueeze(0)).squeeze(0)
        target = normalize(target_direction.unsqueeze(0)).squeeze(0).to(projected)
        score = F.cosine_similarity(projected.unsqueeze(0), target.unsqueeze(0)).item()
        return float(max(0.0, min(1.0, score)))

    def _resize_buffer(
        self,
        old: torch.Tensor,
        *,
        new_shape: tuple[int, ...],
        fill_value: float | int | bool,
    ) -> torch.Tensor:
        new_tensor = torch.full(new_shape, fill_value, dtype=old.dtype, device=old.device)
        slices = tuple(slice(0, min(old.shape[index], new_shape[index])) for index in range(len(new_shape)))
        if slices:
            new_tensor[slices] = old[slices]
        return new_tensor

    @torch.no_grad()
    def set_pool_size(self, pool_size: int) -> None:
        target_size = int(max(1, pool_size))
        if target_size == self.pool_size:
            return
        target_neighbors = min(int(self.config.max_neighbors), max(target_size - 1, 1))
        self.neighbor_indices = self._resize_buffer(
            self.neighbor_indices,
            new_shape=(target_size, target_neighbors),
            fill_value=-1,
        )
        self.neighbor_mask = self._resize_buffer(
            self.neighbor_mask,
            new_shape=(target_size, target_neighbors),
            fill_value=False,
        )
        self.lateral_weights = self._resize_buffer(
            self.lateral_weights,
            new_shape=(target_size, target_neighbors, self.concept_dim),
            fill_value=1.0,
        )
        self.last_prediction_scores = self._resize_buffer(
            self.last_prediction_scores,
            new_shape=(target_size,),
            fill_value=0.0,
        )
        self.last_error_scores = self._resize_buffer(
            self.last_error_scores,
            new_shape=(target_size,),
            fill_value=0.0,
        )
        self.last_attention_scores = self._resize_buffer(
            self.last_attention_scores,
            new_shape=(target_size,),
            fill_value=1.0,
        )
        self.pool_size = target_size

    @torch.no_grad()
    def copy_state_from(self, other: "LateralPredictionNetwork") -> None:
        self.set_pool_size(other.pool_size)
        row_count = min(self.pool_size, other.pool_size)
        common_neighbors = min(self.neighbor_indices.shape[1], other.neighbor_indices.shape[1])
        self.neighbor_indices[:row_count, :common_neighbors].copy_(
            other.neighbor_indices[:row_count, :common_neighbors]
        )
        self.neighbor_mask[:row_count, :common_neighbors].copy_(
            other.neighbor_mask[:row_count, :common_neighbors]
        )
        self.lateral_weights[:row_count, :common_neighbors].copy_(
            other.lateral_weights[:row_count, :common_neighbors]
        )
        self.last_prediction_scores[:row_count].copy_(other.last_prediction_scores[:row_count])
        self.last_error_scores[:row_count].copy_(other.last_error_scores[:row_count])
        self.last_attention_scores[:row_count].copy_(other.last_attention_scores[:row_count])
        self.last_mean_error.copy_(other.last_mean_error)
        self.refresh_counter.copy_(other.refresh_counter)

    @torch.no_grad()
    def refresh_neighbors(
        self,
        concept_directions: torch.Tensor,
        *,
        source_indices: list[int] | None = None,
        force: bool = False,
    ) -> None:
        if self.pool_size <= 1:
            return
        concept_batch = concept_directions.to(self.lateral_weights.device, dtype=torch.float32)
        active_mask = self._active_mask(concept_batch)
        if int(active_mask.sum().item()) <= 1:
            self.neighbor_indices.fill_(-1)
            self.neighbor_mask.zero_()
            return
        if source_indices is None:
            sources = active_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        else:
            sources = [int(index) for index in source_indices if 0 <= int(index) < self.pool_size]
        if not sources:
            return
        should_refresh = force or int(self.refresh_counter.item()) % int(self.config.refresh_interval) == 0
        if not should_refresh and source_indices is None:
            return
        normalized = normalize(concept_batch)
        available = self.neighbor_indices.shape[1]
        for source in sources:
            if not bool(active_mask[source].item()):
                self.neighbor_indices[source].fill_(-1)
                self.neighbor_mask[source].zero_()
                continue
            similarities = torch.matmul(normalized, normalized[source].unsqueeze(-1)).squeeze(-1)
            similarities[source] = -1.0
            similarities = similarities.masked_fill(~active_mask, -1.0)
            top_k = min(available, max(int(active_mask.sum().item()) - 1, 0))
            new_indices = torch.full_like(self.neighbor_indices[source], -1)
            new_mask = torch.zeros_like(self.neighbor_mask[source])
            old_weights = {
                int(target): self.lateral_weights[source, slot].detach().clone()
                for slot, target in enumerate(self.neighbor_indices[source].tolist())
                if self.neighbor_mask[source, slot] and target >= 0
            }
            if top_k > 0:
                values, indices = torch.topk(similarities, k=top_k)
                valid_targets = [(int(target), float(value)) for target, value in zip(indices.tolist(), values.tolist(), strict=False) if value > 0.0]
                for slot, (target, _) in enumerate(valid_targets[:available]):
                    new_indices[slot] = target
                    new_mask[slot] = True
                    if target in old_weights:
                        self.lateral_weights[source, slot].copy_(old_weights[target])
                    else:
                        self.lateral_weights[source, slot].fill_(1.0)
                if len(valid_targets) < available:
                    self.lateral_weights[source, len(valid_targets) :].fill_(1.0)
            else:
                self.lateral_weights[source].fill_(1.0)
            self.neighbor_indices[source].copy_(new_indices)
            self.neighbor_mask[source].copy_(new_mask)
        self.refresh_counter.add_(1)

    @torch.no_grad()
    def predict_lateral(
        self,
        fired_indices: list[int],
        concept_directions: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Given which CCCs fired, predict what else should fire."""

        if not fired_indices:
            self.last_prediction_scores.zero_()
            self.last_error_scores.zero_()
            self.last_attention_scores.fill_(1.0)
            self.last_mean_error.zero_()
            return {}
        self.refresh_neighbors(concept_directions, source_indices=fired_indices)
        concept_batch = concept_directions.to(self.lateral_weights.device, dtype=torch.float32)
        normalized = normalize(concept_batch)
        predictions: dict[int, torch.Tensor] = {}
        for source in fired_indices:
            if source < 0 or source >= self.pool_size:
                continue
            source_direction = normalized[source]
            if float(source_direction.norm().item()) <= 1e-6:
                continue
            for slot, target in enumerate(self.neighbor_indices[source].tolist()):
                if not bool(self.neighbor_mask[source, slot].item()) or target < 0 or target >= self.pool_size:
                    continue
                target_direction = normalized[target]
                score = self._prediction_score(
                    source_direction,
                    self.lateral_weights[source, slot],
                    target_direction,
                )
                if score < float(self.config.prediction_threshold):
                    continue
                score_tensor = torch.tensor(score, dtype=torch.float32, device=concept_batch.device)
                if target in predictions:
                    predictions[target] = torch.maximum(predictions[target], score_tensor)
                else:
                    predictions[target] = score_tensor
        return predictions

    def compute_lateral_errors(
        self,
        predictions: dict[int, torch.Tensor],
        actual_fired: list[int],
    ) -> dict[int, float]:
        """Compare lateral predictions against actual firing pattern."""

        actual = {int(index) for index in actual_fired}
        candidates = set(predictions) | actual
        errors: dict[int, float] = {}
        for index in candidates:
            predicted = float(predictions.get(index, torch.tensor(0.0)).item()) if index in predictions else 0.0
            observed = 1.0 if index in actual else 0.0
            errors[index] = float(abs(observed - predicted))
        return errors

    def summarize_errors(self, errors: dict[int, float]) -> float:
        if not errors:
            return 0.0
        return float(sum(errors.values()) / max(len(errors), 1))

    @torch.no_grad()
    def cache_error_state(
        self,
        predictions: dict[int, torch.Tensor],
        errors: dict[int, float],
        attention: dict[int, float] | None = None,
    ) -> None:
        self.last_prediction_scores.zero_()
        self.last_error_scores.zero_()
        self.last_attention_scores.fill_(1.0)
        for index, prediction in predictions.items():
            if 0 <= int(index) < self.pool_size:
                self.last_prediction_scores[int(index)] = float(prediction.item())
        for index, error in errors.items():
            if 0 <= int(index) < self.pool_size:
                self.last_error_scores[int(index)] = float(error)
        if attention is not None:
            for index, value in attention.items():
                if 0 <= int(index) < self.pool_size:
                    self.last_attention_scores[int(index)] = float(value)
        self.last_mean_error.fill_(self.summarize_errors(errors))

    @torch.no_grad()
    def hebbian_update(
        self,
        fired_indices: list[int],
        concept_directions: torch.Tensor,
    ) -> None:
        """Update sparse lateral weights with a local co-firing rule."""

        if not fired_indices:
            return
        self.refresh_neighbors(concept_directions, source_indices=fired_indices, force=True)
        concept_batch = concept_directions.to(self.lateral_weights.device, dtype=torch.float32)
        normalized = normalize(concept_batch).abs()
        fired_set = {int(index) for index in fired_indices if 0 <= int(index) < self.pool_size}
        for source in fired_set:
            source_direction = normalized[source]
            if float(source_direction.norm().item()) <= 1e-6:
                continue
            for slot, target in enumerate(self.neighbor_indices[source].tolist()):
                if not bool(self.neighbor_mask[source, slot].item()) or target < 0 or target >= self.pool_size:
                    continue
                overlap = source_direction * normalized[target]
                if target in fired_set:
                    self.lateral_weights[source, slot].add_(
                        float(self.config.hebbian_lr) * overlap.to(self.lateral_weights.dtype)
                    )
                else:
                    self.lateral_weights[source, slot].sub_(
                        float(self.config.anti_hebbian_lr) * overlap.to(self.lateral_weights.dtype)
                    )
                self.lateral_weights[source, slot].clamp_(
                    min=float(self.config.min_weight),
                    max=float(self.config.max_weight),
                )


__all__ = ["LateralPredictionNetwork"]
