"""Local synaptic consolidation for Hebbian CCC learning."""

from __future__ import annotations

import torch


class SynapticConsolidation:
    """Track CCC importance from firing frequency, confidence, and recency."""

    def __init__(
        self,
        num_slots: int,
        *,
        strength: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.strength = float(max(0.0, strength))
        self.fire_counts = torch.zeros(int(max(0, num_slots)), device=device, dtype=dtype)
        self.confidence_sums = torch.zeros_like(self.fire_counts)
        self.mean_confidence = torch.zeros_like(self.fire_counts)
        self.last_fire_step = torch.full_like(self.fire_counts, -1.0)
        self.importance = torch.zeros_like(self.fire_counts)
        self.total_updates = 0

    @property
    def enabled(self) -> bool:
        return self.strength > 0.0

    @property
    def size(self) -> int:
        return int(self.importance.numel())

    def ensure_capacity(self, num_slots: int) -> None:
        target = int(max(0, num_slots))
        if target <= self.size:
            return
        pad = target - self.size
        zeros = torch.zeros(
            pad,
            device=self.importance.device,
            dtype=self.importance.dtype,
        )
        self.fire_counts = torch.cat((self.fire_counts, zeros), dim=0)
        self.confidence_sums = torch.cat((self.confidence_sums, zeros.clone()), dim=0)
        self.mean_confidence = torch.cat((self.mean_confidence, zeros.clone()), dim=0)
        self.last_fire_step = torch.cat((self.last_fire_step, torch.full_like(zeros, -1.0)), dim=0)
        self.importance = torch.cat((self.importance, zeros.clone()), dim=0)

    def copy_from(self, other: "SynapticConsolidation") -> None:
        self.ensure_capacity(other.size)
        self.fire_counts.zero_()
        self.confidence_sums.zero_()
        self.mean_confidence.zero_()
        self.last_fire_step.fill_(-1.0)
        self.importance.zero_()
        self.fire_counts[: other.size].copy_(other.fire_counts)
        self.confidence_sums[: other.size].copy_(other.confidence_sums)
        self.mean_confidence[: other.size].copy_(other.mean_confidence)
        self.last_fire_step[: other.size].copy_(other.last_fire_step)
        self.importance[: other.size].copy_(other.importance)
        self.total_updates = int(other.total_updates)

    def copy_slice_from(
        self,
        other: "SynapticConsolidation",
        *,
        source_start: int = 0,
        target_start: int = 0,
        length: int | None = None,
    ) -> None:
        if length is None:
            length = max(0, other.size - int(source_start))
        length = int(max(0, length))
        if length == 0:
            return
        self.ensure_capacity(target_start + length)
        source_slice = slice(int(source_start), int(source_start) + length)
        target_slice = slice(int(target_start), int(target_start) + length)
        self.fire_counts[target_slice].copy_(other.fire_counts[source_slice])
        self.confidence_sums[target_slice].copy_(other.confidence_sums[source_slice])
        self.mean_confidence[target_slice].copy_(other.mean_confidence[source_slice])
        self.last_fire_step[target_slice].copy_(other.last_fire_step[source_slice])
        self.importance[target_slice].copy_(other.importance[source_slice])
        self.total_updates = max(int(self.total_updates), int(other.total_updates))

    @torch.no_grad()
    def update_importance(
        self,
        fired_indices: list[int] | torch.Tensor,
        *,
        current_size: int | None = None,
        weight: float = 1.0,
        confidences: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        size = self.size if current_size is None else int(max(0, current_size))
        self.ensure_capacity(size)
        self.total_updates += 1

        if isinstance(fired_indices, torch.Tensor):
            indices = fired_indices.to(device=self.fire_counts.device, dtype=torch.long).reshape(-1)
        else:
            indices = torch.tensor(
                [int(index) for index in fired_indices],
                device=self.fire_counts.device,
                dtype=torch.long,
            )
        if confidences is None:
            confidence_values = torch.ones(
                indices.numel(),
                device=self.fire_counts.device,
                dtype=self.fire_counts.dtype,
            )
        elif isinstance(confidences, torch.Tensor):
            confidence_values = confidences.to(
                device=self.fire_counts.device,
                dtype=self.fire_counts.dtype,
            ).reshape(-1)
        else:
            confidence_values = torch.tensor(
                [float(value) for value in confidences],
                device=self.fire_counts.device,
                dtype=self.fire_counts.dtype,
            )
        if confidence_values.numel() != indices.numel():
            raise ValueError(
                "confidences must have the same number of elements as fired_indices."
            )
        if indices.numel() > 0:
            valid = indices[(indices >= 0) & (indices < size)]
            if valid.numel() > 0:
                valid_confidences = confidence_values[(indices >= 0) & (indices < size)].clamp(0.0, 1.0)
                updates = torch.full(
                    (valid.numel(),),
                    float(weight),
                    device=self.fire_counts.device,
                    dtype=self.fire_counts.dtype,
                )
                self.fire_counts.index_add_(0, valid, updates)
                self.confidence_sums.index_add_(0, valid, valid_confidences * float(weight))
                self.last_fire_step.index_fill_(
                    0,
                    valid,
                    float(self.total_updates),
                )

        if size <= 0:
            return self.importance

        active_counts = self.fire_counts[:size]
        active_confidence = self.confidence_sums[:size]
        self.mean_confidence[:size].zero_()
        nonzero = active_counts > 0
        self.mean_confidence[:size] = torch.where(
            nonzero,
            active_confidence / active_counts.clamp_min(1.0),
            torch.zeros_like(active_confidence),
        )
        peak = float(active_counts.max().item()) if active_counts.numel() else 0.0
        self.importance[:size].zero_()
        if peak > 0.0:
            frequency = active_counts / peak
            steps_since_fire = torch.where(
                self.last_fire_step[:size] >= 0.0,
                torch.full_like(self.last_fire_step[:size], float(self.total_updates))
                - self.last_fire_step[:size],
                torch.full_like(self.last_fire_step[:size], float(self.total_updates)),
            )
            recency_decay = max(32.0, float(size) * 2.0)
            recency_weight = torch.exp(-steps_since_fire / recency_decay)
            recency_weight = torch.where(nonzero, recency_weight, torch.zeros_like(recency_weight))
            self.importance[:size].copy_(
                (frequency * self.mean_confidence[:size].clamp(0.0, 1.0) * recency_weight)
                .clamp(0.0, 1.0)
            )
        if size < self.size:
            self.mean_confidence[size:].zero_()
            self.last_fire_step[size:].fill_(-1.0)
            self.importance[size:].zero_()
        return self.importance[:size]

    def learning_rate_scales(
        self,
        *,
        current_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        size = self.size if current_size is None else int(max(0, current_size))
        if size <= 0:
            scale = torch.empty(0, device=device or self.importance.device, dtype=dtype or self.importance.dtype)
            return scale
        importance = self.importance[:size]
        scale = (1.0 - (self.strength * importance.clamp(0.0, 1.0))).clamp(0.0, 1.0)
        return scale.to(device=device or scale.device, dtype=dtype or scale.dtype)

    def effective_lr(self, learning_rate: float, index: int) -> float:
        if index < 0 or index >= self.size:
            return float(learning_rate)
        scale = float(self.learning_rate_scales(current_size=index + 1)[index].item())
        return float(learning_rate) * scale


__all__ = ["SynapticConsolidation"]
