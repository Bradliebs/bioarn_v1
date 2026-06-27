"""Local synaptic consolidation for Hebbian CCC learning."""

from __future__ import annotations

import torch


class SynapticConsolidation:
    """Track CCC importance from firing frequency and scale Hebbian plasticity."""

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
        self.importance = torch.cat((self.importance, zeros.clone()), dim=0)

    def copy_from(self, other: "SynapticConsolidation") -> None:
        self.ensure_capacity(other.size)
        self.fire_counts.zero_()
        self.importance.zero_()
        self.fire_counts[: other.size].copy_(other.fire_counts)
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
        self.importance[target_slice].copy_(other.importance[source_slice])
        self.total_updates = max(int(self.total_updates), int(other.total_updates))

    @torch.no_grad()
    def update_importance(
        self,
        fired_indices: list[int] | torch.Tensor,
        *,
        current_size: int | None = None,
        weight: float = 1.0,
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
        if indices.numel() > 0:
            valid = indices[(indices >= 0) & (indices < size)]
            if valid.numel() > 0:
                updates = torch.full(
                    (valid.numel(),),
                    float(weight),
                    device=self.fire_counts.device,
                    dtype=self.fire_counts.dtype,
                )
                self.fire_counts.index_add_(0, valid, updates)

        if size <= 0:
            return self.importance

        active_counts = self.fire_counts[:size]
        peak = float(active_counts.max().item()) if active_counts.numel() else 0.0
        self.importance[:size].zero_()
        if peak > 0.0:
            self.importance[:size].copy_(active_counts / peak)
        if size < self.size:
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
