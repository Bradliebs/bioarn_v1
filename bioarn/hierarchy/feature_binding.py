"""Cross-layer feature binding via simple Hebbian synchrony."""

from __future__ import annotations

import torch


class FeatureBinding:
    """Bind features across hierarchy layers via temporal synchrony."""

    def __init__(
        self,
        layer_pool_sizes: list[int],
        *,
        binding_strength: float = 0.1,
        decay: float = 0.995,
    ) -> None:
        if len(layer_pool_sizes) < 2:
            raise ValueError("At least two layers are required for feature binding.")
        self.binding_strength = float(binding_strength)
        self.decay = float(decay)
        self.weights = [
            torch.zeros(int(lower), int(higher), dtype=torch.float32)
            for lower, higher in zip(layer_pool_sizes[:-1], layer_pool_sizes[1:], strict=False)
        ]

    @staticmethod
    def _flatten(indices: list[int] | list[list[int]]) -> list[int]:
        if not indices:
            return []
        if isinstance(indices[0], list):
            unique = {
                int(value)
                for sublist in indices
                for value in sublist
            }
            return sorted(unique)
        return sorted({int(value) for value in indices})

    @torch.no_grad()
    def strengthen(
        self,
        pair_index: int,
        lower_indices: list[int] | list[list[int]],
        higher_indices: list[int] | list[list[int]],
        *,
        delay: int = 1,
    ) -> float:
        """Strengthen bindings when lower and higher features co-activate."""

        lower = self._flatten(lower_indices)
        higher = self._flatten(higher_indices)
        if not lower or not higher:
            return 0.0

        matrix = self.weights[int(pair_index)]
        matrix.mul_(self.decay)
        delta = self.binding_strength / max(int(delay), 1)
        for lower_index in lower:
            matrix[lower_index, higher] = (matrix[lower_index, higher] + delta).clamp_(0.0, 1.0)
        return float(matrix[lower][:, higher].mean().item())

    @torch.no_grad()
    def get_strength(
        self,
        pair_index: int,
        lower_indices: list[int] | list[list[int]],
        higher_indices: list[int] | list[list[int]],
    ) -> float:
        """Return the mean binding strength for a feature pair."""

        lower = self._flatten(lower_indices)
        higher = self._flatten(higher_indices)
        if not lower or not higher:
            return 0.0
        matrix = self.weights[int(pair_index)]
        return float(matrix[lower][:, higher].mean().item())

    @torch.no_grad()
    def reset(self) -> None:
        """Reset all binding weights."""

        for matrix in self.weights:
            matrix.zero_()


__all__ = ["FeatureBinding"]
