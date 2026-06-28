"""Temporal context buffering for workspace-conditioned sequence learning."""

from __future__ import annotations

from collections import deque
import math

import torch
import torch.nn.functional as F

from bioarn.core.math_utils import normalize


class TemporalContextBuffer:
    """Maintain a short rolling history of recent concept activations."""

    def __init__(self, window_size: int = 8, concept_dim: int = 256) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if concept_dim <= 0:
            raise ValueError("concept_dim must be positive.")
        self.window_size = int(window_size)
        self.concept_dim = int(concept_dim)
        self._activations: deque[torch.Tensor] = deque(maxlen=self.window_size)
        self._fired_history: deque[list[int]] = deque(maxlen=self.window_size)

    def _align(self, concept_activation: torch.Tensor) -> torch.Tensor:
        vector = concept_activation.detach().reshape(-1).to(torch.float32)
        if vector.numel() > self.concept_dim:
            vector = vector[: self.concept_dim]
        elif vector.numel() < self.concept_dim:
            vector = F.pad(vector, (0, self.concept_dim - vector.numel()))
        return vector

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        if float(vector.norm().item()) <= 1e-8:
            return torch.zeros_like(vector)
        return normalize(vector.unsqueeze(0)).squeeze(0)

    def push(self, concept_activation: torch.Tensor, fired_indices: list[int]) -> None:
        """Add a new frame's concept activity to the rolling context."""

        aligned = self._align(concept_activation)
        self._activations.append(aligned.detach().clone())
        sanitized = [
            int(index) for index in fired_indices
            if 0 <= int(index) < self.concept_dim
        ]
        self._fired_history.append(sanitized)

    def get_context(self) -> torch.Tensor:
        """Return a recency-weighted summary vector over the buffered frames."""

        if not self._activations:
            return torch.zeros(self.concept_dim, dtype=torch.float32)

        stacked = torch.stack(list(self._activations), dim=0)
        count = stacked.shape[0]
        weights = torch.tensor(
            [math.exp(-(count - position - 1) / max(1.0, self.window_size / 2.0)) for position in range(count)],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        context = (weights.unsqueeze(-1) * stacked).sum(dim=0) / weights.sum().clamp_min(1e-6)
        return self._normalize(context)

    def get_temporal_pattern(self) -> list[list[int]]:
        """Return the buffered sequence of sparse frame activations."""

        return [indices.copy() for indices in self._fired_history]

    def clear(self) -> None:
        """Reset the temporal buffer."""

        self._activations.clear()
        self._fired_history.clear()
