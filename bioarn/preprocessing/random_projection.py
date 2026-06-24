"""Sparse random projection for fast dimensionality reduction."""

from __future__ import annotations

import torch


class SparseRandomProjection:
    """Johnson-Lindenstrauss random projection for dimensionality reduction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        density: float = 0.1,
        *,
        seed: int = 0,
    ):
        if not 0.0 < density <= 1.0:
            raise ValueError("density must be in (0, 1].")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.density = float(density)
        self.seed = int(seed)
        self.is_fitted = True
        generator = torch.Generator().manual_seed(self.seed)
        active = torch.rand(
            self.output_dim, self.input_dim, generator=generator, dtype=torch.float32
        ) < self.density
        signs = torch.where(
            torch.rand(self.output_dim, self.input_dim, generator=generator) < 0.5,
            -torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
        )
        scale = (self.output_dim * self.density) ** -0.5
        self.projection = (active.to(torch.float32) * signs) * scale

    def get_output_dim(self, input_dim: int) -> int:
        if int(input_dim) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, received {input_dim}.")
        return self.output_dim

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            if x.numel() != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.numel()}.")
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SparseRandomProjection expects shape ({self.input_dim},) or (batch, {self.input_dim})."
            )
        return x.to(torch.float32), False

    def fit(self, data: torch.Tensor) -> "SparseRandomProjection":
        del data
        return self

    def partial_fit(self, x: torch.Tensor) -> "SparseRandomProjection":
        del x
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        projected = batch @ self.projection.to(batch).transpose(0, 1)
        return projected.squeeze(0) if squeeze else projected

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)


__all__ = ["SparseRandomProjection"]
