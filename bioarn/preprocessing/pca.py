"""Streaming PCA with bounded-memory incremental SVD updates."""

from __future__ import annotations

import torch


class OnlinePCA:
    """Incremental PCA for streaming data (no full dataset needed)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        *,
        max_samples: int = 512,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(min(output_dim, input_dim))
        self.max_samples = int(max(2, max_samples))
        self.seed = int(seed)
        self.is_fitted = False
        self.sample_count = 0
        self.mean = torch.zeros(self.input_dim, dtype=torch.float32)
        self.components = torch.eye(self.output_dim, self.input_dim, dtype=torch.float32)
        self._samples = torch.empty(0, self.input_dim, dtype=torch.float32)
        self._generator = torch.Generator().manual_seed(self.seed)

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
                f"OnlinePCA expects shape ({self.input_dim},) or (batch, {self.input_dim})."
            )
        return x.to(torch.float32), False

    def _downsample_buffer(self) -> None:
        if self._samples.shape[0] <= self.max_samples:
            return
        keep = torch.randperm(
            self._samples.shape[0], generator=self._generator, device=self._samples.device
        )[: self.max_samples]
        self._samples = self._samples.index_select(0, keep).contiguous()

    def _recompute_components(self) -> None:
        if self._samples.shape[0] < 2:
            return
        self.mean = self._samples.mean(dim=0)
        centered = self._samples - self.mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: self.output_dim].contiguous()
        if components.numel():
            max_abs_index = torch.argmax(components.abs(), dim=1)
            signs = torch.sign(
                components[torch.arange(components.shape[0]), max_abs_index]
            ).unsqueeze(-1)
            components = components * torch.where(signs == 0, torch.ones_like(signs), signs)
        if components.shape[0] < self.output_dim:
            padding = torch.zeros(
                self.output_dim - components.shape[0],
                self.input_dim,
                dtype=components.dtype,
                device=components.device,
            )
            components = torch.cat([components, padding], dim=0)
        self.components = components
        self.is_fitted = True

    def partial_fit(self, x: torch.Tensor) -> "OnlinePCA":
        batch, _ = self._ensure_batch(x)
        self.sample_count += batch.shape[0]
        if self._samples.numel() == 0:
            self._samples = batch.detach().clone()
        else:
            self._samples = torch.cat([self._samples, batch.detach().clone()], dim=0)
        self._downsample_buffer()
        self._recompute_components()
        return self

    def fit(self, data: torch.Tensor) -> "OnlinePCA":
        batch, _ = self._ensure_batch(data)
        self.sample_count = batch.shape[0]
        if batch.shape[0] > self.max_samples:
            indices = torch.linspace(0, batch.shape[0] - 1, self.max_samples).round().to(torch.long)
            batch = batch.index_select(0, indices)
        self._samples = batch.detach().clone()
        self._recompute_components()
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        centered = batch - self.mean.to(batch)
        projected = centered @ self.components.to(batch).transpose(0, 1)
        return projected.squeeze(0) if squeeze else projected

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


__all__ = ["OnlinePCA"]
