"""Patch-based encodings for vision tensors."""

from __future__ import annotations

import torch


class PatchEncoder:
    """Split images into patches and encode each independently."""

    def __init__(
        self,
        image_size: tuple = (32, 32, 3),
        patch_size: int = 8,
        output_dim: int = 128,
        *,
        seed: int = 0,
        sparsity: float = 0.25,
    ):
        height, width, channels = (int(value) for value in image_size)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("patch_size must evenly divide the image height and width.")
        self.image_size = (height, width, channels)
        self.patch_size = int(patch_size)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.sparsity = float(sparsity)
        self.is_fitted = True

        self.num_patches_y = height // self.patch_size
        self.num_patches_x = width // self.patch_size
        self.num_patches = self.num_patches_y * self.num_patches_x
        if self.output_dim < self.num_patches:
            raise ValueError("output_dim must be at least the number of patches.")

        self.patch_dim = self.patch_size * self.patch_size * channels
        base_dim, remainder = divmod(self.output_dim, self.num_patches)
        self.patch_code_dims = [
            base_dim + (1 if index < remainder else 0) for index in range(self.num_patches)
        ]

        generator = torch.Generator().manual_seed(self.seed)
        self.patch_projections = [
            torch.randn(dim, self.patch_dim, generator=generator, dtype=torch.float32)
            for dim in self.patch_code_dims
        ]

    def get_output_dim(self, input_dim: int) -> int:
        expected = self.image_size[0] * self.image_size[1] * self.image_size[2]
        if int(input_dim) != expected:
            raise ValueError(f"Expected input_dim={expected}, received {input_dim}.")
        return self.output_dim

    def fit(self, data: torch.Tensor) -> "PatchEncoder":
        del data
        return self

    def partial_fit(self, x: torch.Tensor) -> "PatchEncoder":
        del x
        return self

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        expected = self.image_size[0] * self.image_size[1] * self.image_size[2]
        if x.dim() == 1:
            if x.numel() != expected:
                raise ValueError(f"Expected {expected} features, got {x.numel()}.")
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2 or x.shape[-1] != expected:
            raise ValueError(f"PatchEncoder expects shape ({expected},) or (batch, {expected}).")
        return x.to(torch.float32), False

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        batch_size = batch.shape[0]
        channels = self.image_size[2]
        images = batch.reshape(batch_size, channels, self.image_size[0], self.image_size[1])
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size, self.num_patches, self.patch_dim
        )

        codes: list[torch.Tensor] = []
        for patch_index, (projection, code_dim) in enumerate(
            zip(self.patch_projections, self.patch_code_dims, strict=False)
        ):
            patch = patches[:, patch_index]
            patch = patch - patch.mean(dim=-1, keepdim=True)
            patch = patch / patch.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-5)
            projected = patch @ projection.to(patch).transpose(0, 1)
            top_k = max(1, int(round(code_dim * self.sparsity)))
            _, active_indices = torch.topk(projected, k=top_k, dim=-1)
            binary = torch.zeros(batch_size, code_dim, device=patch.device, dtype=torch.float32)
            binary.scatter_(1, active_indices, 1.0)
            codes.append(binary)

        encoded = torch.cat(codes, dim=-1)
        return encoded.squeeze(0) if squeeze else encoded

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)


__all__ = ["PatchEncoder"]
