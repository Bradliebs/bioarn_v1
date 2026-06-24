"""Receptive field extraction utilities for hierarchical vision."""

from __future__ import annotations

import torch


class ReceptiveFieldExtractor:
    """Extract patches at multiple scales mimicking visual receptive fields."""

    def __init__(
        self,
        image_size: tuple[int, int, int] = (32, 32, 3),
        *,
        include_position: bool = True,
    ) -> None:
        height, width, channels = (int(value) for value in image_size)
        self.image_size = (height, width, channels)
        self.include_position = bool(include_position)
        self.position_dim = 2 if self.include_position else 0
        self.last_grid_shape: tuple[int, int] = (0, 0)
        self.last_patch_positions: list[tuple[int, int]] = []

    def _ensure_image(self, image: torch.Tensor) -> torch.Tensor:
        height, width, channels = self.image_size
        expected = height * width * channels
        if image.dim() == 1:
            if image.numel() != expected:
                raise ValueError(f"Expected flattened image with {expected} values.")
            return image.reshape(channels, height, width).to(torch.float32)
        if image.dim() == 3:
            if tuple(image.shape) == (channels, height, width):
                return image.to(torch.float32)
            if tuple(image.shape) == (height, width, channels):
                return image.permute(2, 0, 1).to(torch.float32)
        raise ValueError("image must have shape (C, H, W), (H, W, C), or (H*W,).")

    def _position_encoding(self, top: int, left: int, patch_size: int) -> torch.Tensor:
        if not self.include_position:
            return torch.empty(0, dtype=torch.float32)
        height, width, _ = self.image_size
        center_y = (top + (patch_size / 2.0)) / max(height, 1)
        center_x = (left + (patch_size / 2.0)) / max(width, 1)
        scale = max(1.0, ((patch_size * patch_size * self.image_size[2]) / 2.0) ** 0.5)
        return torch.tensor(
            [((2.0 * center_y) - 1.0) * scale, ((2.0 * center_x) - 1.0) * scale],
            dtype=torch.float32,
        )

    @torch.no_grad()
    def extract_patches(
        self,
        image: torch.Tensor,
        patch_size: int,
        stride: int,
    ) -> list[torch.Tensor]:
        """Extract patches with optional spatial position encoding."""

        frame = self._ensure_image(image)
        channels, height, width = frame.shape
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive.")
        if patch_size > height or patch_size > width:
            raise ValueError("patch_size must fit within the configured image.")

        patches: list[torch.Tensor] = []
        positions: list[tuple[int, int]] = []
        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                patch = frame[:, top : top + patch_size, left : left + patch_size].reshape(-1)
                patch = patch - patch.mean()
                patch = patch / patch.std(unbiased=False).clamp_min(1e-5)
                if self.include_position:
                    patch = torch.cat((patch, self._position_encoding(top, left, patch_size)))
                patches.append(patch.to(torch.float32))
                positions.append((top, left))

        self.last_patch_positions = positions
        self.last_grid_shape = (
            ((height - patch_size) // stride) + 1,
            ((width - patch_size) // stride) + 1,
        )
        return patches

    @staticmethod
    def make_grouping(
        grid_shape: tuple[int, int],
        *,
        group_size: int,
        stride: int | None = None,
    ) -> list[list[int]]:
        """Group adjacent receptive fields by 2D neighborhood."""

        rows, cols = (int(value) for value in grid_shape)
        stride = group_size if stride is None else int(stride)
        if rows <= 0 or cols <= 0:
            return []
        if group_size <= 0 or stride <= 0:
            raise ValueError("group_size and stride must be positive.")

        groups: list[list[int]] = []
        for top in range(0, rows - group_size + 1, stride):
            for left in range(0, cols - group_size + 1, stride):
                group: list[int] = []
                for y in range(top, top + group_size):
                    for x in range(left, left + group_size):
                        group.append((y * cols) + x)
                groups.append(group)
        return groups

    @staticmethod
    @torch.no_grad()
    def pool_activations(
        activations: list[torch.Tensor],
        grouping: list[list[int]],
    ) -> list[torch.Tensor]:
        """Concatenate grouped activations for the next hierarchy layer."""

        if not grouping:
            return []
        return [
            torch.cat([activations[index].reshape(-1).to(torch.float32) for index in group], dim=0)
            for group in grouping
        ]

    @torch.no_grad()
    def extract_multi_scale(
        self,
        image: torch.Tensor,
        scales: list[int],
    ) -> dict[int, list[torch.Tensor]]:
        """Extract patches at multiple receptive-field sizes."""

        return {
            int(scale): self.extract_patches(image, patch_size=int(scale), stride=int(scale))
            for scale in scales
        }


__all__ = ["ReceptiveFieldExtractor"]
