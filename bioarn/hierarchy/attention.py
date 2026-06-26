"""Spatial attention utilities for the visual hierarchy."""

from __future__ import annotations

import torch


class SpatialAttention:
    """Contrast-and-center-biased patch gain control."""

    def __init__(
        self,
        image_size: tuple[int, int, int],
        *,
        gain_strength: float = 0.35,
        center_bias: float = 0.2,
    ) -> None:
        height, width, channels = (int(value) for value in image_size)
        self.image_size = (height, width, channels)
        self.gain_strength = float(max(0.0, gain_strength))
        self.center_bias = float(max(0.0, center_bias))

    @staticmethod
    def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.to(torch.float32)
        centered = scores.to(torch.float32) - scores.to(torch.float32).mean()
        scale = centered.std(unbiased=False).clamp_min(1e-4)
        return centered / scale

    def _center_score(self, top: int, left: int, patch_size: int) -> float:
        height, width, _ = self.image_size
        center_y = (top + (patch_size / 2.0)) / max(height, 1)
        center_x = (left + (patch_size / 2.0)) / max(width, 1)
        distance = (((center_y - 0.5) ** 2) + ((center_x - 0.5) ** 2)) ** 0.5
        return max(0.0, 1.0 - (distance / 0.7072))

    @torch.no_grad()
    def patch_gains(
        self,
        frame: torch.Tensor,
        positions: list[tuple[int, int]],
        *,
        patch_size: int,
    ) -> torch.Tensor:
        if not positions:
            return torch.empty(0, dtype=torch.float32, device=frame.device)

        scores: list[float] = []
        for top, left in positions:
            patch = frame[:, top : top + patch_size, left : left + patch_size].to(torch.float32)
            contrast = float(patch.std(unbiased=False).item())
            edges = 0.0
            if patch_size > 1:
                vertical = float((patch[:, 1:, :] - patch[:, :-1, :]).abs().mean().item())
                horizontal = float((patch[:, :, 1:] - patch[:, :, :-1]).abs().mean().item())
                edges = vertical + horizontal
            chroma = float(patch.mean(dim=(1, 2)).std(unbiased=False).item())
            center = self._center_score(top, left, patch_size)
            scores.append(contrast + (0.75 * edges) + (0.25 * chroma) + (self.center_bias * center))

        normalized = self._normalize_scores(torch.tensor(scores, dtype=torch.float32, device=frame.device))
        gains = 1.0 + (self.gain_strength * torch.tanh(normalized))
        return gains.clamp_(0.65, 1.5)

    @torch.no_grad()
    def apply_to_patches(
        self,
        patches: list[torch.Tensor],
        gains: torch.Tensor,
        *,
        sensory_dim: int,
    ) -> list[torch.Tensor]:
        if not patches or gains.numel() == 0:
            return [patch.to(torch.float32) for patch in patches]

        gated: list[torch.Tensor] = []
        for patch, gain in zip(patches, gains, strict=False):
            vector = patch.to(torch.float32).clone()
            vector[:sensory_dim] *= float(gain.item())
            gated.append(vector)
        return gated


__all__ = ["SpatialAttention"]
