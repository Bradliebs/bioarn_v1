"""Auditory hierarchy for Bio-ARN audio inputs."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import AudioHierarchyConfig


class AudioHierarchy(nn.Module):
    """Frozen A1 → A2 → Belt hierarchy over mel spectrograms."""

    def __init__(self, config: AudioHierarchyConfig):
        super().__init__()
        self.config = replace(config)
        self.num_bands = max(1, min(8, int(self.config.n_mels)))
        generator = torch.Generator().manual_seed(int(self.config.init_seed))

        a1_weights = torch.randn(
            self.config.a1_channels,
            1,
            self.config.temporal_kernel,
            generator=generator,
            dtype=torch.float32,
        )
        a1_weights = F.normalize(a1_weights.reshape(self.config.a1_channels, -1), dim=1).reshape_as(a1_weights)
        self.register_buffer("a1_weights", a1_weights)
        self.register_buffer(
            "a1_bias",
            torch.linspace(-0.15, 0.15, self.config.a1_channels, dtype=torch.float32),
        )

        a2_in_channels = self.num_bands * 3
        a2_weights = torch.randn(
            self.config.a2_channels,
            a2_in_channels,
            3,
            generator=generator,
            dtype=torch.float32,
        )
        a2_weights = F.normalize(a2_weights.reshape(self.config.a2_channels, -1), dim=1).reshape_as(a2_weights)
        self.register_buffer("a2_weights", a2_weights)
        self.register_buffer(
            "a2_bias",
            torch.linspace(-0.1, 0.1, self.config.a2_channels, dtype=torch.float32),
        )

        belt_gain = 0.75 + (0.5 * torch.rand(self.config.belt_dim, generator=generator, dtype=torch.float32))
        self.register_buffer("belt_gain", belt_gain)
        self.register_buffer("belt_bias", torch.zeros(self.config.belt_dim, dtype=torch.float32))

    @property
    def output_dim(self) -> int:
        return int(self.config.belt_dim)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Encode ``(n_mels, frames)`` or ``(batch, n_mels, frames)`` into Belt concepts."""

        x = mel_spectrogram.to(torch.float32)
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3 or x.shape[1] != self.config.n_mels:
            raise ValueError("mel_spectrogram must have shape (n_mels, frames) or (batch, n_mels, frames).")

        batch_size, _, frames = x.shape
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)

        a1 = F.conv1d(
            x.reshape(batch_size * self.config.n_mels, 1, frames),
            self.a1_weights,
            self.a1_bias,
            padding=self.config.temporal_kernel // 2,
        )
        a1 = F.gelu(a1)
        a1 = a1.reshape(batch_size, self.config.n_mels, self.config.a1_channels, frames).permute(0, 2, 1, 3)

        raw_bands = F.adaptive_avg_pool2d(x.unsqueeze(1), (self.num_bands, frames)).squeeze(1)
        a1_mean = F.adaptive_avg_pool2d(a1.mean(dim=1, keepdim=True), (self.num_bands, frames)).squeeze(1)
        a1_energy = F.adaptive_max_pool2d(a1.abs().mean(dim=1, keepdim=True), (self.num_bands, frames)).squeeze(1)
        a2_input = torch.cat([raw_bands, a1_mean, a1_energy], dim=1)
        a2 = F.conv1d(a2_input, self.a2_weights, self.a2_bias, padding=1)
        a2 = F.gelu(a2)

        belt_source = torch.cat(
            [
                raw_bands.reshape(batch_size, -1),
                a1_mean.reshape(batch_size, -1),
                a1_energy.reshape(batch_size, -1),
                a2.reshape(batch_size, -1),
            ],
            dim=1,
        )
        belt = F.adaptive_avg_pool1d(belt_source.unsqueeze(1), self.config.belt_dim).squeeze(1)
        belt = torch.tanh((belt * self.belt_gain) + self.belt_bias)
        belt = F.normalize(belt, dim=-1, eps=1e-6)
        return belt.squeeze(0) if squeeze else belt


__all__ = ["AudioHierarchy"]
