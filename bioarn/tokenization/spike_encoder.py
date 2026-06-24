"""Token-to-spike encoding bridge for Bio-ARN."""

from __future__ import annotations

import math

import torch
from torch import nn

from bioarn.core.spiking import rate_encode


class DecodedTokenMatch(int):
    """Integer token id with an attached confidence score."""

    def __new__(cls, token_id: int, confidence: float) -> "DecodedTokenMatch":
        match = int.__new__(cls, token_id)
        match.confidence = float(confidence)
        return match


class SpikeTokenEncoder(nn.Module):
    """Convert token IDs to spike patterns for Bio-ARN processing."""

    def __init__(self, vocab_size: int, spike_dim: int = 256, num_timesteps: int = 8) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if spike_dim <= 0 or num_timesteps <= 0:
            raise ValueError("spike_dim and num_timesteps must be positive.")

        self.vocab_size = int(vocab_size)
        self.spike_dim = int(spike_dim)
        self.num_timesteps = int(num_timesteps)
        self.sparsity = min(0.2, max(0.1, 0.15))

        token_patterns = self._build_token_patterns()
        rate_templates = token_patterns * 0.85 + (1.0 - token_patterns) * 0.15

        self.register_buffer("token_patterns", token_patterns)
        self.register_buffer("rate_templates", rate_templates)

    def encode_token(self, token_id: int) -> torch.Tensor:
        """Map a token id to its fixed spike pattern."""

        self._validate_token_id(token_id)
        return self.token_patterns[token_id].clone()

    def encode_sequence(self, token_ids: list[int]) -> torch.Tensor:
        """Encode a token sequence into a temporal spike train."""

        if not token_ids:
            return torch.zeros((0, self.spike_dim), dtype=self.token_patterns.dtype, device=self.token_patterns.device)

        blocks = [self._temporal_block(token_id) for token_id in token_ids]
        return torch.cat(blocks, dim=0)

    def encode_to_rate(self, token_ids: list[int]) -> torch.Tensor:
        """Rate-code tokens using the core Bernoulli spike encoder."""

        if not token_ids:
            return torch.zeros((0, self.spike_dim), dtype=self.token_patterns.dtype, device=self.token_patterns.device)

        blocks = []
        for token_id in token_ids:
            self._validate_token_id(token_id)
            rate_block = rate_encode(self.rate_templates[token_id], num_steps=self.num_timesteps)
            blocks.append(rate_block)
        return torch.cat(blocks, dim=0)

    def decode_spikes(self, spike_pattern: torch.Tensor) -> DecodedTokenMatch:
        """Decode a spike pattern to the nearest token id using Hamming distance."""

        if spike_pattern.dim() != 1 or spike_pattern.shape[0] != self.spike_dim:
            raise ValueError(f"spike_pattern must have shape ({self.spike_dim},).")

        binary_pattern = (spike_pattern.to(self.token_patterns.device) > 0.5).to(self.token_patterns.dtype)
        distances = (self.token_patterns != binary_pattern.unsqueeze(0)).to(torch.float32).sum(dim=1)
        best_index = int(torch.argmin(distances).item())
        confidence = 1.0 - float(distances[best_index].item() / self.spike_dim)
        return DecodedTokenMatch(best_index, confidence)

    def _validate_token_id(self, token_id: int) -> None:
        if not 0 <= int(token_id) < self.vocab_size:
            raise ValueError(f"token_id must be in [0, {self.vocab_size}).")

    def _build_token_patterns(self) -> torch.Tensor:
        active_count = max(1, math.ceil(self.spike_dim * self.sparsity))
        patterns = torch.zeros((self.vocab_size, self.spike_dim), dtype=torch.float32)

        for token_id in range(self.vocab_size):
            best_pattern = None
            best_score = float("inf")

            for attempt in range(8):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(17_171 + (token_id * 7_919) + attempt)
                candidate = torch.zeros(self.spike_dim, dtype=torch.float32)
                active_indices = torch.randperm(self.spike_dim, generator=generator)[:active_count]
                candidate[active_indices] = 1.0

                if token_id == 0:
                    best_pattern = candidate
                    break

                overlaps = torch.matmul(patterns[:token_id], candidate)
                score = float(overlaps.max().item() * 2.0 + overlaps.mean().item())
                if score < best_score:
                    best_score = score
                    best_pattern = candidate

            patterns[token_id] = best_pattern if best_pattern is not None else candidate

        return patterns

    def _temporal_block(self, token_id: int) -> torch.Tensor:
        self._validate_token_id(token_id)
        base_pattern = self.token_patterns[token_id]
        block = torch.zeros(
            (self.num_timesteps, self.spike_dim),
            dtype=self.token_patterns.dtype,
            device=self.token_patterns.device,
        )

        if self.num_timesteps == 1:
            block[0] = base_pattern
            return block

        block[0] = base_pattern
        active_indices = torch.nonzero(base_pattern, as_tuple=False).flatten()
        half_count = max(1, active_indices.numel() // 2)

        for step in range(1, self.num_timesteps - 1):
            start = (step - 1) % active_indices.numel()
            sustain_indices = active_indices.roll(-start)[:half_count]
            block[step, sustain_indices] = max(0.25, 0.6 - (0.1 * (step - 1)))

        shifted = torch.roll(base_pattern, shifts=(token_id % max(1, self.spike_dim - 1)) + 1, dims=0)
        block[-1] = torch.maximum(base_pattern * 0.2, shifted * 0.8)
        return block


__all__ = ["DecodedTokenMatch", "SpikeTokenEncoder"]

