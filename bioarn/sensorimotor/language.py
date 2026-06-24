"""Language sensory stream with spike-based temporal encoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from bioarn.config import PredictiveConfig, SpikingConfig
from bioarn.core.math_utils import sparse_top_k
from bioarn.core.spiking import LIFLayer, rate_encode


@dataclass
class LanguageOutput:
    """Outputs from the language sensory stream."""

    features: torch.Tensor
    spike_train: torch.Tensor
    suppressed_fraction: float


class LanguageEncoder(nn.Module):
    """Character/token encoder using spike trains and predictive suppression."""

    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int, config: SpikingConfig):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if embedding_dim <= 0 or output_dim <= 0:
            raise ValueError("embedding_dim and output_dim must be positive.")

        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim)
        self.config = config
        self.predictive_config = PredictiveConfig()
        self.steps_per_token = max(6, min(16, embedding_dim))
        self.output_top_k = max(1, self.output_dim // 20)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.temporal_layer = LIFLayer(
            self.embedding_dim,
            self.output_dim,
            bias=False,
            config=config,
            spike_history_steps=self.steps_per_token,
        )

        self.register_buffer("last_rate_pattern", torch.empty(0))
        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self) -> None:
        token_axis = torch.arange(self.vocab_size, dtype=self.embedding.weight.dtype).unsqueeze(1)
        feature_axis = torch.arange(self.embedding_dim, dtype=self.embedding.weight.dtype).unsqueeze(0)
        weights = (
            torch.sin((token_axis + 1.0) * (feature_axis + 1.0) * 0.17)
            + torch.cos((token_axis + 1.0) * (feature_axis + 1.0) * 0.11)
        ) * 0.5
        self.embedding.weight.copy_(weights)

        projection = torch.empty_like(self.temporal_layer.linear.weight)
        gain = max(1.25, min(2.5, math.sqrt(self.embedding_dim) / 2.0))
        for idx in range(self.output_dim):
            row = torch.sin(
                torch.linspace(
                    0.0,
                    math.pi * (idx + 1),
                    self.embedding_dim,
                    device=projection.device,
                    dtype=projection.dtype,
                )
            )
            projection[idx] = row
        self.temporal_layer.linear.weight.copy_(projection * gain)

    def reset_state(self) -> None:
        """Reset temporal spiking state and predictive carry-over."""
        self.temporal_layer.reset_state()
        self.last_rate_pattern = torch.empty(
            0,
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )

    def _prepare_inputs(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if token_ids.dim() == 1:
            return token_ids.unsqueeze(0).long(), True
        if token_ids.dim() == 2:
            return token_ids.long(), False
        raise ValueError("token_ids must have shape (seq_len,) or (batch, seq_len).")

    def _predict_rates(self, rates: torch.Tensor) -> torch.Tensor:
        predicted = torch.zeros_like(rates)
        if rates.shape[1] > 1:
            predicted[:, 1:] = rates[:, :-1].detach()
        if self.last_rate_pattern.numel() == self.embedding_dim:
            predicted[:, 0] = self.last_rate_pattern.to(device=rates.device, dtype=rates.dtype).unsqueeze(0)
        return predicted

    def _encode_spike_blocks(self, rates: torch.Tensor) -> list[torch.Tensor]:
        return [rate_encode(rates[:, idx], num_steps=self.steps_per_token) for idx in range(rates.shape[1])]

    def forward(self, token_ids: torch.Tensor) -> LanguageOutput:
        token_batch, squeeze = self._prepare_inputs(token_ids)
        embeddings = self.embedding(token_batch)
        rates = torch.sigmoid(embeddings)
        predicted_rates = self._predict_rates(rates)
        surprise_mask = (rates - predicted_rates).abs() > self.predictive_config.error_threshold

        spike_blocks = self._encode_spike_blocks(rates)
        spike_train = torch.cat(spike_blocks, dim=0)

        suppressed_blocks = [
            block * surprise_mask[:, idx].unsqueeze(0).to(block.dtype)
            for idx, block in enumerate(spike_blocks)
        ]
        suppressed_train = torch.cat(suppressed_blocks, dim=0)

        self.temporal_layer.reset_state()
        temporal_spikes, _ = self.temporal_layer(suppressed_train)
        features = sparse_top_k(temporal_spikes.sum(dim=0), k=self.output_top_k)

        with torch.no_grad():
            self.last_rate_pattern = rates[:, -1].mean(dim=0).detach().clone()

        return LanguageOutput(
            features=features.squeeze(0) if squeeze and features.shape[0] == 1 else features,
            spike_train=spike_train,
            suppressed_fraction=float((~surprise_mask).float().mean().item()),
        )

    def encode_text(self, text: str, char_to_idx: dict) -> LanguageOutput:
        """Convenience wrapper for raw character strings."""
        if not text:
            raise ValueError("text must be non-empty.")
        token_ids = torch.tensor([char_to_idx.get(char, 0) for char in text], dtype=torch.long)
        return self.forward(token_ids.unsqueeze(0))


__all__ = ["LanguageEncoder", "LanguageOutput"]
