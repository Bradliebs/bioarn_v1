"""Spike-compatible selective attention with competitive inhibition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bioarn.core.math_utils import cosine_similarity


class SpikeAttention:
    """Attention-like selection using raw vectors and winner-take-all dynamics."""

    def __init__(self, dim: int = 256, num_heads: int = 4) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.inhibition_strength = 0.25
        self.head_masks = self._build_masks()
        self.last_head_winners: list[int] = []

    def _build_masks(self) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        for head in range(self.num_heads):
            mask = torch.zeros(self.dim, dtype=torch.float32)
            mask[head :: self.num_heads] = 1.0
            if not torch.count_nonzero(mask):
                mask.fill_(1.0)
            masks.append(mask)
        return masks

    def _align(self, vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if flattened.numel() > self.dim:
            flattened = flattened[: self.dim]
        elif flattened.numel() < self.dim:
            flattened = F.pad(flattened, (0, self.dim - flattened.numel()))
        return flattened

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        norm = float(vector.norm().item())
        if norm <= 1e-8:
            return torch.zeros_like(vector)
        return vector / norm

    @torch.no_grad()
    def attend(self, query: torch.Tensor, keys: list[torch.Tensor]) -> tuple[torch.Tensor, list[float]]:
        """Select context keys via cosine similarity and lateral inhibition."""

        normalized_query = self._normalize(self._align(query))
        if not keys:
            return torch.zeros(self.dim, dtype=normalized_query.dtype, device=normalized_query.device), []

        key_matrix = torch.stack(
            [self._normalize(self._align(key)).to(normalized_query) for key in keys],
            dim=0,
        )
        aggregate = torch.zeros(len(keys), dtype=normalized_query.dtype, device=normalized_query.device)
        selected: set[int] = set()
        self.last_head_winners = []

        for head_index, base_mask in enumerate(self.head_masks):
            del head_index
            mask = base_mask.to(normalized_query)
            masked_query = normalized_query * mask
            if float(masked_query.norm().item()) <= 1e-8:
                masked_query = normalized_query
            masked_query = self._normalize(masked_query)

            masked_keys = key_matrix * mask.unsqueeze(0)
            if float(masked_keys.abs().sum().item()) <= 1e-8:
                masked_keys = key_matrix

            scores = cosine_similarity(
                masked_keys,
                masked_query.unsqueeze(0).expand_as(masked_keys),
            )
            if selected and len(selected) < len(keys):
                inhibition = torch.zeros_like(scores)
                inhibition[list(selected)] = self.inhibition_strength
                scores = scores - inhibition

            winner_index = int(torch.argmax(scores).item())
            winner_score = max(0.0, float(scores[winner_index].item()))
            if winner_score <= 0.0 and len(selected) >= len(keys):
                winner_score = max(0.0, float(cosine_similarity(key_matrix[winner_index], normalized_query).item()))

            aggregate[winner_index] += max(winner_score, 1e-6)
            self.last_head_winners.append(winner_index)
            if len(selected) < len(keys):
                selected.add(winner_index)

        if float(aggregate.sum().item()) <= 1e-8:
            best_index = int(
                torch.argmax(
                    cosine_similarity(
                        key_matrix,
                        normalized_query.unsqueeze(0).expand_as(key_matrix),
                    )
                ).item()
            )
            aggregate[best_index] = 1.0

        weights = aggregate / aggregate.sum().clamp_min(1e-6)
        attended = (weights.unsqueeze(-1) * key_matrix).sum(dim=0)
        attended = self._normalize(attended)
        return attended.detach().clone(), [float(weight) for weight in weights.tolist()]
