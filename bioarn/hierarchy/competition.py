"""Local competition helpers for the visual hierarchy."""

from __future__ import annotations

import torch

from bioarn.core.math_utils import cosine_similarity, normalize


class CompetitiveLateralInhibition:
    """Greedy winner selection that suppresses highly redundant concepts."""

    def __init__(self, *, similarity_threshold: float = 0.9) -> None:
        self.similarity_threshold = float(max(0.0, min(0.999, similarity_threshold)))

    @staticmethod
    def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
        if vectors.numel() == 0:
            return vectors.to(torch.float32)
        return normalize(vectors.to(torch.float32))

    @torch.no_grad()
    def select(
        self,
        candidate_indices: list[int],
        confidences: torch.Tensor,
        concept_directions: torch.Tensor,
        *,
        limit: int,
    ) -> tuple[list[int], torch.Tensor]:
        if not candidate_indices:
            return [], torch.empty(0, dtype=torch.float32, device=confidences.device)

        top_k = min(max(int(limit), 1), len(candidate_indices))
        ordered_positions = torch.argsort(confidences.to(torch.float32), descending=True).tolist()
        directions = self._normalize_rows(
            concept_directions.index_select(
                0,
                torch.tensor(candidate_indices, device=concept_directions.device, dtype=torch.long),
            )
        )

        selected_positions: list[int] = []
        for position in ordered_positions:
            if len(selected_positions) >= top_k:
                break
            if not selected_positions:
                selected_positions.append(int(position))
                continue

            candidate = directions[position].unsqueeze(0)
            winners = directions.index_select(
                0,
                torch.tensor(selected_positions, device=directions.device, dtype=torch.long),
            )
            overlaps = cosine_similarity(
                winners,
                candidate.expand_as(winners),
            )
            if bool((overlaps > self.similarity_threshold).any().item()):
                continue
            selected_positions.append(int(position))

        if not selected_positions:
            selected_positions = [int(ordered_positions[0])]

        selected_indices = [int(candidate_indices[position]) for position in selected_positions]
        selected_confidences = confidences.index_select(
            0,
            torch.tensor(selected_positions, device=confidences.device, dtype=torch.long),
        ).to(torch.float32)
        return selected_indices, selected_confidences


__all__ = ["CompetitiveLateralInhibition"]
