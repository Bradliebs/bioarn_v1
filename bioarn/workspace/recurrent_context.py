"""Recurrent context integration without backpropagation through time."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bioarn.core.math_utils import cosine_similarity


class RecurrentContext:
    """Integrate concepts across time for coherent generation."""

    def __init__(self, context_dim: int = 256, integration_rate: float = 0.1) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if not 0.0 < integration_rate <= 1.0:
            raise ValueError("integration_rate must be in (0, 1].")

        self.context_dim = int(context_dim)
        self.integration_rate = float(integration_rate)
        self.state = torch.zeros(self.context_dim, dtype=torch.float32)
        self.history: list[torch.Tensor] = []

    def _align(self, concept: torch.Tensor) -> torch.Tensor:
        vector = concept.detach().reshape(-1).to(torch.float32)
        if vector.numel() > self.context_dim:
            vector = vector[: self.context_dim]
        elif vector.numel() < self.context_dim:
            vector = F.pad(vector, (0, self.context_dim - vector.numel()))
        return vector

    @staticmethod
    def _normalize(concept: torch.Tensor) -> torch.Tensor:
        norm = float(concept.norm().item())
        if norm <= 1e-8:
            return torch.zeros_like(concept)
        return concept / norm

    @torch.no_grad()
    def integrate(self, new_concept: torch.Tensor) -> torch.Tensor:
        """Leaky-integrate a concept into the running context state."""

        aligned = self._normalize(self._align(new_concept)).to(self.state)
        if float(self.state.norm().item()) == 0.0:
            self.state = aligned.detach().clone()
        else:
            carry = 1.0 - self.integration_rate
            blended = (carry * self.state) + (self.integration_rate * aligned)
            self.state = self._normalize(blended).detach().clone()
        self.history.append(self.state.detach().clone())
        return self.state.detach().clone()

    @torch.no_grad()
    def prime_retrieval(self, context: torch.Tensor, candidate_scores: torch.Tensor) -> torch.Tensor:
        """Bias candidate retrieval toward context-consistent continuations."""

        aligned_context = self._normalize(self._align(context)).to(candidate_scores)
        if float(aligned_context.norm().item()) == 0.0:
            return candidate_scores.detach().clone()

        if candidate_scores.dim() == 2:
            candidates = candidate_scores.detach().to(torch.float32)
            if candidates.shape[-1] != self.context_dim:
                if candidates.shape[-1] > self.context_dim:
                    candidates = candidates[..., : self.context_dim]
                else:
                    candidates = F.pad(candidates, (0, self.context_dim - candidates.shape[-1]))
            normalized_candidates = candidates / candidates.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return cosine_similarity(
                normalized_candidates,
                aligned_context.unsqueeze(0).expand_as(normalized_candidates),
            ).detach().clone()

        centered = candidate_scores.detach().to(torch.float32)
        modulation = torch.relu(centered - centered.mean())
        gain = min(0.5, 0.1 + float(aligned_context.abs().mean().item()))
        return (centered + (gain * modulation)).detach().clone()

    @torch.no_grad()
    def detect_repetition(self, context_history: list[torch.Tensor], window: int = 20) -> float:
        """Return a high score when context trajectories are cycling."""

        if window <= 1 or len(context_history) < 4:
            return 0.0

        recent = [
            self._normalize(self._align(vector)).to(self.state)
            for vector in context_history[-max(4, int(window)) :]
            if float(vector.detach().reshape(-1).norm().item()) > 0.0
        ]
        if len(recent) < 4:
            return 0.0

        max_lag = min(len(recent) // 2, 8)
        best_cycle = 0.0
        for lag in range(1, max_lag + 1):
            similarities: list[float] = []
            for index in range(lag, len(recent)):
                similarities.append(
                    float(
                        cosine_similarity(
                            recent[index].unsqueeze(0),
                            recent[index - lag].unsqueeze(0),
                        ).item()
                    )
                )
            if similarities:
                best_cycle = max(best_cycle, sum(similarities) / len(similarities))
        return float(max(0.0, min(1.0, best_cycle)))

    @torch.no_grad()
    def reset(self) -> None:
        """Clear the recurrent state."""

        self.state.zero_()
        self.history = []

