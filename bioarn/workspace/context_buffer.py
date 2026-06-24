"""Extended working-memory buffer for longer-lived context maintenance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bioarn.core.math_utils import cosine_similarity


@dataclass
class BufferedConcept:
    """One concept retained in extended working memory."""

    concept: torch.Tensor
    strength: float
    age: int
    timestamp: int


class ContextBuffer:
    """Extended working memory beyond the limited-capacity GNW."""

    def __init__(
        self,
        buffer_size: int = 128,
        context_dim: int = 256,
        decay: float = 0.95,
        eviction_threshold: float = 0.05,
        ema_rate: float = 0.2,
        drift_window: int = 16,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1].")
        if not 0.0 <= eviction_threshold < 1.0:
            raise ValueError("eviction_threshold must be in [0, 1).")
        if not 0.0 < ema_rate <= 1.0:
            raise ValueError("ema_rate must be in (0, 1].")

        self.buffer_size = int(buffer_size)
        self.context_dim = int(context_dim)
        self.decay = float(decay)
        self.eviction_threshold = float(eviction_threshold)
        self.ema_rate = float(ema_rate)
        self.drift_window = max(2, int(drift_window))

        self.items: list[BufferedConcept] = []
        self.context_vector = torch.zeros(self.context_dim, dtype=torch.float32)
        self._recent_contexts: list[torch.Tensor] = []
        self._timestamp = 0

    @property
    def buffer(self) -> list[BufferedConcept]:
        """Alias for tests and call sites that prefer buffer terminology."""

        return self.items

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

    def _trim(self) -> None:
        self.items = [item for item in self.items if item.strength >= self.eviction_threshold]
        while len(self.items) > self.buffer_size:
            weakest_index = min(
                range(len(self.items)),
                key=lambda index: (self.items[index].strength, self.items[index].timestamp),
            )
            self.items.pop(weakest_index)

    def _summary_from_items(self) -> torch.Tensor:
        if not self.items:
            return torch.zeros(self.context_dim, dtype=self.context_vector.dtype, device=self.context_vector.device)

        concepts = torch.stack([item.concept.to(self.context_vector) for item in self.items], dim=0)
        recency = torch.tensor(
            [self.decay ** (len(self.items) - position - 1) for position in range(len(self.items))],
            dtype=concepts.dtype,
            device=concepts.device,
        )
        strengths = torch.tensor(
            [max(item.strength, 0.0) for item in self.items],
            dtype=concepts.dtype,
            device=concepts.device,
        )
        weights = (recency * strengths).clamp_min(1e-6)
        summary = (weights.unsqueeze(-1) * concepts).sum(dim=0) / weights.sum().clamp_min(1e-6)
        return self._normalize(summary)

    def _record_context(self, context_vector: torch.Tensor) -> None:
        self._recent_contexts.append(context_vector.detach().clone())
        if len(self._recent_contexts) > self.drift_window:
            self._recent_contexts = self._recent_contexts[-self.drift_window :]

    @torch.no_grad()
    def update(self, concept: torch.Tensor, strength: float) -> None:
        """Insert a concept, decay old items, and refresh the running context."""

        self._timestamp += 1
        normalized = self._normalize(self._align(concept)).to(self.context_vector)

        updated_items: list[BufferedConcept] = []
        for item in self.items:
            decayed_strength = float(item.strength) * self.decay
            if decayed_strength < self.eviction_threshold:
                continue
            updated_items.append(
                BufferedConcept(
                    concept=item.concept.to(self.context_vector),
                    strength=decayed_strength,
                    age=item.age + 1,
                    timestamp=item.timestamp,
                )
            )

        self.items = updated_items
        incoming_strength = max(0.0, float(strength))
        if incoming_strength > 0.0 and float(normalized.norm().item()) > 0.0:
            self.items.append(
                BufferedConcept(
                    concept=normalized.detach().clone(),
                    strength=incoming_strength,
                    age=0,
                    timestamp=self._timestamp,
                )
            )

        self._trim()
        summary = self._summary_from_items().to(self.context_vector)
        if float(self.context_vector.norm().item()) == 0.0:
            self.context_vector = summary.detach().clone()
        else:
            blended = ((1.0 - self.ema_rate) * self.context_vector) + (self.ema_rate * summary)
            self.context_vector = self._normalize(blended).detach().clone()
        self._record_context(self.context_vector)

    @torch.no_grad()
    def get_context_vector(self) -> torch.Tensor:
        """Return a recency-weighted summary of retained concepts."""

        if not self.items:
            return self.context_vector.detach().clone()
        summary = self._summary_from_items().to(self.context_vector)
        if float(self.context_vector.norm().item()) == 0.0:
            return summary.detach().clone()
        blended = self._normalize((0.65 * summary) + (0.35 * self.context_vector))
        return blended.detach().clone()

    @torch.no_grad()
    def attend(self, query: torch.Tensor, top_k: int = 5) -> list[tuple[torch.Tensor, float]]:
        """Retrieve the most query-relevant buffered concepts."""

        if top_k <= 0 or not self.items:
            return []

        normalized_query = self._normalize(self._align(query)).to(self.context_vector)
        if float(normalized_query.norm().item()) == 0.0:
            return []

        retrieved: list[tuple[torch.Tensor, float]] = []
        for position, item in enumerate(self.items):
            similarity = float(
                cosine_similarity(
                    item.concept.to(normalized_query).unsqueeze(0),
                    normalized_query.unsqueeze(0),
                ).item()
            )
            recency_weight = self.decay ** (len(self.items) - position - 1)
            score = similarity * (0.5 + 0.5 * min(1.0, item.strength)) * recency_weight
            retrieved.append((item.concept.detach().clone(), float(score)))

        retrieved.sort(key=lambda pair: pair[1], reverse=True)
        return retrieved[: min(top_k, len(retrieved))]

    @torch.no_grad()
    def get_topic_drift(self) -> float:
        """Estimate how quickly the retained context is changing."""

        if len(self._recent_contexts) < 2:
            return 0.0

        drifts: list[float] = []
        for previous, current in zip(self._recent_contexts[:-1], self._recent_contexts[1:], strict=False):
            if float(previous.norm().item()) <= 1e-8 or float(current.norm().item()) <= 1e-8:
                continue
            similarity = float(
                cosine_similarity(
                    previous.unsqueeze(0).to(current),
                    current.unsqueeze(0),
                ).item()
            )
            drifts.append(max(0.0, min(1.0, 1.0 - similarity)))

        if not drifts:
            return 0.0
        return float(sum(drifts) / len(drifts))

    @torch.no_grad()
    def clear(self) -> None:
        """Reset the buffer and its running summary."""

        self.items = []
        self.context_vector.zero_()
        self._recent_contexts = []
        self._timestamp = 0

