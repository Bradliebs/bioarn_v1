"""Margin gate with honest abstention and resonance checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bioarn.config import MarginGateConfig
from bioarn.core.math_utils import cosine_similarity


@dataclass
class MarginGateOutput:
    """Output of the margin gate."""

    output: torch.Tensor
    confidence: torch.Tensor
    fired: torch.Tensor
    abstained: torch.Tensor


@dataclass
class ResonanceOutput:
    """Output of the resonance check."""

    match_score: torch.Tensor
    resonated: torch.Tensor
    learn_signal: torch.Tensor


class MarginGate(nn.Module):
    """Honest abstention gate based on cosine similarity margins."""

    def __init__(self, config: MarginGateConfig):
        super().__init__()
        self.theta_margin_lr = float(config.theta_margin_lr)

        self.register_buffer(
            "theta_margin",
            torch.tensor(float(config.theta_margin), dtype=torch.float32),
        )
        self.register_buffer(
            "theta_resonance",
            torch.tensor(float(config.theta_resonance), dtype=torch.float32),
        )

        self.register_buffer("total_presentations", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_fires", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_abstentions", torch.tensor(0, dtype=torch.long))
        self.register_buffer("fire_rate", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "avg_confidence_when_fired",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "avg_confidence_when_abstained",
            torch.tensor(0.0, dtype=torch.float32),
        )

    def forward(
        self,
        input_activation: torch.Tensor,
        concept_direction: torch.Tensor,
    ) -> MarginGateOutput:
        """Gate activations using a cosine-similarity threshold."""
        confidence = cosine_similarity(input_activation, concept_direction)
        fired = confidence > self.theta_margin
        abstained = ~fired
        output = torch.where(
            fired.unsqueeze(-1),
            input_activation,
            torch.zeros_like(input_activation),
        )

        self._update_stats(confidence=confidence, fired=fired)

        return MarginGateOutput(
            output=output,
            confidence=confidence,
            fired=fired,
            abstained=abstained,
        )

    @torch.no_grad()
    def adapt_threshold(self, recent_fire_rate: float) -> None:
        """Adapt the margin threshold to keep firing selective."""
        if recent_fire_rate > 0.8:
            self.theta_margin.add_(self.theta_margin_lr)
        elif recent_fire_rate < 0.05:
            self.theta_margin.sub_(self.theta_margin_lr)

        self.theta_margin.clamp_(0.1, 0.95)

    def check_resonance(
        self,
        prediction: torch.Tensor,
        actual_input: torch.Tensor,
    ) -> ResonanceOutput:
        """Check whether a prediction resonates strongly enough to learn from."""
        match_score = cosine_similarity(prediction, actual_input)
        resonated = match_score > self.theta_resonance
        learn_signal = torch.where(
            resonated,
            ((match_score - self.theta_resonance) / (1.0 - self.theta_resonance)).clamp(
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(match_score),
        )

        return ResonanceOutput(
            match_score=match_score,
            resonated=resonated,
            learn_signal=learn_signal,
        )

    @torch.no_grad()
    def get_stats(self) -> dict[str, float | int]:
        """Return running gate statistics as Python scalars."""
        return {
            "theta_margin": float(self.theta_margin.item()),
            "theta_resonance": float(self.theta_resonance.item()),
            "total_presentations": int(self.total_presentations.item()),
            "total_fires": int(self.total_fires.item()),
            "total_abstentions": int(self.total_abstentions.item()),
            "fire_rate": float(self.fire_rate.item()),
            "avg_confidence_when_fired": float(
                self.avg_confidence_when_fired.item()
            ),
            "avg_confidence_when_abstained": float(
                self.avg_confidence_when_abstained.item()
            ),
        }

    @torch.no_grad()
    def _update_stats(self, confidence: torch.Tensor, fired: torch.Tensor) -> None:
        batch_confidence = confidence.reshape(-1).detach()
        batch_fired = fired.reshape(-1).detach()

        batch_size = batch_fired.numel()
        fired_count = int(batch_fired.sum().item())
        abstained_count = batch_size - fired_count

        prev_fires = int(self.total_fires.item())
        prev_abstentions = int(self.total_abstentions.item())

        if fired_count:
            fired_confidences = batch_confidence[batch_fired]
            fired_total = prev_fires + fired_count
            fired_average = (
                self.avg_confidence_when_fired * prev_fires
                + fired_confidences.sum().to(self.avg_confidence_when_fired.dtype)
            ) / fired_total
            self.avg_confidence_when_fired.copy_(fired_average)

        if abstained_count:
            abstained_confidences = batch_confidence[~batch_fired]
            abstained_total = prev_abstentions + abstained_count
            abstained_average = (
                self.avg_confidence_when_abstained * prev_abstentions
                + abstained_confidences.sum().to(
                    self.avg_confidence_when_abstained.dtype
                )
            ) / abstained_total
            self.avg_confidence_when_abstained.copy_(abstained_average)

        self.total_presentations.add_(batch_size)
        self.total_fires.add_(fired_count)
        self.total_abstentions.add_(abstained_count)
        self.fire_rate.copy_(
            self.total_fires.float() / self.total_presentations.clamp_min(1).float()
        )


__all__ = ["MarginGate", "MarginGateOutput", "ResonanceOutput"]
