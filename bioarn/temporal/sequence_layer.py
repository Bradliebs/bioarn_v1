"""STDP-driven temporal sequence learning over concept activations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import STDPConfig, TemporalConfig
from bioarn.core.math_utils import normalize


@dataclass
class TemporalOutput:
    """Temporal prediction and surprise after observing one frame."""

    prediction: torch.Tensor
    predicted_indices: list[int]
    prior_prediction: torch.Tensor
    prior_predicted_indices: list[int]
    surprise: float
    actual_indices: list[int]


class TemporalSequenceLayer(nn.Module):
    """STDP-based temporal pattern learning across concept activations."""

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.context_window = int(config.context_window)
        self.concept_dim = int(config.concept_dim)
        self.stdp_config = STDPConfig(
            tau_plus=config.stdp_tau_plus,
            tau_minus=config.stdp_tau_minus,
            A_plus=config.stdp_lr,
            A_minus=config.stdp_lr,
        )
        self.register_buffer(
            "temporal_weights",
            torch.zeros(self.concept_dim, self.concept_dim, dtype=torch.float32),
        )
        self.register_buffer("last_prediction", torch.zeros(self.concept_dim, dtype=torch.float32))
        self._activation_history: deque[torch.Tensor] = deque(maxlen=self.context_window)
        self._index_history: deque[list[int]] = deque(maxlen=self.context_window)

    def _align(self, concept_activations: torch.Tensor) -> torch.Tensor:
        vector = concept_activations.detach().reshape(-1).to(
            device=self.temporal_weights.device,
            dtype=self.temporal_weights.dtype,
        )
        if vector.numel() > self.concept_dim:
            vector = vector[: self.concept_dim]
        elif vector.numel() < self.concept_dim:
            vector = F.pad(vector, (0, self.concept_dim - vector.numel()))
        return vector

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        if float(vector.norm().item()) <= 1e-8:
            return torch.zeros_like(vector)
        return normalize(vector.unsqueeze(0)).squeeze(0)

    def _sanitize_indices(self, indices: list[int]) -> list[int]:
        sanitized = {
            int(index)
            for index in indices
            if 0 <= int(index) < self.concept_dim
        }
        return sorted(sanitized)

    def _predicted_indices(self, prediction: torch.Tensor) -> list[int]:
        vector = prediction.detach().reshape(-1)
        if vector.numel() == 0 or float(vector.max().item()) <= 1e-8:
            return []
        normalized = vector / vector.max().clamp_min(1e-6)
        above = torch.nonzero(
            normalized >= float(self.config.prediction_threshold),
            as_tuple=False,
        ).reshape(-1)
        if above.numel() > 0:
            return [int(index) for index in above.tolist()]
        top_index = int(torch.argmax(normalized).item())
        return [top_index]

    @torch.no_grad()
    def predict_next(self) -> torch.Tensor:
        """Predict which concepts will fire in the next frame."""

        if not self._activation_history:
            return torch.zeros_like(self.last_prediction)

        history = list(self._activation_history)
        count = len(history)
        recency_weights = torch.tensor(
            [math.exp(-(count - position - 1)) for position in range(count)],
            device=self.temporal_weights.device,
            dtype=self.temporal_weights.dtype,
        )
        stacked = torch.stack(history, dim=0).to(self.temporal_weights)
        context = (recency_weights.unsqueeze(-1) * stacked).sum(dim=0)
        context = context / recency_weights.sum().clamp_min(1e-6)
        latest = stacked[-1]
        query = self._normalize((0.7 * latest.clamp_min(0.0)) + (0.3 * context.clamp_min(0.0)))
        positive_weights = self.temporal_weights.clamp_min(0.0)
        prediction = query @ positive_weights
        if float(prediction.max().item()) <= 1e-8:
            return torch.zeros_like(prediction)
        return prediction / prediction.max().clamp_min(1e-6)

    def temporal_surprise(self, actual_fired: list[int]) -> float:
        """How surprising was the current frame relative to the prior prediction?"""

        prediction = self.last_prediction.detach().reshape(-1).clamp_min(0.0)
        actual_indices = self._sanitize_indices(actual_fired)
        if prediction.numel() == 0 or float(prediction.max().item()) <= 1e-8:
            return 0.0 if not actual_indices else 1.0

        normalized = prediction / prediction.max().clamp_min(1e-6)
        predicted = set(self._predicted_indices(normalized))
        actual = set(actual_indices)
        if not predicted and not actual:
            return 0.0
        if not predicted or not actual:
            return 1.0

        actual_tensor = torch.tensor(actual_indices, device=normalized.device, dtype=torch.long)
        hit_strength = float(normalized.index_select(0, actual_tensor).mean().item())
        false_positive = [index for index in predicted if index not in actual]
        false_positive_strength = 0.0
        if false_positive:
            fp_tensor = torch.tensor(false_positive, device=normalized.device, dtype=torch.long)
            false_positive_strength = float(normalized.index_select(0, fp_tensor).mean().item())
        surprise = (0.8 * (1.0 - hit_strength)) + (0.2 * false_positive_strength)
        return float(max(0.0, min(1.0, surprise)))

    @torch.no_grad()
    def learn_stdp(self, pre_indices: list[int], post_indices: list[int], dt: float):
        """Apply pairwise STDP updates to the temporal transition matrix."""

        pre = self._sanitize_indices(pre_indices)
        post = self._sanitize_indices(post_indices)
        if not pre or not post or abs(float(dt)) <= 1e-8:
            return

        pre_tensor = torch.tensor(pre, device=self.temporal_weights.device, dtype=torch.long)
        post_tensor = torch.tensor(post, device=self.temporal_weights.device, dtype=torch.long)
        if dt > 0.0:
            delta = float(self.config.stdp_lr) * math.exp(
                -float(dt) / max(float(self.stdp_config.tau_plus), 1e-6)
            )
            self.temporal_weights[pre_tensor.unsqueeze(1), post_tensor.unsqueeze(0)] += delta
        else:
            delta = float(self.config.stdp_lr) * math.exp(
                float(dt) / max(float(self.stdp_config.tau_minus), 1e-6)
            )
            self.temporal_weights[pre_tensor.unsqueeze(1), post_tensor.unsqueeze(0)] -= delta

        self.temporal_weights.clamp_(-1.0, 1.0)
        self.temporal_weights.fill_diagonal_(0.0)

    @torch.no_grad()
    def observe_frame(
        self,
        concept_activations: torch.Tensor,
        fired_indices: list[int],
    ) -> TemporalOutput:
        """Process one frame and return a next-step prediction plus surprise."""

        aligned = self._normalize(self._align(concept_activations))
        actual_indices = self._sanitize_indices(fired_indices)
        prior_prediction = self.last_prediction.detach().clone()
        prior_predicted_indices = self._predicted_indices(prior_prediction)
        surprise = self.temporal_surprise(actual_indices)

        for distance, previous_indices in enumerate(reversed(self._index_history), start=1):
            dt = float(distance)
            self.learn_stdp(previous_indices, actual_indices, dt)

        self._activation_history.append(aligned.detach().clone())
        self._index_history.append(actual_indices)
        next_prediction = self.predict_next().detach().clone()
        self.last_prediction.copy_(next_prediction)
        return TemporalOutput(
            prediction=next_prediction,
            predicted_indices=self._predicted_indices(next_prediction),
            prior_prediction=prior_prediction,
            prior_predicted_indices=prior_predicted_indices,
            surprise=surprise,
            actual_indices=actual_indices,
        )

    @torch.no_grad()
    def reset_state(self, *, clear_weights: bool = False) -> None:
        """Reset short-term temporal state while optionally preserving learned weights."""

        self._activation_history.clear()
        self._index_history.clear()
        self.last_prediction.zero_()
        if clear_weights:
            self.temporal_weights.zero_()
