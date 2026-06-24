"""Competitive feature learning for self-organized prototype discovery."""

from __future__ import annotations

import torch

from bioarn.core.math_utils import normalize


class CompetitiveLearner:
    """Competitive codebook learning with optional neighborhood updates."""

    def __init__(
        self,
        input_dim: int,
        num_neurons: int = 128,
        learning_rate: float = 0.01,
        neighborhood: bool = False,
        *,
        sigma: float = 2.0,
        seed: int = 0,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.input_dim = int(input_dim)
        self.num_neurons = int(num_neurons)
        self.learning_rate = float(learning_rate)
        self.neighborhood = bool(neighborhood)
        self.sigma = float(max(sigma, 1e-3))
        self.seed = int(seed)
        self.is_fitted = False
        self.sample_count = 0
        self.mean_vector = torch.zeros(self.input_dim, dtype=torch.float32)
        self.activation_counts = torch.zeros(self.num_neurons, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        weights = torch.randn(
            self.num_neurons,
            self.input_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.weights = normalize(weights)

    def get_output_dim(self, input_dim: int) -> int:
        if int(input_dim) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, received {input_dim}.")
        return self.num_neurons

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            if x.numel() != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.numel()}.")
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"CompetitiveLearner expects shape ({self.input_dim},) or (batch, {self.input_dim})."
            )
        return x.to(torch.float32), False

    def _center(self, batch: torch.Tensor) -> torch.Tensor:
        return batch - self.mean_vector.to(batch)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        centered = self._center(batch)
        scores = centered @ self.weights.to(centered).transpose(0, 1)
        winners = scores.argmax(dim=-1, keepdim=True)
        codes = torch.zeros_like(scores)
        if self.neighborhood:
            neuron_positions = torch.arange(self.num_neurons, device=scores.device, dtype=torch.float32)
            for row_index, winner in enumerate(winners.squeeze(-1).tolist()):
                distance = (neuron_positions - float(winner)).abs()
                neighborhood = torch.exp(-(distance.square()) / (2.0 * (self.sigma**2)))
                codes[row_index] = scores[row_index] * neighborhood
        else:
            winner_scores = torch.gather(scores, 1, winners)
            codes.scatter_(1, winners, winner_scores)
        return codes.squeeze(0) if squeeze else codes

    def reconstruct(self, code: torch.Tensor) -> torch.Tensor:
        if code.dim() == 1:
            if code.numel() != self.num_neurons:
                raise ValueError(f"Expected {self.num_neurons} code values, got {code.numel()}.")
            return (code.to(torch.float32) @ self.weights) + self.mean_vector.to(code)
        if code.dim() != 2 or code.shape[-1] != self.num_neurons:
            raise ValueError(
                f"CompetitiveLearner expects code shape ({self.num_neurons},) or (batch, {self.num_neurons})."
            )
        return (code.to(torch.float32) @ self.weights.to(code)) + self.mean_vector.to(code)

    def get_codebook(self) -> torch.Tensor:
        return self.weights.detach().clone()

    def reconstruction_error(self, x: torch.Tensor) -> float:
        batch, _ = self._ensure_batch(x)
        codes = self.transform(batch)
        reconstructions = self.reconstruct(codes)
        return float(torch.mean((reconstructions - batch) ** 2).item())

    @torch.no_grad()
    def partial_fit(self, x: torch.Tensor) -> "CompetitiveLearner":
        batch, _ = self._ensure_batch(x)
        neuron_positions = torch.arange(self.num_neurons, dtype=torch.float32)
        for sample in batch:
            centered = sample - self.mean_vector
            sample_direction = normalize(centered.unsqueeze(0)).squeeze(0)
            scores = self.weights.to(centered) @ centered
            usage_penalty = 0.1 * (
                self.activation_counts.to(torch.float32) / max(self.sample_count + 1, 1)
            )
            winner = int((scores - usage_penalty.to(scores)).argmax().item())
            self.activation_counts[winner] += 1
            if self.neighborhood:
                distance = (neuron_positions - float(winner)).abs()
                factors = torch.exp(-(distance.square()) / (2.0 * (self.sigma**2))).unsqueeze(1)
                updated = self.weights + (
                    self.learning_rate * factors.to(self.weights) * (sample_direction.unsqueeze(0) - self.weights)
                )
                self.weights = normalize(updated)
            else:
                updated = self.weights[winner] + (self.learning_rate * (sample_direction - self.weights[winner]))
                self.weights[winner] = normalize(updated.unsqueeze(0)).squeeze(0)
            self.sample_count += 1
            self.mean_vector += (sample - self.mean_vector) / float(self.sample_count)
            self.is_fitted = True
        return self

    def fit(self, data: torch.Tensor) -> "CompetitiveLearner":
        return self.partial_fit(data)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


__all__ = ["CompetitiveLearner"]
