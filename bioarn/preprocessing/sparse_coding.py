"""Hebbian sparse coding for biologically plausible feature learning."""

from __future__ import annotations

import torch

from bioarn.core.math_utils import normalize


class HebbianSparseCoder:
    """Learn sparse overcomplete representations with competitive Hebbian updates."""

    def __init__(
        self,
        input_dim: int,
        num_features: int = 512,
        sparsity: float = 0.05,
        learning_rate: float = 0.01,
        *,
        anti_hebbian_rate: float | None = None,
        seed: int = 0,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("sparsity must be in (0, 1].")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        self.sparsity = float(sparsity)
        self.learning_rate = float(learning_rate)
        self.anti_hebbian_rate = float(
            anti_hebbian_rate if anti_hebbian_rate is not None else (learning_rate * 0.1)
        )
        self.seed = int(seed)
        self.k_active = max(1, int(round(self.num_features * self.sparsity)))
        self.is_fitted = False
        self.sample_count = 0
        self.mean_vector = torch.zeros(self.input_dim, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed)
        dictionary = torch.randn(
            self.num_features,
            self.input_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.dictionary = normalize(dictionary)

    def get_output_dim(self, input_dim: int) -> int:
        if int(input_dim) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, received {input_dim}.")
        return self.num_features

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            if x.numel() != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.numel()}.")
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"HebbianSparseCoder expects shape ({self.input_dim},) or (batch, {self.input_dim})."
            )
        return x.to(torch.float32), False

    def _center(self, batch: torch.Tensor) -> torch.Tensor:
        return batch - self.mean_vector.to(batch)

    def _encode_centered(self, centered: torch.Tensor) -> torch.Tensor:
        scores = centered @ self.dictionary.to(centered).transpose(0, 1)
        inhibited = scores - scores.mean(dim=-1, keepdim=True)
        threshold = (
            inhibited.abs().mean(dim=-1, keepdim=True)
            + inhibited.std(dim=-1, keepdim=True, unbiased=False)
        ).clamp_min(1e-6) * 0.2
        activations = torch.sign(inhibited) * torch.relu(inhibited.abs() - threshold)
        topk = min(self.k_active, activations.shape[-1])
        values, indices = torch.topk(activations.abs(), k=topk, dim=-1)
        selected = torch.gather(activations, 1, indices)
        code = torch.zeros_like(activations)
        code.scatter_(1, indices, selected)

        inactive_rows = code.count_nonzero(dim=-1) == 0
        if inactive_rows.any():
            fallback_scores = scores[inactive_rows]
            best_indices = fallback_scores.abs().argmax(dim=-1, keepdim=True)
            best_values = torch.gather(fallback_scores, 1, best_indices)
            code[inactive_rows].scatter_(1, best_indices, best_values)
        return code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        encoded = self._encode_centered(self._center(batch))
        return encoded.squeeze(0) if squeeze else encoded

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            if z.numel() != self.num_features:
                raise ValueError(f"Expected {self.num_features} code values, got {z.numel()}.")
            reconstructed = z.to(torch.float32) @ self.dictionary
            return reconstructed + self.mean_vector.to(reconstructed)
        if z.dim() != 2 or z.shape[-1] != self.num_features:
            raise ValueError(
                f"HebbianSparseCoder expects code shape ({self.num_features},) or (batch, {self.num_features})."
            )
        reconstructed = z.to(torch.float32) @ self.dictionary.to(z)
        return reconstructed + self.mean_vector.to(reconstructed)

    def get_dictionary(self) -> torch.Tensor:
        return self.dictionary.detach().clone()

    def reconstruction_error(self, x: torch.Tensor) -> float:
        batch, _ = self._ensure_batch(x)
        codes = self.transform(batch)
        reconstructions = self.reconstruct(codes)
        return float(torch.mean((reconstructions - batch) ** 2).item())

    @torch.no_grad()
    def learn(self, x: torch.Tensor) -> "HebbianSparseCoder":
        batch, _ = self._ensure_batch(x)
        for sample in batch:
            centered = sample - self.mean_vector
            code = self._encode_centered(centered.unsqueeze(0)).squeeze(0)
            active = torch.nonzero(code.abs() > 1e-6, as_tuple=False).flatten()
            if active.numel() > 0:
                weights = self.dictionary.index_select(0, active)
                reconstruction = code @ self.dictionary.to(code)
                residual = centered - reconstruction
                learning_signal = torch.tanh(code.index_select(0, active)).unsqueeze(1)
                updated = weights + (self.learning_rate * learning_signal * residual.unsqueeze(0))
                updated = updated - (
                    self.learning_rate
                    * 0.05
                    * learning_signal.square().clamp_max(1.0)
                    * weights
                )
                if active.numel() > 1:
                    coactivity = torch.outer(
                        code.index_select(0, active).abs(),
                        code.index_select(0, active).abs(),
                    )
                    coactivity.fill_diagonal_(0.0)
                    updated = updated - (self.anti_hebbian_rate * (coactivity @ updated))
                self.dictionary[active] = normalize(updated)
            self.sample_count += 1
            self.mean_vector += (sample - self.mean_vector) / float(self.sample_count)
            self.is_fitted = True
        return self

    def partial_fit(self, x: torch.Tensor) -> "HebbianSparseCoder":
        return self.learn(x)

    def fit(self, data: torch.Tensor) -> "HebbianSparseCoder":
        return self.learn(data)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


__all__ = ["HebbianSparseCoder"]
