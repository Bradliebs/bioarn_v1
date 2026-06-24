"""Online dictionary learning with Hebbian-style updates and matching pursuit."""

from __future__ import annotations

import torch

from bioarn.core.math_utils import normalize


class OnlineDictionaryLearner:
    """Learn a sparse dictionary incrementally from streaming samples."""

    def __init__(
        self,
        input_dim: int,
        dict_size: int = 256,
        sparsity_target: float = 0.1,
        *,
        learning_rate: float = 0.02,
        max_matching_iters: int = 20,
        seed: int = 0,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if dict_size <= 0:
            raise ValueError("dict_size must be positive.")
        if not 0.0 < sparsity_target <= 1.0:
            raise ValueError("sparsity_target must be in (0, 1].")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.input_dim = int(input_dim)
        self.dict_size = int(dict_size)
        self.sparsity_target = float(sparsity_target)
        self.learning_rate = float(learning_rate)
        self.max_matching_iters = int(max(1, max_matching_iters))
        self.seed = int(seed)
        self.max_active = max(1, int(round(self.dict_size * self.sparsity_target)))
        self.is_fitted = False
        self.sample_count = 0
        self.mean_vector = torch.zeros(self.input_dim, dtype=torch.float32)
        self.update_ema = 0.0
        generator = torch.Generator().manual_seed(self.seed)
        dictionary = torch.randn(
            self.dict_size,
            self.input_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.dictionary = normalize(dictionary)

    def get_output_dim(self, input_dim: int) -> int:
        if int(input_dim) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, received {input_dim}.")
        return self.dict_size

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            if x.numel() != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.numel()}.")
            return x.unsqueeze(0).to(torch.float32), True
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"OnlineDictionaryLearner expects shape ({self.input_dim},) or (batch, {self.input_dim})."
            )
        return x.to(torch.float32), False

    def _center(self, batch: torch.Tensor) -> torch.Tensor:
        return batch - self.mean_vector.to(batch)

    def matching_pursuit(self, x: torch.Tensor, max_iters: int | None = None) -> torch.Tensor:
        if x.dim() != 1 or x.numel() != self.input_dim:
            raise ValueError(f"matching_pursuit expects a vector of shape ({self.input_dim},).")
        centered = x.to(torch.float32) - self.mean_vector.to(x)
        residual = centered.clone()
        code = torch.zeros(self.dict_size, device=centered.device, dtype=torch.float32)
        max_steps = min(self.max_matching_iters if max_iters is None else int(max_iters), self.max_active)
        for _ in range(max_steps):
            scores = self.dictionary.to(centered) @ residual
            best = int(scores.abs().argmax().item())
            coefficient = scores[best]
            if float(coefficient.abs().item()) < 1e-6:
                break
            code[best] += coefficient
            residual = residual - (coefficient * self.dictionary[best].to(centered))
            if float(residual.norm().item()) < 0.01:
                break
        return code

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = self._ensure_batch(x)
        codes = torch.stack([self.matching_pursuit(sample) for sample in batch], dim=0)
        return codes.squeeze(0) if squeeze else codes

    def reconstruct(self, code: torch.Tensor) -> torch.Tensor:
        if code.dim() == 1:
            if code.numel() != self.dict_size:
                raise ValueError(f"Expected {self.dict_size} code values, got {code.numel()}.")
            return (code.to(torch.float32) @ self.dictionary) + self.mean_vector.to(code)
        if code.dim() != 2 or code.shape[-1] != self.dict_size:
            raise ValueError(
                f"OnlineDictionaryLearner expects code shape ({self.dict_size},) or (batch, {self.dict_size})."
            )
        return (code.to(torch.float32) @ self.dictionary.to(code)) + self.mean_vector.to(code)

    def get_dictionary(self) -> torch.Tensor:
        return self.dictionary.detach().clone()

    def reconstruction_error(self, x: torch.Tensor) -> float:
        batch, _ = self._ensure_batch(x)
        codes = self.transform(batch)
        reconstructions = self.reconstruct(codes)
        return float(torch.mean((reconstructions - batch) ** 2).item())

    @torch.no_grad()
    def partial_fit(self, x: torch.Tensor) -> "OnlineDictionaryLearner":
        batch, _ = self._ensure_batch(x)
        for sample in batch:
            centered = sample - self.mean_vector
            code = self.matching_pursuit(sample)
            active = torch.nonzero(code.abs() > 1e-6, as_tuple=False).flatten()
            if active.numel() > 0:
                weights = self.dictionary.index_select(0, active)
                coeffs = torch.tanh(code.index_select(0, active)).unsqueeze(1)
                reconstruction = code @ self.dictionary.to(code)
                residual = centered - reconstruction
                updated = weights + (self.learning_rate * coeffs * residual.unsqueeze(0))
                updated = updated + (
                    0.25 * self.learning_rate * coeffs.sign() * (centered.unsqueeze(0) - weights)
                )
                if active.numel() > 1:
                    gram = updated @ updated.transpose(0, 1)
                    gram.fill_diagonal_(0.0)
                    updated = updated - (0.05 * self.learning_rate * (gram @ updated))
                delta = torch.mean((updated - weights).abs()).item()
                self.update_ema = (0.98 * self.update_ema) + (0.02 * delta)
                self.dictionary[active] = normalize(updated)
            self.sample_count += 1
            self.mean_vector += (sample - self.mean_vector) / float(self.sample_count)
            self.is_fitted = True
        return self

    def fit(self, data: torch.Tensor) -> "OnlineDictionaryLearner":
        return self.partial_fit(data)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


__all__ = ["OnlineDictionaryLearner"]
