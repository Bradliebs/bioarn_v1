"""ZCA whitening utilities for vision streams."""

from __future__ import annotations

from itertools import islice
from typing import Any, Mapping

import torch

from bioarn.data.base import DataSample, StreamingDataSource
from bioarn.data.vision import CIFAR10Stream


class ZCAWhitening:
    """ZCA whitening for decorrelating image inputs."""

    def __init__(self, epsilon: float = 1e-5, n_components: int | None = None):
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if n_components is not None and int(n_components) <= 0:
            raise ValueError("n_components must be positive when provided.")
        self.epsilon = float(epsilon)
        self.n_components = int(n_components) if n_components is not None else None
        self.mean: torch.Tensor | None = None
        self.whitening_matrix: torch.Tensor | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_output_dim(self, input_dim: int) -> int:
        input_dim = int(input_dim)
        if self.mean is not None and int(self.mean.numel()) != input_dim:
            raise ValueError(f"Expected input_dim={self.mean.numel()}, received {input_dim}.")
        return input_dim

    @staticmethod
    def _ensure_batch(data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if data.dim() == 1:
            return data.unsqueeze(0).to(torch.float32), True
        if data.dim() != 2:
            raise ValueError("ZCAWhitening expects a tensor with shape [D] or [N, D].")
        return data.to(torch.float32), False

    @staticmethod
    def _canonicalize_eigenvectors(eigenvectors: torch.Tensor) -> torch.Tensor:
        if eigenvectors.numel() == 0:
            return eigenvectors
        column_indices = torch.arange(eigenvectors.shape[1], device=eigenvectors.device)
        max_abs_index = torch.argmax(eigenvectors.abs(), dim=0)
        signs = torch.sign(eigenvectors[max_abs_index, column_indices])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return eigenvectors * signs.unsqueeze(0)

    def _require_fitted(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._fitted or self.mean is None or self.whitening_matrix is None:
            raise RuntimeError("ZCAWhitening must be fitted before calling transform().")
        return self.mean, self.whitening_matrix

    def fit(self, data: torch.Tensor) -> "ZCAWhitening":
        """Compute ZCA whitening parameters from data [N, D]."""

        batch, _ = self._ensure_batch(data)
        if batch.shape[0] < 2:
            raise ValueError("ZCAWhitening requires at least two samples.")

        batch64 = batch.to(torch.float64)
        mean64 = batch64.mean(dim=0)
        centered = batch64 - mean64
        covariance = centered.transpose(0, 1) @ centered / batch64.shape[0]

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order)
        eigenvectors = eigenvectors.index_select(1, order)
        eigenvectors = self._canonicalize_eigenvectors(eigenvectors)

        if self.n_components is not None:
            component_count = min(int(self.n_components), eigenvectors.shape[1])
            eigenvalues = eigenvalues[:component_count]
            eigenvectors = eigenvectors[:, :component_count]

        inverse_sqrt = torch.rsqrt(eigenvalues.clamp_min(0.0) + self.epsilon)
        whitening_matrix = (eigenvectors * inverse_sqrt.unsqueeze(0)) @ eigenvectors.transpose(0, 1)

        self.mean = mean64.to(torch.float32).contiguous()
        self.whitening_matrix = whitening_matrix.to(torch.float32).contiguous()
        self._fitted = True
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply ZCA whitening to data [N, D]."""

        batch, squeeze = self._ensure_batch(data)
        mean, whitening_matrix = self._require_fitted()
        centered = batch - mean.to(device=batch.device, dtype=batch.dtype)
        whitened = centered @ whitening_matrix.to(device=batch.device, dtype=batch.dtype)
        return whitened.squeeze(0) if squeeze else whitened

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""

        return self.fit(data).transform(data)

    def state_dict(self) -> dict[str, Any]:
        mean, whitening_matrix = self._require_fitted()
        return {
            "epsilon": self.epsilon,
            "n_components": self.n_components,
            "mean": mean.detach().clone(),
            "whitening_matrix": whitening_matrix.detach().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> "ZCAWhitening":
        mean = state_dict.get("mean")
        whitening_matrix = state_dict.get("whitening_matrix")
        if not isinstance(mean, torch.Tensor) or not isinstance(whitening_matrix, torch.Tensor):
            raise ValueError("state_dict must contain tensor entries for 'mean' and 'whitening_matrix'.")
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        n_components = state_dict.get("n_components", self.n_components)
        self.n_components = int(n_components) if n_components is not None else None
        self.mean = mean.detach().clone().to(torch.float32).contiguous()
        self.whitening_matrix = whitening_matrix.detach().clone().to(torch.float32).contiguous()
        self._fitted = True
        return self


class WhitenedCIFARStream(StreamingDataSource):
    """CIFAR-10 stream with a fixed offline ZCA whitening transform."""

    image_shape = (3, 32, 32)

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "data/",
        *,
        flatten: bool = True,
        normalize: bool = True,
        shuffle: bool | None = None,
        class_sequential: bool = False,
        seed: int = 0,
        epsilon: float = 1e-5,
        n_fit_samples: int = 5000,
        n_components: int | None = None,
        whitening: ZCAWhitening | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        self.split = split
        self.flatten = bool(flatten)
        self.normalize = bool(normalize)
        self.n_fit_samples = int(max(2, n_fit_samples))
        self.base_stream = CIFAR10Stream(
            split=split,
            data_dir=data_dir,
            flatten=True,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=None,
        )
        self.whitener = whitening if whitening is not None else ZCAWhitening(epsilon=epsilon, n_components=n_components)
        if not self.whitener.is_fitted:
            if split != "train":
                raise ValueError("Test whitening requires a fitted ZCAWhitening instance from training data.")
            self._fit_whitener()

    def __len__(self) -> int:
        return len(self.base_stream)

    def _fit_whitener(self) -> None:
        fit_limit = min(len(self.base_stream), self.n_fit_samples)
        if fit_limit < 2:
            raise ValueError("WhitenedCIFARStream requires at least two fit samples.")
        fit_batch = torch.stack(
            [sample.data.to(torch.float32).reshape(-1) for sample in islice(self.base_stream.stream(), fit_limit)],
            dim=0,
        )
        self.whitener.fit(fit_batch)

    def stream(self):
        for sample in self.base_stream.stream():
            whitened = self.whitener.transform(sample.data.to(torch.float32).reshape(-1))
            output = whitened if self.flatten else whitened.reshape(self.image_shape)
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "whitened": True,
                    "whitening": "zca",
                    "whitening_fit_samples": min(len(self.base_stream), self.n_fit_samples),
                    "whitening_epsilon": self.whitener.epsilon,
                    "whitening_components": self.whitener.n_components,
                }
            )
            yield DataSample(
                data=self._move_tensor(output),
                label=sample.label,
                modality=sample.modality,
                metadata=metadata,
            )


__all__ = ["WhitenedCIFARStream", "ZCAWhitening"]
