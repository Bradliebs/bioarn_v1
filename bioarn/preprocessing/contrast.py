"""Contrast normalization for flattened or image-shaped tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ContrastNormalizer:
    """Local contrast normalization for vision inputs."""

    def __init__(self, kernel_size: int = 3, eps: float = 1e-5):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)
        self.is_fitted = True

    def get_output_dim(self, input_dim: int) -> int:
        return int(input_dim)

    def fit(self, data: torch.Tensor) -> "ContrastNormalizer":
        del data
        return self

    def partial_fit(self, x: torch.Tensor) -> "ContrastNormalizer":
        del x
        return self

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)

    def _to_nchw(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, bool, bool, bool]:
        squeeze_batch = False
        flatten_output = False
        channels_last = False

        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() == 2:
            flatten_output = True
            if x.shape[-1] == 3072:
                return x.reshape(x.shape[0], 3, 32, 32), squeeze_batch, flatten_output, False
            if x.shape[-1] == 784:
                return x.reshape(x.shape[0], 1, 28, 28), squeeze_batch, flatten_output, False
            raise ValueError(f"Cannot infer image shape from flattened dimension {x.shape[-1]}.")

        if x.dim() == 3:
            squeeze_batch = True
            if x.shape[0] in {1, 3}:
                return x.unsqueeze(0), squeeze_batch, False, False
            if x.shape[-1] in {1, 3}:
                channels_last = True
                return x.permute(2, 0, 1).unsqueeze(0), squeeze_batch, False, channels_last
            raise ValueError("3D tensors must be CHW or HWC images.")

        if x.dim() == 4:
            if x.shape[1] in {1, 3}:
                return x, squeeze_batch, False, False
            if x.shape[-1] in {1, 3}:
                channels_last = True
                return x.permute(0, 3, 1, 2), squeeze_batch, False, channels_last
            raise ValueError("4D tensors must be NCHW or NHWC batches.")

        raise ValueError("ContrastNormalizer expects a 1D, 2D, 3D, or 4D tensor.")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        images, squeeze_batch, flatten_output, channels_last = self._to_nchw(
            x.to(torch.float32)
        )
        padding = self.kernel_size // 2
        padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
        local_mean = F.avg_pool2d(padded, self.kernel_size, stride=1)
        local_energy = F.avg_pool2d(padded.square(), self.kernel_size, stride=1)
        local_var = (local_energy - local_mean.square()).clamp_min(0.0)
        normalized = (images - local_mean) / torch.sqrt(local_var + self.eps)

        if flatten_output:
            result = normalized.reshape(normalized.shape[0], -1)
        elif channels_last:
            result = normalized.permute(0, 2, 3, 1)
        else:
            result = normalized

        return result.squeeze(0) if squeeze_batch else result


__all__ = ["ContrastNormalizer"]
