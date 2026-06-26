"""Online data augmentation for streamed samples."""

from __future__ import annotations

import math

import torch

from bioarn.data.base import DataSample


class OnlineAugmenter:
    """Augment data on-the-fly during streaming."""

    def __init__(
        self,
        *,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.25,
        crop_prob: float = 0.25,
        noise_prob: float = 0.5,
        occlusion_prob: float = 0.2,
        char_dropout_prob: float = 0.1,
        char_swap_prob: float = 0.05,
        char_insertion_prob: float = 0.05,
        noise_std: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.crop_prob = crop_prob
        self.noise_prob = noise_prob
        self.occlusion_prob = occlusion_prob
        self.char_dropout_prob = char_dropout_prob
        self.char_swap_prob = char_swap_prob
        self.char_insertion_prob = char_insertion_prob
        self.noise_std = noise_std
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def augment(self, sample: DataSample) -> DataSample:
        if sample.modality == "vision":
            data = self._augment_vision(sample.data)
        elif sample.modality == "language":
            data = self._augment_language(sample.data)
        else:
            data = sample.data.clone()
        return DataSample(
            data=data,
            label=sample.label,
            modality=sample.modality,
            metadata={**sample.metadata, "augmented": True},
        )

    def _rand(self) -> float:
        return torch.rand(1, generator=self.generator).item()

    def _infer_image_shape(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.ndim == 3:
            return tensor.clone(), False
        if tensor.ndim != 1:
            raise ValueError("Vision augmentation expects a flattened vector or CHW image tensor")
        numel = tensor.numel()
        if numel == 784:
            return tensor.reshape(1, 28, 28).clone(), True
        if numel == 3072:
            return tensor.reshape(3, 32, 32).clone(), True
        side = int(round(math.sqrt(numel)))
        if side * side == numel:
            return tensor.reshape(1, side, side).clone(), True
        raise ValueError(f"Cannot infer image shape from flattened tensor with {numel} elements")

    def _augment_vision(self, tensor: torch.Tensor) -> torch.Tensor:
        image, should_flatten = self._infer_image_shape(tensor)

        if self._rand() < self.flip_prob:
            image = torch.flip(image, dims=(-1,))

        if self._rand() < self.rotation_prob:
            k = int(torch.randint(1, 4, (1,), generator=self.generator).item())
            image = torch.rot90(image, k=k, dims=(-2, -1))

        if self._rand() < self.crop_prob:
            image = self._random_crop(image)

        if self._rand() < self.noise_prob:
            noise = torch.randn(image.shape, generator=self.generator) * self.noise_std
            image = image + noise.to(device=image.device, dtype=image.dtype)

        if self._rand() < self.occlusion_prob:
            image = self._occlude(image)

        image = image.clamp(0.0, 1.0)
        if should_flatten:
            image = image.reshape(-1)
        return image.to(tensor.device)

    def _random_crop(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        crop_size = max(1, min(height, width) - 2)
        padded = torch.nn.functional.pad(image.unsqueeze(0), (2, 2, 2, 2), mode="replicate").squeeze(0)
        max_y = padded.shape[-2] - crop_size
        max_x = padded.shape[-1] - crop_size
        offset_y = int(torch.randint(0, max_y + 1, (1,), generator=self.generator).item())
        offset_x = int(torch.randint(0, max_x + 1, (1,), generator=self.generator).item())
        cropped = padded[:, offset_y : offset_y + crop_size, offset_x : offset_x + crop_size]
        return torch.nn.functional.interpolate(
            cropped.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).reshape(channels, height, width)

    def _occlude(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        block_h = max(1, height // 4)
        block_w = max(1, width // 4)
        start_y = int(torch.randint(0, max(1, height - block_h + 1), (1,), generator=self.generator).item())
        start_x = int(torch.randint(0, max(1, width - block_w + 1), (1,), generator=self.generator).item())
        result = image.clone()
        result[:, start_y : start_y + block_h, start_x : start_x + block_w] = 0.0
        return result

    def _augment_language(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 1:
            raise ValueError("Language augmentation expects a 1D token tensor")
        tokens = tensor.clone()
        if tokens.numel() == 0:
            return tokens

        if self._rand() < self.char_dropout_prob:
            mask = torch.rand(tokens.shape, generator=self.generator) < self.char_dropout_prob
            mask = mask.to(tokens.device)
            tokens[mask] = 0

        if tokens.numel() > 1 and self._rand() < self.char_swap_prob:
            index = int(torch.randint(0, tokens.numel() - 1, (1,), generator=self.generator).item())
            left = tokens[index].clone()
            tokens[index] = tokens[index + 1]
            tokens[index + 1] = left

        if self._rand() < self.char_insertion_prob:
            insert_at = int(torch.randint(0, tokens.numel(), (1,), generator=self.generator).item())
            token = int(tokens[insert_at].item())
            inserted = torch.cat(
                [
                    tokens[:insert_at],
                    torch.tensor([token], dtype=tokens.dtype, device=tokens.device),
                    tokens[insert_at:],
                ]
            )
            tokens = inserted[: tensor.numel()]

        return tokens


__all__ = ["OnlineAugmenter"]
