"""Composable preprocessing pipelines."""

from __future__ import annotations

from typing import Any

import torch


class PreprocessingPipeline:
    """Chain preprocessing steps: normalize → contrast → reduce → encode."""

    def __init__(self, steps: list[tuple[str, Any]]):
        self.steps = list(steps)

    @property
    def is_fitted(self) -> bool:
        return all(bool(getattr(step, "is_fitted", True)) for _, step in self.steps)

    def get_output_dim(self, input_dim: int) -> int:
        current_dim = int(input_dim)
        for _, step in self.steps:
            if hasattr(step, "get_output_dim"):
                current_dim = int(step.get_output_dim(current_dim))
        return current_dim

    def fit(self, data: torch.Tensor) -> "PreprocessingPipeline":
        transformed = data.to(torch.float32)
        for _, step in self.steps:
            if hasattr(step, "fit_transform"):
                transformed = step.fit_transform(transformed)
            else:
                if hasattr(step, "fit"):
                    step.fit(transformed)
                transformed = step.transform(transformed)
        return self

    def partial_fit(self, x: torch.Tensor) -> "PreprocessingPipeline":
        transformed = x.to(torch.float32)
        for _, step in self.steps:
            if hasattr(step, "partial_fit"):
                step.partial_fit(transformed)
            transformed = step.transform(transformed)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        transformed = x.to(torch.float32)
        for _, step in self.steps:
            transformed = step.transform(transformed)
        return transformed

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        transformed = data.to(torch.float32)
        for _, step in self.steps:
            if hasattr(step, "fit_transform"):
                transformed = step.fit_transform(transformed)
            else:
                if hasattr(step, "fit"):
                    step.fit(transformed)
                transformed = step.transform(transformed)
        return transformed


__all__ = ["PreprocessingPipeline"]
