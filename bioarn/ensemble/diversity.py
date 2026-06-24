"""Diversity helpers for building heterogeneous Bio-ARN experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.ensemble.config import ExpertConfig
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.preprocessing import CompetitiveLearner, HebbianSparseCoder, OnlinePCA, PatchEncoder, PreprocessingPipeline
from bioarn.scaling import BatchedCCCPool


@dataclass
class DiversityManager:
    """Ensure experts are meaningfully different instead of redundant clones."""

    def measure_diversity(self, expert_predictions: list[list[int]]) -> float:
        """Pairwise disagreement rate across expert prediction histories."""

        if len(expert_predictions) < 2:
            return 0.0
        num_samples = min((len(predictions) for predictions in expert_predictions), default=0)
        if num_samples == 0:
            return 0.0

        disagreements = 0.0
        comparisons = 0
        for sample_index in range(num_samples):
            for first in range(len(expert_predictions)):
                for second in range(first + 1, len(expert_predictions)):
                    left = int(expert_predictions[first][sample_index])
                    right = int(expert_predictions[second][sample_index])
                    if left == -1 and right == -1:
                        continue
                    comparisons += 1
                    disagreements += float(left != right)
        return disagreements / comparisons if comparisons else 0.0

    @staticmethod
    def _value(base_config: Any, key: str, default: Any) -> Any:
        if isinstance(base_config, dict):
            return base_config.get(key, default)
        return getattr(base_config, key, default)

    @staticmethod
    def _build_pool(
        *,
        input_dim: int,
        concept_dim: int,
        max_pool_size: int,
        threshold: float,
        learning_rate: float,
    ) -> BatchedCCCPool:
        ccc_config = CCCConfig(
            input_dim=int(input_dim),
            concept_dim=int(concept_dim),
            num_f1_features=max(24, min(128, max(16, int(input_dim // 2)))),
            f1_top_k=max(4, min(32, max(4, int(min(128, max(16, input_dim // 2)) // 4)))),
            fast_lr=1.0,
            slow_lr=float(learning_rate),
            feedback_lr=float(learning_rate),
            max_pool_size=int(max_pool_size),
        )
        margin_config = MarginGateConfig(
            theta_margin=float(threshold),
            theta_margin_lr=0.001,
            theta_resonance=min(0.95, float(threshold) + 0.2),
        )
        return BatchedCCCPool(ccc_config, margin_config)

    def create_diverse_experts(self, base_config: Any, num_experts: int = 4) -> list[ExpertConfig]:
        """Create a heterogeneous set of preprocessors and pools."""

        input_dim = int(self._value(base_config, "input_dim", 3072))
        concept_dim = int(self._value(base_config, "concept_dim", 128))
        pool_size = int(self._value(base_config, "max_pool_size", 200))
        image_size = tuple(self._value(base_config, "image_size", (32, 32, 3)))
        learning_rate = float(self._value(base_config, "learning_rate", 0.02))
        num_classes = int(self._value(base_config, "num_classes", 10))

        patch_pipeline = PreprocessingPipeline(
            [
                ("patches", PatchEncoder(image_size=image_size, patch_size=8, output_dim=192, seed=29)),
                ("competitive", CompetitiveLearner(192, num_neurons=128, learning_rate=0.02, seed=31)),
            ]
        )

        experts = [
            ExpertConfig(
                name="pca-global",
                preprocessor=PreprocessingPipeline(
                    [("pca", OnlinePCA(input_dim, output_dim=128, max_samples=256, seed=11))]
                ),
                pool=self._build_pool(
                    input_dim=128,
                    concept_dim=concept_dim,
                    max_pool_size=pool_size,
                    threshold=0.34,
                    learning_rate=learning_rate,
                ),
            ),
            ExpertConfig(
                name="sparse-texture",
                preprocessor=PreprocessingPipeline(
                    [("sparse", HebbianSparseCoder(input_dim, num_features=512, sparsity=0.08, learning_rate=0.015, seed=17))]
                ),
                pool=self._build_pool(
                    input_dim=512,
                    concept_dim=concept_dim,
                    max_pool_size=pool_size,
                    threshold=0.38,
                    learning_rate=learning_rate,
                ),
            ),
            ExpertConfig(
                name="hierarchy-multiscale",
                pool=VisualHierarchy(
                    HierarchyConfig(
                        image_size=image_size,
                        pool_sizes=[36, 56, 84, 36],
                        concept_dims=[24, 40, 64, 32],
                        thresholds=[0.22, 0.28, 0.34, 0.38],
                        learning_rates=[0.05, 0.04, 0.03, 0.02],
                        class_count=num_classes,
                    )
                ),
            ),
            ExpertConfig(
                name="patch-competitive",
                preprocessor=patch_pipeline,
                pool=self._build_pool(
                    input_dim=128,
                    concept_dim=concept_dim,
                    max_pool_size=pool_size,
                    threshold=0.3,
                    learning_rate=learning_rate,
                ),
            ),
            ExpertConfig(
                name="raw-widefield",
                pool=self._build_pool(
                    input_dim=input_dim,
                    concept_dim=concept_dim,
                    max_pool_size=pool_size,
                    threshold=0.42,
                    learning_rate=learning_rate,
                ),
            ),
        ]
        return experts[: int(max(1, num_experts))]


__all__ = ["DiversityManager"]
