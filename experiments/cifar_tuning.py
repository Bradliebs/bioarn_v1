"""Tune Bio-ARN CIFAR-10 configurations beyond the current 26% plateau."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.preprocessing import ContrastNormalizer, OnlinePCA, PreprocessingPipeline
from bioarn.training import (
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

SEED = 7
TRAIN_N = 5000
TEST_N = 500


@dataclass(frozen=True)
class VisionSpec:
    name: str
    train_samples: int
    max_pool_size: int
    margin_threshold: float
    num_passes: int
    interleave_classes: bool
    preprocessing_warmup_samples: int
    pipeline_factory: Callable[[], PreprocessingPipeline | None]
    concept_dim: int = 256
    batch_size: int = 32


@dataclass(frozen=True)
class HierarchySpec:
    name: str
    train_samples: int
    warmup_ratio: float
    config_factory: Callable[[], HierarchyConfig]


@dataclass
class RunResult:
    name: str
    family: str
    train_samples: int
    accuracy: float
    covered_accuracy: float
    coverage: float
    abstention_rate: float
    committed_units: int
    extra: str


def _make_pca_pipeline(*, output_dim: int = 128, max_samples: int = 1024, seed: int = 17) -> PreprocessingPipeline:
    return PreprocessingPipeline(
        [("pca", OnlinePCA(3072, output_dim=output_dim, max_samples=max_samples, seed=seed))]
    )


def _make_contrast_pca_pipeline(
    *,
    kernel_size: int = 3,
    output_dim: int = 128,
    max_samples: int = 1024,
    seed: int = 17,
) -> PreprocessingPipeline:
    return PreprocessingPipeline(
        [
            ("contrast", ContrastNormalizer(kernel_size=kernel_size)),
            ("pca", OnlinePCA(3072, output_dim=output_dim, max_samples=max_samples, seed=seed)),
        ]
    )


def _vision_config(spec: VisionSpec) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=spec.concept_dim,
        max_pool_size=spec.max_pool_size,
        margin_threshold=spec.margin_threshold,
        use_batched=True,
        batch_size=spec.batch_size,
        learning_rate=0.01,
        num_train_samples=spec.train_samples,
        num_test_samples=TEST_N,
        preprocessing_warmup_samples=spec.preprocessing_warmup_samples,
    )


def _hierarchy_default_config() -> HierarchyConfig:
    return HierarchyConfig(
        image_size=(32, 32, 3),
        patch_sizes=[8, 2, 1, 1],
        pool_sizes=[100, 200, 500, 200],
        concept_dims=[32, 64, 128, 64],
        thresholds=[0.25, 0.30, 0.35, 0.40],
        learning_rates=[0.05, 0.03, 0.02, 0.01],
        class_count=10,
    )


def _hierarchy_wide_config() -> HierarchyConfig:
    return HierarchyConfig(
        image_size=(32, 32, 3),
        patch_sizes=[8, 2, 1, 1],
        pool_sizes=[128, 256, 640, 256],
        concept_dims=[48, 96, 160, 96],
        thresholds=[0.22, 0.27, 0.32, 0.37],
        learning_rates=[0.05, 0.03, 0.02, 0.01],
        class_count=10,
    )


def _vision_specs() -> list[VisionSpec]:
    return [
        VisionSpec(
            name="vision-pca-control",
            train_samples=2000,
            max_pool_size=384,
            margin_threshold=0.40,
            num_passes=2,
            interleave_classes=True,
            preprocessing_warmup_samples=256,
            pipeline_factory=lambda: _make_pca_pipeline(output_dim=128, max_samples=256, seed=17),
        ),
        VisionSpec(
            name="vision-pca-5k-theta40",
            train_samples=5000,
            max_pool_size=512,
            margin_threshold=0.40,
            num_passes=3,
            interleave_classes=True,
            preprocessing_warmup_samples=512,
            pipeline_factory=lambda: _make_pca_pipeline(output_dim=128, max_samples=1024, seed=19),
        ),
        VisionSpec(
            name="vision-pca-5k-theta25",
            train_samples=5000,
            max_pool_size=512,
            margin_threshold=0.25,
            num_passes=3,
            interleave_classes=True,
            preprocessing_warmup_samples=512,
            pipeline_factory=lambda: _make_pca_pipeline(output_dim=128, max_samples=1024, seed=29),
        ),
        VisionSpec(
            name="vision-contrast-pca-5k",
            train_samples=5000,
            max_pool_size=512,
            margin_threshold=0.25,
            num_passes=3,
            interleave_classes=True,
            preprocessing_warmup_samples=512,
            pipeline_factory=lambda: _make_contrast_pca_pipeline(
                kernel_size=3,
                output_dim=128,
                max_samples=1024,
                seed=31,
            ),
        ),
    ]


def _hierarchy_specs() -> list[HierarchySpec]:
    return [
        HierarchySpec(
            name="hierarchy-control",
            train_samples=2000,
            warmup_ratio=1 / 3,
            config_factory=_hierarchy_default_config,
        ),
        HierarchySpec(
            name="hierarchy-5k-warm40",
            train_samples=5000,
            warmup_ratio=0.40,
            config_factory=_hierarchy_default_config,
        ),
        HierarchySpec(
            name="hierarchy-wide-5k-warm40",
            train_samples=5000,
            warmup_ratio=0.40,
            config_factory=_hierarchy_wide_config,
        ),
    ]


def _run_vision(
    spec: VisionSpec,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
) -> RunResult:
    subset = train_samples[: spec.train_samples]
    trainer = VisionTrainer(_vision_config(spec), preprocessing=spec.pipeline_factory())
    print(
        f"[vision] {spec.name} train={len(subset)} pool={spec.max_pool_size} "
        f"theta={spec.margin_threshold:.2f} passes={spec.num_passes}"
    )
    train_metrics = trainer.train_online(
        subset,
        num_samples=len(subset),
        num_passes=spec.num_passes,
        interleave_classes=spec.interleave_classes,
    )
    eval_metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
    analysis = trainer.get_ccc_analysis()
    result = RunResult(
        name=spec.name,
        family="vision",
        train_samples=len(subset),
        accuracy=float(eval_metrics["accuracy"]),
        covered_accuracy=float(eval_metrics["covered_accuracy"]),
        coverage=float(eval_metrics["coverage"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        committed_units=int(analysis["committed_cccs"]),
        extra=(
            f"warmup={int(train_metrics['warmup_samples'])} "
            f"passes={spec.num_passes} theta={spec.margin_threshold:.2f}"
        ),
    )
    print(
        f"[vision-result] {result.name} acc={result.accuracy:.3f} "
        f"covered={result.covered_accuracy:.3f} coverage={result.coverage:.3f} "
        f"abstain={result.abstention_rate:.3f} cccs={result.committed_units}"
    )
    return result


def _run_hierarchy(
    spec: HierarchySpec,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
) -> RunResult:
    subset = train_samples[: spec.train_samples]
    hierarchy = VisualHierarchy(spec.config_factory())
    warmup_count = min(max(int(len(subset) * spec.warmup_ratio), 1), len(subset) - 1)
    print(
        f"[hierarchy] {spec.name} train={len(subset)} warmup={warmup_count} "
        f"ratio={spec.warmup_ratio:.2f}"
    )
    progress_interval = max(1, len(subset) // 5)
    with torch.inference_mode():
        for index, (tensor, _) in enumerate(subset[:warmup_count], start=1):
            hierarchy.learn(tensor)
            if index % progress_interval == 0 or index == warmup_count:
                print(f"[hierarchy-train] {spec.name} warmup {index}/{warmup_count}")
        for offset, (tensor, label) in enumerate(subset[warmup_count:], start=1):
            if label is not None:
                hierarchy.learn(tensor, int(label))
            trained = warmup_count + offset
            if trained % progress_interval == 0 or trained == len(subset):
                print(f"[hierarchy-train] {spec.name} total {trained}/{len(subset)}")

        total = 0
        correct = 0
        covered = 0
        abstained = 0
        for tensor, label in test_samples:
            predicted, _ = hierarchy.classify(tensor)
            total += 1
            if predicted == -1:
                abstained += 1
                continue
            covered += 1
            if label is not None and int(predicted) == int(label):
                correct += 1

    result = RunResult(
        name=spec.name,
        family="hierarchy",
        train_samples=len(subset),
        accuracy=correct / max(total, 1),
        covered_accuracy=correct / max(covered, 1),
        coverage=covered / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        committed_units=sum(layer.pool.committed_count for layer in hierarchy.layers),
        extra=f"warmup_ratio={spec.warmup_ratio:.2f}",
    )
    print(
        f"[hierarchy-result] {result.name} acc={result.accuracy:.3f} "
        f"covered={result.covered_accuracy:.3f} coverage={result.coverage:.3f} "
        f"abstain={result.abstention_rate:.3f} units={result.committed_units}"
    )
    return result


def _format_table(results: list[RunResult]) -> str:
    headers = [
        "config",
        "family",
        "train",
        "acc",
        "covered",
        "coverage",
        "abstain",
        "units",
        "notes",
    ]
    rows = [
        [
            result.name,
            result.family,
            str(result.train_samples),
            f"{result.accuracy:.3f}",
            f"{result.covered_accuracy:.3f}",
            f"{result.coverage:.3f}",
            f"{result.abstention_rate:.3f}",
            str(result.committed_units),
            result.extra,
        ]
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def run_experiment() -> list[RunResult]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=TRAIN_N,
        test_samples=TEST_N,
        seed=SEED,
        timeout_seconds=5.0,
    )
    train_samples = take_samples(train_stream, TRAIN_N)
    test_samples = take_samples(test_stream, TEST_N)
    print(f"data_source: {source}")
    print(f"train_samples: {len(train_samples)}")
    print(f"test_samples: {len(test_samples)}")

    results: list[RunResult] = []
    for spec in _vision_specs():
        results.append(_run_vision(spec, train_samples, test_samples))
    for spec in _hierarchy_specs():
        results.append(_run_hierarchy(spec, train_samples, test_samples))

    ordered = sorted(
        results,
        key=lambda item: (
            item.accuracy,
            item.covered_accuracy,
            item.coverage,
            -item.abstention_rate,
        ),
        reverse=True,
    )
    best = ordered[0]
    print("\ncomparison_table:")
    print(_format_table(ordered))
    print(
        f"\nbest_result: {best.name} "
        f"acc={best.accuracy:.3f} covered={best.covered_accuracy:.3f} "
        f"coverage={best.coverage:.3f}"
    )
    return ordered


if __name__ == "__main__":
    run_experiment()
