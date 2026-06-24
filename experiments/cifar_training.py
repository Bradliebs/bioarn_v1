"""Run CIFAR-10 training with multiple preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from bioarn.preprocessing import (
    ContrastNormalizer,
    OnlinePCA,
    PatchEncoder,
    PreprocessingPipeline,
    SparseRandomProjection,
)
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import (
    SyntheticCIFAR10Stream,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)


@dataclass
class RunResult:
    name: str
    threshold: float
    accuracy: float
    covered_accuracy: float
    abstention_rate: float
    committed_cccs: int
    specialized_cccs: int
    warmup_samples: int
    data_source: str


def _select_streams(config: VisionTrainConfig):
    data_root = Path("data") / "cifar-10-batches-py"
    if not data_root.exists():
        print("cifar10 not cached locally; using synthetic fallback")
        return (
            SyntheticCIFAR10Stream(config.num_train_samples, seed=7),
            SyntheticCIFAR10Stream(config.num_test_samples, seed=8, shuffle=False),
            "synthetic-cifar10",
        )
    return load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=config.num_train_samples,
        test_samples=config.num_test_samples,
        seed=7,
        timeout_seconds=5.0,
    )


def _base_config(threshold: float) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=384,
        margin_threshold=threshold,
        use_batched=True,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=1200,
        num_test_samples=300,
        preprocessing_warmup_samples=200,
    )


def _pipeline_factories() -> list[tuple[str, Callable[[], PreprocessingPipeline | None]]]:
    return [
        ("raw", lambda: None),
        (
            "random-projection",
            lambda: PreprocessingPipeline(
                [("random_projection", SparseRandomProjection(3072, output_dim=256, density=0.1, seed=11))]
            ),
        ),
        (
            "pca-128",
            lambda: PreprocessingPipeline([("pca", OnlinePCA(3072, output_dim=128, max_samples=256, seed=13))]),
        ),
        (
            "contrast-pca",
            lambda: PreprocessingPipeline(
                [
                    ("contrast", ContrastNormalizer(kernel_size=3)),
                    ("pca", OnlinePCA(3072, output_dim=128, max_samples=256, seed=17)),
                ]
            ),
        ),
        (
            "patch-hash",
            lambda: PreprocessingPipeline(
                [("patches", PatchEncoder(image_size=(32, 32, 3), patch_size=8, output_dim=128, seed=19))]
            ),
        ),
    ]


def _run_single(
    *,
    name: str,
    threshold: float,
    pipeline: PreprocessingPipeline | None,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    data_source: str,
) -> RunResult:
    trainer = VisionTrainer(_base_config(threshold), preprocessing=pipeline)
    train_metrics = trainer.train_online(train_samples, num_samples=len(train_samples))
    eval_metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
    analysis = trainer.get_ccc_analysis()
    print(
        f"[result] {name} theta={threshold:.2f} "
        f"acc={float(eval_metrics['accuracy']):.3f} "
        f"covered={float(eval_metrics['covered_accuracy']):.3f} "
        f"abstain={float(eval_metrics['abstention_rate']):.3f} "
        f"cccs={int(analysis['committed_cccs'])}"
    )
    return RunResult(
        name=name,
        threshold=threshold,
        accuracy=float(eval_metrics["accuracy"]),
        covered_accuracy=float(eval_metrics["covered_accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        committed_cccs=int(analysis["committed_cccs"]),
        specialized_cccs=int(analysis["specialized_cccs"]),
        warmup_samples=int(train_metrics["warmup_samples"]),
        data_source=data_source,
    )


def _pick_best(results: list[RunResult]) -> RunResult:
    return max(
        results,
        key=lambda item: (
            item.accuracy,
            item.covered_accuracy,
            item.committed_cccs,
            -item.abstention_rate,
        ),
    )


def _format_table(results: list[RunResult]) -> str:
    headers = ["config", "theta", "accuracy", "covered", "CCCs Used", "abstention", "warmup"]
    rows = [
        [
            result.name,
            f"{result.threshold:.2f}",
            f"{result.accuracy:.3f}",
            f"{result.covered_accuracy:.3f}",
            str(result.committed_cccs),
            f"{result.abstention_rate:.3f}",
            str(result.warmup_samples),
        ]
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    row_lines = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, divider, *row_lines])


def train_hierarchical_cifar() -> dict[str, object]:
    """Train Bio-ARN with a stacked visual hierarchy on CIFAR-like images."""

    base_config = _base_config(0.35)
    train_stream, test_stream, source = _select_streams(base_config)
    train_samples = take_samples(train_stream, 5000)
    test_samples = take_samples(test_stream, 1000)

    hierarchy = VisualHierarchy(HierarchyConfig())

    unsupervised_samples = train_samples[:2000]
    supervised_samples = train_samples[2000:5000]

    print(f"[hierarchy] data_source={source}")
    print(f"[hierarchy] phase1_unsupervised={len(unsupervised_samples)}")
    for tensor, _ in unsupervised_samples:
        hierarchy.learn(tensor)

    print(f"[hierarchy] phase2_supervised={len(supervised_samples)}")
    for tensor, label in supervised_samples:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))

    total = 0
    correct = 0
    covered = 0
    abstained = 0
    for tensor, label in test_samples:
        predicted, _ = hierarchy.classify(tensor)
        total += 1
        abstained += int(predicted == -1)
        if predicted != -1:
            covered += 1
        if label is not None and predicted == label:
            correct += 1

    result = {
        "data_source": source,
        "accuracy": correct / max(total, 1),
        "coverage": covered / max(total, 1),
        "covered_accuracy": correct / max(covered, 1),
        "abstention_rate": abstained / max(total, 1),
        "per_layer_committed": [layer.pool.committed_count for layer in hierarchy.layers],
    }
    print(
        "[hierarchy] "
        f"acc={result['accuracy']:.3f} "
        f"covered={result['covered_accuracy']:.3f} "
        f"coverage={result['coverage']:.3f} "
        f"abstain={result['abstention_rate']:.3f}"
    )
    for index, committed in enumerate(result["per_layer_committed"], start=1):
        print(f"[hierarchy] L{index}_committed={committed}")
    return result


def main() -> None:
    thresholds = [0.30, 0.35, 0.40, 0.45]
    base_config = _base_config(thresholds[0])
    train_stream, test_stream, source = _select_streams(base_config)
    train_samples = take_samples(train_stream, base_config.num_train_samples)
    test_samples = take_samples(test_stream, base_config.num_test_samples)

    print(f"data_source: {source}")
    print(f"train_samples: {len(train_samples)}")
    print(f"test_samples: {len(test_samples)}")

    best_results: list[RunResult] = []
    for name, factory in _pipeline_factories():
        runs = []
        for threshold in thresholds:
            print(f"[run] config={name} theta={threshold:.2f}")
            runs.append(
                _run_single(
                    name=name,
                    threshold=threshold,
                    pipeline=factory(),
                    train_samples=train_samples,
                    test_samples=test_samples,
                    data_source=source,
                )
            )
        best = _pick_best(runs)
        best_results.append(best)
        print(
            f"[best] {name}: theta={best.threshold:.2f} acc={best.accuracy:.3f} "
            f"cccs={best.committed_cccs} abstain={best.abstention_rate:.3f}"
        )

    print("comparison_table:")
    print(_format_table(best_results))


if __name__ == "__main__":
    main()
