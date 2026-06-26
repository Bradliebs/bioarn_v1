"""Continual learning benchmarks for Bio-ARN's online vision stack."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.data import MNISTStream
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples

TASK_GROUPS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


@dataclass
class StageSummary:
    stage_name: str
    average_accuracy: float
    backward_transfer: float
    forward_transfer: float | None
    cumulative_accuracy: float | None
    committed_cccs: list[int]


@dataclass
class ContinualResult:
    name: str
    data_source: str
    task_groups: list[tuple[int, int]]
    evaluation_matrix: list[list[float | None]]
    stage_summaries: list[StageSummary]
    forgetting_by_task: list[float]
    final_average_accuracy: float
    final_backward_transfer: float
    mean_forward_transfer: float


def _build_hierarchy() -> VisualHierarchy:
    return VisualHierarchy(
        HierarchyConfig(
            image_size=(32, 32, 3),
            patch_sizes=[8, 2, 1, 1],
            pool_sizes=[100, 200, 500, 200],
            concept_dims=[32, 64, 128, 64],
            thresholds=[0.25, 0.3, 0.35, 0.4],
            learning_rates=[0.05, 0.03, 0.02, 0.01],
            class_count=10,
        )
    )


def _build_mnist_trainer(
    *,
    train_samples: int,
    test_samples: int,
) -> VisionTrainer:
    return VisionTrainer(
        VisionTrainConfig(
            input_dim=784,
            concept_dim=128,
            max_pool_size=256,
            margin_threshold=0.5,
            use_batched=True,
            batch_size=32,
            learning_rate=0.02,
            num_train_samples=train_samples,
            num_test_samples=test_samples,
            preprocessing_warmup_samples=min(train_samples, max(50, train_samples // 3)),
        )
    )


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    buckets: dict[int, list[tuple[torch.Tensor, int | None]]] = {}
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
            continue
        buckets.setdefault(int(label), []).append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int | None]] = []
    labels = sorted(buckets)
    while any(buckets[label] for label in labels):
        for label in labels:
            if buckets[label]:
                interleaved.append(buckets[label].pop(0))
    interleaved.extend(unlabeled)
    return interleaved


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _format_task_group(group: Sequence[int]) -> str:
    return f"{group[0]}/{group[1]}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "   -"
    return f"{value * 100:5.1f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    divider = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(divider)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _evaluate_hierarchy(
    hierarchy: VisualHierarchy,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> dict[str, float]:
    total = 0
    correct = 0
    covered = 0
    abstained = 0
    for tensor, label in samples:
        prediction, _ = hierarchy.classify(tensor)
        total += 1
        abstained += int(prediction == -1)
        covered += int(prediction != -1)
        correct += int(label is not None and prediction == label)
    return {
        "accuracy": correct / max(total, 1),
        "coverage": covered / max(total, 1),
        "abstention_rate": abstained / max(total, 1),
    }


def _train_hierarchy_task(
    hierarchy: VisualHierarchy,
    task_samples: Sequence[tuple[torch.Tensor, int | None]],
    *,
    warmup_fraction: float,
    stage_name: str,
) -> list[int]:
    ordered = _interleave_by_class(list(task_samples))
    if not ordered:
        raise RuntimeError(f"{stage_name}: no training samples collected.")

    warmup_count = min(
        int(len(ordered) * warmup_fraction),
        max(0, len(ordered) - max(20, len(ordered) // 5)),
    )
    warmup_count = max(0, warmup_count)
    supervised_count = len(ordered) - warmup_count
    print(
        f"[{stage_name}] warmup={warmup_count} supervised={supervised_count} "
        f"task_classes={sorted({int(label) for _, label in ordered if label is not None})}"
    )

    for tensor, _ in ordered[:warmup_count]:
        hierarchy.learn(tensor)
    for tensor, label in ordered[warmup_count:]:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))

    committed = [int(layer.pool.committed_count) for layer in hierarchy.layers]
    print(f"[{stage_name}] committed_cccs={committed}")
    return committed


def _combine_task_samples(
    task_samples: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    *,
    upto: int,
) -> list[tuple[torch.Tensor, int | None]]:
    combined: list[tuple[torch.Tensor, int | None]] = []
    for task_index in range(upto + 1):
        combined.extend(task_samples[task_index])
    return _interleave_by_class(combined)


def _average_accuracy_from_row(row: Sequence[float | None], *, upto: int) -> float:
    return _mean([float(row[index]) for index in range(upto + 1) if row[index] is not None])


def _backward_transfer(
    matrix: Sequence[Sequence[float | None]],
    *,
    stage_index: int,
) -> float:
    if stage_index == 0:
        return 0.0
    deltas = [
        float(matrix[stage_index][task_index]) - float(matrix[task_index][task_index])
        for task_index in range(stage_index)
        if matrix[stage_index][task_index] is not None and matrix[task_index][task_index] is not None
    ]
    return _mean(deltas) if deltas else 0.0


def _forgetting_by_task(matrix: Sequence[Sequence[float | None]]) -> list[float]:
    final_row = matrix[-1]
    forgetting: list[float] = []
    for task_index in range(len(final_row)):
        history = [
            float(matrix[stage_index][task_index])
            for stage_index in range(task_index, len(matrix))
            if matrix[stage_index][task_index] is not None
        ]
        final_accuracy = float(final_row[task_index]) if final_row[task_index] is not None else 0.0
        forgetting.append(max(history, default=0.0) - final_accuracy)
    return forgetting


def _print_result(result: ContinualResult) -> None:
    print("\n" + "=" * 88)
    print(f"{result.name} (data_source={result.data_source})")
    print("=" * 88)

    headers = ["after task", *[f"T{index + 1} ({_format_task_group(group)})" for index, group in enumerate(result.task_groups)]]
    rows = []
    for stage_index, row in enumerate(result.evaluation_matrix):
        rows.append(
            [f"T{stage_index + 1}"]
            + [_format_percent(row[task_index]) for task_index in range(len(result.task_groups))]
        )
    print("\nPer-task accuracy (%)")
    _print_table(headers, rows)

    summary_headers = ["stage", "avg acc", "BWT", "FWT", "cum acc", "committed CCCs"]
    summary_rows = []
    for summary in result.stage_summaries:
        summary_rows.append(
            [
                summary.stage_name,
                _format_percent(summary.average_accuracy),
                _format_percent(summary.backward_transfer),
                _format_percent(summary.forward_transfer),
                _format_percent(summary.cumulative_accuracy),
                "/".join(str(value) for value in summary.committed_cccs),
            ]
        )
    print("\nStage summary (%)")
    _print_table(summary_headers, summary_rows)

    forgetting_headers = ["task", "classes", "forgetting"]
    forgetting_rows = [
        [f"T{index + 1}", _format_task_group(result.task_groups[index]), _format_percent(value)]
        for index, value in enumerate(result.forgetting_by_task)
    ]
    print("\nFinal forgetting by task (%)")
    _print_table(forgetting_headers, forgetting_rows)

    print(
        "\nFinal summary: "
        f"avg_acc={result.final_average_accuracy:.3f} "
        f"BWT={result.final_backward_transfer:+.3f} "
        f"FWT={result.mean_forward_transfer:+.3f} "
        f"mean_forgetting={_mean(result.forgetting_by_task):.3f}"
    )


def _run_continual_benchmark(
    *,
    name: str,
    data_source: str,
    task_groups: Sequence[Sequence[int]] | None,
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    random_baseline_fn: Callable[[int], float],
    train_stage_fn: Callable[[int, Sequence[tuple[torch.Tensor, int | None]], str], list[int]],
    eval_stage_fn: Callable[[Sequence[tuple[torch.Tensor, int | None]]], dict[str, float]],
    cumulative_eval_fn: Callable[[int], float | None] | None = None,
) -> ContinualResult:
    matrix: list[list[float | None]] = []
    stage_summaries: list[StageSummary] = []
    forward_transfers: list[float] = []

    for stage_index, task_samples in enumerate(train_by_task):
        stage_name = f"{name.lower().replace(' ', '-')}-task-{stage_index + 1}"
        forward_transfer = None
        if stage_index > 0:
            forward_eval = eval_stage_fn(test_by_task[stage_index])
            forward_transfer = float(forward_eval["accuracy"]) - random_baseline_fn(stage_index)
            forward_transfers.append(forward_transfer)
            print(
                f"[{stage_name}] pretrain_acc={forward_eval['accuracy']:.3f} "
                f"random={random_baseline_fn(stage_index):.3f} "
                f"FWT={forward_transfer:+.3f}"
            )

        committed = train_stage_fn(stage_index, task_samples, stage_name)

        row: list[float | None] = [None] * len(train_by_task)
        for eval_index in range(stage_index + 1):
            metrics = eval_stage_fn(test_by_task[eval_index])
            row[eval_index] = float(metrics["accuracy"])
        matrix.append(row)

        stage_summaries.append(
            StageSummary(
                stage_name=f"T{stage_index + 1}",
                average_accuracy=_average_accuracy_from_row(row, upto=stage_index),
                backward_transfer=_backward_transfer(matrix, stage_index=stage_index),
                forward_transfer=forward_transfer,
                cumulative_accuracy=(
                    None if cumulative_eval_fn is None else float(cumulative_eval_fn(stage_index))
                ),
                committed_cccs=committed,
            )
        )

    final_row = matrix[-1]
    final_average_accuracy = _mean([float(value) for value in final_row if value is not None])
    final_backward_transfer = _backward_transfer(matrix, stage_index=len(matrix) - 1)
    forgetting_by_task = _forgetting_by_task(matrix)
    mean_forward_transfer = _mean(forward_transfers) if forward_transfers else 0.0
    return ContinualResult(
        name=name,
        data_source=data_source,
        task_groups=[tuple(int(label) for label in group) for group in (task_groups or TASK_GROUPS)],
        evaluation_matrix=matrix,
        stage_summaries=stage_summaries,
        forgetting_by_task=forgetting_by_task,
        final_average_accuracy=final_average_accuracy,
        final_backward_transfer=final_backward_transfer,
        mean_forward_transfer=mean_forward_transfer,
    )


def run_split_cifar10(
    *,
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    data_source: str,
    warmup_fraction: float,
) -> ContinualResult:
    hierarchy = _build_hierarchy()

    def train_stage(
        stage_index: int,
        task_samples: Sequence[tuple[torch.Tensor, int | None]],
        stage_name: str,
    ) -> list[int]:
        del stage_index
        return _train_hierarchy_task(
            hierarchy,
            task_samples,
            warmup_fraction=warmup_fraction,
            stage_name=stage_name,
        )

    return _run_continual_benchmark(
        name="Split-CIFAR-10",
        data_source=data_source,
        task_groups=TASK_GROUPS,
        train_by_task=train_by_task,
        test_by_task=test_by_task,
        random_baseline_fn=lambda _stage_index: 1.0 / len(TASK_GROUPS[0]),
        train_stage_fn=train_stage,
        eval_stage_fn=lambda samples: _evaluate_hierarchy(hierarchy, samples),
    )


def run_class_incremental_cifar10(
    *,
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    data_source: str,
    warmup_fraction: float,
) -> ContinualResult:
    hierarchy = _build_hierarchy()

    def train_stage(
        stage_index: int,
        task_samples: Sequence[tuple[torch.Tensor, int | None]],
        stage_name: str,
    ) -> list[int]:
        del stage_index
        return _train_hierarchy_task(
            hierarchy,
            task_samples,
            warmup_fraction=warmup_fraction,
            stage_name=stage_name,
        )

    def cumulative_eval(stage_index: int) -> float:
        metrics = _evaluate_hierarchy(hierarchy, _combine_task_samples(test_by_task, upto=stage_index))
        return float(metrics["accuracy"])

    return _run_continual_benchmark(
        name="Class-Incremental CIFAR-10",
        data_source=data_source,
        task_groups=TASK_GROUPS,
        train_by_task=train_by_task,
        test_by_task=test_by_task,
        random_baseline_fn=lambda _stage_index: 1.0 / len(TASK_GROUPS[0]),
        train_stage_fn=train_stage,
        eval_stage_fn=lambda samples: _evaluate_hierarchy(hierarchy, samples),
        cumulative_eval_fn=cumulative_eval,
    )


def _apply_permutation(
    samples: Sequence[tuple[torch.Tensor, int | None]],
    permutation: torch.Tensor,
) -> list[tuple[torch.Tensor, int | None]]:
    return [
        (tensor.to(torch.float32).reshape(-1).index_select(0, permutation), label)
        for tensor, label in samples
    ]


def run_permuted_mnist(
    *,
    train_samples: int,
    test_samples: int,
    num_tasks: int,
    seed: int,
) -> ContinualResult:
    trainer = _build_mnist_trainer(
        train_samples=train_samples,
        test_samples=test_samples,
    )
    train_stream = MNISTStream(split="train", data_dir="data", flatten=True, normalize=True, shuffle=True, seed=seed)
    test_stream = MNISTStream(split="test", data_dir="data", flatten=True, normalize=True, shuffle=False, seed=seed)
    base_train = take_samples(train_stream, train_samples)
    base_test = take_samples(test_stream, test_samples)
    if len(base_train) < train_samples or len(base_test) < test_samples:
        raise RuntimeError("Unable to collect enough MNIST samples for permuted benchmark.")

    train_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    test_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    task_groups = [(task_index, task_index) for task_index in range(num_tasks)]
    for task_index in range(num_tasks):
        generator = torch.Generator().manual_seed(seed + task_index)
        permutation = torch.randperm(784, generator=generator)
        train_by_task.append(_apply_permutation(base_train, permutation))
        test_by_task.append(_apply_permutation(base_test, permutation))

    def train_stage(
        stage_index: int,
        task_samples: Sequence[tuple[torch.Tensor, int | None]],
        stage_name: str,
    ) -> list[int]:
        del stage_index
        print(f"[{stage_name}] train_samples={len(task_samples)}")
        trainer.train_online(
            list(task_samples),
            num_samples=len(task_samples),
            num_passes=1,
            interleave_classes=True,
        )
        committed = int(trainer.get_ccc_analysis()["committed_cccs"])
        print(f"[{stage_name}] committed_cccs={committed}")
        return [committed]

    def eval_stage(samples: Sequence[tuple[torch.Tensor, int | None]]) -> dict[str, float]:
        metrics = trainer.evaluate(list(samples), num_samples=len(samples))
        return {
            "accuracy": float(metrics["accuracy"]),
            "coverage": float(metrics["coverage"]),
            "abstention_rate": float(metrics["abstention_rate"]),
        }

    result = _run_continual_benchmark(
        name="Permuted-MNIST",
        data_source="mnist",
        task_groups=task_groups,
        train_by_task=train_by_task,
        test_by_task=test_by_task,
        random_baseline_fn=lambda _stage_index: 0.1,
        train_stage_fn=train_stage,
        eval_stage_fn=eval_stage,
        cumulative_eval_fn=None,
    )
    return result


def _load_mnist_task_data(
    *,
    train_per_task: int,
    test_per_task: int,
    seed: int,
) -> tuple[list[list[tuple[torch.Tensor, int | None]]], list[list[tuple[torch.Tensor, int | None]]]]:
    train_stream = MNISTStream(
        split="train",
        data_dir="data",
        flatten=True,
        normalize=True,
        shuffle=True,
        seed=seed,
    )
    test_stream = MNISTStream(
        split="test",
        data_dir="data",
        flatten=True,
        normalize=True,
        shuffle=False,
        seed=seed,
    )
    train_by_task = [
        take_samples(train_stream, train_per_task, allowed_labels=set(task_group))
        for task_group in TASK_GROUPS
    ]
    test_by_task = [
        take_samples(test_stream, test_per_task, allowed_labels=set(task_group))
        for task_group in TASK_GROUPS
    ]

    for task_index, (train_samples_for_task, test_samples_for_task) in enumerate(
        zip(train_by_task, test_by_task, strict=False)
    ):
        if len(train_samples_for_task) < train_per_task:
            raise RuntimeError(
                f"MNIST task {task_index + 1} only yielded {len(train_samples_for_task)} training samples."
            )
        if len(test_samples_for_task) < test_per_task:
            raise RuntimeError(
                f"MNIST task {task_index + 1} only yielded {len(test_samples_for_task)} test samples."
            )

    return train_by_task, test_by_task


def run_split_mnist(
    *,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> ContinualResult:
    trainer = _build_mnist_trainer(
        train_samples=train_samples,
        test_samples=test_samples,
    )
    train_by_task, test_by_task = _load_mnist_task_data(
        train_per_task=train_samples,
        test_per_task=test_samples,
        seed=seed,
    )

    def train_stage(
        stage_index: int,
        task_samples: Sequence[tuple[torch.Tensor, int | None]],
        stage_name: str,
    ) -> list[int]:
        del stage_index
        print(
            f"[{stage_name}] train_samples={len(task_samples)} "
            f"task_classes={sorted({int(label) for _, label in task_samples if label is not None})}"
        )
        trainer.train_online(
            list(task_samples),
            num_samples=len(task_samples),
            num_passes=1,
            interleave_classes=True,
        )
        committed = int(trainer.get_ccc_analysis()["committed_cccs"])
        print(f"[{stage_name}] committed_cccs={committed}")
        return [committed]

    def eval_stage(samples: Sequence[tuple[torch.Tensor, int | None]]) -> dict[str, float]:
        metrics = trainer.evaluate(list(samples), num_samples=len(samples))
        return {
            "accuracy": float(metrics["accuracy"]),
            "coverage": float(metrics["coverage"]),
            "abstention_rate": float(metrics["abstention_rate"]),
        }

    def cumulative_eval(stage_index: int) -> float:
        metrics = eval_stage(_combine_task_samples(test_by_task, upto=stage_index))
        return float(metrics["accuracy"])

    return _run_continual_benchmark(
        name="Split-MNIST",
        data_source="mnist",
        task_groups=TASK_GROUPS,
        train_by_task=train_by_task,
        test_by_task=test_by_task,
        random_baseline_fn=lambda _stage_index: 1.0 / len(TASK_GROUPS[0]),
        train_stage_fn=train_stage,
        eval_stage_fn=eval_stage,
        cumulative_eval_fn=cumulative_eval,
    )


def _load_cifar_task_data(
    *,
    train_per_task: int,
    test_per_task: int,
    seed: int,
) -> tuple[list[list[tuple[torch.Tensor, int | None]]], list[list[tuple[torch.Tensor, int | None]]], str]:
    fallback_train = max(8000, train_per_task * len(TASK_GROUPS) * 2)
    fallback_test = max(2000, test_per_task * len(TASK_GROUPS) * 2)
    train_stream, test_stream, data_source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=fallback_train,
        test_samples=fallback_test,
        seed=seed,
        timeout_seconds=5.0,
    )
    train_by_task = [
        take_samples(train_stream, train_per_task, allowed_labels=set(task_group))
        for task_group in TASK_GROUPS
    ]
    test_by_task = [
        take_samples(test_stream, test_per_task, allowed_labels=set(task_group))
        for task_group in TASK_GROUPS
    ]

    for task_index, (train_samples_for_task, test_samples_for_task) in enumerate(
        zip(train_by_task, test_by_task, strict=False)
    ):
        if len(train_samples_for_task) < train_per_task:
            raise RuntimeError(
                f"Task {task_index + 1} only yielded {len(train_samples_for_task)} training samples."
            )
        if len(test_samples_for_task) < test_per_task:
            raise RuntimeError(
                f"Task {task_index + 1} only yielded {len(test_samples_for_task)} test samples."
            )
    return train_by_task, test_by_task, data_source


def summarize_findings(results: Sequence[ContinualResult]) -> list[str]:
    findings: list[str] = []
    for result in results:
        mean_forgetting = _mean(result.forgetting_by_task)
        findings.append(
            f"{result.name}: avg_acc={result.final_average_accuracy:.3f}, "
            f"BWT={result.final_backward_transfer:+.3f}, "
            f"mean_forgetting={mean_forgetting:.3f}"
        )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bio-ARN continual learning benchmarks.")
    parser.add_argument("--train-per-task", type=int, default=500, help="Training samples per CIFAR task.")
    parser.add_argument("--test-per-task", type=int, default=150, help="Test samples per CIFAR task.")
    parser.add_argument("--warmup-fraction", type=float, default=0.25, help="Unlabelled warmup fraction per task.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    parser.add_argument(
        "--include-permuted-mnist",
        action="store_true",
        help="Also run the optional Permuted-MNIST benchmark.",
    )
    parser.add_argument("--mnist-train-samples", type=int, default=1000, help="Training samples for each permuted-MNIST task.")
    parser.add_argument("--mnist-test-samples", type=int, default=200, help="Test samples for each permuted-MNIST task.")
    parser.add_argument("--mnist-tasks", type=int, default=5, help="Number of permutation tasks to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_by_task, test_by_task, data_source = _load_cifar_task_data(
        train_per_task=args.train_per_task,
        test_per_task=args.test_per_task,
        seed=args.seed,
    )
    print(
        f"Loaded {data_source} for continual learning. "
        f"train_per_task={args.train_per_task} test_per_task={args.test_per_task}"
    )

    split_result = run_split_cifar10(
        train_by_task=copy.deepcopy(train_by_task),
        test_by_task=test_by_task,
        data_source=data_source,
        warmup_fraction=args.warmup_fraction,
    )
    _print_result(split_result)

    class_incremental_result = run_class_incremental_cifar10(
        train_by_task=copy.deepcopy(train_by_task),
        test_by_task=test_by_task,
        data_source=data_source,
        warmup_fraction=args.warmup_fraction,
    )
    _print_result(class_incremental_result)

    all_results = [split_result, class_incremental_result]
    if args.include_permuted_mnist:
        permuted_result = run_permuted_mnist(
            train_samples=args.mnist_train_samples,
            test_samples=args.mnist_test_samples,
            num_tasks=args.mnist_tasks,
            seed=args.seed,
        )
        _print_result(permuted_result)
        all_results.append(permuted_result)

    print("\nKey findings")
    for finding in summarize_findings(all_results):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
