"""Formal continual learning benchmark with ACC/BWT/FWT metrics."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import PrecisionConfig
from bioarn.data import FashionMNISTStream, MNISTStream
from bioarn.training import VisionTrainConfig, VisionTrainer, take_samples

TASK_GROUPS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    enable_elastic_protection: bool = False
    enable_replay: bool = False
    enable_eviction: bool = False
    enable_precision_routing: bool = False


@dataclass
class ConfigMetrics:
    name: str
    acc: float
    bwt: float
    fwt: float
    forgetting: float
    evaluation_matrix: list[list[float | None]]
    forgetting_by_task: list[float]
    committed_cccs: list[int]
    mean_protection: float
    replay_boosts: int


@dataclass
class BenchmarkResult:
    benchmark_name: str
    config_metrics: list[ConfigMetrics]


CONFIG_SPECS: tuple[ConfigSpec, ...] = (
    ConfigSpec(name="Naive"),
    ConfigSpec(
        name="Elastic",
        enable_elastic_protection=True,
        enable_precision_routing=True,
    ),
    ConfigSpec(name="Replay", enable_replay=True),
    ConfigSpec(
        name="Full",
        enable_elastic_protection=True,
        enable_replay=True,
        enable_eviction=True,
        enable_precision_routing=True,
    ),
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _build_precision_config(pool_size: int) -> PrecisionConfig:
    return PrecisionConfig(
        enabled=True,
        pool_size=pool_size,
        entropy_window=100,
        precision_alpha=5.0,
        precision_threshold=0.5,
        min_precision=0.1,
        max_precision=1.0,
    )


def _format_metric(value: float) -> str:
    return f"{value * 100:6.1f}%"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    divider = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print("| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    print(divider)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |")


def _load_split_task_data(
    stream_cls: type[MNISTStream] | type[FashionMNISTStream],
    *,
    train_per_task: int,
    test_per_task: int,
    seed: int,
    data_dir: Path,
) -> tuple[list[list[tuple[torch.Tensor, int | None]]], list[list[tuple[torch.Tensor, int | None]]]]:
    train_stream = stream_cls(
        split="train",
        data_dir=str(data_dir),
        flatten=True,
        normalize=True,
        shuffle=True,
        seed=seed,
    )
    test_stream = stream_cls(
        split="test",
        data_dir=str(data_dir),
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
    for task_index, (train_samples, test_samples) in enumerate(zip(train_by_task, test_by_task, strict=False)):
        if len(train_samples) < train_per_task:
            raise RuntimeError(
                f"Task {task_index + 1} only yielded {len(train_samples)} training samples."
            )
        if len(test_samples) < test_per_task:
            raise RuntimeError(
                f"Task {task_index + 1} only yielded {len(test_samples)} test samples."
            )
    return train_by_task, test_by_task


def _apply_permutation(
    samples: Sequence[tuple[torch.Tensor, int | None]],
    permutation: torch.Tensor,
) -> list[tuple[torch.Tensor, int | None]]:
    return [
        (tensor.to(torch.float32).reshape(-1).index_select(0, permutation), label)
        for tensor, label in samples
    ]


def _load_permuted_mnist_data(
    *,
    train_per_task: int,
    test_per_task: int,
    num_tasks: int,
    seed: int,
    data_dir: Path,
) -> tuple[list[list[tuple[torch.Tensor, int | None]]], list[list[tuple[torch.Tensor, int | None]]]]:
    train_stream = MNISTStream(
        split="train",
        data_dir=str(data_dir),
        flatten=True,
        normalize=True,
        shuffle=True,
        seed=seed,
    )
    test_stream = MNISTStream(
        split="test",
        data_dir=str(data_dir),
        flatten=True,
        normalize=True,
        shuffle=False,
        seed=seed,
    )
    base_train = take_samples(train_stream, train_per_task, allowed_labels=None)
    base_test = take_samples(test_stream, test_per_task, allowed_labels=None)
    if len(base_train) < train_per_task or len(base_test) < test_per_task:
        raise RuntimeError("Unable to collect enough MNIST samples for Permuted-MNIST.")

    train_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    test_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    for task_index in range(num_tasks):
        generator = torch.Generator().manual_seed(seed + task_index)
        permutation = torch.randperm(784, generator=generator)
        train_by_task.append(_apply_permutation(base_train, permutation))
        test_by_task.append(_apply_permutation(base_test, permutation))
    return train_by_task, test_by_task


def _build_trainer(
    spec: ConfigSpec,
    *,
    input_dim: int,
    concept_dim: int,
    pool_size: int,
    train_per_task: int,
    test_per_task: int,
    learning_rate: float,
    margin_threshold: float,
    replay_interval: int,
    num_f1_features: int | None,
    f1_top_k: int | None,
) -> VisionTrainer:
    return VisionTrainer(
        VisionTrainConfig(
            input_dim=input_dim,
            concept_dim=concept_dim,
            max_pool_size=pool_size,
            max_growth_factor=1.0,
            margin_threshold=margin_threshold,
            use_batched=False,
            batch_size=32,
            learning_rate=learning_rate,
            consolidation_strength=0.0,
            freeze_f1_after=0,
            f1_adapter_dim=16,
            num_train_samples=train_per_task,
            num_test_samples=test_per_task,
            preprocessing_warmup_samples=0,
            protection_growth_rate=0.1,
            protection_decay_rate=0.01,
            replay_interval=replay_interval,
            enable_elastic_protection=spec.enable_elastic_protection,
            enable_replay=spec.enable_replay,
            enable_eviction=spec.enable_eviction,
            precision=(
                _build_precision_config(pool_size) if spec.enable_precision_routing else None
            ),
            num_f1_features=num_f1_features,
            f1_top_k=f1_top_k,
        )
    )


def _preview_accuracy(
    trainer: VisionTrainer,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> float:
    correct = 0
    total = 0
    with torch.inference_mode():
        for tensor, label in samples:
            step_result = trainer._step_pool(  # noqa: SLF001
                trainer._prepare_tensor(tensor),  # noqa: SLF001
                allow_recruit=False,
                preview=True,
            )
            prediction = (
                None
                if step_result.abstained
                else trainer._recognition_label(  # noqa: SLF001
                    step_result.concept_direction,
                    step_result.fired_indices,
                )
            )
            correct += int(label is not None and prediction == label)
            total += int(label is not None)
    return correct / max(total, 1)


def _backward_transfer(matrix: Sequence[Sequence[float | None]]) -> float:
    if len(matrix) <= 1:
        return 0.0
    final_row = matrix[-1]
    deltas = [
        float(final_row[task_index]) - float(matrix[task_index][task_index])
        for task_index in range(len(matrix) - 1)
        if final_row[task_index] is not None and matrix[task_index][task_index] is not None
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
        final_accuracy = 0.0 if final_row[task_index] is None else float(final_row[task_index])
        forgetting.append(max(history, default=0.0) - final_accuracy)
    return forgetting


def _run_config(
    spec: ConfigSpec,
    *,
    task_name: str,
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    baseline_fn: Callable[[int], float],
    concept_dim: int,
    pool_size: int,
    learning_rate: float,
    margin_threshold: float,
    replay_interval: int,
    num_passes: int,
    num_f1_features: int | None,
    f1_top_k: int | None,
    verbose_training: bool,
) -> ConfigMetrics:
    del task_name
    trainer = _build_trainer(
        spec,
        input_dim=784,
        concept_dim=concept_dim,
        pool_size=pool_size,
        train_per_task=max(len(task) for task in train_by_task),
        test_per_task=max(len(task) for task in test_by_task),
        learning_rate=learning_rate,
        margin_threshold=margin_threshold,
        replay_interval=replay_interval,
        num_f1_features=num_f1_features,
        f1_top_k=f1_top_k,
    )
    evaluation_matrix: list[list[float | None]] = []
    forward_transfers: list[float] = []
    committed_cccs: list[int] = []
    replay_boosts = 0

    for stage_index, task_samples in enumerate(train_by_task):
        if stage_index > 0:
            pretrain_accuracy = _preview_accuracy(trainer, test_by_task[stage_index])
            forward_transfers.append(pretrain_accuracy - baseline_fn(stage_index))

        trainer.start_new_task()
        if verbose_training:
            train_result = trainer.train_online(
                list(task_samples),
                num_samples=len(task_samples),
                num_passes=num_passes,
                interleave_classes=True,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                train_result = trainer.train_online(
                    list(task_samples),
                    num_samples=len(task_samples),
                    num_passes=num_passes,
                    interleave_classes=True,
                )
        replay_boosts += int(train_result.get("concept_replay_boosts", 0))

        row: list[float | None] = [None] * len(train_by_task)
        for eval_index in range(stage_index + 1):
            row[eval_index] = _preview_accuracy(trainer, test_by_task[eval_index])
        evaluation_matrix.append(row)
        committed_cccs.append(int(trainer.system.ccc_pool.get_pool_stats()["num_committed"]))

    final_row = evaluation_matrix[-1]
    final_acc = _mean([float(value) for value in final_row if value is not None])
    forgetting_by_task = _forgetting_by_task(evaluation_matrix)
    forgetting = _mean(forgetting_by_task[:-1]) if len(forgetting_by_task) > 1 else _mean(forgetting_by_task)
    pool_stats = trainer.system.ccc_pool.get_pool_stats()
    return ConfigMetrics(
        name=spec.name,
        acc=final_acc,
        bwt=_backward_transfer(evaluation_matrix),
        fwt=_mean(forward_transfers) if forward_transfers else 0.0,
        forgetting=forgetting,
        evaluation_matrix=evaluation_matrix,
        forgetting_by_task=forgetting_by_task,
        committed_cccs=committed_cccs,
        mean_protection=float(pool_stats.get("mean_protection", 0.0)),
        replay_boosts=replay_boosts,
    )


def _run_benchmark(
    *,
    benchmark_name: str,
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    baseline_fn: Callable[[int], float],
    concept_dim: int,
    pool_size: int,
    learning_rate: float,
    margin_threshold: float,
    replay_interval: int,
    num_passes: int,
    num_f1_features: int | None,
    f1_top_k: int | None,
    verbose_training: bool,
) -> BenchmarkResult:
    config_metrics = [
        _run_config(
            spec,
            task_name=benchmark_name,
            train_by_task=train_by_task,
            test_by_task=test_by_task,
            baseline_fn=baseline_fn,
            concept_dim=concept_dim,
            pool_size=pool_size,
            learning_rate=learning_rate,
            margin_threshold=margin_threshold,
            replay_interval=replay_interval,
            num_passes=num_passes,
            num_f1_features=num_f1_features,
            f1_top_k=f1_top_k,
            verbose_training=verbose_training,
        )
        for spec in CONFIG_SPECS
    ]
    return BenchmarkResult(benchmark_name=benchmark_name, config_metrics=config_metrics)


def _print_benchmark_result(result: BenchmarkResult) -> None:
    print(f"\n{result.benchmark_name}")
    rows = [
        [
            metrics.name,
            _format_metric(metrics.acc),
            _format_metric(metrics.bwt),
            _format_metric(metrics.fwt),
            _format_metric(metrics.forgetting),
        ]
        for metrics in result.config_metrics
    ]
    _print_table(["Config", "ACC", "BWT", "FWT", "Forgetting"], rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal continual learning benchmark for Bio-ARN.")
    parser.add_argument("--train-per-task", type=int, default=120, help="Training samples per task.")
    parser.add_argument("--test-per-task", type=int, default=60, help="Test samples per task.")
    parser.add_argument("--num-passes", type=int, default=1, help="Training passes per task.")
    parser.add_argument("--pool-size", type=int, default=100, help="Fixed CCC pool size.")
    parser.add_argument("--concept-dim", type=int, default=128, help="CCC concept dimension.")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Slow/replay learning rate.")
    parser.add_argument("--margin-threshold", type=float, default=0.5, help="CCC margin threshold.")
    parser.add_argument("--replay-interval", type=int, default=64, help="Samples between replay sweeps.")
    parser.add_argument("--num-f1-features", type=int, default=64, help="Shared F1 feature count.")
    parser.add_argument("--f1-top-k", type=int, default=16, help="Top-k active F1 features.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling/permutations.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root.")
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Show per-task trainer progress logs.",
    )
    parser.add_argument(
        "--include-permuted-mnist",
        action="store_true",
        help="Also run the optional Permuted-MNIST benchmark.",
    )
    parser.add_argument(
        "--permuted-tasks",
        type=int,
        default=5,
        help="Number of tasks for optional Permuted-MNIST.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    print(
        "Running continual learning benchmark with "
        f"train_per_task={args.train_per_task}, test_per_task={args.test_per_task}, "
        f"pool_size={args.pool_size}, concept_dim={args.concept_dim}, passes={args.num_passes}, "
        f"seed={args.seed}"
    )
    print(
        "Elastic/Full configs also enable precision-aware routing because CCCPool routes "
        "protected concepts only when precision gating is present."
    )

    mnist_train, mnist_test = _load_split_task_data(
        MNISTStream,
        train_per_task=args.train_per_task,
        test_per_task=args.test_per_task,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    mnist_result = _run_benchmark(
        benchmark_name="Split-MNIST",
        train_by_task=mnist_train,
        test_by_task=mnist_test,
        baseline_fn=lambda _stage_index: 0.5,
        concept_dim=args.concept_dim,
        pool_size=args.pool_size,
        learning_rate=args.learning_rate,
        margin_threshold=args.margin_threshold,
        replay_interval=args.replay_interval,
        num_passes=args.num_passes,
        num_f1_features=args.num_f1_features,
        f1_top_k=args.f1_top_k,
        verbose_training=args.verbose_training,
    )
    _print_benchmark_result(mnist_result)

    fashion_train, fashion_test = _load_split_task_data(
        FashionMNISTStream,
        train_per_task=args.train_per_task,
        test_per_task=args.test_per_task,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    fashion_result = _run_benchmark(
        benchmark_name="Split-Fashion-MNIST",
        train_by_task=fashion_train,
        test_by_task=fashion_test,
        baseline_fn=lambda _stage_index: 0.5,
        concept_dim=args.concept_dim,
        pool_size=args.pool_size,
        learning_rate=args.learning_rate,
        margin_threshold=args.margin_threshold,
        replay_interval=args.replay_interval,
        num_passes=args.num_passes,
        num_f1_features=args.num_f1_features,
        f1_top_k=args.f1_top_k,
        verbose_training=args.verbose_training,
    )
    _print_benchmark_result(fashion_result)

    if args.include_permuted_mnist:
        permuted_train, permuted_test = _load_permuted_mnist_data(
            train_per_task=args.train_per_task,
            test_per_task=args.test_per_task,
            num_tasks=args.permuted_tasks,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        permuted_result = _run_benchmark(
            benchmark_name="Permuted-MNIST",
            train_by_task=permuted_train,
            test_by_task=permuted_test,
            baseline_fn=lambda _stage_index: 0.1,
            concept_dim=args.concept_dim,
            pool_size=args.pool_size,
            learning_rate=args.learning_rate,
            margin_threshold=args.margin_threshold,
            replay_interval=args.replay_interval,
            num_passes=args.num_passes,
            num_f1_features=args.num_f1_features,
            f1_top_k=args.f1_top_k,
            verbose_training=args.verbose_training,
        )
        _print_benchmark_result(permuted_result)


if __name__ == "__main__":
    main()
