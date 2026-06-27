"""Sprint E combined benchmark: locking, convolutional CCCs, and precision weighting."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from bioarn.config import ConvCCCConfig, PrecisionConfig
from bioarn.core.conv_ccc import ConvCCCPool, ConvCCCPoolOutput
from bioarn.predictive.precision_weighting import PrecisionWeightedGate
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 7
OOD_SEED = 42
TRAIN_N = 2_000
TEST_N = 500
OOD_N = 300
POOL_SIZE = 100

TASK_GROUPS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
TASK_TRAIN_N = 500
TASK_TEST_N = 200

TRACE_CHECKPOINTS = (100, 500, 1_000, 1_500, 2_000)
TRACE_SMOOTHING_WINDOW = 50


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    use_conv_ccc: bool = False
    concept_locking: bool = False
    precision_weighting: bool = False


@dataclass
class AccuracyResult:
    name: str
    accuracy: float
    ood_auroc: float
    num_locked: int
    num_committed: int
    pool_size: int
    fire_rate: float
    mean_importance: float


@dataclass
class ContinualResult:
    name: str
    evaluation_matrix: list[list[float]]
    final_average_accuracy: float
    final_backward_transfer: float
    mean_forward_transfer: float
    forgetting_by_task: list[float]


@dataclass
class PrecisionTracePoint:
    samples_seen: int
    pool_entropy: float
    precision: float
    learning_rate_multiplier: float


ACCURACY_SPECS = [
    BenchmarkSpec("baseline"),
    BenchmarkSpec("locked", concept_locking=True),
    BenchmarkSpec("precision", precision_weighting=True),
    BenchmarkSpec("locked_precision", concept_locking=True, precision_weighting=True),
    BenchmarkSpec("conv", use_conv_ccc=True),
    BenchmarkSpec("conv_locked", use_conv_ccc=True, concept_locking=True),
    BenchmarkSpec("conv_precision", use_conv_ccc=True, precision_weighting=True),
    BenchmarkSpec("conv_all", use_conv_ccc=True, concept_locking=True, precision_weighting=True),
]

CONTINUAL_SPECS = [
    BenchmarkSpec("baseline"),
    BenchmarkSpec("locked", concept_locking=True),
    BenchmarkSpec("conv_locked", use_conv_ccc=True, concept_locking=True),
    BenchmarkSpec("conv_all", use_conv_ccc=True, concept_locking=True, precision_weighting=True),
]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _format_percent(value: float) -> str:
    return f"{value * 100:6.1f}%"


def _format_signed_percent(value: float) -> str:
    return f"{value * 100:+6.1f}"


def _format_float(value: float) -> str:
    return f"{value:7.3f}"


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


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    buckets: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
            continue
        buckets[int(label)].append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int | None]] = []
    labels = sorted(buckets)
    while any(buckets[label] for label in labels):
        for label in labels:
            if buckets[label]:
                interleaved.append(buckets[label].pop(0))
    interleaved.extend(unlabeled)
    return interleaved


def _auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    positives = [(score, 1) for score in id_scores]
    negatives = [(score, 0) for score in ood_scores]
    all_points = sorted(positives + negatives, key=lambda item: item[0], reverse=True)

    total_pos = len(id_scores)
    total_neg = len(ood_scores)
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    prev_score: float | None = None
    area = 0.0

    for score, label in all_points:
        if prev_score is not None and score != prev_score:
            area += (fp - prev_fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    area += (fp - prev_fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
    area += (total_neg - fp) / total_neg * (tp + prev_tp) / (2 * total_pos)
    return float(area)


def _backward_transfer(matrix: Sequence[Sequence[float]], *, stage_index: int) -> float:
    if stage_index == 0:
        return 0.0
    deltas = [
        float(matrix[stage_index][task_index]) - float(matrix[task_index][task_index])
        for task_index in range(stage_index)
    ]
    return _mean(deltas) if deltas else 0.0


def _forgetting_by_task(matrix: Sequence[Sequence[float]]) -> list[float]:
    final_row = matrix[-1]
    forgetting: list[float] = []
    for task_index in range(len(final_row)):
        history = [float(matrix[stage_index][task_index]) for stage_index in range(task_index, len(matrix))]
        forgetting.append(max(history, default=0.0) - float(final_row[task_index]))
    return forgetting


def _make_ood_samples(num_samples: int, *, seed: int = OOD_SEED) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(3072, generator=generator) for _ in range(num_samples)]


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


def _require_conv_pool(pool: object) -> ConvCCCPool:
    if not isinstance(pool, ConvCCCPool):
        raise TypeError(f"Expected ConvCCCPool, got {type(pool).__name__}.")
    return pool


def _build_trainer(
    spec: BenchmarkSpec,
    *,
    num_train_samples: int,
    num_test_samples: int,
    pool_size: int,
) -> tuple[VisionTrainer, PrecisionWeightedGate | None]:
    config = VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=pool_size,
        margin_threshold=0.55,
        use_batched=False,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        preprocessing_warmup_samples=200,
        use_conv_ccc=spec.use_conv_ccc,
        precision=(
            None
            if spec.use_conv_ccc or not spec.precision_weighting
            else _build_precision_config(pool_size)
        ),
    )
    trainer = VisionTrainer(config)
    pool = trainer.system.ccc_pool
    pool.config.lock_threshold = 0.8 if spec.concept_locking else 1.1

    external_precision: PrecisionWeightedGate | None = None
    if spec.use_conv_ccc and spec.precision_weighting:
        conv_pool = _require_conv_pool(pool)
        conv_config: ConvCCCConfig = conv_pool.config
        external_precision = PrecisionWeightedGate(_build_precision_config(conv_config.max_pool_size))
        external_precision.set_pool_size(int(conv_config.max_pool_size))
    return trainer, external_precision


def _pool_locked_count(trainer: VisionTrainer) -> int:
    pool = trainer.system.ccc_pool
    pool_stats = pool.get_pool_stats()
    if "num_locked" in pool_stats:
        return int(pool_stats["num_locked"])
    cccs = getattr(pool, "cccs", None)
    if cccs is None:
        return 0
    count = 0
    for ccc in cccs:
        if hasattr(ccc, "locked"):
            count += int(bool(ccc.locked.item()))
        elif hasattr(ccc, "is_locked"):
            count += int(bool(ccc.is_locked.item()))
    return count


def _precision_controller(
    trainer: VisionTrainer,
    external_precision: PrecisionWeightedGate | None,
) -> PrecisionWeightedGate | None:
    if external_precision is not None:
        return external_precision
    controller = getattr(trainer.system.ccc_pool, "precision_gate", None)
    return controller if isinstance(controller, PrecisionWeightedGate) else None


def _preview_precision(
    trainer: VisionTrainer,
    tensor: torch.Tensor,
    external_precision: PrecisionWeightedGate | None,
) -> tuple[float, float, float]:
    prepared = trainer._prepare_tensor(tensor)  # noqa: SLF001
    if external_precision is not None:
        conv_preview: ConvCCCPoolOutput = _require_conv_pool(trainer.system.ccc_pool).preview(prepared)
        entropy = external_precision.entropy_estimator.compute_entropy(candidate=conv_preview.fired_indices)
        precision = float(external_precision.preview_pool_output(conv_preview.fired_indices))
        return entropy, precision, precision

    preview_result = trainer._step_pool(prepared, allow_recruit=False, preview=True)  # noqa: SLF001
    controller = getattr(trainer.system.ccc_pool, "precision_gate", None)
    if isinstance(controller, PrecisionWeightedGate):
        entropy = controller.entropy_estimator.compute_entropy(candidate=preview_result.fired_indices)
        precision = float(controller.current_precision)
        return entropy, precision, precision
    return 1.0, 1.0, 1.0


def _run_online_training(
    trainer: VisionTrainer,
    train_samples: Sequence[tuple[torch.Tensor, int | None]],
    *,
    external_precision: PrecisionWeightedGate | None = None,
    trace_checkpoints: Sequence[int] = (),
) -> dict[str, object]:
    ordered = _interleave_by_class(list(train_samples))
    precision_history: deque[float] = deque(maxlen=TRACE_SMOOTHING_WINDOW)
    entropy_history: deque[float] = deque(maxlen=TRACE_SMOOTHING_WINDOW)
    lr_history: deque[float] = deque(maxlen=TRACE_SMOOTHING_WINDOW)
    trace_points: list[PrecisionTracePoint] = []
    trace_targets = set(int(value) for value in trace_checkpoints)

    processed = 0
    labeled = 0
    correct = 0
    abstained = 0
    progress_interval = max(1, len(ordered) // 10)

    for tensor, label in ordered:
        learning_rate_multiplier = 1.0
        pool_entropy = 1.0
        precision = 1.0
        controller = _precision_controller(trainer, external_precision)
        if controller is not None:
            pool_entropy, precision, learning_rate_multiplier = _preview_precision(
                trainer,
                tensor,
                external_precision,
            )
            entropy_history.append(pool_entropy)
            precision_history.append(precision)
            lr_history.append(learning_rate_multiplier)

        prediction, abstained_flag, _, confidence, step_result = trainer._train_single_sample(  # noqa: SLF001
            tensor,
            label,
            learning_rate_multiplier=learning_rate_multiplier,
        )

        if external_precision is not None:
            external_precision.observe_pool_output(step_result.fired_indices)

        processed += 1
        if label is not None:
            labeled += 1
            correct += int(prediction == label)
        abstained += int(abstained_flag)

        if processed in trace_targets and precision_history:
            trace_points.append(
                PrecisionTracePoint(
                    samples_seen=processed,
                    pool_entropy=_mean(list(entropy_history)),
                    precision=_mean(list(precision_history)),
                    learning_rate_multiplier=_mean(list(lr_history)),
                )
            )

        if processed % progress_interval == 0 or processed == len(ordered):
            print(
                f"[train:{processed:4d}/{len(ordered)}] "
                f"acc={correct / max(labeled, 1):.3f} "
                f"abstain={abstained / max(processed, 1):.3f} "
                f"committed={int(trainer.system.ccc_pool.get_pool_stats()['num_committed'])}"
            )

    return {
        "processed_samples": processed,
        "accuracy": correct / max(labeled, 1),
        "abstention_rate": abstained / max(processed, 1),
        "mean_learning_rate_multiplier": _mean(list(lr_history)) if lr_history else 1.0,
        "trace": trace_points,
    }


def _evaluate_samples(
    trainer: VisionTrainer,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> dict[str, object]:
    correct = 0
    labeled = 0
    covered = 0
    abstained = 0
    predictions: list[int | None] = []
    labels: list[int | None] = []
    confidences: list[float] = []
    per_class_totals: Counter[int] = Counter()
    per_class_correct: Counter[int] = Counter()

    for tensor, label in samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
            preview=True,
        )
        prediction = (
            None
            if step_result.abstained
            else trainer._recognition_label(step_result.concept_direction, step_result.fired_indices)  # noqa: SLF001
        )
        confidence = float(step_result.confidence)

        labeled += int(label is not None)
        correct += int(label is not None and prediction == label)
        covered += int(prediction is not None)
        abstained += int(step_result.abstained)
        predictions.append(prediction)
        labels.append(label)
        confidences.append(confidence)
        if label is not None:
            per_class_totals[int(label)] += 1
            per_class_correct[int(label)] += int(prediction == label)

    return {
        "accuracy": correct / max(labeled, 1),
        "coverage": covered / max(len(samples), 1),
        "abstention_rate": abstained / max(len(samples), 1),
        "predictions": predictions,
        "labels": labels,
        "confidences": confidences,
        "per_class_accuracy": {
            label: per_class_correct[label] / max(per_class_totals[label], 1)
            for label in sorted(per_class_totals)
        },
    }


def _ood_confidences(trainer: VisionTrainer, ood_samples: Sequence[torch.Tensor]) -> list[float]:
    confidences: list[float] = []
    for tensor in ood_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
            preview=True,
        )
        confidences.append(float(step_result.confidence))
    return confidences


def _prepare_split_cifar10(
    train_stream,
    test_stream,
) -> tuple[
    list[list[tuple[torch.Tensor, int | None]]],
    list[list[tuple[torch.Tensor, int | None]]],
]:
    train_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    test_by_task: list[list[tuple[torch.Tensor, int | None]]] = []
    for task_group in TASK_GROUPS:
        labels = {int(label) for label in task_group}
        train_by_task.append(take_samples(train_stream, TASK_TRAIN_N, allowed_labels=labels))
        test_by_task.append(take_samples(test_stream, TASK_TEST_N, allowed_labels=labels))
    return train_by_task, test_by_task


def _run_accuracy_benchmark(
    train_samples: Sequence[tuple[torch.Tensor, int | None]],
    test_samples: Sequence[tuple[torch.Tensor, int | None]],
    ood_samples: Sequence[torch.Tensor],
) -> list[AccuracyResult]:
    results: list[AccuracyResult] = []
    for spec in ACCURACY_SPECS:
        print(f"\n[accuracy] running {spec.name}")
        trainer, external_precision = _build_trainer(
            spec,
            num_train_samples=TRAIN_N,
            num_test_samples=TEST_N,
            pool_size=POOL_SIZE,
        )
        _run_online_training(trainer, train_samples, external_precision=external_precision)
        eval_result = _evaluate_samples(trainer, test_samples)
        ood_scores = _ood_confidences(trainer, ood_samples)
        pool_stats = trainer.system.ccc_pool.get_pool_stats()
        results.append(
            AccuracyResult(
                name=spec.name,
                accuracy=float(eval_result["accuracy"]),
                ood_auroc=_auroc(list(eval_result["confidences"]), ood_scores),
                num_locked=_pool_locked_count(trainer),
                num_committed=int(pool_stats["num_committed"]),
                pool_size=int(pool_stats["total_concepts"]),
                fire_rate=float(pool_stats["fire_rate"]),
                mean_importance=float(pool_stats["mean_importance"]),
            )
        )
    return results


def _run_continual_benchmark(
    train_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
    test_by_task: Sequence[Sequence[tuple[torch.Tensor, int | None]]],
) -> list[ContinualResult]:
    results: list[ContinualResult] = []

    for spec in CONTINUAL_SPECS:
        print(f"\n[continual] running {spec.name}")
        trainer, external_precision = _build_trainer(
            spec,
            num_train_samples=TASK_TRAIN_N,
            num_test_samples=TASK_TEST_N,
            pool_size=POOL_SIZE,
        )
        baseline_trainer, baseline_external_precision = _build_trainer(
            spec,
            num_train_samples=TASK_TRAIN_N,
            num_test_samples=TASK_TEST_N,
            pool_size=POOL_SIZE,
        )
        del baseline_external_precision
        initial_task_accuracy = [
            float(_evaluate_samples(baseline_trainer, task_samples)["accuracy"])
            for task_samples in test_by_task
        ]

        evaluation_matrix: list[list[float]] = []
        forward_transfers: list[float] = []

        for stage_index, task_samples in enumerate(train_by_task):
            if stage_index > 0:
                pretrain_accuracy = float(_evaluate_samples(trainer, test_by_task[stage_index])["accuracy"])
                forward_transfers.append(pretrain_accuracy - initial_task_accuracy[stage_index])
            trainer.start_new_task()
            _run_online_training(trainer, task_samples, external_precision=external_precision)
            row = [
                float(_evaluate_samples(trainer, eval_samples)["accuracy"])
                for eval_samples in test_by_task
            ]
            evaluation_matrix.append(row)

        forgetting = _forgetting_by_task(evaluation_matrix)
        results.append(
            ContinualResult(
                name=spec.name,
                evaluation_matrix=evaluation_matrix,
                final_average_accuracy=_mean(evaluation_matrix[-1]),
                final_backward_transfer=_backward_transfer(
                    evaluation_matrix,
                    stage_index=len(evaluation_matrix) - 1,
                ),
                mean_forward_transfer=_mean(forward_transfers) if forward_transfers else 0.0,
                forgetting_by_task=forgetting,
            )
        )
    return results


def _run_precision_dynamics(
    train_samples: Sequence[tuple[torch.Tensor, int | None]],
) -> list[PrecisionTracePoint]:
    print("\n[precision] running dynamics trace")
    trainer, external_precision = _build_trainer(
        BenchmarkSpec("precision", precision_weighting=True),
        num_train_samples=TRAIN_N,
        num_test_samples=TEST_N,
        pool_size=POOL_SIZE,
    )
    result = _run_online_training(
        trainer,
        train_samples,
        external_precision=external_precision,
        trace_checkpoints=TRACE_CHECKPOINTS,
    )
    return list(result["trace"])


def _print_accuracy_results(results: Sequence[AccuracyResult]) -> None:
    print("\n=== CIFAR-10 Accuracy Comparison ===")
    headers = [
        "Config",
        "Accuracy",
        "OOD AUROC",
        "Locked",
        "Committed",
        "Pool Size",
        "Fire Rate",
        "MeanImp",
    ]
    rows = [
        [
            result.name,
            _format_percent(result.accuracy),
            f"{result.ood_auroc:8.3f}",
            str(result.num_locked),
            str(result.num_committed),
            str(result.pool_size),
            f"{result.fire_rate:8.3f}",
            f"{result.mean_importance:7.3f}",
        ]
        for result in results
    ]
    _print_table(headers, rows)


def _print_continual_results(results: Sequence[ContinualResult]) -> None:
    print("\n=== Continual Learning (Split-CIFAR-10) ===")
    headers = [
        "Config",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "Final Acc",
        "BWT",
        "FWT",
        "Forgetting",
    ]
    rows: list[list[str]] = []
    for result in results:
        final_row = result.evaluation_matrix[-1]
        rows.append(
            [
                result.name,
                *[_format_percent(value) for value in final_row],
                _format_percent(result.final_average_accuracy),
                _format_signed_percent(result.final_backward_transfer),
                _format_signed_percent(result.mean_forward_transfer),
                _format_percent(_mean(result.forgetting_by_task)),
            ]
        )
    _print_table(headers, rows)


def _print_precision_trace(trace_points: Sequence[PrecisionTracePoint]) -> None:
    print("\n=== Precision Dynamics ===")
    headers = ["Samples seen", "Pool Entropy", "Precision", "Effective LR"]
    rows = [
        [
            str(point.samples_seen),
            _format_float(point.pool_entropy),
            _format_float(point.precision),
            _format_float(point.learning_rate_multiplier),
        ]
        for point in trace_points
    ]
    _print_table(headers, rows)


def _build_decision_markdown(
    *,
    data_source: str,
    accuracy_results: Sequence[AccuracyResult],
    continual_results: Sequence[ContinualResult],
    precision_trace: Sequence[PrecisionTracePoint],
) -> str:
    lines = [
        "# Sprint E combined benchmark",
        "",
        f"- Dataset source: `{data_source}`",
        f"- Accuracy benchmark: {TRAIN_N} train / {TEST_N} test / {OOD_N} OOD samples",
        f"- Continual benchmark: {len(TASK_GROUPS)} Split-CIFAR-10 tasks, {TASK_TRAIN_N} train + {TASK_TEST_N} test per task",
        "",
        "## CIFAR-10 accuracy",
        "",
        "| Config | Accuracy | OOD AUROC | Locked | Committed | Pool Size | Fire Rate | Mean Importance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in accuracy_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    f"{result.accuracy * 100:.1f}%",
                    f"{result.ood_auroc:.3f}",
                    str(result.num_locked),
                    str(result.num_committed),
                    str(result.pool_size),
                    f"{result.fire_rate:.3f}",
                    f"{result.mean_importance:.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Continual learning",
            "",
            "| Config | T1 | T2 | T3 | T4 | T5 | Final Acc | BWT | FWT | Mean Forgetting |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in continual_results:
        final_row = result.evaluation_matrix[-1]
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    *[f"{value * 100:.1f}%" for value in final_row],
                    f"{result.final_average_accuracy * 100:.1f}%",
                    f"{result.final_backward_transfer * 100:+.1f}",
                    f"{result.mean_forward_transfer * 100:+.1f}",
                    f"{_mean(result.forgetting_by_task) * 100:.1f}%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Precision dynamics",
            "",
            "| Samples seen | Pool Entropy | Precision | Effective LR |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for point in precision_trace:
        lines.append(
            f"| {point.samples_seen} | {point.pool_entropy:.3f} | {point.precision:.3f} | {point.learning_rate_multiplier:.3f} |"
        )

    best_accuracy = max(accuracy_results, key=lambda item: item.accuracy)
    best_continual = max(continual_results, key=lambda item: item.final_average_accuracy)
    first_precision = precision_trace[0] if precision_trace else None
    last_precision = precision_trace[-1] if precision_trace else None

    lines.extend(
        [
            "",
            "## Key takeaways",
            "",
            f"- Best short-run accuracy: `{best_accuracy.name}` at {best_accuracy.accuracy * 100:.1f}% accuracy.",
            (
                f"- Best continual retention: `{best_continual.name}` with "
                f"{best_continual.final_average_accuracy * 100:.1f}% final average accuracy and "
                f"{_mean(best_continual.forgetting_by_task) * 100:.1f}% mean forgetting."
            ),
        ]
    )
    if first_precision is not None and last_precision is not None:
        lines.append(
            "- Precision weighting trend: "
            f"entropy {first_precision.pool_entropy:.3f} -> {last_precision.pool_entropy:.3f}, "
            f"precision {first_precision.precision:.3f} -> {last_precision.precision:.3f}, "
            f"effective LR {first_precision.learning_rate_multiplier:.3f} -> "
            f"{last_precision.learning_rate_multiplier:.3f}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    torch.manual_seed(SEED)

    train_stream, test_stream, data_source = load_cifar10_or_synthetic(
        data_dir=Path("data"),
        train_samples=12_000,
        test_samples=4_000,
        seed=SEED,
    )
    print(f"Loaded {data_source} benchmark stream.")

    train_samples = take_samples(train_stream, TRAIN_N)
    test_samples = take_samples(test_stream, TEST_N)
    ood_samples = _make_ood_samples(OOD_N)
    split_train, split_test = _prepare_split_cifar10(train_stream, test_stream)

    accuracy_results = _run_accuracy_benchmark(train_samples, test_samples, ood_samples)
    continual_results = _run_continual_benchmark(split_train, split_test)
    precision_trace = _run_precision_dynamics(train_samples)

    _print_accuracy_results(accuracy_results)
    _print_continual_results(continual_results)
    _print_precision_trace(precision_trace)

    print("\n=== Decision Note ===")
    print(_build_decision_markdown(
        data_source=data_source,
        accuracy_results=accuracy_results,
        continual_results=continual_results,
        precision_trace=precision_trace,
    ))


if __name__ == "__main__":
    main()
