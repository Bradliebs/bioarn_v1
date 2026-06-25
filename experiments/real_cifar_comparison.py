"""Compare Bio-ARN configurations on real CIFAR-10 images."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import (
    EnsembleTrainer,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

TRAIN_N = 2000
TEST_N = 500
OOD_N = 300
NUM_PASSES = 2
SEED = 7
OOD_SEED = 42


@dataclass
class RunResult:
    name: str
    accuracy: float
    abstention_rate: float
    ood_auroc: float


def _auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    """Trapezoidal AUROC with ID confidence as the positive class."""

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


def _interleave_by_class(
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    labeled: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
        else:
            labeled[int(label)].append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int | None]] = []
    classes = sorted(labeled)
    while any(labeled[class_id] for class_id in classes):
        for class_id in classes:
            if labeled[class_id]:
                interleaved.append(labeled[class_id].pop(0))
    interleaved.extend(unlabeled)
    return interleaved


def _make_ood_samples(num_samples: int, seed: int = OOD_SEED) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(3072, generator=generator) for _ in range(num_samples)]


def _base_train_config() -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=384,
        margin_threshold=0.35,
        use_batched=True,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=TRAIN_N,
        num_test_samples=TEST_N,
        preprocessing_warmup_samples=200,
    )


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


def _build_ensemble() -> EnsemblePool:
    expert_configs = DiversityManager().create_diverse_experts(
        {
            "input_dim": 3072,
            "concept_dim": 128,
            "max_pool_size": 256,
            "learning_rate": 0.01,
            "image_size": (32, 32, 3),
            "num_classes": 10,
        },
        num_experts=5,
    )
    return EnsemblePool(
        EnsembleConfig(
            num_experts=5,
            voting_method="weighted",
            abstention_threshold=0.5,
            use_boosting=True,
            diversity_target=0.3,
            expert_configs=expert_configs,
        )
    )


def run_baseline(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    trainer = VisionTrainer(_base_train_config())
    trainer.train_online(
        train_samples,
        num_samples=TRAIN_N,
        interleave_classes=True,
        num_passes=NUM_PASSES,
    )
    eval_metrics = trainer.evaluate(test_samples, num_samples=TEST_N)

    id_scores: list[float] = []
    for tensor, _ in test_samples:
        _, _, confidence, _ = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        _, _, confidence, _ = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(confidence))

    return RunResult(
        name="baseline",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_hierarchy(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    hierarchy = _build_hierarchy()
    interleaved_train = _interleave_by_class(train_samples)
    warmup_end = len(interleaved_train) // 3

    for tensor, _ in interleaved_train[:warmup_end]:
        hierarchy.learn(tensor)
    for tensor, label in interleaved_train[warmup_end:]:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))

    total = correct = abstained = 0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        predicted, confidence = hierarchy.classify(tensor)
        total += 1
        if predicted == -1:
            abstained += 1
            id_scores.append(0.0)
            continue
        id_scores.append(float(confidence))
        if label is not None and predicted == int(label):
            correct += 1

    ood_scores = [float(hierarchy.classify(tensor)[1]) for tensor in ood_samples]
    return RunResult(
        name="hierarchy",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_ensemble(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    pool = _build_ensemble()
    interleaved_train = _interleave_by_class(train_samples)
    EnsembleTrainer(pool, log_every=max(1, len(interleaved_train) // 10)).train(interleaved_train)

    total = correct = abstained = 0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        result = pool.classify(tensor)
        total += 1
        if result.abstained:
            abstained += 1
            id_scores.append(0.0)
            continue
        id_scores.append(float(result.confidence))
        if label is not None and result.predicted_class == int(label):
            correct += 1

    ood_scores = [float(pool.classify(tensor).confidence) for tensor in ood_samples]
    return RunResult(
        name="ensemble",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_both(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    hierarchy = _build_hierarchy()
    pool = _build_ensemble()
    interleaved_train = _interleave_by_class(train_samples)
    warmup_end = len(interleaved_train) // 3

    for tensor, _ in interleaved_train[:warmup_end]:
        hierarchy.learn(tensor)
    for tensor, label in interleaved_train[warmup_end:]:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))

    EnsembleTrainer(pool, log_every=max(1, len(interleaved_train) // 10)).train(interleaved_train)

    total = correct = abstained = 0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        final_prediction, final_confidence = _classify_with_fallback(hierarchy, pool, tensor)
        total += 1
        if final_prediction == -1:
            abstained += 1
            id_scores.append(0.0)
            continue
        id_scores.append(final_confidence)
        if label is not None and final_prediction == int(label):
            correct += 1

    ood_scores = [
        _classify_with_fallback(hierarchy, pool, tensor)[1]
        for tensor in ood_samples
    ]
    return RunResult(
        name="both",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def _classify_with_fallback(
    hierarchy: VisualHierarchy,
    pool: EnsemblePool,
    tensor: torch.Tensor,
) -> tuple[int, float]:
    hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
    if hierarchy_prediction != -1:
        return int(hierarchy_prediction), float(hierarchy_confidence)

    ensemble_result = pool.classify(tensor)
    if ensemble_result.abstained:
        return -1, 0.0
    return int(ensemble_result.predicted_class), float(ensemble_result.confidence)


def _format_table(rows: list[RunResult]) -> str:
    headers = ("Config", "Accuracy", "Abstention", "OOD AUROC")
    values = [
        (
            row.name,
            f"{row.accuracy * 100:.1f}%",
            f"{row.abstention_rate * 100:.1f}%",
            f"{row.ood_auroc:.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    header_line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider_line = "  ".join("─" * widths[index] for index in range(len(headers)))
    body_lines = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in values
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def run_experiment() -> list[RunResult]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))

    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=TRAIN_N,
        test_samples=TEST_N,
        seed=SEED,
    )
    train_samples = take_samples(train_stream, TRAIN_N)
    test_samples = take_samples(test_stream, TEST_N)
    ood_samples = _make_ood_samples(OOD_N)

    runners = (
        ("baseline", run_baseline),
        ("hierarchy", run_hierarchy),
        ("ensemble", run_ensemble),
        ("both", run_both),
    )
    results: list[RunResult] = []
    for index, (name, runner) in enumerate(runners, start=1):
        print(f"[{index}/{len(runners)}] running {name}")
        results.append(runner(train_samples, test_samples, ood_samples))

    print("\n=== Bio-ARN Real CIFAR-10 Comparison ===")
    print(f"Data source: {source}")
    print(f"Training: {TRAIN_N} samples, {NUM_PASSES} passes, interleaved")
    print(f"Testing: {TEST_N} samples | OOD: {OOD_N} noise samples\n")
    print(_format_table(results))
    return results


if __name__ == "__main__":
    run_experiment()
