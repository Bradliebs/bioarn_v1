"""Formal lateral-prediction ablation for MNIST-family OOD detection."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import GNWConfig, LateralPredictionConfig, PrecisionConfig
from bioarn.data import FashionMNISTStream, MNISTStream
from bioarn.training import VisionTrainConfig, VisionTrainer

TRAIN_PER_CLASS = 500
TEST_ID_PER_CLASS = 100
TEST_OOD_PER_CLASS = 100
NUM_PASSES = 1
POOL_SIZE = 100
SEED = 7
CONFIDENCE_PENALTY = 0.15


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    stream_cls: type[MNISTStream] | type[FashionMNISTStream]
    id_classes: tuple[int, ...] = tuple(range(8))
    ood_classes: tuple[int, ...] = (8, 9)
    seed_offset: int = 0


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    use_gnw: bool
    use_lateral: bool
    use_precision: bool


@dataclass(frozen=True)
class AblationResult:
    config: str
    id_accuracy: float
    auroc: float
    aupr: float
    fpr95: float
    mean_confidence: float
    mean_lateral_error: float
    mean_uncertainty: float
    train_seconds: float
    committed_cccs: int


def _collect_balanced_samples(
    stream_cls: type[MNISTStream] | type[FashionMNISTStream],
    *,
    split: str,
    classes: Sequence[int],
    per_class: int,
    data_dir: Path,
    seed: int,
    shuffle: bool,
) -> list[tuple[torch.Tensor, int]]:
    counts: defaultdict[int, int] = defaultdict(int)
    target_classes = tuple(int(label) for label in classes)
    target_set = set(target_classes)
    samples: list[tuple[torch.Tensor, int]] = []
    stream = stream_cls(
        split=split,
        data_dir=str(data_dir),
        flatten=True,
        normalize=True,
        shuffle=shuffle,
        seed=seed,
    )
    for sample in stream.stream():
        label = int(sample.label)
        if label not in target_set or counts[label] >= per_class:
            continue
        samples.append((sample.data.to(torch.float32), label))
        counts[label] += 1
        if all(counts[label] >= per_class for label in target_classes):
            return samples
    raise RuntimeError(
        f"Unable to collect {per_class} samples for classes {target_classes}; counts={dict(counts)}"
    )


def _workspace_config() -> GNWConfig:
    return GNWConfig(
        capacity=7,
        broadcast_gain=2.0,
        fatigue_rate=0.1,
        fatigue_threshold=0.3,
        competition_temp=1.0,
    )


def _lateral_config() -> LateralPredictionConfig:
    return LateralPredictionConfig(
        enabled=True,
        max_neighbors=8,
        hebbian_lr=0.05,
        anti_hebbian_lr=0.02,
        prediction_threshold=0.10,
        surprise_gain=1.5,
    )


def _precision_config(pool_size: int) -> PrecisionConfig:
    return PrecisionConfig(
        enabled=True,
        pool_size=pool_size,
        entropy_window=100,
        precision_alpha=5.0,
        precision_threshold=0.5,
        min_precision=0.1,
        max_precision=1.0,
        lateral_error_weight=0.35,
        hierarchy_error_weight=0.0,
        external_signal_decay=0.85,
        surprise_gain=1.5,
    )


def _trainer_config(
    spec: ConfigSpec,
    *,
    num_train_samples: int,
    num_test_samples: int,
) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=784,
        concept_dim=128,
        max_pool_size=POOL_SIZE,
        margin_threshold=0.50,
        use_batched=True,
        batch_size=32,
        learning_rate=0.02,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        preprocessing_warmup_samples=128,
        workspace=_workspace_config() if spec.use_gnw else None,
        lateral_prediction=_lateral_config() if spec.use_lateral else None,
        precision=_precision_config(POOL_SIZE) if spec.use_precision else None,
        num_f1_features=256,
        f1_top_k=50,
    )


def _clear_workspace_state(trainer: VisionTrainer) -> None:
    if getattr(trainer.system, "workspace_enabled", False):
        trainer.system.gnw.clear()
        trainer.system.last_thought = trainer.system._empty_thought()


def _preview_step(
    trainer: VisionTrainer,
    tensor: torch.Tensor,
) -> tuple[object, int | None]:
    _clear_workspace_state(trainer)
    prepared = trainer._prepare_tensor(tensor)  # noqa: SLF001
    step = trainer._step_pool(prepared, allow_recruit=False, preview=True)  # noqa: SLF001
    prediction = (
        None
        if step.abstained
        else trainer._recognition_label(step.concept_direction, step.fired_indices)  # noqa: SLF001
    )
    return step, prediction


def _ood_score(trainer: VisionTrainer, step) -> tuple[float, float, float]:
    pool = trainer.system.ccc_pool
    confidence = max(0.0, min(1.0, float(step.confidence)))
    lateral_error = (
        float(pool.get_lateral_prediction_error())
        if getattr(pool, "lateral_network", None) is not None
        else 0.0
    )
    precision_gate = getattr(pool, "precision_gate", None)
    uncertainty = (
        float(getattr(precision_gate, "current_uncertainty", 0.0))
        if precision_gate is not None
        else 0.0
    )
    score = confidence - (CONFIDENCE_PENALTY * lateral_error) - (CONFIDENCE_PENALTY * uncertainty)
    return max(0.0, min(1.0, score)), lateral_error, uncertainty


def _evaluate_id(trainer: VisionTrainer, samples: Sequence[tuple[torch.Tensor, int]]) -> dict[str, float | list[float]]:
    correct = 0
    scores: list[float] = []
    confidences: list[float] = []
    lateral_errors: list[float] = []
    uncertainties: list[float] = []
    for tensor, label in samples:
        step, prediction = _preview_step(trainer, tensor)
        score, lateral_error, uncertainty = _ood_score(trainer, step)
        correct += int(prediction == label)
        scores.append(score)
        confidences.append(float(step.confidence))
        lateral_errors.append(lateral_error)
        uncertainties.append(uncertainty)
    return {
        "accuracy": correct / max(len(samples), 1),
        "scores": scores,
        "mean_confidence": fmean(confidences) if confidences else 0.0,
        "mean_lateral_error": fmean(lateral_errors) if lateral_errors else 0.0,
        "mean_uncertainty": fmean(uncertainties) if uncertainties else 0.0,
    }


def _evaluate_ood(trainer: VisionTrainer, samples: Sequence[tuple[torch.Tensor, int]]) -> dict[str, float | list[float]]:
    scores: list[float] = []
    confidences: list[float] = []
    lateral_errors: list[float] = []
    uncertainties: list[float] = []
    for tensor, _ in samples:
        step, _ = _preview_step(trainer, tensor)
        score, lateral_error, uncertainty = _ood_score(trainer, step)
        scores.append(score)
        confidences.append(float(step.confidence))
        lateral_errors.append(lateral_error)
        uncertainties.append(uncertainty)
    return {
        "scores": scores,
        "mean_confidence": fmean(confidences) if confidences else 0.0,
        "mean_lateral_error": fmean(lateral_errors) if lateral_errors else 0.0,
        "mean_uncertainty": fmean(uncertainties) if uncertainties else 0.0,
    }


def _fpr_at_95_tpr(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
    labels = ([1] * len(id_scores)) + ([0] * len(ood_scores))
    scores = list(id_scores) + list(ood_scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = [float(fp) for fp, tp in zip(fpr, tpr, strict=False) if float(tp) >= 0.95]
    return min(valid) if valid else 1.0


def _run_config(
    dataset: DatasetSpec,
    config_spec: ConfigSpec,
    *,
    data_dir: Path,
    train_samples: Sequence[tuple[torch.Tensor, int]],
    id_test_samples: Sequence[tuple[torch.Tensor, int]],
    ood_test_samples: Sequence[tuple[torch.Tensor, int]],
    num_passes: int,
) -> AblationResult:
    print(f"[run] {dataset.name} / {config_spec.name}", flush=True)
    torch.manual_seed(SEED + dataset.seed_offset)
    trainer = VisionTrainer(
        _trainer_config(
            config_spec,
            num_train_samples=len(train_samples),
            num_test_samples=len(id_test_samples),
        )
    )
    _ = data_dir
    start = time.perf_counter()
    trainer.train_online(
        list(train_samples),
        num_samples=len(train_samples),
        num_passes=num_passes,
        interleave_classes=True,
    )
    train_seconds = time.perf_counter() - start
    id_eval = _evaluate_id(trainer, id_test_samples)
    ood_eval = _evaluate_ood(trainer, ood_test_samples)
    id_scores = list(id_eval["scores"])
    ood_scores = list(ood_eval["scores"])
    labels = ([1] * len(id_scores)) + ([0] * len(ood_scores))
    scores = id_scores + ood_scores
    stats = trainer.system.ccc_pool.get_pool_stats()
    return AblationResult(
        config=config_spec.name,
        id_accuracy=float(id_eval["accuracy"]),
        auroc=float(roc_auc_score(labels, scores)),
        aupr=float(average_precision_score(labels, scores)),
        fpr95=float(_fpr_at_95_tpr(id_scores, ood_scores)),
        mean_confidence=float(id_eval["mean_confidence"]),
        mean_lateral_error=float(
            0.5 * (float(id_eval["mean_lateral_error"]) + float(ood_eval["mean_lateral_error"]))
        ),
        mean_uncertainty=float(
            0.5 * (float(id_eval["mean_uncertainty"]) + float(ood_eval["mean_uncertainty"]))
        ),
        train_seconds=float(train_seconds),
        committed_cccs=int(stats["num_committed"]),
    )


def _print_table(dataset_name: str, results: Sequence[AblationResult]) -> None:
    print(f"\n{dataset_name} Ablation:", flush=True)
    print("| Config | ID Acc | AUROC | AUPR | FPR@95 |", flush=True)
    print("|--------|--------|-------|------|--------|", flush=True)
    for result in results:
        print(
            f"| {result.config} | "
            f"{result.id_accuracy * 100:5.1f}% | "
            f"{result.auroc:0.3f} | "
            f"{result.aupr:0.3f} | "
            f"{result.fpr95:0.3f} |"
        , flush=True)


def _print_summary(dataset_name: str, results: Sequence[AblationResult]) -> None:
    base = next(result for result in results if result.config == "Base")
    gnw = next(result for result in results if result.config == "+GNW")
    lateral = next(result for result in results if result.config == "+Lateral")
    gnw_lateral = next(result for result in results if result.config == "+GNW+Lateral")
    full = next(result for result in results if result.config == "+GNW+Lateral+Precision")
    print(f"\n{dataset_name} summary:", flush=True)
    print(
        f"- GNW vs Base: AUROC {base.auroc:0.3f} -> {gnw.auroc:0.3f} "
        f"({(gnw.auroc - base.auroc) * 100:+0.1f} pts)"
    , flush=True)
    print(
        f"- Lateral vs Base: AUROC {base.auroc:0.3f} -> {lateral.auroc:0.3f} "
        f"({(lateral.auroc - base.auroc) * 100:+0.1f} pts)"
    , flush=True)
    print(
        f"- GNW+Lateral vs GNW: AUROC {gnw.auroc:0.3f} -> {gnw_lateral.auroc:0.3f} "
        f"({(gnw_lateral.auroc - gnw.auroc) * 100:+0.1f} pts)"
    , flush=True)
    print(
        f"- Full vs GNW+Lateral: AUROC {gnw_lateral.auroc:0.3f} -> {full.auroc:0.3f} "
        f"({(full.auroc - gnw_lateral.auroc) * 100:+0.1f} pts)"
    , flush=True)


def run_ablation(
    *,
    data_dir: Path,
    train_per_class: int,
    test_id_per_class: int,
    test_ood_per_class: int,
    num_passes: int,
    dataset_filter: str,
) -> dict[str, list[AblationResult]]:
    configs = [
        ConfigSpec(name="Base", use_gnw=False, use_lateral=False, use_precision=False),
        ConfigSpec(name="+GNW", use_gnw=True, use_lateral=False, use_precision=False),
        ConfigSpec(name="+Lateral", use_gnw=False, use_lateral=True, use_precision=False),
        ConfigSpec(name="+GNW+Lateral", use_gnw=True, use_lateral=True, use_precision=False),
        ConfigSpec(name="+GNW+Lateral+Precision", use_gnw=True, use_lateral=True, use_precision=True),
    ]
    datasets = [
        DatasetSpec(name="MNIST", stream_cls=MNISTStream, seed_offset=0),
        DatasetSpec(name="Fashion-MNIST", stream_cls=FashionMNISTStream, seed_offset=100),
    ]
    if dataset_filter == "mnist":
        datasets = [datasets[0]]
    elif dataset_filter == "fashion":
        datasets = [datasets[1]]
    all_results: dict[str, list[AblationResult]] = {}
    for dataset in datasets:
        print(
            f"\n[dataset] {dataset.name} "
            f"(train {train_per_class}/class, ID test {test_id_per_class}/class, OOD test {test_ood_per_class}/class)"
        , flush=True)
        train_samples = _collect_balanced_samples(
            dataset.stream_cls,
            split="train",
            classes=dataset.id_classes,
            per_class=train_per_class,
            data_dir=data_dir,
            seed=SEED + dataset.seed_offset,
            shuffle=True,
        )
        id_test_samples = _collect_balanced_samples(
            dataset.stream_cls,
            split="test",
            classes=dataset.id_classes,
            per_class=test_id_per_class,
            data_dir=data_dir,
            seed=SEED + dataset.seed_offset + 1,
            shuffle=False,
        )
        ood_test_samples = _collect_balanced_samples(
            dataset.stream_cls,
            split="test",
            classes=dataset.ood_classes,
            per_class=test_ood_per_class,
            data_dir=data_dir,
            seed=SEED + dataset.seed_offset + 2,
            shuffle=False,
        )
        results = [
            _run_config(
                dataset,
                config_spec,
                data_dir=data_dir,
                train_samples=train_samples,
                id_test_samples=id_test_samples,
                ood_test_samples=ood_test_samples,
                num_passes=num_passes,
            )
            for config_spec in configs
        ]
        all_results[dataset.name] = results
        _print_table(dataset.name, results)
        _print_summary(dataset.name, results)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lateral-prediction OOD ablation on MNIST-family data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset",
        choices=("all", "mnist", "fashion"),
        default="all",
        help="Run all datasets or just one benchmark family.",
    )
    parser.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--test-id-per-class", type=int, default=TEST_ID_PER_CLASS)
    parser.add_argument("--test-ood-per-class", type=int, default=TEST_OOD_PER_CLASS)
    parser.add_argument("--num-passes", type=int, default=NUM_PASSES)
    args = parser.parse_args()

    results = run_ablation(
        data_dir=args.data_dir,
        train_per_class=max(1, int(args.train_per_class)),
        test_id_per_class=max(1, int(args.test_id_per_class)),
        test_ood_per_class=max(1, int(args.test_ood_per_class)),
        num_passes=max(1, int(args.num_passes)),
        dataset_filter=str(args.dataset),
    )
    print("\nPaper-ready conclusion:", flush=True)
    for dataset_name, dataset_results in results.items():
        base = next(result for result in dataset_results if result.config == "Base")
        full = next(result for result in dataset_results if result.config == "+GNW+Lateral+Precision")
        lateral = next(result for result in dataset_results if result.config == "+Lateral")
        print(
            f"- {dataset_name}: lateral-only changed AUROC by {(lateral.auroc - base.auroc) * 100:+0.1f} pts; "
            f"the full stack changed AUROC by {(full.auroc - base.auroc) * 100:+0.1f} pts."
        , flush=True)


if __name__ == "__main__":
    main()
