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

from bioarn.config import GNWConfig, PredictiveConfig
from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import (
    EnsembleTrainer,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)
from combined_config_sweep import run_best_combined
from sprint_d_benchmark import run_best_sprint_d

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


@dataclass(frozen=True)
class HybridRoutingPolicy:
    hierarchy_threshold: float
    ensemble_override_margin: float
    min_ensemble_agreement: float


@dataclass(frozen=True)
class HybridPredictionRecord:
    label: int | None
    hierarchy_prediction: int
    hierarchy_confidence: float
    ensemble_prediction: int
    ensemble_confidence: float
    ensemble_agreement: float


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


def _ensemble_signal_strength(result) -> float:
    if result.abstained:
        return 0.0
    agreement_bonus = 0.7 + (0.3 * float(result.agreement))
    abstention_penalty = 1.0 - (0.35 * float(result.abstention_fraction))
    return float(max(0.0, min(1.0, float(result.confidence) * agreement_bonus * abstention_penalty)))


def _collect_hybrid_records(
    hierarchy: VisualHierarchy,
    pool: EnsemblePool,
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[HybridPredictionRecord]:
    records: list[HybridPredictionRecord] = []
    for tensor, label in samples:
        hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
        ensemble_result = pool.classify(tensor)
        records.append(
            HybridPredictionRecord(
                label=None if label is None else int(label),
                hierarchy_prediction=int(hierarchy_prediction),
                hierarchy_confidence=float(hierarchy_confidence),
                ensemble_prediction=-1 if ensemble_result.abstained else int(ensemble_result.predicted_class),
                ensemble_confidence=_ensemble_signal_strength(ensemble_result),
                ensemble_agreement=float(ensemble_result.agreement),
            )
        )
    return records


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


def _workspace_config() -> GNWConfig:
    return GNWConfig(
        capacity=5,
        broadcast_gain=2.2,
        fatigue_rate=0.08,
        fatigue_threshold=0.18,
        competition_temp=0.45,
        context_size=192,
        context_decay=0.97,
        context_update_rate=0.25,
        attention_heads=4,
        context_top_k=6,
    )


def _default_predictive_config(*, mode: str = "error_gating") -> PredictiveConfig:
    return PredictiveConfig(
        gamma=0.12,
        eta=0.008,
        precision_init=1.0,
        error_threshold=0.02,
        settling_steps=6,
        mode=mode,
    )


def _tuned_predictive_config(*, mode: str = "error_gating") -> PredictiveConfig:
    return PredictiveConfig(
        gamma=0.16,
        eta=0.006,
        precision_init=2.4,
        error_threshold=0.01,
        settling_steps=14,
        mode=mode,
    )


def _build_hierarchy(
    *,
    predictive_config: PredictiveConfig | None = None,
    feedback_strength: float = 0.0,
) -> VisualHierarchy:
    return VisualHierarchy(
        HierarchyConfig(
            image_size=(32, 32, 3),
            patch_sizes=[8, 2, 1, 1],
            pool_sizes=[100, 200, 500, 200],
            concept_dims=[32, 64, 128, 64],
            thresholds=[0.25, 0.3, 0.35, 0.4],
            learning_rates=[0.05, 0.03, 0.02, 0.01],
            class_count=10,
            feedback_strength=feedback_strength,
            predictive=predictive_config,
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
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(step_result.confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(step_result.confidence))

    return RunResult(
        name="baseline",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_workspace(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    train_config = _base_train_config()
    train_config.workspace = _workspace_config()
    trainer = VisionTrainer(train_config)
    trainer.train_online(
        train_samples,
        num_samples=TRAIN_N,
        interleave_classes=True,
        num_passes=NUM_PASSES,
    )
    eval_metrics = trainer.evaluate(test_samples, num_samples=TEST_N)

    id_scores: list[float] = []
    for tensor, _ in test_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(step_result.confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(step_result.confidence))

    return RunResult(
        name="workspace",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_curriculum_curiosity(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    train_config = _base_train_config()
    train_config.curiosity_weight = 0.8
    train_config.curriculum = True
    train_config.contrastive_curiosity = True
    trainer = VisionTrainer(train_config)
    trainer.train_online(
        train_samples,
        num_samples=TRAIN_N,
        interleave_classes=True,
        num_passes=NUM_PASSES,
    )
    eval_metrics = trainer.evaluate(test_samples, num_samples=TEST_N)

    id_scores: list[float] = []
    for tensor, _ in test_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(step_result.confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(step_result.confidence))

    return RunResult(
        name="curriculum+curiosity",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_gnw_consensus(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    train_config = _base_train_config()
    workspace = _workspace_config()
    workspace.context_bonus = 0.15  # type: ignore[attr-defined]
    workspace.gnw_learning_gain = 0.8  # type: ignore[attr-defined]
    train_config.workspace = workspace
    trainer = VisionTrainer(train_config)
    trainer.train_online(
        train_samples,
        num_samples=TRAIN_N,
        interleave_classes=True,
        num_passes=NUM_PASSES,
    )
    eval_metrics = trainer.evaluate(test_samples, num_samples=TEST_N)

    id_scores: list[float] = []
    for tensor, _ in test_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        id_scores.append(float(step_result.confidence))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        step_result = trainer._step_pool(  # noqa: SLF001
            trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        ood_scores.append(float(step_result.confidence))

    return RunResult(
        name="gnw_consensus",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_hierarchy(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
    *,
    predictive_config: PredictiveConfig | None = None,
    feedback_strength: float = 0.0,
    predictive_tag: str = "predictive",
) -> RunResult:
    hierarchy = _build_hierarchy(
        predictive_config=predictive_config,
        feedback_strength=feedback_strength,
    )
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
    predictive_enabled = predictive_config is not None
    if predictive_enabled and feedback_strength > 0.0:
        name = f"hierarchy+{predictive_tag}+feedback"
    elif predictive_enabled:
        name = f"hierarchy+{predictive_tag}"
    elif feedback_strength > 0.0:
        name = "hierarchy+feedback"
    else:
        name = "hierarchy"
    return RunResult(
        name=name,
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
    calibration_records = _collect_hybrid_records(hierarchy, pool, interleaved_train[::6])
    ood_records = _collect_hybrid_records(hierarchy, pool, [(tensor, None) for tensor in ood_samples[: min(100, len(ood_samples))]])
    routing_policy = _calibrate_hybrid_policy(
        calibration_records,
        ood_records,
    )

    total = correct = abstained = 0
    id_scores: list[float] = []
    for tensor, label in test_samples:
        final_prediction, final_confidence = _classify_hybrid(hierarchy, pool, tensor, routing_policy)
        total += 1
        if final_prediction == -1:
            abstained += 1
            id_scores.append(0.0)
            continue
        id_scores.append(final_confidence)
        if label is not None and final_prediction == int(label):
            correct += 1

    ood_scores = [
        _classify_hybrid(hierarchy, pool, tensor, routing_policy)[1]
        for tensor in ood_samples
    ]
    return RunResult(
        name="both",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


def run_best_d(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    result = run_best_sprint_d(train_samples, test_samples, ood_samples)
    return RunResult(
        name="best_d",
        accuracy=float(result.accuracy),
        abstention_rate=float(result.abstention_rate),
        ood_auroc=float(result.ood_auroc),
    )


def _classify_hybrid(
    hierarchy: VisualHierarchy,
    pool: EnsemblePool,
    tensor: torch.Tensor,
    policy: HybridRoutingPolicy,
) -> tuple[int, float]:
    hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
    ensemble_result = pool.classify(tensor)
    record = HybridPredictionRecord(
        label=None,
        hierarchy_prediction=int(hierarchy_prediction),
        hierarchy_confidence=float(hierarchy_confidence),
        ensemble_prediction=-1 if ensemble_result.abstained else int(ensemble_result.predicted_class),
        ensemble_confidence=_ensemble_signal_strength(ensemble_result),
        ensemble_agreement=float(ensemble_result.agreement),
    )
    return _classify_hybrid_record(record, policy)


def _classify_hybrid_record(
    record: HybridPredictionRecord,
    policy: HybridRoutingPolicy,
) -> tuple[int, float]:
    if record.hierarchy_prediction == -1:
        if record.ensemble_prediction == -1:
            return -1, 0.0
        return record.ensemble_prediction, record.ensemble_confidence

    if record.ensemble_prediction == -1:
        return record.hierarchy_prediction, record.hierarchy_confidence

    if record.hierarchy_confidence < policy.hierarchy_threshold:
        return record.ensemble_prediction, record.ensemble_confidence

    if (
        record.ensemble_prediction != record.hierarchy_prediction
        and record.ensemble_agreement >= policy.min_ensemble_agreement
        and record.ensemble_confidence > record.hierarchy_confidence + policy.ensemble_override_margin
    ):
        return record.ensemble_prediction, record.ensemble_confidence

    return record.hierarchy_prediction, record.hierarchy_confidence


def _calibrate_hybrid_policy(
    calibration_records: list[HybridPredictionRecord],
    ood_records: list[HybridPredictionRecord],
) -> HybridRoutingPolicy:
    candidate_thresholds = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    candidate_margins = (0.0, 0.05, 0.1, 0.15)
    candidate_agreements = (0.5, 0.6, 0.7, 0.8, 1.0)

    best_policy = HybridRoutingPolicy(
        hierarchy_threshold=0.65,
        ensemble_override_margin=0.05,
        min_ensemble_agreement=0.7,
    )
    best_score = (-1.0, -1.0, -1.0)

    for threshold in candidate_thresholds:
        for margin in candidate_margins:
            for min_agreement in candidate_agreements:
                policy = HybridRoutingPolicy(
                    hierarchy_threshold=threshold,
                    ensemble_override_margin=margin,
                    min_ensemble_agreement=min_agreement,
                )
                accuracy, auroc, ensemble_usage = _evaluate_hybrid_policy(
                    calibration_records,
                    ood_records,
                    policy,
                )
                score = (accuracy, auroc, ensemble_usage)
                if score > best_score:
                    best_score = score
                    best_policy = policy

    return best_policy


def _evaluate_hybrid_policy(
    calibration_records: list[HybridPredictionRecord],
    ood_records: list[HybridPredictionRecord],
    policy: HybridRoutingPolicy,
) -> tuple[float, float, float]:
    total = correct = ensemble_routed = 0
    id_scores: list[float] = []

    for record in calibration_records:
        predicted, confidence = _classify_hybrid_record(record, policy)
        if predicted != record.hierarchy_prediction or abs(confidence - record.hierarchy_confidence) > 1e-6:
            ensemble_routed += 1
        total += 1
        if predicted != -1:
            id_scores.append(float(confidence))
        else:
            id_scores.append(0.0)
        if record.label is not None and predicted == record.label:
            correct += 1

    ood_scores = [_classify_hybrid_record(record, policy)[1] for record in ood_records]
    accuracy = correct / max(total, 1)
    auroc = _auroc(id_scores, ood_scores)
    usage_rate = ensemble_routed / max(total, 1)
    return accuracy, auroc, usage_rate


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
        ("workspace", run_workspace),
        ("curriculum+curiosity", run_curriculum_curiosity),
        ("gnw_consensus", run_gnw_consensus),
        ("hierarchy", run_hierarchy),
        (
            "hierarchy+feedback",
            lambda tr, te, od: run_hierarchy(tr, te, od, feedback_strength=0.2),
        ),
        (
            "hierarchy+settling",
            lambda tr, te, od: run_hierarchy(
                tr,
                te,
                od,
                predictive_config=_default_predictive_config(mode="settling"),
                predictive_tag="settling",
            ),
        ),
        (
            "hierarchy+error_gated",
            lambda tr, te, od: run_hierarchy(
                tr,
                te,
                od,
                predictive_config=_default_predictive_config(mode="error_gating"),
                predictive_tag="error_gated",
            ),
        ),
        (
            "hierarchy+settling+tuned",
            lambda tr, te, od: run_hierarchy(
                tr,
                te,
                od,
                predictive_config=_tuned_predictive_config(mode="settling"),
                predictive_tag="settling+tuned",
            ),
        ),
        (
            "hierarchy+settling+tuned+feedback",
            lambda tr, te, od: run_hierarchy(
                tr,
                te,
                od,
                predictive_config=_tuned_predictive_config(mode="settling"),
                feedback_strength=0.2,
                predictive_tag="settling+tuned",
            ),
        ),
        ("ensemble", run_ensemble),
        ("both", run_both),
        ("best_d", run_best_d),
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
