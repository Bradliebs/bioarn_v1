"""Sprint D combined benchmark for Bio-ARN on real CIFAR-10."""

from __future__ import annotations

import contextlib
import io
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import GNWConfig, PredictiveConfig, STDPConfig
from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import (
    EnsembleTrainer,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

SEED = 7
OOD_SEED = 42
BASELINE_TRAIN_N = 500
WINNER_1K_TRAIN_N = 1000
WINNER_2K_TRAIN_N = 2000
TEST_N = 200
OOD_N = 200


@dataclass(frozen=True)
class SprintDConfigSpec:
    name: str
    train_samples: int
    tier: str
    predictive_mode: str | None = None
    curiosity_weight: float = 0.0
    curriculum: bool = False
    contrastive_curiosity: bool = False
    workspace: bool = False
    stdp: bool = False
    feedback_strength: float = 0.0
    ensemble: bool = False
    num_passes: int = 1
    interleave_classes: bool = False


@dataclass
class BenchmarkResult:
    name: str
    tier: str
    train_samples: int
    accuracy: float
    abstention_rate: float
    ood_auroc: float
    train_seconds: float
    hierarchy_seconds: float
    workspace_seconds: float
    ensemble_seconds: float
    replay_samples: int
    predictive_mode: str | None
    curiosity_weight: float
    curriculum: bool
    contrastive_curiosity: bool
    workspace_enabled: bool
    stdp_enabled: bool
    feedback_strength: float
    ensemble_enabled: bool
    num_passes: int
    interleave_classes: bool


@dataclass
class WorkspaceClassifier:
    trainer: VisionTrainer
    train_seconds: float

    def predict(self, tensor: torch.Tensor) -> tuple[int | None, float]:
        step_result = self.trainer._step_pool(  # noqa: SLF001
            self.trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        if step_result.abstained:
            return None, float(step_result.confidence)
        return self.trainer._recognition_label(  # noqa: SLF001
            step_result.concept_direction,
            step_result.fired_indices,
        ), float(step_result.confidence)


@dataclass
class EnsembleClassifier:
    pool: EnsemblePool
    train_seconds: float

    def predict(self, tensor: torch.Tensor) -> tuple[int | None, float, float]:
        result = self.pool.classify(tensor)
        if result.abstained:
            return None, 0.0, float(result.agreement)
        return int(result.predicted_class), _ensemble_signal_strength(result), float(result.agreement)


@dataclass(frozen=True)
class RoutingPolicy:
    hierarchy_threshold: float
    aux_margin: float
    min_ensemble_agreement: float


@dataclass
class RoutingRecord:
    label: int | None
    hierarchy_prediction: int
    hierarchy_confidence: float
    workspace_prediction: int | None
    workspace_confidence: float
    ensemble_prediction: int | None
    ensemble_confidence: float
    ensemble_agreement: float


@dataclass(frozen=True)
class AuxiliaryCandidate:
    source: str
    prediction: int
    confidence: float
    agreement: float = 1.0


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


def _workspace_config() -> GNWConfig:
    workspace = GNWConfig(
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
    workspace.context_bonus = 0.15  # type: ignore[attr-defined]
    workspace.gnw_learning_gain = 0.8  # type: ignore[attr-defined]
    return workspace


def _predictive_config(mode: str = "error_gating") -> PredictiveConfig:
    return PredictiveConfig(
        gamma=0.12,
        eta=0.008,
        precision_init=1.0,
        error_threshold=0.02,
        settling_steps=6,
        mode=mode,
    )


def _build_hierarchy(spec: SprintDConfigSpec) -> VisualHierarchy:
    predictive = _predictive_config(spec.predictive_mode) if spec.predictive_mode else None
    return VisualHierarchy(
        HierarchyConfig(
            image_size=(32, 32, 3),
            patch_sizes=[8, 2, 1, 1],
            pool_sizes=[100, 200, 500, 200],
            concept_dims=[32, 64, 128, 64],
            thresholds=[0.25, 0.3, 0.35, 0.4],
            learning_rates=[0.05, 0.03, 0.02, 0.01],
            class_count=10,
            feedback_strength=spec.feedback_strength,
            predictive=predictive,
            stdp=STDPConfig() if spec.stdp else None,
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


def _order_samples(
    samples: list[tuple[torch.Tensor, int | None]],
    hierarchy: VisualHierarchy,
    spec: SprintDConfigSpec,
    *,
    pass_index: int,
) -> list[tuple[torch.Tensor, int | None]]:
    ordered = list(samples)
    if spec.curriculum:
        scored: list[tuple[int, float, int, torch.Tensor, int | None]] = []
        with torch.inference_mode():
            for tensor, label in ordered:
                predicted, confidence = hierarchy.classify(tensor)
                is_easy = int(label is not None and predicted == int(label))
                label_key = -1 if label is None else int(label)
                scored.append((is_easy, float(confidence), label_key, tensor, label))
        if spec.interleave_classes:
            buckets: defaultdict[int, list[tuple[int, float, int, torch.Tensor, int | None]]] = defaultdict(list)
            unlabeled: list[tuple[int, float, int, torch.Tensor, int | None]] = []
            for item in sorted(scored, key=lambda row: (-row[0], -row[1], row[2])):
                if item[4] is None:
                    unlabeled.append(item)
                else:
                    buckets[int(item[4])].append(item)
            ranked: list[tuple[torch.Tensor, int | None]] = []
            classes = sorted(buckets)
            while any(buckets[class_id] for class_id in classes):
                for class_id in classes:
                    if buckets[class_id]:
                        _, _, _, tensor, label = buckets[class_id].pop(0)
                        ranked.append((tensor, label))
            ranked.extend((tensor, label) for _, _, _, tensor, label in unlabeled)
            return ranked
        return [(tensor, label) for _, _, _, tensor, label in sorted(scored, key=lambda row: (-row[0], -row[1], row[2]))]

    if spec.interleave_classes:
        ordered = _interleave_by_class(ordered)
    elif pass_index > 0 and ordered:
        order = torch.randperm(len(ordered)).tolist()
        ordered = [ordered[index] for index in order]
    return ordered


def _curiosity_threshold(weight: float) -> float:
    normalized = min(max(weight, 0.0), 1.5) / 1.5
    return 0.55 + (0.15 * normalized)


def _replay_candidates(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    spec: SprintDConfigSpec,
) -> list[tuple[torch.Tensor, int | None]]:
    if spec.curiosity_weight <= 0.0:
        return []

    threshold = _curiosity_threshold(spec.curiosity_weight)
    candidates: list[tuple[float, int, float, torch.Tensor, int | None]] = []
    with torch.inference_mode():
        for tensor, label in train_samples:
            if label is None:
                continue
            predicted, confidence = hierarchy.classify(tensor)
            confidence = float(confidence)
            incorrect = predicted != int(label)
            if incorrect or confidence < threshold:
                boundary_priority = 1.0 - confidence
                priority = boundary_priority + (0.75 if incorrect else 0.0)
                if spec.contrastive_curiosity:
                    priority += 0.5 * boundary_priority
                candidates.append((priority, int(label), confidence, tensor, label))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(tensor, label) for _, _, _, tensor, label in candidates]


def _apply_curiosity_replay(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    spec: SprintDConfigSpec,
) -> tuple[int, float]:
    replay_samples = _replay_candidates(hierarchy, train_samples, spec)
    if not replay_samples:
        return 0, 0.0

    replay_start = time.perf_counter()
    ordered = _interleave_by_class(replay_samples) if spec.interleave_classes else replay_samples
    with torch.inference_mode():
        for tensor, label in ordered:
            if label is not None:
                hierarchy.learn(tensor, int(label))

        if spec.contrastive_curiosity and len(ordered) >= 4:
            focused = ordered[: max(1, len(ordered) // 2)]
            for tensor, label in focused:
                if label is not None:
                    hierarchy.learn(tensor, int(label))
    return len(replay_samples), time.perf_counter() - replay_start


def _train_hierarchy(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    spec: SprintDConfigSpec,
) -> tuple[float, int]:
    initial_order = _order_samples(train_samples, hierarchy, spec, pass_index=0)
    warmup_end = max(1, len(initial_order) // 3)
    warmup = initial_order[:warmup_end]
    labelled = initial_order[warmup_end:]
    replay_samples = 0

    start = time.perf_counter()
    with torch.inference_mode():
        for tensor, _ in warmup:
            hierarchy.learn(tensor)

        for pass_index in range(max(1, int(spec.num_passes))):
            current = _order_samples(labelled, hierarchy, spec, pass_index=pass_index)
            for tensor, label in current:
                if label is not None:
                    hierarchy.learn(tensor, int(label))
            replay_count, _ = _apply_curiosity_replay(hierarchy, current, spec)
            replay_samples += replay_count
    return time.perf_counter() - start, replay_samples


def _train_workspace_classifier(
    train_samples: list[tuple[torch.Tensor, int | None]],
    spec: SprintDConfigSpec,
) -> WorkspaceClassifier:
    config = VisionTrainConfig(
        input_dim=3072,
        concept_dim=256,
        max_pool_size=384,
        margin_threshold=0.35,
        use_batched=True,
        batch_size=32,
        learning_rate=0.01,
        num_train_samples=len(train_samples),
        num_test_samples=TEST_N,
        preprocessing_warmup_samples=200,
        curiosity_weight=spec.curiosity_weight,
        curriculum=spec.curriculum,
        contrastive_curiosity=spec.contrastive_curiosity,
        workspace=_workspace_config(),
    )
    trainer = VisionTrainer(config)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_online(
            train_samples,
            num_samples=len(train_samples),
            num_passes=max(1, int(spec.num_passes)),
            interleave_classes=spec.interleave_classes,
        )
    return WorkspaceClassifier(trainer=trainer, train_seconds=time.perf_counter() - start)


def _train_ensemble_classifier(
    train_samples: list[tuple[torch.Tensor, int | None]],
    spec: SprintDConfigSpec,
) -> EnsembleClassifier:
    pool = _build_ensemble()
    base_samples = _interleave_by_class(train_samples) if spec.interleave_classes else list(train_samples)
    trainer = EnsembleTrainer(pool, log_every=max(1, len(base_samples) // 10))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for pass_index in range(max(1, int(spec.num_passes))):
            if pass_index == 0:
                current = base_samples
            else:
                order = torch.randperm(len(base_samples)).tolist()
                current = [base_samples[index] for index in order]
            trainer.train(current)
    return EnsembleClassifier(pool=pool, train_seconds=time.perf_counter() - start)


def _ensemble_signal_strength(result) -> float:
    if result.abstained:
        return 0.0
    agreement_bonus = 0.7 + (0.3 * float(result.agreement))
    abstention_penalty = 1.0 - (0.35 * float(result.abstention_fraction))
    return float(max(0.0, min(1.0, float(result.confidence) * agreement_bonus * abstention_penalty)))


def _collect_routing_records(
    hierarchy: VisualHierarchy,
    workspace: WorkspaceClassifier | None,
    ensemble: EnsembleClassifier | None,
    samples: list[tuple[torch.Tensor, int | None]],
) -> list[RoutingRecord]:
    records: list[RoutingRecord] = []
    with torch.inference_mode():
        for tensor, label in samples:
            hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
            workspace_prediction, workspace_confidence = (
                workspace.predict(tensor) if workspace is not None else (None, 0.0)
            )
            ensemble_prediction, ensemble_confidence, ensemble_agreement = (
                ensemble.predict(tensor) if ensemble is not None else (None, 0.0, 0.0)
            )
            records.append(
                RoutingRecord(
                    label=None if label is None else int(label),
                    hierarchy_prediction=int(hierarchy_prediction),
                    hierarchy_confidence=float(hierarchy_confidence),
                    workspace_prediction=None if workspace_prediction is None else int(workspace_prediction),
                    workspace_confidence=float(workspace_confidence),
                    ensemble_prediction=None if ensemble_prediction is None else int(ensemble_prediction),
                    ensemble_confidence=float(ensemble_confidence),
                    ensemble_agreement=float(ensemble_agreement),
                )
            )
    return records


def _best_auxiliary(record: RoutingRecord) -> AuxiliaryCandidate | None:
    candidates: list[AuxiliaryCandidate] = []
    if record.workspace_prediction is not None:
        candidates.append(
            AuxiliaryCandidate(
                source="workspace",
                prediction=int(record.workspace_prediction),
                confidence=float(record.workspace_confidence),
            )
        )
    if record.ensemble_prediction is not None:
        candidates.append(
            AuxiliaryCandidate(
                source="ensemble",
                prediction=int(record.ensemble_prediction),
                confidence=float(record.ensemble_confidence),
                agreement=float(record.ensemble_agreement),
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.confidence, candidate.agreement))


def _combine_predictions(
    record: RoutingRecord,
    policy: RoutingPolicy | None,
) -> tuple[int, float]:
    if policy is None:
        return record.hierarchy_prediction, record.hierarchy_confidence

    auxiliary = _best_auxiliary(record)
    if record.hierarchy_prediction == -1:
        if auxiliary is None:
            return -1, 0.0
        return auxiliary.prediction, auxiliary.confidence

    if auxiliary is None:
        return record.hierarchy_prediction, record.hierarchy_confidence

    if auxiliary.prediction == record.hierarchy_prediction:
        return record.hierarchy_prediction, max(record.hierarchy_confidence, auxiliary.confidence)

    if auxiliary.source == "ensemble" and auxiliary.agreement < policy.min_ensemble_agreement:
        return record.hierarchy_prediction, record.hierarchy_confidence

    if (
        record.hierarchy_confidence < policy.hierarchy_threshold
        and auxiliary.confidence > record.hierarchy_confidence + policy.aux_margin
    ):
        return auxiliary.prediction, auxiliary.confidence

    if (
        auxiliary.source == "ensemble"
        and auxiliary.agreement >= policy.min_ensemble_agreement
        and auxiliary.confidence > record.hierarchy_confidence + policy.aux_margin
    ):
        return auxiliary.prediction, auxiliary.confidence

    return record.hierarchy_prediction, record.hierarchy_confidence


def _evaluate_policy(
    calibration_records: list[RoutingRecord],
    ood_records: list[RoutingRecord],
    policy: RoutingPolicy,
) -> tuple[float, float, float]:
    total = correct = rerouted = 0
    id_scores: list[float] = []
    for record in calibration_records:
        final_prediction, final_confidence = _combine_predictions(record, policy)
        baseline_aux = _best_auxiliary(record)
        if baseline_aux is not None and final_prediction == baseline_aux.prediction and final_prediction != record.hierarchy_prediction:
            rerouted += 1
        total += 1
        id_scores.append(0.0 if final_prediction == -1 else float(final_confidence))
        if record.label is not None and final_prediction == int(record.label):
            correct += 1

    ood_scores = [float(_combine_predictions(record, policy)[1]) for record in ood_records]
    return correct / max(total, 1), _auroc(id_scores, ood_scores), rerouted / max(total, 1)


def _calibrate_routing_policy(
    calibration_records: list[RoutingRecord],
    ood_records: list[RoutingRecord],
) -> RoutingPolicy:
    thresholds = (0.45, 0.55, 0.65, 0.75)
    margins = (0.0, 0.05, 0.1, 0.15)
    agreements = (0.5, 0.7, 1.0)
    best_policy = RoutingPolicy(hierarchy_threshold=0.55, aux_margin=0.1, min_ensemble_agreement=0.7)
    best_score = (-1.0, -1.0, -1.0)

    for threshold in thresholds:
        for margin in margins:
            for min_agreement in agreements:
                policy = RoutingPolicy(
                    hierarchy_threshold=threshold,
                    aux_margin=margin,
                    min_ensemble_agreement=min_agreement,
                )
                score = _evaluate_policy(calibration_records, ood_records, policy)
                if score > best_score:
                    best_score = score
                    best_policy = policy
    return best_policy


def run_sprint_d_config(
    spec: SprintDConfigSpec,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> BenchmarkResult:
    subset = train_samples[: spec.train_samples]
    hierarchy = _build_hierarchy(spec)
    hierarchy_seconds, replay_samples = _train_hierarchy(hierarchy, subset, spec)

    workspace_classifier: WorkspaceClassifier | None = None
    if spec.workspace:
        workspace_classifier = _train_workspace_classifier(subset, spec)

    ensemble_classifier: EnsembleClassifier | None = None
    if spec.ensemble:
        ensemble_classifier = _train_ensemble_classifier(subset, spec)

    routing_policy: RoutingPolicy | None = None
    if workspace_classifier is not None or ensemble_classifier is not None:
        stride = max(1, len(subset) // 64)
        calibration_records = _collect_routing_records(
            hierarchy,
            workspace_classifier,
            ensemble_classifier,
            subset[::stride],
        )
        ood_records = _collect_routing_records(
            hierarchy,
            workspace_classifier,
            ensemble_classifier,
            [(tensor, None) for tensor in ood_samples[: max(32, OOD_N // 2)]],
        )
        routing_policy = _calibrate_routing_policy(calibration_records, ood_records)

    total = correct = abstained = 0
    id_scores: list[float] = []
    evaluation_records = _collect_routing_records(
        hierarchy,
        workspace_classifier,
        ensemble_classifier,
        test_samples,
    )
    for record in evaluation_records:
        final_prediction, final_confidence = _combine_predictions(record, routing_policy)
        total += 1
        if final_prediction == -1:
            abstained += 1
            id_scores.append(0.0)
            continue
        id_scores.append(float(final_confidence))
        if record.label is not None and final_prediction == int(record.label):
            correct += 1

    ood_records = _collect_routing_records(
        hierarchy,
        workspace_classifier,
        ensemble_classifier,
        [(tensor, None) for tensor in ood_samples],
    )
    ood_scores = [float(_combine_predictions(record, routing_policy)[1]) for record in ood_records]

    workspace_seconds = (
        0.0 if workspace_classifier is None else float(workspace_classifier.train_seconds)
    )
    ensemble_seconds = (
        0.0 if ensemble_classifier is None else float(ensemble_classifier.train_seconds)
    )
    return BenchmarkResult(
        name=spec.name,
        tier=spec.tier,
        train_samples=spec.train_samples,
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
        train_seconds=float(hierarchy_seconds + workspace_seconds + ensemble_seconds),
        hierarchy_seconds=float(hierarchy_seconds),
        workspace_seconds=workspace_seconds,
        ensemble_seconds=ensemble_seconds,
        replay_samples=int(replay_samples),
        predictive_mode=spec.predictive_mode,
        curiosity_weight=float(spec.curiosity_weight),
        curriculum=bool(spec.curriculum),
        contrastive_curiosity=bool(spec.contrastive_curiosity),
        workspace_enabled=bool(spec.workspace),
        stdp_enabled=bool(spec.stdp),
        feedback_strength=float(spec.feedback_strength),
        ensemble_enabled=bool(spec.ensemble),
        num_passes=int(spec.num_passes),
        interleave_classes=bool(spec.interleave_classes),
    )


def _tier_1_specs() -> list[SprintDConfigSpec]:
    return [
        SprintDConfigSpec(name="baseline", train_samples=BASELINE_TRAIN_N, tier="tier1"),
        SprintDConfigSpec(
            name="error-gated",
            train_samples=BASELINE_TRAIN_N,
            tier="tier1",
            predictive_mode="error_gating",
        ),
        SprintDConfigSpec(
            name="curriculum",
            train_samples=BASELINE_TRAIN_N,
            tier="tier1",
            curiosity_weight=0.8,
            curriculum=True,
        ),
        SprintDConfigSpec(
            name="gnw-consensus",
            train_samples=BASELINE_TRAIN_N,
            tier="tier1",
            workspace=True,
        ),
    ]


def _tier_2_specs() -> list[SprintDConfigSpec]:
    return [
        SprintDConfigSpec(
            name="best-d1",
            train_samples=BASELINE_TRAIN_N,
            tier="tier2",
            predictive_mode="error_gating",
            curiosity_weight=0.8,
            curriculum=True,
        ),
        SprintDConfigSpec(
            name="best-d1-gnw",
            train_samples=BASELINE_TRAIN_N,
            tier="tier2",
            predictive_mode="error_gating",
            curiosity_weight=0.8,
            curriculum=True,
            workspace=True,
        ),
        SprintDConfigSpec(
            name="best-d1-multi",
            train_samples=BASELINE_TRAIN_N,
            tier="tier2",
            predictive_mode="error_gating",
            curiosity_weight=0.8,
            curriculum=True,
            num_passes=2,
            interleave_classes=True,
        ),
        SprintDConfigSpec(
            name="kitchen-sink",
            train_samples=BASELINE_TRAIN_N,
            tier="tier2",
            predictive_mode="error_gating",
            curiosity_weight=0.8,
            curriculum=True,
            contrastive_curiosity=True,
            workspace=True,
            stdp=True,
            feedback_strength=0.2,
            ensemble=True,
            num_passes=2,
            interleave_classes=True,
        ),
    ]


def _format_table(rows: list[BenchmarkResult]) -> str:
    headers = ("Tier", "Config", "Train", "Acc", "OOD", "Time(s)", "Replay")
    values = [
        (
            row.tier,
            row.name,
            str(row.train_samples),
            f"{row.accuracy * 100:.1f}%",
            f"{row.ood_auroc:.3f}",
            f"{row.train_seconds:.1f}",
            str(row.replay_samples),
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


def _best_result(rows: list[BenchmarkResult]) -> BenchmarkResult:
    return max(rows, key=lambda row: (row.accuracy, row.ood_auroc, -row.train_seconds))


BEST_SPRINT_D_SPEC = SprintDConfigSpec(
    name="best_d",
    train_samples=WINNER_2K_TRAIN_N,
    tier="best",
    predictive_mode="error_gating",
    curiosity_weight=0.8,
    curriculum=True,
    num_passes=2,
    interleave_classes=True,
)


def run_best_sprint_d(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> BenchmarkResult:
    spec = replace(
        BEST_SPRINT_D_SPEC,
        train_samples=min(int(BEST_SPRINT_D_SPEC.train_samples), len(train_samples)),
    )
    return run_sprint_d_config(spec, train_samples, test_samples, ood_samples)


def run_benchmark() -> tuple[list[BenchmarkResult], BenchmarkResult, BenchmarkResult]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))

    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=WINNER_2K_TRAIN_N,
        test_samples=TEST_N,
        seed=SEED,
    )
    train_samples = take_samples(train_stream, WINNER_2K_TRAIN_N)
    test_samples = take_samples(test_stream, TEST_N)
    ood_samples = _make_ood_samples(OOD_N)

    print("=== Bio-ARN Sprint D Combined Benchmark ===")
    print(f"Data source: {source}")
    print(f"Shared eval: {TEST_N} test | {OOD_N} OOD")

    results: list[BenchmarkResult] = []

    tier_1_results: list[BenchmarkResult] = []
    print(f"\nTier 1 — individual features ({BASELINE_TRAIN_N} train)")
    for index, spec in enumerate(_tier_1_specs(), start=1):
        print(f"[tier1 {index}/{len(_tier_1_specs())}] {spec.name}")
        result = run_sprint_d_config(spec, train_samples, test_samples, ood_samples)
        tier_1_results.append(result)
        results.append(result)
        print(
            f"  acc={result.accuracy * 100:.1f}% "
            f"ood={result.ood_auroc:.3f} "
            f"time={result.train_seconds:.1f}s"
        )

    tier_2_results: list[BenchmarkResult] = []
    print(f"\nTier 2 — best combinations ({BASELINE_TRAIN_N} train)")
    for index, spec in enumerate(_tier_2_specs(), start=1):
        print(f"[tier2 {index}/{len(_tier_2_specs())}] {spec.name}")
        result = run_sprint_d_config(spec, train_samples, test_samples, ood_samples)
        tier_2_results.append(result)
        results.append(result)
        print(
            f"  acc={result.accuracy * 100:.1f}% "
            f"ood={result.ood_auroc:.3f} "
            f"time={result.train_seconds:.1f}s"
        )

    winner_tier_2 = _best_result(tier_2_results)
    winner_feature_spec = next(spec for spec in _tier_2_specs() if spec.name == winner_tier_2.name)
    scaling_specs = [
        replace(winner_feature_spec, name="winner-1k", tier="tier3", train_samples=WINNER_1K_TRAIN_N),
        replace(winner_feature_spec, name="winner-2k", tier="tier3", train_samples=WINNER_2K_TRAIN_N),
    ]

    print("\nTier 3 — scaling the Tier 2 winner")
    for index, spec in enumerate(scaling_specs, start=1):
        print(f"[tier3 {index}/{len(scaling_specs)}] {spec.name} ({winner_tier_2.name})")
        result = run_sprint_d_config(spec, train_samples, test_samples, ood_samples)
        results.append(result)
        print(
            f"  acc={result.accuracy * 100:.1f}% "
            f"ood={result.ood_auroc:.3f} "
            f"time={result.train_seconds:.1f}s"
        )

    best_overall = _best_result(results)
    print("\n=== Results ===")
    print(_format_table(results))
    print(
        "\nTier 2 winner: "
        f"{winner_tier_2.name} | acc={winner_tier_2.accuracy * 100:.1f}% | "
        f"ood={winner_tier_2.ood_auroc:.3f} | train={winner_tier_2.train_seconds:.1f}s"
    )
    print(
        "Best overall: "
        f"{best_overall.name} | acc={best_overall.accuracy * 100:.1f}% | "
        f"ood={best_overall.ood_auroc:.3f} | train={best_overall.train_seconds:.1f}s"
    )
    print(
        "Targets: "
        f"500-train >=35% {'yes' if any(row.train_samples == BASELINE_TRAIN_N and row.accuracy >= 0.35 for row in results) else 'no'} | "
        f"2000-train >=40% {'yes' if any(row.train_samples == WINNER_2K_TRAIN_N and row.accuracy >= 0.40 for row in results) else 'no'}"
    )
    return results, winner_tier_2, best_overall


if __name__ == "__main__":
    run_benchmark()
