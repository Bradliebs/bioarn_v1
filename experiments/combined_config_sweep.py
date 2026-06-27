"""Sweep combined Bio-ARN configurations on real CIFAR-10."""

from __future__ import annotations

import contextlib
import copy
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

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, MarginGateConfig, PredictiveConfig, SDMConfig, STDPConfig
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.scaling import ScaledBioARN
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples

SEED = 7
OOD_SEED = 42
TEST_N = 200
OOD_N = 200
INITIAL_TRAIN_N = 800
MAX_TRAIN_N = 2000


@dataclass(frozen=True)
class CombinedConfigSpec:
    name: str
    train_samples: int
    workspace: bool
    curiosity_weight: float
    predictive: bool
    stdp: bool
    feedback_strength: float
    hierarchy_pool_scale: float = 1.0
    hierarchy_passes: int = 1
    aux_train_passes: int = 2


@dataclass
class CombinedRunResult:
    name: str
    train_samples: int
    accuracy: float
    abstention_rate: float
    ood_auroc: float
    train_seconds: float
    hierarchy_seconds: float
    workspace_seconds: float
    replay_samples: int
    workspace_enabled: bool
    curiosity_weight: float
    predictive: bool
    stdp: bool
    feedback_strength: float
    hierarchy_pool_scale: float
    hierarchy_passes: int


@dataclass(frozen=True)
class RoutingPolicy:
    hierarchy_threshold: float
    workspace_margin: float


@dataclass
class WorkspaceClassifier:
    trainer: VisionTrainer
    train_seconds: float

    def predict(self, tensor: torch.Tensor) -> tuple[int | None, float]:
        fired_indices, concept, confidence, abstained = self.trainer._step_pool(  # noqa: SLF001
            self.trainer._prepare_tensor(tensor),  # noqa: SLF001
            allow_recruit=False,
        )
        if abstained:
            return None, float(confidence)
        return self.trainer._recognition_label(concept, fired_indices), float(confidence)  # noqa: SLF001


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


def _scaled_pool_sizes(scale: float) -> tuple[list[int], list[int]]:
    base_pool_sizes = [100, 200, 500, 200]
    base_max_pool_sizes = [150, 300, 750, 600]
    if abs(scale - 1.0) < 1e-6:
        return base_pool_sizes, base_max_pool_sizes
    pool_sizes = [max(base, int(round(base * scale))) for base in base_pool_sizes]
    max_pool_sizes = [
        max(max_size, int(round(max_size * scale)))
        for max_size in base_max_pool_sizes
    ]
    return pool_sizes, max_pool_sizes


def _build_hierarchy(spec: CombinedConfigSpec) -> VisualHierarchy:
    pool_sizes, max_pool_sizes = _scaled_pool_sizes(spec.hierarchy_pool_scale)
    return VisualHierarchy(
        HierarchyConfig(
            image_size=(32, 32, 3),
            patch_sizes=[8, 2, 1, 1],
            pool_sizes=pool_sizes,
            max_pool_sizes=max_pool_sizes,
            concept_dims=[32, 64, 128, 64],
            thresholds=[0.25, 0.3, 0.35, 0.4],
            learning_rates=[0.05, 0.03, 0.02, 0.01],
            class_count=10,
            feedback_strength=spec.feedback_strength,
            predictive=(
                PredictiveConfig(
                    gamma=0.12,
                    eta=0.008,
                    precision_init=1.0,
                    error_threshold=0.02,
                    settling_steps=6,
                )
                if spec.predictive
                else None
            ),
            stdp=STDPConfig() if spec.stdp else None,
        )
    )


def _train_hierarchy(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    *,
    num_passes: int,
) -> float:
    ordered = _interleave_by_class(train_samples)
    warmup_end = max(1, len(ordered) // 3)
    start = time.perf_counter()
    with torch.inference_mode():
        for pass_index in range(max(1, int(num_passes))):
            if pass_index == 0:
                current = ordered
            else:
                order = torch.randperm(len(ordered)).tolist()
                current = [ordered[index] for index in order]
            for sample_index, (tensor, label) in enumerate(current):
                if pass_index == 0 and sample_index < warmup_end:
                    hierarchy.learn(tensor)
                elif label is not None:
                    hierarchy.learn(tensor, int(label))
    return time.perf_counter() - start


def _curiosity_threshold(weight: float) -> float:
    normalized = min(max(weight, 0.0), 1.5) / 1.5
    return 0.55 + (0.15 * normalized)


def _replay_candidates(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    *,
    curiosity_weight: float,
) -> list[tuple[torch.Tensor, int | None]]:
    if curiosity_weight <= 0.0:
        return []
    threshold = _curiosity_threshold(curiosity_weight)
    candidates: list[tuple[torch.Tensor, int | None, float]] = []
    with torch.inference_mode():
        for tensor, label in train_samples:
            if label is None:
                continue
            predicted, confidence = hierarchy.classify(tensor)
            if predicted != int(label) or confidence < threshold:
                candidates.append((tensor, label, float(confidence)))
    candidates.sort(key=lambda item: (int(item[1]), item[2]))
    return [(tensor, label) for tensor, label, _ in candidates]


def _apply_curiosity_replay(
    hierarchy: VisualHierarchy,
    train_samples: list[tuple[torch.Tensor, int | None]],
    *,
    curiosity_weight: float,
) -> tuple[int, float]:
    replay_samples = _replay_candidates(
        hierarchy,
        train_samples,
        curiosity_weight=curiosity_weight,
    )
    if not replay_samples:
        return 0, 0.0

    replay_start = time.perf_counter()
    with torch.inference_mode():
        ordered = _interleave_by_class(replay_samples)
        for tensor, label in ordered:
            if label is not None:
                hierarchy.learn(tensor, int(label))

        if curiosity_weight > 1.0 and len(ordered) >= 4:
            focused = ordered[: max(1, len(ordered) // 2)]
            for tensor, label in focused:
                if label is not None:
                    hierarchy.learn(tensor, int(label))
    return len(replay_samples), time.perf_counter() - replay_start


def _build_workspace_system(config: VisionTrainConfig) -> ScaledBioARN:
    ccc_features = 64 if config.input_dim >= 1024 else max(16, min(64, config.input_dim))
    workspace_config = copy.deepcopy(config.workspace)
    bio_config = BioARNConfig(
        ccc=CCCConfig(
            input_dim=config.input_dim,
            concept_dim=config.concept_dim,
            num_f1_features=ccc_features,
            f1_top_k=max(8, ccc_features // 4),
            fast_lr=1.0,
            slow_lr=config.learning_rate,
            feedback_lr=config.learning_rate,
            max_pool_size=config.max_pool_size,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=config.margin_threshold,
            theta_margin_lr=0.001,
            theta_resonance=min(0.9, config.margin_threshold + 0.25),
        ),
        sdm=SDMConfig(
            address_dim=max(512, config.concept_dim * 4),
            hamming_radius=max(16, config.concept_dim // 4),
            num_hard_locations=256,
            data_dim=config.concept_dim,
            decay_rate=0.999,
            stdp_window=10,
        ),
        gnw=copy.deepcopy(workspace_config) if workspace_config is not None else GNWConfig(),
        workspace=workspace_config,
        seed=42,
    )
    return ScaledBioARN(bio_config, use_optimized=True)


def _train_workspace_classifier(
    train_samples: list[tuple[torch.Tensor, int | None]],
    *,
    curiosity_weight: float,
    num_passes: int,
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
        curiosity_weight=curiosity_weight,
        workspace=_workspace_config(),
    )
    trainer = VisionTrainer(config)
    trainer.system = _build_workspace_system(config)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_online(
            train_samples,
            num_samples=len(train_samples),
            num_passes=max(1, int(num_passes)),
            interleave_classes=True,
        )
    return WorkspaceClassifier(trainer=trainer, train_seconds=time.perf_counter() - start)


def _combine_predictions(
    hierarchy_prediction: int,
    hierarchy_confidence: float,
    workspace_prediction: int | None,
    workspace_confidence: float,
    policy: RoutingPolicy | None,
) -> tuple[int, float]:
    if policy is None or workspace_prediction is None:
        return hierarchy_prediction, hierarchy_confidence
    if hierarchy_prediction == -1:
        return int(workspace_prediction), float(workspace_confidence)
    if (
        hierarchy_confidence < policy.hierarchy_threshold
        and workspace_confidence > hierarchy_confidence + policy.workspace_margin
    ):
        return int(workspace_prediction), float(workspace_confidence)
    return hierarchy_prediction, hierarchy_confidence


def _calibrate_routing_policy(
    hierarchy: VisualHierarchy,
    workspace: WorkspaceClassifier,
    calibration_samples: list[tuple[torch.Tensor, int | None]],
) -> RoutingPolicy:
    thresholds = (0.45, 0.55, 0.65)
    margins = (0.05, 0.1, 0.15)
    best = RoutingPolicy(hierarchy_threshold=0.55, workspace_margin=0.1)
    best_accuracy = -1.0

    cached_predictions: list[tuple[int, float, int | None, float, int | None]] = []
    with torch.inference_mode():
        for tensor, label in calibration_samples:
            hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
            workspace_prediction, workspace_confidence = workspace.predict(tensor)
            cached_predictions.append(
                (
                    int(hierarchy_prediction),
                    float(hierarchy_confidence),
                    None if workspace_prediction is None else int(workspace_prediction),
                    float(workspace_confidence),
                    None if label is None else int(label),
                )
            )

    for threshold in thresholds:
        for margin in margins:
            policy = RoutingPolicy(hierarchy_threshold=threshold, workspace_margin=margin)
            correct = total = 0
            for hierarchy_prediction, hierarchy_confidence, workspace_prediction, workspace_confidence, label in cached_predictions:
                final_prediction, _ = _combine_predictions(
                    hierarchy_prediction,
                    hierarchy_confidence,
                    workspace_prediction,
                    workspace_confidence,
                    policy,
                )
                if label is not None:
                    total += 1
                    correct += int(final_prediction == label)
            accuracy = correct / max(total, 1)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best = policy

    return best


def run_combined_config(
    spec: CombinedConfigSpec,
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> CombinedRunResult:
    subset = train_samples[: spec.train_samples]
    hierarchy = _build_hierarchy(spec)
    hierarchy_seconds = _train_hierarchy(
        hierarchy,
        subset,
        num_passes=spec.hierarchy_passes,
    )
    replay_samples, replay_seconds = _apply_curiosity_replay(
        hierarchy,
        subset,
        curiosity_weight=spec.curiosity_weight,
    )

    workspace_classifier: WorkspaceClassifier | None = None
    routing_policy: RoutingPolicy | None = None
    if spec.workspace:
        workspace_classifier = _train_workspace_classifier(
            subset,
            curiosity_weight=spec.curiosity_weight,
            num_passes=spec.aux_train_passes,
        )
        routing_policy = _calibrate_routing_policy(
            hierarchy,
            workspace_classifier,
            subset[:: max(1, len(subset) // 64)],
        )

    total = correct = abstained = 0
    id_scores: list[float] = []
    with torch.inference_mode():
        for tensor, label in test_samples:
            hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
            workspace_prediction, workspace_confidence = (
                workspace_classifier.predict(tensor)
                if workspace_classifier is not None
                else (None, 0.0)
            )
            final_prediction, final_confidence = _combine_predictions(
                hierarchy_prediction,
                hierarchy_confidence,
                workspace_prediction,
                workspace_confidence,
                routing_policy,
            )
            total += 1
            if final_prediction == -1:
                abstained += 1
                id_scores.append(0.0)
                continue
            id_scores.append(float(final_confidence))
            if label is not None and final_prediction == int(label):
                correct += 1

    ood_scores: list[float] = []
    with torch.inference_mode():
        for tensor in ood_samples:
            hierarchy_prediction, hierarchy_confidence = hierarchy.classify(tensor)
            workspace_prediction, workspace_confidence = (
                workspace_classifier.predict(tensor)
                if workspace_classifier is not None
                else (None, 0.0)
            )
            _, final_confidence = _combine_predictions(
                hierarchy_prediction,
                hierarchy_confidence,
                workspace_prediction,
                workspace_confidence,
                routing_policy,
            )
            ood_scores.append(float(final_confidence))

    workspace_seconds = (
        0.0 if workspace_classifier is None else float(workspace_classifier.train_seconds)
    )
    total_train_seconds = hierarchy_seconds + replay_seconds + workspace_seconds
    return CombinedRunResult(
        name=spec.name,
        train_samples=spec.train_samples,
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
        train_seconds=float(total_train_seconds),
        hierarchy_seconds=float(hierarchy_seconds + replay_seconds),
        workspace_seconds=workspace_seconds,
        replay_samples=int(replay_samples),
        workspace_enabled=spec.workspace,
        curiosity_weight=spec.curiosity_weight,
        predictive=spec.predictive,
        stdp=spec.stdp,
        feedback_strength=spec.feedback_strength,
        hierarchy_pool_scale=spec.hierarchy_pool_scale,
        hierarchy_passes=spec.hierarchy_passes,
    )


def _initial_specs() -> list[CombinedConfigSpec]:
    return [
        CombinedConfigSpec(
            name="baseline",
            train_samples=INITIAL_TRAIN_N,
            workspace=False,
            curiosity_weight=0.0,
            predictive=False,
            stdp=False,
            feedback_strength=0.0,
        ),
        CombinedConfigSpec(
            name="curiosity-only",
            train_samples=INITIAL_TRAIN_N,
            workspace=False,
            curiosity_weight=0.8,
            predictive=False,
            stdp=False,
            feedback_strength=0.0,
        ),
        CombinedConfigSpec(
            name="workspace+curiosity",
            train_samples=INITIAL_TRAIN_N,
            workspace=True,
            curiosity_weight=0.8,
            predictive=False,
            stdp=False,
            feedback_strength=0.0,
        ),
        CombinedConfigSpec(
            name="full-no-feedback",
            train_samples=INITIAL_TRAIN_N,
            workspace=True,
            curiosity_weight=0.8,
            predictive=True,
            stdp=True,
            feedback_strength=0.0,
        ),
        CombinedConfigSpec(
            name="full",
            train_samples=INITIAL_TRAIN_N,
            workspace=True,
            curiosity_weight=0.8,
            predictive=True,
            stdp=True,
            feedback_strength=0.1,
        ),
        CombinedConfigSpec(
            name="full-strong-feedback",
            train_samples=INITIAL_TRAIN_N,
            workspace=True,
            curiosity_weight=0.8,
            predictive=True,
            stdp=True,
            feedback_strength=0.3,
        ),
        CombinedConfigSpec(
            name="kitchen-sink",
            train_samples=INITIAL_TRAIN_N,
            workspace=True,
            curiosity_weight=1.0,
            predictive=True,
            stdp=True,
            feedback_strength=0.2,
            hierarchy_pool_scale=1.5,
            hierarchy_passes=2,
        ),
    ]


def _escalation_specs(best_initial: CombinedRunResult) -> list[CombinedConfigSpec]:
    base_spec = next(spec for spec in _initial_specs() if spec.name == best_initial.name)
    specs = [
        CombinedConfigSpec(
            name="baseline-2k",
            train_samples=MAX_TRAIN_N,
            workspace=False,
            curiosity_weight=0.0,
            predictive=False,
            stdp=False,
            feedback_strength=0.0,
        ),
        replace(
            base_spec,
            name=f"{base_spec.name}-2k",
            train_samples=MAX_TRAIN_N,
        ),
        CombinedConfigSpec(
            name="curiosity-only-2k-curiosity1.5",
            train_samples=MAX_TRAIN_N,
            workspace=False,
            curiosity_weight=1.5,
            predictive=False,
            stdp=False,
            feedback_strength=0.0,
            hierarchy_passes=2,
        ),
    ]
    unique: dict[str, CombinedConfigSpec] = {}
    for spec in specs:
        unique[spec.name] = spec
    return list(unique.values())


def _format_table(rows: list[CombinedRunResult]) -> str:
    headers = ("Config", "Train", "Acc", "OOD", "Time(s)", "Replay", "Workspace")
    values = [
        (
            row.name,
            str(row.train_samples),
            f"{row.accuracy * 100:.1f}%",
            f"{row.ood_auroc:.3f}",
            f"{row.train_seconds:.1f}",
            str(row.replay_samples),
            "yes" if row.workspace_enabled else "no",
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


def _best_result(rows: list[CombinedRunResult]) -> CombinedRunResult:
    return max(rows, key=lambda row: (row.accuracy, row.ood_auroc, -row.train_seconds))


def run_combined_sweep() -> tuple[list[CombinedRunResult], CombinedRunResult]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))

    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=MAX_TRAIN_N,
        test_samples=TEST_N,
        seed=SEED,
    )
    train_samples = take_samples(train_stream, MAX_TRAIN_N)
    test_samples = take_samples(test_stream, TEST_N)
    ood_samples = _make_ood_samples(OOD_N)

    print("=== Bio-ARN Combined CIFAR-10 Sweep ===")
    print(f"Data source: {source}")
    print(f"Phase 1: {INITIAL_TRAIN_N} train | {TEST_N} test | {OOD_N} OOD")

    results: list[CombinedRunResult] = []
    initial_specs = _initial_specs()
    for index, spec in enumerate(initial_specs, start=1):
        print(f"[phase1 {index}/{len(initial_specs)}] {spec.name}")
        result = run_combined_config(spec, train_samples, test_samples, ood_samples)
        results.append(result)
        print(
            f"  acc={result.accuracy * 100:.1f}% "
            f"ood={result.ood_auroc:.3f} "
            f"time={result.train_seconds:.1f}s"
        )

    best = _best_result(results)
    if best.accuracy < 0.40:
        print(
            "\nNo phase-1 config reached 40%; escalating to 2k training samples "
            "and larger combined settings."
        )
        escalation_specs = _escalation_specs(best)
        for index, spec in enumerate(escalation_specs, start=1):
            print(f"[phase2 {index}/{len(escalation_specs)}] {spec.name}")
            result = run_combined_config(spec, train_samples, test_samples, ood_samples)
            results.append(result)
            print(
                f"  acc={result.accuracy * 100:.1f}% "
                f"ood={result.ood_auroc:.3f} "
                f"time={result.train_seconds:.1f}s"
            )
        best = _best_result(results)

    print("\n=== Results ===")
    print(_format_table(results))
    print(
        "\nBest config: "
        f"{best.name} | acc={best.accuracy * 100:.1f}% | "
        f"ood={best.ood_auroc:.3f} | train={best.train_seconds:.1f}s"
    )
    return results, best


BEST_COMBINED_SPEC = CombinedConfigSpec(
    name="baseline-2k",
    train_samples=MAX_TRAIN_N,
    workspace=False,
    curiosity_weight=0.0,
    predictive=False,
    stdp=False,
    feedback_strength=0.0,
)


def run_best_combined(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> CombinedRunResult:
    return run_combined_config(BEST_COMBINED_SPEC, train_samples, test_samples, ood_samples)


if __name__ == "__main__":
    run_combined_sweep()
