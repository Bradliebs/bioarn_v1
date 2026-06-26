"""Scale the real CIFAR-10 hierarchy experiment with more data and full-stream replay."""

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

from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import load_cifar10_or_synthetic, take_samples

TRAIN_N = 5000
TEST_N = 500
OOD_N = 300
NUM_PASSES = 3
SEED = 7
OOD_SEED = 42
REFERENCE_ACCURACY = 0.264


@dataclass
class ScalingResult:
    accuracy: float
    abstention_rate: float
    ood_auroc: float


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


def run_experiment() -> ScalingResult:
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

    hierarchy = _build_hierarchy()
    interleaved_train = _interleave_by_class(train_samples)
    warmup_end = len(interleaved_train) // 3
    print("=== Bio-ARN CIFAR-10 Hierarchy Scaling ===")
    print(f"Data source: {source}")
    print(f"Training: {TRAIN_N} samples, interleaved, {NUM_PASSES} full passes")
    print(f"Warmup per pass: {warmup_end} unsupervised samples")
    print(f"Evaluation: {TEST_N} test samples | {OOD_N} OOD noise samples")

    pass_progress = max(1, len(interleaved_train) // 4)
    with torch.inference_mode():
        for pass_index in range(NUM_PASSES):
            for index, (tensor, label) in enumerate(interleaved_train, start=1):
                if index <= warmup_end:
                    hierarchy.learn(tensor)
                elif label is not None:
                    hierarchy.learn(tensor, label=int(label))
                if index % pass_progress == 0 or index == len(interleaved_train):
                    phase = "warmup" if index <= warmup_end else "train"
                    print(f"[{phase}] pass {pass_index + 1}/{NUM_PASSES} {index}/{len(interleaved_train)}")

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
            if label is not None and int(predicted) == int(label):
                correct += 1

    ood_scores = [float(hierarchy.classify(tensor)[1]) for tensor in ood_samples]
    result = ScalingResult(
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )
    delta = result.accuracy - REFERENCE_ACCURACY

    print("\nresult:")
    print(f"accuracy: {result.accuracy * 100:.1f}%")
    print(f"abstention_rate: {result.abstention_rate * 100:.1f}%")
    print(f"ood_auroc: {result.ood_auroc:.3f}")
    print(f"delta_vs_hierarchy_2k: {delta * 100:+.1f} percentage points")
    return result


if __name__ == "__main__":
    run_experiment()
