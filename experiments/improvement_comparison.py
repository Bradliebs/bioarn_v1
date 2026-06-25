"""Compare four BioARN configurations on a synthetic CIFAR-like dataset.

Configurations
--------------
1. baseline    – standard VisionTrainer, no extras
2. hierarchy   – VisualHierarchy (ventral-stream) classification
3. ensemble    – diverse EnsemblePool with weighted voting + Hebbian boosting
4. both        – hierarchy for primary classification, ensemble as fallback

Metrics
-------
- accuracy        : overall correct / total
- abstention_rate : fraction of samples where the model abstained
- ood_auroc       : area under ROC curve using confidence as discriminator
                    (ID test samples = positive, pure-noise samples = negative)

Data
----
Synthetic CIFAR-10 stream (no download required).  The entire experiment runs
in roughly 1-2 minutes on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

import torch

from bioarn.config import BioARNConfig
from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool
from bioarn.hierarchy import HierarchyConfig, VisualHierarchy
from bioarn.training import SyntheticCIFAR10Stream, VisionTrainConfig, VisionTrainer, take_samples

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_N = 600
TEST_N = 200
OOD_N = 150
SEED_TRAIN = 7
SEED_TEST = 8
SEED_OOD = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _make_id_samples(n: int, seed: int, shuffle: bool = True) -> list[tuple[torch.Tensor, int | None]]:
    return take_samples(SyntheticCIFAR10Stream(n, seed=seed, shuffle=shuffle), n)


def _make_ood_samples(n: int, seed: int = SEED_OOD) -> list[torch.Tensor]:
    """Pure uniform noise — maximally out-of-distribution."""
    gen = torch.Generator().manual_seed(seed)
    return [torch.rand(3072, generator=gen) for _ in range(n)]


# ---------------------------------------------------------------------------
# AUROC helper (no sklearn dependency)
# ---------------------------------------------------------------------------

def _auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    """Trapezoidal AUROC; id_scores = positive class, higher score = more ID."""
    positives = [(s, 1) for s in id_scores]
    negatives = [(s, 0) for s in ood_scores]
    all_points = sorted(positives + negatives, key=lambda x: x[0], reverse=True)

    total_pos = len(id_scores)
    total_neg = len(ood_scores)
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    auroc = 0.0
    prev_thresh = None

    for score, label in all_points:
        if prev_thresh is not None and score != prev_thresh:
            auroc += (fp - prev_fp) / total_neg * (tp + prev_tp) / 2 / total_pos
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_thresh = score

    auroc += (fp - prev_fp) / total_neg * (tp + prev_tp) / 2 / total_pos
    auroc += (total_neg - fp) / total_neg * (tp + prev_tp) / 2 / total_pos
    return float(auroc)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    name: str
    accuracy: float
    abstention_rate: float
    ood_auroc: float


# ---------------------------------------------------------------------------
# 1. Baseline
# ---------------------------------------------------------------------------

def run_baseline(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    cfg = BioARNConfig()  # hierarchy=None, ensemble=None

    train_cfg = VisionTrainConfig(
        input_dim=3072,
        concept_dim=128,
        max_pool_size=150,
        margin_threshold=0.35,
        use_batched=True,
        batch_size=32,
        learning_rate=0.02,
        num_train_samples=TRAIN_N,
        num_test_samples=TEST_N,
    )
    _ = cfg  # BioARNConfig carries the intent; VisionTrainer builds its own sub-config
    trainer = VisionTrainer(train_cfg)
    trainer.train_online(train_samples, num_samples=TRAIN_N)
    eval_metrics = trainer.evaluate(test_samples, num_samples=TEST_N)

    # Collect confidence scores for AUROC
    id_scores: list[float] = []
    for tensor, _ in test_samples:
        _, _, conf, _ = trainer._step_pool(trainer._prepare_tensor(tensor), allow_recruit=False)
        id_scores.append(float(conf))

    ood_scores: list[float] = []
    for tensor in ood_samples:
        _, _, conf, _ = trainer._step_pool(trainer._prepare_tensor(tensor.to(torch.float32)), allow_recruit=False)
        ood_scores.append(float(conf))

    return RunResult(
        name="baseline",
        accuracy=float(eval_metrics["accuracy"]),
        abstention_rate=float(eval_metrics["abstention_rate"]),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


# ---------------------------------------------------------------------------
# 2. With Hierarchy
# ---------------------------------------------------------------------------

def run_hierarchy(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    cfg = BioARNConfig(hierarchy=HierarchyConfig())

    hierarchy = VisualHierarchy(cfg.hierarchy)

    # Phase 1: unsupervised warmup
    warmup = train_samples[: TRAIN_N // 3]
    for tensor, _ in warmup:
        hierarchy.learn(tensor)

    # Phase 2: supervised learning
    for tensor, label in train_samples[TRAIN_N // 3 :]:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))

    # Evaluate
    total = correct = covered = abstained = 0
    id_scores: list[float] = []

    for tensor, label in test_samples:
        predicted, conf = hierarchy.classify(tensor)
        total += 1
        if predicted == -1:
            abstained += 1
            id_scores.append(0.0)
        else:
            covered += 1
            id_scores.append(float(conf))
            if label is not None and predicted == label:
                correct += 1

    ood_scores: list[float] = []
    for tensor in ood_samples:
        _, conf = hierarchy.classify(tensor.to(torch.float32))
        ood_scores.append(float(conf))

    return RunResult(
        name="hierarchy",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


# ---------------------------------------------------------------------------
# 3. With Ensemble
# ---------------------------------------------------------------------------

def _build_ensemble() -> EnsemblePool:
    cfg = EnsembleConfig(
        num_experts=4,
        voting_method="weighted",
        abstention_threshold=0.5,
        use_boosting=True,
        diversity_target=0.3,
    )
    manager = DiversityManager()
    expert_configs = manager.create_diverse_experts(
        {
            "input_dim": 3072,
            "concept_dim": 128,
            "max_pool_size": 150,
            "learning_rate": 0.02,
            "image_size": (32, 32, 3),
            "num_classes": 10,
        },
        num_experts=4,
    )
    cfg.expert_configs = expert_configs
    return EnsemblePool(cfg)


def run_ensemble(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    cfg = BioARNConfig(ensemble=EnsembleConfig())

    _ = cfg  # EnsembleConfig used to configure the pool below
    pool = _build_ensemble()

    for tensor, label in train_samples:
        for expert_state in pool.experts:
            pool._learn_expert(expert_state, tensor, label)  # noqa: SLF001

    total = correct = covered = abstained = 0
    id_scores: list[float] = []

    for tensor, label in test_samples:
        result = pool.classify(tensor)
        total += 1
        if result.abstained:
            abstained += 1
            id_scores.append(0.0)
        else:
            covered += 1
            id_scores.append(float(result.confidence))
            if label is not None and result.predicted_class == label:
                correct += 1

    ood_scores: list[float] = []
    for tensor in ood_samples:
        result = pool.classify(tensor.to(torch.float32))
        ood_scores.append(float(result.confidence))

    return RunResult(
        name="ensemble",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


# ---------------------------------------------------------------------------
# 4. With Both
# ---------------------------------------------------------------------------

def run_both(
    train_samples: list[tuple[torch.Tensor, int | None]],
    test_samples: list[tuple[torch.Tensor, int | None]],
    ood_samples: list[torch.Tensor],
) -> RunResult:
    """Hierarchy as primary classifier; ensemble acts as a tie-breaking fallback.

    Decision logic:
    - If hierarchy fires (predicted != -1): use hierarchy prediction.
    - Else if ensemble doesn't abstain: use ensemble prediction.
    - Else: abstain.
    """
    cfg = BioARNConfig(hierarchy=HierarchyConfig(), ensemble=EnsembleConfig())

    hierarchy = VisualHierarchy(cfg.hierarchy)
    pool = _build_ensemble()

    # Train both
    warmup = train_samples[: TRAIN_N // 3]
    for tensor, _ in warmup:
        hierarchy.learn(tensor)

    for tensor, label in train_samples[TRAIN_N // 3 :]:
        if label is not None:
            hierarchy.learn(tensor, label=int(label))
        for expert_state in pool.experts:
            pool._learn_expert(expert_state, tensor, label)  # noqa: SLF001

    # Evaluate combined
    total = correct = abstained = 0
    id_scores: list[float] = []

    for tensor, label in test_samples:
        h_pred, h_conf = hierarchy.classify(tensor)
        total += 1

        if h_pred != -1:
            final_pred = h_pred
            final_conf = float(h_conf)
        else:
            e_result = pool.classify(tensor)
            if not e_result.abstained:
                final_pred = e_result.predicted_class
                final_conf = float(e_result.confidence)
            else:
                final_pred = -1
                final_conf = 0.0

        if final_pred == -1:
            abstained += 1
            id_scores.append(0.0)
        else:
            id_scores.append(final_conf)
            if label is not None and final_pred == label:
                correct += 1

    ood_scores: list[float] = []
    for tensor in ood_samples:
        h_pred, h_conf = hierarchy.classify(tensor.to(torch.float32))
        if h_pred != -1:
            ood_scores.append(float(h_conf))
        else:
            e_result = pool.classify(tensor.to(torch.float32))
            ood_scores.append(float(e_result.confidence))

    return RunResult(
        name="both",
        accuracy=correct / max(total, 1),
        abstention_rate=abstained / max(total, 1),
        ood_auroc=_auroc(id_scores, ood_scores),
    )


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _format_table(rows: list[RunResult]) -> str:
    headers = ["config", "accuracy", "abstention", "ood_auroc"]
    values = [
        [
            row.name,
            f"{row.accuracy:.3f}",
            f"{row.abstention_rate:.3f}",
            f"{row.ood_auroc:.3f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in values))
        for i in range(len(headers))
    ]
    divider = "-+-".join("-" * w for w in widths)
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body = [
        " | ".join(v.ljust(widths[i]) for i, v in enumerate(row))
        for row in values
    ]
    return "\n".join([header_line, divider, *body])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> list[RunResult]:
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))

    print(f"data_source: synthetic-cifar10  train={TRAIN_N}  test={TEST_N}  ood={OOD_N}")

    train_samples = _make_id_samples(TRAIN_N, SEED_TRAIN)
    test_samples = _make_id_samples(TEST_N, SEED_TEST, shuffle=False)
    ood_samples = _make_ood_samples(OOD_N)

    results: list[RunResult] = []

    print("\n--- [1/4] baseline ---")
    results.append(run_baseline(train_samples, test_samples, ood_samples))

    print("\n--- [2/4] hierarchy ---")
    results.append(run_hierarchy(train_samples, test_samples, ood_samples))

    print("\n--- [3/4] ensemble ---")
    results.append(run_ensemble(train_samples, test_samples, ood_samples))

    print("\n--- [4/4] both ---")
    results.append(run_both(train_samples, test_samples, ood_samples))

    print("\n\n=== COMPARISON TABLE ===")
    print(_format_table(results))

    best_acc = max(results, key=lambda r: r.accuracy)
    best_ood = max(results, key=lambda r: r.ood_auroc)
    print(f"\nbest_accuracy  : {best_acc.name}  ({best_acc.accuracy:.3f})")
    print(f"best_ood_auroc : {best_ood.name}  ({best_ood.ood_auroc:.3f})")

    return results


if __name__ == "__main__":
    run_experiment()
