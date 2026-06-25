"""Test whether interleaved class presentation improves Bio-ARN's MNIST accuracy.

The existing mnist_poc.py achieves ~82% accuracy using CCCPool + LocalPrototypeBank
with one-pass streaming training. This experiment tests whether interleaving
class order (the same technique that gave +63% on synthetic CIFAR-10) can push
MNIST accuracy higher.

Configurations tested:
1. baseline     — original sequential order (same as mnist_poc.py)
2. interleaved  — round-robin class order before training
3. multi-pass   — 2 passes over the data (second pass shuffled)
4. both         — interleaved first pass + shuffled second pass
"""

from __future__ import annotations

import argparse
import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bioarn.config import BioARNConfig, CCCConfig, MarginGateConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import cosine_similarity, normalize
from experiments.mnist_poc import (
    LocalPrototypeBank,
    calibrate_abstention_threshold,
    collect_samples,
    evaluate_classifier,
    format_percent,
    load_mnist,
)

DEFAULT_TRAIN_SAMPLES = 5000
DEFAULT_TEST_SAMPLES = 1000
DEFAULT_CALIBRATION_SAMPLES = 200
CURRENT_HEADLINE = 0.82


@dataclass
class RunResult:
    name: str
    accuracy: float
    covered_accuracy: float
    abstention_rate: float
    cccs_committed: int
    prototype_entries: int
    train_time: float


def build_config(seed: int) -> BioARNConfig:
    """Same config as mnist_poc.py for fair comparison."""
    return BioARNConfig(
        seed=seed,
        ccc=CCCConfig(
            input_dim=784,
            concept_dim=128,
            num_f1_features=256,
            f1_top_k=50,
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=100,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.50,
            theta_margin_lr=0.0005,
            theta_resonance=0.65,
        ),
    )


def interleave_by_class(
    samples: list[tuple[torch.Tensor, int]],
) -> list[tuple[torch.Tensor, int]]:
    """Round-robin interleave samples by class label."""
    buckets: defaultdict[int, list[tuple[torch.Tensor, int]]] = defaultdict(list)
    for tensor, label in samples:
        buckets[label].append((tensor, label))

    interleaved: list[tuple[torch.Tensor, int]] = []
    keys = sorted(buckets.keys())
    max_len = max(len(buckets[k]) for k in keys) if keys else 0
    for i in range(max_len):
        for k in keys:
            if i < len(buckets[k]):
                interleaved.append(buckets[k][i])
    return interleaved


def train_streaming(
    pool: CCCPool,
    bank: LocalPrototypeBank,
    samples: list[tuple[torch.Tensor, int]],
    *,
    timestep_offset: int = 0,
    label: str = "",
) -> float:
    """One pass of streaming training. Returns elapsed time."""
    start = time.perf_counter()
    for idx, (vector, lbl) in enumerate(samples):
        pool_output = pool(vector, timestep=timestep_offset + idx)
        bank.observe(vector, lbl)
        if (idx + 1) % 1000 == 0:
            stats = pool.get_pool_stats()
            print(
                f"    {label}{idx + 1:>5}/{len(samples)} "
                f"cccs={stats['num_committed']:>3} protos={bank.num_entries:>3}"
            )
    return time.perf_counter() - start


def run_configuration(
    name: str,
    train_samples: list[tuple[torch.Tensor, int]],
    test_samples: list[tuple[torch.Tensor, int]],
    calibration_samples: list[tuple[torch.Tensor, int]],
    *,
    interleave: bool = False,
    num_passes: int = 1,
    seed: int = 42,
) -> RunResult:
    """Run one configuration and return results."""
    torch.manual_seed(seed)
    config = build_config(seed)
    pool = CCCPool(config.ccc, config.margin_gate)
    bank = LocalPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)

    # Prepare training order
    ordered_samples = interleave_by_class(train_samples) if interleave else list(train_samples)

    # Pass 1
    print(f"\n  [{name}] Pass 1/{num_passes} ({len(ordered_samples)} samples)...")
    elapsed = train_streaming(pool, bank, ordered_samples, timestep_offset=0, label=f"{name} p1 ")

    # Additional passes (shuffled)
    for pass_idx in range(1, num_passes):
        perm = torch.randperm(len(train_samples)).tolist()
        shuffled = [train_samples[i] for i in perm]
        print(f"  [{name}] Pass {pass_idx + 1}/{num_passes} (shuffled)...")
        elapsed += train_streaming(
            pool, bank, shuffled,
            timestep_offset=len(train_samples) * pass_idx,
            label=f"{name} p{pass_idx + 1} ",
        )

    # Evaluate
    threshold_result = calibrate_abstention_threshold(bank, calibration_samples)
    classification = evaluate_classifier(
        bank, test_samples, threshold=threshold_result.threshold
    )
    stats = pool.get_pool_stats()

    print(
        f"  [{name}] accuracy={format_percent(classification.overall_accuracy)} "
        f"covered={format_percent(classification.covered_accuracy)} "
        f"abstention={format_percent(classification.abstention_rate)} "
        f"cccs={stats['num_committed']}"
    )

    return RunResult(
        name=name,
        accuracy=classification.overall_accuracy,
        covered_accuracy=classification.covered_accuracy,
        abstention_rate=classification.abstention_rate,
        cccs_committed=int(stats["num_committed"]),
        prototype_entries=bank.num_entries,
        train_time=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bio-ARN MNIST: interleaved training comparison.")
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(__file__).resolve().parents[1] / "data"
    print("Loading MNIST...")
    train_dataset = load_mnist(data_root, train=True)
    test_dataset = load_mnist(data_root, train=False)

    train_samples = collect_samples(train_dataset, limit=args.train_samples)
    calibration_samples = collect_samples(
        train_dataset, limit=args.calibration_samples, offset=args.train_samples
    )
    if len(calibration_samples) < args.calibration_samples:
        calibration_samples = train_samples[-args.calibration_samples:]
    test_samples = collect_samples(test_dataset, limit=args.test_samples)

    print(f"Train: {len(train_samples)} | Test: {len(test_samples)} | Calibration: {len(calibration_samples)}")

    # Run all configurations
    configs = [
        ("baseline (1-pass)", False, 1),
        ("interleaved (1-pass)", True, 1),
        ("multi-pass (2-pass)", False, 2),
        ("interleaved+2-pass", True, 2),
    ]

    results: list[RunResult] = []
    for name, interleave, passes in configs:
        result = run_configuration(
            name,
            train_samples,
            test_samples,
            calibration_samples,
            interleave=interleave,
            num_passes=passes,
            seed=args.seed,
        )
        results.append(result)

    # Print summary
    baseline_acc = results[0].accuracy
    print("\n" + "=" * 70)
    print("=== Bio-ARN MNIST Improved ===")
    print(f"Training: {len(train_samples)} samples | Testing: {len(test_samples)} samples")
    print(f"Current headline: {CURRENT_HEADLINE * 100:.0f}% accuracy")
    print("=" * 70)

    print(f"\n{'Config':<24} {'Accuracy':<10} {'vs Baseline':<13} {'Covered':<10} {'Abstain':<10} {'CCCs':<6}")
    print("-" * 73)
    for r in results:
        delta = r.accuracy - baseline_acc
        delta_str = f"{delta:+.1%}" if r.name != results[0].name else "—"
        print(
            f"{r.name:<24} {r.accuracy:<10.1%} {delta_str:<13} "
            f"{r.covered_accuracy:<10.1%} {r.abstention_rate:<10.1%} {r.cccs_committed:<6}"
        )

    best = max(results, key=lambda r: r.accuracy)
    print(f"\n{'✅' if best.accuracy > CURRENT_HEADLINE else '⚠️'} Best result: {best.accuracy:.1%} ({best.name})")
    if best.accuracy > CURRENT_HEADLINE:
        print(f"   New headline: {best.accuracy:.1%} (was {CURRENT_HEADLINE:.0%})")
    else:
        print(f"   Current headline {CURRENT_HEADLINE:.0%} not beaten (best: {best.accuracy:.1%})")


if __name__ == "__main__":
    main()

