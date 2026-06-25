"""Measure how interleaving and multi-pass training scale with dataset size."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from bioarn.training import SyntheticCIFAR10Stream, VisionTrainConfig, VisionTrainer, take_samples

DEFAULT_SIZES = [200, 500, 1000, 2000, 5000]
DEFAULT_TEST_SAMPLES = 200
DEFAULT_TEST_SEED = 99


@dataclass(frozen=True)
class VariantSpec:
    name: str
    interleave_classes: bool
    num_passes: int


def build_trainer_config(train_size: int, test_size: int) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=3072,
        concept_dim=128,
        max_pool_size=150,
        margin_threshold=0.35,
        use_batched=True,
        batch_size=32,
        learning_rate=0.02,
        num_train_samples=train_size,
        num_test_samples=test_size,
    )


def build_sorted_train_samples(size: int, *, seed: int) -> list[tuple[torch.Tensor, int]]:
    samples = take_samples(SyntheticCIFAR10Stream(size, seed=seed, shuffle=True), size)
    labeled = [(tensor, int(label)) for tensor, label in samples if label is not None]
    return sorted(labeled, key=lambda item: item[1])


def build_test_samples(size: int, *, seed: int) -> list[tuple[torch.Tensor, int]]:
    samples = take_samples(SyntheticCIFAR10Stream(size, seed=seed, shuffle=True), size)
    return [(tensor, int(label)) for tensor, label in samples if label is not None]


def run_variant(
    train_samples: list[tuple[torch.Tensor, int]],
    test_samples: list[tuple[torch.Tensor, int]],
    variant: VariantSpec,
    *,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    trainer = VisionTrainer(build_trainer_config(len(train_samples), len(test_samples)))
    trainer.train_online(
        train_samples,
        num_samples=len(train_samples),
        interleave_classes=variant.interleave_classes,
        num_passes=variant.num_passes,
    )
    evaluation = trainer.evaluate(test_samples, num_samples=len(test_samples))
    return float(evaluation["accuracy"])


def format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def format_signed_percent(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100.0:.1f}%"


def describe_trend(start_delta: float, end_delta: float) -> str:
    tolerance = 0.01
    if end_delta > start_delta + tolerance:
        return "growing"
    if end_delta < start_delta - tolerance:
        return "diminishing"
    return "stable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    variants = [
        VariantSpec("default", interleave_classes=False, num_passes=1),
        VariantSpec("interleaved", interleave_classes=True, num_passes=1),
        VariantSpec("multi-pass", interleave_classes=False, num_passes=2),
        VariantSpec("both", interleave_classes=True, num_passes=2),
    ]
    test_samples = build_test_samples(args.test_samples, seed=args.test_seed)

    rows: list[tuple[int, dict[str, float]]] = []
    for size_index, train_size in enumerate(args.sizes):
        print(f"[size] train_samples={train_size}")
        train_samples = build_sorted_train_samples(train_size, seed=args.train_seed + size_index)
        metrics: dict[str, float] = {}
        for variant_index, variant in enumerate(variants):
            print(f"[run] size={train_size} variant={variant.name}")
            metrics[variant.name] = run_variant(
                train_samples,
                test_samples,
                variant,
                seed=args.seed + (size_index * 10) + variant_index,
            )
        rows.append((train_size, metrics))

    print("=== Interleaved Training Scaling ===")
    print(f"Test set: {len(test_samples)} samples (fixed seed={args.test_seed})")
    print()
    print(
        "Train Size    Default    Interleaved    Multi-Pass    Both       Interleave Delta"
    )
    print(
        "----------    -------    -----------    ----------    ----       ----------------"
    )
    for train_size, metrics in rows:
        interleave_delta = metrics["interleaved"] - metrics["default"]
        print(
            f"{train_size:<12}"
            f"{format_percent(metrics['default']):<11}"
            f"{format_percent(metrics['interleaved']):<15}"
            f"{format_percent(metrics['multi-pass']):<14}"
            f"{format_percent(metrics['both']):<11}"
            f"{format_signed_percent(interleave_delta):<12}"
        )

    first_delta = rows[0][1]["interleaved"] - rows[0][1]["default"]
    last_delta = rows[-1][1]["interleaved"] - rows[-1][1]["default"]
    print()
    print("Scaling analysis:")
    print(f"- Interleave advantage at {rows[0][0]} samples: {format_signed_percent(first_delta)}")
    print(f"- Interleave advantage at {rows[-1][0]} samples: {format_signed_percent(last_delta)}")
    print(f"- Trend: {describe_trend(first_delta, last_delta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
