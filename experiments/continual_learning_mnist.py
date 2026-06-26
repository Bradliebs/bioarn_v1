"""MNIST continual learning benchmarks for Bio-ARN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from experiments.continual_learning import _print_result, run_permuted_mnist, run_split_mnist, summarize_findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bio-ARN continual learning benchmarks on MNIST.")
    parser.add_argument("--split-train-samples", type=int, default=300, help="Training samples per split-MNIST task.")
    parser.add_argument("--split-test-samples", type=int, default=150, help="Test samples per split-MNIST task.")
    parser.add_argument(
        "--permuted-train-samples",
        type=int,
        default=300,
        help="Training samples per permuted-MNIST task.",
    )
    parser.add_argument(
        "--permuted-test-samples",
        type=int,
        default=150,
        help="Test samples per permuted-MNIST task.",
    )
    parser.add_argument("--permuted-tasks", type=int, default=5, help="Number of permutation tasks to evaluate.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    print(
        "Running MNIST continual learning benchmarks with "
        f"split_train={args.split_train_samples}, split_test={args.split_test_samples}, "
        f"permuted_train={args.permuted_train_samples}, permuted_test={args.permuted_test_samples}, "
        f"permuted_tasks={args.permuted_tasks}, seed={args.seed}"
    )

    split_result = run_split_mnist(
        train_samples=args.split_train_samples,
        test_samples=args.split_test_samples,
        seed=args.seed,
    )
    _print_result(split_result)

    permuted_result = run_permuted_mnist(
        train_samples=args.permuted_train_samples,
        test_samples=args.permuted_test_samples,
        num_tasks=args.permuted_tasks,
        seed=args.seed,
    )
    _print_result(permuted_result)

    print("\nKey findings")
    for finding in summarize_findings([split_result, permuted_result]):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
