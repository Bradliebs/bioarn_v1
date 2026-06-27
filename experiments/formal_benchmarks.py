"""Formal multi-dataset benchmark suite for Bio-ARN 2.0."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioarn.config import GNWConfig, PrecisionConfig
from bioarn.hardware.energy_model import EnergyModel
from bioarn.predictive.precision_weighting import PrecisionWeightedGate
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples

try:
    from torchvision import datasets, transforms

    HAS_TORCHVISION = True
except Exception:
    datasets = None
    transforms = None
    HAS_TORCHVISION = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 7
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
MNIST_TRAIN_N = 5_000
MNIST_TEST_N = 1_000
CIFAR_TRAIN_N = 3_000
CIFAR_TEST_N = 1_000


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_samples: list[tuple[torch.Tensor, int | None]]
    test_samples: list[tuple[torch.Tensor, int | None]]
    ood_samples: list[torch.Tensor]
    input_dim: int
    concept_dim: int
    max_pool_size: int
    learning_rate: float
    margin_threshold: float
    source: str


@dataclass(frozen=True)
class BenchmarkResult:
    config_name: str
    accuracy: float
    ood_auroc: float
    energy_joules: float
    committed: int
    locked: int
    fire_rate: float
    train_seconds: float


@contextlib.contextmanager
def _socket_timeout(timeout_seconds: float):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _interleave_by_class(
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> list[tuple[torch.Tensor, int | None]]:
    buckets: defaultdict[int, list[tuple[torch.Tensor, int | None]]] = defaultdict(list)
    unlabeled: list[tuple[torch.Tensor, int | None]] = []
    for tensor, label in samples:
        if label is None:
            unlabeled.append((tensor, label))
            continue
        buckets[int(label)].append((tensor, label))

    ordered: list[tuple[torch.Tensor, int | None]] = []
    labels = sorted(buckets)
    while any(buckets[label] for label in labels):
        for label in labels:
            if buckets[label]:
                ordered.append(buckets[label].pop(0))
    ordered.extend(unlabeled)
    return ordered


def _auroc(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
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


def _make_random_ood(num_samples: int, input_dim: int, *, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(input_dim, generator=generator, dtype=torch.float32) for _ in range(num_samples)]


def _build_precision_config(pool_size: int) -> PrecisionConfig:
    return PrecisionConfig(
        enabled=True,
        pool_size=pool_size,
        entropy_window=100,
        precision_alpha=5.0,
        precision_threshold=0.5,
        min_precision=0.1,
        max_precision=1.0,
    )


def _workspace_config(concept_dim: int) -> GNWConfig:
    return GNWConfig(
        capacity=5,
        broadcast_gain=2.2,
        fatigue_rate=0.08,
        fatigue_threshold=0.18,
        competition_temp=0.45,
        concept_dim=concept_dim,
        context_size=192,
        context_decay=0.97,
        context_update_rate=0.25,
        attention_heads=4,
        context_top_k=6,
    )


def _base_config(bundle: DatasetBundle, *, use_batched: bool) -> VisionTrainConfig:
    return VisionTrainConfig(
        input_dim=bundle.input_dim,
        concept_dim=bundle.concept_dim,
        max_pool_size=bundle.max_pool_size,
        margin_threshold=bundle.margin_threshold,
        use_batched=use_batched,
        batch_size=32,
        learning_rate=bundle.learning_rate,
        num_train_samples=len(bundle.train_samples),
        num_test_samples=len(bundle.test_samples),
        preprocessing_warmup_samples=min(200, max(64, len(bundle.train_samples) // 10)),
    )


def _evaluate_samples(
    trainer: VisionTrainer,
    samples: Sequence[tuple[torch.Tensor, int | None]],
) -> dict[str, object]:
    correct = 0
    labeled = 0
    covered = 0
    abstained = 0
    confidences: list[float] = []
    firing_counts: list[int] = []

    with torch.inference_mode():
        for tensor, label in samples:
            step_result = trainer._step_pool(  # noqa: SLF001
                trainer._prepare_tensor(tensor),  # noqa: SLF001
                allow_recruit=False,
                preview=True,
            )
            prediction = (
                None
                if step_result.abstained
                else trainer._recognition_label(step_result.concept_direction, step_result.fired_indices)  # noqa: SLF001
            )
            labeled += int(label is not None)
            correct += int(label is not None and prediction == label)
            covered += int(prediction is not None)
            abstained += int(step_result.abstained)
            confidences.append(float(step_result.confidence))
            firing_counts.append(len(step_result.fired_indices))

    return {
        "accuracy": correct / max(labeled, 1),
        "coverage": covered / max(len(samples), 1),
        "abstention_rate": abstained / max(len(samples), 1),
        "confidences": confidences,
        "mean_firing_count": sum(firing_counts) / max(len(firing_counts), 1),
    }


def _ood_confidences(trainer: VisionTrainer, samples: Sequence[torch.Tensor]) -> list[float]:
    confidences: list[float] = []
    with torch.inference_mode():
        for tensor in samples:
            step_result = trainer._step_pool(  # noqa: SLF001
                trainer._prepare_tensor(tensor),  # noqa: SLF001
                allow_recruit=False,
                preview=True,
            )
            confidences.append(float(step_result.confidence))
    return confidences


def _estimate_total_energy(
    trainer: VisionTrainer,
    *,
    train_samples: int,
    eval_samples: int,
    ood_samples: int,
) -> float:
    pool_stats = trainer.system.ccc_pool.get_pool_stats()
    active_cccs = max(1, int(round(float(pool_stats["num_committed"]) * float(pool_stats["fire_rate"]))))
    energy_model = EnergyModel()
    train_energy = energy_model.estimate_learning_energy(trainer.system.config, "loihi2").total_joules
    infer_energy = energy_model.estimate_inference_energy(
        trainer.system.config,
        "loihi2",
        active_cccs,
    ).total_joules
    return float((train_energy * train_samples) + (infer_energy * (eval_samples + ood_samples)))


def _pool_locked_count(trainer: VisionTrainer) -> int:
    pool = trainer.system.ccc_pool
    pool_stats = pool.get_pool_stats()
    if "num_locked" in pool_stats:
        return int(pool_stats["num_locked"])
    cccs = getattr(pool, "cccs", None)
    if cccs is None:
        return 0
    count = 0
    for ccc in cccs:
        if hasattr(ccc, "locked"):
            count += int(bool(ccc.locked.item()))
        elif hasattr(ccc, "is_locked"):
            count += int(bool(ccc.is_locked.item()))
    return count


def _format_percent(value: float) -> str:
    return f"{value * 100:7.2f}%"


def _format_auroc(value: float) -> str:
    return f"{value:9.3f}"


def _format_energy(value_joules: float) -> str:
    return f"{value_joules * 1_000:9.3f}"


def _format_time(seconds: float) -> str:
    return f"{seconds:8.2f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_dataset_summary(bundle: DatasetBundle, results: Sequence[BenchmarkResult]) -> None:
    print(f"\n{bundle.name} ({'28×28 grayscale' if bundle.input_dim == 784 else '32×32 RGB'}, 10 classes)")
    print("-" * 60)
    print(f"Source: {bundle.source}")
    headers = ["Config", "Accuracy", "OOD AUROC", "Energy mJ", "Committed", "Locked", "FireRate", "Time (s)"]
    rows = [
        [
            result.config_name,
            _format_percent(result.accuracy),
            _format_auroc(result.ood_auroc),
            _format_energy(result.energy_joules),
            str(result.committed),
            str(result.locked),
            f"{result.fire_rate:8.3f}",
            _format_time(result.train_seconds),
        ]
        for result in results
    ]
    _print_table(headers, rows)


def _synthetic_grayscale_prototypes(*, fashion: bool) -> list[torch.Tensor]:
    prototypes: list[torch.Tensor] = []
    for label in range(10):
        image = torch.zeros(28, 28, dtype=torch.float32)
        if fashion:
            top = 3 + (label % 5) * 2
            left = 2 + (label // 5) * 10
            image[top : top + 18, left : left + 6] = 0.75
            image[top + 4 : top + 14, left - 1 : left + 7] = 0.45
            image[22 - (label % 4) : 24 - (label % 4), max(0, left - 2) : min(28, left + 8)] = 0.95
            image[:, (label * 2) % 28] += 0.08
        else:
            row = 2 + (label // 5) * 12
            col = 2 + (label % 5) * 5
            image[row : row + 20, col : col + 2] = 0.85
            image[row : row + 2, col : col + 10] = 0.65
            image[row + 9 : row + 11, col : col + 10] = 0.55
            image[row + 18 : row + 20, col : col + 10] = 0.75
            image[(label * 2) % 28, :] += 0.05
        prototypes.append(image.clamp_(0.0, 1.0))
    return prototypes


def _make_synthetic_grayscale_samples(
    num_samples: int,
    *,
    seed: int,
    fashion: bool,
) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    prototypes = _synthetic_grayscale_prototypes(fashion=fashion)
    samples: list[tuple[torch.Tensor, int]] = []
    for index in range(num_samples):
        label = index % 10
        prototype = prototypes[label].clone()
        shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
        shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
        prototype = torch.roll(prototype, shifts=(shift_y, shift_x), dims=(0, 1))
        gain = 0.9 + (0.2 * torch.rand(1, generator=generator).item())
        noise = torch.randn((28, 28), generator=generator) * (0.08 if fashion else 0.06)
        sample = (prototype * gain + noise).clamp_(0.0, 1.0).reshape(-1)
        samples.append((sample, label))
    return samples


def _collect_balanced_torchvision_samples(
    dataset,
    *,
    per_label: int,
    seed: int,
) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    counts: defaultdict[int, int] = defaultdict(int)
    samples: list[tuple[torch.Tensor, int]] = []
    for index in order:
        image, label = dataset[index]
        label = int(label)
        if counts[label] >= per_label:
            continue
        samples.append((image.to(torch.float32).reshape(-1), label))
        counts[label] += 1
        if all(counts[class_id] >= per_label for class_id in range(10)):
            break
    if len(samples) != per_label * 10:
        raise RuntimeError(f"Unable to collect {per_label} samples per label.")
    return samples


def _load_mnist_family(
    *,
    name: str,
    dataset_cls,
    train_n: int,
    test_n: int,
    seed: int,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]], str]:
    if HAS_TORCHVISION and dataset_cls is not None and transforms is not None:
        try:
            with _socket_timeout(20.0):
                transform = transforms.ToTensor()
                train_dataset = dataset_cls(root=str(DATA_ROOT), train=True, download=True, transform=transform)
                test_dataset = dataset_cls(root=str(DATA_ROOT), train=False, download=True, transform=transform)
            return (
                _collect_balanced_torchvision_samples(train_dataset, per_label=train_n // 10, seed=seed),
                _collect_balanced_torchvision_samples(test_dataset, per_label=test_n // 10, seed=seed + 1),
                f"torchvision-{name.lower()}",
            )
        except Exception as exc:
            print(f"[warn] {name} download failed, using synthetic fallback: {exc}")

    fashion = "fashion" in name.lower()
    return (
        _make_synthetic_grayscale_samples(train_n, seed=seed, fashion=fashion),
        _make_synthetic_grayscale_samples(test_n, seed=seed + 1, fashion=fashion),
        f"synthetic-{name.lower()}",
    )


def _load_mnist_bundle() -> DatasetBundle:
    train_samples, test_samples, source = _load_mnist_family(
        name="MNIST",
        dataset_cls=None if not HAS_TORCHVISION else datasets.MNIST,
        train_n=MNIST_TRAIN_N,
        test_n=MNIST_TEST_N,
        seed=SEED,
    )
    return DatasetBundle(
        name="MNIST",
        train_samples=train_samples,
        test_samples=test_samples,
        ood_samples=[],
        input_dim=784,
        concept_dim=96,
        max_pool_size=96,
        learning_rate=0.02,
        margin_threshold=0.50,
        source=source,
    )


def _load_fashion_bundle() -> DatasetBundle:
    train_samples, test_samples, source = _load_mnist_family(
        name="Fashion-MNIST",
        dataset_cls=None if not HAS_TORCHVISION else datasets.FashionMNIST,
        train_n=MNIST_TRAIN_N,
        test_n=MNIST_TEST_N,
        seed=SEED + 11,
    )
    return DatasetBundle(
        name="Fashion-MNIST",
        train_samples=train_samples,
        test_samples=test_samples,
        ood_samples=[],
        input_dim=784,
        concept_dim=96,
        max_pool_size=96,
        learning_rate=0.02,
        margin_threshold=0.50,
        source=source,
    )


def _load_cifar_bundle() -> DatasetBundle:
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir=DATA_ROOT,
        train_samples=6_000,
        test_samples=2_000,
        seed=SEED,
    )
    train_samples = take_samples(train_stream, CIFAR_TRAIN_N)
    test_samples = take_samples(test_stream, CIFAR_TEST_N)
    return DatasetBundle(
        name="CIFAR-10",
        train_samples=train_samples,
        test_samples=test_samples,
        ood_samples=_make_random_ood(CIFAR_TEST_N, 3072, seed=SEED + 99),
        input_dim=3072,
        concept_dim=128,
        max_pool_size=128,
        learning_rate=0.01,
        margin_threshold=0.35,
        source=source,
    )


def _attach_cross_dataset_ood(
    mnist: DatasetBundle,
    fashion: DatasetBundle,
) -> tuple[DatasetBundle, DatasetBundle]:
    mnist_ood = [tensor.clone() for tensor, _ in fashion.test_samples]
    fashion_ood = [tensor.clone() for tensor, _ in mnist.test_samples]
    return (
        DatasetBundle(**{**mnist.__dict__, "ood_samples": mnist_ood}),
        DatasetBundle(**{**fashion.__dict__, "ood_samples": fashion_ood}),
    )


def _run_trainer_benchmark(
    bundle: DatasetBundle,
    *,
    config_name: str,
    use_locking: bool,
    use_precision: bool,
    use_workspace: bool,
    use_curiosity: bool,
) -> BenchmarkResult:
    use_batched = not use_locking
    config = _base_config(bundle, use_batched=use_batched)
    if use_precision:
        config.precision = _build_precision_config(bundle.max_pool_size)
    if use_workspace:
        config.workspace = _workspace_config(bundle.concept_dim)
    if use_curiosity:
        config.curiosity_weight = 0.8

    trainer = VisionTrainer(config)
    trainer.system.ccc_pool.config.lock_threshold = 0.8 if use_locking else 1.1

    print(f"[run] {bundle.name} / {config_name}")
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_online(
            bundle.train_samples,
            num_samples=len(bundle.train_samples),
            num_passes=1,
            interleave_classes=True,
        )
    train_seconds = time.perf_counter() - start

    eval_result = _evaluate_samples(trainer, bundle.test_samples)
    ood_scores = _ood_confidences(trainer, bundle.ood_samples)
    pool_stats = trainer.system.ccc_pool.get_pool_stats()
    return BenchmarkResult(
        config_name=config_name,
        accuracy=float(eval_result["accuracy"]),
        ood_auroc=_auroc(eval_result["confidences"], ood_scores),
        energy_joules=_estimate_total_energy(
            trainer,
            train_samples=len(bundle.train_samples),
            eval_samples=len(bundle.test_samples),
            ood_samples=len(bundle.ood_samples),
        ),
        committed=int(pool_stats["num_committed"]),
        locked=_pool_locked_count(trainer),
        fire_rate=float(pool_stats["fire_rate"]),
        train_seconds=train_seconds,
    )


def _run_conv_benchmark(
    bundle: DatasetBundle,
    *,
    config_name: str,
    use_locking: bool,
    use_precision: bool,
) -> BenchmarkResult:
    config = VisionTrainConfig(
        input_dim=bundle.input_dim,
        concept_dim=bundle.concept_dim,
        max_pool_size=bundle.max_pool_size,
        margin_threshold=0.55,
        use_batched=False,
        batch_size=32,
        learning_rate=bundle.learning_rate,
        num_train_samples=len(bundle.train_samples),
        num_test_samples=len(bundle.test_samples),
        preprocessing_warmup_samples=0,
        use_conv_ccc=True,
    )
    trainer = VisionTrainer(config)
    trainer.system.ccc_pool.config.lock_threshold = 0.8 if use_locking else 1.1

    precision_gate: PrecisionWeightedGate | None = None
    if use_precision:
        precision_gate = PrecisionWeightedGate(_build_precision_config(bundle.max_pool_size))
        precision_gate.set_pool_size(int(trainer.system.ccc_pool.config.max_pool_size))

    ordered_train = _interleave_by_class(bundle.train_samples)
    print(f"[run] {bundle.name} / {config_name}")
    progress_interval = max(1, len(ordered_train) // 5)
    start = time.perf_counter()
    with torch.inference_mode():
        for index, (tensor, label) in enumerate(ordered_train, start=1):
            learning_rate_multiplier = 1.0
            if precision_gate is not None:
                prepared = trainer._prepare_tensor(tensor)  # noqa: SLF001
                preview = trainer.system.ccc_pool.preview(prepared)
                learning_rate_multiplier = float(precision_gate.preview_pool_output(preview.fired_indices))
            _, _, _, _, step_result = trainer._train_single_sample(  # noqa: SLF001
                tensor,
                label,
                learning_rate_multiplier=learning_rate_multiplier,
            )
            if precision_gate is not None:
                precision_gate.observe_pool_output(step_result.fired_indices)
            if index % progress_interval == 0 or index == len(ordered_train):
                print(f"  processed {index}/{len(ordered_train)}")
    train_seconds = time.perf_counter() - start

    eval_result = _evaluate_samples(trainer, bundle.test_samples)
    ood_scores = _ood_confidences(trainer, bundle.ood_samples)
    pool_stats = trainer.system.ccc_pool.get_pool_stats()
    return BenchmarkResult(
        config_name=config_name,
        accuracy=float(eval_result["accuracy"]),
        ood_auroc=_auroc(eval_result["confidences"], ood_scores),
        energy_joules=_estimate_total_energy(
            trainer,
            train_samples=len(bundle.train_samples),
            eval_samples=len(bundle.test_samples),
            ood_samples=len(bundle.ood_samples),
        ),
        committed=int(pool_stats["num_committed"]),
        locked=_pool_locked_count(trainer),
        fire_rate=float(pool_stats["fire_rate"]),
        train_seconds=train_seconds,
    )


def _run_dataset_benchmarks(bundle: DatasetBundle) -> list[BenchmarkResult]:
    results = [
        _run_trainer_benchmark(
            bundle,
            config_name="baseline",
            use_locking=False,
            use_precision=False,
            use_workspace=False,
            use_curiosity=False,
        ),
        _run_trainer_benchmark(
            bundle,
            config_name="with_locking",
            use_locking=True,
            use_precision=False,
            use_workspace=False,
            use_curiosity=False,
        ),
        _run_trainer_benchmark(
            bundle,
            config_name="with_precision",
            use_locking=False,
            use_precision=True,
            use_workspace=False,
            use_curiosity=False,
        ),
        _run_trainer_benchmark(
            bundle,
            config_name="best_combined",
            use_locking=True,
            use_precision=True,
            use_workspace=True,
            use_curiosity=True,
        ),
    ]
    if bundle.name == "CIFAR-10":
        results.extend(
            [
                _run_conv_benchmark(
                    bundle,
                    config_name="conv_baseline",
                    use_locking=False,
                    use_precision=False,
                ),
                _run_conv_benchmark(
                    bundle,
                    config_name="conv_all",
                    use_locking=True,
                    use_precision=True,
                ),
            ]
        )
    return results


def _print_overall_summary(results_by_dataset: dict[str, tuple[DatasetBundle, list[BenchmarkResult]]]) -> None:
    print("\n" + "=" * 60)
    print("      FORMAL BENCHMARK RESULTS — Bio-ARN 2.0")
    print("=" * 60)
    for bundle, results in results_by_dataset.values():
        _print_dataset_summary(bundle, results)


def main() -> None:
    torch.manual_seed(SEED)

    mnist = _load_mnist_bundle()
    fashion = _load_fashion_bundle()
    cifar = _load_cifar_bundle()
    mnist, fashion = _attach_cross_dataset_ood(mnist, fashion)

    bundles = [mnist, fashion, cifar]
    results_by_dataset: dict[str, tuple[DatasetBundle, list[BenchmarkResult]]] = {}
    for bundle in bundles:
        results_by_dataset[bundle.name] = (bundle, _run_dataset_benchmarks(bundle))

    _print_overall_summary(results_by_dataset)


if __name__ == "__main__":
    main()
