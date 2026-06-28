"""Run a full-scale SoftHebb benchmark on real CIFAR-10."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig, deep_cifar_config
from bioarn.core.conv_ccc import ConvF1Layer

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for fullscale_softhebb.py") from exc


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config_factory: Callable[[], ConvCCCConfig]
    use_layerwise_pretrain: bool = False


@dataclass
class BenchmarkResult:
    name: str
    per_pass_accuracy: list[float]
    best_accuracy: float
    best_pass: int
    best_per_class_accuracy: list[float]
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    default_workers = min(3, len(_make_specs()), max(1, (os.cpu_count() or 1) // 4))
    default_threads = max(1, (os.cpu_count() or 1) // max(default_workers, 1))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=10_000, help="Balanced CIFAR-10 train subset size.")
    parser.add_argument("--test-samples", type=int, default=2_000, help="Balanced CIFAR-10 test subset size.")
    parser.add_argument("--passes", type=int, default=10, help="Number of full-model passes per configuration.")
    parser.add_argument("--batch-size", type=int, default=32, help="Hebbian mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for feature extraction/evaluation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and initialization.")
    parser.add_argument("--workers", type=int, default=default_workers, help="Parallel config workers.")
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=default_threads,
        help="Torch CPU threads per worker process.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive.")
    if args.train_samples % 10 != 0 or args.test_samples % 10 != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    if args.passes < 10:
        raise ValueError("passes must be at least 10 for the full-scale benchmark.")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.workers <= 0 or args.torch_threads_per_worker <= 0:
        raise ValueError("workers and torch_threads_per_worker must be positive.")


def _collect_balanced_samples(dataset, *, per_label: int, seed: int) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    counts: defaultdict[int, int] = defaultdict(int)
    samples: list[tuple[torch.Tensor, int]] = []
    for index in order:
        image, label = dataset[index]
        label = int(label)
        if counts[label] >= per_label:
            continue
        samples.append((image.to(torch.float32), label))
        counts[label] += 1
        if all(counts[class_id] >= per_label for class_id in range(10)):
            break
    expected = per_label * 10
    if len(samples) != expected:
        raise RuntimeError(f"Unable to collect {expected} balanced CIFAR-10 samples.")
    return samples


def _load_real_cifar10(
    *,
    data_dir: Path,
    train_samples: int,
    test_samples: int,
    seed: int,
    download: bool,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=download, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=download, transform=transform)
    if tuple(train_dataset.classes) != CLASS_NAMES:
        raise RuntimeError("Unexpected CIFAR-10 class order from torchvision.")
    return (
        _collect_balanced_samples(train_dataset, per_label=train_samples // 10, seed=seed),
        _collect_balanced_samples(test_dataset, per_label=test_samples // 10, seed=seed + 1),
    )


def _ensure_cifar10_available(data_dir: Path) -> None:
    transform = transforms.ToTensor()
    datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)


def _stack_samples(samples: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image for image, _ in samples], dim=0).to(torch.float32)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    return images, labels


def _build_layer(config: ConvCCCConfig) -> ConvF1Layer:
    return ConvF1Layer(
        in_channels=config.in_channels,
        num_features=config.num_conv_features,
        spatial_size=config.spatial_size,
        top_k=config.f1_top_k,
        spatial_grid=config.spatial_grid,
        num_layers=config.num_conv_layers,
        hidden_channels=config.conv_hidden_channels,
        kernel_sizes=config.conv_kernel_sizes,
        spatial_top_k=config.spatial_top_k,
        competitive_k=config.conv_competitive_k,
        hebbian_lr=config.conv_hebbian_lr,
        hebbian_batch_size=config.hebbian_batch_size,
        weight_norm_target=config.conv_weight_norm,
        enable_local_contrast_norm=config.enable_local_contrast_norm,
        contrast_kernel_size=config.contrast_kernel_size,
        response_norm_eps=config.response_norm_eps,
        feature_pool_avg_mix=config.feature_pool_avg_mix,
        hebbian_oja_decay=config.hebbian_oja_decay,
        filter_decorrelation=config.filter_decorrelation,
        softhebb_enabled=config.softhebb_enabled,
        softhebb_gamma=config.softhebb_gamma,
        softhebb_beta=config.softhebb_beta,
        softhebb_theta_decay=config.softhebb_theta_decay,
    )


def _baseline_config() -> ConvCCCConfig:
    return ConvCCCConfig(
        softhebb_enabled=False,
        num_conv_features=64,
        spatial_size=32,
        f1_top_k=32,
        conv_hebbian_lr=0.005,
        hebbian_batch_size=32,
    )


def _softhebb_default_config() -> ConvCCCConfig:
    return ConvCCCConfig(
        softhebb_enabled=True,
        softhebb_gamma=4.0,
        softhebb_beta=2.0,
        softhebb_theta_decay=0.99,
        num_conv_features=64,
        spatial_size=32,
        f1_top_k=32,
        conv_hebbian_lr=0.005,
        hebbian_batch_size=32,
    )


def _softhebb_soft_config() -> ConvCCCConfig:
    return ConvCCCConfig(
        softhebb_enabled=True,
        softhebb_gamma=2.0,
        softhebb_beta=1.5,
        softhebb_theta_decay=0.99,
        num_conv_features=128,
        spatial_size=32,
        f1_top_k=64,
        conv_hebbian_lr=0.003,
        hebbian_batch_size=32,
    )


def _softhebb_deep_config() -> ConvCCCConfig:
    config = deep_cifar_config()
    config.softhebb_enabled = True
    config.softhebb_gamma = 4.0
    config.softhebb_beta = 2.0
    config.softhebb_theta_decay = 0.99
    config.hebbian_batch_size = 32
    config.layerwise_train.freeze_after_training = False
    return config


def _make_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(name="baseline", config_factory=_baseline_config),
        BenchmarkSpec(name="softhebb_default", config_factory=_softhebb_default_config),
        BenchmarkSpec(name="softhebb_soft", config_factory=_softhebb_soft_config),
        BenchmarkSpec(name="softhebb_deep", config_factory=_softhebb_deep_config, use_layerwise_pretrain=True),
    ]


def _class_index_buckets(labels: torch.Tensor) -> dict[int, list[int]]:
    buckets: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        buckets[int(label)].append(index)
    return {label: buckets[label] for label in sorted(buckets)}


def _build_interleaved_indices(class_indices: dict[int, list[int]], *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shuffled_buckets: dict[int, list[int]] = {}
    for label, indices in class_indices.items():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_buckets[label] = [indices[position] for position in order]
    ordered: list[int] = []
    bucket_positions = {label: 0 for label in shuffled_buckets}
    labels = sorted(shuffled_buckets)
    while True:
        emitted = False
        for label in labels:
            bucket = shuffled_buckets[label]
            position = bucket_positions[label]
            if position < len(bucket):
                ordered.append(bucket[position])
                bucket_positions[label] += 1
                emitted = True
        if not emitted:
            break
    return torch.tensor(ordered, dtype=torch.long)


def _batch_indices(indices: torch.Tensor, *, batch_size: int):
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _run_hebbian_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    pass_indices: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    for batch_indices in _batch_indices(pass_indices, batch_size=batch_size):
        batch = train_images[batch_indices]
        layer.hebbian_update(batch, learning_signal=torch.ones(batch.shape[0], dtype=torch.float32))
    layer.flush_hebbian_updates()


def _run_layerwise_pretrain(
    layer: ConvF1Layer,
    config: ConvCCCConfig,
    train_images: torch.Tensor,
    class_indices: dict[int, list[int]],
    *,
    batch_size: int,
    seed: int,
    verbose: bool,
) -> None:
    pretrain_samples = min(config.layerwise_train.samples_per_layer, train_images.shape[0])
    pretrain_indices = _build_interleaved_indices(class_indices, seed=seed)[:pretrain_samples]
    pretrain_images = train_images[pretrain_indices]
    pretrain_batches = [pretrain_images[start : start + batch_size] for start in range(0, pretrain_images.shape[0], batch_size)]
    if verbose:
        print(
            f"[softhebb_deep] layerwise pretraining "
            f"samples_per_layer={pretrain_samples} passes_per_layer={config.layerwise_train.passes_per_layer}"
        )
    layer.train_layerwise(
        pretrain_batches,
        samples_per_layer=pretrain_samples,
        passes_per_layer=config.layerwise_train.passes_per_layer,
        lr_schedule=list(config.layerwise_train.lr_schedule),
    )
    for layer_index in range(layer.num_layers):
        layer._unfreeze_layer(layer_index)  # noqa: SLF001 - benchmark needs full-model passes after pretraining


def _prototype_head(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    sums = torch.zeros(10, layer.output_dim, dtype=torch.float32)
    counts = torch.zeros(10, dtype=torch.float32)
    with torch.inference_mode():
        indices = torch.arange(train_images.shape[0], dtype=torch.long)
        for batch_indices in _batch_indices(indices, batch_size=batch_size):
            images = train_images[batch_indices]
            labels = train_labels[batch_indices]
            features = layer(images).to(torch.float32)
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones(labels.shape[0], dtype=torch.float32))
    prototypes = sums / counts.unsqueeze(1).clamp_min(1.0)
    return F.normalize(prototypes, dim=1)


def _evaluate(
    layer: ConvF1Layer,
    prototypes: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, list[float]]:
    per_class_correct = torch.zeros(10, dtype=torch.float32)
    per_class_total = torch.zeros(10, dtype=torch.float32)
    with torch.inference_mode():
        indices = torch.arange(test_images.shape[0], dtype=torch.long)
        for batch_indices in _batch_indices(indices, batch_size=batch_size):
            images = test_images[batch_indices]
            labels = test_labels[batch_indices]
            features = F.normalize(layer(images).to(torch.float32), dim=1)
            predictions = torch.argmax(features @ prototypes.T, dim=1)
            correct = predictions.eq(labels)
            per_class_correct.index_add_(0, labels, correct.to(torch.float32))
            per_class_total.index_add_(0, labels, torch.ones(labels.shape[0], dtype=torch.float32))
    per_class_accuracy = (
        per_class_correct / per_class_total.clamp_min(1.0)
    ).tolist()
    accuracy = float(per_class_correct.sum().item() / per_class_total.sum().clamp_min(1.0).item())
    return accuracy, [float(value) for value in per_class_accuracy]


def _run_spec(
    spec: BenchmarkSpec,
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    args: argparse.Namespace,
    verbose: bool = True,
) -> BenchmarkResult:
    run_start = time.perf_counter()
    torch.manual_seed(args.seed)
    config = spec.config_factory()
    config.hebbian_batch_size = args.batch_size
    layer = _build_layer(config)
    class_indices = _class_index_buckets(train_labels)
    if spec.use_layerwise_pretrain:
        _run_layerwise_pretrain(
            layer,
            config,
            train_images,
            class_indices,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=verbose,
        )

    per_pass_accuracy: list[float] = []
    best_accuracy = 0.0
    best_pass = 0
    best_per_class_accuracy = [0.0 for _ in range(10)]

    if verbose:
        print(f"\n=== {spec.name} ===")
    for pass_index in range(args.passes):
        pass_start = time.perf_counter()
        pass_indices = _build_interleaved_indices(class_indices, seed=args.seed + pass_index)
        _run_hebbian_pass(layer, train_images, pass_indices, batch_size=args.batch_size)
        prototypes = _prototype_head(layer, train_images, train_labels, batch_size=args.eval_batch_size)
        accuracy, per_class_accuracy = _evaluate(
            layer,
            prototypes,
            test_images,
            test_labels,
            batch_size=args.eval_batch_size,
        )
        per_pass_accuracy.append(accuracy)
        if pass_index == 0 or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pass = pass_index + 1
            best_per_class_accuracy = per_class_accuracy
        if verbose:
            print(
                f"[{spec.name}] pass {pass_index + 1}/{args.passes} "
                f"accuracy={accuracy * 100:.2f}% elapsed={time.perf_counter() - pass_start:.1f}s"
            )

    return BenchmarkResult(
        name=spec.name,
        per_pass_accuracy=per_pass_accuracy,
        best_accuracy=best_accuracy,
        best_pass=best_pass,
        best_per_class_accuracy=best_per_class_accuracy,
        elapsed_seconds=time.perf_counter() - run_start,
    )


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_learning_curves(results: list[BenchmarkResult], passes: int) -> None:
    headers = ["Pass", *[result.name for result in results]]
    separator = "|" + "|".join("-" * (len(header) + 2) for header in headers) + "|"
    print("Learning curves:")
    print("| " + " | ".join(headers) + " |")
    print(separator)
    for pass_index in range(passes):
        row = [str(pass_index + 1)]
        for result in results:
            row.append(_format_accuracy(result.per_pass_accuracy[pass_index]))
        print("| " + " | ".join(row) + " |")


def _print_per_class_table(result: BenchmarkResult) -> None:
    print("\nPer-class accuracy (best config):")
    print("| Class | Accuracy |")
    print("|-------|----------|")
    for class_name, accuracy in zip(CLASS_NAMES, result.best_per_class_accuracy, strict=True):
        print(f"| {class_name} | {_format_accuracy(accuracy)} |")


def _worker_run_spec(spec_name: str, args_dict: dict[str, object]) -> BenchmarkResult:
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(int(args.torch_threads_per_worker))
    spec_map = {spec.name: spec for spec in _make_specs()}
    train_samples, test_samples = _load_real_cifar10(
        data_dir=Path(args.data_dir),
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        seed=int(args.seed),
        download=False,
    )
    train_images, train_labels = _stack_samples(train_samples)
    test_images, test_labels = _stack_samples(test_samples)
    return _run_spec(
        spec_map[spec_name],
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        args=args,
        verbose=False,
    )


def _run_all_specs(args: argparse.Namespace) -> list[BenchmarkResult]:
    specs = _make_specs()
    if args.workers <= 1 or len(specs) == 1:
        torch.set_num_threads(args.torch_threads_per_worker)
        print("Loading real CIFAR-10 via torchvision...")
        train_samples, test_samples = _load_real_cifar10(
            data_dir=args.data_dir,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seed=args.seed,
            download=True,
        )
        train_images, train_labels = _stack_samples(train_samples)
        test_images, test_labels = _stack_samples(test_samples)
        print(
            f"\n=== Full-Scale SoftHebb — CIFAR-10 "
            f"({train_images.shape[0]} train, {test_images.shape[0]} test) ===\n"
        )
        return [
            _run_spec(
                spec,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                args=args,
                verbose=True,
            )
            for spec in specs
        ]

    print("Ensuring real CIFAR-10 is available via torchvision...")
    _ensure_cifar10_available(args.data_dir)
    print(
        f"\n=== Full-Scale SoftHebb — CIFAR-10 "
        f"({args.train_samples} train, {args.test_samples} test) ===\n"
    )
    print(
        f"Running {len(specs)} configs with {min(args.workers, len(specs))} workers "
        f"and {args.torch_threads_per_worker} torch threads per worker..."
    )

    args_dict = {
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "passes": args.passes,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "workers": args.workers,
        "torch_threads_per_worker": args.torch_threads_per_worker,
    }
    spec_order = {spec.name: index for index, spec in enumerate(specs)}
    execution_names = [spec.name for spec in sorted(specs, key=lambda spec: spec.use_layerwise_pretrain, reverse=True)]
    results_by_name: dict[str, BenchmarkResult] = {}
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(specs)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_worker_run_spec, name, args_dict): name for name in execution_names}
        for future in as_completed(futures):
            name = futures[future]
            result = future.result()
            results_by_name[name] = result
            print(
                f"[done] {name}: best={_format_accuracy(result.best_accuracy)} "
                f"pass={result.best_pass} elapsed={result.elapsed_seconds / 60.0:.1f}m"
            )
    return [results_by_name[spec.name] for spec in sorted(specs, key=lambda spec: spec_order[spec.name])]


def main() -> int:
    args = parse_args()
    _validate_args(args)
    results = _run_all_specs(args)

    print()
    _print_learning_curves(results, args.passes)

    best_result = max(results, key=lambda result: (result.best_accuracy, result.best_pass))
    baseline = next(result for result in results if result.name == "baseline")
    best_softhebb = max(
        (result for result in results if result.name != "baseline"),
        key=lambda result: (result.best_accuracy, result.best_pass),
    )
    softhebb_delta = (best_softhebb.best_accuracy - baseline.best_accuracy) * 100.0

    print(
        f"\nBest: {best_result.name} at {_format_accuracy(best_result.best_accuracy)} "
        f"(pass {best_result.best_pass})"
    )
    if softhebb_delta >= 0.0:
        print(f"SoftHebb vs baseline at scale: improved by {softhebb_delta:.2f} percentage points.")
    else:
        print(f"SoftHebb vs baseline at scale: trailed by {abs(softhebb_delta):.2f} percentage points.")

    _print_per_class_table(best_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
