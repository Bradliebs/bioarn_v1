"""Decision-grade SoftHebb validation against the CIFAR-10 baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig
from bioarn.core.conv_ccc import ConvF1Layer

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for softhebb_final_validation.py") from exc


SEEDS = (0, 7, 42, 123, 2024)
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
THRESHOLD_PP = 3.0


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config_factory: Callable[[], ConvCCCConfig]


@dataclass(frozen=True)
class RunResult:
    nearest_centroid_accuracy: float
    linear_probe_accuracy: float
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=5_000, help="Balanced CIFAR-10 train subset size.")
    parser.add_argument("--test-samples", type=int, default=1_000, help="Balanced CIFAR-10 test subset size.")
    parser.add_argument("--passes", type=int, default=5, help="Hebbian training passes per run.")
    parser.add_argument("--batch-size", type=int, default=32, help="Hebbian mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for feature extraction.")
    parser.add_argument("--probe-batch-size", type=int, default=256, help="Linear-probe SGD mini-batch size.")
    parser.add_argument("--probe-epochs", type=int, default=100, help="Linear-probe training epochs.")
    parser.add_argument("--probe-lr", type=float, default=0.01, help="Linear-probe SGD learning rate.")
    parser.add_argument("--probe-momentum", type=float, default=0.9, help="Linear-probe SGD momentum.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive.")
    if args.train_samples % 10 != 0 or args.test_samples % 10 != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    if args.passes <= 0:
        raise ValueError("passes must be positive.")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.probe_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.probe_epochs <= 0:
        raise ValueError("probe_epochs must be positive.")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cifar10_datasets(data_dir: Path):
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    if tuple(train_dataset.classes) != CLASS_NAMES:
        raise RuntimeError("Unexpected CIFAR-10 class order from torchvision.")
    return train_dataset, test_dataset


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
        conv_competitive_k=8,
    )


def _softhebb_best_config() -> ConvCCCConfig:
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
        conv_competitive_k=8,
    )


def _softhebb_tight_config() -> ConvCCCConfig:
    return ConvCCCConfig(
        softhebb_enabled=True,
        softhebb_gamma=6.0,
        softhebb_beta=2.0,
        softhebb_theta_decay=0.99,
        num_conv_features=64,
        spatial_size=32,
        f1_top_k=32,
        conv_hebbian_lr=0.005,
        hebbian_batch_size=32,
        conv_competitive_k=8,
    )


def _make_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(name="baseline", config_factory=_baseline_config),
        BenchmarkSpec(name="softhebb_best", config_factory=_softhebb_best_config),
        BenchmarkSpec(name="softhebb_tight", config_factory=_softhebb_tight_config),
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


def _batch_indices(indices: torch.Tensor, *, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _iter_batches(
    values: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor | None]]:
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(values.shape[0], generator=generator)
    else:
        indices = torch.arange(values.shape[0])
    for start in range(0, values.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_values = values[batch_indices]
        batch_labels = None if labels is None else labels[batch_indices]
        yield batch_values, batch_labels


def _run_hebbian_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    pass_indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    for batch in _batch_indices(pass_indices, batch_size=batch_size):
        images = train_images[batch].to(device=device, dtype=torch.float32)
        learning_signal = torch.ones(images.shape[0], device=device, dtype=torch.float32)
        layer.hebbian_update(images, learning_signal=learning_signal)
    layer.flush_hebbian_updates()


@torch.inference_mode()
def _extract_features(
    layer: ConvF1Layer,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for batch_images, _ in _iter_batches(images, None, batch_size=batch_size, shuffle=False, seed=0):
        features.append(layer(batch_images.to(device=device, dtype=torch.float32)).cpu().to(torch.float32))
    return torch.cat(features, dim=0)


def _nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    train_norm = F.normalize(train_features, dim=1)
    test_norm = F.normalize(test_features, dim=1)
    prototypes: list[torch.Tensor] = []
    for label in range(10):
        class_features = train_norm[train_labels == label]
        if class_features.numel() == 0:
            raise RuntimeError(f"Missing class {label} in train features.")
        prototypes.append(F.normalize(class_features.mean(dim=0, keepdim=True), dim=1).squeeze(0))
    prototype_matrix = torch.stack(prototypes, dim=1)
    predictions = (test_norm @ prototype_matrix).argmax(dim=1)
    return float((predictions == test_labels).float().mean().item())


def _linear_probe_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    probe_epochs: int,
    probe_lr: float,
    probe_momentum: float,
    probe_batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    _set_seed(seed)
    probe = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.SGD([probe.weight, probe.bias], lr=probe_lr, momentum=probe_momentum)
    criterion = nn.CrossEntropyLoss()

    train_features = train_features.to(torch.float32)
    test_features = test_features.to(torch.float32)

    for epoch in range(probe_epochs):
        probe.train()
        for batch_features, batch_labels in _iter_batches(
            train_features,
            train_labels,
            batch_size=probe_batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            features = batch_features.to(device=device, dtype=torch.float32)
            labels = batch_labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_features, batch_labels in _iter_batches(
            test_features,
            test_labels,
            batch_size=probe_batch_size,
            shuffle=False,
            seed=0,
        ):
            features = batch_features.to(device=device, dtype=torch.float32)
            labels = batch_labels.to(device=device, dtype=torch.long)
            predictions = probe(features).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def _run_single_spec(
    spec: BenchmarkSpec,
    *,
    seed: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> RunResult:
    start = time.perf_counter()
    _set_seed(seed)
    layer = _build_layer(spec.config_factory()).to(device)
    class_indices = _class_index_buckets(train_labels)
    print(f"\n[{spec.name}] seed={seed}")
    for pass_index in range(args.passes):
        pass_indices = _build_interleaved_indices(class_indices, seed=seed + pass_index)
        pass_start = time.perf_counter()
        _run_hebbian_pass(
            layer,
            train_images,
            pass_indices,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"  pass {pass_index + 1}/{args.passes} done in {time.perf_counter() - pass_start:.1f}s")

    layer.eval()
    train_features = _extract_features(layer, train_images, batch_size=args.eval_batch_size, device=device)
    test_features = _extract_features(layer, test_images, batch_size=args.eval_batch_size, device=device)
    nearest_centroid = _nearest_centroid_accuracy(train_features, train_labels, test_features, test_labels)
    linear_probe = _linear_probe_accuracy(
        train_features,
        train_labels,
        test_features,
        test_labels,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_momentum=args.probe_momentum,
        probe_batch_size=args.probe_batch_size,
        seed=seed,
        device=device,
    )
    elapsed = time.perf_counter() - start
    print(
        f"  nearest-centroid={nearest_centroid * 100:.2f}% "
        f"linear-probe={linear_probe * 100:.2f}% elapsed={elapsed / 60:.1f}m"
    )
    return RunResult(
        nearest_centroid_accuracy=nearest_centroid,
        linear_probe_accuracy=linear_probe,
        elapsed_seconds=elapsed,
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values) * 100
    std = (statistics.stdev(values) * 100) if len(values) > 1 else 0.0
    return f"{mean:.2f}±{std:.2f}"


def _metric_table(
    results: dict[str, dict[int, RunResult]],
    *,
    metric_name: str,
    metric_getter: Callable[[RunResult], float],
) -> str:
    rows = [
        f"{metric_name}:",
        "| Config         | Seed 0 | Seed 7 | Seed 42 | Seed 123 | Seed 2024 | Mean ± Std |",
        "|----------------|--------|--------|---------|----------|-----------|------------|",
    ]
    for config_name in ("baseline", "softhebb_best", "softhebb_tight"):
        seed_values = [metric_getter(results[config_name][seed]) for seed in SEEDS]
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{config_name:<14}",
                    *(_format_percent(value) for value in seed_values),
                    _format_mean_std(seed_values),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _delta_summary(
    results: dict[str, dict[int, RunResult]],
    *,
    contender: str,
    metric_getter: Callable[[RunResult], float],
) -> tuple[float, float, int]:
    deltas = [
        (metric_getter(results[contender][seed]) - metric_getter(results["baseline"][seed])) * 100
        for seed in SEEDS
    ]
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    wins = sum(delta > 0.0 for delta in deltas)
    return mean, std, wins


def _best_softhebb_by_metric(
    results: dict[str, dict[int, RunResult]],
    *,
    metric_getter: Callable[[RunResult], float],
) -> str:
    contenders = ("softhebb_best", "softhebb_tight")
    return max(
        contenders,
        key=lambda contender: statistics.fmean(metric_getter(results[contender][seed]) for seed in SEEDS),
    )


def _build_report(results: dict[str, dict[int, RunResult]]) -> tuple[str, str]:
    nearest_table = _metric_table(
        results,
        metric_name="Nearest-Centroid",
        metric_getter=lambda result: result.nearest_centroid_accuracy,
    )
    probe_table = _metric_table(
        results,
        metric_name="Linear Probe",
        metric_getter=lambda result: result.linear_probe_accuracy,
    )

    nc_best_contender = _best_softhebb_by_metric(results, metric_getter=lambda result: result.nearest_centroid_accuracy)
    lp_best_contender = _best_softhebb_by_metric(results, metric_getter=lambda result: result.linear_probe_accuracy)

    nc_best_mean, nc_best_std, nc_best_wins = _delta_summary(
        results,
        contender=nc_best_contender,
        metric_getter=lambda result: result.nearest_centroid_accuracy,
    )
    lp_best_mean, lp_best_std, lp_best_wins = _delta_summary(
        results,
        contender=lp_best_contender,
        metric_getter=lambda result: result.linear_probe_accuracy,
    )

    nc_primary_mean, nc_primary_std, _ = _delta_summary(
        results,
        contender="softhebb_best",
        metric_getter=lambda result: result.nearest_centroid_accuracy,
    )
    lp_primary_mean, lp_primary_std, _ = _delta_summary(
        results,
        contender="softhebb_best",
        metric_getter=lambda result: result.linear_probe_accuracy,
    )

    nearest_go = nc_best_mean > THRESHOLD_PP and nc_best_wins >= 4
    probe_go = lp_best_mean > THRESHOLD_PP and lp_best_wins >= 4
    decision = "GO" if nearest_go or probe_go else "STOP"

    if decision == "GO":
        conclusion = (
            f"SoftHebb clears the >{THRESHOLD_PP:.0f}pp threshold on "
            f"{'linear-probe' if probe_go else 'nearest-centroid'} evaluation "
            f"with repeatability ({max(nc_best_wins, lp_best_wins)}/5 seed wins)."
        )
    else:
        conclusion = (
            "No SoftHebb variant clears the >3pp threshold with repeatability; "
            "the unsupervised Hebbian feature ceiling appears unchanged."
        )

    report = "\n".join(
        [
            "=== DECISION-GRADE VALIDATION — CIFAR-10 (5 seeds) ===",
            "",
            nearest_table,
            "",
            probe_table,
            "",
            f"DECISION: {decision}",
            f"- Δ_nc(softhebb_best - baseline) = {nc_primary_mean:.2f} ± {nc_primary_std:.2f} pp "
            f"→ {'>' if nc_primary_mean > THRESHOLD_PP else '<='} 3pp threshold",
            f"- Δ_lp(softhebb_best - baseline) = {lp_primary_mean:.2f} ± {lp_primary_std:.2f} pp "
            f"→ {'>' if lp_primary_mean > THRESHOLD_PP else '<='} 3pp threshold",
            f"- Best SoftHebb nearest-centroid contender: {nc_best_contender} "
            f"({nc_best_mean:.2f} ± {nc_best_std:.2f} pp, {nc_best_wins}/5 seed wins)",
            f"- Best SoftHebb linear-probe contender: {lp_best_contender} "
            f"({lp_best_mean:.2f} ± {lp_best_std:.2f} pp, {lp_best_wins}/5 seed wins)",
            f"- Conclusion: {conclusion}",
        ]
    )
    return report, decision


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = _select_device()
    print(f"Using device: {device}")
    print("Loading CIFAR-10 via torchvision...")
    train_dataset, test_dataset = _load_cifar10_datasets(args.data_dir)

    results: dict[str, dict[int, RunResult]] = {spec.name: {} for spec in _make_specs()}
    overall_start = time.perf_counter()
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_samples = _collect_balanced_samples(train_dataset, per_label=args.train_samples // 10, seed=seed)
        test_samples = _collect_balanced_samples(test_dataset, per_label=args.test_samples // 10, seed=seed + 1)
        train_images, train_labels = _stack_samples(train_samples)
        test_images, test_labels = _stack_samples(test_samples)
        for spec in _make_specs():
            results[spec.name][seed] = _run_single_spec(
                spec,
                seed=seed,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                args=args,
                device=device,
            )

    report, _ = _build_report(results)
    print()
    print(report)
    total_minutes = (time.perf_counter() - overall_start) / 60.0
    print(f"\nTotal elapsed: {total_minutes:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
