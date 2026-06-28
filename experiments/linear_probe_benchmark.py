"""Compare nearest-centroid and linear-probe evaluation on Hebbian CIFAR-10 features."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig
from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.data.base import DataSample, StreamingDataSource
from bioarn.data.vision import AugmentedCIFARStream, HebbianAugmentation

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for linear_probe_benchmark.py") from exc


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config: ConvCCCConfig
    use_augmentation: bool = False
    use_layerwise: bool = False


@dataclass
class BenchmarkResult:
    name: str
    nearest_centroid_accuracy: float
    linear_probe_accuracy: float
    error: str | None = None


class TensorSampleStream(StreamingDataSource):
    """Expose in-memory labelled tensors via the streaming interface."""

    def __init__(self, samples: list[tuple[torch.Tensor, int]]) -> None:
        super().__init__()
        self.samples = [(image.to(torch.float32).clone(), int(label)) for image, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def stream(self):
        for index, (image, label) in enumerate(self.samples):
            yield DataSample(
                data=image.clone(),
                label=label,
                modality="vision",
                metadata={"index": index, "dataset": "cifar10", "source": "balanced-subset"},
            )


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=5000, help="Balanced train subset size.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Balanced test subset size.")
    parser.add_argument("--passes", type=int, default=3, help="Hebbian feature-learning passes.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for Hebbian updates.")
    parser.add_argument("--probe-batch-size", type=int, default=256, help="Mini-batch size for probe SGD.")
    parser.add_argument("--probe-epochs", type=int, default=50, help="Linear-probe training epochs.")
    parser.add_argument("--probe-lr", type=float, default=0.01, help="Linear-probe SGD learning rate.")
    parser.add_argument("--probe-momentum", type=float, default=0.9, help="Linear-probe SGD momentum.")
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=2,
        help="Views per training sample for augmented Hebbian configs.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and initialization.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive.")
    if args.train_samples % 10 != 0 or args.test_samples % 10 != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    if args.passes < 1:
        raise ValueError("passes must be positive.")
    if args.batch_size <= 0 or args.probe_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.probe_epochs <= 0:
        raise ValueError("probe_epochs must be positive.")
    if args.augmentation_factor <= 0:
        raise ValueError("augmentation_factor must be positive.")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    return (
        _collect_balanced_samples(train_dataset, per_label=train_samples // 10, seed=seed),
        _collect_balanced_samples(test_dataset, per_label=test_samples // 10, seed=seed + 1),
    )


def _stack_samples(samples: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image for image, _ in samples], dim=0).to(torch.float32)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    return images, labels


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


def _iter_image_batches(
    images: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[torch.Tensor]:
    return [batch for batch, _ in _iter_batches(images, None, batch_size=batch_size, shuffle=shuffle, seed=seed)]


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


def _moderate_deeper_softhebb(batch_size: int) -> ConvCCCConfig:
    return ConvCCCConfig(
        in_channels=3,
        spatial_size=32,
        num_conv_features=192,
        num_conv_layers=5,
        conv_hidden_channels=(48, 64, 96, 128),
        conv_kernel_sizes=(5, 3, 3, 3, 3),
        spatial_grid=4,
        f1_top_k=64,
        conv_hebbian_lr=0.005,
        hebbian_batch_size=batch_size,
        conv_competitive_k=16,
        spatial_top_k=8,
        softhebb_enabled=True,
        softhebb_gamma=4.0,
        softhebb_beta=2.0,
    )


def _make_benchmark_specs(batch_size: int) -> list[BenchmarkSpec]:
    baseline = ConvCCCConfig(hebbian_batch_size=batch_size)
    softhebb = ConvCCCConfig(
        hebbian_batch_size=batch_size,
        softhebb_enabled=True,
        softhebb_gamma=4.0,
        softhebb_beta=2.0,
    )
    softhebb_tuned = ConvCCCConfig(
        hebbian_batch_size=batch_size,
        softhebb_enabled=True,
        softhebb_gamma=2.0,
        softhebb_beta=1.5,
    )
    softhebb_aug = copy.deepcopy(softhebb)
    deeper_softhebb = _moderate_deeper_softhebb(batch_size)
    return [
        BenchmarkSpec(name="baseline", config=baseline),
        BenchmarkSpec(name="+softhebb", config=softhebb),
        BenchmarkSpec(name="+softhebb_tuned", config=softhebb_tuned),
        BenchmarkSpec(name="+softhebb+aug", config=softhebb_aug, use_augmentation=True),
        BenchmarkSpec(name="+deeper_softhebb", config=deeper_softhebb, use_layerwise=True),
    ]


def _collect_augmented_views(
    base_samples: list[tuple[torch.Tensor, int]],
    *,
    augmentation_factor: int,
    seed: int,
) -> list[tuple[torch.Tensor, int]]:
    stream = AugmentedCIFARStream(
        num_samples=len(base_samples),
        augmentation=HebbianAugmentation(random_flip=True, random_crop=True),
        augmentation_factor=augmentation_factor,
        seed=seed,
        base_stream=TensorSampleStream(base_samples),
    )
    return [(sample.data.to(torch.float32), int(sample.label)) for sample in stream.stream()]


def _training_samples_for_pass(
    spec: BenchmarkSpec,
    *,
    raw_train_samples: list[tuple[torch.Tensor, int]],
    augmentation_factor: int,
    pass_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.use_augmentation:
        augmented_samples = _collect_augmented_views(
            raw_train_samples,
            augmentation_factor=augmentation_factor,
            seed=seed + pass_index,
        )
        return _stack_samples(augmented_samples)
    return _stack_samples(raw_train_samples)


def _run_standard_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    for images, _ in _iter_batches(train_images, None, batch_size=batch_size, shuffle=True, seed=seed):
        batch = images.to(device=device, dtype=torch.float32)
        layer.hebbian_update(batch, learning_signal=torch.ones(batch.shape[0], device=device, dtype=torch.float32))
    layer.flush_hebbian_updates()


def _run_layerwise_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    layerwise_batches = [
        batch.to(device=device, dtype=torch.float32)
        for batch in _iter_image_batches(train_images, batch_size=batch_size, shuffle=True, seed=seed)
    ]
    return layer.train_layerwise(
        layerwise_batches,
        samples_per_layer=train_images.shape[0],
        passes_per_layer=1,
    )


@torch.inference_mode()
def _extract_features(
    layer: ConvF1Layer,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    feature_batches: list[torch.Tensor] = []
    for batch_images, _ in _iter_batches(images, None, batch_size=batch_size, shuffle=False, seed=0):
        feature_batches.append(layer(batch_images.to(device=device, dtype=torch.float32)).cpu())
    return torch.cat(feature_batches, dim=0)


def _nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int = 10,
) -> float:
    train_norm = F.normalize(train_features, dim=1)
    test_norm = F.normalize(test_features, dim=1)
    prototypes: list[torch.Tensor] = []
    for label in range(num_classes):
        class_features = train_norm[train_labels == label]
        if class_features.numel() == 0:
            raise RuntimeError(f"Missing class {label} in train features.")
        prototype = F.normalize(class_features.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        prototypes.append(prototype)
    prototype_matrix = torch.stack(prototypes, dim=1)
    logits = test_norm @ prototype_matrix
    predictions = logits.argmax(dim=1)
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
    torch.manual_seed(seed)
    probe = LinearProbe(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=probe_lr, momentum=probe_momentum)
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


def _run_spec(
    spec: BenchmarkSpec,
    *,
    raw_train_samples: list[tuple[torch.Tensor, int]],
    base_train_images: torch.Tensor,
    base_train_labels: torch.Tensor,
    base_test_images: torch.Tensor,
    base_test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    config = copy.deepcopy(spec.config)
    config.hebbian_batch_size = args.batch_size
    torch.manual_seed(args.seed)
    layer = _build_layer(config).to(device)

    print(
        f"\n=== {spec.name} ===\n"
        f"train={args.train_samples} test={args.test_samples} passes={args.passes} "
        f"hebbian_batch={args.batch_size} probe_batch={args.probe_batch_size} "
        f"augmentation={spec.use_augmentation} layerwise={spec.use_layerwise}"
    )

    for pass_index in range(args.passes):
        train_images, train_labels = _training_samples_for_pass(
            spec,
            raw_train_samples=raw_train_samples,
            augmentation_factor=args.augmentation_factor,
            pass_index=pass_index,
            seed=args.seed,
        )
        if spec.use_layerwise:
            _run_layerwise_pass(
                layer,
                train_images,
                batch_size=args.batch_size,
                seed=args.seed + pass_index,
                device=device,
            )
        else:
            _run_standard_pass(
                layer,
                train_images,
                batch_size=args.batch_size,
                seed=args.seed + pass_index,
                device=device,
            )
        print(f"[pass {pass_index + 1}/{args.passes}] train_views={train_images.shape[0]}")

    train_features = _extract_features(layer, base_train_images, batch_size=args.batch_size, device=device)
    test_features = _extract_features(layer, base_test_images, batch_size=args.batch_size, device=device)

    nearest_centroid = _nearest_centroid_accuracy(
        train_features,
        base_train_labels,
        test_features,
        base_test_labels,
    )
    linear_probe = _linear_probe_accuracy(
        train_features,
        base_train_labels,
        test_features,
        base_test_labels,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_momentum=args.probe_momentum,
        probe_batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    print(
        f"nearest-centroid={nearest_centroid * 100:.2f}% "
        f"linear-probe={linear_probe * 100:.2f}%"
    )
    return BenchmarkResult(
        name=spec.name,
        nearest_centroid_accuracy=nearest_centroid,
        linear_probe_accuracy=linear_probe,
    )


def _format_percent(value: float | None) -> str:
    if value is None:
        return "ERR"
    return f"{value * 100:.2f}%"


def _summarize_table(results: list[BenchmarkResult]) -> str:
    rows = [
        "| Config | Nearest-Centroid | Linear Probe |",
        "|--------|-------------------|--------------|",
    ]
    for result in results:
        rows.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    _format_percent(None if result.error else result.nearest_centroid_accuracy),
                    _format_percent(None if result.error else result.linear_probe_accuracy),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _print_key_findings(results: list[BenchmarkResult]) -> None:
    successful = [result for result in results if result.error is None]
    if not successful:
        print("\nKey findings:\n- All configurations failed.")
        return

    better_probe_count = sum(
        int(result.linear_probe_accuracy > result.nearest_centroid_accuracy)
        for result in successful
    )
    best_linear = max(successful, key=lambda result: result.linear_probe_accuracy)
    best_centroid = max(successful, key=lambda result: result.nearest_centroid_accuracy)

    print("\nKey findings:")
    print(
        f"- Linear probe beat nearest-centroid on {better_probe_count}/{len(successful)} successful configs."
    )
    print(
        f"- Best linear-probe config: {best_linear.name} "
        f"({best_linear.linear_probe_accuracy * 100:.2f}%)."
    )
    print(
        f"- Best nearest-centroid config: {best_centroid.name} "
        f"({best_centroid.nearest_centroid_accuracy * 100:.2f}%)."
    )
    for result in results:
        if result.error is not None:
            print(f"- {result.name} failed: {result.error}")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = _select_device()

    print(f"Using device: {device}")
    print("Loading real CIFAR-10 via torchvision...")
    raw_train_samples, raw_test_samples = _load_real_cifar10(
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    raw_train_images, raw_train_labels = _stack_samples(raw_train_samples)
    raw_test_images, raw_test_labels = _stack_samples(raw_test_samples)
    print(
        f"Collected {raw_train_images.shape[0]} balanced train samples and "
        f"{raw_test_images.shape[0]} balanced test samples."
    )

    specs = _make_benchmark_specs(args.batch_size)
    results: list[BenchmarkResult] = []
    for spec in specs:
        try:
            results.append(
                _run_spec(
                    spec,
                    raw_train_samples=raw_train_samples,
                    base_train_images=raw_train_images,
                    base_train_labels=raw_train_labels,
                    base_test_images=raw_test_images,
                    base_test_labels=raw_test_labels,
                    args=args,
                    device=device,
                )
            )
        except Exception as exc:  # pragma: no cover - long-running benchmark robustness
            print(f"[error] {spec.name} failed: {exc}")
            results.append(
                BenchmarkResult(
                    name=spec.name,
                    nearest_centroid_accuracy=0.0,
                    linear_probe_accuracy=0.0,
                    error=str(exc),
                )
            )

    print("\n=== Linear Probe vs Nearest-Centroid — CIFAR-10 ===\n")
    print(_summarize_table(results))
    _print_key_findings(results)
    return 0 if any(result.error is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
