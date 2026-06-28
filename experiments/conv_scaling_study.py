"""Scale Hebbian convolutional features on real CIFAR-10."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig
from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.core.math_utils import cosine_similarity, normalize

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for conv_scaling_study.py") from exc


SEED = 7


@dataclass(frozen=True)
class ConvScalingSpec:
    name: str
    train_samples: int
    passes: int
    batch_size: int
    config: ConvCCCConfig


@dataclass(frozen=True)
class ScalingResult:
    name: str
    train_samples: int
    passes: int
    best_accuracy: float
    final_accuracy: float
    per_pass_accuracy: list[float]
    concept_dim: int


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
        raise RuntimeError(f"Unable to collect {expected} balanced samples from CIFAR-10.")
    return samples


def _load_real_cifar10(
    *,
    data_dir: Path,
    train_samples: int,
    test_samples: int,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    if train_samples % 10 != 0 or test_samples % 10 != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    return (
        _collect_balanced_samples(train_dataset, per_label=train_samples // 10, seed=SEED),
        _collect_balanced_samples(test_dataset, per_label=test_samples // 10, seed=SEED + 1),
    )


def _iter_batches(
    samples: list[tuple[torch.Tensor, int]],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    indices = list(range(len(samples)))
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=generator).tolist()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = [samples[index] for index in batch_indices]
        images = torch.stack([image for image, _ in batch], dim=0)
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        yield images, labels


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
    )


def _prototype_head(
    layer: ConvF1Layer,
    train_samples: list[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    sums: dict[int, torch.Tensor] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    with torch.inference_mode():
        for images, labels in _iter_batches(train_samples, batch_size, shuffle=False, seed=SEED):
            features = layer(images)
            for feature, label in zip(features, labels.tolist(), strict=True):
                if label not in sums:
                    sums[label] = torch.zeros(layer.output_dim, dtype=feature.dtype)
                sums[label].add_(feature)
                counts[label] += 1
    return {
        label: normalize((feature_sum / max(counts[label], 1)).reshape(1, -1)).squeeze(0)
        for label, feature_sum in sums.items()
    }


def _evaluate(
    layer: ConvF1Layer,
    prototypes: dict[int, torch.Tensor],
    test_samples: list[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
) -> float:
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in _iter_batches(test_samples, batch_size, shuffle=False, seed=SEED):
            features = layer(images)
            for feature, label in zip(features, labels.tolist(), strict=True):
                feature = normalize(feature.reshape(1, -1)).squeeze(0)
                scores = {
                    class_label: float(cosine_similarity(prototype, feature).item())
                    for class_label, prototype in prototypes.items()
                }
                prediction = max(scores.items(), key=lambda item: item[1])[0]
                correct += int(prediction == int(label))
                total += 1
    return correct / max(total, 1)


def _run_spec(
    spec: ConvScalingSpec,
    train_samples: list[tuple[torch.Tensor, int]],
    test_samples: list[tuple[torch.Tensor, int]],
) -> ScalingResult:
    config = copy.deepcopy(spec.config)
    config.hebbian_batch_size = spec.batch_size
    layer = _build_layer(config)
    per_pass_accuracy: list[float] = []

    print(
        f"\n=== {spec.name} ===\n"
        f"train={spec.train_samples} test={len(test_samples)} passes={spec.passes} "
        f"batch={spec.batch_size} dim={layer.output_dim}"
    )
    for pass_index in range(spec.passes):
        for images, _ in _iter_batches(
            train_samples,
            spec.batch_size,
            shuffle=True,
            seed=SEED + pass_index,
        ):
            layer.hebbian_update(images, learning_signal=torch.ones(images.shape[0], dtype=torch.float32))
        layer.flush_hebbian_updates()
        prototypes = _prototype_head(layer, train_samples, batch_size=spec.batch_size)
        accuracy = _evaluate(layer, prototypes, test_samples, batch_size=spec.batch_size)
        per_pass_accuracy.append(accuracy)
        print(f"[pass {pass_index + 1}/{spec.passes}] accuracy={accuracy * 100:.2f}%")

    return ScalingResult(
        name=spec.name,
        train_samples=spec.train_samples,
        passes=spec.passes,
        best_accuracy=max(per_pass_accuracy, default=0.0),
        final_accuracy=per_pass_accuracy[-1] if per_pass_accuracy else 0.0,
        per_pass_accuracy=per_pass_accuracy,
        concept_dim=layer.output_dim,
    )


def _format_summary(results: list[ScalingResult]) -> str:
    headers = ["config", "train", "passes", "best_acc", "final_acc", "per_pass"]
    rows = [
        [
            result.name,
            str(result.train_samples),
            str(result.passes),
            f"{result.best_accuracy * 100:.2f}%",
            f"{result.final_accuracy * 100:.2f}%",
            ", ".join(f"{value * 100:.2f}%" for value in result.per_pass_accuracy),
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    row_lines = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([header_line, divider, *row_lines])


def _specs() -> list[ConvScalingSpec]:
    return [
        ConvScalingSpec(
            name="small",
            train_samples=5000,
            passes=2,
            batch_size=128,
            config=ConvCCCConfig(
                num_conv_features=96,
                num_conv_layers=4,
                conv_hidden_channels=(48, 96, 128),
                conv_kernel_sizes=(5, 3, 3, 3),
                spatial_grid=4,
                f1_top_k=96,
                conv_hebbian_lr=0.0045,
                conv_competitive_k=24,
                spatial_top_k=8,
                feature_pool_avg_mix=0.35,
                hebbian_oja_decay=0.06,
                filter_decorrelation=0.03,
            ),
        ),
        ConvScalingSpec(
            name="medium",
            train_samples=10000,
            passes=3,
            batch_size=128,
            config=ConvCCCConfig(
                num_conv_features=160,
                num_conv_layers=5,
                conv_hidden_channels=(64, 96, 128, 160),
                conv_kernel_sizes=(5, 3, 3, 3, 3),
                spatial_grid=4,
                f1_top_k=128,
                conv_hebbian_lr=0.0040,
                conv_competitive_k=32,
                spatial_top_k=10,
                feature_pool_avg_mix=0.40,
                hebbian_oja_decay=0.08,
                filter_decorrelation=0.04,
            ),
        ),
        ConvScalingSpec(
            name="large",
            train_samples=20000,
            passes=4,
            batch_size=256,
            config=ConvCCCConfig(
                num_conv_features=224,
                num_conv_layers=5,
                conv_hidden_channels=(96, 160, 224, 224),
                conv_kernel_sizes=(5, 3, 3, 3, 3),
                spatial_grid=4,
                f1_top_k=160,
                conv_hebbian_lr=0.0035,
                conv_competitive_k=48,
                spatial_top_k=12,
                feature_pool_avg_mix=0.45,
                hebbian_oja_decay=0.10,
                filter_decorrelation=0.05,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-samples", type=int, default=1000)
    args = parser.parse_args()

    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    results: list[ScalingResult] = []
    for spec in _specs():
        train_samples, test_samples = _load_real_cifar10(
            data_dir=args.data_dir,
            train_samples=spec.train_samples,
            test_samples=args.test_samples,
        )
        results.append(_run_spec(spec, train_samples, test_samples))

    print("\nsummary:")
    print(_format_summary(results))
    best = max(results, key=lambda result: result.best_accuracy)
    print(
        f"\nbest_config={best.name} "
        f"best_accuracy={best.best_accuracy * 100:.2f}% "
        f"final_accuracy={best.final_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
