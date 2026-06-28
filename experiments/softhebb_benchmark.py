"""Run a SoftHebb ablation study on real CIFAR-10."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.config import ConvCCCConfig, deep_cifar_config
from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.data.base import DataSample, StreamingDataSource
from bioarn.data.vision import AugmentedCIFARStream, HebbianAugmentation
from bioarn.data.whitening import ZCAWhitening

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for softhebb_benchmark.py") from exc


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config: ConvCCCConfig
    use_augmentation: bool = False
    use_whitening: bool = False
    use_layerwise: bool = False


@dataclass
class BenchmarkResult:
    name: str
    per_pass_accuracy: list[float]
    best_accuracy: float
    final_accuracy: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=1000, help="Balanced train subset size.")
    parser.add_argument("--test-samples", type=int, default=500, help="Balanced test subset size.")
    parser.add_argument("--passes", type=int, default=3, help="Number of benchmark passes per configuration.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for updates and evaluation.")
    parser.add_argument("--augmentation-factor", type=int, default=2, help="Views per training sample when augmentation is enabled.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and initialization.")
    parser.add_argument(
        "--whitening-fit-samples",
        type=int,
        default=1000,
        help="Maximum number of train samples used to fit ZCA whitening.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive.")
    if args.train_samples % 10 != 0 or args.test_samples % 10 != 0:
        raise ValueError("train_samples and test_samples must be divisible by 10 for balanced sampling.")
    if args.passes < 3:
        raise ValueError("passes must be at least 3 for the ablation study.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.augmentation_factor <= 0:
        raise ValueError("augmentation_factor must be positive.")
    if args.whitening_fit_samples < 2:
        raise ValueError("whitening_fit_samples must be at least 2.")


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
    images: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor | None]]:
    indices = list(range(images.shape[0]))
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(images.shape[0], generator=generator).tolist()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_images = images[batch_indices]
        batch_labels = None if labels is None else labels[batch_indices]
        yield batch_images, batch_labels


def _iter_image_batches(
    images: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[torch.Tensor]:
    return [batch for batch, _ in _iter_batches(images, None, batch_size=batch_size, shuffle=shuffle, seed=seed)]


def _apply_whitening(
    train_images: torch.Tensor,
    test_images: torch.Tensor,
    *,
    fit_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, ZCAWhitening]:
    fit_count = min(max(2, fit_samples), train_images.shape[0])
    flat_train = train_images.reshape(train_images.shape[0], -1)
    flat_test = test_images.reshape(test_images.shape[0], -1)
    zca = ZCAWhitening()
    zca.fit(flat_train[:fit_count])
    whitened_train = zca.transform(flat_train).reshape_as(train_images)
    whitened_test = zca.transform(flat_test).reshape_as(test_images)
    return whitened_train, whitened_test, zca


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


def _make_benchmark_specs(batch_size: int) -> list[BenchmarkSpec]:
    baseline = ConvCCCConfig(hebbian_batch_size=batch_size)
    softhebb = ConvCCCConfig(
        hebbian_batch_size=batch_size,
        softhebb_enabled=True,
        softhebb_gamma=4.0,
        softhebb_beta=2.0,
    )
    deeper = deep_cifar_config()
    deeper.hebbian_batch_size = batch_size
    deeper.layerwise_train.samples_per_layer = max(deeper.layerwise_train.samples_per_layer, batch_size)
    all_config = copy.deepcopy(deeper)
    all_config.softhebb_enabled = True
    all_config.softhebb_gamma = 4.0
    all_config.softhebb_beta = 2.0
    return [
        BenchmarkSpec(name="baseline", config=baseline),
        BenchmarkSpec(name="+softhebb", config=softhebb),
        BenchmarkSpec(name="+augmentation", config=copy.deepcopy(baseline), use_augmentation=True),
        BenchmarkSpec(name="+whitening", config=copy.deepcopy(baseline), use_whitening=True),
        BenchmarkSpec(name="+deeper", config=deeper, use_layerwise=True),
        BenchmarkSpec(name="+softhebb+aug", config=copy.deepcopy(softhebb), use_augmentation=True),
        BenchmarkSpec(
            name="+all",
            config=all_config,
            use_augmentation=True,
            use_whitening=True,
            use_layerwise=True,
        ),
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
    whitened_train_images: torch.Tensor | None,
    whitening: ZCAWhitening | None,
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
        train_images, train_labels = _stack_samples(augmented_samples)
        if whitening is not None:
            train_images = whitening.transform(train_images.reshape(train_images.shape[0], -1)).reshape_as(train_images)
        return train_images, train_labels
    if whitened_train_images is not None:
        _, raw_train_labels = _stack_samples(raw_train_samples)
        return whitened_train_images, raw_train_labels
    raw_train_images, raw_train_labels = _stack_samples(raw_train_samples)
    return raw_train_images, raw_train_labels


def _prototype_head(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    sums: dict[int, torch.Tensor] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    with torch.inference_mode():
        for images, labels in _iter_batches(train_images, train_labels, batch_size=batch_size, shuffle=False, seed=0):
            features = layer(images)
            assert labels is not None
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
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in _iter_batches(test_images, test_labels, batch_size=batch_size, shuffle=False, seed=0):
            features = layer(images)
            assert labels is not None
            for feature, label in zip(features, labels.tolist(), strict=True):
                normalized = normalize(feature.reshape(1, -1)).squeeze(0)
                scores = {
                    class_label: float(cosine_similarity(prototype, normalized).item())
                    for class_label, prototype in prototypes.items()
                }
                prediction = max(scores.items(), key=lambda item: item[1])[0]
                correct += int(prediction == int(label))
                total += 1
    return correct / max(total, 1)


def _run_standard_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> None:
    for images, _ in _iter_batches(train_images, None, batch_size=batch_size, shuffle=True, seed=seed):
        layer.hebbian_update(images, learning_signal=torch.ones(images.shape[0], dtype=torch.float32))
    layer.flush_hebbian_updates()


def _run_layerwise_pass(
    layer: ConvF1Layer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    layerwise_batches = _iter_image_batches(train_images, batch_size=batch_size, shuffle=True, seed=seed)
    return layer.train_layerwise(
        layerwise_batches,
        samples_per_layer=train_images.shape[0],
        passes_per_layer=1,
    )


def _run_spec(
    spec: BenchmarkSpec,
    *,
    raw_train_samples: list[tuple[torch.Tensor, int]],
    base_train_images: torch.Tensor,
    base_test_images: torch.Tensor,
    base_test_labels: torch.Tensor,
    whitened_train_images: torch.Tensor | None,
    whitened_test_images: torch.Tensor | None,
    whitening: ZCAWhitening | None,
    args: argparse.Namespace,
) -> BenchmarkResult:
    config = copy.deepcopy(spec.config)
    config.hebbian_batch_size = args.batch_size
    torch.manual_seed(args.seed)
    layer = _build_layer(config)
    eval_test_images = whitened_test_images if spec.use_whitening and whitened_test_images is not None else base_test_images
    per_pass_accuracy: list[float] = []

    print(
        f"\n=== {spec.name} ===\n"
        f"train={args.train_samples} test={args.test_samples} passes={args.passes} "
        f"batch={args.batch_size} layerwise={spec.use_layerwise} "
        f"augmentation={spec.use_augmentation} whitening={spec.use_whitening}"
    )

    for pass_index in range(args.passes):
        train_images, train_labels = _training_samples_for_pass(
            spec,
            raw_train_samples=raw_train_samples,
            whitened_train_images=whitened_train_images if spec.use_whitening else None,
            whitening=whitening if spec.use_whitening else None,
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
            )
        else:
            _run_standard_pass(
                layer,
                train_images,
                batch_size=args.batch_size,
                seed=args.seed + pass_index,
            )
        prototypes = _prototype_head(
            layer,
            train_images,
            train_labels,
            batch_size=args.batch_size,
        )
        accuracy = _evaluate(
            layer,
            prototypes,
            eval_test_images,
            base_test_labels,
            batch_size=args.batch_size,
        )
        per_pass_accuracy.append(accuracy)
        print(
            f"[pass {pass_index + 1}/{args.passes}] "
            f"train_views={train_images.shape[0]} accuracy={accuracy * 100:.2f}%"
        )

    best_accuracy = max(per_pass_accuracy, default=0.0)
    final_accuracy = per_pass_accuracy[-1] if per_pass_accuracy else 0.0
    return BenchmarkResult(
        name=spec.name,
        per_pass_accuracy=per_pass_accuracy,
        best_accuracy=best_accuracy,
        final_accuracy=final_accuracy,
    )


def _format_value(value: float | None) -> str:
    if value is None:
        return "ERR"
    return f"{value * 100:.2f}%"


def _summarize_table(results: list[BenchmarkResult], passes: int) -> str:
    headers = ["Config", *[f"Pass {index + 1}" for index in range(passes)], "Best", "Final"]
    separator = "|" + "|".join("-" * (len(header) + 2) for header in headers) + "|"
    rows = ["| " + " | ".join(headers) + " |", separator]
    for result in results:
        pass_cells = [
            _format_value(result.per_pass_accuracy[index]) if index < len(result.per_pass_accuracy) else "ERR"
            for index in range(passes)
        ]
        if result.error is not None:
            pass_cells = ["ERR" for _ in range(passes)]
        rows.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    *pass_cells,
                    _format_value(None if result.error is not None else result.best_accuracy),
                    _format_value(None if result.error is not None else result.final_accuracy),
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
    ranked = sorted(successful, key=lambda result: (result.best_accuracy, result.final_accuracy), reverse=True)
    best = ranked[0]
    baseline = next((result for result in successful if result.name == "baseline"), None)
    print("\nKey findings:")
    print(f"- Best configuration: {best.name} (best={best.best_accuracy * 100:.2f}%, final={best.final_accuracy * 100:.2f}%).")
    if baseline is not None and best.name != baseline.name:
        delta = (best.best_accuracy - baseline.best_accuracy) * 100.0
        print(f"- Improvement over baseline best accuracy: {delta:+.2f} percentage points.")
    for result in results:
        if result.error is not None:
            print(f"- {result.name} failed: {result.error}")


def main() -> int:
    args = parse_args()
    _validate_args(args)

    print("Loading real CIFAR-10 via torchvision...")
    raw_train_samples, raw_test_samples = _load_real_cifar10(
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    raw_train_images, _ = _stack_samples(raw_train_samples)
    raw_test_images, raw_test_labels = _stack_samples(raw_test_samples)

    print(
        f"Collected {raw_train_images.shape[0]} balanced train samples and "
        f"{raw_test_images.shape[0]} balanced test samples."
    )

    print("Fitting shared ZCA whitening cache...")
    whitened_train_images, whitened_test_images, whitening = _apply_whitening(
        raw_train_images,
        raw_test_images,
        fit_samples=args.whitening_fit_samples,
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
                    base_test_images=raw_test_images,
                    base_test_labels=raw_test_labels,
                    whitened_train_images=whitened_train_images,
                    whitened_test_images=whitened_test_images,
                    whitening=whitening,
                    args=args,
                )
            )
        except Exception as exc:  # pragma: no cover - handled for long-running benchmark robustness
            print(f"[error] {spec.name} failed: {exc}")
            results.append(
                BenchmarkResult(
                    name=spec.name,
                    per_pass_accuracy=[],
                    best_accuracy=0.0,
                    final_accuracy=0.0,
                    error=str(exc),
                )
            )

    print("\n=== SoftHebb Ablation Study — CIFAR-10 ===\n")
    print(_summarize_table(results, args.passes))
    _print_key_findings(results)
    return 0 if any(result.error is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
