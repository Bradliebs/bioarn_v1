"""Progressive Hebbian scaling experiments on real CIFAR-10."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.core.conv_ccc import ConvF1Layer

try:
    from torchvision import datasets
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for hebbian_scaling.py") from exc


SEED = 42
NUM_CLASSES = 10
SUBSET_TRAIN_SAMPLES = 5_000
SUBSET_TEST_SAMPLES = 1_000
FULL_TRAIN_SAMPLES = 50_000
FULL_TEST_SAMPLES = 10_000
TRAIN_PASSES = 50
CHECKPOINTS = (1, 5, 10, 20, 30, 40, 50)
NUM_FEATURES = 256
HIDDEN_CHANNELS = (128, 256)
KERNEL_SIZES = (5, 3, 3)
SPATIAL_GRID = 4
SPATIAL_TOP_K = 8
TOP_K = 128
COMPETITIVE_K = 32
HEBBIAN_LR = 0.005
HEBBIAN_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
PROBE_EPOCHS = 100
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9
BASELINE_LINEAR_PROBE = 0.20
FULL_DATA_RUNTIME_LIMIT_HOURS = 3.0
FULL_DATA_PASS_CAP = 30
FULL_DATA_MIN_PASSES = 5
DIVISIVE_SIGMA = 0.1
DIVISIVE_NEIGHBORHOOD = 5
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
class SparseFeatureSet:
    indices: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    feature_dim: int


@dataclass(frozen=True)
class PassMetrics:
    pass_num: int
    nearest_centroid: float
    linear_probe: float


@dataclass
class ExperimentResult:
    name: str
    train_samples: int
    test_samples: int
    planned_passes: int
    executed_passes: int
    pass_metrics: list[PassMetrics]
    notes: list[str] = field(default_factory=list)

    @property
    def best_linear_probe(self) -> float:
        return max((metric.linear_probe for metric in self.pass_metrics), default=0.0)

    @property
    def best_nearest_centroid(self) -> float:
        return max((metric.nearest_centroid for metric in self.pass_metrics), default=0.0)

    @property
    def best_pass(self) -> int:
        if not self.pass_metrics:
            return 0
        return max(self.pass_metrics, key=lambda metric: metric.linear_probe).pass_num

    @property
    def metrics_by_pass(self) -> dict[int, PassMetrics]:
        return {metric.pass_num: metric for metric in self.pass_metrics}


class DivisiveNormalization(nn.Module):
    """Bio-plausible divisive normalization."""

    def __init__(self, num_channels: int, sigma: float = DIVISIVE_SIGMA, neighborhood_size: int = DIVISIVE_NEIGHBORHOOD) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.sigma = float(sigma)
        self.neighborhood_size = int(max(1, neighborhood_size))
        kernel = torch.ones(1, 1, self.neighborhood_size, dtype=torch.float32) / float(self.neighborhood_size)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.square()
        batch_size, num_channels, height, width = x.shape
        x_sq_flat = x_sq.permute(0, 2, 3, 1).reshape(-1, 1, num_channels)
        pad = self.neighborhood_size // 2
        channel_pool = F.conv1d(x_sq_flat, self.kernel, padding=pad)
        channel_pool = channel_pool.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)
        spatial_pool = F.avg_pool2d(
            x_sq,
            kernel_size=self.neighborhood_size,
            stride=1,
            padding=self.neighborhood_size // 2,
        )
        norm_pool = 0.5 * (channel_pool + spatial_pool)
        denominator = torch.sqrt(self.sigma + norm_pool)
        return x / denominator


class NormalizedConvF1Layer(ConvF1Layer):
    """ConvF1Layer variant with optional divisive normalization after each activation."""

    def __init__(
        self,
        *args,
        use_divisive_norm: bool = False,
        divisive_sigma: float = DIVISIVE_SIGMA,
        divisive_neighborhood_size: int = DIVISIVE_NEIGHBORHOOD,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_divisive_norm = bool(use_divisive_norm)
        self.divisive_norms = nn.ModuleList(
            [
                DivisiveNormalization(
                    layer.out_channels,
                    sigma=divisive_sigma,
                    neighborhood_size=divisive_neighborhood_size,
                )
                for layer in self.conv_layers
            ]
        )
        self._layer_indices = {id(layer): index for index, layer in enumerate(self.conv_layers)}
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _apply_layer(self, layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        activations = super()._apply_layer(layer, x)
        if not self.use_divisive_norm:
            return activations
        layer_index = self._layer_indices[id(layer)]
        return self.divisive_norms[layer_index](activations)


class SparseLinearProbe(nn.Module):
    """Linear classifier that consumes compact sparse feature batches."""

    def __init__(self, feature_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(self.feature_dim, self.num_classes, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(self.num_classes, dtype=torch.float32))
        bound = 1.0 / math.sqrt(max(self.feature_dim, 1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(indices, self.weight)
        return (selected * values.unsqueeze(-1)).sum(dim=1) + self.bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Torchvision dataset root.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--subset-train-samples", type=int, default=SUBSET_TRAIN_SAMPLES, help="Balanced subset training size for experiment 1.")
    parser.add_argument("--subset-test-samples", type=int, default=SUBSET_TEST_SAMPLES, help="Balanced subset test size for experiment 1.")
    parser.add_argument("--full-train-samples", type=int, default=FULL_TRAIN_SAMPLES, help="Training size for experiments 2 and 3.")
    parser.add_argument("--full-test-samples", type=int, default=FULL_TEST_SAMPLES, help="Test size for experiments 2 and 3.")
    parser.add_argument("--subset-passes", type=int, default=TRAIN_PASSES, help="Experiment 1 Hebbian passes.")
    parser.add_argument("--full-passes", type=int, default=TRAIN_PASSES, help="Experiments 2 and 3 Hebbian passes.")
    parser.add_argument("--probe-epochs", type=int, default=PROBE_EPOCHS, help="Linear probe SGD epochs.")
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Hebbian batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE, help="Feature extraction batch size.")
    parser.add_argument("--probe-batch-size", type=int, default=PROBE_BATCH_SIZE, help="Linear probe batch size.")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS), help="Pass numbers to evaluate.")
    parser.add_argument("--full-runtime-limit-hours", type=float, default=FULL_DATA_RUNTIME_LIMIT_HOURS, help="If projected runtime exceeds this, cap full-data runs.")
    parser.add_argument("--full-pass-cap", type=int, default=FULL_DATA_PASS_CAP, help="Reduced full-data pass budget when the runtime cap triggers.")
    parser.add_argument("--threads", type=int, default=min(8, max(1, os.cpu_count() or 1)), help="Torch CPU thread count.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pp"


def _load_cifar10_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    if tuple(train_dataset.classes) != CLASS_NAMES:
        raise RuntimeError("Unexpected CIFAR-10 class order from torchvision.")

    train_images = torch.from_numpy(train_dataset.data).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
    test_images = torch.from_numpy(test_dataset.data).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)
    return train_images, train_labels, test_images, test_labels


def _collect_balanced_indices(labels: torch.Tensor, *, per_label: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(labels.shape[0], generator=generator).tolist()
    counts: defaultdict[int, int] = defaultdict(int)
    selected: list[int] = []
    for index in order:
        label = int(labels[index].item())
        if counts[label] >= per_label:
            continue
        selected.append(index)
        counts[label] += 1
        if all(counts[class_id] >= per_label for class_id in range(NUM_CLASSES)):
            break
    expected = per_label * NUM_CLASSES
    if len(selected) != expected:
        raise RuntimeError(f"Unable to collect {expected} balanced CIFAR-10 samples.")
    return torch.tensor(selected, dtype=torch.long)


def _subset_by_indices(images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return images[indices].clone(), labels[indices].clone()


def _class_index_buckets(labels: torch.Tensor) -> dict[int, list[int]]:
    buckets: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        buckets[int(label)].append(index)
    return {label: buckets[label] for label in range(NUM_CLASSES)}


def _build_interleaved_indices(class_indices: dict[int, list[int]], *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shuffled: dict[int, list[int]] = {}
    for label in range(NUM_CLASSES):
        indices = class_indices[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled[label] = [indices[position] for position in order]
    ordered: list[int] = []
    positions = {label: 0 for label in range(NUM_CLASSES)}
    while True:
        emitted = False
        for label in range(NUM_CLASSES):
            position = positions[label]
            bucket = shuffled[label]
            if position < len(bucket):
                ordered.append(bucket[position])
                positions[label] += 1
                emitted = True
        if not emitted:
            break
    return torch.tensor(ordered, dtype=torch.long)


def _batch_indices(indices: torch.Tensor, *, batch_size: int):
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _iter_label_batches(labels: torch.Tensor, *, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(labels.shape[0], generator=generator)
    else:
        indices = torch.arange(labels.shape[0])
    for start in range(0, labels.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _build_conv_f1(*, hebbian_batch_size: int = HEBBIAN_BATCH_SIZE) -> ConvF1Layer:
    return ConvF1Layer(
        in_channels=3,
        num_features=NUM_FEATURES,
        spatial_size=32,
        top_k=TOP_K,
        spatial_grid=SPATIAL_GRID,
        num_layers=3,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_sizes=KERNEL_SIZES,
        spatial_top_k=SPATIAL_TOP_K,
        competitive_k=COMPETITIVE_K,
        hebbian_lr=HEBBIAN_LR,
        hebbian_batch_size=hebbian_batch_size,
        weight_norm_target=1.0,
        enable_local_contrast_norm=True,
        hebbian_oja_decay=0.05,
        filter_decorrelation=0.02,
    )


def _prepare_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    requested_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if requested_samples == images.shape[0]:
        return images, labels
    if requested_samples % NUM_CLASSES != 0:
        raise ValueError("Reduced train/test sample counts must be divisible by 10.")
    indices = _collect_balanced_indices(labels, per_label=requested_samples // NUM_CLASSES, seed=seed)
    return _subset_by_indices(images, labels, indices)


def _build_normalized_conv_f1(*, use_divisive_norm: bool, hebbian_batch_size: int = HEBBIAN_BATCH_SIZE) -> NormalizedConvF1Layer:
    return NormalizedConvF1Layer(
        in_channels=3,
        num_features=NUM_FEATURES,
        spatial_size=32,
        top_k=TOP_K,
        spatial_grid=SPATIAL_GRID,
        num_layers=3,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_sizes=KERNEL_SIZES,
        spatial_top_k=SPATIAL_TOP_K,
        competitive_k=COMPETITIVE_K,
        hebbian_lr=HEBBIAN_LR,
        hebbian_batch_size=hebbian_batch_size,
        weight_norm_target=1.0,
        enable_local_contrast_norm=True,
        hebbian_oja_decay=0.05,
        filter_decorrelation=0.02,
        use_divisive_norm=use_divisive_norm,
    )


def _run_hebbian_pass(model: ConvF1Layer, images: torch.Tensor, ordered_indices: torch.Tensor, *, batch_size: int, device: torch.device) -> None:
    for batch_indices in _batch_indices(ordered_indices, batch_size=batch_size):
        batch_images = images[batch_indices].to(device=device, dtype=torch.float32)
        learning_signal = torch.ones(batch_images.shape[0], device=device, dtype=torch.float32)
        model.hebbian_update(batch_images, learning_signal=learning_signal)
    model.flush_hebbian_updates()


@torch.inference_mode()
def _extract_sparse_features(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    sparse_k: int,
    device: torch.device,
) -> SparseFeatureSet:
    model.eval()
    total = images.shape[0]
    feature_dim = getattr(model, "output_dim")
    top_k = min(int(sparse_k), int(feature_dim))
    all_indices = torch.empty((total, top_k), dtype=torch.int32)
    all_values = torch.empty((total, top_k), dtype=torch.float32)
    cursor = 0
    for batch_indices in _iter_label_batches(labels, batch_size=batch_size, shuffle=False, seed=0):
        batch_images = images[batch_indices].to(device=device, dtype=torch.float32)
        dense_batch = model(batch_images).cpu().to(torch.float32)
        top_values, top_indices = torch.topk(dense_batch, k=top_k, dim=1)
        batch_size_now = dense_batch.shape[0]
        all_indices[cursor : cursor + batch_size_now] = top_indices.to(torch.int32)
        all_values[cursor : cursor + batch_size_now] = top_values
        cursor += batch_size_now
    return SparseFeatureSet(
        indices=all_indices,
        values=all_values,
        labels=labels.clone(),
        feature_dim=int(feature_dim),
    )


def _iter_sparse_batches(features: SparseFeatureSet, *, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(features.labels.shape[0], generator=generator)
    else:
        order = torch.arange(features.labels.shape[0])
    for start in range(0, order.shape[0], batch_size):
        batch = order[start : start + batch_size]
        yield features.indices[batch], features.values[batch], features.labels[batch]


def _nearest_centroid_accuracy(train_features: SparseFeatureSet, test_features: SparseFeatureSet) -> float:
    prototype_sums = torch.zeros((NUM_CLASSES, train_features.feature_dim), dtype=torch.float32)
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)

    for batch_indices, batch_values, batch_labels in _iter_sparse_batches(
        train_features,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        seed=0,
    ):
        label_rows = batch_labels.unsqueeze(1).expand_as(batch_indices).reshape(-1).to(torch.long)
        feature_cols = batch_indices.reshape(-1).to(torch.long)
        prototype_sums.index_put_(
            (label_rows, feature_cols),
            batch_values.reshape(-1),
            accumulate=True,
        )
        class_counts += torch.bincount(batch_labels, minlength=NUM_CLASSES).to(torch.float32)

    if bool((class_counts == 0).any().item()):
        missing = torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Missing classes in train features: {missing}")

    prototypes = F.normalize(prototype_sums / class_counts.unsqueeze(1), dim=1)
    prototype_lookup = prototypes.transpose(0, 1).contiguous()

    correct = 0
    total = 0
    for batch_indices, batch_values, batch_labels in _iter_sparse_batches(
        test_features,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        seed=0,
    ):
        selected = F.embedding(batch_indices.to(torch.long), prototype_lookup)
        logits = (selected * batch_values.unsqueeze(-1)).sum(dim=1)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == batch_labels).sum().item())
        total += int(batch_labels.numel())
    return correct / max(total, 1)


def _linear_probe_accuracy(
    train_features: SparseFeatureSet,
    test_features: SparseFeatureSet,
    *,
    seed: int,
    probe_epochs: int,
    probe_batch_size: int,
    device: torch.device,
) -> float:
    _set_seed(seed)
    probe = SparseLinearProbe(train_features.feature_dim).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(probe_epochs):
        probe.train()
        for batch_indices, batch_values, batch_labels in _iter_sparse_batches(
            train_features,
            batch_size=probe_batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_indices.to(device=device, dtype=torch.long), batch_values.to(device=device, dtype=torch.float32))
            loss = criterion(logits, batch_labels.to(device=device, dtype=torch.long))
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_indices, batch_values, batch_labels in _iter_sparse_batches(
            test_features,
            batch_size=probe_batch_size,
            shuffle=False,
            seed=0,
        ):
            logits = probe(batch_indices.to(device=device, dtype=torch.long), batch_values.to(device=device, dtype=torch.float32))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch_labels.to(device)).sum().item())
            total += int(batch_labels.numel())
    return correct / max(total, 1)


def _evaluate_checkpoint(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    eval_batch_size: int,
    probe_batch_size: int,
    probe_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    train_features = _extract_sparse_features(
        model,
        train_images,
        train_labels,
        batch_size=eval_batch_size,
        sparse_k=TOP_K,
        device=device,
    )
    test_features = _extract_sparse_features(
        model,
        test_images,
        test_labels,
        batch_size=eval_batch_size,
        sparse_k=TOP_K,
        device=device,
    )
    nearest_centroid = _nearest_centroid_accuracy(train_features, test_features)
    linear_probe = _linear_probe_accuracy(
        train_features,
        test_features,
        seed=seed,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        device=device,
    )
    return nearest_centroid, linear_probe


def _maybe_apply_runtime_cap(
    *,
    pass_times: list[float],
    planned_passes: int,
    current_limit_hours: float,
    pass_cap: int,
    min_passes: int = FULL_DATA_MIN_PASSES,
) -> tuple[int | None, str | None]:
    if planned_passes <= pass_cap or len(pass_times) < 5 or current_limit_hours <= 0:
        return None, None
    average_pass_seconds = sum(pass_times) / len(pass_times)
    projected_hours = average_pass_seconds * planned_passes / 3600.0
    if projected_hours <= current_limit_hours:
        return None, None
    limit_seconds = current_limit_hours * 3600.0
    max_passes_by_limit = int(limit_seconds // max(average_pass_seconds, 1e-6))
    adjusted_cap = min(planned_passes, pass_cap, max(min_passes, max_passes_by_limit))
    return (
        adjusted_cap,
        f"Projected full-data runtime {projected_hours:.1f}h exceeded {current_limit_hours:.1f}h; "
        f"capped subsequent full-data runs to {adjusted_cap} passes."
    )


def _run_experiment(
    *,
    name: str,
    model_builder,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    planned_passes: int,
    checkpoints: tuple[int, ...],
    eval_batch_size: int,
    probe_batch_size: int,
    probe_epochs: int,
    seed: int,
    train_batch_size: int,
    device: torch.device,
    runtime_limit_hours: float = 0.0,
    runtime_pass_cap: int | None = None,
) -> ExperimentResult:
    _set_seed(seed)
    model = model_builder().to(device)
    class_indices = _class_index_buckets(train_labels)
    metrics: list[PassMetrics] = []
    notes: list[str] = []
    pass_times: list[float] = []
    pass_limit = int(planned_passes)

    print(
        f"[{name}] train={train_images.shape[0]} test={test_images.shape[0]} "
        f"passes={planned_passes} dim={model.output_dim}",
        flush=True,
    )

    for pass_index in range(pass_limit):
        pass_num = pass_index + 1
        ordered_indices = _build_interleaved_indices(class_indices, seed=seed + pass_index)
        pass_start = time.perf_counter()
        _run_hebbian_pass(model, train_images, ordered_indices, batch_size=train_batch_size, device=device)
        pass_time = time.perf_counter() - pass_start
        pass_times.append(pass_time)
        print(f"[{name}] pass {pass_num}/{pass_limit} train {pass_time:.1f}s", flush=True)

        if runtime_pass_cap is not None:
            adjusted_cap, cap_note = _maybe_apply_runtime_cap(
                pass_times=pass_times,
                planned_passes=planned_passes,
                current_limit_hours=runtime_limit_hours,
                pass_cap=runtime_pass_cap,
            )
            if cap_note and adjusted_cap is not None and pass_limit != adjusted_cap:
                pass_limit = int(adjusted_cap)
                notes.append(cap_note)
                print(f"[{name}] NOTE: {cap_note}", flush=True)

        if pass_num in checkpoints or pass_num == pass_limit:
            eval_start = time.perf_counter()
            nearest_centroid, linear_probe = _evaluate_checkpoint(
                model,
                train_images,
                train_labels,
                test_images,
                test_labels,
                eval_batch_size=eval_batch_size,
                probe_batch_size=probe_batch_size,
                probe_epochs=probe_epochs,
                seed=seed,
                device=device,
            )
            metrics.append(
                PassMetrics(
                    pass_num=pass_num,
                    nearest_centroid=nearest_centroid,
                    linear_probe=linear_probe,
                )
            )
            print(
                f"[{name}] eval pass {pass_num}: nc={nearest_centroid * 100:.2f}% "
                f"lp={linear_probe * 100:.2f}% elapsed={(time.perf_counter() - eval_start) / 60:.1f}m",
                flush=True,
            )

        if pass_num >= pass_limit:
            break

    return ExperimentResult(
        name=name,
        train_samples=int(train_images.shape[0]),
        test_samples=int(test_images.shape[0]),
        planned_passes=int(planned_passes),
        executed_passes=int(pass_limit),
        pass_metrics=metrics,
        notes=notes,
    )


def _format_experiment_table(result: ExperimentResult, *, checkpoints: tuple[int, ...]) -> str:
    display_passes = sorted({*checkpoints, result.executed_passes})
    lines = [
        "| Pass | Nearest-Centroid | Linear Probe |",
        "|------|------------------|--------------|",
    ]
    by_pass = result.metrics_by_pass
    for checkpoint in display_passes:
        metric = by_pass.get(checkpoint)
        lines.append(
            f"| {checkpoint:<4} | {_pct(None if metric is None else metric.nearest_centroid):<16} | "
            f"{_pct(None if metric is None else metric.linear_probe):<12} |"
        )
    return "\n".join(lines)


def _format_div_norm_table(
    no_norm: ExperimentResult,
    div_norm: ExperimentResult,
    *,
    checkpoints: tuple[int, ...],
) -> str:
    display_passes = sorted({*checkpoints, no_norm.executed_passes, div_norm.executed_passes})
    lines = [
        "| Pass | NC (no norm) | NC (div norm) | LP (no norm) | LP (div norm) |",
        "|------|--------------|---------------|--------------|---------------|",
    ]
    no_norm_by_pass = no_norm.metrics_by_pass
    div_norm_by_pass = div_norm.metrics_by_pass
    for checkpoint in display_passes:
        no_metric = no_norm_by_pass.get(checkpoint)
        div_metric = div_norm_by_pass.get(checkpoint)
        lines.append(
            f"| {checkpoint:<4} | {_pct(None if no_metric is None else no_metric.nearest_centroid):<12} | "
            f"{_pct(None if div_metric is None else div_metric.nearest_centroid):<13} | "
            f"{_pct(None if no_metric is None else no_metric.linear_probe):<12} | "
            f"{_pct(None if div_metric is None else div_metric.linear_probe):<13} |"
        )
    return "\n".join(lines)


def _write_decision(
    *,
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    exp3_div: ExperimentResult,
    revised_ceiling: float,
) -> Path:
    capacity_gain = exp1.best_linear_probe - BASELINE_LINEAR_PROBE
    data_gain = exp2.best_linear_probe - exp1.best_linear_probe
    norm_gain = exp3_div.best_linear_probe - exp2.best_linear_probe

    if norm_gain >= 0.02:
        next_step = "Keep divisive normalization and probe deeper/layerwise schedules; normalization is still buying clear headroom."
    elif data_gain >= 0.02:
        next_step = "Prioritize more data or stronger curriculum/augmentation; scaling data still matters more than extra normalization."
    else:
        next_step = "Diminishing returns are setting in; the next step should be a new local learning rule or hybrid supervised readout."

    decision_path = Path(".squad") / "decisions" / "inbox" / "tank-scaling-results.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        "\n".join(
            [
                "### 2026-06-29: Hebbian Scaling Results",
                "**By:** Tank (Core Neural)",
                (
                    f"**What:** Progressive scaling achieved {revised_ceiling * 100:.1f}% on CIFAR-10 "
                    "(up from ~20% baseline)"
                ),
                (
                    f"**Key findings:** capacity: {_pp(capacity_gain)}, full data: {_pp(data_gain)}, "
                    f"div norm: {_pp(norm_gain)}"
                ),
                f"**Revised ceiling:** {revised_ceiling * 100:.1f}%",
                f"**Next recommended step:** {next_step}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return decision_path


def _print_report(
    *,
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    exp3_no_norm: ExperimentResult,
    exp3_div_norm: ExperimentResult,
    checkpoints: tuple[int, ...],
    device: torch.device,
) -> None:
    exp1_delta = exp1.best_linear_probe - BASELINE_LINEAR_PROBE
    exp2_delta = exp2.best_linear_probe - exp1.best_linear_probe
    exp3_delta = exp3_div_norm.best_linear_probe - exp2.best_linear_probe
    best_overall = max(exp1.best_linear_probe, exp2.best_linear_probe, exp3_div_norm.best_linear_probe)

    print("\n================================================================", flush=True)
    print("HEBBIAN SCALING — PROGRESSIVE EXPERIMENTS", flush=True)
    print("================================================================", flush=True)
    print(f"Seed: 42 | Device: {device}", flush=True)
    print("", flush=True)

    print(
        f"--- EXPERIMENT 1: Combined Scale (256 feat × {exp1.planned_passes} passes × "
        f"{exp1.train_samples // 1000}K data) ---",
        flush=True,
    )
    print(_format_experiment_table(exp1, checkpoints=checkpoints), flush=True)
    print(f"\nBest: {_pct(exp1.best_linear_probe)} (linear probe, pass {exp1.best_pass})", flush=True)
    for note in exp1.notes:
        print(f"Note: {note}", flush=True)
    print("", flush=True)

    print(
        f"--- EXPERIMENT 2: Full Dataset (256 feat × {exp2.planned_passes} passes × "
        f"{exp2.train_samples // 1000}K data) ---",
        flush=True,
    )
    print(_format_experiment_table(exp2, checkpoints=checkpoints), flush=True)
    print(f"\nBest: {_pct(exp2.best_linear_probe)} (linear probe, pass {exp2.best_pass})", flush=True)
    print(f"Δ vs Exp 1: {_pp(exp2_delta)} (effect of 10x more data)", flush=True)
    for note in exp2.notes:
        print(f"Note: {note}", flush=True)
    print("", flush=True)

    print(
        f"--- EXPERIMENT 3: Divisive Normalization (256 feat × {exp3_div_norm.planned_passes} passes × "
        f"{exp3_div_norm.train_samples // 1000}K data) ---",
        flush=True,
    )
    print(_format_div_norm_table(exp3_no_norm, exp3_div_norm, checkpoints=checkpoints), flush=True)
    print(f"\nBest: {_pct(exp3_div_norm.best_linear_probe)} (linear probe, pass {exp3_div_norm.best_pass})", flush=True)
    print(f"Δ vs Exp 2 (no norm): {_pp(exp3_delta)} (effect of bio-plausible normalization)", flush=True)
    print("Note: NC/LP no-norm columns reuse Experiment 2 (same ConvF1Layer architecture without divisive normalization).", flush=True)
    for note in exp3_no_norm.notes:
        print(f"Note: {note}", flush=True)
    for note in exp3_div_norm.notes:
        print(f"Note: {note}", flush=True)

    print("\n================================================================", flush=True)
    print("PROGRESSIVE SCALING SUMMARY", flush=True)
    print("================================================================", flush=True)
    print("| Experiment | Config | Best LP Accuracy | Δ vs Previous |", flush=True)
    print("|-----------|--------|------------------|---------------|", flush=True)
    print("| Bias audit baseline | 64 feat, 10 pass, 5K | ~20.0% | — |", flush=True)
    print(
        f"| Exp 1: Scale | 256 feat, {exp1.executed_passes} pass, 5K | {_pct(exp1.best_linear_probe)} | {_pp(exp1_delta)} |",
        flush=True,
    )
    print(
        f"| Exp 2: Full data | 256 feat, {exp2.executed_passes} pass, 50K | {_pct(exp2.best_linear_probe)} | {_pp(exp2_delta)} |",
        flush=True,
    )
    print(
        f"| Exp 3: Div norm | 256 feat, {exp3_div_norm.executed_passes} pass, 50K, norm | "
        f"{_pct(exp3_div_norm.best_linear_probe)} | {_pp(exp3_delta)} |",
        flush=True,
    )
    print("", flush=True)
    print(f"Total improvement over original baseline: {_pp(best_overall - BASELINE_LINEAR_PROBE)}", flush=True)
    print(f"Revised CIFAR-10 ceiling: {_pct(best_overall)}", flush=True)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using torch threads: {torch.get_num_threads()}", flush=True)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    checkpoints = tuple(sorted({int(checkpoint) for checkpoint in args.checkpoints if int(checkpoint) > 0}))
    if not checkpoints:
        raise ValueError("At least one positive checkpoint is required.")
    if args.subset_train_samples % NUM_CLASSES != 0 or args.subset_test_samples % NUM_CLASSES != 0:
        raise ValueError("Subset train/test sample counts must be divisible by 10.")
    if args.full_train_samples <= 0 or args.full_test_samples <= 0:
        raise ValueError("Full train/test sample counts must be positive.")
    if args.full_train_samples > FULL_TRAIN_SAMPLES or args.full_test_samples > FULL_TEST_SAMPLES:
        raise ValueError("Requested full-data sample counts exceed CIFAR-10 dataset size.")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0 or args.probe_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.probe_epochs <= 0:
        raise ValueError("Probe epochs must be positive.")

    print("Loading real CIFAR-10 via torchvision...", flush=True)
    load_start = time.perf_counter()
    train_images_all, train_labels_all, test_images_all, test_labels_all = _load_cifar10_tensors(args.data_dir)
    print(f"Loaded CIFAR-10 in {time.perf_counter() - load_start:.1f}s", flush=True)

    subset_train_indices = _collect_balanced_indices(
        train_labels_all,
        per_label=args.subset_train_samples // NUM_CLASSES,
        seed=args.seed,
    )
    subset_test_indices = _collect_balanced_indices(
        test_labels_all,
        per_label=args.subset_test_samples // NUM_CLASSES,
        seed=args.seed + 1,
    )
    subset_train_images, subset_train_labels = _subset_by_indices(train_images_all, train_labels_all, subset_train_indices)
    subset_test_images, subset_test_labels = _subset_by_indices(test_images_all, test_labels_all, subset_test_indices)

    full_train_images, full_train_labels = _prepare_split(
        train_images_all,
        train_labels_all,
        requested_samples=args.full_train_samples,
        seed=args.seed + 2,
    )
    full_test_images, full_test_labels = _prepare_split(
        test_images_all,
        test_labels_all,
        requested_samples=args.full_test_samples,
        seed=args.seed + 3,
    )

    print("\nRunning experiment 1...", flush=True)
    exp1 = _run_experiment(
        name="exp1-scale",
        model_builder=lambda: _build_conv_f1(hebbian_batch_size=args.train_batch_size),
        train_images=subset_train_images,
        train_labels=subset_train_labels,
        test_images=subset_test_images,
        test_labels=subset_test_labels,
        planned_passes=args.subset_passes,
        checkpoints=checkpoints,
        eval_batch_size=args.eval_batch_size,
        probe_batch_size=args.probe_batch_size,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        device=device,
    )

    print("\nRunning experiment 2...", flush=True)
    exp2 = _run_experiment(
        name="exp2-full",
        model_builder=lambda: _build_conv_f1(hebbian_batch_size=args.train_batch_size),
        train_images=full_train_images,
        train_labels=full_train_labels,
        test_images=full_test_images,
        test_labels=full_test_labels,
        planned_passes=args.full_passes,
        checkpoints=checkpoints,
        eval_batch_size=args.eval_batch_size,
        probe_batch_size=args.probe_batch_size,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        device=device,
        runtime_limit_hours=args.full_runtime_limit_hours,
        runtime_pass_cap=args.full_pass_cap,
    )

    full_pass_budget = exp2.executed_passes

    exp3_no_norm = exp2

    print("\nRunning experiment 3 (with divisive norm)...", flush=True)
    exp3_div_norm = _run_experiment(
        name="exp3-div-norm",
        model_builder=lambda: _build_normalized_conv_f1(
            use_divisive_norm=True,
            hebbian_batch_size=args.train_batch_size,
        ),
        train_images=full_train_images,
        train_labels=full_train_labels,
        test_images=full_test_images,
        test_labels=full_test_labels,
        planned_passes=full_pass_budget,
        checkpoints=checkpoints,
        eval_batch_size=args.eval_batch_size,
        probe_batch_size=args.probe_batch_size,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        device=device,
    )

    revised_ceiling = max(exp1.best_linear_probe, exp2.best_linear_probe, exp3_div_norm.best_linear_probe)
    _print_report(
        exp1=exp1,
        exp2=exp2,
        exp3_no_norm=exp3_no_norm,
        exp3_div_norm=exp3_div_norm,
        checkpoints=checkpoints,
        device=device,
    )

    decision_path = _write_decision(
        exp1=exp1,
        exp2=exp2,
        exp3_div=exp3_div_norm,
        revised_ceiling=revised_ceiling,
    )
    print(f"\nDecision written to {decision_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
