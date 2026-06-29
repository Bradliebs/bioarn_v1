"""Audit survivorship bias in the CIFAR-10 Hebbian feature-space ceiling claim."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import os
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
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - exercised in CLI usage
    raise RuntimeError("torchvision is required for bias_audit.py") from exc


SEED = 42
TRAIN_SAMPLES = 5_000
TEST_SAMPLES = 1_000
FALLBACK_TRAIN_SAMPLES = 3_000
NUM_CLASSES = 10
TRAIN_PASSES = 10
DURATION_PASSES = 50
HEBBIAN_LR = 0.005
HEBBIAN_BATCH_SIZE = 32
PROBE_EPOCHS = 100
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
SPATIAL_GRID = 4
BIAS_THRESHOLD_PP = 3.0
TAIL_RISE_THRESHOLD_PP = 0.5
DURATION_CHECKPOINTS = (1, 5, 10, 20, 30, 40, 50)
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
CAPACITY_CONFIGS = {
    "cap_64": dict(num_features=64, hidden_channels=(32, 64), top_k=32, competitive_k=8),
    "cap_128": dict(num_features=128, hidden_channels=(64, 128), top_k=64, competitive_k=16),
    "cap_256": dict(num_features=256, hidden_channels=(128, 256), top_k=128, competitive_k=32),
}


@dataclass(frozen=True)
class FeatureMetrics:
    nearest_centroid: float
    linear_probe: float
    mlp_probe: float | None = None


@dataclass(frozen=True)
class CapacityResult:
    features: int
    output_dim: int
    metrics: FeatureMetrics
    train_samples: int


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PureHebbianCNN(nn.Module):
    """Minimal Hebbian conv — no CCC, no Oja, no decorrelation, no contrast norm."""

    def __init__(self, num_features: int = 64) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, self.num_features, 3, padding=1)
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                with torch.no_grad():
                    conv.bias.zero_()
        for conv in (self.conv1, self.conv2, self.conv3):
            with torch.no_grad():
                flat = conv.weight.view(conv.weight.shape[0], -1)
                norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
                flat.div_(norms)

    @property
    def output_dim(self) -> int:
        return self.num_features * SPATIAL_GRID * SPATIAL_GRID

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.conv1(x))
        h1 = F.max_pool2d(h1, 2)
        h2 = F.relu(self.conv2(h1))
        h2 = F.max_pool2d(h2, 2)
        h3 = F.relu(self.conv3(h2))
        h3 = F.adaptive_avg_pool2d(h3, SPATIAL_GRID)
        return h3.flatten(1)

    @torch.no_grad()
    def hebbian_update(self, x: torch.Tensor, lr: float = HEBBIAN_LR) -> None:
        x = x.to(torch.float32)
        layers = [self.conv1, self.conv2, self.conv3]
        activations = [x]
        current = x
        for index, conv in enumerate(layers):
            current = F.relu(conv(current))
            activations.append(current)
            if index < 2:
                current = F.max_pool2d(current, 2)
                activations.append(current)

        input_maps = [activations[0], activations[2], activations[4]]
        output_maps = [activations[1], activations[3], activations[5]]
        for conv, pre, post in zip(layers, input_maps, output_maps, strict=True):
            patches = F.unfold(
                pre,
                kernel_size=conv.kernel_size,
                padding=conv.padding,
                stride=conv.stride,
            )
            post_flat = post.reshape(post.shape[0], post.shape[1], -1)
            channel_energy = post.mean(dim=(2, 3))
            k = max(1, post.shape[1] // 4)
            _, top_idx = torch.topk(channel_energy, k, dim=1)
            mask = torch.zeros_like(channel_energy)
            mask.scatter_(1, top_idx, 1.0)
            post_flat = post_flat * mask.unsqueeze(-1)
            spatial = max(post_flat.shape[-1], 1)
            delta = torch.einsum("bol,bil->oi", post_flat, patches) / spatial / max(post.shape[0], 1)
            conv.weight.add_(lr * delta.reshape_as(conv.weight))
            flat = conv.weight.view(conv.weight.shape[0], -1)
            flat.sub_(flat.mean(dim=1, keepdim=True))
            norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
            flat.div_(norms)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


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
        if all(counts[class_id] >= per_label for class_id in range(NUM_CLASSES)):
            break
    expected = per_label * NUM_CLASSES
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
    train_dataset, test_dataset = _load_cifar10_datasets(data_dir)
    return (
        _collect_balanced_samples(train_dataset, per_label=train_samples // NUM_CLASSES, seed=seed),
        _collect_balanced_samples(test_dataset, per_label=test_samples // NUM_CLASSES, seed=seed + 1),
    )


def _stack_samples(samples: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image for image, _ in samples], dim=0).to(torch.float32)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    return images, labels


def _balanced_prefix_subset(
    samples: list[tuple[torch.Tensor, int]],
    *,
    per_label: int,
) -> list[tuple[torch.Tensor, int]]:
    counts: defaultdict[int, int] = defaultdict(int)
    subset: list[tuple[torch.Tensor, int]] = []
    for image, label in samples:
        if counts[label] >= per_label:
            continue
        subset.append((image, label))
        counts[label] += 1
        if all(counts[class_id] >= per_label for class_id in range(NUM_CLASSES)):
            break
    expected = per_label * NUM_CLASSES
    if len(subset) != expected:
        raise RuntimeError(f"Unable to build balanced subset of {expected} samples.")
    return subset


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


def _iter_batches(
    values: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
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


def _build_ccc_layer(*, num_features: int, hidden_channels: tuple[int, ...], top_k: int, competitive_k: int) -> ConvF1Layer:
    return ConvF1Layer(
        in_channels=3,
        num_features=num_features,
        spatial_size=32,
        top_k=top_k,
        spatial_grid=SPATIAL_GRID,
        num_layers=3,
        hidden_channels=hidden_channels,
        kernel_sizes=(5, 3, 3),
        spatial_top_k=4,
        competitive_k=competitive_k,
        hebbian_lr=HEBBIAN_LR,
        hebbian_batch_size=HEBBIAN_BATCH_SIZE,
        weight_norm_target=1.0,
        enable_local_contrast_norm=True,
        contrast_kernel_size=5,
        response_norm_eps=1e-4,
        feature_pool_avg_mix=0.25,
        hebbian_oja_decay=0.05,
        filter_decorrelation=0.02,
        softhebb_enabled=False,
        softhebb_gamma=4.0,
        softhebb_beta=2.0,
        softhebb_theta_decay=0.99,
    )


def _run_ccc_pass(layer: ConvF1Layer, images: torch.Tensor, ordered_indices: torch.Tensor) -> None:
    for batch in _batch_indices(ordered_indices, batch_size=TRAIN_BATCH_SIZE):
        batch_images = images[batch].to(torch.float32)
        learning_signal = torch.ones(batch_images.shape[0], dtype=torch.float32)
        layer.hebbian_update(batch_images, learning_signal=learning_signal)
    layer.flush_hebbian_updates()


def _run_pure_pass(model: PureHebbianCNN, images: torch.Tensor, ordered_indices: torch.Tensor) -> None:
    for batch in _batch_indices(ordered_indices, batch_size=TRAIN_BATCH_SIZE):
        model.hebbian_update(images[batch].to(torch.float32), lr=HEBBIAN_LR)


@torch.inference_mode()
def _extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    feature_batches: list[torch.Tensor] = []
    for batch_images, _ in _iter_batches(images, None, batch_size=EVAL_BATCH_SIZE, shuffle=False, seed=0):
        feature_batches.append(model(batch_images.to(torch.float32)).cpu().to(torch.float32))
    return torch.cat(feature_batches, dim=0)


def _nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    train_norm = F.normalize(train_features, dim=1)
    test_norm = F.normalize(test_features, dim=1)
    prototypes: list[torch.Tensor] = []
    for label in range(NUM_CLASSES):
        class_features = train_norm[train_labels == label]
        if class_features.numel() == 0:
            raise RuntimeError(f"Missing class {label} in train features.")
        prototypes.append(F.normalize(class_features.mean(dim=0, keepdim=True), dim=1).squeeze(0))
    prototype_matrix = torch.stack(prototypes, dim=1)
    predictions = (test_norm @ prototype_matrix).argmax(dim=1)
    return float((predictions == test_labels).float().mean().item())


def _train_probe(
    probe: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    seed: int,
) -> float:
    _set_seed(seed)
    probe = probe.to(torch.float32)
    optimizer = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    train_features = train_features.to(torch.float32)
    test_features = test_features.to(torch.float32)

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        for batch_features, batch_labels in _iter_batches(
            train_features,
            train_labels,
            batch_size=PROBE_BATCH_SIZE,
            shuffle=True,
            seed=seed + epoch,
        ):
            assert batch_labels is not None
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels.to(torch.long))
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_features, batch_labels in _iter_batches(
            test_features,
            test_labels,
            batch_size=PROBE_BATCH_SIZE,
            shuffle=False,
            seed=0,
        ):
            assert batch_labels is not None
            predictions = probe(batch_features.to(torch.float32)).argmax(dim=1)
            correct += int((predictions == batch_labels).sum().item())
            total += int(batch_labels.numel())
    return correct / max(total, 1)


def _feature_metrics(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    include_mlp: bool,
    seed: int,
) -> FeatureMetrics:
    nearest_centroid = _nearest_centroid_accuracy(train_features, train_labels, test_features, test_labels)
    linear_probe = _train_probe(
        nn.Linear(train_features.shape[1], NUM_CLASSES),
        train_features,
        train_labels,
        test_features,
        test_labels,
        seed=seed,
    )
    mlp_probe = None
    if include_mlp:
        mlp_probe = _train_probe(
            MLPProbe(train_features.shape[1], hidden_dim=128, num_classes=NUM_CLASSES),
            train_features,
            train_labels,
            test_features,
            test_labels,
            seed=seed,
        )
    return FeatureMetrics(
        nearest_centroid=nearest_centroid,
        linear_probe=linear_probe,
        mlp_probe=mlp_probe,
    )


def _capacity_result_from_features(
    *,
    features: int,
    output_dim: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    include_mlp: bool,
    train_samples: int,
) -> CapacityResult:
    metrics = _feature_metrics(
        train_features,
        train_labels,
        test_features,
        test_labels,
        include_mlp=include_mlp,
        seed=SEED,
    )
    return CapacityResult(
        features=features,
        output_dim=output_dim,
        metrics=metrics,
        train_samples=train_samples,
    )


def _train_ccc_capacity(
    *,
    name: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    passes: int,
    result_pass: int | None = None,
    accuracy_checkpoints: set[int] | None = None,
    feature_checkpoints: set[int] | None = None,
    include_mlp: bool = False,
) -> tuple[CapacityResult | None, dict[int, float]]:
    config = CAPACITY_CONFIGS[name]
    _set_seed(SEED)
    layer = _build_ccc_layer(
        num_features=int(config["num_features"]),
        hidden_channels=tuple(config["hidden_channels"]),
        top_k=int(config["top_k"]),
        competitive_k=int(config["competitive_k"]),
    )
    class_indices = _class_index_buckets(train_labels)
    duration_accuracy: dict[int, float] = {}
    final_result: CapacityResult | None = None
    wanted_accuracy_checkpoints = accuracy_checkpoints or set()
    wanted_feature_checkpoints = feature_checkpoints or set()
    target_result_pass = passes if result_pass is None else int(result_pass)

    print(
        f"[train] {name} features={config['num_features']} output_dim={layer.output_dim} "
        f"train={train_images.shape[0]} passes={passes}"
    )
    for pass_index in range(passes):
        pass_num = pass_index + 1
        ordered_indices = _build_interleaved_indices(class_indices, seed=SEED + pass_index)
        pass_start = time.perf_counter()
        _run_ccc_pass(layer, train_images, ordered_indices)
        print(f"  pass {pass_num}/{passes} done in {time.perf_counter() - pass_start:.1f}s")
        needs_eval = (
            pass_num in wanted_accuracy_checkpoints
            or pass_num in wanted_feature_checkpoints
            or pass_num == target_result_pass
        )
        if needs_eval:
            train_features = _extract_features(layer, train_images)
            test_features = _extract_features(layer, test_images)
            nearest_centroid = _nearest_centroid_accuracy(
                train_features,
                train_labels,
                test_features,
                test_labels,
            )
            if pass_num in wanted_accuracy_checkpoints:
                duration_accuracy[pass_num] = nearest_centroid
            if pass_num == target_result_pass:
                final_result = _capacity_result_from_features(
                    features=int(config["num_features"]),
                    output_dim=layer.output_dim,
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    include_mlp=include_mlp,
                    train_samples=int(train_images.shape[0]),
                )
    return final_result, duration_accuracy


def _train_pure_capacity(
    *,
    num_features: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    passes: int,
    include_mlp: bool = False,
) -> CapacityResult:
    _set_seed(SEED)
    model = PureHebbianCNN(num_features=num_features)
    class_indices = _class_index_buckets(train_labels)
    print(
        f"[train] pure_hebbian_{num_features} features={num_features} output_dim={model.output_dim} "
        f"train={train_images.shape[0]} passes={passes}"
    )
    for pass_index in range(passes):
        ordered_indices = _build_interleaved_indices(class_indices, seed=SEED + pass_index)
        pass_start = time.perf_counter()
        _run_pure_pass(model, train_images, ordered_indices)
        print(f"  pass {pass_index + 1}/{passes} done in {time.perf_counter() - pass_start:.1f}s")
    train_features = _extract_features(model, train_images)
    test_features = _extract_features(model, test_images)
    return _capacity_result_from_features(
        features=num_features,
        output_dim=model.output_dim,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        include_mlp=include_mlp,
        train_samples=int(train_images.shape[0]),
    )


def _pp(delta: float) -> str:
    return f"{delta:+.1f} pp"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bias_message(detected: bool, message: str) -> str:
    prefix = "⚠️ BIAS DETECTED" if detected else "✅ No bias"
    return f"{prefix}: {message}"


def _write_decision(
    *,
    report_summary: str,
    key_finding: str,
    revised_conclusion: str,
) -> Path:
    decision_path = Path(".squad") / "decisions" / "inbox" / "tank-bias-audit-results.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        "\n".join(
            [
                "### 2026-06-28: Survivorship Bias Audit Results",
                "**By:** Tank (Core Neural)",
                f"**What:** {report_summary}",
                f"**Key finding:** {key_finding}",
                f"**Revised conclusion:** {revised_conclusion}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return decision_path


def main() -> int:
    start = time.perf_counter()
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    print("Loading real CIFAR-10 via torchvision...")
    train_samples, test_samples = _load_real_cifar10(
        data_dir=Path("data"),
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        seed=SEED,
    )
    train_images, train_labels = _stack_samples(train_samples)
    test_images, test_labels = _stack_samples(test_samples)
    print(
        f"Collected {train_images.shape[0]} balanced train samples and "
        f"{test_images.shape[0]} balanced test samples."
    )

    reduced_large_train = False
    large_train_images = train_images
    large_train_labels = train_labels

    cap64_result, duration_accuracy = _train_ccc_capacity(
        name="cap_64",
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=DURATION_PASSES,
        result_pass=10,
        accuracy_checkpoints=set(DURATION_CHECKPOINTS),
        include_mlp=True,
    )
    if cap64_result is None:
        raise RuntimeError("Missing cap_64 result at pass 10.")

    elapsed_after_64 = time.perf_counter() - start
    if elapsed_after_64 > 45 * 60:
        reduced_large_train = True
        reduced_train_samples = _balanced_prefix_subset(
            train_samples,
            per_label=FALLBACK_TRAIN_SAMPLES // NUM_CLASSES,
        )
        large_train_images, large_train_labels = _stack_samples(reduced_train_samples)
        print(
            f"Runtime safeguard triggered after {elapsed_after_64 / 60:.1f}m; "
            f"256-feature runs will use {FALLBACK_TRAIN_SAMPLES} train samples."
        )

    cap128_result, _ = _train_ccc_capacity(
        name="cap_128",
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=TRAIN_PASSES,
        result_pass=TRAIN_PASSES,
        include_mlp=False,
    )
    if cap128_result is None:
        raise RuntimeError("Missing cap_128 result.")

    cap256_result, _ = _train_ccc_capacity(
        name="cap_256",
        train_images=large_train_images,
        train_labels=large_train_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=TRAIN_PASSES,
        result_pass=TRAIN_PASSES,
        include_mlp=True,
    )
    if cap256_result is None:
        raise RuntimeError("Missing cap_256 result.")

    pure64_result = _train_pure_capacity(
        num_features=64,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=TRAIN_PASSES,
        include_mlp=False,
    )
    pure256_result = _train_pure_capacity(
        num_features=256,
        train_images=large_train_images,
        train_labels=large_train_labels,
        test_images=test_images,
        test_labels=test_labels,
        passes=TRAIN_PASSES,
        include_mlp=False,
    )

    capacity_results = [cap64_result, cap128_result, cap256_result]

    capacity_delta_pp = (cap256_result.metrics.linear_probe - cap64_result.metrics.linear_probe) * 100.0
    capacity_bias = capacity_delta_pp >= BIAS_THRESHOLD_PP
    duration_gain_pp = (duration_accuracy[50] - duration_accuracy[10]) * 100.0
    still_rising = (duration_accuracy[50] - duration_accuracy[40]) * 100.0 > TAIL_RISE_THRESHOLD_PP
    duration_bias = duration_gain_pp >= BIAS_THRESHOLD_PP
    eval_delta_pp = ((cap64_result.metrics.mlp_probe or 0.0) - cap64_result.metrics.nearest_centroid) * 100.0
    eval_delta_256_pp = ((cap256_result.metrics.mlp_probe or 0.0) - cap256_result.metrics.nearest_centroid) * 100.0
    eval_bias = max(eval_delta_pp, eval_delta_256_pp) >= BIAS_THRESHOLD_PP
    framework_delta_pp = (pure64_result.metrics.linear_probe - cap64_result.metrics.linear_probe) * 100.0
    framework_delta_256_pp = (pure256_result.metrics.linear_probe - cap256_result.metrics.linear_probe) * 100.0
    framework_bias = max(framework_delta_pp, framework_delta_256_pp) >= BIAS_THRESHOLD_PP
    biases_found = sum((capacity_bias, duration_bias, eval_bias, framework_bias))

    overall_candidates = [
        ("CCC 64 linear probe", cap64_result.metrics.linear_probe),
        ("CCC 64 MLP probe", cap64_result.metrics.mlp_probe or 0.0),
        ("CCC 128 linear probe", cap128_result.metrics.linear_probe),
        ("CCC 256 linear probe", cap256_result.metrics.linear_probe),
        ("CCC 256 MLP probe", cap256_result.metrics.mlp_probe or 0.0),
        ("Pure Hebbian 64 linear probe", pure64_result.metrics.linear_probe),
        ("Pure Hebbian 256 linear probe", pure256_result.metrics.linear_probe),
        ("Duration pass 50 nearest-centroid", duration_accuracy[50]),
    ]
    best_label, best_value = max(overall_candidates, key=lambda item: item[1])

    if biases_found == 0:
        conclusion = "the original ceiling claim holds up in this audit slice."
    else:
        conclusion = f"the prior ceiling claim was incomplete; this audit reached {_pct(best_value)} via {best_label}."
    if reduced_large_train:
        conclusion += f" Large 256-feature runs used {FALLBACK_TRAIN_SAMPLES} train samples to stay within runtime budget."

    summary_lines = [
        "================================================================",
        "SURVIVORSHIP BIAS AUDIT — CIFAR-10",
        "================================================================",
        "",
        "--- TEST 1: CAPACITY CHECK (64 → 128 → 256 features) ---",
        "| Features | Output Dim | Nearest-Centroid | Linear Probe |",
        "|----------|------------|------------------|--------------|",
    ]
    for result in capacity_results:
        summary_lines.append(
            f"| {result.features:<8} | {result.output_dim:<10} | {_pct(result.metrics.nearest_centroid):<16} | {_pct(result.metrics.linear_probe):<12} |"
        )
    summary_lines.extend(
        [
            "",
            f"Capacity effect: Δ(256 vs 64, linear probe) = {_pp(capacity_delta_pp)}",
            _bias_message(
                capacity_bias,
                (
                    "feature count materially changed the ceiling."
                    if capacity_bias
                    else "larger CCC capacity did not move linear-probe accuracy by 3pp."
                ),
            ),
            "",
            "--- TEST 2: TRAINING DURATION (50 passes, 64 features) ---",
            "| Pass | Accuracy |",
            "|------|----------|",
        ]
    )
    for checkpoint in DURATION_CHECKPOINTS:
        summary_lines.append(f"| {checkpoint:<4} | {_pct(duration_accuracy[checkpoint]):<8} |")
    summary_lines.extend(
        [
            "",
            f"Still rising at pass 50? {'Yes' if still_rising else 'No'}",
            _bias_message(
                duration_bias,
                (
                    f"nearest-centroid improved {_pp(duration_gain_pp)} from pass 10 to pass 50."
                    if duration_bias
                    else "extra training did not add a 3pp gain over the 10-pass baseline."
                ),
            ),
            "",
            "--- TEST 3: EVALUATION METHOD (NC vs LP vs MLP) ---",
            "| Features | Nearest-Centroid | Linear Probe | MLP Probe |",
            "|----------|------------------|--------------|-----------|",
            f"| 64       | {_pct(cap64_result.metrics.nearest_centroid):<16} | {_pct(cap64_result.metrics.linear_probe):<12} | {_pct(cap64_result.metrics.mlp_probe or 0.0):<9} |",
            f"| 256      | {_pct(cap256_result.metrics.nearest_centroid):<16} | {_pct(cap256_result.metrics.linear_probe):<12} | {_pct(cap256_result.metrics.mlp_probe or 0.0):<9} |",
            "",
            f"Eval effect: Δ(MLP vs NC, 64 feat) = {_pp(eval_delta_pp)}",
            _bias_message(
                eval_bias,
                (
                    "a stronger probe extracted materially more class signal."
                    if eval_bias
                    else "MLP probes did not add a 3pp gain over nearest-centroid."
                ),
            ),
            "",
            "--- TEST 4: PURE HEBBIAN vs CCC (no framework overhead) ---",
            "| Model         | Features | Nearest-Centroid | Linear Probe |",
            "|---------------|----------|------------------|--------------|",
            f"| CCC           | 64       | {_pct(cap64_result.metrics.nearest_centroid):<16} | {_pct(cap64_result.metrics.linear_probe):<12} |",
            f"| Pure Hebbian  | 64       | {_pct(pure64_result.metrics.nearest_centroid):<16} | {_pct(pure64_result.metrics.linear_probe):<12} |",
            f"| CCC           | 256      | {_pct(cap256_result.metrics.nearest_centroid):<16} | {_pct(cap256_result.metrics.linear_probe):<12} |",
            f"| Pure Hebbian  | 256      | {_pct(pure256_result.metrics.nearest_centroid):<16} | {_pct(pure256_result.metrics.linear_probe):<12} |",
            "",
            f"Framework effect: Δ(Pure vs CCC, 64 feat, linear probe) = {_pp(framework_delta_pp)}",
            _bias_message(
                framework_bias,
                (
                    "the stripped-down Hebbian CNN outperformed CCC by at least 3pp."
                    if framework_bias
                    else "removing CCC machinery did not buy a 3pp linear-probe gain."
                ),
            ),
            "",
            "================================================================",
            "SUMMARY",
            "================================================================",
            f"Biases found: {biases_found} / 4",
            f"Revised ceiling estimate: {_pct(best_value)} ({best_label})",
            f"Conclusion: {conclusion}",
        ]
    )
    report = "\n".join(summary_lines)
    print()
    print(report)

    key_finding = (
        f"Best audit result was {_pct(best_value)} from {best_label}; "
        f"capacity Δ256vs64={capacity_delta_pp:+.1f}pp, duration Δ50vs10={duration_gain_pp:+.1f}pp."
    )
    decision_path = _write_decision(
        report_summary=(
            "Ran four single-seed CIFAR-10 survivorship-bias checks on the Hebbian feature ceiling: "
            "capacity, training duration, evaluation method, and pure-Hebbian-vs-CCC."
        ),
        key_finding=key_finding,
        revised_conclusion=f"Biases found: {biases_found}/4; {conclusion}",
    )
    print(f"\nDecision written to {decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
