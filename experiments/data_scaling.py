"""Data-scale Hebbian experiments: multi-dataset + augmentation on CIFAR-10.

Four experiments that test whether more data improves Hebbian feature learning.
The Hebbian layer is fully unsupervised — labels are never used during Hebbian
training, so image data from any natural-image source is valid.  The linear
probe at evaluation time uses only CIFAR-10 train/test labels.

Free datasets used (auto-downloaded via torchvision):
  • CIFAR-10  — 50K images 32×32  (eval labels always from here)
  • CIFAR-100 — 50K images 32×32  (unlabeled for Hebbian training)
  • SVHN train — ~73K images 32×32 (street-view house numbers)

Experiments:
  Exp 1 (aug-c10):      CIFAR-10 50K + strong online aug,             512 feat, 50 passes
  Exp 2 (multi-100k):   CIFAR-10+CIFAR-100 100K, no aug,              512 feat, 30 passes
  Exp 3 (multi-173k):   CIFAR-10+CIFAR-100+SVHN ~173K, no aug,        512 feat, 25 passes
  Exp 4 (multi-173k-aug): same ~173K + strong online aug,             512 feat, 20 passes
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
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
except ImportError as exc:
    raise RuntimeError("torchvision is required for data_scaling.py") from exc

# ─── Architecture constants ────────────────────────────────────────────────────
SEED = 42
NUM_CLASSES = 10
NUM_FEATURES = 512
HIDDEN_CHANNELS = (256, 512)  # scaled with features
KERNEL_SIZES = (5, 3, 3)
SPATIAL_GRID = 4
SPATIAL_TOP_K = 8
TOP_K = 256          # 50% sparsity at 512 features
COMPETITIVE_K = 64   # ~25% of TOP_K
HEBBIAN_LR = 0.005
HEBBIAN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
PROBE_EPOCHS = 100
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9

# Experiment pass counts
PASSES_AUG = 50       # Exp 1: aug CIFAR-10
PASSES_MULTI = 30     # Exp 2: multi-dataset no aug
PASSES_MULTI_AUG = 25 # Exp 3: multi-dataset + aug
PASSES_MAX_AUG = 20   # Exp 4: max data + aug

CHECKPOINTS = (1, 5, 10, 20, 25, 30, 40, 50)
RUNTIME_LIMIT_HOURS = 4.0   # abort individual experiment if projected > 4 hr
RUNTIME_PASS_CAP = 20       # minimum cap

# Previous ceiling (from hebbian_scaling.py) for delta reporting
PREVIOUS_CEILING = 0.376

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ─── Dataclasses ──────────────────────────────────────────────────────────────

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
    hebbian_samples: int
    eval_train_samples: int
    planned_passes: int
    executed_passes: int
    pass_metrics: list[PassMetrics]
    notes: list[str] = field(default_factory=list)

    @property
    def best_linear_probe(self) -> float:
        return max((m.linear_probe for m in self.pass_metrics), default=0.0)

    @property
    def best_nearest_centroid(self) -> float:
        return max((m.nearest_centroid for m in self.pass_metrics), default=0.0)

    @property
    def best_pass(self) -> int:
        if not self.pass_metrics:
            return 0
        return max(self.pass_metrics, key=lambda m: m.linear_probe).pass_num

    @property
    def metrics_by_pass(self) -> dict[int, PassMetrics]:
        return {m.pass_num: m for m in self.pass_metrics}


# ─── Model ────────────────────────────────────────────────────────────────────

class SparseLinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(self.feature_dim, self.num_classes))
        self.bias = nn.Parameter(torch.empty(self.num_classes))
        bound = 1.0 / math.sqrt(max(self.feature_dim, 1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(indices, self.weight)
        return (selected * values.unsqueeze(-1)).sum(dim=1) + self.bias


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


# ─── Utilities ────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pp"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_cifar10_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (train_images, train_labels, test_images, test_labels) as float32 [0,1]."""
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    if tuple(train_ds.classes) != CLASS_NAMES:
        raise RuntimeError("Unexpected CIFAR-10 class order from torchvision.")
    train_images = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
    test_images = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
    train_labels = torch.tensor(train_ds.targets, dtype=torch.long)
    test_labels = torch.tensor(test_ds.targets, dtype=torch.long)
    return train_images, train_labels, test_images, test_labels


def _load_cifar100_images(data_dir: Path) -> torch.Tensor:
    """CIFAR-100 train images as float32 [0,1], shape [50000, 3, 32, 32]."""
    ds = datasets.CIFAR100(root=str(data_dir), train=True, download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0


def _load_stl10_unlabeled(data_dir: Path) -> torch.Tensor:
    """STL-10 unlabeled images resized to 32×32, shape [100000, 3, 32, 32] float32 [0,1].

    Raw data is 96×96; we resize in chunks of 2000 to avoid a large float32
    intermediate (which would be ~11 GB if done all at once).
    """
    ds = datasets.STL10(root=str(data_dir), split="unlabeled", download=True)
    raw = torch.from_numpy(ds.data)  # [100000, 3, 96, 96] uint8
    n = raw.shape[0]
    result = torch.empty((n, 3, 32, 32), dtype=torch.float32)
    chunk = 2000
    print(f"  Resizing {n // 1000}K STL-10 images from 96×96 → 32×32 ...", flush=True)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_f32 = raw[start:end].to(torch.float32) / 255.0
        result[start:end] = F.interpolate(chunk_f32, size=(32, 32), mode="bilinear", align_corners=False)
    return result


def _load_svhn_images(data_dir: Path) -> torch.Tensor:
    """SVHN train split as float32 [0,1], shape [N, 3, 32, 32].

    Only the `train` split (~73K) is loaded to avoid the 530K `extra` split
    which would use ~6 GB RAM.  Pass ``--include-svhn-extra`` at the CLI to
    also include the extra split (adds ~6.3 GB of RAM and download traffic).
    """
    ds = datasets.SVHN(root=str(data_dir), split="train", download=True)
    # SVHN .data is already (N, 3, 32, 32) uint8
    return torch.from_numpy(ds.data).to(torch.float32) / 255.0


def _load_svhn_extra_images(data_dir: Path) -> torch.Tensor:
    """SVHN extra split (~531K). Only used when --include-svhn-extra is set."""
    ds = datasets.SVHN(root=str(data_dir), split="extra", download=True)
    return torch.from_numpy(ds.data).to(torch.float32) / 255.0


# ─── Augmentation ─────────────────────────────────────────────────────────────

def _augment_batch(batch: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Online augmentation on device. batch: [B, 3, 32, 32] float32 [0,1].

    Applies (per batch, same transform for all items in batch):
      • Random horizontal flip
      • Random crop  (pad 4, then crop back to 32×32)
      • Mild brightness jitter
      • Mild contrast jitter

    Using a single crop/flip per batch is much faster than per-sample and still
    provides substantial diversity because each pass sees a different augmentation.
    """
    # Random horizontal flip
    if rng.random() > 0.5:
        batch = batch.flip(-1)

    # Pad then random crop
    pad = 4
    padded = F.pad(batch, [pad, pad, pad, pad], mode="reflect")  # → [B, 3, 40, 40]
    i = rng.randint(0, 2 * pad)  # row offset
    j = rng.randint(0, 2 * pad)  # col offset
    batch = padded[:, :, i : i + 32, j : j + 32]

    # Brightness jitter: scale pixel values
    brightness = rng.uniform(0.7, 1.3)
    batch = torch.clamp(batch * brightness, 0.0, 1.0)

    # Contrast jitter: shift toward/away from per-channel mean
    contrast = rng.uniform(0.8, 1.2)
    mean = batch.mean(dim=(-1, -2), keepdim=True)  # [B, 3, 1, 1]
    batch = torch.clamp((batch - mean) * contrast + mean, 0.0, 1.0)

    return batch


# ─── Index helpers ────────────────────────────────────────────────────────────

def _shuffle_indices(n: int, seed: int) -> torch.Tensor:
    """Random permutation of [0, n)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)


def _batch_indices(indices: torch.Tensor, *, batch_size: int):
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _iter_label_batches(labels: torch.Tensor, *, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(labels.shape[0], generator=g)
    else:
        indices = torch.arange(labels.shape[0])
    for start in range(0, labels.shape[0], batch_size):
        yield indices[start : start + batch_size]


# ─── Hebbian pass ─────────────────────────────────────────────────────────────

def _run_hebbian_pass(
    model: ConvF1Layer,
    images: torch.Tensor,
    ordered_indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    augment: bool = False,
    rng: random.Random | None = None,
) -> None:
    for batch_indices in _batch_indices(ordered_indices, batch_size=batch_size):
        batch = images[batch_indices].to(device=device, dtype=torch.float32)
        if augment and rng is not None:
            batch = _augment_batch(batch, rng)
        learning_signal = torch.ones(batch.shape[0], device=device, dtype=torch.float32)
        model.hebbian_update(batch, learning_signal=learning_signal)
    model.flush_hebbian_updates()


# ─── Feature extraction & evaluation ──────────────────────────────────────────

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
    for batch_idxs in _iter_label_batches(labels, batch_size=batch_size, shuffle=False, seed=0):
        batch_imgs = images[batch_idxs].to(device=device, dtype=torch.float32)
        dense = model(batch_imgs).cpu().to(torch.float32)
        top_vals, top_idxs = torch.topk(dense, k=top_k, dim=1)
        bs = dense.shape[0]
        all_indices[cursor : cursor + bs] = top_idxs.to(torch.int32)
        all_values[cursor : cursor + bs] = top_vals
        cursor += bs
    return SparseFeatureSet(
        indices=all_indices,
        values=all_values,
        labels=labels.clone(),
        feature_dim=int(feature_dim),
    )


def _iter_sparse_batches(features: SparseFeatureSet, *, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(features.labels.shape[0], generator=g)
    else:
        order = torch.arange(features.labels.shape[0])
    for start in range(0, order.shape[0], batch_size):
        batch = order[start : start + batch_size]
        yield features.indices[batch], features.values[batch], features.labels[batch]


def _nearest_centroid_accuracy(train_features: SparseFeatureSet, test_features: SparseFeatureSet) -> float:
    prototype_sums = torch.zeros((NUM_CLASSES, train_features.feature_dim), dtype=torch.float32)
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for idxs, vals, lbls in _iter_sparse_batches(train_features, batch_size=PROBE_BATCH_SIZE, shuffle=False, seed=0):
        label_rows = lbls.unsqueeze(1).expand_as(idxs).reshape(-1).to(torch.long)
        feature_cols = idxs.reshape(-1).to(torch.long)
        prototype_sums.index_put_((label_rows, feature_cols), vals.reshape(-1), accumulate=True)
        class_counts += torch.bincount(lbls, minlength=NUM_CLASSES).to(torch.float32)
    if bool((class_counts == 0).any().item()):
        missing = torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Missing classes in train features: {missing}")
    prototypes = F.normalize(prototype_sums / class_counts.unsqueeze(1), dim=1)
    proto_lookup = prototypes.transpose(0, 1).contiguous()
    correct = total = 0
    for idxs, vals, lbls in _iter_sparse_batches(test_features, batch_size=PROBE_BATCH_SIZE, shuffle=False, seed=0):
        selected = F.embedding(idxs.to(torch.long), proto_lookup)
        logits = (selected * vals.unsqueeze(-1)).sum(dim=1)
        correct += int((logits.argmax(dim=1) == lbls).sum().item())
        total += int(lbls.numel())
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
        for idxs, vals, lbls in _iter_sparse_batches(train_features, batch_size=probe_batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = probe(idxs.to(device=device, dtype=torch.long), vals.to(device=device, dtype=torch.float32))
            criterion(logits, lbls.to(device=device, dtype=torch.long)).backward()
            optimizer.step()
    probe.eval()
    correct = total = 0
    with torch.inference_mode():
        for idxs, vals, lbls in _iter_sparse_batches(test_features, batch_size=probe_batch_size, shuffle=False, seed=0):
            logits = probe(idxs.to(device=device, dtype=torch.long), vals.to(device=device, dtype=torch.float32))
            correct += int((logits.argmax(dim=1) == lbls.to(device)).sum().item())
            total += int(lbls.numel())
    return correct / max(total, 1)


def _evaluate_checkpoint(
    model: nn.Module,
    eval_train_images: torch.Tensor,
    eval_train_labels: torch.Tensor,
    eval_test_images: torch.Tensor,
    eval_test_labels: torch.Tensor,
    *,
    eval_batch_size: int,
    probe_batch_size: int,
    probe_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    train_feats = _extract_sparse_features(
        model, eval_train_images, eval_train_labels,
        batch_size=eval_batch_size, sparse_k=TOP_K, device=device,
    )
    test_feats = _extract_sparse_features(
        model, eval_test_images, eval_test_labels,
        batch_size=eval_batch_size, sparse_k=TOP_K, device=device,
    )
    nc = _nearest_centroid_accuracy(train_feats, test_feats)
    lp = _linear_probe_accuracy(
        train_feats, test_feats,
        seed=seed, probe_epochs=probe_epochs, probe_batch_size=probe_batch_size, device=device,
    )
    return nc, lp


# ─── Runtime cap ──────────────────────────────────────────────────────────────

def _maybe_apply_runtime_cap(
    *,
    pass_times: list[float],
    planned_passes: int,
    current_limit_hours: float,
    pass_cap: int,
    min_passes: int = 5,
) -> tuple[int | None, str | None]:
    if planned_passes <= pass_cap or len(pass_times) < 3 or current_limit_hours <= 0:
        return None, None
    avg_s = sum(pass_times) / len(pass_times)
    projected_h = avg_s * planned_passes / 3600.0
    if projected_h <= current_limit_hours:
        return None, None
    limit_s = current_limit_hours * 3600.0
    max_by_limit = int(limit_s // max(avg_s, 1e-6))
    adjusted = min(planned_passes, pass_cap, max(min_passes, max_by_limit))
    return (
        adjusted,
        f"Projected runtime {projected_h:.1f}h > {current_limit_hours:.1f}h limit; "
        f"capped to {adjusted} passes (avg pass {avg_s:.1f}s).",
    )


# ─── Experiment runner ────────────────────────────────────────────────────────

def _run_experiment(
    *,
    name: str,
    model_builder,
    # Hebbian training data — no labels needed, can be multi-dataset
    hebbian_images: torch.Tensor,
    # CIFAR-10 eval data — used only for checkpoint evaluation
    eval_train_images: torch.Tensor,
    eval_train_labels: torch.Tensor,
    eval_test_images: torch.Tensor,
    eval_test_labels: torch.Tensor,
    planned_passes: int,
    checkpoints: tuple[int, ...],
    eval_batch_size: int,
    probe_batch_size: int,
    probe_epochs: int,
    seed: int,
    train_batch_size: int,
    device: torch.device,
    use_augmentation: bool = False,
    runtime_limit_hours: float = 0.0,
    runtime_pass_cap: int | None = None,
) -> ExperimentResult:
    _set_seed(seed)
    model = model_builder().to(device)
    rng = random.Random(seed) if use_augmentation else None
    n_hebbian = int(hebbian_images.shape[0])
    metrics: list[PassMetrics] = []
    notes: list[str] = []
    pass_times: list[float] = []
    pass_limit = int(planned_passes)

    aug_label = "+aug" if use_augmentation else ""
    print(
        f"[{name}] hebbian={n_hebbian // 1000}K{aug_label} "
        f"eval_train={eval_train_images.shape[0] // 1000}K "
        f"passes={planned_passes} features={NUM_FEATURES} dim={model.output_dim}",
        flush=True,
    )

    for pass_index in range(pass_limit):
        pass_num = pass_index + 1
        shuffled = _shuffle_indices(n_hebbian, seed=seed + pass_index)
        t0 = time.perf_counter()
        _run_hebbian_pass(
            model, hebbian_images, shuffled,
            batch_size=train_batch_size, device=device,
            augment=use_augmentation, rng=rng,
        )
        pass_time = time.perf_counter() - t0
        pass_times.append(pass_time)
        print(f"[{name}] pass {pass_num}/{pass_limit} hebbian {pass_time:.1f}s", flush=True)

        # Runtime cap check after 3 passes
        if runtime_pass_cap is not None:
            adjusted, cap_note = _maybe_apply_runtime_cap(
                pass_times=pass_times,
                planned_passes=planned_passes,
                current_limit_hours=runtime_limit_hours,
                pass_cap=runtime_pass_cap,
            )
            if cap_note and adjusted is not None and pass_limit != adjusted:
                pass_limit = int(adjusted)
                notes.append(cap_note)
                print(f"[{name}] NOTE: {cap_note}", flush=True)

        if pass_num in checkpoints or pass_num == pass_limit:
            t_eval = time.perf_counter()
            nc, lp = _evaluate_checkpoint(
                model,
                eval_train_images, eval_train_labels,
                eval_test_images, eval_test_labels,
                eval_batch_size=eval_batch_size,
                probe_batch_size=probe_batch_size,
                probe_epochs=probe_epochs,
                seed=seed,
                device=device,
            )
            metrics.append(PassMetrics(pass_num=pass_num, nearest_centroid=nc, linear_probe=lp))
            print(
                f"[{name}] eval pass {pass_num}: nc={nc * 100:.2f}% lp={lp * 100:.2f}% "
                f"eval={(time.perf_counter() - t_eval) / 60:.1f}m",
                flush=True,
            )

        if pass_num >= pass_limit:
            break

    return ExperimentResult(
        name=name,
        hebbian_samples=n_hebbian,
        eval_train_samples=int(eval_train_images.shape[0]),
        planned_passes=int(planned_passes),
        executed_passes=int(pass_limit),
        pass_metrics=metrics,
        notes=notes,
    )


# ─── Reporting ────────────────────────────────────────────────────────────────

def _format_experiment_table(result: ExperimentResult, *, checkpoints: tuple[int, ...]) -> str:
    display_passes = sorted({*checkpoints, result.executed_passes})
    by_pass = result.metrics_by_pass
    lines = [
        "| Pass | Nearest-Centroid | Linear Probe |",
        "|------|------------------|--------------|",
    ]
    for cp in display_passes:
        m = by_pass.get(cp)
        lines.append(
            f"| {cp:<4} | {_pct(None if m is None else m.nearest_centroid):<16} | "
            f"{_pct(None if m is None else m.linear_probe):<12} |"
        )
    return "\n".join(lines)


def _print_report(
    *,
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    exp3: ExperimentResult,
    exp4: ExperimentResult,
    checkpoints: tuple[int, ...],
    device: torch.device,
) -> None:
    print("\n================================================================", flush=True)
    print("DATA SCALING — RESULTS", flush=True)
    print("================================================================", flush=True)
    print(f"Seed: {SEED} | Device: {device} | Features: {NUM_FEATURES} | TOP_K: {TOP_K}", flush=True)

    for exp, label in [
        (exp1, "Exp 1: aug-c10      (50K + aug)"),
        (exp2, "Exp 2: multi-100k   (~100K, no aug)"),
        (exp3, "Exp 3: multi-173k   (~173K, no aug)"),
        (exp4, "Exp 4: multi-173k-aug (~173K + aug)"),
    ]:
        print(f"\n--- {label} ---", flush=True)
        print(_format_experiment_table(exp, checkpoints=checkpoints), flush=True)
        print(f"Best: {_pct(exp.best_linear_probe)} LP (pass {exp.best_pass})", flush=True)
        for note in exp.notes:
            print(f"Note: {note}", flush=True)

    best_overall = max(
        exp1.best_linear_probe,
        exp2.best_linear_probe,
        exp3.best_linear_probe,
        exp4.best_linear_probe,
    )

    print("\n================================================================", flush=True)
    print("DATA SCALING SUMMARY", flush=True)
    print("================================================================", flush=True)
    print("| Experiment | Data | Best LP | Δ vs prev ceiling |", flush=True)
    print("|-----------|------|---------|-------------------|", flush=True)
    print(f"| Previous ceiling | 256 feat, 50K | {_pct(PREVIOUS_CEILING)} | — |", flush=True)
    for exp, label, note in [
        (exp1, f"Exp 1: 512 feat, {exp1.hebbian_samples // 1000}K aug", "aug CIFAR-10"),
        (exp2, f"Exp 2: 512 feat, ~{exp2.hebbian_samples // 1000}K no aug", "C10+C100"),
        (exp3, f"Exp 3: 512 feat, ~{exp3.hebbian_samples // 1000}K no aug", "C10+C100+SVHN"),
        (exp4, f"Exp 4: 512 feat, ~{exp4.hebbian_samples // 1000}K aug", "C10+C100+SVHN+aug"),
    ]:
        delta = exp.best_linear_probe - PREVIOUS_CEILING
        print(f"| {note} | {label} | {_pct(exp.best_linear_probe)} | {_pp(delta)} |", flush=True)
    print(f"\nNew ceiling: {_pct(best_overall)}", flush=True)
    print(f"Total improvement over original ~20% baseline: {_pp(best_overall - 0.20)}", flush=True)


def _write_decision(
    *,
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    exp3: ExperimentResult,
    exp4: ExperimentResult,
) -> Path:
    best = max(exp1.best_linear_probe, exp2.best_linear_probe,
               exp3.best_linear_probe, exp4.best_linear_probe)
    aug_gain = exp1.best_linear_probe - PREVIOUS_CEILING
    data_gain = exp2.best_linear_probe - PREVIOUS_CEILING
    combined_gain = exp3.best_linear_probe - PREVIOUS_CEILING
    max_gain = exp4.best_linear_probe - PREVIOUS_CEILING

    ts = time.strftime("%Y-%m-%d")
    path = Path(".squad") / "decisions" / "inbox" / "tank-data-scaling.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"### {ts}: Data Scaling Results",
            "**By:** Tank (Core Neural)",
            f"**What:** Multi-dataset Hebbian training reached {best * 100:.1f}% LP on CIFAR-10",
            f"**Previous ceiling:** {PREVIOUS_CEILING * 100:.1f}% (256 feat, 50K images)",
            f"**Exp 1 (aug only):** {_pct(exp1.best_linear_probe)} ({_pp(aug_gain)} vs prev)",
            f"**Exp 2 (100K no aug, C10+C100):** {_pct(exp2.best_linear_probe)} ({_pp(data_gain)} vs prev)",
            f"**Exp 3 (173K no aug, C10+C100+SVHN):** {_pct(exp3.best_linear_probe)} ({_pp(combined_gain)} vs prev)",
            f"**Exp 4 (173K + aug, C10+C100+SVHN):** {_pct(exp4.best_linear_probe)} ({_pp(max_gain)} vs prev)",
            f"**New ceiling:** {best * 100:.1f}%",
            "",
        ]),
        encoding="utf-8",
    )
    return path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Torchvision dataset root.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--probe-epochs", type=int, default=PROBE_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=HEBBIAN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--probe-batch-size", type=int, default=PROBE_BATCH_SIZE)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--passes-aug", type=int, default=PASSES_AUG,
                        help="Exp 1 passes (aug CIFAR-10).")
    parser.add_argument("--passes-multi", type=int, default=PASSES_MULTI,
                        help="Exp 2 passes (multi-dataset, no aug).")
    parser.add_argument("--passes-multi-aug", type=int, default=PASSES_MULTI_AUG,
                        help="Exp 3 passes (multi-dataset + aug).")
    parser.add_argument("--passes-max-aug", type=int, default=PASSES_MAX_AUG,
                        help="Exp 4 passes (all datasets + aug).")
    parser.add_argument("--runtime-limit-hours", type=float, default=RUNTIME_LIMIT_HOURS,
                        help="Abort individual experiment if projected > N hours.")
    parser.add_argument("--runtime-pass-cap", type=int, default=RUNTIME_PASS_CAP,
                        help="Minimum passes to run when runtime cap triggers.")
    parser.add_argument("--include-svhn-extra", action="store_true",
                        help="Also download SVHN extra split (~531K extra images, ~6 GB).")
    parser.add_argument("--threads", type=int,
                        default=min(8, max(1, os.cpu_count() or 1)))
    # Skip flags for iterating on individual experiments
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--skip-exp3", action="store_true")
    parser.add_argument("--skip-exp4", action="store_true")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    print(f"Threads: {torch.get_num_threads()}", flush=True)
    print(f"Features: {NUM_FEATURES} | TOP_K: {TOP_K} | COMPETITIVE_K: {COMPETITIVE_K}", flush=True)

    checkpoints = tuple(sorted({int(c) for c in args.checkpoints if int(c) > 0}))

    # ── Load CIFAR-10 (always needed, fast — already cached) ─────────────────
    print("\nLoading CIFAR-10 ...", flush=True)
    t_load = time.perf_counter()
    c10_train_imgs, c10_train_lbls, c10_test_imgs, c10_test_lbls = _load_cifar10_tensors(args.data_dir)
    print(f"  {c10_train_imgs.shape[0] // 1000}K train, {c10_test_imgs.shape[0] // 1000}K test — {time.perf_counter() - t_load:.1f}s\n", flush=True)

    common_kwargs = dict(
        eval_train_images=c10_train_imgs,
        eval_train_labels=c10_train_lbls,
        eval_test_images=c10_test_imgs,
        eval_test_labels=c10_test_lbls,
        checkpoints=checkpoints,
        eval_batch_size=args.eval_batch_size,
        probe_batch_size=args.probe_batch_size,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        device=device,
        runtime_limit_hours=args.runtime_limit_hours,
        runtime_pass_cap=args.runtime_pass_cap,
    )
    builder = lambda: _build_conv_f1(hebbian_batch_size=args.train_batch_size)

    # ── Experiment 1: augmented CIFAR-10 (starts immediately, no new downloads) ──
    if not args.skip_exp1:
        print(f"=== EXPERIMENT 1: augmented CIFAR-10 ({c10_train_imgs.shape[0] // 1000}K + aug) ===", flush=True)
        exp1 = _run_experiment(
            name="exp1-aug-c10",
            model_builder=builder,
            hebbian_images=c10_train_imgs,
            planned_passes=args.passes_aug,
            use_augmentation=True,
            **common_kwargs,
        )
    else:
        print("(Skipping Exp 1)", flush=True)
        exp1 = ExperimentResult("exp1-skip", 0, 0, 0, 0, [])

    # ── Download/load CIFAR-100 + SVHN before Exp 2/3/4 ────────────────────────
    need_multi = not (args.skip_exp2 and args.skip_exp3 and args.skip_exp4)
    if need_multi:
        print("\nLoading multi-dataset sources (may download if not cached) ...", flush=True)
        print("  CIFAR-100 (~169 MB) ...", flush=True)
        c100_imgs = _load_cifar100_images(args.data_dir)
        print(f"  CIFAR-100: {c100_imgs.shape[0] // 1000}K unlabeled images", flush=True)
        combined_100k = torch.cat([c10_train_imgs, c100_imgs], dim=0)
        print(f"  combined_100k: {combined_100k.shape[0] // 1000}K images", flush=True)

        print("  SVHN train (~174 MB) ...", flush=True)
        svhn_imgs = _load_svhn_images(args.data_dir)
        if args.include_svhn_extra:
            print("  Loading SVHN extra (~531K images, ~6 GB) ...", flush=True)
            svhn_extra = _load_svhn_extra_images(args.data_dir)
            svhn_imgs = torch.cat([svhn_imgs, svhn_extra], dim=0)
        combined_173k = torch.cat([c10_train_imgs, c100_imgs, svhn_imgs], dim=0)
        print(f"  combined_173k: {combined_173k.shape[0] // 1000}K images (C10+C100+SVHN)\n", flush=True)
    else:
        combined_100k = torch.empty(0, 3, 32, 32)
        combined_173k = torch.empty(0, 3, 32, 32)

    # ── Experiment 2: C10+C100 100K, no aug ───────────────────────────────────
    if not args.skip_exp2:
        print(f"=== EXPERIMENT 2: multi-dataset {combined_100k.shape[0] // 1000}K (C10+C100), no aug ===", flush=True)
        exp2 = _run_experiment(
            name="exp2-multi-100k",
            model_builder=builder,
            hebbian_images=combined_100k,
            planned_passes=args.passes_multi,
            use_augmentation=False,
            **common_kwargs,
        )
    else:
        print("(Skipping Exp 2)", flush=True)
        exp2 = ExperimentResult("exp2-skip", 0, 0, 0, 0, [])

    # ── Experiment 3: C10+C100+SVHN 173K, no aug ──────────────────────────────
    if not args.skip_exp3:
        print(f"=== EXPERIMENT 3: multi-dataset {combined_173k.shape[0] // 1000}K (C10+C100+SVHN), no aug ===", flush=True)
        exp3 = _run_experiment(
            name="exp3-multi-173k",
            model_builder=builder,
            hebbian_images=combined_173k,
            planned_passes=args.passes_multi_aug,
            use_augmentation=False,
            **common_kwargs,
        )
    else:
        print("(Skipping Exp 3)", flush=True)
        exp3 = ExperimentResult("exp3-skip", 0, 0, 0, 0, [])

    # ── Experiment 4: C10+C100+SVHN 173K + aug ────────────────────────────────
    if not args.skip_exp4:
        print(f"=== EXPERIMENT 4: multi-dataset {combined_173k.shape[0] // 1000}K (C10+C100+SVHN) + aug ===", flush=True)
        exp4 = _run_experiment(
            name="exp4-multi-173k-aug",
            model_builder=builder,
            hebbian_images=combined_173k,
            planned_passes=args.passes_max_aug,
            use_augmentation=True,
            **common_kwargs,
        )
    else:
        print("(Skipping Exp 4)", flush=True)
        exp4 = ExperimentResult("exp4-skip", 0, 0, 0, 0, [])

    # ── Report ────────────────────────────────────────────────────────────────
    _print_report(
        exp1=exp1,
        exp2=exp2,
        exp3=exp3,
        exp4=exp4,
        checkpoints=checkpoints,
        device=device,
    )

    decision_path = _write_decision(exp1=exp1, exp2=exp2, exp3=exp3, exp4=exp4)
    print(f"\nDecision written to {decision_path}", flush=True)

    # ── Log to file ───────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"data_scaling_gpu_{ts}.log"
    print(f"Log at: {log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
