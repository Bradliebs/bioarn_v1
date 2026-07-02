"""Phase 3c: L2-normalise features before sparse probe. One-line fix over Phase 3b to test whether GAP scale was killing the LP gradient."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.core.local_contrastive import LocalContrastiveEncoder
from bioarn.core.local_predictive import LocalPredictiveEncoder
from bioarn.core.softhebb_net import SoftHebbNet

try:
    from torchvision import datasets
except ImportError as exc:
    raise RuntimeError("torchvision required for local_ssl_3b.py") from exc

SEED = 42
NUM_CLASSES = 10
GAMMA_SWEEP = (0.5, 1.0, 2.0, 5.0, 10.0)
SOFT_CHANNELS = (96, 384, 512)
SOFT_KERNELS = (5, 3, 3)
SOFT_ETA = 0.01
PROBE_EPOCHS = 100
PASSES = 30
CHECKPOINTS = (1, 5, 10, 20, 30)
HEBBIAN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9
DIAG_SAMPLE = 5000
DEAD_THRESHOLD = 0.005
PHASE3_CEILING = 0.3887

A_NUM_FEATURES = 512
A_HIDDEN_CHANNELS = (256, 512)
A_KERNEL_SIZES = (5, 3, 3)
A_SPATIAL_GRID = 4
A_SPATIAL_TOP_K = 8
A_TOP_K = 256
A_COMPETITIVE_K = 64


@dataclass(frozen=True)
class SparseFeatureSet:
    indices: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    feature_dim: int


@dataclass(frozen=True)
class CollapseMetrics:
    effective_rank: float
    dead_feature_pct: float
    mean_sparsity: float
    feature_var_mean: float
    filter_cosim: float


@dataclass(frozen=True)
class PassMetrics:
    pass_num: int
    nc: float
    lp: float
    collapse: CollapseMetrics


@dataclass
class ExperimentResult:
    name: str
    label: str
    pass_metrics: list[PassMetrics]
    best_lp_pass: int
    best_health_pass: int
    notes: list[str] = field(default_factory=list)

    @property
    def best_lp(self) -> float:
        return max((m.lp for m in self.pass_metrics), default=0.0)

    @property
    def best_health_lp(self) -> float:
        for metric in self.pass_metrics:
            if metric.pass_num == self.best_health_pass:
                return metric.lp
        return 0.0

    @property
    def peak_effective_rank(self) -> float:
        return max((m.collapse.effective_rank for m in self.pass_metrics), default=0.0)

    @property
    def min_dead_feature_pct(self) -> float:
        return min((m.collapse.dead_feature_pct for m in self.pass_metrics), default=0.0)

    @property
    def metrics_by_pass(self) -> dict[int, PassMetrics]:
        return {metric.pass_num: metric for metric in self.pass_metrics}


class SparseLinearProbe(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.weight = nn.Parameter(torch.empty(self.feature_dim, NUM_CLASSES))
        self.bias = nn.Parameter(torch.empty(NUM_CLASSES))
        bound = 1.0 / math.sqrt(max(self.feature_dim, 1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(indices, self.weight)
        return (selected * values.unsqueeze(-1)).sum(dim=1) + self.bias


def _build_conv_f1() -> ConvF1Layer:
    return ConvF1Layer(
        in_channels=3,
        num_features=A_NUM_FEATURES,
        spatial_size=32,
        top_k=A_TOP_K,
        spatial_grid=A_SPATIAL_GRID,
        num_layers=3,
        hidden_channels=A_HIDDEN_CHANNELS,
        kernel_sizes=A_KERNEL_SIZES,
        spatial_top_k=A_SPATIAL_TOP_K,
        competitive_k=A_COMPETITIVE_K,
        hebbian_lr=0.005,
        hebbian_batch_size=HEBBIAN_BATCH_SIZE,
        weight_norm_target=1.0,
        enable_local_contrast_norm=True,
        hebbian_oja_decay=0.05,
        filter_decorrelation=0.02,
    )


def _build_softhebb(gamma: float, global_pool: bool = True) -> SoftHebbNet:
    return SoftHebbNet(
        channels=SOFT_CHANNELS,
        kernel_sizes=SOFT_KERNELS,
        gamma=gamma,
        eta=SOFT_ETA,
        global_pool=global_pool,
    )


def _build_contrastive(gamma: float, global_pool: bool = True) -> LocalContrastiveEncoder:
    return LocalContrastiveEncoder(
        channels=SOFT_CHANNELS,
        kernel_sizes=SOFT_KERNELS,
        gamma=gamma,
        eta=SOFT_ETA,
        global_pool=global_pool,
    )


def _build_predictive(gamma: float, global_pool: bool = True) -> LocalPredictiveEncoder:
    return LocalPredictiveEncoder(
        channels=SOFT_CHANNELS,
        kernel_sizes=SOFT_KERNELS,
        gamma=gamma,
        eta=SOFT_ETA,
        global_pool=global_pool,
    )


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.2f} pp"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def _load_cifar10(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    train_images = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).float() / 255.0
    test_images = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).float() / 255.0
    train_labels = torch.tensor(train_ds.targets, dtype=torch.long)
    test_labels = torch.tensor(test_ds.targets, dtype=torch.long)
    return train_images, train_labels, test_images, test_labels


def _augment_batch(batch: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() > 0.5:
        batch = batch.flip(-1)
    pad = 4
    padded = F.pad(batch, [pad, pad, pad, pad], mode="reflect")
    i, j = rng.randint(0, 2 * pad), rng.randint(0, 2 * pad)
    batch = padded[:, :, i : i + 32, j : j + 32]
    batch = (batch * rng.uniform(0.7, 1.3)).clamp(0.0, 1.0)
    mean = batch.mean(dim=(-1, -2), keepdim=True)
    batch = ((batch - mean) * rng.uniform(0.8, 1.2) + mean).clamp(0.0, 1.0)
    return batch


def _run_hebbian_pass(
    model: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    augment: bool = False,
    rng: random.Random | None = None,
) -> None:
    n = images.shape[0]
    perm = torch.randperm(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        batch = images[idx].to(device=device, dtype=torch.float32)
        if augment and rng is not None:
            batch = _augment_batch(batch, rng)
        signal = torch.ones(batch.shape[0], device=device)
        model.hebbian_update(batch, learning_signal=signal)
    model.flush_hebbian_updates()


@torch.inference_mode()
def _extract_dense_features(
    model: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device=device, dtype=torch.float32)
        chunks.append(model(batch).cpu().float())
    return torch.cat(chunks, dim=0)


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
    feature_dim = int(model.output_dim)
    top_k = min(sparse_k, feature_dim)
    all_indices = torch.empty((total, top_k), dtype=torch.int32)
    all_values = torch.empty((total, top_k), dtype=torch.float32)
    cursor = 0
    for start in range(0, total, batch_size):
        batch = images[start : start + batch_size].to(device=device, dtype=torch.float32)
        dense = model(batch).cpu().float()
        dense = F.normalize(dense, p=2, dim=1)  # Phase 3c: scale-normalise before probe
        vals, idxs = torch.topk(dense, k=top_k, dim=1)
        batch_size_actual = dense.shape[0]
        all_indices[cursor : cursor + batch_size_actual] = idxs.to(torch.int32)
        all_values[cursor : cursor + batch_size_actual] = vals
        cursor += batch_size_actual
    return SparseFeatureSet(indices=all_indices, values=all_values, labels=labels.clone(), feature_dim=feature_dim)


def _nc_accuracy(train: SparseFeatureSet, test: SparseFeatureSet) -> float:
    proto_sums = torch.zeros((NUM_CLASSES, train.feature_dim))
    counts = torch.zeros(NUM_CLASSES)
    for start in range(0, train.labels.shape[0], PROBE_BATCH_SIZE):
        end = start + PROBE_BATCH_SIZE
        idxs = train.indices[start:end].long()
        vals = train.values[start:end]
        lbls = train.labels[start:end]
        label_rows = lbls.unsqueeze(1).expand_as(idxs).reshape(-1)
        proto_sums.index_put_((label_rows, idxs.reshape(-1)), vals.reshape(-1), accumulate=True)
        counts += torch.bincount(lbls, minlength=NUM_CLASSES).float()
    prototypes = F.normalize(proto_sums / counts.unsqueeze(1).clamp(min=1), dim=1).T.contiguous()
    correct = total = 0
    for start in range(0, test.labels.shape[0], PROBE_BATCH_SIZE):
        end = start + PROBE_BATCH_SIZE
        idxs = test.indices[start:end].long()
        vals = test.values[start:end]
        lbls = test.labels[start:end]
        selected = F.embedding(idxs, prototypes)
        logits = (selected * vals.unsqueeze(-1)).sum(dim=1)
        correct += int((logits.argmax(dim=1) == lbls).sum())
        total += lbls.numel()
    return correct / max(total, 1)


def _lp_accuracy(
    train: SparseFeatureSet,
    test: SparseFeatureSet,
    *,
    seed: int,
    probe_epochs: int,
    device: torch.device,
) -> float:
    _set_seed(seed)
    probe = SparseLinearProbe(train.feature_dim).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    crit = nn.CrossEntropyLoss()
    n_train = train.labels.shape[0]
    for epoch in range(probe_epochs):
        probe.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, n_train, PROBE_BATCH_SIZE):
            end = start + PROBE_BATCH_SIZE
            batch = perm[start:end]
            idxs = train.indices[batch].to(device, dtype=torch.long)
            vals = train.values[batch].to(device)
            lbls = train.labels[batch].to(device)
            opt.zero_grad(set_to_none=True)
            crit(probe(idxs, vals), lbls).backward()
            opt.step()
    probe.eval()
    correct = total = 0
    with torch.inference_mode():
        for start in range(0, test.labels.shape[0], PROBE_BATCH_SIZE):
            end = start + PROBE_BATCH_SIZE
            idxs = test.indices[start:end].to(device, dtype=torch.long)
            vals = test.values[start:end].to(device)
            lbls = test.labels[start:end].to(device)
            correct += int((probe(idxs, vals).argmax(dim=1) == lbls).sum())
            total += lbls.numel()
    return correct / max(total, 1)


def _evaluate(
    model: nn.Module,
    train_imgs: torch.Tensor,
    train_lbls: torch.Tensor,
    test_imgs: torch.Tensor,
    test_lbls: torch.Tensor,
    *,
    sparse_k: int,
    device: torch.device,
    probe_epochs: int,
    seed: int,
) -> tuple[float, float]:
    train_f = _extract_sparse_features(model, train_imgs, train_lbls, batch_size=EVAL_BATCH_SIZE, sparse_k=sparse_k, device=device)
    test_f = _extract_sparse_features(model, test_imgs, test_lbls, batch_size=EVAL_BATCH_SIZE, sparse_k=sparse_k, device=device)
    nc = _nc_accuracy(train_f, test_f)
    lp = _lp_accuracy(train_f, test_f, seed=seed, probe_epochs=probe_epochs, device=device)
    return nc, lp


def _compute_collapse(
    model: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    sample_n: int = DIAG_SAMPLE,
) -> CollapseMetrics:
    actual_n = min(int(sample_n), int(images.shape[0]))
    sample_idx = torch.randperm(images.shape[0])[:actual_n]
    sample_images = images[sample_idx]
    features = _extract_dense_features(model, sample_images, batch_size=EVAL_BATCH_SIZE, device=device)
    features = F.normalize(features, p=2, dim=1)  # Phase 3c: normalise for scale-fair diagnostics
    feature_var = features.var(dim=0, unbiased=False)
    feature_var_mean = float(feature_var.mean().item())
    dead_feature_pct = float((feature_var < DEAD_THRESHOLD).float().mean().item())
    mean_sparsity = float((features.abs() < 0.01).float().mean(dim=1).mean().item())

    svd_input = features.T.contiguous() if features.shape[1] > features.shape[0] else features.contiguous()
    singular_values = torch.linalg.svdvals(svd_input)
    prob = singular_values / singular_values.sum().clamp_min(1e-12)
    spectral_entropy = -(prob * (prob + 1e-12).log()).sum()
    effective_rank = float(torch.exp(spectral_entropy).item())

    filter_cosim = float("nan")
    base_model = model.encoder if hasattr(model, "encoder") else model
    if isinstance(base_model, SoftHebbNet):
        weights = base_model.layers[-1].weight.data.float().reshape(base_model.layers[-1].weight.shape[0], -1)
        if weights.shape[0] > 1:
            weights = F.normalize(weights, dim=1, eps=1e-12)
            cos = (weights @ weights.T).abs()
            n_filters = cos.shape[0]
            off_diag_sum = cos.sum() - torch.diagonal(cos).sum()
            filter_cosim = float((off_diag_sum / (n_filters * (n_filters - 1))).item())

    return CollapseMetrics(
        effective_rank=effective_rank,
        dead_feature_pct=dead_feature_pct,
        mean_sparsity=mean_sparsity,
        feature_var_mean=feature_var_mean,
        filter_cosim=filter_cosim,
    )


def _debug_predictive(
    model: LocalPredictiveEncoder,
    images: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        pair = images[:2].to(device=device, dtype=torch.float32)
        pair_feats = model(pair)
        two_image_diff = float(torch.norm(pair_feats[0] - pair_feats[1], p=2).item())

        sample_idx = torch.randperm(images.shape[0])[:10]
        sample_feats = model(images[sample_idx].to(device=device, dtype=torch.float32)).float()
        per_image_std = sample_feats.std(dim=1)
        feature_nonconst = bool(torch.all(per_image_std > 0.001).item())

    batch1 = images[:HEBBIAN_BATCH_SIZE].to(device=device, dtype=torch.float32)
    batch2 = images[HEBBIAN_BATCH_SIZE : 2 * HEBBIAN_BATCH_SIZE].to(device=device, dtype=torch.float32)
    model.hebbian_update(batch1)
    pred_loss_initial = float(model._last_pred_loss)
    model.hebbian_update(batch2)
    pred_loss_after = float(model._last_pred_loss)
    model.flush_hebbian_updates()

    pred_loss_finite = math.isfinite(pred_loss_initial) and math.isfinite(pred_loss_after)
    pred_loss_decreased = pred_loss_finite and pred_loss_after <= pred_loss_initial
    pred_loss_trend = "down" if pred_loss_decreased else ("finite_no_drop" if pred_loss_finite else "nonfinite")

    issues: list[str] = []
    if two_image_diff <= 0.0:
        issues.append("two_image_diff<=0")
    if not feature_nonconst:
        issues.append("feature std <= 0.001 for one or more samples")
    if not pred_loss_finite:
        issues.append("predictive loss non-finite")

    status = "ok" if not issues else f"WARN: {', '.join(issues)}"
    result = {
        "status": status,
        "two_image_diff": two_image_diff,
        "pred_loss_initial": pred_loss_initial,
        "pred_loss_after": pred_loss_after,
        "pred_loss_trend": pred_loss_trend,
        "feature_nonconst": feature_nonconst,
    }

    print("\n[D debug] LocalPredictiveEncoder sanity", flush=True)
    print(f"  two_image_diff   : {_fmt(two_image_diff, 6)}", flush=True)
    print(f"  pred_loss_initial: {_fmt(pred_loss_initial, 6)}", flush=True)
    print(f"  pred_loss_after  : {_fmt(pred_loss_after, 6)}", flush=True)
    print(f"  pred_loss_trend  : {pred_loss_trend}", flush=True)
    print(f"  feature_nonconst : {feature_nonconst}", flush=True)
    print(f"  status           : {status}", flush=True)
    if status != "ok":
        print(f"⚠️ D DIAGNOSTIC FAILED: {', '.join(issues)}", flush=True)
    return result


def _run_experiment(
    *,
    name: str,
    label: str,
    model: nn.Module,
    sparse_k: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    planned_passes: int,
    checkpoints: tuple[int, ...],
    use_augmentation: bool,
    device: torch.device,
    probe_epochs: int,
    seed: int,
    smoke_test: bool = False,
    diag_sample: int = DIAG_SAMPLE,
) -> ExperimentResult:
    model = model.to(device)
    rng = random.Random(seed)
    metrics: list[PassMetrics] = []
    actual_passes = 1 if smoke_test else planned_passes
    actual_probe_epochs = 2 if smoke_test else probe_epochs
    actual_diag_sample = min(diag_sample, 512) if smoke_test else diag_sample

    print(
        f"[{name}] train={train_images.shape[0]//1000}K device={device} passes={actual_passes} aug={use_augmentation}",
        flush=True,
    )

    for pass_num in range(1, actual_passes + 1):
        t0 = time.perf_counter()
        _run_hebbian_pass(model, train_images, batch_size=HEBBIAN_BATCH_SIZE, device=device, augment=use_augmentation, rng=rng)
        t_hebb = time.perf_counter() - t0

        if pass_num in checkpoints or pass_num == actual_passes:
            t_eval = time.perf_counter()
            nc, lp = _evaluate(
                model,
                train_images,
                train_labels,
                test_images,
                test_labels,
                sparse_k=sparse_k,
                device=device,
                probe_epochs=actual_probe_epochs,
                seed=seed,
            )
            collapse = _compute_collapse(model, train_images, device=device, sample_n=actual_diag_sample)
            t_eval = time.perf_counter() - t_eval
            metrics.append(PassMetrics(pass_num=pass_num, nc=nc, lp=lp, collapse=collapse))
            print(
                f"[{name}] pass {pass_num}/{actual_passes}  "
                f"hebbian {t_hebb:.1f}s  "
                f"nc={nc*100:.2f}%  "
                f"lp={lp*100:.2f}%  "
                f"eff_rank={collapse.effective_rank:.1f}  "
                f"dead={collapse.dead_feature_pct*100:.1f}%  "
                f"filter_cosim={_fmt(collapse.filter_cosim, 2)}  "
                f"eval {t_eval/60:.1f}m",
                flush=True,
            )
        else:
            print(f"[{name}] pass {pass_num}/{actual_passes}  hebbian {t_hebb:.1f}s", flush=True)

    best_lp_pass = max(metrics, key=lambda item: item.lp).pass_num if metrics else 0
    best_health_pass = max(metrics, key=lambda item: item.collapse.effective_rank).pass_num if metrics else 0
    return ExperimentResult(
        name=name,
        label=label,
        pass_metrics=metrics,
        best_lp_pass=best_lp_pass,
        best_health_pass=best_health_pass,
    )


def _format_pass_table(result: ExperimentResult) -> str:
    lines = [
        "| Pass | NC | LP | Eff_rank | Dead% | Sparsity | Filter_cosim |",
        "|------|----|----|----------|-------|----------|--------------|",
    ]
    for metric in result.pass_metrics:
        lines.append(
            "| "
            f"{metric.pass_num} | "
            f"{_pct(metric.nc)} | "
            f"{_pct(metric.lp)} | "
            f"{_fmt(metric.collapse.effective_rank, 2)} | "
            f"{metric.collapse.dead_feature_pct * 100:.2f}% | "
            f"{metric.collapse.mean_sparsity * 100:.2f}% | "
            f"{_fmt(metric.collapse.filter_cosim, 3)} |"
        )
    return "\n".join(lines)


def _print_result_block(result: ExperimentResult) -> None:
    print(result.label, flush=True)
    print(_format_pass_table(result), flush=True)
    print(
        f"Best LP pass: {result.best_lp_pass} ({_pct(result.best_lp)}) / "
        f"Best health pass: {result.best_health_pass} ({_pct(result.best_health_lp)})",
        flush=True,
    )
    for note in result.notes:
        print(f"Note: {note}", flush=True)


def _print_report(
    *,
    seed: int,
    device: torch.device,
    probe_epochs: int,
    conv_result: ExperimentResult | None,
    sweep_results: dict[float, ExperimentResult],
    spatial_diag_result: ExperimentResult | None,
    contrastive_result: ExperimentResult | None,
    predictive_result: ExperimentResult | None,
    predictive_diag: dict[str, Any] | None,
) -> None:
    print("\n================================================================", flush=True)
    print("PHASE 3c — RESULTS (L2-normalised features)", flush=True)
    print("================================================================", flush=True)
    print(f"Seed: {seed} | Device: {device} | Probe epochs: {probe_epochs}", flush=True)

    if conv_result is not None:
        print("\nExp A: ConvF1 control", flush=True)
        _print_result_block(conv_result)

    if sweep_results:
        print("\nExp B: SoftHebb γ sweep (global pool 512-dim)", flush=True)
        print(
            "| γ | Best LP | Best LP pass | LP@health | Health pass | Peak eff_rank | Min dead% |",
            flush=True,
        )
        print(
            "|---|---------|--------------|-----------|-------------|---------------|-----------|",
            flush=True,
        )
        for gamma in sorted(sweep_results):
            result = sweep_results[gamma]
            print(
                f"| {gamma:g} | {_pct(result.best_lp)} | {result.best_lp_pass} | {_pct(result.best_health_lp)} | "
                f"{result.best_health_pass} | {_fmt(result.peak_effective_rank, 2)} | {result.min_dead_feature_pct * 100:.2f}% |",
                flush=True,
            )

    if spatial_diag_result is not None:
        print("\nExp B diagnostic: SoftHebb γ=10 spatial (8192-dim)", flush=True)
        _print_result_block(spatial_diag_result)

    if contrastive_result is not None:
        print("\nExp C: LocalContrastive (γ=BEST, global pool)", flush=True)
        _print_result_block(contrastive_result)

    if predictive_result is not None:
        print("\nExp D: LocalPredictive (γ=BEST, global pool)", flush=True)
        if predictive_diag is not None:
            print("| D diag status | pred_loss trend | two_image_diff |", flush=True)
            print("|---------------|-----------------|----------------|", flush=True)
            print(
                f"| {predictive_diag['status']} | {predictive_diag['pred_loss_trend']} | "
                f"{_fmt(float(predictive_diag['two_image_diff']), 6)} |",
                flush=True,
            )
        _print_result_block(predictive_result)

    print("\nSUMMARY vs Phase 3 ceiling (38.87%):", flush=True)
    print("| Exp | Model | Best LP | Δ vs 38.87% | Health LP |", flush=True)
    print("|-----|-------|---------|-------------|-----------|", flush=True)
    if conv_result is not None:
        print(
            f"| A | ConvF1 control | {_pct(conv_result.best_lp)} | {_pp(conv_result.best_lp - PHASE3_CEILING)} | {_pct(conv_result.best_health_lp)} |",
            flush=True,
        )
    if sweep_results:
        best_gamma, best_sweep = max(sweep_results.items(), key=lambda item: item[1].best_lp)
        print(
            f"| B | SoftHebb γ={best_gamma:g} | {_pct(best_sweep.best_lp)} | {_pp(best_sweep.best_lp - PHASE3_CEILING)} | {_pct(best_sweep.best_health_lp)} |",
            flush=True,
        )
    if spatial_diag_result is not None:
        print(
            f"| B* | SoftHebb γ=10 spatial | {_pct(spatial_diag_result.best_lp)} | {_pp(spatial_diag_result.best_lp - PHASE3_CEILING)} | {_pct(spatial_diag_result.best_health_lp)} |",
            flush=True,
        )
    if contrastive_result is not None:
        print(
            f"| C | LocalContrastive | {_pct(contrastive_result.best_lp)} | {_pp(contrastive_result.best_lp - PHASE3_CEILING)} | {_pct(contrastive_result.best_health_lp)} |",
            flush=True,
        )
    if predictive_result is not None:
        print(
            f"| D | LocalPredictive | {_pct(predictive_result.best_lp)} | {_pp(predictive_result.best_lp - PHASE3_CEILING)} | {_pct(predictive_result.best_health_lp)} |",
            flush=True,
        )


def _write_decision(
    *,
    best_gamma: float,
    conv_result: ExperimentResult | None,
    sweep_results: dict[float, ExperimentResult],
    spatial_diag_result: ExperimentResult | None,
    contrastive_result: ExperimentResult | None,
    predictive_result: ExperimentResult | None,
    predictive_diag: dict[str, Any] | None,
) -> Path:
    ts = time.strftime("%Y-%m-%d")
    path = ROOT / ".squad" / "decisions" / "inbox" / "tank-local-ssl-3c.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"### {ts}: Phase 3b local SSL repair",
        "**By:** Tank (Core Neural)",
        "**What:** Re-tested local SSL after lowering SoftWTA γ and switching to global-pooled 512-dim features.",
        f"**Phase 3 ceiling:** {PHASE3_CEILING * 100:.2f}% LP",
        f"**Selected γ:** {best_gamma:g}",
        "",
    ]

    if conv_result is not None:
        lines.append(
            f"- **Exp A / ConvF1:** best LP {_pct(conv_result.best_lp)} at pass {conv_result.best_lp_pass}; "
            f"health LP {_pct(conv_result.best_health_lp)} at pass {conv_result.best_health_pass}."
        )
    if sweep_results:
        lines.append("- **Exp B / SoftHebb global pool sweep:**")
        for gamma in sorted(sweep_results):
            result = sweep_results[gamma]
            lines.append(
                f"  - γ={gamma:g}: best LP {_pct(result.best_lp)} (pass {result.best_lp_pass}); "
                f"LP@health {_pct(result.best_health_lp)} (pass {result.best_health_pass}); "
                f"peak rank {result.peak_effective_rank:.2f}; min dead {result.min_dead_feature_pct * 100:.2f}%."
            )
    if spatial_diag_result is not None:
        lines.append(
            f"- **Exp B diagnostic / SoftHebb γ=10 spatial:** best LP {_pct(spatial_diag_result.best_lp)}; "
            f"health LP {_pct(spatial_diag_result.best_health_lp)}."
        )
    if contrastive_result is not None:
        lines.append(
            f"- **Exp C / LocalContrastive:** best LP {_pct(contrastive_result.best_lp)} at pass {contrastive_result.best_lp_pass}; "
            f"health LP {_pct(contrastive_result.best_health_lp)}."
        )
    if predictive_result is not None:
        lines.append(
            f"- **Exp D / LocalPredictive:** best LP {_pct(predictive_result.best_lp)} at pass {predictive_result.best_lp_pass}; "
            f"health LP {_pct(predictive_result.best_health_lp)}."
        )
    if predictive_diag is not None:
        lines.append(
            f"- **Predictive diagnostic:** status={predictive_diag['status']}; trend={predictive_diag['pred_loss_trend']}; "
            f"two_image_diff={float(predictive_diag['two_image_diff']):.6f}."
        )

    all_results = [result for result in [conv_result, spatial_diag_result, contrastive_result, predictive_result] if result is not None]
    all_results.extend(sweep_results.values())
    best_overall = max(all_results, key=lambda result: result.best_lp, default=None)
    if best_overall is not None:
        delta = best_overall.best_lp - PHASE3_CEILING
        verdict = "beat" if delta > 0 else "did not beat"
        lines.extend(
            [
                "",
                f"**Decision:** Phase 3b {verdict} the Phase 3 ceiling. "
                f"Best overall run: {best_overall.label} at {_pct(best_overall.best_lp)} ({_pp(delta)}).",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="1 pass, 2 probe epochs, γ=2.0 only")
    parser.add_argument("--skip-a", action="store_true", help="Skip Exp A (ConvF1 control)")
    parser.add_argument("--skip-b", action="store_true", help="Skip Exp B (SoftHebb γ sweep)")
    parser.add_argument("--skip-c", action="store_true", help="Skip Exp C (LocalContrastive)")
    parser.add_argument("--skip-d", action="store_true", help="Skip Exp D (LocalPredictive)")
    parser.add_argument("--gamma-only", type=float, default=None, help="Run Exp B sweep with a single γ value")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--no-spatial-diag", action="store_true", help="Skip γ=10 / global_pool=False mismatch diagnostic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})", flush=True)
    print(f"Threads: {torch.get_num_threads()}", flush=True)

    checkpoints = (1,) if args.smoke_test else CHECKPOINTS
    probe_epochs = 2 if args.smoke_test else PROBE_EPOCHS
    planned_passes = 1 if args.smoke_test else args.passes
    if args.smoke_test:
        print("SMOKE TEST MODE: 1 pass, 2 probe epochs, γ=2.0 only", flush=True)

    sweep_gammas = (2.0,) if args.smoke_test else ((float(args.gamma_only),) if args.gamma_only is not None else GAMMA_SWEEP)

    print("\nLoading CIFAR-10 ...", flush=True)
    t0 = time.perf_counter()
    train_imgs, train_lbls, test_imgs, test_lbls = _load_cifar10(args.data_dir)
    print(
        f"  {train_imgs.shape[0]//1000}K train, {test_imgs.shape[0]//1000}K test — {time.perf_counter() - t0:.1f}s\n",
        flush=True,
    )

    common = dict(
        train_images=train_imgs,
        train_labels=train_lbls,
        test_images=test_imgs,
        test_labels=test_lbls,
        planned_passes=planned_passes,
        checkpoints=checkpoints,
        device=device,
        probe_epochs=probe_epochs,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    conv_result: ExperimentResult | None = None
    sweep_results: dict[float, ExperimentResult] = {}
    spatial_diag_result: ExperimentResult | None = None
    contrastive_result: ExperimentResult | None = None
    predictive_result: ExperimentResult | None = None
    predictive_diag: dict[str, Any] | None = None

    if not args.skip_a:
        print("=== EXPERIMENT A: ConvF1 control ===", flush=True)
        conv_result = _run_experiment(
            name="expA-conv-f1-control",
            label="Exp A: ConvF1 control",
            model=_build_conv_f1(),
            sparse_k=A_TOP_K,
            use_augmentation=True,
            **common,
        )
    else:
        print("(Skipping Exp A)", flush=True)

    if not args.skip_b:
        print("\n=== EXPERIMENT B: SoftHebb γ sweep (global pool) ===", flush=True)
        for gamma in sweep_gammas:
            sweep_results[gamma] = _run_experiment(
                name=f"expB-softhebb-g{gamma:g}",
                label=f"Exp B: SoftHebb γ={gamma:g} (global pool)",
                model=_build_softhebb(gamma=gamma, global_pool=True),
                sparse_k=256,
                use_augmentation=True,
                **common,
            )
        if not args.no_spatial_diag:
            print("\n=== EXPERIMENT B diagnostic: SoftHebb γ=10 spatial mismatch ===", flush=True)
            spatial_diag_result = _run_experiment(
                name="expB-softhebb-g10-spatial",
                label="Exp B diagnostic: SoftHebb γ=10 (spatial 8192-dim)",
                model=_build_softhebb(gamma=10.0, global_pool=False),
                sparse_k=256,
                use_augmentation=True,
                **common,
            )
    else:
        print("(Skipping Exp B)", flush=True)

    if args.smoke_test or args.gamma_only is not None or not sweep_results:
        best_gamma = 2.0
    else:
        best_gamma = max(sweep_results.items(), key=lambda item: item[1].best_lp)[0]
    print(f"Selected γ for Exp C/D: {best_gamma:g}", flush=True)

    if not args.skip_c:
        print("\n=== EXPERIMENT C: LocalContrastive ===", flush=True)
        contrastive_result = _run_experiment(
            name=f"expC-local-contrastive-g{best_gamma:g}",
            label=f"Exp C: LocalContrastive γ={best_gamma:g} (global pool)",
            model=_build_contrastive(gamma=best_gamma, global_pool=True),
            sparse_k=256,
            use_augmentation=False,
            **common,
        )
    else:
        print("(Skipping Exp C)", flush=True)

    if not args.skip_d:
        print("\n=== EXPERIMENT D: LocalPredictive ===", flush=True)
        predictive_diag = _debug_predictive(_build_predictive(gamma=best_gamma, global_pool=True), train_imgs, device=device)
        predictive_result = _run_experiment(
            name=f"expD-local-predictive-g{best_gamma:g}",
            label=f"Exp D: LocalPredictive γ={best_gamma:g} (global pool)",
            model=_build_predictive(gamma=best_gamma, global_pool=True),
            sparse_k=256,
            use_augmentation=False,
            **common,
        )
        if predictive_diag["status"] != "ok":
            predictive_result.notes.append(f"Predictive diagnostic warning: {predictive_diag['status']}")
    else:
        print("(Skipping Exp D)", flush=True)

    _print_report(
        seed=args.seed,
        device=device,
        probe_epochs=probe_epochs,
        conv_result=conv_result,
        sweep_results=sweep_results,
        spatial_diag_result=spatial_diag_result,
        contrastive_result=contrastive_result,
        predictive_result=predictive_result,
        predictive_diag=predictive_diag,
    )
    decision_path = _write_decision(
        best_gamma=best_gamma,
        conv_result=conv_result,
        sweep_results=sweep_results,
        spatial_diag_result=spatial_diag_result,
        contrastive_result=contrastive_result,
        predictive_result=predictive_result,
        predictive_diag=predictive_diag,
    )
    print(f"\nDecision written to {decision_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

