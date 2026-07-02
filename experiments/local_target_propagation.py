"""Phase 5: bounded credit-assignment test via Difference Target Propagation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.core.conv_ccc import ConvF1Layer
from bioarn.core.dtp_encoder import DTPContrastiveEncoder
from bioarn.core.simclr import simclr_augment

try:
    from torchvision import datasets
except ImportError as exc:
    raise RuntimeError("torchvision required for experiments/local_target_propagation.py") from exc

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
NUM_CLASSES = 10
PROBE_EPOCHS = 100
PASSES = 30
CHECKPOINTS = (1, 5, 10, 20, 30)
BATCH_SIZE_DTP = 256
BATCH_SIZE_HEBBIAN = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9
DIAG_SAMPLE = 5000
DEAD_THRESHOLD = 0.005
PHASE4_CONVF1_LP = 0.3800
PHASE4_SIMCLR_LP = 0.6554
PHASE4_LOCAL_LP = 0.3195

A_NUM_FEATURES = 512
A_HIDDEN_CHANNELS = (256, 512)
A_KERNEL_SIZES = (5, 3, 3)
A_SPATIAL_GRID = 4
A_SPATIAL_TOP_K = 8
A_TOP_K = 256
A_COMPETITIVE_K = 64

FEATURE_TOP_K = 256
SMOKE_TRAIN_SIZE = 12_000
SMOKE_TEST_SIZE = 1_000


# ── Data structures ───────────────────────────────────────────────────────────

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
class DTPStepMetrics:
    contrastive_loss: float
    layer_losses: tuple[float, float, float]
    inv_losses: tuple[float, float]
    target_displacements: tuple[float, float, float]


@dataclass
class PassMetrics:
    pass_num: int
    nc: float
    lp: float
    collapse: CollapseMetrics
    dtp_metrics: DTPStepMetrics | None = None  # last-batch DTP metrics for this pass


@dataclass
class ExperimentResult:
    name: str
    label: str
    pass_metrics: list[PassMetrics] = field(default_factory=list)
    best_lp_pass: int = 0
    best_health_pass: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def best_lp(self) -> float:
        return max((m.lp for m in self.pass_metrics), default=0.0)

    @property
    def best_health_lp(self) -> float:
        if not self.pass_metrics:
            return 0.0
        best = max(self.pass_metrics, key=lambda m: m.collapse.effective_rank)
        return best.lp


# ── CIFAR-10 loading ──────────────────────────────────────────────────────────

def _load_cifar10(data_root: Path, smoke_test: bool = False):
    def _to_tensor(ds):
        imgs = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        lbls = torch.tensor(ds.targets, dtype=torch.long)
        return imgs, lbls

    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=False)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=False)
    train_imgs, train_lbls = _to_tensor(train_ds)
    test_imgs, test_lbls = _to_tensor(test_ds)

    if smoke_test:
        g = torch.Generator()
        g.manual_seed(SEED)
        idx = torch.randperm(len(train_imgs), generator=g)[:SMOKE_TRAIN_SIZE]
        train_imgs, train_lbls = train_imgs[idx], train_lbls[idx]
        idx2 = torch.randperm(len(test_imgs), generator=g)[:SMOKE_TEST_SIZE]
        test_imgs, test_lbls = test_imgs[idx2], test_lbls[idx2]
        print(f"  smoke subset: {len(train_imgs):,} train / {len(test_imgs):,} test")

    return train_imgs, train_lbls, test_imgs, test_lbls


# ── Evaluation utilities ──────────────────────────────────────────────────────

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
        dense = F.normalize(dense, p=2, dim=1)
        vals, idxs = torch.topk(dense, k=top_k, dim=1)
        n = dense.shape[0]
        all_indices[cursor : cursor + n] = idxs.to(torch.int32)
        all_values[cursor : cursor + n] = vals
        cursor += n
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
    g = torch.Generator()
    g.manual_seed(seed)
    probe = nn.EmbeddingBag(train.feature_dim, NUM_CLASSES, mode="sum", sparse=True)
    nn.init.zeros_(probe.weight)
    opt = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    crit = nn.CrossEntropyLoss()
    probe = probe.to(device)
    probe.train()
    n = train.labels.shape[0]
    for _ in range(probe_epochs):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, PROBE_BATCH_SIZE):
            end = start + PROBE_BATCH_SIZE
            p = perm[start:end]
            idxs = train.indices[p].to(device, dtype=torch.long)
            vals = train.values[p].to(device)
            lbls = train.labels[p].to(device)
            opt.zero_grad(set_to_none=True)
            crit(probe(idxs, per_sample_weights=vals), lbls).backward()
            opt.step()
    probe.eval()
    correct = total = 0
    with torch.inference_mode():
        for start in range(0, test.labels.shape[0], PROBE_BATCH_SIZE):
            end = start + PROBE_BATCH_SIZE
            idxs = test.indices[start:end].to(device, dtype=torch.long)
            vals = test.values[start:end].to(device)
            lbls = test.labels[start:end].to(device)
            correct += int((probe(idxs, per_sample_weights=vals).argmax(dim=1) == lbls).sum())
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


def _extract_last_filter_bank(model: nn.Module) -> torch.Tensor | None:
    if isinstance(model, DTPContrastiveEncoder):
        layer = model.enc_layers[-1][0]  # Conv2d is first element of nn.Sequential
        return layer.weight.data.float().reshape(layer.weight.shape[0], -1)
    if isinstance(model, ConvF1Layer):
        return None
    return None


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
    features_raw = _extract_dense_features(model, sample_images, batch_size=EVAL_BATCH_SIZE, device=device)
    features = F.normalize(features_raw, p=2, dim=1)
    raw_var = features_raw.var(dim=0, unbiased=False)
    feature_var_mean = float(raw_var.mean().item())
    dead_feature_pct = float((raw_var < DEAD_THRESHOLD).float().mean().item())
    mean_sparsity = float((features.abs() < 0.01).float().mean(dim=1).mean().item())

    svd_input = features.T.contiguous() if features.shape[1] > features.shape[0] else features.contiguous()
    singular_values = torch.linalg.svdvals(svd_input)
    prob = singular_values / singular_values.sum().clamp_min(1e-12)
    spectral_entropy = -(prob * (prob + 1e-12).log()).sum()
    effective_rank = float(torch.exp(spectral_entropy).item())

    filter_cosim = float("nan")
    weights = _extract_last_filter_bank(model)
    if weights is not None and weights.shape[0] > 1:
        weights = F.normalize(weights, dim=1, eps=1e-12)
        cos = (weights @ weights.T).abs()
        n_filters = cos.shape[0]
        off_diag = cos.sum() - torch.diagonal(cos).sum()
        filter_cosim = float((off_diag / (n_filters * (n_filters - 1))).item())

    return CollapseMetrics(
        effective_rank=effective_rank,
        dead_feature_pct=dead_feature_pct,
        mean_sparsity=mean_sparsity,
        feature_var_mean=feature_var_mean,
        filter_cosim=filter_cosim,
    )


# ── ConvF1 pass runner ────────────────────────────────────────────────────────

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


def _run_convf1_pass(
    model: ConvF1Layer,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> None:
    n = images.shape[0]
    perm = torch.randperm(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        batch = images[idx].to(device=device, dtype=torch.float32)
        batch = _augment_batch(batch, rng)
        signal = torch.ones(batch.shape[0], device=device)
        model.hebbian_update(batch, learning_signal=signal)
    model.flush_hebbian_updates()


# ── DTP pass runner ───────────────────────────────────────────────────────────

def _run_dtp_pass(
    model: DTPContrastiveEncoder,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> DTPStepMetrics:
    """Run one full pass over the dataset with DTP. Returns last-batch metrics."""
    model.train()
    n = images.shape[0]
    perm = list(range(n))
    rng.shuffle(perm)
    last_metrics: DTPStepMetrics | None = None
    for start in range(0, n, batch_size):
        batch_idx = perm[start : start + batch_size]
        if len(batch_idx) < 4:
            continue
        batch = images[batch_idx].to(device=device, dtype=torch.float32)
        view_a = simclr_augment(batch, rng)
        view_b = simclr_augment(batch, rng)
        raw = model.train_step(view_a, view_b)
        last_metrics = DTPStepMetrics(
            contrastive_loss=raw.contrastive_loss,
            layer_losses=raw.layer_losses,
            inv_losses=raw.inv_losses,
            target_displacements=raw.target_displacements,
        )
    return last_metrics or DTPStepMetrics(
        contrastive_loss=float("nan"),
        layer_losses=(float("nan"), float("nan"), float("nan")),
        inv_losses=(float("nan"), float("nan")),
        target_displacements=(float("nan"), float("nan"), float("nan")),
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt(v: float, decimals: int = 3) -> str:
    return "—" if v != v else f"{v:.{decimals}f}"  # nan check


# ── Experiment runners ────────────────────────────────────────────────────────

def _run_convf1_experiment(
    *,
    name: str,
    label: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    probe_epochs: int,
    seed: int,
    smoke_test: bool = False,
    diag_sample: int = DIAG_SAMPLE,
) -> ExperimentResult:
    model = ConvF1Layer(
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
        hebbian_batch_size=BATCH_SIZE_HEBBIAN,
        weight_norm_target=1.0,
        enable_local_contrast_norm=True,
        hebbian_oja_decay=0.05,
        filter_decorrelation=0.02,
    ).to(device)

    rng = random.Random(seed)
    metrics: list[PassMetrics] = []
    actual_passes = 1 if smoke_test else PASSES
    actual_probe = 2 if smoke_test else probe_epochs
    actual_diag = min(diag_sample, 512) if smoke_test else diag_sample

    print(f"[{name}] train={train_images.shape[0]//1000}K device={device} passes={actual_passes}", flush=True)
    for pass_num in range(1, actual_passes + 1):
        t0 = time.perf_counter()
        _run_convf1_pass(model, train_images, batch_size=BATCH_SIZE_HEBBIAN, device=device, rng=rng)
        t_train = time.perf_counter() - t0

        if pass_num in CHECKPOINTS or pass_num == actual_passes:
            t_eval = time.perf_counter()
            nc, lp = _evaluate(
                model, train_images, train_labels, test_images, test_labels,
                sparse_k=A_TOP_K, device=device, probe_epochs=actual_probe, seed=seed,
            )
            collapse = _compute_collapse(model, train_images, device=device, sample_n=actual_diag)
            t_eval = time.perf_counter() - t_eval
            metrics.append(PassMetrics(pass_num=pass_num, nc=nc, lp=lp, collapse=collapse))
            print(
                f"[{name}] pass {pass_num}/{actual_passes}  train {t_train:.1f}s  "
                f"nc={_pct(nc)}  lp={_pct(lp)}  eff_rank={collapse.effective_rank:.1f}  "
                f"dead={_pct(collapse.dead_feature_pct)}  filter_cosim={_fmt(collapse.filter_cosim)}  "
                f"eval {(time.perf_counter() - t0 - t_train)/60:.1f}m",
                flush=True,
            )
        else:
            print(f"[{name}] pass {pass_num}/{actual_passes}  train {t_train:.1f}s", flush=True)

    best_lp_pass = max(metrics, key=lambda m: m.lp).pass_num if metrics else 0
    best_health_pass = max(metrics, key=lambda m: m.collapse.effective_rank).pass_num if metrics else 0
    return ExperimentResult(name=name, label=label, pass_metrics=metrics,
                            best_lp_pass=best_lp_pass, best_health_pass=best_health_pass)


def _run_dtp_experiment(
    *,
    name: str,
    label: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    probe_epochs: int,
    seed: int,
    smoke_test: bool = False,
    diag_sample: int = DIAG_SAMPLE,
) -> ExperimentResult:
    model = DTPContrastiveEncoder().to(device)

    rng = random.Random(seed)
    metrics: list[PassMetrics] = []
    actual_passes = 1 if smoke_test else PASSES
    actual_probe = 2 if smoke_test else probe_epochs
    actual_diag = min(diag_sample, 512) if smoke_test else diag_sample

    print(
        f"[{name}] BIO_ARN_DTP train={train_images.shape[0]//1000}K device={device} passes={actual_passes}",
        flush=True,
    )
    for pass_num in range(1, actual_passes + 1):
        t0 = time.perf_counter()
        dtp_metrics = _run_dtp_pass(model, train_images, batch_size=BATCH_SIZE_DTP, device=device, rng=rng)
        t_train = time.perf_counter() - t0

        if pass_num in CHECKPOINTS or pass_num == actual_passes:
            t_eval = time.perf_counter()
            nc, lp = _evaluate(
                model, train_images, train_labels, test_images, test_labels,
                sparse_k=FEATURE_TOP_K, device=device, probe_epochs=actual_probe, seed=seed,
            )
            collapse = _compute_collapse(model, train_images, device=device, sample_n=actual_diag)
            t_eval = time.perf_counter() - t_eval
            metrics.append(PassMetrics(pass_num=pass_num, nc=nc, lp=lp, collapse=collapse, dtp_metrics=dtp_metrics))
            l3, l2, l1 = dtp_metrics.layer_losses
            d3, d2, d1 = dtp_metrics.target_displacements
            inv0, inv1 = dtp_metrics.inv_losses
            print(
                f"[{name}] BIO_ARN_DTP pass {pass_num}/{actual_passes}  train {t_train:.1f}s  "
                f"cont_loss={dtp_metrics.contrastive_loss:.4f}  "
                f"L=(l3={l3:.3f},l2={l2:.3f},l1={l1:.3f})  "
                f"inv=({inv0:.3f},{inv1:.3f})  "
                f"disp=({d3:.3f},{d2:.3f},{d1:.3f})  "
                f"nc={_pct(nc)}  lp={_pct(lp)}  eff_rank={collapse.effective_rank:.1f}  "
                f"dead={_pct(collapse.dead_feature_pct)}  "
                f"eval {t_eval/60:.1f}m",
                flush=True,
            )
        else:
            l3, l2, l1 = dtp_metrics.layer_losses
            print(
                f"[{name}] BIO_ARN_DTP pass {pass_num}/{actual_passes}  train {t_train:.1f}s  "
                f"cont_loss={dtp_metrics.contrastive_loss:.4f}  L=({l3:.3f},{l2:.3f},{l1:.3f})",
                flush=True,
            )

    best_lp_pass = max(metrics, key=lambda m: m.lp).pass_num if metrics else 0
    best_health_pass = max(metrics, key=lambda m: m.collapse.effective_rank).pass_num if metrics else 0
    return ExperimentResult(name=name, label=label, pass_metrics=metrics,
                            best_lp_pass=best_lp_pass, best_health_pass=best_health_pass)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _format_pass_table(result: ExperimentResult) -> str:
    lines = [
        "| Pass | NC | LP | Eff_rank | Dead% | Sparsity | cont_loss | disp3 |",
        "|------|----|----|----------|-------|----------|-----------|-------|",
    ]
    for m in result.pass_metrics:
        cont = _fmt(m.dtp_metrics.contrastive_loss, 4) if m.dtp_metrics else "—"
        disp3 = _fmt(m.dtp_metrics.target_displacements[0], 3) if m.dtp_metrics else "—"
        lines.append(
            f"| {m.pass_num} | {_pct(m.nc)} | {_pct(m.lp)} | "
            f"{_fmt(m.collapse.effective_rank, 2)} | "
            f"{m.collapse.dead_feature_pct*100:.2f}% | "
            f"{m.collapse.mean_sparsity*100:.2f}% | "
            f"{cont} | {disp3} |"
        )
    return "\n".join(lines)


def _print_report(
    *,
    seed: int,
    device: torch.device,
    probe_epochs: int,
    conv_result: ExperimentResult | None,
    dtp_result: ExperimentResult | None,
) -> None:
    print("\n================================================================", flush=True)
    print("PHASE 5 RESULTS", flush=True)
    print("================================================================", flush=True)
    print(f"Seed: {seed} | Device: {device} | Probe epochs: {probe_epochs}", flush=True)

    if conv_result is not None:
        print("\nExp A: ConvF1 control", flush=True)
        print(_format_pass_table(conv_result), flush=True)
        print(
            f"Best LP pass: {conv_result.best_lp_pass} ({_pct(conv_result.best_lp)}) / "
            f"Best health pass: {conv_result.best_health_pass} ({_pct(conv_result.best_health_lp)})",
            flush=True,
        )

    if dtp_result is not None:
        print("\nExp G: DTPContrastiveEncoder (BIO_ARN_DTP — layer-local credit, no cross-layer backprop)", flush=True)
        print(_format_pass_table(dtp_result), flush=True)
        print(
            f"Best LP pass: {dtp_result.best_lp_pass} ({_pct(dtp_result.best_lp)}) / "
            f"Best health pass: {dtp_result.best_health_pass} ({_pct(dtp_result.best_health_lp)})",
            flush=True,
        )

    print("\nPhase 4 reference (do not re-run):", flush=True)
    print(f"  ConvF1 (Hebbian control): {PHASE4_CONVF1_LP*100:.2f}%", flush=True)
    print(f"  SimCLR BACKPROP_UPPER_BOUND: {PHASE4_SIMCLR_LP*100:.2f}%", flush=True)
    print(f"  LocalInfoNCE BIO_ARN_LOCAL: {PHASE4_LOCAL_LP*100:.2f}%", flush=True)

    print("\nSUMMARY vs Phase 5 acceptance ladder:", flush=True)
    print("| Exp | Model | Best LP | Δ vs 38.00% | Verdict |", flush=True)
    print("|-----|-------|---------|-------------|---------|", flush=True)
    if conv_result is not None:
        delta = conv_result.best_lp - PHASE4_CONVF1_LP
        print(f"| A | ConvF1 control | {_pct(conv_result.best_lp)} | {delta:+.2f} pp | control |", flush=True)
    if dtp_result is not None:
        delta = dtp_result.best_lp - PHASE4_CONVF1_LP
        verdict = (
            "BREAKTHROUGH (>50%)" if dtp_result.best_lp > 0.50
            else "STRONG SUCCESS (>45%)" if dtp_result.best_lp > 0.45
            else "SUCCESS (>38%)" if dtp_result.best_lp > PHASE4_CONVF1_LP
            else "WEAK SIGNAL (>LocalInfoNCE)" if dtp_result.best_lp > PHASE4_LOCAL_LP
            else "FAIL (≤LocalInfoNCE)"
        )
        print(f"| G | DTP BIO_ARN_DTP | {_pct(dtp_result.best_lp)} | {delta:+.2f} pp | {verdict} |", flush=True)
    print(f"\nPhase 4 LocalInfoNCE floor: {PHASE4_LOCAL_LP*100:.2f}%", flush=True)
    print(f"Phase 4 ConvF1 ceiling:     {PHASE4_CONVF1_LP*100:.2f}%", flush=True)
    print(f"Phase 4 SimCLR upper bound: {PHASE4_SIMCLR_LP*100:.2f}%", flush=True)

    print("\nACCEPTANCE:", flush=True)
    print(f"  DTP LP > {PHASE4_CONVF1_LP*100:.2f}% (beats ConvF1): SUCCESS — credit assignment works without encoder backprop", flush=True)
    print(f"  DTP LP > 45.00%: STRONG SUCCESS", flush=True)
    print(f"  DTP LP > 50.00%: BREAKTHROUGH", flush=True)
    print(f"  DTP LP <= {PHASE4_LOCAL_LP*100:.2f}% (no better than LocalInfoNCE): FINAL NEGATIVE RESULT", flush=True)
    print(f"  Write final conclusion: Bio-ARN local mechanisms cannot approximate contrastive SSL", flush=True)

    # Write decision to squad inbox
    _write_decision(conv_result=conv_result, dtp_result=dtp_result)


def _write_decision(
    *,
    conv_result: ExperimentResult | None,
    dtp_result: ExperimentResult | None,
) -> None:
    inbox_dir = ROOT / ".squad" / "decisions" / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = inbox_dir / f"tank-local-ssl-5-{ts}.md"
    lines = [
        "### Phase 5 DTP credit-assignment result",
        "",
        f"**A ConvF1:** {_pct(conv_result.best_lp) if conv_result else 'N/A'}",
        f"**G DTPContrastive:** {_pct(dtp_result.best_lp) if dtp_result else 'N/A'}",
        f"**Phase4 reference — SimCLR:** {PHASE4_SIMCLR_LP*100:.2f}%",
        f"**Phase4 reference — LocalInfoNCE:** {PHASE4_LOCAL_LP*100:.2f}%",
    ]
    if dtp_result:
        verdict = (
            "BREAKTHROUGH" if dtp_result.best_lp > 0.50
            else "STRONG SUCCESS" if dtp_result.best_lp > 0.45
            else "SUCCESS" if dtp_result.best_lp > PHASE4_CONVF1_LP
            else "WEAK SIGNAL" if dtp_result.best_lp > PHASE4_LOCAL_LP
            else "FINAL NEGATIVE RESULT"
        )
        lines += ["", f"**Verdict:** {verdict}"]
        if verdict == "FINAL NEGATIVE RESULT":
            lines += [
                "",
                "**Final conclusion:**",
                "Bio-ARN pure/local Hebbian mechanisms learn useful low-level visual features,",
                "but CIFAR-10 class-useful representation learning requires a credit-assignment",
                "mechanism beyond local co-activation and local contrastive modulation.",
                "DTP (layer-local credit without cross-layer backprop) also fails to close the gap.",
                "The bottleneck is deep credit assignment across layers, not the contrastive objective.",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDecision written to {path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"Threads: {num_threads}", flush=True)
    if args.smoke:
        print("SMOKE TEST MODE: 1 pass, 5 probe epochs\n", flush=True)

    t0 = time.perf_counter()
    print("Loading CIFAR-10 ...", flush=True)
    train_imgs, train_lbls, test_imgs, test_lbls = _load_cifar10(
        Path(args.data_root), smoke_test=args.smoke
    )
    print(f"  {len(train_imgs):,} train, {len(test_imgs):,} test — {time.perf_counter()-t0:.1f}s\n", flush=True)

    probe_epochs = 5 if args.smoke else PROBE_EPOCHS

    print("=== EXPERIMENT A: ConvF1 control ===", flush=True)
    conv_result = _run_convf1_experiment(
        name="expA-conv-f1-control",
        label="Exp A: ConvF1 control",
        train_images=train_imgs,
        train_labels=train_lbls,
        test_images=test_imgs,
        test_labels=test_lbls,
        device=device,
        probe_epochs=probe_epochs,
        seed=args.seed,
        smoke_test=args.smoke,
    )

    print("\n=== EXPERIMENT G: DTPContrastiveEncoder BIO_ARN_DTP ===", flush=True)
    dtp_result = _run_dtp_experiment(
        name="expG-dtp-bio-arn-dtp",
        label="Exp G: DTPContrastiveEncoder (BIO_ARN_DTP — layer-local credit, no cross-layer backprop)",
        train_images=train_imgs,
        train_labels=train_lbls,
        test_images=test_imgs,
        test_labels=test_lbls,
        device=device,
        probe_epochs=probe_epochs,
        seed=args.seed,
        smoke_test=args.smoke,
    )

    _print_report(
        seed=args.seed,
        device=device,
        probe_epochs=probe_epochs,
        conv_result=conv_result,
        dtp_result=dtp_result,
    )


if __name__ == "__main__":
    main()
