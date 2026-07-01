"""Local Self-Supervised Feature Learning — Bio-ARN Phase 3.

Tests whether changing the LEARNING SIGNAL (not more data/capacity) breaks the
37.6% LP ceiling confirmed by data_scaling.py.

Four experiments with identical evaluation (same probe, same dataset, same checkpoints):

  Exp A (control):     ConvF1Layer baseline                  expected ~37.6%
  Exp B (softhebb):    SoftHebbNet — Journé et al. WTA arch  target 45–65%
  Exp C (contrastive): LocalContrastiveEncoder — CLAPP-style  target 50%+
  Exp D (predictive):  LocalPredictiveEncoder — patch masking target 50%+

Key rule: all four use CIFAR-10 50K / 10K train/test, 100-epoch linear probe,
checkpoints at passes 1, 5, 10, 20, 30.  Any accuracy difference reflects the
learning rule, not dataset or evaluation changes.
"""

from __future__ import annotations

import argparse
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
from bioarn.core.softhebb_net import SoftHebbNet
from bioarn.core.local_contrastive import LocalContrastiveEncoder
from bioarn.core.local_predictive import LocalPredictiveEncoder

try:
    from torchvision import datasets
except ImportError as exc:
    raise RuntimeError("torchvision required for local_ssl.py") from exc

# ─── Architecture / training constants ────────────────────────────────────────

SEED = 42
NUM_CLASSES = 10

# ConvF1 (Exp A) — same config as data_scaling.py Exp 1
A_NUM_FEATURES = 512
A_HIDDEN_CHANNELS = (256, 512)
A_KERNEL_SIZES = (5, 3, 3)
A_SPATIAL_GRID = 4
A_SPATIAL_TOP_K = 8
A_TOP_K = 256
A_COMPETITIVE_K = 64

# SoftHebbNet (Exp B, C, D) — compact Journé variant
SOFT_CHANNELS = (96, 384, 512)
SOFT_KERNELS = (5, 3, 3)
SOFT_GAMMA = 10.0
SOFT_ETA = 0.01

# Evaluation
HEBBIAN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 256
PROBE_EPOCHS = 100
PROBE_LR = 0.01
PROBE_MOMENTUM = 0.9
CHECKPOINTS = (1, 5, 10, 20, 30)

PASSES_A = 30
PASSES_B = 30
PASSES_C = 30
PASSES_D = 30

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


# ─── Models ───────────────────────────────────────────────────────────────────

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
        in_channels=3, num_features=A_NUM_FEATURES, spatial_size=32,
        top_k=A_TOP_K, spatial_grid=A_SPATIAL_GRID, num_layers=3,
        hidden_channels=A_HIDDEN_CHANNELS, kernel_sizes=A_KERNEL_SIZES,
        spatial_top_k=A_SPATIAL_TOP_K, competitive_k=A_COMPETITIVE_K,
        hebbian_lr=0.005, hebbian_batch_size=HEBBIAN_BATCH_SIZE,
        weight_norm_target=1.0, enable_local_contrast_norm=True,
        hebbian_oja_decay=0.05, filter_decorrelation=0.02,
    )


def _build_softhebb() -> SoftHebbNet:
    return SoftHebbNet(channels=SOFT_CHANNELS, kernel_sizes=SOFT_KERNELS, gamma=SOFT_GAMMA, eta=SOFT_ETA)


def _build_contrastive() -> LocalContrastiveEncoder:
    return LocalContrastiveEncoder(channels=SOFT_CHANNELS, kernel_sizes=SOFT_KERNELS, gamma=SOFT_GAMMA, eta=SOFT_ETA)


def _build_predictive() -> LocalPredictiveEncoder:
    return LocalPredictiveEncoder(channels=SOFT_CHANNELS, kernel_sizes=SOFT_KERNELS, gamma=SOFT_GAMMA, eta=SOFT_ETA)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pp"


# ─── Data ─────────────────────────────────────────────────────────────────────

def _load_cifar10(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    train_images = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).float() / 255.0
    test_images = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).float() / 255.0
    train_labels = torch.tensor(train_ds.targets, dtype=torch.long)
    test_labels = torch.tensor(test_ds.targets, dtype=torch.long)
    return train_images, train_labels, test_images, test_labels


# ─── Augmentation ─────────────────────────────────────────────────────────────

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


# ─── Training pass ────────────────────────────────────────────────────────────

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
    feature_dim = int(model.output_dim)
    top_k = min(sparse_k, feature_dim)
    all_indices = torch.empty((total, top_k), dtype=torch.int32)
    all_values = torch.empty((total, top_k), dtype=torch.float32)
    cursor = 0
    for start in range(0, total, batch_size):
        batch = images[start : start + batch_size].to(device=device, dtype=torch.float32)
        dense = model(batch).cpu().float()
        vals, idxs = torch.topk(dense, k=top_k, dim=1)
        bs = dense.shape[0]
        all_indices[cursor : cursor + bs] = idxs.to(torch.int32)
        all_values[cursor : cursor + bs] = vals
        cursor += bs
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
    train_imgs: torch.Tensor, train_lbls: torch.Tensor,
    test_imgs: torch.Tensor, test_lbls: torch.Tensor,
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


# ─── Experiment runner ────────────────────────────────────────────────────────

def _run_experiment(
    *,
    name: str,
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
) -> ExperimentResult:
    model = model.to(device)
    rng = random.Random(seed)
    metrics: list[PassMetrics] = []
    _actual_passes = 1 if smoke_test else planned_passes
    _actual_probe_epochs = 2 if smoke_test else probe_epochs

    n = train_images.shape[0]
    print(f"[{name}] train={n//1000}K device={device} passes={_actual_passes} aug={use_augmentation}", flush=True)

    for pass_num in range(1, _actual_passes + 1):
        t0 = time.perf_counter()
        _run_hebbian_pass(model, train_images, batch_size=HEBBIAN_BATCH_SIZE, device=device, augment=use_augmentation, rng=rng)
        t_hebb = time.perf_counter() - t0

        if pass_num in checkpoints or pass_num == _actual_passes:
            t_eval = time.perf_counter()
            nc, lp = _evaluate(
                model, train_images, train_labels, test_images, test_labels,
                sparse_k=sparse_k, device=device, probe_epochs=_actual_probe_epochs, seed=seed,
            )
            t_eval = time.perf_counter() - t_eval
            metrics.append(PassMetrics(pass_num=pass_num, nearest_centroid=nc, linear_probe=lp))
            print(
                f"[{name}] pass {pass_num}/{_actual_passes}  "
                f"hebbian {t_hebb:.1f}s  "
                f"nc={nc*100:.2f}%  lp={lp*100:.2f}%  eval {t_eval/60:.1f}m",
                flush=True,
            )
        else:
            print(f"[{name}] pass {pass_num}/{_actual_passes}  hebbian {t_hebb:.1f}s", flush=True)

    return ExperimentResult(name=name, planned_passes=planned_passes, executed_passes=_actual_passes, pass_metrics=metrics)


# ─── Reporting ────────────────────────────────────────────────────────────────

def _format_table(result: ExperimentResult, *, checkpoints: tuple[int, ...]) -> str:
    by_pass = result.metrics_by_pass
    display = sorted({*checkpoints, result.executed_passes})
    lines = ["| Pass | Nearest-Centroid | Linear Probe |", "|------|-----------------|--------------|"]
    for cp in display:
        m = by_pass.get(cp)
        nc = _pct(None if m is None else m.nearest_centroid)
        lp = _pct(None if m is None else m.linear_probe)
        lines.append(f"| {cp:<4} | {nc:<15} | {lp:<12} |")
    return "\n".join(lines)


def _print_report(
    results: dict[str, ExperimentResult],
    labels: dict[str, str],
    checkpoints: tuple[int, ...],
    device: torch.device,
) -> None:
    print("\n================================================================", flush=True)
    print("LOCAL SSL — RESULTS", flush=True)
    print("================================================================", flush=True)
    print(f"Seed: {SEED} | Device: {device} | Probe epochs: {PROBE_EPOCHS}", flush=True)
    for key, label in labels.items():
        r = results.get(key)
        if r is None:
            continue
        print(f"\n--- {label} ---", flush=True)
        print(_format_table(r, checkpoints=checkpoints), flush=True)
        print(f"Best: {_pct(r.best_linear_probe)} LP (pass {r.best_pass})", flush=True)

    print("\n================================================================", flush=True)
    print("LOCAL SSL SUMMARY", flush=True)
    print("================================================================", flush=True)
    print(f"| Experiment | Rule | Best LP | Δ vs {_pct(PREVIOUS_CEILING)} ceiling |", flush=True)
    print("|-----------|------|---------|-------------------------|", flush=True)
    print(f"| Previous ceiling | pure Hebbian ConvF1 (256 feat, 50K) | {_pct(PREVIOUS_CEILING)} | — |", flush=True)
    for key, label in labels.items():
        r = results.get(key)
        if r is None:
            continue
        delta = r.best_linear_probe - PREVIOUS_CEILING
        print(f"| {key} | {label} | {_pct(r.best_linear_probe)} | {_pp(delta)} |", flush=True)

    best = max((r.best_linear_probe for r in results.values()), default=0.0)
    print(f"\nNew ceiling: {_pct(best)}", flush=True)


def _write_decision(results: dict[str, ExperimentResult]) -> Path:
    ts = time.strftime("%Y-%m-%d")
    best = max((r.best_linear_probe for r in results.values()), default=0.0)
    path = Path(".squad") / "decisions" / "inbox" / "tank-local-ssl.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"### {ts}: Local SSL Results",
        "**By:** Tank (Core Neural)",
        f"**What:** Local self-supervised feature learning reached {best*100:.1f}% LP on CIFAR-10",
        f"**Previous ceiling:** {PREVIOUS_CEILING*100:.1f}% (pure Hebbian ConvF1)",
    ]
    for key, r in results.items():
        delta = r.best_linear_probe - PREVIOUS_CEILING
        lines.append(f"**{key}:** {_pct(r.best_linear_probe)} ({_pp(delta)} vs prev)")
    lines.append(f"**New ceiling:** {best*100:.1f}%")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--probe-epochs", type=int, default=PROBE_EPOCHS)
    p.add_argument("--passes-a", type=int, default=PASSES_A)
    p.add_argument("--passes-b", type=int, default=PASSES_B)
    p.add_argument("--passes-c", type=int, default=PASSES_C)
    p.add_argument("--passes-d", type=int, default=PASSES_D)
    p.add_argument("--threads", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    p.add_argument("--skip-a", action="store_true", help="Skip Exp A (ConvF1 control)")
    p.add_argument("--skip-b", action="store_true", help="Skip Exp B (SoftHebbNet)")
    p.add_argument("--skip-c", action="store_true", help="Skip Exp C (LocalContrastive)")
    p.add_argument("--skip-d", action="store_true", help="Skip Exp D (LocalPredictive)")
    p.add_argument("--smoke-test", action="store_true", help="1 pass + 2 probe epochs (quick sanity check)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})", flush=True)
    print(f"Threads: {torch.get_num_threads()}", flush=True)

    checkpoints = CHECKPOINTS
    if args.smoke_test:
        checkpoints = (1,)
        print("SMOKE TEST MODE: 1 pass, 2 probe epochs", flush=True)

    # ── Load CIFAR-10 ─────────────────────────────────────────────────────────
    print("\nLoading CIFAR-10 ...", flush=True)
    t0 = time.perf_counter()
    train_imgs, train_lbls, test_imgs, test_lbls = _load_cifar10(args.data_dir)
    print(f"  {train_imgs.shape[0]//1000}K train, {test_imgs.shape[0]//1000}K test — {time.perf_counter()-t0:.1f}s\n", flush=True)

    results: dict[str, ExperimentResult] = {}
    labels: dict[str, str] = {}

    common = dict(
        train_images=train_imgs, train_labels=train_lbls,
        test_images=test_imgs, test_labels=test_lbls,
        checkpoints=checkpoints, device=device,
        probe_epochs=args.probe_epochs, seed=args.seed,
        smoke_test=args.smoke_test,
    )

    # ── Exp A: ConvF1 baseline (control) ─────────────────────────────────────
    if not args.skip_a:
        print("=== EXPERIMENT A: ConvF1Layer baseline (512 feat, aug) ===", flush=True)
        results["A"] = _run_experiment(
            name="expA-conv-f1-baseline",
            model=_build_conv_f1(),
            sparse_k=A_TOP_K,
            planned_passes=args.passes_a,
            use_augmentation=True,
            **common,
        )
        labels["A"] = "Exp A: ConvF1 baseline (512 feat, aug)"
    else:
        print("(Skipping Exp A)", flush=True)

    # ── Exp B: SoftHebbNet ────────────────────────────────────────────────────
    if not args.skip_b:
        print("\n=== EXPERIMENT B: SoftHebbNet (96→384→512, SoftWTA) ===", flush=True)
        results["B"] = _run_experiment(
            name="expB-softhebb-net",
            model=_build_softhebb(),
            sparse_k=256,
            planned_passes=args.passes_b,
            use_augmentation=True,
            **common,
        )
        labels["B"] = "Exp B: SoftHebbNet (Journé-style WTA, aug)"
    else:
        print("(Skipping Exp B)", flush=True)

    # ── Exp C: Local Contrastive ──────────────────────────────────────────────
    if not args.skip_c:
        print("\n=== EXPERIMENT C: LocalContrastiveEncoder (CLAPP-style) ===", flush=True)
        results["C"] = _run_experiment(
            name="expC-local-contrastive",
            model=_build_contrastive(),
            sparse_k=256,
            planned_passes=args.passes_c,
            use_augmentation=False,  # contrastive encoder generates views internally
            **common,
        )
        labels["C"] = "Exp C: LocalContrastiveEncoder (view-consistency modulation)"
    else:
        print("(Skipping Exp C)", flush=True)

    # ── Exp D: Local Predictive ───────────────────────────────────────────────
    if not args.skip_d:
        print("\n=== EXPERIMENT D: LocalPredictiveEncoder (patch masking) ===", flush=True)
        results["D"] = _run_experiment(
            name="expD-local-predictive",
            model=_build_predictive(),
            sparse_k=256,
            planned_passes=args.passes_d,
            use_augmentation=False,  # predictive encoder applies masking internally
            **common,
        )
        labels["D"] = "Exp D: LocalPredictiveEncoder (masked patch prediction)"
    else:
        print("(Skipping Exp D)", flush=True)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_report(results, labels, checkpoints, device)

    decision_path = _write_decision(results)
    print(f"\nDecision written to {decision_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
