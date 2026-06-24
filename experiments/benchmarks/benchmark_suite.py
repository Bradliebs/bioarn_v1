"""Bio-ARN 2.0 Benchmark Suite — Bio-ARN vs MLP vs Transformer on MNIST.

Run as:
    python experiments\\benchmarks\\benchmark_suite.py
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ── Path setup: repo root + experiments/ both importable ─────────────────────
_BENCH_DIR = Path(__file__).resolve().parent
_EXP_DIR = _BENCH_DIR.parent
_REPO_ROOT = _EXP_DIR.parent
for _p in (_REPO_ROOT, _EXP_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from bioarn.config import BioARNConfig, CCCConfig, MarginGateConfig
from bioarn.core.ccc import CCCPool
from mnist_poc import (
    LocalPrototypeBank,
    calibrate_abstention_threshold,
    collect_samples,
    evaluate_classifier,
    load_fashion_mnist,
    load_mnist,
    rotate_batch,
    run_streaming_training,
    subset_indices_by_label,
)

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── Run configuration ─────────────────────────────────────────────────────────

SEEDS: list[int] = [42, 123, 777]
TRAIN_SIZE: int = 5_000
TEST_SIZE: int = 1_000
CAL_SIZE: int = 500
FEW_SHOT_K: list[int] = [1, 5, 10]
DATA_ROOT: Path = _REPO_ROOT / "data"
OUTPUT_FILE: Path = _BENCH_DIR / "results.json"

# Bio-ARN benchmark config — smaller pool than the full mnist_poc config so
# that the streaming loop (which is Python-level per-sample) finishes quickly.
# Edit these to trade accuracy for speed or vice-versa.
_BENCH_POOL_SIZE: int = 25
_BENCH_F1_FEATURES: int = 128
_BENCH_CONCEPT_DIM: int = 64
_BENCH_F1_TOP_K: int = 16

# Continual-learning sub-scenario: keep small to avoid extra training overhead
_CL_PER_LABEL: int = 120   # training samples per class per phase
_CL_EPOCHS: int = 2        # epochs per phase for NN baselines


def _bench_config(seed: int) -> BioARNConfig:
    """Speed-optimised Bio-ARN config for the benchmark suite."""
    return BioARNConfig(
        seed=seed,
        ccc=CCCConfig(
            input_dim=784,
            concept_dim=_BENCH_CONCEPT_DIM,
            num_f1_features=_BENCH_F1_FEATURES,
            f1_top_k=_BENCH_F1_TOP_K,
            fast_lr=1.0,
            slow_lr=0.02,
            feedback_lr=0.02,
            max_pool_size=_BENCH_POOL_SIZE,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.50,
            theta_margin_lr=0.0005,
            theta_resonance=0.65,
        ),
    )

# ── Models ────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """784→256→ReLU→128→ReLU→10; trained with AdamW, 5 epochs."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PatchTransformer(nn.Module):
    """2-layer Transformer encoder: 28×28 → 16 patches of 7×7 → d_model=128, 4 heads."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(49, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 10)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        imgs = x.reshape(b, 28, 28)
        # 4×4 grid of non-overlapping 7×7 patches → (B, 16, 49)
        return imgs.unfold(1, 7, 7).unfold(2, 7, 7).contiguous().view(b, 16, 49)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)                       # (B, 16, 49)
        emb = self.patch_embed(patches) + self.pos_embed    # (B, 16, d_model)
        encoded = self.encoder(emb)                          # (B, 16, d_model)
        return self.head(encoded.mean(dim=1))               # (B, 10)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Dataset wrapper for flat-784 samples ─────────────────────────────────────

class _FlatDataset(Dataset):
    """Wraps a list of (flat-784 tensor, label) pairs as a PyTorch Dataset."""

    def __init__(self, samples: list[tuple[torch.Tensor, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        v, lbl = self.samples[i]
        return v.reshape(784), lbl


# ── NN training & evaluation ──────────────────────────────────────────────────

def train_nn(
    model: nn.Module,
    dataset: Dataset,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    label: str = "",
    verbose: bool = True,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = n = 0
        for inputs, targets in loader:
            opt.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * targets.shape[0]
            n += targets.shape[0]
        if verbose and label:
            print(f"    {label} epoch {epoch + 1}/{epochs}: loss={total_loss / max(n, 1):.4f}")


@torch.no_grad()
def eval_nn_acc(model: nn.Module, dataset: Dataset) -> float:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        correct += int((model(inputs).argmax(dim=1) == targets).sum())
        total += targets.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_flat_nn(model: nn.Module, test_samples: list[tuple[torch.Tensor, int]]) -> float:
    """Accuracy on a list of (flat-784 tensor, label) pairs."""
    model.eval()
    vecs = torch.stack([v.reshape(784) for v, _ in test_samples])
    labels = torch.tensor([lbl for _, lbl in test_samples])
    preds = model(vecs).argmax(dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def nn_scores(model: nn.Module, flat_vectors: list[torch.Tensor]) -> list[float]:
    """Max-softmax confidence scores for flat-784 tensors."""
    model.eval()
    batch = torch.stack(flat_vectors).reshape(-1, 784)
    return F.softmax(model(batch), dim=1).max(dim=1).values.tolist()


# ── Bio-ARN helpers ───────────────────────────────────────────────────────────

def train_bioarn(
    train_samples: list[tuple[torch.Tensor, int]],
    cal_samples: list[tuple[torch.Tensor, int]],
    config: BioARNConfig,
) -> tuple[CCCPool, LocalPrototypeBank, float, float]:
    """Train Bio-ARN streaming. Returns (pool, bank, threshold, mean_fired)."""
    pool = CCCPool(config.ccc, config.margin_gate)
    bank = LocalPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)
    _, fired_per_input, _ = run_streaming_training(pool, bank, train_samples)
    cal = calibrate_abstention_threshold(bank, cal_samples)
    mean_fired = sum(fired_per_input) / max(len(fired_per_input), 1)
    return pool, bank, cal.threshold, mean_fired


def bioarn_scores(bank: LocalPrototypeBank, vectors: list[torch.Tensor]) -> list[float]:
    """Raw cosine-similarity scores from the prototype bank (no threshold applied)."""
    return [bank.predict(v, threshold=-2.0).score for v in vectors]


def bioarn_param_count(pool: CCCPool, bank: LocalPrototypeBank, config: BioARNConfig) -> int:
    pool_n = (
        sum(p.numel() for p in pool.parameters())
        + sum(b.numel() for b in pool.buffers())
    )
    bank_n = len(bank.entries) * config.ccc.concept_dim
    return pool_n + bank_n


def bioarn_memory_mb(pool: CCCPool, bank: LocalPrototypeBank, config: BioARNConfig) -> float:
    return bioarn_param_count(pool, bank, config) * 4 / (1024 * 1024)


# ── OOD helpers ───────────────────────────────────────────────────────────────

def make_ood_vectors(
    id_samples: list[tuple[torch.Tensor, int]],
    fashion_dataset: Dataset | None,
    n: int = 500,
) -> dict[str, list[torch.Tensor]]:
    flat = torch.stack([v for v, _ in id_samples[:n]])
    ood: dict[str, list[torch.Tensor]] = {
        "random-noise": list(torch.rand_like(flat)),
        "rotated-90":   list(rotate_batch(flat, 90.0)),
        "inverted":     list(1.0 - flat),
    }
    if fashion_dataset is not None:
        fashion_samps = collect_samples(fashion_dataset, limit=n)
        ood["fashion-mnist"] = [v for v, _ in fashion_samps]
    return ood


def compute_auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    """AUROC via Wilcoxon rank-sum: P(score_ID > score_OOD). Higher = better."""
    n_pos, n_neg = len(id_scores), len(ood_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    combined = sorted(
        [(s, 1) for s in id_scores] + [(s, 0) for s in ood_scores],
        key=lambda x: x[0],
    )
    rank_sum = sum(r + 1 for r, (_, lbl) in enumerate(combined) if lbl == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_best_f1(id_scores: list[float], ood_scores: list[float]) -> tuple[float, float]:
    """Sweep thresholds; return (best_F1, threshold) where score<thr → OOD."""
    all_unique = sorted(set(id_scores + ood_scores))
    step = max(1, len(all_unique) // 100)
    best_f1, best_thr = 0.0, 0.0
    for thr in all_unique[::step]:
        tp = sum(1 for s in ood_scores if s < thr)
        fp = sum(1 for s in id_scores if s < thr)
        fn = sum(1 for s in ood_scores if s >= thr)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_f1, best_thr


def _find_threshold_for_fp(scores: list[float], target_fp: float) -> float:
    """Return score-threshold that yields ~target_fp fraction of scores below it."""
    s = sorted(scores)
    idx = max(0, min(int(target_fp * len(s)), len(s) - 1))
    return s[idx]


# ── FLOPs estimation ──────────────────────────────────────────────────────────

MLP_FLOPS_MAC: int = 784 * 256 + 256 * 128 + 128 * 10  # 234,752 MACs


def transformer_flops_mac(seq: int = 16, d: int = 128, n_heads: int = 4, layers: int = 2) -> int:
    patch_emb = seq * 49 * d
    attn_per = (3 * seq * d * d) + (seq * seq * d) + (seq * seq * d) + (seq * d * d)
    ffn_per = 2 * seq * d * (d * 2)
    head = d * 10
    return patch_emb + layers * (attn_per + ffn_per) + head


def bioarn_dense_flops_mac(config: BioARNConfig) -> int:
    per_ccc = (config.ccc.input_dim * config.ccc.num_f1_features
               + config.ccc.num_f1_features * config.ccc.concept_dim)
    return config.ccc.max_pool_size * per_ccc


def bioarn_active_flops_mac(config: BioARNConfig, mean_fired: float) -> int:
    per_ccc = (config.ccc.input_dim * config.ccc.num_f1_features
               + config.ccc.num_f1_features * config.ccc.concept_dim)
    return int(mean_fired * per_ccc)


# ── Latency & memory ──────────────────────────────────────────────────────────

def measure_nn_latency_ms(
    model: nn.Module, vectors: list[torch.Tensor], n_warmup: int = 5
) -> float:
    model.eval()
    batch = torch.stack(vectors).reshape(-1, 784)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(batch[: min(8, batch.shape[0])])
        t0 = time.perf_counter()
        model(batch)
    return (time.perf_counter() - t0) * 1000.0 / max(batch.shape[0], 1)


def measure_bioarn_latency_ms(
    bank: LocalPrototypeBank, vectors: list[torch.Tensor], n_warmup: int = 5
) -> float:
    for v in vectors[:n_warmup]:
        bank.predict(v, threshold=-2.0)
    t0 = time.perf_counter()
    for v in vectors:
        bank.predict(v, threshold=-2.0)
    return (time.perf_counter() - t0) * 1000.0 / max(len(vectors), 1)


def nn_memory_mb(model: nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return n * 4 / (1024 * 1024)


def process_memory_mb() -> float:
    if _HAS_PSUTIL:
        return _psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


# ── Scenario A: Standard classification ──────────────────────────────────────

def scenario_a(
    bank: LocalPrototypeBank,
    mlp: MLPBaseline,
    transformer: PatchTransformer,
    test_samples: list[tuple[torch.Tensor, int]],
    test_dataset_sub: Dataset,
    threshold: float,
    config: BioARNConfig,
    pool: CCCPool,
) -> dict[str, Any]:
    bioarn_acc = evaluate_classifier(bank, test_samples, threshold=threshold).overall_accuracy
    mlp_acc = eval_nn_acc(mlp, test_dataset_sub)
    txf_acc = eval_nn_acc(transformer, test_dataset_sub)

    return {
        "bioarn":      {"accuracy": bioarn_acc,  "params": bioarn_param_count(pool, bank, config), "flops_mac": bioarn_dense_flops_mac(config)},
        "mlp":         {"accuracy": mlp_acc,      "params": mlp.param_count(),         "flops_mac": MLP_FLOPS_MAC},
        "transformer": {"accuracy": txf_acc,      "params": transformer.param_count(), "flops_mac": transformer_flops_mac()},
    }


# ── Scenario B: Few-shot learning ─────────────────────────────────────────────

def scenario_b(
    train_dataset: Dataset,
    test_samples: list[tuple[torch.Tensor, int]],
) -> dict[str, Any]:
    results: dict[str, dict[str, float]] = {"bioarn": {}, "mlp": {}, "transformer": {}}

    for k in FEW_SHOT_K:
        few = collect_samples(train_dataset, per_label=k, labels=set(range(10)))
        if not few:
            for m in results:
                results[m][f"k{k}"] = 0.0
            continue

        # Bio-ARN: fresh prototype bank on k training examples
        few_bank = LocalPrototypeBank(max_entries_per_label=k + 2, recruit_threshold=0.60)
        for v, lbl in few:
            few_bank.observe(v, lbl)
        report = evaluate_classifier(few_bank, test_samples, threshold=0.0)
        results["bioarn"][f"k{k}"] = report.overall_accuracy

        # NN baselines: retrain from scratch on k examples
        few_ds = _FlatDataset(few)
        n_epochs = max(10, 100 // max(k * 10, 1))
        bs = max(4, len(few))

        mlp_k = MLPBaseline()
        train_nn(mlp_k, few_ds, epochs=n_epochs, batch_size=bs, lr=1e-3, verbose=False)
        results["mlp"][f"k{k}"] = evaluate_flat_nn(mlp_k, test_samples)

        txf_k = PatchTransformer()
        train_nn(txf_k, few_ds, epochs=n_epochs, batch_size=bs, lr=1e-3, verbose=False)
        results["transformer"][f"k{k}"] = evaluate_flat_nn(txf_k, test_samples)

    return results


# ── Scenario C: Continual learning ───────────────────────────────────────────

def scenario_c(
    train_dataset: Dataset,
    test_dataset: Dataset,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    per_label_train = _CL_PER_LABEL
    per_label_test = 100

    train_04 = collect_samples(train_dataset, labels=set(range(5)),    per_label=per_label_train)
    train_59 = collect_samples(train_dataset, labels=set(range(5, 10)), per_label=per_label_train)
    test_04  = collect_samples(test_dataset,  labels=set(range(5)),    per_label=per_label_test)

    # Bio-ARN: no forgetting by design (additive prototype memory)
    bank_c = LocalPrototypeBank(max_entries_per_label=5, recruit_threshold=0.80)
    for v, lbl in train_04:
        bank_c.observe(v, lbl)
    cal_c = calibrate_abstention_threshold(bank_c, train_04[:200])
    ba_before = evaluate_classifier(bank_c, test_04, threshold=cal_c.threshold).overall_accuracy
    for v, lbl in train_59:
        bank_c.observe(v, lbl)
    ba_after = evaluate_classifier(bank_c, test_04, threshold=cal_c.threshold).overall_accuracy

    idx_04   = subset_indices_by_label(train_dataset, range(5),     per_label=per_label_train)
    idx_59   = subset_indices_by_label(train_dataset, range(5, 10), per_label=per_label_train)
    t04_idx  = subset_indices_by_label(test_dataset,  range(5),     per_label=per_label_test)
    test_sub_04 = Subset(test_dataset, t04_idx)

    # MLP: sequential fine-tuning (catastrophic forgetting expected)
    mlp_c = MLPBaseline()
    train_nn(mlp_c, Subset(train_dataset, idx_04), epochs=_CL_EPOCHS, verbose=False)
    mlp_before = eval_nn_acc(mlp_c, test_sub_04)
    train_nn(mlp_c, Subset(train_dataset, idx_59), epochs=_CL_EPOCHS, verbose=False)
    mlp_after = eval_nn_acc(mlp_c, test_sub_04)

    # Transformer: sequential fine-tuning
    txf_c = PatchTransformer()
    train_nn(txf_c, Subset(train_dataset, idx_04), epochs=_CL_EPOCHS, verbose=False)
    txf_before = eval_nn_acc(txf_c, test_sub_04)
    train_nn(txf_c, Subset(train_dataset, idx_59), epochs=_CL_EPOCHS, verbose=False)
    txf_after = eval_nn_acc(txf_c, test_sub_04)

    def _pack(before: float, after: float) -> dict[str, float]:
        return {"acc_before": before, "acc_after": after, "forgetting": before - after}

    return {
        "bioarn":      _pack(ba_before, ba_after),
        "mlp":         _pack(mlp_before, mlp_after),
        "transformer": _pack(txf_before, txf_after),
    }


# ── Scenario D: OOD detection ─────────────────────────────────────────────────

def scenario_d(
    bank: LocalPrototypeBank,
    mlp: MLPBaseline,
    transformer: PatchTransformer,
    id_samples: list[tuple[torch.Tensor, int]],
    fashion_dataset: Dataset | None,
    threshold: float,
) -> dict[str, Any]:
    n_id = min(500, len(id_samples))
    id_vecs = [v for v, _ in id_samples[:n_id]]

    ood_sets = make_ood_vectors(id_samples, fashion_dataset, n=n_id)

    ba_id  = bioarn_scores(bank, id_vecs)
    mlp_id = nn_scores(mlp, id_vecs)
    txf_id = nn_scores(transformer, id_vecs)

    # Calibrate NN thresholds to match Bio-ARN false-positive rate on ID data
    ba_fp_rate = sum(1 for s in ba_id if s < threshold) / max(len(ba_id), 1)
    mlp_thr = _find_threshold_for_fp(mlp_id, ba_fp_rate)
    txf_thr = _find_threshold_for_fp(txf_id, ba_fp_rate)

    def _ood_metrics(
        id_sc: list[float],
        score_fn: Any,
        thr: float,
    ) -> dict[str, Any]:
        all_ood_sc: list[float] = []
        per_type: dict[str, float] = {}
        for ood_type, ood_vecs in ood_sets.items():
            ood_sc = score_fn(ood_vecs[:n_id])
            per_type[ood_type] = compute_auroc(id_sc, ood_sc)
            all_ood_sc.extend(ood_sc)
        overall_auroc = compute_auroc(id_sc, all_ood_sc)
        f1, _ = compute_best_f1(id_sc, all_ood_sc)
        abstention_ood = sum(1 for s in all_ood_sc if s < thr) / max(len(all_ood_sc), 1)
        return {
            "auroc": overall_auroc,
            "abstention_rate_ood": abstention_ood,
            "f1": f1,
            "auroc_per_type": per_type,
        }

    return {
        "bioarn":      _ood_metrics(ba_id,  lambda v: bioarn_scores(bank, v), threshold),
        "mlp":         _ood_metrics(mlp_id, lambda v: nn_scores(mlp, v),      mlp_thr),
        "transformer": _ood_metrics(txf_id, lambda v: nn_scores(transformer, v), txf_thr),
    }


# ── Scenario E: Energy efficiency ─────────────────────────────────────────────

def scenario_e(
    pool: CCCPool,
    bank: LocalPrototypeBank,
    mlp: MLPBaseline,
    transformer: PatchTransformer,
    test_vectors: list[torch.Tensor],
    mean_fired: float,
    config: BioARNConfig,
) -> dict[str, Any]:
    n = min(200, len(test_vectors))
    vecs = test_vectors[:n]

    ba_lat  = measure_bioarn_latency_ms(bank, vecs)
    mlp_lat = measure_nn_latency_ms(mlp, vecs)
    txf_lat = measure_nn_latency_ms(transformer, vecs)

    dense_ba  = bioarn_dense_flops_mac(config)
    active_ba = bioarn_active_flops_mac(config, mean_fired)
    sparsity_ba = mean_fired / max(config.ccc.max_pool_size, 1)

    mlp_mac = MLP_FLOPS_MAC
    txf_mac = transformer_flops_mac()

    return {
        "bioarn": {
            "dense_flops_mac":  dense_ba,
            "active_flops_mac": active_ba,
            "sparsity":         sparsity_ba,
            "energy_proxy_mac": active_ba,             # FLOPs × activation_density
            "flops_savings":    dense_ba / max(active_ba, 1),
            "latency_ms":       ba_lat,
            "params":           bioarn_param_count(pool, bank, config),
            "memory_mb":        bioarn_memory_mb(pool, bank, config),
        },
        "mlp": {
            "dense_flops_mac":  mlp_mac,
            "active_flops_mac": mlp_mac,
            "sparsity":         1.0,
            "energy_proxy_mac": mlp_mac,
            "flops_savings":    1.0,
            "latency_ms":       mlp_lat,
            "params":           mlp.param_count(),
            "memory_mb":        nn_memory_mb(mlp),
        },
        "transformer": {
            "dense_flops_mac":  txf_mac,
            "active_flops_mac": txf_mac,
            "sparsity":         1.0,
            "energy_proxy_mac": txf_mac,
            "flops_savings":    1.0,
            "latency_ms":       txf_lat,
            "params":           transformer.param_count(),
            "memory_mb":        nn_memory_mb(transformer),
        },
    }


# ── Per-seed runner ───────────────────────────────────────────────────────────

def run_seed(
    seed: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    fashion_dataset: Dataset | None,
) -> dict[str, Any]:
    print(f"\n{'=' * 62}")
    print(f"  Seed {seed}")
    print(f"{'=' * 62}")
    torch.manual_seed(seed)

    train_samples = collect_samples(train_dataset, limit=TRAIN_SIZE)
    cal_samples   = collect_samples(train_dataset, limit=CAL_SIZE, offset=TRAIN_SIZE)
    if len(cal_samples) < CAL_SIZE // 2:
        cal_samples = train_samples[-CAL_SIZE:]
    test_samples = collect_samples(test_dataset, limit=TEST_SIZE)
    test_vecs    = [v for v, _ in test_samples]

    # ---------- Bio-ARN ----------
    print("\n  Training Bio-ARN ...")
    config = _bench_config(seed)
    pool, bank, threshold, mean_fired = train_bioarn(train_samples, cal_samples, config)

    # ---------- MLP ----------
    print("  Training MLP ...")
    train_sub = Subset(train_dataset, list(range(min(TRAIN_SIZE, len(train_dataset)))))
    test_sub  = Subset(test_dataset,  list(range(min(TEST_SIZE,  len(test_dataset)))))
    mlp = MLPBaseline()
    train_nn(mlp, train_sub, epochs=5, label="MLP", verbose=True)

    # ---------- Transformer ----------
    print("  Training Transformer ...")
    transformer = PatchTransformer()
    train_nn(transformer, train_sub, epochs=5, label="TXF", verbose=True)

    # ---------- Scenarios ----------
    print("\n  Scenario A: standard classification ...")
    sa = scenario_a(bank, mlp, transformer, test_samples, test_sub, threshold, config, pool)

    print("  Scenario B: few-shot learning ...")
    sb = scenario_b(train_dataset, test_samples)

    print("  Scenario C: continual learning ...")
    sc = scenario_c(train_dataset, test_dataset, seed)

    print("  Scenario D: OOD detection ...")
    sd = scenario_d(bank, mlp, transformer, test_samples, fashion_dataset, threshold)

    print("  Scenario E: energy efficiency ...")
    se = scenario_e(pool, bank, mlp, transformer, test_vecs, mean_fired, config)

    return {
        "seed": seed,
        "bioarn_threshold": threshold,
        "mean_fired_cccs":  mean_fired,
        "scenario_a": sa,
        "scenario_b": sb,
        "scenario_c": sc,
        "scenario_d": sd,
        "scenario_e": se,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _ms(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var  = sum((x - mean) ** 2 for x in vals) / n
    return mean, var ** 0.5


def _pct(vals: list[float]) -> str:
    m, s = _ms(vals)
    return f"{m * 100:5.1f}±{s * 100:.1f}%"


def _flt(vals: list[float], d: int = 4) -> str:
    m, s = _ms(vals)
    return f"{m:.{d}f}±{s:.{d}f}"


def _int_k(v: int) -> str:
    """Format integer as k/M/B for readability."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return str(v)


def _collect(seed_results: list[dict], *keys: str) -> list[float]:
    """Navigate nested dicts by sequence of keys, collecting one float per seed."""
    out: list[float] = []
    for sr in seed_results:
        node: Any = sr
        for k in keys:
            if not isinstance(node, dict):
                break
            node = node.get(k, 0.0)
        out.append(float(node) if isinstance(node, (int, float)) else 0.0)
    return out


# ── Table printer ─────────────────────────────────────────────────────────────

def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    hdr = "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers)) + "|"
    body = [
        "|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(headers))) + "|"
        for row in rows
    ]
    lines = [f"\n{title}", sep, hdr, sep] + body + [sep]
    return "\n".join(lines)


def print_results_tables(seed_results: list[dict], config_summary: dict) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = platform.python_version()
    torch_ver = torch.__version__
    os_info = platform.system() + " " + platform.release()
    cpu = platform.processor() or platform.machine()

    W = 74
    border = "+" + "-" * W + "+"
    def _hdr_line(text: str) -> str:
        return "| " + text.ljust(W - 2) + " |"

    print(border)
    print(_hdr_line("Bio-ARN 2.0 Benchmark Suite"))
    print(_hdr_line(f"Timestamp : {now}"))
    print(_hdr_line(f"Platform  : {os_info} | Python {py_ver} | PyTorch {torch_ver}"))
    print(_hdr_line(f"CPU       : {cpu[:W-14]}"))
    print(_hdr_line(f"Seeds     : {SEEDS}"))
    print(_hdr_line(f"Dataset   : MNIST {TRAIN_SIZE:,} train / {TEST_SIZE:,} test | cal {CAL_SIZE}"))
    print(_hdr_line(f"Config    : MLP 784→256→128→10  |  TXF 2-layer d=128 nhead=4"))
    print(_hdr_line(f"Bio-ARN   : pool={_BENCH_POOL_SIZE} f1={_BENCH_F1_FEATURES} d={_BENCH_CONCEPT_DIM}  (speed-opt; edit _BENCH_* consts to resize)"))
    print(border)

    models = ["bioarn", "mlp", "transformer"]
    labels = {"bioarn": "Bio-ARN", "mlp": "MLP", "transformer": "Transformer"}

    # ── Scenario A ───────────────────────────────────────────────────────────
    rows_a = []
    for m in models:
        acc  = _pct(_collect(seed_results, "scenario_a", m, "accuracy"))
        par  = _int_k(int(_ms(_collect(seed_results, "scenario_a", m, "params"))[0]))
        flo  = _int_k(int(_ms(_collect(seed_results, "scenario_a", m, "flops_mac"))[0]))
        rows_a.append([labels[m], acc, par, flo])
    print(_table(
        "Scenario A — Standard Classification (MNIST subset)",
        ["Model", "Accuracy", "Params", "FLOPs (MACs)"],
        rows_a,
    ))

    # ── Scenario B ───────────────────────────────────────────────────────────
    rows_b = []
    for m in models:
        row = [labels[m]]
        for k in FEW_SHOT_K:
            row.append(_pct(_collect(seed_results, "scenario_b", m, f"k{k}")))
        rows_b.append(row)
    print(_table(
        "Scenario B — Few-Shot Accuracy (retrained baselines / Bio-ARN prototype bank)",
        ["Model"] + [f"k={k}" for k in FEW_SHOT_K],
        rows_b,
    ))

    # ── Scenario C ───────────────────────────────────────────────────────────
    rows_c = []
    for m in models:
        before  = _pct(_collect(seed_results, "scenario_c", m, "acc_before"))
        after   = _pct(_collect(seed_results, "scenario_c", m, "acc_after"))
        forget  = _pct(_collect(seed_results, "scenario_c", m, "forgetting"))
        rows_c.append([labels[m], before, after, forget])
    print(_table(
        "Scenario C — Continual Learning (train 0-4 → then 5-9; evaluate on 0-4)",
        ["Model", "Acc Before", "Acc After", "Forgetting ↓"],
        rows_c,
    ))

    # ── Scenario D ───────────────────────────────────────────────────────────
    rows_d = []
    for m in models:
        auroc  = _flt(_collect(seed_results, "scenario_d", m, "auroc"), 3)
        abst   = _pct(_collect(seed_results, "scenario_d", m, "abstention_rate_ood"))
        f1     = _flt(_collect(seed_results, "scenario_d", m, "f1"), 3)
        rows_d.append([labels[m], auroc, abst, f1])
    print(_table(
        "Scenario D — OOD Detection (noise / rotated-90 / inverted / Fashion-MNIST)",
        ["Model", "AUROC ↑", "OOD Abstention ↑", "Best-F1 ↑"],
        rows_d,
    ))

    # Per-type AUROC sub-table
    ood_types_all: set[str] = set()
    for sr in seed_results:
        for m in models:
            ood_types_all |= sr["scenario_d"][m]["auroc_per_type"].keys()
    ood_types = sorted(ood_types_all)
    if ood_types:
        rows_dtype = []
        for m in models:
            row = [labels[m]]
            for ot in ood_types:
                vals = [
                    sr["scenario_d"][m]["auroc_per_type"].get(ot, 0.0)
                    for sr in seed_results
                ]
                row.append(_flt(vals, 3))
            rows_dtype.append(row)
        print(_table(
            "Scenario D (per-type AUROC)",
            ["Model"] + ood_types,
            rows_dtype,
        ))

    # ── Scenario E ───────────────────────────────────────────────────────────
    rows_e = []
    for m in models:
        dense   = _int_k(int(_ms(_collect(seed_results, "scenario_e", m, "dense_flops_mac"))[0]))
        active  = _int_k(int(_ms(_collect(seed_results, "scenario_e", m, "active_flops_mac"))[0]))
        spar    = _flt(_collect(seed_results, "scenario_e", m, "sparsity"), 3)
        saving  = _flt(_collect(seed_results, "scenario_e", m, "flops_savings"), 1)
        lat     = _flt(_collect(seed_results, "scenario_e", m, "latency_ms"), 3) + "ms"
        mem     = f"{_ms(_collect(seed_results, 'scenario_e', m, 'memory_mb'))[0]:.1f}MB"
        rows_e.append([labels[m], dense, active, spar, saving, lat, mem])
    print(_table(
        "Scenario E — Energy Efficiency",
        ["Model", "Dense MACs", "Active MACs", "Sparsity", "Savings", "Latency", "Mem"],
        rows_e,
    ))

    # ── Final SUMMARY ────────────────────────────────────────────────────────
    _summarise(seed_results)


def _summarise(seed_results: list[dict]) -> None:
    """Print a SUMMARY line identifying which model wins each key metric."""
    models = ["bioarn", "mlp", "transformer"]
    labels = {"bioarn": "Bio-ARN", "mlp": "MLP", "transformer": "Transformer"}

    def _winner(path_tpl: tuple, higher_is_better: bool = True) -> str:
        scores = {m: _ms(_collect(seed_results, *[p.format(model=m) for p in path_tpl]))[0] for m in models}
        best = max(scores, key=scores.__getitem__) if higher_is_better else min(scores, key=scores.__getitem__)
        return labels[best]

    wins: dict[str, list[str]] = {lbl: [] for lbl in labels.values()}

    metrics = [
        ("Accuracy",         ("scenario_a", "{model}", "accuracy"),  True),
        ("Few-shot (k=1)",   ("scenario_b", "{model}", "k1"),        True),
        ("Few-shot (k=5)",   ("scenario_b", "{model}", "k5"),        True),
        ("Forgetting ↓",     ("scenario_c", "{model}", "forgetting"), False),
        ("OOD AUROC",        ("scenario_d", "{model}", "auroc"),     True),
        ("OOD F1",           ("scenario_d", "{model}", "f1"),        True),
        ("Active FLOPs ↓",   ("scenario_e", "{model}", "active_flops_mac"), False),
        ("Latency ↓",        ("scenario_e", "{model}", "latency_ms"), False),
        ("Memory ↓",         ("scenario_e", "{model}", "memory_mb"), False),
    ]

    for metric_name, path, higher in metrics:
        scores = {}
        for m in models:
            filled = tuple(p.replace("{model}", m) for p in path)
            scores[m] = _ms(_collect(seed_results, *filled))[0]
        best = max(scores, key=scores.__getitem__) if higher else min(scores, key=scores.__getitem__)
        wins[labels[best]].append(metric_name)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for lbl, won in wins.items():
        if won:
            print(f"  {lbl:<14}: wins on  {', '.join(won)}")
    print("=" * 74)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.perf_counter()

    print("Loading datasets ...")
    train_dataset = load_mnist(DATA_ROOT, train=True)
    test_dataset  = load_mnist(DATA_ROOT, train=False)

    fashion_dataset: Dataset | None = None
    try:
        fashion_dataset = load_fashion_mnist(DATA_ROOT, train=False)
        print("Fashion-MNIST loaded for OOD evaluation.")
    except Exception as exc:
        print(f"Fashion-MNIST unavailable ({exc}); OOD will use noise/rotation/invert only.")

    seed_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = run_seed(seed, train_dataset, test_dataset, fashion_dataset)
        seed_results.append(result)

    # ── Aggregate and display ─────────────────────────────────────────────────
    config_summary = {
        "seeds": SEEDS,
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "cal_size": CAL_SIZE,
        "few_shot_k": FEW_SHOT_K,
    }
    print_results_tables(seed_results, config_summary)

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.processor(),
        },
        "config": config_summary,
        "seed_results": seed_results,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as fh:
        json.dump(output, fh, indent=2, default=float)
    print(f"\nRaw results saved → {OUTPUT_FILE}")
    print(f"Total wall time   : {(time.perf_counter() - total_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
