# Hebbian Convolutional Feature Learning — CIFAR-10 Ceiling Analysis

**Status:** Closed (STOP decision)
**Date:** 2026-06-28
**Authors:** Bio-ARN Team (Sprint J + Final Validation)

## Summary

Pure unsupervised Hebbian convolutional learning reaches a ceiling of **~20–23% accuracy on CIFAR-10**, regardless of the competition mechanism used. This ceiling is a property of the unsupervised feature space, not the plasticity rule. Changing from hard top-K competition to SoftHebb (soft winner-take-all + BCM plasticity) does not materially improve performance.

This is a **negative-but-useful finding**: it characterizes the boundary of what unsupervised Hebbian conv learning can achieve on a standard benchmark, saving future effort on this specific axis of improvement.

## What Was Tried

### Sprint J Improvements (6 techniques)

| Technique | Description | Effect on CIFAR-10 |
|-----------|-------------|--------------------|
| **SoftHebb rule** | Soft-WTA with configurable γ sharpness + BCM per-filter thresholds | Neutral (matches baseline at γ=4) |
| **BCM plasticity** | Per-filter sliding threshold (θ decay=0.99) that stabilizes learning | No measurable benefit |
| **Data augmentation** | Random flip, crop, color jitter, cutout | Slight degradation at small scale |
| **ZCA whitening** | Decorrelates input channels (off-diagonal cov 5.33→0.001) | No accuracy improvement |
| **Deeper architecture** | 5-layer conv stack with max-pooling between stages | Underperforms 3-layer (19.25% vs 23.35%) |
| **Layer-wise training** | Train each layer independently, then freeze and stack | No improvement over joint training |

### Key Configurations Tested

All use `num_features=64`, `spatial_size=32`, `top_k=32`, `competitive_k=8`, `hebbian_lr=0.005`.

| Config | Competition | γ | Accuracy (NC) | Accuracy (LP) |
|--------|------------|---|---------------|---------------|
| Baseline (hard top-K) | Hard competitive | — | **20.34 ± 1.15%** | **20.38 ± 1.31%** |
| SoftHebb best (γ=4) | Soft-WTA + BCM | 4.0 | 19.38 ± 1.89% | 19.66 ± 1.88% |
| SoftHebb tight (γ=6) | Soft-WTA + BCM | 6.0 | 19.84 ± 1.78% | 20.20 ± 1.44% |
| SoftHebb soft (γ=2) | Soft-WTA + BCM | 2.0 | ~20.65%* | — |
| SoftHebb deep (5-layer) | Soft-WTA + BCM | 4.0 | ~19.25%* | — |

*Single-seed results from full-scale benchmark (10K samples, 10 passes).

## Decision-Grade Final Validation

**Protocol:** 5 random seeds × 3 configs × 5 training passes, evaluated with both nearest-centroid and linear probe (100 epochs SGD).

### Results (5 seeds: 0, 7, 42, 123, 2024)

**Nearest-Centroid:**

| Config | Seed 0 | Seed 7 | Seed 42 | Seed 123 | Seed 2024 | Mean ± Std |
|--------|--------|--------|---------|----------|-----------|------------|
| baseline | 18.40% | 20.20% | 21.20% | 20.90% | 21.00% | 20.34 ± 1.15% |
| softhebb_best | 16.30% | 19.00% | 20.80% | 20.90% | 19.90% | 19.38 ± 1.89% |
| softhebb_tight | 16.80% | 20.50% | 20.30% | 21.50% | 20.10% | 19.84 ± 1.78% |

**Linear Probe:**

| Config | Seed 0 | Seed 7 | Seed 42 | Seed 123 | Seed 2024 | Mean ± Std |
|--------|--------|--------|---------|----------|-----------|------------|
| baseline | 18.30% | 20.00% | 21.60% | 21.30% | 20.70% | 20.38 ± 1.31% |
| softhebb_best | 16.80% | 18.90% | 21.00% | 21.50% | 20.10% | 19.66 ± 1.88% |
| softhebb_tight | 18.10% | 20.00% | 21.40% | 21.70% | 19.80% | 20.20 ± 1.44% |

### Decision

**STOP** — No SoftHebb variant clears the >3 percentage point threshold.

- Δ(softhebb_best − baseline) = **−0.96 ± 0.81 pp** (nearest-centroid)
- Δ(softhebb_best − baseline) = **−0.72 ± 0.64 pp** (linear probe)
- Δ(softhebb_tight − baseline) = **−0.50 ± 0.92 pp** (nearest-centroid)
- Δ(softhebb_tight − baseline) = **−0.18 ± 0.47 pp** (linear probe)

SoftHebb slightly *underperforms* baseline on average. The competition mechanism is not the bottleneck.

## Why the Ceiling Exists

The ~20% ceiling is **not** caused by:
- ❌ The competition mechanism (hard vs. soft — both reach the same band)
- ❌ Input preprocessing (ZCA whitening didn't help)
- ❌ Data augmentation (didn't improve feature quality)
- ❌ Network depth (deeper networks performed worse)

The ceiling **is** caused by:
- ✅ **Feature space quality:** Unsupervised Hebbian updates learn local correlations (edges, textures) but cannot organize features into class-discriminative representations without some form of supervision signal
- ✅ **No error feedback:** Without any gradient or error signal flowing back, the network cannot adjust features to be more discriminative for downstream tasks
- ✅ **Single-pass local learning:** Each filter learns independently from local patches; there is no mechanism to coordinate filters toward complementary, class-relevant features

## What Would Break the Ceiling

Based on the literature and our experiments, breaking this ceiling would require fundamentally different approaches:

1. **Contrastive local learning** (e.g., SimCLR-style objectives adapted for local rules) — augmentation-based self-supervision could provide a richer learning signal while remaining backprop-free
2. **Error-driven local rules** (e.g., target propagation, difference target propagation) — some form of top-down error signal, even if not backpropagation, to guide feature organization
3. **Predictive coding** (hierarchical prediction error minimization) — already in the Bio-ARN architecture for lateral connections; extending to drive conv feature learning
4. **Accepting limited supervision** (e.g., contrastive Hebbian learning with labels) — a small amount of label information could dramatically improve feature discriminability

## Context: Literature Comparison

| Method | CIFAR-10 Accuracy | Supervision |
|--------|-------------------|-------------|
| Bio-ARN Hebbian (this work) | ~20% | None |
| SoftHebb MLP (Journé et al., ICLR 2023) | 54.5% (MLP) | None |
| Modern Hebbian CNNs (literature) | 64–76% | None* |
| Supervised CNN baseline | 90%+ | Full |

*Modern Hebbian CNN results typically use architectural tricks (batch norm, larger networks, careful initialization) and significantly more compute than our current setup.

The gap between our 20% and literature's 64–76% likely comes from: (a) network scale (we use 64 features, literature uses 256–512+), (b) architectural components (batch norm, residual connections), and (c) training budget (we use 5K–10K samples × 5–10 passes; literature uses full 50K × hundreds of epochs).

**Important:** These factors represent engineering investment, not fundamental algorithmic breakthroughs. The SoftHebb investigation showed that changing the plasticity rule alone — without scaling up — does not meaningfully move the needle.

## Files

| File | Description |
|------|-------------|
| `experiments/softhebb_final_validation.py` | Decision-grade validation script (5 seeds × 3 configs) |
| `experiments/fullscale_softhebb.py` | Full-scale benchmark (10K samples, 10 passes, 4 configs) |
| `experiments/softhebb_benchmark.py` | Original ablation (7 configs, 1K samples) |
| `experiments/linear_probe_benchmark.py` | Linear probe evaluation framework |
| `experiments/softhebb_hyperparam_sweep.py` | Hyperparameter grid search |
| `bioarn/core/conv_ccc.py` | ConvF1Layer with SoftHebb mode |
| `bioarn/config.py` | SoftHebb configuration (ConvCCCConfig) |

## Recommendation

The SoftHebb infrastructure (soft-WTA, BCM thresholds, layer-wise training) remains in the codebase and is well-tested. It is available for future use if:
- Significantly more compute becomes available for scaling experiments
- A complementary technique (contrastive learning, predictive coding) is combined with it
- The target benchmark changes to one where soft competition might matter more

For the CIFAR-10 Hebbian ceiling specifically, this investigation is **closed**.
