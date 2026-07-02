# Hebbian Convolutional Feature Learning — CIFAR-10 Ceiling Analysis

**Status:** Phase 4 complete (2026-07-02) — contrastive objective valid, local approximation failed
**Date:** 2026-06-29
**Authors:** Bio-ARN Team (Sprint J + Final Validation + Bias Audit + Progressive Scaling)

## Summary

Pure unsupervised Hebbian convolutional learning reaches a **confirmed ceiling of ~37.6% accuracy on CIFAR-10** — up from the original ~20% estimate (+17.6 pp). All four scaling axes have now been tested and closed.

**Final results:**
- **Capacity** (64→256 features): +7.4 pp
- **Full data** (5K→50K training images): **+10.2 pp** — the largest single gain
- **Bio-plausible divisive normalization**: **−5.6 pp** — hurts
- **More data + 512 features + augmentation**: **+0.1 pp** — flat; ceiling is real

The ceiling is **not** a data problem. Adding 3.5× more images (50K → 173K via CIFAR-100 + SVHN), doubling features to 512, and applying augmentation all leave the result in the 37.3–37.7% band. **The bottleneck is the learning rule:** pure Hebbian updates cannot organise features into class-discriminative representations without a contrastive or error-driven signal.

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
| Bio-ARN Hebbian (this work, initial) | ~20% | None |
| Bio-ARN Hebbian (this work, combined scale, 5K) | 27.4% | None |
| Bio-ARN Hebbian (this work, full data 50K) | **37.6%** | None |
| Bio-ARN Hebbian (this work, data scaling — ceiling confirmed) | **37.7%** (best of 4 exps) | None |
| Bio-ARN ConvF1 Phase 3 control (512 feat, 30 pass, aug) | **38.87%** | None |
| Bio-ARN SoftHebbNet γ=10 (this work, Phase 3) | 20.18% peak (pass 10), collapses | None |
| Bio-ARN LocalContrastive γ=10 (this work, Phase 3) | 21.05% peak (pass 10), collapses | None |
| SoftHebb MLP (Journé et al., ICLR 2023) | 54.5% (MLP) | None |
| Modern Hebbian CNNs (literature) | 64–76% | None* |
| Supervised CNN baseline | 90%+ | Full |

*Modern Hebbian CNN results typically use architectural tricks (batch norm, larger networks, careful initialization) and significantly more compute than our current setup.

The gap between our 27% and literature's 64–76% likely comes from: (a) network scale (we use 256 features max, literature uses 512+), (b) architectural components (batch norm, residual connections), and (c) training budget (we use 5K samples × 50 passes; literature uses full 50K × hundreds of epochs).

**Important:** The bias audit confirmed these are real confounds, not theoretical. Scaling capacity and training duration produced measurable gains (+5.3 pp and +7.0 pp respectively). Further scaling is likely to yield further gains — the curve was still rising at 50 passes.

## Survivorship Bias Audit (Post-Hoc Correction)

After the initial STOP decision, we identified that our experimental design suffered from survivorship bias: we varied only the competition mechanism while holding capacity, training duration, evaluation method, and architectural framework constant.

### Biases Tested

| Test | Question | Result | Bias Found? |
|------|----------|--------|-------------|
| **Capacity** (64→128→256) | Were we underfitting? | +5.3 pp at 256 features | ⚠️ **YES** |
| **Duration** (5→50 passes) | Did we stop too early? | +7.0 pp at 50 passes | ⚠️ **YES** |
| **Evaluation** (NC vs LP vs MLP) | Are features better than measured? | MLP ≈ LP | ✅ No |
| **Pure Hebbian** (stripped CCC) | Is the framework hurting? | CCC ≈ Pure | ✅ No |

### Capacity Check Results (10 passes, seed=42)

| Features | Output Dim | Nearest-Centroid | Linear Probe |
|----------|-----------|-----------------|-------------|
| 64       | ~5K       | ~19.5%          | ~19.8%      |
| 128      | ~12K      | ~22.1%          | ~23.4%      |
| 256      | ~25K      | ~24.0%          | ~25.1%      |

**Capacity effect:** Δ(256 vs 64) = **+5.3 pp** (linear probe). The 64-feature "ceiling" was actually a capacity floor.

### Duration Check Results (64 features, seed=42)

| Pass | Nearest-Centroid |
|------|-----------------|
| 1    | ~14.2%          |
| 5    | ~17.8%          |
| 10   | ~19.7%          |
| 20   | ~23.1%          |
| 30   | ~25.0%          |
| 40   | ~26.1%          |
| 50   | ~26.7%          |

**Duration effect:** Accuracy was still climbing at pass 50 (+7.0 pp over pass 10). We dramatically undertrained.

### What This Means

1. **The competition mechanism conclusion stands** — SoftHebb ≈ hard top-K regardless of scale
2. **The "~20% ceiling" was premature** — it was an artifact of testing at one capacity and one duration
3. **The revised ceiling is at least 27%** and likely higher with combined scaling (256 features × 50+ passes, which was not tested due to runtime constraints)
4. **The gap to literature (64-76%) is narrower than we thought** — and the remaining gap is likely explained by: more features (512+), more data (50K), more passes (hundreds), and batch norm

### Implications for Next Steps

The survivorship bias audit opens two concrete paths:

**Path A — Scale the validated approach:**
- Run 256 features × 50 passes (not tested due to time) — likely >30%
- Run with full 50K CIFAR-10 training set — more data = better features
- Add batch normalization (bio-plausible implementations exist)

**Path B — Accept the diminishing returns:**
- Each doubling of scale (features, duration) yields ~3-5 pp
- Getting to literature's 64-76% would require 512+ features, 200+ passes, full dataset, and batch norm
- That's valid engineering but the fundamental question (Hebbian vs competition mechanism) is answered

## Progressive Scaling Experiments (2026-06-29) — COMPLETE

All three experiments ran on GPU (NVIDIA GeForce RTX 3070, CUDA 12.4) using `experiments/hebbian_scaling.py`. Architecture: 3-layer ConvF1, 256 features, spatial_grid=4, top_k=128, competitive_k=32, hebbian_lr=0.005. Seed: 42.

### Experiment 1: Combined Scale — COMPLETE

**Config:** 256 features × 50 passes × 5,000 training samples × 1,000 test samples

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 18.10%          | 18.80%       |
| 5    | 22.20%          | 21.20%       |
| 10   | 23.00%          | 22.50%       |
| 20   | 25.50%          | 25.00%       |
| 30   | **28.40%**      | **27.40%**   |
| 40   | 27.10%          | 27.30%       |
| 50   | 27.60%          | 26.80%       |

**Best:** 28.40% NC / **27.4% LP** (pass 30) — +7.4 pp over 20% baseline. Peaks at pass 30–40, then saturates at 5K scale.

### Experiment 2: Full Dataset — COMPLETE

**Config:** 256 features × 50 passes × 50,000 training samples × 10,000 test samples

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 24.76%          | 28.29%       |
| 5    | 28.35%          | 33.54%       |
| 10   | **29.88%**      | 36.12%       |
| 20   | 28.69%          | 36.30%       |
| 30   | 27.73%          | 36.87%       |
| 40   | 28.24%          | **37.60%**   |
| 50   | 28.39%          | 37.59%       |

**Best:** 29.88% NC (pass 10) / **37.6% LP (pass 40)** — **+10.2 pp over Exp 1**, biggest single gain of the whole investigation.

**Key observations:** NC peaks at pass 10; LP keeps climbing to pass 40 then flattens. At pass 50 LP is essentially identical to pass 40 — curve is saturating but not done. Data scale is a bigger bottleneck than capacity or duration.

### Experiment 3: Bio-plausible Divisive Normalization — COMPLETE

**Config:** Same as Exp 2 + `DivisiveNormalization(σ=0.1, neighborhood=5)` between conv layers

| Pass | NC (no norm) | LP (no norm) | NC (div norm) | LP (div norm) |
|------|-------------|-------------|--------------|--------------|
| 1    | 24.76%      | 28.29%      | **31.01%**   | 32.04%       |
| 5    | 28.35%      | 33.54%      | 27.52%       | 29.64%       |
| 10   | **29.88%**  | 36.12%      | 27.53%       | 29.50%       |
| 20   | 28.69%      | 36.30%      | 29.68%       | 30.96%       |
| 30   | 27.73%      | 36.87%      | 29.83%       | 31.15%       |
| 40   | 28.24%      | **37.60%**  | 29.73%       | 30.88%       |
| 50   | 28.39%      | 37.59%      | 29.57%       | 30.74%       |

**Best with div norm:** 32.0% LP (pass 1). **Δ vs Exp 2: −5.6 pp.** Divisive normalization actively hurts.

**Interpretation:** Suppression removes useful feature variance along with noise. This is the same pattern as the competition mechanism — local normalisation/competition mechanisms are not the bottleneck. The representations are informative; what's missing is scale.

### Progressive Scaling Summary

| Experiment | Config | Best LP | Δ vs Previous |
|------------|--------|---------|--------------|
| Bias audit baseline | 64 feat, 10 pass, 5K | ~20.0% | — |
| Exp 1: Combined scale | 256 feat, 50 pass, 5K | 27.4% | +7.4 pp |
| Exp 2: Full data | 256 feat, 50 pass, 50K | **37.6%** | **+10.2 pp** |
| Exp 3: Divisive norm | 256 feat, 50 pass, 50K + norm | 32.0% | −5.6 pp |

**Total improvement over original baseline: +17.6 pp (20% → 37.6%)**

## Data Scaling Experiments (2026-06-30) — IN PROGRESS

Testing whether **more data + 512 features** breaks through the 37.6% LP ceiling. The Hebbian layer is fully unsupervised — labels are never used during training, so images from any natural source are valid. Linear probe evaluation always uses CIFAR-10 train/test labels.

**Architecture:** 3-layer ConvF1, **512 features**, spatial_grid=4, top_k=256 (50% sparse), competitive_k=64, hebbian_lr=0.005. Seed: 42. GPU: NVIDIA GeForce RTX 3070.

**Datasets (free via torchvision):**
- CIFAR-10 train: 50K × 32×32 (eval labels always from here)
- CIFAR-100 train: 50K × 32×32 (unlabeled Hebbian training)
- SVHN train: ~73K × 32×32 (street-view house numbers, unlabeled)

**Online augmentation (per batch):** random horizontal flip + pad4/crop + brightness/contrast jitter.

### Experiment 1: aug-c10 — 50K + aug, 512 feat, 50 passes

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 25.58%          | 28.75%       |
| 5    | 28.47%          | 34.05%       |
| 10   | 30.18%          | 36.82%       |
| 20   | 29.72%          | 37.29%       |
| 25   | 28.30%          | 37.14%       |
| 30   | 28.27%          | 37.23%       |
| 40   | 27.73%          | 37.27%       |
| 50   | 27.85%          | **37.68%**   |

**Best: 37.68% LP (pass 50)** — just barely edges the prior ceiling. NC peaks early (pass 10) then declines as LP keeps climbing.

### Experiment 2: multi-100k — C10+C100 100K, no aug, 30 passes

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 26.54%          | 31.11%       |
| 5    | 30.81%          | 36.82%       |
| 10   | 29.31%          | **37.35%**   |
| 20   | 27.43%          | 37.15%       |
| 25   | 27.48%          | 37.07%       |
| 30   | 27.44%          | 37.06%       |

**Best: 37.35% LP (pass 10)** — peaks earlier than Exp 1, then saturates. 2× data with no augmentation matches the ceiling but doesn't beat it.

### Experiment 3: multi-173k — C10+C100+SVHN ~173K, no aug, 25 passes

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 27.08%          | 33.25%       |
| 5    | 27.48%          | 35.95%       |
| 10   | 27.59%          | **37.49%**   |
| 20   | 28.22%          | 36.64%       |
| 25   | 27.49%          | 36.72%       |

**Best: 37.49% LP (pass 10)** — 3.5× data, same result as 2× data. LP peaks at pass 10 then dips slightly. SVHN (different domain) does not add useful signal.

### Experiment 4: multi-173k-aug — C10+C100+SVHN ~173K + aug, 20 passes

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 27.47%          | 33.44%       |
| 5    | 28.67%          | **37.31%**   |
| 10   | 28.00%          | 37.02%       |
| 20   | 27.68%          | 36.83%       |

**Best: 37.31% LP (pass 5)** — augmentation + more data peaks fastest but no higher. Adding augmentation to a larger heterogeneous dataset does not compound the gains.

### Data Scaling Summary

| Experiment | Data | Aug | Features | Passes | Best LP | Δ vs 37.6% |
|------------|------|-----|----------|--------|---------|------------|
| Previous ceiling | CIFAR-10 50K | ✗ | 256 | 50 | 37.6% | — |
| Exp 1: aug-c10 | CIFAR-10 50K | ✓ | 512 | 50 | **37.68%** | +0.1 pp |
| Exp 2: multi-100k | C10+C100 100K | ✗ | 512 | 30 | 37.35% | −0.3 pp |
| Exp 3: multi-173k | C10+C100+SVHN 173K | ✗ | 512 | 25 | 37.49% | −0.1 pp |
| Exp 4: multi-173k-aug | C10+C100+SVHN 173K | ✓ | 512 | 20 | 37.31% | −0.3 pp |

**All four experiments cluster within 37.3–37.7% — a spread of just 0.4 pp across 3.5× data range.** The ceiling is real and is not a data problem.

**Script:** `experiments/data_scaling.py` | **Commit:** `4787808`

## Local Self-Supervised Feature Learning (2026-07-01) — COMPLETE (Phase 3)

**Label: Negative result, but not a valid falsification of local SSL.**
**Primary failure: representation collapse (γ=10) + evaluation mismatch (spatial 8192-dim vs global 512-dim).**
**Next move: Phase 3b repair, not new architecture exploration.**

Phase 3 tests whether **changing the learning signal** breaks the 37.6% LP ceiling confirmed by Phases 1–2. All four experiments use identical evaluation: CIFAR-10 50K train / 10K test, 100-epoch linear probe, checkpoints at passes 1/5/10/20/30.

**Hypothesis:** The ceiling is the learning rule, not data volume. Adding contrastive or predictive objectives should break the ceiling even with the same data.

**Script:** `experiments/local_ssl.py` | **Commit:** `1f5f3d1`

### Experiment A: ConvF1Layer baseline (512 feat, aug) — control

Same architecture and data as Phase 2 Exp 1. Expected ~37.6%.

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 25.63%          | 28.78%       |
| 5    | 28.15%          | 33.97%       |
| 10   | 30.34%          | 37.00%       |
| 20   | 28.34%          | 37.88%       |
| 30   | 28.56%          | **38.87%**   |

**Best:** 38.87% LP (pass 30)

### Experiment B: SoftHebbNet (Journé-style WTA, aug)

Clean standalone implementation of Journé et al. ICLR 2023. Three SoftHebb layers (96→384→512 ch, γ=10, η=0.01), SoftWTA competition, per-filter L2 normalisation. Output: 512×4×4 = 8192 features. Target: >45%.

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 24.82%          | 10.00%       |
| 5    | 24.52%          | 10.00%       |
| 10   | 21.28%          | **20.18%**   |
| 20   | 18.69%          | 13.84%       |
| 30   | 18.40%          | 12.03%       |

**Best:** 20.18% LP (pass 10)

> ⚠️ **Feature collapse observed.** NC declines from 24.82% (pass 1) to 18.40% (pass 30) — filters converge to mean patterns rather than discriminative features. LP peaks at pass 10 (20.18%) then collapses as features degrade. With γ=10, SoftWTA acts nearly like a hard argmax per spatial position. LP stays well below baseline due to (a) spatial features (512×4×4) not aligning with the sparse per-sample top-256 probe, and (b) feature collapse reducing discriminability. Proposed fix: reduce γ to 2–4 and/or use global average pooling to produce 512-dim global features.

### Experiment C: LocalContrastiveEncoder (CLAPP-style view consistency)

Wraps SoftHebbNet. Generates two augmented views of each batch; weights Hebbian update by cosine similarity between views: `modulation = 0.5 + 0.5 * cos_sim(f1, f2) ∈ [0, 1]`. Target: >50%.

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 20.93%          | 10.00%       |
| 5    | 20.45%          | 10.29%       |
| 10   | 20.99%          | **21.05%**   |
| 20   | 18.48%          | 14.55%       |
| 30   | 18.48%          | 15.27%       |

**Best:** 21.05% LP (pass 10)

> ⚠️ **Same collapse pattern as B.** Peak at pass 10 (21.05% LP vs B's 20.18%) then declines. Contrastive modulation gives +0.9 pp advantage at peak and slower collapse (15.27% final vs B's 12.03%), but does not prevent the underlying feature collapse. The view-consistency signal is slightly beneficial but insufficient to overcome the γ=10 WTA collapse.

Wraps SoftHebbNet. Masks one of 16 8×8 patches; gradient-trained MLP prediction head predicts masked patch from features; prediction error modulates Hebbian update: `sigmoid(2 * error − 1)`. Target: >50%.

| Pass | Nearest-Centroid | Linear Probe |
|------|-----------------|--------------|
| 1    | 20.84%          | 10.00%       |
| 5    | 20.95%          | 10.00%       |
| 10   | 21.10%          | 10.00%       |
| 20   | 20.60%          | 10.00%       |
| 30   | 20.45%          | 10.00%       |

**Best:** 10.00% LP (random chance throughout)

> ⚠️ **LP stuck at random throughout all 30 passes.** Unlike B and C (which peaked at 20% at pass 10), D shows LP = 10.00% throughout. NC is relatively stable (20.84% → 20.45%) — less collapse than B/C, but no improvement either. Prediction error modulation provides no LP benefit. Likely cause: without augmentation, the unmodulated Hebbian update (error modulation ≈ 1.0 for all samples, since errors are large with random prediction head) is effectively the same as a constant full-strength update, but on unaugmented images — which may not provide sufficient feature diversity for the probe to find discriminative directions.

### Phase 3 Summary

| Experiment | Rule | Best LP | Best Pass | Δ vs 37.6% ceiling |
|------------|------|---------|-----------|---------------------|
| A (control) | Pure Hebbian ConvF1 512 feat aug | **38.87%** | 30 | +1.3 pp |
| B (SoftHebb) | Journé WTA aug, γ=10 | 20.18% | 10 | −17.4 pp |
| C (contrastive) | View-consistency modulation | 21.05% | 10 | −16.6 pp |
| D (predictive) | Masked-patch prediction error | 10.00% | — | −27.6 pp |

**Phase 3 interpretation:**
- B and C peak at pass 10 (~20% LP) then collapse to 12–15% by pass 30 — **feature collapse** from aggressive WTA (γ=10)
- D is stuck at random (10.00%) throughout — prediction-error modulation inactive when pred_head is random (errors always large → modulation ≈ 1.0 always → essentially unmodulated Hebbian on non-augmented images)
- The ceiling was not broken. The Phase 3 negative result is driven by **two compounding implementation issues**, not a fundamental failure of the local SSL learning principle:

  1. **Architectural mismatch:** SoftHebbNet outputs 8192 spatial features (512 ch × 4×4); the sparse top-256 probe works for ConvF1's global 512 features but cannot capture class-discriminative signal from position-specific spatial features where objects appear at different locations per image.
  2. **Hyperparameter collapse:** γ=10 in SoftWTA acts as near hard-argmax, collapsing filter diversity. NC declines from ~25% to ~18% over 30 passes rather than improving — the opposite of what discriminative learning requires.

**These are fixable.** Phase 3b implemented:
- Reduced γ sweep: {0.5, 1, 2, 5, 10} to diagnose temperature-driven collapse
- Global average pooling: 512×4×4 → 512-dim global features (fair comparison with ConvF1)
- Collapse diagnostics per pass: effective_rank, dead_feature_pct, filter_cosim, sparsity
- Best-pass selected by unsupervised health metric (effective_rank, no labels)
- D debug: pred_loss trend, feature variance, two-image distinguishability

**Script:** `experiments/local_ssl.py` | **Commit:** `1f5f3d1`

---

## Local SSL Phase 3b Repair (2026-07-02) — COMPLETE

Repairs the Phase 3 mismatch issues (γ=10 collapse, spatial 8192-dim vs global 512-dim). Fair comparison: all models output 512-dim global features.

**Label: Negative result, but STILL not a valid falsification of local SSL — third evaluation artifact found.**
**Script:** `experiments/local_ssl_3b.py` | **Commit:** `ee20842`

### Phase 3b Acceptance Ladder

| Target | Fail | Weak pass | Real pass | Strong pass |
|--------|------|-----------|-----------|-------------|
| A control | — | — | ~38–39% | — |
| B SoftHebb γ sweep + GAP | <25% | 30–35% | >38.87% | >45% |
| C LocalContrastive + GAP | <25% | 30–35% | >38.87% | >45–50% |
| D Predictive bug-fixed + GAP | ~10% | >20% | >30% | >38.87% |

### Phase 3b Results

**Exp A: ConvF1 control (512-dim, aug)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1    | 25.63% | 28.78% | 3981 |
| 5    | 28.10% | 34.00% | 3728 |
| 10   | 30.14% | 37.29% | 3726 |
| 20   | 28.96% | 37.95% | 3570 |
| 30   | 28.43% | **38.00%** | 3388 |

**Exp B: SoftHebb γ sweep (global pool 512-dim)**

| γ | Best LP | Peak eff_rank | NC trend | Collapse? |
|---|---------|---------------|----------|-----------|
| 0.5 | 10.00% | 17.7 | 30.4–30.6% (stable) | No |
| 1   | 10.00% | 17.7 | 29.9% (stable) | No |
| 2   | 10.00% | 26.3 | 29.7–29.8% (stable) | No |
| 5   | 15.45% | 56.0 → 1.3 | 26.8%→15.9% | Yes (pass 10→30) |
| 10  | 14.39% | 95.1 → 1.0 | 24.4%→10.0% | Yes (pass 5→20) |

γ sweep confirmed: lower γ prevents collapse. But LP=10% at all lower γ values. Collapse γ (5, 10) produce slightly above-chance LP (13–15%) only at the moment of collapse.

**Exp B★ diagnostic: SoftHebb γ=10 spatial 8192-dim (Phase 3 conditions)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1 | 20.84% | 10.00% | 1561 |
| 10 | 19.79% | 14.25% | 65 |
| 20 | 18.51% | 19.82% | 4 |
| 30 | 18.53% | **20.72%** | 4 |

Phase 3 B result reproduced (LP peaks ~20% at collapse with spatial features).

**Exp C: LocalContrastive γ=5 (global pool 512-dim)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1  | 28.12% | 10.00% | 45.7 |
| 10 | 27.07% | 10.00% | 43.0 |
| 20 | 24.47% | 10.00% | 25.3 |
| 30 | 23.36% | **10.00%** | 10.4 |

Declining NC and eff_rank over passes. LP never moved.

---

## Phase 3c Repair — L2-Normalisation Fix (2026-07-02) — COMPLETE

**Label: True gap found. Phase 3b result invalidated by feature-scale bug.**
**Script:** `experiments/local_ssl_3c.py` | **Commit:** `00c00b6`

Third evaluation artifact: global average pooling reduces feature magnitudes ~16× relative to spatial features. LP gradient ∝ raw feature value → vanishes. NC (argmax-based, scale-invariant) was unaffected and correctly showed SoftHebb NC ≈ ConvF1 NC. Fix: `F.normalize(dense, p=2, dim=1)` before sparse probe selection.

### Phase 3c Results (50K, 30 passes, L2-norm fix applied)

| Exp | Model | Pass 1 LP | Pass 30 LP (best) | Δ vs ConvF1 | eff_rank (pass 30) |
|-----|-------|-----------|-------------------|-------------|---------------------|
| A | ConvF1 control | 28.78% | **38.00%** | — | 3387 |
| B | SoftHebb γ=0.5 | — | 29.97% | −8.0 pp | 18 |
| B | SoftHebb γ=1 | — | 30.42% | −7.6 pp | 18 |
| B | SoftHebb γ=2 | — | **31.38%** | −6.6 pp | 26 |
| B | SoftHebb γ=5 | — | 31.87% peak → collapse | — | — |
| B | SoftHebb γ=10 | — | 32.47% pass 1 → collapse | — | 1 |
| C | LocalContrastive γ=10 | — | 31.14% peak → collapse | −6.9 pp | — |
| D | LocalPredictive γ=10 | — | 29.19% (slow decline) | −8.8 pp | — |

**True gap (post all-artifact-repair): SoftHebb/local-SSL ~30–32% vs ConvF1 38.00% — a 6–9 pp gap.**

**Cause:** ConvF1 eff_rank ≈ 3700 (near full-rank, 3 diversity mechanisms: Oja decay, filter decorr, competitive_k). SoftHebb eff_rank ≈ 17–30 (stable but low — no explicit filter diversity beyond soft-WTA). ConvF1 keeps improving pass 1→30 (+9.2 pp). SoftHebb plateau from pass 1 (+0.3 pp).

**Interpretation:** The gap is caused by filter diversity, not the evaluation artifact. Phase 3 view-consistency (Exp C) is not true contrastive learning — it has no negative pairs or anti-collapse mechanism. The missing ingredient is explicit representational separation with negatives.

---

## Phase 4 — Contrastive InfoNCE: Two-Lane Experiment (2026-07-02) — COMPLETE

**Status:** Phase 4 control matched, backprop ceiling shattered, local approximation failed.
**Script:** `experiments/local_contrastive_infonce.py` | **Commit:** `bd8649b`

### Design

Two lanes with identical CIFAR-10 evaluation protocol (50K train / 10K test, 100-epoch LP):

- **Lane 1 (Exp E) — Backprop InfoNCE upper bound:** SimCLR-style NT-Xent loss, two augmented views, backprop through encoder. Labelled BACKPROP_UPPER_BOUND. Used only as a diagnostic ceiling — not Bio-ARN-pure.
- **Lane 2 (Exp F) — Local InfoNCE (BIO_ARN_LOCAL):** Same batch construction. SoftHebbNet backbone (γ=2.0). Positive Hebbian modulation for same-image view pairs; negative anti-Hebbian modulation for shuffled-batch (different-image) pairs. No backprop through encoder. Oja decay=0.05, filter decorrelation=0.02.

**Augmentation:** RandomResizedCrop (scale 0.2–1.0), horizontal flip, colour jitter (brightness/contrast/saturation 0.6–1.4, p=0.8), random grayscale (p=0.2). Implemented as batched `affine_grid + grid_sample` (19× speedup over per-image interpolation).

### Phase 4 Results (50K, 30 passes/epochs, 100-epoch LP, seed=42)

| Exp | Model | Pass 1 LP | Pass 5 LP | Pass 10 LP | Pass 20 LP | Pass 30 LP | eff_rank (final) |
|-----|-------|-----------|-----------|------------|------------|------------|-------------------|
| A | ConvF1 control | 28.78% | 34.00% | 37.29% | 37.95% | **38.00%** | 3388 |
| E | SimCLR BACKPROP_UPPER_BOUND | 42.64% | 53.32% | 58.19% | 62.73% | **65.54%** | 246 |
| F | LocalInfoNCE BIO_ARN_LOCAL | 31.06% | 31.16% | 31.29% | 31.78% | **31.95%** | 28 |

### Acceptance Ladder Verdict

| Target | Threshold | Result | Verdict |
|--------|-----------|--------|---------|
| Local LP > 38.00% | SUCCESS | 31.95% | ❌ FAIL |
| Local LP > 45.00% | STRONG SUCCESS | 31.95% | ❌ FAIL |
| Local LP > 50.00% | BREAKTHROUGH | 31.95% | ❌ FAIL |
| Objective valid (only E > 38%) | — | E=65.54% | ✅ CONFIRMED |

**Verdict: "Objective valid, local approximation not yet solved."**

### Phase 4 Interpretation

**What Exp E (SimCLR) proved:**
- The contrastive objective shatters the 38% ceiling: +27.54 pp (38.00% → 65.54%)
- LP improves monotonically through all 30 epochs (42.64% → 65.54%)
- eff_rank grows with training (66 → 246) — backprop actively diversifies representations
- Loss falls steadily (epoch 1: ~3.60 → epoch 30: ~2.60) — training is stable
- **The architecture is not the bottleneck. The objective is.**

**What Exp F (LocalInfoNCE) showed:**
- LP is near-flat throughout: 31.06% → 31.16% → 31.29% → 31.78% → 31.95%
- eff_rank is near-constant: 29.79 → 29.57 → 29.27 → 28.64 → 28.02 (stable, very low)
- filter_cosim near-zero and falling (0.013 → 0.004) — filters ARE becoming more diverse
- NC stable at ~30% throughout — same as SoftHebb Phase 3c without any contrastive signal
- **The local contrastive modulation is not transmitting into the representations.**

**Why the local approximation fails:**
1. **LP matches pure SoftHebb** (31.95% vs 31.38% Phase 3c γ=2 SoftHebb — identical within noise). The positive/negative Hebbian modulation is not adding representational signal beyond the baseline SoftHebb dynamics.
2. **eff_rank does not grow** (28 stable vs SimCLR's 66→246). Backprop InfoNCE actively diversifies representations across all layers simultaneously. Local updates modify weights per-layer without coordinating the representational geometry across the full encoder.
3. **The anti-Hebbian push is weak.** Negative modulation uses shuffled-batch pairs with weight −0.5 × neg_mod. At SoftHebb γ=2 where many filters weakly co-activate, the negative signal diffuses rather than selectively pushing discriminative dimensions apart.
4. **No credit assignment.** The representational objective (push different-class features apart) requires distributing that error signal across multiple layers. Local Hebbian updates can only modify the immediately post-synaptic weights — they cannot propagate the "these two features are too similar" signal to earlier layers.

### Phase 4 Confirmed Findings

| Finding | Evidence |
|---------|----------|
| The 38% ConvF1 ceiling is not architectural — it is objective-limited | SimCLR (same architecture backbone) reaches 65.54% |
| The contrastive objective alone is sufficient to break the ceiling | +27.54 pp with backprop InfoNCE |
| Local Hebbian contrastive modulation does not approximate the backprop contrastive benefit | LocalInfoNCE LP = 31.95% ≈ pure SoftHebb 31.38% |
| Layer-local weight updates cannot coordinate multi-layer representational geometry | eff_rank stable at 28 vs SimCLR's 66→246 growth |
| The credit-assignment problem remains unsolved for this architecture | No LP improvement from negative pairs across 30 passes |

### What This Rules Out

- ❌ "The ceiling is architectural" — disproved by SimCLR 65.54% on same backbone width
- ❌ "More Hebbian passes will eventually close the gap" — LP flat 31→32% across all 30 passes
- ❌ "Local positive+negative modulation approximates InfoNCE" — LP = SoftHebb baseline
- ❌ "Filter diversity alone explains the gap" — LocalInfoNCE has very low filter_cosim (filters ARE diverse) but LP does not improve

### What Remains Open

- ✅ Can a stronger local objective (BYOL-style, LFNA, predictive coding) transmit more contrastive signal?
- ✅ Can layer-wise target propagation (instead of pure Hebbian) approximate the backprop credit path?
- ✅ Does recurrence or temporal continuity (missing in current Bio-ARN) provide implicit credit assignment?

### Recommended Phase 5

If continuing: the next experiment should test **local target propagation** (e.g., Difference Target Propagation or DRTP) as the credit-assignment mechanism, keeping the contrastive objective from Exp E but replacing backprop with a locally-computed target.

Alternatively: accept the Phase 4 result as the final Bio-ARN finding — that biologically-constrained local learning cannot approximate contrastive self-supervision without some form of non-local credit signal — and document this as the principal conclusion.

**Exp D: LocalPredictive γ=5 (global pool 512-dim)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1  | 25.83% | 10.00% | 34.60 |
| 10 | 25.77% | 10.00% | 33.97 |
| 20 | 25.71% | 10.00% | 33.33 |
| 30 | 25.88% | **10.00%** | 32.68 |

Most stable of all experiments: NC and eff_rank barely change over 30 passes. LP stuck at 10%. D diagnostic confirmed: pred_loss is decreasing (pipeline works), but features near-constant at init — magnitude too small for LP probe.

### Phase 3b Diagnosis: Third Evaluation Artifact

**Critical observation:** SoftHebb γ=0.5/1/2 shows NC=29–30% — **identical to ConvF1's NC (28–30%)**. The nearest-centroid classifier (which normalises prototypes, making it scale-invariant) reports features as equally discriminative to ConvF1. But LP=10%.

**Root cause — feature scale:** Global average pooling computes mean over 4×4=16 spatial positions. With ReLU half-sparsity, activations are ~16× smaller in magnitude than individual spatial activations used in Phase 3 (spatial features, no pooling). The linear probe learns via gradient ∝ feature_value. Sub-threshold feature magnitudes cause vanishing gradient in the probe — it never learns.

| Evaluation metric | Scale sensitivity | SoftHebb | ConvF1 |
|---|---|---|---|
| NC (nearest-centroid, normalised prototypes) | **Scale-invariant** | 29–30% | 28–30% |
| LP (SGD probe, raw feature values) | **Scale-dependent** | 10% | 38% |

NC says the features ARE discriminative. LP can't learn from them because global avg pooling reduced their magnitude ~16×.

This is not a feature quality failure — it is an evaluation mismatch between a scale-invariant metric (NC) and a scale-sensitive metric (LP) applied to GAP-reduced features.

**Phase 3b is ALSO not a valid falsification of local SSL.** Three compounding evaluation bugs now found:
1. Phase 3: γ=10 causes hard-argmax feature collapse (fixed in 3b via γ sweep)
2. Phase 3: Spatial 8192-dim features vs global 512-dim mismatch (fixed in 3b via GAP)
3. Phase 3b: Global average pooling reduces feature magnitudes ~16×, killing LP gradient signal (fix: L2-normalise before probe)

**Phase 3c fix (one line):**
```python
dense = F.normalize(dense, p=2, dim=1)  # before top-k selection in probe
```
Applied uniformly to all experiments including the ConvF1 control. Makes LP evaluation scale-invariant, matching NC behaviour.

---

## Local SSL Phase 3c: L2-Normalised Probe (2026-07-02) — COMPLETE

Applies `F.normalize(dense, p=2, dim=1)` before probe feature selection. All other settings identical to Phase 3b. This is the final evaluation fix.

**Script:** `experiments/local_ssl_3c.py` | **Commit:** `00c00b6`

### Phase 3c Results

**Exp A: ConvF1 control** — unchanged from Phase 3b (normalization neutral to ConvF1)

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1  | 25.63% | 28.78% | 3981 |
| 10 | 30.14% | 37.29% | 3727 |
| 30 | 28.43% | **38.00%** | 3388 |

**Exp B: SoftHebb γ sweep (global pool 512-dim, L2-normalised)**

| γ | Pass 1 LP | Pass 10 LP | Best LP | Eff_rank start→end | Collapse? |
|---|-----------|------------|---------|---------------------|-----------|
| 0.5 | 29.93% | 29.93% | **29.97%** (p20) | 19→18 | No — stable plateau |
| 1   | 30.15% | 30.24% | **30.42%** (p30) | 20→19 | No — stable plateau |
| 2   | 31.37% | 31.32% | **31.38%** (p30) | 30→28 | No — stable plateau |
| 5   | 31.87% | 30.39% | **31.87%** (p1) | 59→2  | Yes — collapses pass 15→30 |
| 10  | 32.47% | 21.06% | **32.47%** (p1) | 108→1 | Yes — collapses pass 10→ |

**Exp B★ diagnostic: γ=10 spatial (Phase 3 conditions)**

| Pass | LP | Eff_rank |
|------|-----|----------|
| 1  | 32.39% | 2075 |
| 10 | 26.98% | 97 |
| 30 | 19.63% | 5 |

**Exp C: LocalContrastive γ=10 (global pool, L2-normalised)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1  | 27.02% | 30.75% | 97 |
| 5  | 26.86% | **31.14%** | 88 |
| 10 | 25.83% | 29.28% | 50 |
| 20 | 15.58% | 15.75% | 3 |
| 30 | 10.00% | 10.00% | 1 |

Peaks at pass 5, then collapses by pass 20. Best: 31.14%.

**Exp D: LocalPredictive γ=10 (global pool, L2-normalised)**

| Pass | NC | LP | Eff_rank |
|------|----|----|----------|
| 1  | 24.26% | **29.19%** | 58 |
| 10 | 24.67% | 28.47% | 54 |
| 30 | 23.43% | 27.12% | 43 |

Most stable of all: eff_rank declines slowly (58→43), LP stable 27–29%. Does not collapse.

### Phase 3c Summary vs Acceptance Ladder

| Exp | Model | Best LP | Δ vs 38.00% (control) | Acceptance verdict |
|-----|-------|---------|----------------------|-------------------|
| A  | ConvF1 control | **38.00%** | — | ✅ Baseline confirmed |
| B  | SoftHebb γ=2 (stable) | **31.38%** | −6.62 pp | Weak pass (30–35%) |
| B  | SoftHebb γ=1 (stable) | **30.42%** | −7.58 pp | Weak pass |
| B  | SoftHebb γ=10 (collapse, peak) | **32.47%** | −5.53 pp | Weak pass — unstable |
| C  | LocalContrastive | **31.14%** | −6.86 pp | Weak pass — collapses |
| D  | LocalPredictive | **29.19%** | −8.81 pp | Weak pass — stable |

**None of B/C/D reached "real pass" (>38.87%) or "strong pass" (>45%).** All remain 6–9 pp below the ConvF1 control.

### Phase 3c Diagnosis: True Performance Gap Found

With all three evaluation artifacts now fixed (γ collapse, spatial/global mismatch, feature scale), the **true performance gap** between architectures is:

- **ConvF1 (sophisticated Hebbian, explicit diversity):** 38.00% LP
- **SoftHebb γ=2 (simple WTA, no diversity mechanism):** 31.38% LP
- **Local contrastive (view-consistency modulation):** 31.14% LP (peak p5)
- **Local predictive (masked-patch modulation):** 29.19% LP (stable)

**Gap: −6–9 pp.** This is not an artifact. It is the real cost of ConvF1's explicit diversity mechanisms (Oja decay 0.05, filter decorrelation 0.02, competitive_k=64, spatial_grid competition) that SoftHebb lacks.

**Why ConvF1 keeps improving but SoftHebb plateaus:**
- ConvF1 pass 1→30: 28.78% → 38.00% (+9.22 pp) — keeps learning
- SoftHebb γ=1 pass 1→30: 30.15% → 30.42% (+0.27 pp) — flat

ConvF1's diversity mechanisms cause filters to keep refining. SoftHebb learns all it can in pass 1 and stops. With eff_rank=19 (vs ConvF1 eff_rank=3700), SoftHebb uses only ~19 independent feature dimensions out of 512. ConvF1 uses ~3700.

**Why eff_rank is so different:**
- ConvF1's `filter_decorrelation=0.02` explicitly pushes filters apart
- SoftHebb only has soft WTA competition — filters converge toward different dominant stimuli but share no explicit repulsion
- Eff_rank 19 means 512 SoftHebb filters are nearly linearly dependent — most are near-duplicates

**What would fix this:**
1. Add explicit filter decorrelation to SoftHebb (anti-Hebbian lateral inhibition)
2. True contrastive SSL (SimCLR-style InfoNCE, positive/negative pairs with random crop augmentation) — literature reports 50–70%+ on CIFAR-10
3. More training data helps here because the self-supervised signal would benefit from data variety (unlike pure Hebbian where more data saturates)

### Phase 3c Verdict

**Phase 3c is a valid result.** Three evaluation artifacts found and repaired over Phase 3→3b→3c. The true Bio-ARN local SSL ceiling for this architecture class is:

- Pure SoftHebb simple WTA: **~30–32%** (stable, limited by low filter diversity)
- Local view-consistency contrastive: **~31%** (collapses at γ=10, limited by same)
- Local predictive (masked patch): **~29%** (most stable, slowly declining)

vs ConvF1 (carefully tuned Hebbian with explicit diversity): **38%**

The learning signal changes tested in Phase 3–3c do NOT break the ceiling above ConvF1. They fall 6–9 pp below it. To exceed 38%, the next move is genuine contrastive self-supervised learning (InfoNCE, not view-consistency modulation) with explicit negative pairs.

---

## Files

| File | Description |
|------|-------------|
| `experiments/local_ssl_3c.py` | **Phase 3c: L2-normalised probe (final valid evaluation, true gap found)** |
| `experiments/local_ssl_3b.py` | **Phase 3b: repair of collapse/probe issues (γ sweep, GAP, diagnostics)** |
| `experiments/local_ssl.py` | **Phase 3: local self-supervised feature learning (A/B/C/D)** |
| `experiments/data_scaling.py` | **Data scaling experiments (aug, multi-dataset C10+C100+SVHN, 512 feat)** |
| `experiments/hebbian_scaling.py` | **Progressive scaling experiments (combined scale, full data, divisive norm)** |
| `experiments/bias_audit.py` | **Survivorship bias audit (capacity, duration, eval, pure Hebbian)** |
| `experiments/softhebb_final_validation.py` | Decision-grade validation script (5 seeds × 3 configs) |
| `experiments/fullscale_softhebb.py` | Full-scale benchmark (10K samples, 10 passes, 4 configs) |
| `experiments/softhebb_benchmark.py` | Original ablation (7 configs, 1K samples) |
| `experiments/linear_probe_benchmark.py` | Linear probe evaluation framework |
| `experiments/softhebb_hyperparam_sweep.py` | Hyperparameter grid search |
| `bioarn/core/softhebb_net.py` | **SoftHebbNet — clean Journé-style SoftWTA architecture** |
| `bioarn/core/local_contrastive.py` | **LocalContrastiveEncoder — CLAPP-inspired view-consistency modulation** |
| `bioarn/core/local_predictive.py` | **LocalPredictiveEncoder — masked-patch prediction-error modulation** |
| `bioarn/config.py` | SoftHebb configuration (ConvCCCConfig) |

## Recommendation

**Competition mechanism: CLOSED.** SoftHebb ≈ baseline at all scales.
**Normalisation mechanism: CLOSED.** Divisive norm hurts (−5.6 pp).
**Data scaling: CLOSED.** 50K → 173K (3.5×) + augmentation + 512 features → no improvement. Ceiling is real.
**Phase 3 (learning signal change): COMPLETE — negative result due to γ=10 collapse + spatial feature mismatch (not a valid falsification of local SSL).**
**Phase 3b (γ sweep + global pooling): COMPLETE — NC=29–30% ≈ ConvF1, but LP stuck at 10% due to GAP scale reduction. Third evaluation artifact found.**
**Phase 3c (L2-normalise before probe): COMPLETE — true gap found: SoftHebb 30–32% vs ConvF1 38%. Valid result after all three artifacts fixed.**

All prior scaling axes tested and closed:

1. ~~Combined scale (256 feat × 50 pass × 5K)~~ ✅ **27.4% LP**
2. ~~Full dataset (50K)~~ ✅ **37.6% LP — +10.2 pp, biggest gain**
3. ~~Bio-plausible divisive norm~~ ✅ **−5.6 pp — skip this axis**
4. ~~More data + aug + 512 feat~~ ✅ **37.3–37.7% — flat, ceiling confirmed**

**The ceiling is ~37.6% and is not a data problem.** Adding 3.5× more training images from CIFAR-100 and SVHN, doubling features to 512, and applying online augmentation all fail to move the needle. Every experiment lands in the 37.3–37.7% band — a spread of 0.4 pp across wildly different configurations.

**What this means:** The bottleneck is the learning rule itself. Pure Hebbian updates learn local correlations (edges, textures, frequency patches) but cannot organise features into class-discriminative representations without some form of supervision or contrastive signal. More data gives more of the same kind of features, not better-organised ones.

**Phase 3 tests the hypothesis directly:** Changing the learning signal (SoftHebb WTA architecture, local contrastive modulation, predictive patch masking) should break the ceiling if the rule is the bottleneck. Results from Phase 3→3b→3c (after fixing three evaluation artifacts) show that simple SoftHebb and local SSL variants reach **30–32% LP** — 6–9 pp below ConvF1's 38%. The true ceiling for plain SoftHebb without explicit diversity is ~31%. To exceed 38%, genuine contrastive SSL (InfoNCE with random-crop negative pairs) is required.

**Bottom line:** 20% → 38.87% from capacity + data scaling (Phase 1/2 ceiling). Phase 3c confirms SoftHebb simple WTA and local view-consistency signals do not break this ceiling — they fall ~7 pp short. The remaining gap to supervised (90%+) and modern unsupervised Hebbian (64–76%) requires InfoNCE-style contrastive objectives or predictive coding with proper negative sampling, not just view-consistency modulation.
