# Supplementary Material for *Bio-ARN 2.0: A Brain-Inspired Architecture for Honest, Efficient, and Continual Intelligence*

## S1. Benchmark hyperparameters

### S1.1 MNIST benchmark suite (`experiments/benchmarks/results.json`)

| Setting | Value |
|---|---:|
| Seeds | 42, 123, 777 |
| Training samples | 5000 |
| Test samples | 1000 |
| Calibration samples | 500 |
| Few-shot k values | 1, 5, 10 |

### S1.2 Energy-report configuration (`experiments/energy_report_data.json`)

| Component | Key settings |
|---|---|
| Spiking | beta=0.9, threshold=1.0, dt=1.0, refractory=2 |
| Margin gate | theta_margin=0.5, theta_margin_lr=0.0005, theta_resonance=0.65 |
| CCC | input_dim=784, concept_dim=64, num_f1_features=128, f1_top_k=16, max_pool_size=25 |
| SDM | address_dim=10000, radius=451, hard_locations=1000, data_dim=64 |
| Predictive hierarchy | num_levels=4, gamma=0.1, eta=0.01, error_threshold=0.01 |
| GNW | capacity=7, broadcast_gain=2.0, fatigue_rate=0.1 |
| Reward | novelty_threshold=3.0, novelty_boost=5.0, curiosity_weight=0.5 |

### S1.3 Text-generation benchmark configuration (`experiments/benchmarks/text_gen_results.json`)

| Setting | Value |
|---|---:|
| Train characters | 2800 |
| Eval characters | 800 |
| Context lengths tested | 1, 4, 8, 16, 32 |
| Bio-ARN benchmark context length | 16 |
| Bio-ARN train samples | 2200 |
| Vocabulary size | 37 |

### S1.4 CIFAR preprocessing benchmark (`experiments/cifar_training.py`)

| Setting | Value |
|---|---:|
| Input dimensionality | 3072 |
| Concept dimensionality | 256 |
| Max pool size | 384 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Training samples | 1200 |
| Test samples | 300 |
| Thresholds swept | 0.30, 0.35, 0.40, 0.45 |
| Preprocessing warmup | 200 samples for PCA-based runs |

### S1.5 Large-pool scaling benchmark (`experiments/large_pool_scaling.py`)

| Setting | Value |
|---|---:|
| Pool sizes | 100, 500, 1000, 2000, 5000, 10000 |
| Patterns evaluated | 1000 |
| Batch size | 64 |
| Input / concept dim | 32 / 32 |
| F1 top-k | 8 |
| Margin schedule | 0.52 + 0.05 log10(scale), capped at 0.72 |
| Shard size for 10K benchmark | 1000 |

## S2. Additional numerical breakdowns

### S2.1 OOD performance by shift type (Bio-ARN)

| OOD set | AUROC |
|---|---:|
| Random noise | 0.99797 |
| Rotated 90 deg | 0.94323 |
| Inverted digits | 0.998996 |
| Fashion-MNIST | 0.791664 |

### S2.2 Text few-shot case studies (`text_gen_results.json`)

#### One-shot (1 example)

| Case | Bigram | Trigram | Bio-ARN |
|---|---:|---:|---:|
| Phrase continuation | 0.143 | 0.143 | 0.286 |
| Episodic pairs | 0.000 | 0.500 | 0.667 |
| Paired variants | 0.167 | 0.167 | 0.500 |
| Symbol sequence | 0.143 | 0.286 | 0.714 |
| Novel punctuation | 0.250 | 0.250 | 1.000 |

#### Five-shot (5 examples)

| Case | Bigram | Trigram | Bio-ARN |
|---|---:|---:|---:|
| Phrase continuation | 0.286 | 0.000 | 0.429 |
| Episodic pairs | 0.167 | 0.167 | 0.667 |
| Paired variants | 0.167 | 1.000 | 0.500 |
| Symbol sequence | 0.429 | 1.000 | 0.714 |
| Novel punctuation | 0.750 | 0.750 | 1.000 |

### S2.3 Cross-modal per-category retrieval

| Query class | Predicted label | Top-1 correct? |
|---|---|---|
| horizontal_top | horizontal_top | yes |
| horizontal_mid | horizontal_mid | yes |
| vertical_left | vertical_left | yes |
| vertical_mid | vertical_mid | yes |
| diagonal | diagonal | yes |
| anti_diagonal | anti_diagonal | yes |
| cross | horizontal_mid | no |
| x_shape | diagonal | no |
| box | box | yes |
| center_dot | center_dot | yes |

This corresponds to 8/10 top-1 accuracy and MRR 0.950. Errors are structurally interpretable rather than arbitrary.

## S3. Mathematical derivations

### S3.1 Margin-gated abstention as constrained recognition

The CCC can be interpreted as solving a local constrained recognition problem. Let $h_i$ be the concept-space activation and $d_i$ the normalized prototype. A conventional classifier would always choose the maximally scoring concept. Bio-ARN instead imposes the constraint that a concept is allowed to claim the input only if the margin exceeds a threshold:

$$
\\text{claim}_i = \mathbf{1}[\cos(h_i, d_i) > \\theta_i^{margin}].
$$

The abstention region is therefore explicit and geometrically meaningful: it is the complement of the accepted angular cone around the concept direction. Threshold adaptation changes cone width without requiring the model to relearn the concept itself.

### S3.2 Resonance-gated local learning

Slow learning occurs only under resonance. Define a feature reconstruction error

$$
e_i = f^{(1)} - W_i^{fb} h_i.
$$

If $\cos(W_i^{fb} h_i, f^{(1)}) > \\theta^{res}$, then the concept update is applied. This makes learning conditional on internal-consistency checks instead of purely on label loss. In effect, the system asks whether the fired concept can also predict the feature evidence it is supposed to explain.

### S3.3 Predictive-coding objective

Although no global free-energy objective is explicitly optimized by backpropagation, the layer-local updates approximate descent on a weighted prediction-error energy:

$$
\mathcal{F} = \sum_{\ell} \pi_\ell \|x_\ell - f(W_\ell x_{\ell+1})\|^2,
$$

where $\pi_\ell$ are precision-like coefficients. The hierarchy iteratively settles states and updates weights through local correlations, sidestepping explicit end-to-end gradient transport.

### S3.4 Sparse distributed memory retrieval

Given binary addresses $a_i \in \{0,1\}^N$ and stored data vectors $D[m]$, retrieval for cue $c$ within Hamming radius $r$ is

$$
R(c) = \sum_{m \in \mathcal{N}_r(a(c))} D[m].
$$

Noise robustness follows because nearby addresses contribute jointly; partial cues can still intersect enough of the stored neighborhood to recover meaningful associations.

## S4. Hardware requirement estimates

### S4.1 Inference profile (current energy report)

| Component | Dense FLOPs | Sparse FLOPs | Time (ms) | Reported sparsity |
|---|---:|---:|---:|---:|
| CCC Pool | 1,642,368 | 148,659 | 12.30 | 74.5% |
| SDM Retrieval | 98,476,800 | 42,560,000 | 137.49 | 100.0% |
| PE Hierarchy | 2,940,928 | 1,199,012 | 18.04 | 47.3% |
| GNW | 745 | 285 | 1.11 | 48.6% |
| Motor Stream | 622,592 | 304,947 | 10.79 | 77.9% |
| Total | 103,683,433 | 44,212,903 | 179.74 | 82.5% modeled-unit silence |

### S4.2 Loihi deployment pipeline fields

The repository's deployment pipeline estimates:
- core count,
- total neurons,
- total synapses,
- memory per core,
- estimated power in mW,
- estimated latency in ms,
- quantization loss and equivalence tolerance.

These estimates strengthen the deployment narrative even though they do not yet replace real-chip profiling.

### S4.3 Practical edge guidance

From the archived scaling and energy reports, a plausible near-term deployment envelope is:
- small edge configurations: up to 1K CCCs with flat batched pooling,
- medium edge configurations: roughly 5K CCCs with careful memory budgeting,
- large edge or accelerator-backed configurations: 10K+ CCCs with sharded retrieval and hardware-local SDM.

## S5. Complete per-class / per-case reporting note

The archived benchmark JSON files provide per-distribution, per-case, and per-scenario statistics, but they do **not** serialize a full MNIST per-digit table. The Phase-0 MNIST proof-of-concept script prints per-class accuracy during execution, yet those outputs are not preserved in the bundled artifacts. For reproducibility, this supplement therefore includes all archived granular results that are actually present: OOD-type breakdowns, text few-shot case breakdowns, and multimodal per-category retrieval outcomes.
