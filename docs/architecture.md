# Architecture Guide

This guide explains how Bio-ARN 2.0 is assembled, how information moves through the system, and which knobs matter most in practice. For the public API surface, see [API Reference](api_reference.md). For runnable examples, see [Getting Started](getting_started.md).

## System overview

Bio-ARN combines sparse perception, concept-cell competition, associative memory, predictive coding, workspace broadcast, action generation, and reward modulation into one online loop.

```text
Raw input
   │
   ├─► VisualEncoder / LanguageEncoder
   │        │
   │        └─► sparse sensory features + predictive suppression stats
   │
   ├─► PredictiveHierarchy
   │        │
   │        ├─► prediction
   │        ├─► prediction error / free energy
   │        └─► action signal for active inference
   │
   ├─► BioARNCore
   │        │
   │        ├─► CCCPool selects or recruits concept cells
   │        ├─► AssociativeFabric recalls linked concepts
   │        └─► GlobalNeuronalWorkspace broadcasts the winners
   │
   ├─► LanguageMotorStream
   │        │
   │        ├─► concept-to-token rollout
   │        └─► self-monitoring / correction
   │
   └─► RewardSystem
            │
            └─► novelty, curiosity, dopamine-like learning modulation
```

At runtime, the end-to-end orchestrator is `SensorimotorLoop`. It owns the sensory encoders, predictive hierarchy, `BioARNCore`, reward system, and motor stream.

## Core components

### 1. Concept Cell Clusters (CCCs)

**What it does** — Each `ConceptCellCluster` is a sparse concept detector with an F1 feature stage, an F2 concept representation, a margin gate, and a feedback pathway. The surrounding `CCCPool` runs many CCCs, lets them compete, and recruits a fresh CCC when no existing concept explains the input.

**Why it exists** — This is where Bio-ARN gets explicit abstention and one-shot concept recruitment. Instead of forcing every input into a closed class list, CCCs can say “none of the above” until a new concept is worth committing.

**How to use it** — Most users interact with CCCs through `BioARNCore.perceive()` or `SensorimotorLoop.step()`. Tune `ccc.max_pool_size`, `ccc.concept_dim`, and `margin_gate.theta_margin` when adapting the model to new datasets.

### 2. Associative Fabric

**What it does** — `SparseDistributedMemory` provides Kanerva-style content-addressable memory. `AssociativeFabric` adds temporal binding, lateral inhibition, voting, and sequence recall over the CCC outputs.

**Why it exists** — It replaces dense attention with sparse associative retrieval. Stored concepts can trigger nearby or temporally adjacent concepts with local updates and without a context-window bottleneck.

**How to use it** — You normally access it indirectly through `BioARNCore`, but it is useful directly for research experiments around memory capacity, sequential recall, or alternative routing rules.

### 3. Predictive hierarchy

**What it does** — `PCLayer` and `PCStack` implement local predictive coding. `PredictiveHierarchy` packages them into a perception and generation interface that exposes free energy, precision maps, level states, and active-inference directions.

**Why it exists** — Rather than propagate all activity upward, Bio-ARN propagates residual error. Predictable structure gets suppressed early, which is a major reason the architecture stays sparse.

**How to use it** — Use `SensorimotorLoop.predict()` and `SensorimotorLoop.active_inference_step()` for most work. Direct `PredictiveHierarchy` access is helpful when diagnosing stability, suppression, or control behavior.

### 4. Global Neuronal Workspace

**What it does** — `GlobalNeuronalWorkspace` maintains a small, competitive working set of concepts. `StreamOfConsciousness` turns repeated workspace updates into a serial thought chain.

**Why it exists** — The GNW gives Bio-ARN a bottleneck for focus, broadcast, and serial planning. It is also the bridge between recognition and generation.

**How to use it** — `BioARNCore.think()` and `SensorimotorLoop.attend()` are the normal entry points. For explicit generation experiments, you can manually inject a concept into the GNW before calling `run(generate=True)`.

### 5. Sensorimotor encoders and motor stream

**What it does** — `VisualEncoder` converts frames into sparse event features; `LanguageEncoder` converts token sequences into temporal spiking features; `LanguageMotorStream` generates output tokens from concepts and can self-correct when predicted and produced tokens diverge.

**Why it exists** — Bio-ARN is not just a classifier. The architecture is designed to sit inside a perception-action loop where generation is a top-down prediction problem.

**How to use it** — Use `SensorimotorLoop.sense()` for raw encoding, `step()` for full closed-loop behavior, and `generate_text()` for quick text rollouts.

### 6. Reward and novelty system

**What it does** — `RewardSystem` turns prediction error dynamics into intrinsic reward, detects novelty, and adjusts learning-rate modulation. `DopamineScheduler` is the slower tonic/phasic controller behind those modulation signals.

**Why it exists** — The architecture needs a principled way to learn faster from surprising inputs and less from already-explained inputs.

**How to use it** — Usually you let `SensorimotorLoop.step()` call it automatically. In research code, call `RewardSystem.step()` directly if you want to study novelty thresholds or curiosity schedules.

### 7. Scaling and hardware layers

**What it does** — `ScaledBioARN` and `BatchedCCCPool` keep the same semantics while vectorizing core operations. `NeuromorphicBackend`, `PyTorchBackend`, and `LoihiMapping` define how the architecture maps onto software or spike-native hardware.

**Why it exists** — The research claim is partly architectural and partly hardware-software co-design. These modules make that claim executable.

**How to use it** — Start with the default PyTorch path, then switch to scaled or hardware-aware experiments when profiling bottlenecks or deployment strategies.

## The full perception loop

A typical perception step in `SensorimotorLoop.step()` looks like this:

1. **Encode sensory input** — `VisualEncoder` or `LanguageEncoder` turns raw data into a sparse feature vector.
2. **Suppress the predictable** — recently predicted or repeated structure is damped before higher-level processing.
3. **Predict** — `PredictiveHierarchy` runs several local predictive-coding iterations and returns prediction, error, surprise, and free energy.
4. **Recognize** — `BioARNCore.recognize()` asks whether existing CCCs can explain the features above the current margin.
5. **Perceive / recruit** — `BioARNCore.perceive()` runs the full CCC pool; if nothing explains the input, a new CCC is recruited.
6. **Associate** — `AssociativeFabric` registers the active concept(s) and retrieves nearby or temporally linked concepts.
7. **Broadcast** — the GNW selects a small subset of the active concepts and amplifies them into the workspace.
8. **Reward update** — novelty and intrinsic reward update learning multipliers.
9. **Learn locally** — resonance, co-activation, and novelty drive Hebbian/STDP updates. No backward pass is run.
10. **Emit diagnostics** — step outputs expose free energy, abstention, confidence, novelty, action state, and workspace occupancy.

## The generation loop

When `generate=True` or `generate_text()` is called, the flow is inverted:

1. A seed concept is injected directly or indirectly into the GNW.
2. The GNW holds the current conceptual state to express.
3. `PredictiveHierarchy.generate()` produces a top-down expectation for lower layers.
4. `SensorimotorLoop.plan()` turns the conceptual state (and optional goal) into a motor plan.
5. `LanguageMotorStream.plan()` converts the concept direction into token logits.
6. `LanguageMotorStream.execute_step()` samples or selects the next token.
7. The produced token is compared against the model’s own prediction buffer.
8. If self-monitoring detects a mismatch, the loop can self-correct on the next step.
9. The GNW advances the thought chain and generation repeats until the requested length is reached.

## How learning works

Bio-ARN learns continuously with local rules:

- **Fast concept recruitment** — a novel input can immediately claim a free CCC via `learn_fast()`.
- **Slow Hebbian refinement** — once a concept is committed, repeated resonance tunes F1/F2 and feedback weights.
- **STDP-style associative updates** — the SDM/fabric strengthens temporal links when concepts co-occur in order.
- **Precision-weighted prediction error** — the predictive stack updates state and weights from local errors only.
- **Reward modulation** — novelty boosts or tonic dopamine shifts learning-rate multipliers, especially `ccc.slow_lr`.
- **No backprop** — there is no global gradient, no backward pass, and no requirement for replay-heavy batch optimization.

## Configuration guide

Bio-ARN’s defaults are conservative and designed around the bundled MNIST benchmark. Start with `configs\mnist.yaml` unless you are profiling or adding modalities.

### Tune these first

| Parameter | Why tune it | Typical effect |
|---|---|---|
| `ccc.max_pool_size` | Capacity of the concept inventory | Larger pools improve coverage but raise memory/CPU cost |
| `ccc.concept_dim` | Size of concept vectors | Higher dimensions can improve separability at extra memory cost |
| `ccc.f1_top_k` | Sparsity of feature competition | Lower values make representations sparser |
| `margin_gate.theta_margin` | Abstention threshold | Higher values abstain more and recruit more new CCCs |
| `predictive.num_levels` | Depth of predictive hierarchy | More levels add capacity and latency |
| `reward.novelty_threshold` | Surprise sensitivity | Lower values trigger novelty responses sooner |
| `reward.novelty_boost` | Strength of modulation | Higher values accelerate adaptation to novel inputs |

### Usually leave these alone first

| Parameter | Why leave it alone |
|---|---|
| `spiking.dt`, `spiking.refractory_steps` | Stable defaults for the reference implementation |
| `sdm.address_dim`, `sdm.hamming_radius` | Coupled memory-capacity knobs; change only with deliberate SDM experiments |
| `predictive.precision_init` | Good default for balanced error weighting |
| `gnw.capacity` | Intentionally small to preserve a tight workspace bottleneck |
| `reward.curiosity_weight` | Sensible exploratory baseline unless you are studying policy selection |

### Presets

- `mnist` — default vision benchmark preset.
- `cifar` — larger input dimensionality for CIFAR-like images.
- `language_small` / `language_large` — language-oriented feature sizes.
- `production` — larger concept pool for longer-running experiments.

## Practical debugging checklist

- High abstention on familiar inputs → lower `theta_margin` or inspect CCC recruitment.
- Too many CCCs recruiting → raise resonance quality or inspect encoder sparsity.
- Noisy generation → inspect `LanguageMotorStream` self-monitoring and reward modulation.
- Free energy not dropping on repeated inputs → inspect predictive precision, suppression, and encoder reset state.
- CPU prototype too slow → move to `ScaledBioARN` / `BatchedCCCPool` before drawing hardware conclusions.
