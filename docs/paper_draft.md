# Bio-ARN: A Brain-Inspired Architecture for Energy-Efficient Multi-Modal Learning on Neuromorphic Hardware

## Abstract

Modern generative AI systems are increasingly constrained by energy cost, brittle out-of-distribution (OOD) behavior, and reliance on gradient-based batch training. Bio-ARN 2.0 explores a different design point: a brain-inspired architecture built around sparse concept circuits, local Hebbian learning, predictive coding, workspace-style global broadcasting, and a deployment path to neuromorphic hardware. The core computational unit is the Concept Cell Cluster (CCC), a cortical-column-inspired module that can abstain, recruit new concepts online, and refine them through local resonance rather than backpropagation. CCCs are organized into a ventral-stream-like visual hierarchy, linked through predictive coding and top-down feedback, and coupled to a Global Neuronal Workspace (GNW) for competition, broadcast, and multi-step internal routing. The current system also adds opt-in STDP dynamics, synaptic consolidation, dynamic capacity growth, and curiosity-driven replay.

The current system does not match state-of-the-art transformer accuracy on large-scale visual benchmarks. However, it shows promising properties along other axes. On real CIFAR-10, a hierarchy baseline reaches 30.0% accuracy; Sprint D curiosity+curriculum training improves this to 33.8% online and 55.8% in post-training evaluation; and the best combined 2,000-sample configuration reaches 33.0% accuracy with 1.000 OOD AUROC. Sprint E adds three targeted continual-learning mechanisms: concept locking, convolutional CCCs, and precision-weighted predictive processing inspired by hippocampal uncertainty signaling [Frank et al., 2026]. In the latest Split-CIFAR-10 continual-learning benchmark, the combined convolutional+locking path reduces mean forgetting from 34.7% to 20.7%, while the precision gate scales Hebbian plasticity with pool entropy and auto-locking preserves mature concept detectors once their importance crosses threshold. Full accuracy gains from the convolutional path are still pending, but the retention improvement is already material. A Loihi/Lava export path preserves weights exactly, passes round-trip validation, and keeps simulated accuracy within 0.047 of the source model. Analytic energy modeling projects 179.65 µJ per inference on Loihi 2 versus 50.01 mJ for a transformer baseline on an A100 GPU, a 278x advantage at a matched benchmark tier. These results position Bio-ARN not as a drop-in replacement for transformers, but as a candidate architecture for efficient, robust, continuously learning neuromorphic AI.

## 1. Introduction

Large foundation models have demonstrated remarkable capability, but their dominant recipe remains expensive and brittle: dense matrix multiplications, backpropagation-heavy optimization, and high inference energy on von Neumann hardware [Brown et al., 2020; Dosovitskiy et al., 2021]. At the same time, these systems often struggle with calibrated uncertainty, out-of-distribution detection, and online continual adaptation. Biological nervous systems solve a different problem under radically different constraints: sparse activity, local plasticity, continual updating, and tight energy budgets [Hebb, 1949; Friston, 2010].

Bio-ARN 2.0 asks whether some of those constraints can be elevated from inspiration to systems design. Rather than treating biological plausibility as an aesthetic goal, Bio-ARN uses it to guide architecture: sparse concept detectors, predictive suppression, local associative memory, broadcast-limited global access, and event-driven deployment assumptions. The result is a multi-module cognitive stack intended for online learning and neuromorphic execution rather than maximum leaderboard accuracy.

The project's central claim is therefore not that brain-inspired learning already outperforms transformer-scale systems on raw supervised accuracy. It does not. Instead, the claim is that a unified architecture can combine four properties that are rarely demonstrated together in one executable system: (1) local Hebbian learning without backpropagation, (2) practical OOD-aware abstention and uncertainty signals, (3) online continual adaptation with explicit capacity management, and (4) a concrete export path toward neuromorphic hardware.

This paper makes the following contributions:

1. It presents **Concept Cell Clusters (CCCs)** as the basic sparse computational unit for online recruitment, abstention, refinement, and importance-triggered locking of mature concepts.
2. It integrates CCCs into a **visual ventral-stream hierarchy**, a **predictive coding stack with top-down gating and precision weighting**, a **Global Neuronal Workspace**, an **ensemble OOD mechanism**, and **curiosity-driven replay**.
3. It reports new **scaling-law and continual-learning analyses**, including a Sprint E reduction in class-split forgetting through convolutional CCCs and concept locking.
4. It provides a **functional Loihi 2 export path** with round-trip/Lava validation and an analytic energy study suggesting large projected efficiency gains over dense transformer baselines.

[Figure 1: End-to-end Bio-ARN pipeline, from sensory encoding through predictive coding, CCC hierarchies, GNW broadcast, ensemble/OOD heads, curiosity-driven replay, and Loihi 2 export.]

## 2. Architecture

### 2.1 Cortical Column Circuits (CCCs)

The fundamental unit of Bio-ARN is the Concept Cell Cluster (CCC). A CCC contains: (i) an F1 sparse feature projection, (ii) an F2 concept-space activation, (iii) a margin gate that determines whether the current input matches the stored concept strongly enough to fire, and (iv) a top-down feedback pathway that supports resonance and refinement. Conceptually, the CCC is a compact cortical-column-like circuit with explicit abstention behavior.

This abstention behavior is important. Standard classifiers must map every input to one of a fixed set of classes. A CCC can instead emit "none of the above" when the margin gate is not crossed. If no committed CCC explains the input, an unused CCC can be recruited online through fast local learning. Once recruited, repeated resonance gradually refines the concept via slow local weight updates. Synaptic consolidation tracks how often a CCC becomes important and correspondingly reduces its effective learning rate, protecting frequently used concepts from being overwritten too aggressively.

The result is a unit that supports one-shot commitment, sparse competition, continual specialization, and partial stability-plasticity control without a global backward pass. In Bio-ARN, recognition is therefore open-set by construction rather than purely post hoc.

[Figure 2: Internal CCC computation: F1 sparse encoding, F2 concept activation, margin-gate abstention/firing, feedback prediction, and Hebbian/STDP refinement.]

### 2.2 Visual hierarchy: V1 -> V2 -> V4 -> IT

For vision, CCCs are organized into a four-stage hierarchy inspired by the ventral stream: V1, V2, V4, and inferotemporal-like (IT) representations [Riesenhuber and Poggio, 1999]. Inputs are first decomposed into receptive-field patches, then processed layer-by-layer with progressively larger effective receptive fields and more abstract concept vectors.

Each layer is backed by a CCC pool with its own dimensionality, threshold, and winner limit. Lower layers emphasize local feature capture; higher layers summarize increasingly abstract combinations. Intermediate representations are aggregated using confidence-weighted concept summaries. When abstention pressure remains high, pools can expand dynamically up to 3x their initial size, making capacity a controllable resource rather than a fixed ceiling.

This hierarchy is not a conventional deep network trained end-to-end. Each level learns with local rules and its own competitive dynamics. The hierarchy therefore trades gradient-optimized accuracy for online specialization and sparse routing.

### 2.3 Spatial attention and lateral inhibition

Bio-ARN includes an explicit spatial attention stage in the visual hierarchy. Patch gains are modulated by local contrast, edge content, chromatic variation, and center bias, creating a lightweight saliency prior before concept competition. This resembles an early sensory gating mechanism rather than full self-attention.

Within and across layers, lateral inhibition reduces redundant co-activations. Candidate concepts are sorted by confidence and greedily filtered when their cosine similarity exceeds a threshold. In practice, this yields winner-take-most behavior: multiple compatible concepts can survive, but highly overlapping ones are suppressed. The system remains sparse not only because few units fire, but because similar candidates are actively prevented from co-dominating.

### 2.4 Predictive coding and top-down feedback

A predictive hierarchy is attached to the feedforward visual path. Instead of propagating all activity upward, Bio-ARN iteratively settles hidden states to minimize local prediction error and free energy [Rao and Ballard, 1999; Friston, 2010]. Predictable structure is suppressed, while residual error is emphasized.

This predictive path is now coupled to explicit top-down feedback connections from IT -> V4 -> V2 -> V1. Higher-level summaries are projected back to lower levels and applied as multiplicative gating on candidate activations. In effect, higher layers bias lower layers toward interpretations that are globally coherent while still allowing abstention when no concept matches well. In Sprint D, the same predictive machinery also operates in an **error-gating** mode: prediction error is treated as a local learning gate, so novel or poorly predicted features receive larger Hebbian updates while already predictable structure is down-weighted.

Unlike backpropagation, these updates are local to adjacent levels and rely on state/error interactions rather than end-to-end gradient transport. In Bio-ARN, predictive coding is therefore both a representational hypothesis and an optimization constraint.

### 2.5 Global Neuronal Workspace

Bio-ARN uses a Global Neuronal Workspace (GNW) as a small-capacity bottleneck for competition, broadcast, and serial thought routing [Baars, 1988; Dehaene et al., 1998]. Active CCCs compete for entry into workspace slots. Occupants decay over time, accumulate fatigue, and may be displaced by stronger newcomers. The dominant contents are then broadcast to downstream consumers and to the system's own contextual buffer.

An enhanced workspace variant augments the short-term slot mechanism with a longer context buffer and spike-based attention over recent concepts. This lets the system bias current recognition toward recently relevant concepts while still maintaining a strict broadcast bottleneck.

The GNW contributes two useful behaviors. First, it offers a compact mechanism for selective access and internal routing that is missing from purely feedforward sparse classifiers. Second, it provides additional signals for uncertainty and OOD scoring: inputs that fail to stabilize in the workspace or fail to align with workspace context often receive weaker global support. Sprint D further turns this into a **GNW consensus classifier**: active CCCs vote under workspace context, the winning concept is read out through the broadcast, and normalized broadcast strength is recycled as a learning gate so low-consensus samples stay plastic while high-consensus samples consolidate.

### 2.6 Ensemble pool and OOD detection

Bio-ARN's ensemble pool trains multiple CCC-based experts with diverse preprocessing and per-expert perturbations. Predictions are combined with weighted voting, Hebbian-style boosting, abstention-aware agreement scoring, and prototype-based label readout.

This ensemble design is central to Bio-ARN's robustness story. OOD detection is not treated as a separate calibration model bolted onto a closed-set classifier; instead, it emerges from abstention fractions, confidence margins, expert disagreement, and workspace support. Ensemble OOD reaches 0.861 AUROC, GNW workspace scoring reaches 0.868 AUROC, and the best combined configuration reaches 1.000 AUROC. These gains suggest that sparse concept competition plus abstention provides a useful substrate for uncertainty estimation.

### 2.7 Multi-modal binding

Bio-ARN supports multimodal learning by alternating paired visual and textual observations through a shared CCC space. When possible, image and text examples are attached to the same CCC identity; otherwise a new shared concept is recruited and then associated across modalities. A lightweight feature-binding mechanism strengthens co-active cross-layer and cross-modal links through synchrony-like Hebbian updates.

This design differs from contrastive representation learning in transformers. Bio-ARN does not rely on large-batch negatives or a globally optimized embedding geometry. Instead, it seeks to form stable shared concept anchors that can be revisited, extended, and reused online.

### 2.8 Learning without backpropagation

All core learning in Bio-ARN is local. Fast concept recruitment initializes new CCCs. Slow Hebbian refinement updates committed concepts under resonance. Predictive layers update from local error terms. Associative memory strengthens temporal and co-activation links through STDP-like rules [Bi and Poo, 1998]. Reward-style novelty signals modulate learning rates but do not introduce a global backward pass.

This is the architecture's strongest differentiator and also one of its main risks. The upside is online learning, hardware friendliness, and biological plausibility. The downside is weaker optimization efficiency on tasks where backprop-trained deep models dominate.

### 2.9 Stability and exploration extensions

Several new components sharpen Bio-ARN's operating envelope:

- **STDP temporal dynamics** are opt-in and modulate Hebbian updates using pre/post spike traces on feedback pathways.
- **Top-down feedback connections** gate lower-layer activations multiplicatively, encouraging globally consistent interpretations.
- **Prediction-error gating** uses local surprise to amplify learning on novel features and suppress redundant updates on familiar ones.
- **Synaptic consolidation** now scores CCC importance with confidence- and recency-weighted usage, then reduces the learning rate of mature concepts.
- **Dynamic CCC pool growth** expands capacity when abstention remains persistently high.
- **Curriculum learning** orders the first pass toward easier, higher-confidence samples before harder cases are replayed.
- **Curiosity-driven replay** prioritizes novel, abstained, misclassified, or newly recruited samples for immediate replay.
- **A maturation schedule** freezes the shared F1 front-end after warmup, routes later adaptation through task adapters, and progressively shifts plasticity pressure toward newer CCCs.

These modules do not all target the same metric. Some help accuracy, some help OOD behavior, some stabilize temporal learning, and some improve hardware plausibility. That division of labor becomes important in the experimental results.

### 2.10 Concept locking and memory protection

Sprint E adds a stronger protection mechanism for mature concepts: **importance tracking -> threshold -> permanent lock**. Each CCC already accumulates an importance score from usage, confidence, and recency. When that score exceeds `lock_threshold`, the pool marks the CCC as locked and freezes the parameters most responsible for its identity: the `concept_direction` vector and the feedback weights that reconstruct or refine the concept. A locked CCC still participates in the forward pass, can still win competition, and can still classify familiar inputs, but it is skipped by subsequent local updates.

This matters because the newer continual-learning analysis points to a more specific failure mode than generic low-level drift. Shared F1 freezing was not enough. The direct damage occurs later in `learn_slow()`, where repeated local refinement can rotate a committed `concept_direction` and rewrite its feedback pathway toward newer tasks. Concept locking therefore targets the proximate root cause of catastrophic forgetting: once a CCC is consistently useful, it becomes a read-only detector. New concepts must then recruit previously uncommitted CCCs rather than repurposing old ones.

### 2.11 Precision-weighted predictive processing

Sprint E also replaces uniformly applied predictive plasticity with a selective precision mechanism grounded in recent neuroscience. Frank et al. (2026) report that human hippocampal ripples rise before uncertain stimuli and tune cortical responses by signaling predicted uncertainty [Frank et al., 2026]. Bio-ARN implements a computational analogue of that idea. A pool-level entropy estimator tracks how concentrated or diffuse recent CCC firing has been; a sigmoid transform converts that entropy into a precision value; and the resulting precision weights the learning-rate multiplier already used by the local Hebbian update path.

The mapping is direct. Hippocampal ripple signaling corresponds to the pool entropy estimator, which measures uncertainty in the recent CCC firing distribution. Precision weighting corresponds to the sigmoid transform that converts entropy into a learning gate. High uncertainty produces high precision, so surprising or weakly organized contexts learn faster. Low uncertainty produces low precision, so familiar contexts update more cautiously and protect existing memories.

This selective weighting addresses a concrete limitation in the earlier predictive-coding variants. Iterative settling smoothed bottom-up features too uniformly and drove CIFAR accuracy from 30.0% to 11.8%. Plain error gating was far less destructive, but it remained effectively neutral at roughly 30.0% because it treated all residual errors similarly. Precision weighting is more targeted: only high-entropy, novel contexts receive a large plasticity signal. In this sense it provides **soft protection**, while concept locking provides **hard protection** once a representation becomes mature.

### 2.12 Convolutional Concept Cell Clusters

Standard CCCs treat a 32 x 32 x 3 image as a flattened 3072-dimensional vector. That is simple and hardware-friendly, but it discards spatial adjacency before concept matching begins. Sprint E therefore adds **Convolutional CCCs**, implemented as a `ConvCCCPool` with a shared `ConvF1Layer` that applies local 2D convolutions, adaptive spatial pooling, and sparse top-k selection before the usual concept-space competition.

Crucially, the learning rule remains local. The convolutional filters are updated through correlation-based Hebbian rules over pre- and post-synaptic activity rather than gradient backpropagation, and the downstream concept updates still use the same CCC-style local refinement logic. The result is a more vision-appropriate front end that preserves Bio-ARN's bio-plausibility constraint while giving the architecture access to spatial feature extraction.

## 3. Neuromorphic Deployment

Bio-ARN includes a functional export path from trained CCC pools and visual hierarchies to a portable Loihi 2 graph. In this mapping, each CCC is decomposed into LIF populations corresponding to F1 feature neurons, F2 concept neurons, and a margin-gate readout. Feedback connections, local lateral inhibition, and winner-take-all competition are represented as explicit neuromorphic projections. Hierarchical exports further add feedforward layer-to-layer populations and projections.

The export format tracks population allocation, projection matrices, thresholds, delays, and metadata needed for downstream Lava/Loihi execution. This does not yet constitute a full claim of measured large-scale application performance on silicon. Rather, it establishes that the architecture was designed with deployable spike-compatible structure in mind, instead of treating neuromorphic mapping as a post hoc approximation.

The deployment path is now supported by direct validation checks. Export preserves weights exactly, round-trip reconstruction succeeds, and Lava-style simulation stays within 0.047 absolute accuracy of the original model. That is a stronger claim than "export exists": it shows that the graph representation is faithful enough to serve as a real deployment interface rather than a lossy visualization artifact.

From a systems perspective, this matters because Bio-ARN's central mechanisms—sparse event-driven activity, local plasticity, winner-take-all competition, recurrent context, and explicit feedback—map more naturally to neuromorphic substrates than dense attention blocks do [Davies et al., 2021].

[Figure 3: CCC-to-Loihi 2 mapping, showing F1/F2/gate populations, inhibitory competition, feedback projections, and hierarchical composition.]

## 4. Experiments and Results

### 4.1 Experimental framing

The current experimental suite should be read as architectural validation rather than a final scaling study. The codebase includes MNIST, real CIFAR-10, text generation, multimodal demos, continual-learning benchmarks, OOD analyses, scaling sweeps, and analytic energy reports. The most mature strengths are uncertainty estimation, local learning, and hardware-oriented efficiency. The most obvious weakness is raw visual accuracy on harder datasets. The numbers below reflect the current June 2026 repository state and now include the Sprint E retention-focused additions: concept locking, convolutional CCCs, and precision-weighted predictive processing.

### 4.2 CIFAR-10 progression and combined configurations

Real CIFAR-10 remains a challenging benchmark for purely local Hebbian learning. Nevertheless, architectural additions have improved performance over earlier hierarchy-only results.

| Configuration | Accuracy summary | OOD AUROC | Notes |
|---|---:|---:|---|
| Hierarchy baseline | 30.0% | - | reference four-layer hierarchy |
| Sprint D curiosity + curriculum (weight 0.8) | **33.8% online / 55.8% eval** | - | strongest single Sprint D accuracy lift |
| Curiosity sweep at 800 samples | 30.5% | - | intermediate data-scaling reference point |
| Best combined config at 2,000 samples | **33.0%** | **1.000** | best overall accuracy/robustness trade-off |

The absolute numbers remain modest compared with modern convolutional and transformer models. We emphasize this directly because it is important for honest positioning. Bio-ARN should currently be understood as a proof of architectural direction, not a state-of-the-art CIFAR classifier. Within that framing, Sprint D is informative: curriculum + curiosity is the most impactful single improvement for CIFAR sample efficiency, while GNW consensus and predictive gating mostly sharpen selectivity and robustness rather than producing a large additional top-1 jump on their own. Sprint E has not yet displaced the current best closed-set CIFAR configuration; the convolutional CCC path still needs tuning with the full hierarchy. Its most credible measured gain so far is in continual retention rather than top-1 accuracy.

### 4.3 OOD detection

OOD behavior is more encouraging than closed-set accuracy alone would suggest.

| Method | OOD metric |
|---|---:|
| Baseline confidence scoring | 0.778 AUROC |
| Ensemble OOD | **0.861 AUROC** |
| GNW workspace OOD | **0.868 AUROC** |
| Best combined configuration | **1.000 AUROC** |

These results support the view that abstention, expert disagreement, and workspace stability are useful uncertainty signals. In other words, Bio-ARN's sparse concept architecture appears to buy robustness earlier than it buys top-1 accuracy.

[Figure 4: Accuracy/OOD trade-off across hierarchy, curiosity-replay, GNW, ensemble, and best-combined configurations on CIFAR-10.]

### 4.4 Energy efficiency

Bio-ARN's energy claim is based on analytic profiling of actual tensor shapes, non-zero activity, and modeled hardware energy constants, not direct hardware-counter measurements. The strongest published comparison in the repository matches Bio-ARN against a benchmark transformer at roughly the same MNIST accuracy tier and then projects inference energy to Loihi 2 versus an A100 GPU.

| System | Energy per inference | Relative to transformer |
|---|---:|---:|
| Bio-ARN on projected Loihi 2 | **179.65 µJ** | **278x lower** |
| Transformer baseline on A100 | 50.01 mJ | 1x |

The same analysis reports approximately 8,050x lower projected online-training energy for Bio-ARN than dense transformer training in the benchmark setup. This result should be interpreted carefully. It is a hardware-software co-design argument, not a claim that the current dense PyTorch prototype is itself fast or efficient on CPUs. Indeed, the repository's own report explicitly notes that the current CPU implementation is slower than compact dense baselines due to software overheads.

### 4.5 Text generation quality

Bio-ARN also supports sequence generation through a dual character+word stack coupled to CCC concepts, sequence memory, and recurrent context. On the current text-generation benchmark, the dual-level setup reaches:

- **85.7% real-word rate**
- **0.34 repetition**

These numbers do not imply fluent transformer-like generation. The outputs remain short-horizon and structurally constrained. However, they do show that locally learned concept-memory dynamics can produce non-trivial generative behavior while maintaining strong lexical validity relative to the small training regime.

### 4.6 Continual learning

Continual learning is a core motivation for the architecture, and the new benchmark suite makes that strength-and-weakness profile much clearer.

#### 4.6.1 Methodology

We evaluate three sequential settings:

1. **Split-CIFAR-10**: five binary class-split tasks, e.g. (0/1), (2/3), ..., (8/9), trained sequentially with the visual hierarchy.
2. **Split-MNIST**: the same five-way binary class-split protocol on MNIST.
3. **Permuted-MNIST**: five sequential tasks sharing the same labels but each applying a distinct pixel permutation.

After each task, the current model is evaluated on all tasks seen so far. We report backward transfer (BWT) and mean forgetting. This matters because it separates two qualitatively different continual-learning demands: learning new class partitions versus adapting to new input transformations.

#### 4.6.2 Results

| Benchmark | Task type | Backward transfer | Mean forgetting | Interpretation |
|---|---|---:|---:|---|
| Split-CIFAR-10 | class splits on natural images | **-29.0 pts** | **23.2 pts** | slight improvement; error gating + stronger consolidation help, but capacity pressure remains |
| Split-MNIST | class splits on digits | **-54.3%** | **43.5%** | worse than the earlier baseline; aggressive plasticity still destabilizes class ownership |
| Permuted-MNIST | input permutations with shared labels | **-7.5%** | **6.5%** | still far milder than class splits, but slightly worse than the previous run |

The updated retest preserves the same qualitative pattern: forgetting is strongly task-type-dependent. Bio-ARN still struggles most when new tasks introduce new class partitions that compete for CCC capacity and prototype ownership. It degrades much less when the task sequence preserves the label semantics but changes the surface statistics, as in Permuted-MNIST.

#### 4.6.3 Sprint E update

Sprint E specifically targets the class-split forgetting problem on Split-CIFAR-10. The new combined convolutional+locking path reduces mean forgetting from **34.7%** to **20.7%**, which is the clearest retention gain yet observed from a single architectural change set. This should be interpreted as a genuine but still partial success: catastrophic forgetting is substantially lower, not eliminated.

Three additional observations matter. First, the precision-weighting trace behaves as intended: pool entropy and the effective learning-rate multiplier move together, so uncertainty is modulating plasticity rather than merely being logged. Second, CCCs are being auto-locked during training once their importance exceeds threshold, confirming that the hard-protection mechanism actually engages online. Third, the same benchmark does not yet show a decisive accuracy win for the convolutional path, so the current claim is narrower: Sprint E improves retention and representational selectivity, while accuracy tuning with hierarchy integration remains ongoing.

#### 4.6.4 Analysis and mitigation

The stronger Sprint E result refines the root-cause analysis. Capacity pressure still matters, but the more direct failure mode is **concept drift inside already committed CCCs**. In `learn_slow()`, mature `concept_direction` vectors and feedback pathways can still rotate toward later tasks, which erodes earlier class ownership even when the front end is frozen. That failure mode is much weaker on permutation benchmarks because the underlying categories remain aligned.

Across Sprints D and E, the main mitigation mechanisms are now:

- **Prediction-error gating**, which increases local learning on surprising features and suppresses redundant updates.
- **Curiosity + curriculum**, which is highly effective for CIFAR sample efficiency but does not automatically translate into retention.
- **GNW consensus learning gates**, which reduce updates on already well-broadcast concepts and keep low-consensus samples plastic.
- **Precision weighting**, which uses pool entropy to scale plasticity up in uncertain contexts and down in familiar ones.
- **Concept locking**, which permanently freezes mature concept directions and feedback weights once importance crosses threshold.
- **Convolutional CCCs**, which preserve spatial structure so task-specific visual features do not have to be relearned from flat vectors alone.

What helped: Sprint E's conv+locking path reduces mean forgetting from 34.7% to 20.7%, a materially larger gain than the earlier Sprint D-only improvements. This supports the causal hypothesis that protecting mature concept ownership matters more than simply making low-level plasticity selective.

What did not help enough: accuracy on the convolutional path is still under-tuned, and the broader class-split problem has not disappeared. The most plausible explanation is that selective plasticity and locking protect stored concepts, but concept allocation, label binding, and hierarchy-level calibration still determine whether new tasks can be absorbed cleanly. We therefore frame continual learning not as a solved capability, but as the project's clearest open research challenge.

### 4.7 MNIST and deployment maturity

On MNIST, Bio-ARN reaches **85.2% accuracy**, which remains a useful moderate-complexity validation target for online sparse learning. Together with the Loihi/Lava round-trip checks, this suggests that the architecture's implementation maturity is strongest on small-to-medium tasks where local learning and structured sparsity are easier to stabilize.

### 4.8 Scaling laws

Scaling sweeps reveal useful operating points rather than simple monotonic improvement.

| Sweep | Key finding | Practical implication |
|---|---|---|
| CCC pool size | performance saturates at **100** | larger pools increase capacity headroom but do not keep improving accuracy |
| Hierarchy depth | **4 layers** is optimal | V1 -> V2 -> V4 -> IT is the current sweet spot |
| Expert count | **3 experts** is best | more experts add cost faster than they add robustness |
| Data volume | more data helps roughly linearly; **2,000 samples** is best | the architecture is still data-limited rather than fully saturated |

The data-volume sweep is particularly important. Moving from smaller training sets to 2,000 samples continues to improve CIFAR behavior, and the curiosity-driven 800-sample point still reaches 30.5%, suggesting that the current system benefits from both more data and better sample prioritization. These are empirical scaling laws for the present Bio-ARN implementation, not universal laws, but they provide a concrete guide for future tuning.

[Figure 5: Scaling-law summary across pool size, hierarchy depth, expert count, and data volume.]

## 5. Related Work

Bio-ARN intersects several research traditions but does not fit neatly into any single one.

**Transformers and dense sequence models.** Modern language and vision systems are dominated by transformer architectures [Vaswani et al., 2017; Brown et al., 2020; Dosovitskiy et al., 2021]. Relative to these models, Bio-ARN is currently weaker in raw accuracy and generative fluency. Its intended advantages are different: local online learning, sparse activation, OOD-aware abstention, and compatibility with spike-native hardware.

**Neuromorphic computing.** Platforms such as SpiNNaker and BrainScaleS have long argued for event-driven, massively parallel neural computation [Furber et al., 2014; Schmitt et al., 2017]. Intel's Loihi and Loihi 2 further demonstrate programmable neuromorphic substrates with on-chip learning support [Davies et al., 2018; Davies et al., 2021]. Bio-ARN contributes at the architecture level by offering a higher-level cognitive stack expressly shaped for such hardware.

**Hebbian and local-learning systems.** Hebbian learning and STDP have deep biological and computational roots [Hebb, 1949; Bi and Poo, 1998]. More recent work on local-learning alternatives to backpropagation has explored predictive coding, equilibrium methods, and contrastive local objectives. Bio-ARN belongs to this family but is distinctive in combining local concept recruitment, associative memory, workspace dynamics, and deployment-oriented sparsity in one framework.

**Predictive coding.** Predictive coding theories propose that cortical systems minimize prediction error through hierarchical feedback and residual signaling [Rao and Ballard, 1999; Friston, 2010]. Bio-ARN adopts this logic explicitly, using local prediction error to suppress predictable structure and highlight novelty. Sprint E further grounds the learning-gate variant of that idea in recent evidence that hippocampal ripples can tune cortical responses as a function of predicted uncertainty [Frank et al., 2026].

**Global workspace theories.** Global workspace theory and the Global Neuronal Workspace provide a computational story for limited-capacity access, conscious broadcast, and serial integration [Baars, 1988; Dehaene and Changeux, 2011]. Bio-ARN does not claim consciousness. It uses the GNW idea instrumentally, as a bottlenecked global routing mechanism for sparse concepts.

## 6. Discussion

The new results sharpen the paper's main message. Bio-ARN is **not** currently competing on raw CIFAR-10 accuracy, and it should not be presented that way. The more defensible claim is that the architecture combines several properties that are usually studied separately: local online learning, OOD-aware abstention, explicit continual-learning machinery, and a real neuromorphic deployment path.

The combined-feature finding is especially important. Individual modules serve different purposes:

- curiosity-driven replay helps sample efficiency and online adaptation;
- curriculum learning makes that replay materially more effective on CIFAR;
- GNW and ensemble signals help OOD detection;
- prediction-error gating validates the hypothesis that prediction should also act as a learning gate, not only as an inference aid;
- STDP and feedback improve temporal/contextual dynamics;
- consolidation and growth target stability-plasticity;
- the export stack targets energy and hardware deployment.

These modules do **not** simply compound into ever-higher raw accuracy. That is not a flaw in the framing; it is the central result. Bio-ARN's value is the **combination of properties** rather than dominance on any single standard benchmark.

Sprint E sharpens that interpretation rather than overturning it. Precision weighting and concept locking divide stability protection into two complementary regimes: uncertainty-dependent **soft** plasticity control and importance-triggered **hard** memory protection. Convolutional CCCs address a different bottleneck by giving the architecture a spatially structured visual front end without abandoning local learning.

No other system in the current project scope provides all of the following simultaneously in one executable stack:

1. projected neuromorphic energy efficiency,
2. OOD-aware abstention and uncertainty signals,
3. online local learning without backpropagation,
4. explicit continual-learning machinery, and
5. a validated Loihi/Lava export path.

The Frank et al. connection is especially important for how this project now positions itself. Bio-ARN is no longer using predictive coding only as a generic metaphor for top-down error suppression. It now implements a computational analogue of hippocampal precision signaling: uncertainty in recent concept usage modulates how strongly new errors change the system. That creates a more principled bridge between computational neuroscience and neuromorphic engineering. In practical terms, the precision gate offers a neuroscience-grounded way to manage the stability-plasticity dilemma, while concept locking provides a complementary engineering safeguard once a representation is judged mature.

### 6.1 Limitations

Bio-ARN's limitations are equally clear:

1. **CIFAR-10 accuracy is still low.** Even the best combined configuration peaks at 33.0%, and the most optimistic replay-enhanced result is still far below mainstream deep-learning baselines.
2. **Continual learning is highly task-dependent.** Class-split benchmarks show severe forgetting, especially Split-MNIST and Split-CIFAR-10.
3. **The CCC pool is still a bottleneck.** Dynamic growth and consolidation help, but class-split tasks can still saturate useful concept capacity.
4. **OOD gains do not guarantee accuracy gains.** The best robustness configurations are not always the best classifiers.
5. **Energy gains are projected, not yet measured end-to-end on deployed Loihi 2 workloads.**
6. **The current CPU prototype is not optimized.** Hardware-oriented claims should not be confused with present dense-software runtime performance.

Despite these limitations, Bio-ARN suggests a valuable research direction. If the field's next frontier includes efficient on-device intelligence, open-set robustness, and low-power continual adaptation, then architectures like Bio-ARN deserve study even before they match dense backprop-trained models on conventional benchmarks.

Promising future directions include:

- stronger capacity allocation and consolidation for class-split continual learning;
- tighter hierarchy integration for convolutional CCCs so the retention gain also translates into stronger closed-set accuracy;
- protecting F2 / prototype ownership directly rather than only freezing the shared F1 front-end;
- class-aware replay anchors or reserved concept slots for previously learned tasks;
- letting GNW consensus gate not just learning rate but also whether new CCC recruitment is permitted;
- larger multimodal datasets and richer grounding tasks;
- improved generative decoding beyond short-horizon lexical validity;
- measured silicon experiments on Loihi 2 hardware;
- hybrid training schemes that preserve locality while improving optimization.

## 7. Conclusion

Bio-ARN 2.0 is a research architecture built around a simple but ambitious hypothesis: useful intelligence for edge and neuromorphic settings may require a different stack than the one optimized for large datacenter transformers. By combining CCC-based sparse concepts, concept locking, precision-weighted predictive coding, convolutional local feature extractors, top-down feedback, workspace broadcast, multimodal binding, ensemble uncertainty, and local Hebbian learning, Bio-ARN offers a concrete alternative design.

The current evidence is mixed in the right way. Bio-ARN is not yet a high-accuracy vision model, and its continual-learning story remains incomplete on class-split tasks. But it already shows promising OOD behavior, online learning capability, functional neuromorphic export, and compelling projected energy efficiency. We therefore view it as an architecture worth scaling and stress-testing, especially for domains where efficiency, robustness, continual adaptation, and deployability matter as much as raw benchmark score.

[Figure 6: Summary comparison positioning Bio-ARN against transformer-class systems across accuracy, energy, OOD robustness, continual learning, and biological plausibility.]

## 8. References

Frank, D., Moratti, S., Hellerstedt, R., Sarnthein, J., Li, N., Horn, A., Imbach, L., Stieglitz, L., Gil-Nagel, A., Toledano, R., Friston, K. J., & Strange, B. A. (2026). Human hippocampal ripples tune cortical responses based on predicted uncertainty. *Nature Neuroscience*. https://doi.org/10.1038/s41593-026-02345-6
