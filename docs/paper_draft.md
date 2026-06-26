# Bio-ARN: A Brain-Inspired Architecture for Energy-Efficient Multi-Modal Learning on Neuromorphic Hardware

## Abstract

Modern generative AI systems are increasingly constrained by energy cost, brittle out-of-distribution (OOD) behavior, and reliance on gradient-based batch training. Bio-ARN 2.0 explores a different design point: a brain-inspired architecture built around sparse concept circuits, local Hebbian learning, predictive coding, workspace-style global broadcasting, and a deployment path to neuromorphic hardware. The core computational unit is the Concept Cell Cluster (CCC), a cortical-column-inspired module that can abstain, recruit new concepts online, and refine them through local resonance rather than backpropagation. CCCs are organized into a ventral-stream-like visual hierarchy, linked through predictive coding, and coupled to a Global Neuronal Workspace (GNW) for competition, broadcast, and multi-step internal routing. Bio-ARN further supports ensemble-based uncertainty estimation and shared cross-modal concept binding for image-text learning.

The current system does not match state-of-the-art transformer accuracy on large-scale visual benchmarks. However, it shows promising properties along other axes. On real CIFAR-10, the best current hierarchy+ensemble configuration reaches 30.0% accuracy, while ensemble OOD detection reaches 0.861 AUROC versus a 0.778 baseline, and GNW workspace scoring reaches 0.868 AUROC. On MNIST, the architecture reaches 85.2% accuracy. In text generation, a dual character+word setup achieves 85.7% real-word rate with 0.34 repetition. A functional Loihi 2 export path maps CCC pools and visual hierarchies to leaky integrate-and-fire (LIF) populations. Analytic energy modeling projects 179.65 µJ per inference on Loihi 2 versus 50.01 mJ for a transformer baseline on an A100 GPU, a 278x advantage at a matched benchmark tier. These results position Bio-ARN not as a drop-in replacement for transformers, but as a candidate architecture for efficient, robust, continuously learning neuromorphic AI.

## 1. Introduction

Large foundation models have demonstrated remarkable capability, but their dominant recipe remains expensive and brittle: dense matrix multiplications, backpropagation-heavy optimization, and high inference energy on von Neumann hardware [Brown et al., 2020; Dosovitskiy et al., 2021]. At the same time, these systems often struggle with calibrated uncertainty, out-of-distribution detection, and online continual adaptation. Biological nervous systems solve a different problem under radically different constraints: sparse activity, local plasticity, continual updating, and tight energy budgets [Hebb, 1949; Friston, 2010].

Bio-ARN 2.0 asks whether some of those constraints can be elevated from inspiration to systems design. Rather than treating biological plausibility as an aesthetic goal, Bio-ARN uses it to guide architecture: sparse concept detectors, predictive suppression, local associative memory, broadcast-limited global access, and event-driven deployment assumptions. The result is a multi-module cognitive stack intended for online learning and neuromorphic execution rather than maximum leaderboard accuracy.

The project's central claim is therefore not that brain-inspired learning already outperforms transformer-scale systems on raw supervised accuracy. It does not. Instead, the claim is that a unified architecture can combine four properties that are rarely demonstrated together in one executable system: (1) local Hebbian learning without backpropagation, (2) practical OOD-aware abstention and uncertainty signals, (3) multimodal concept binding and generation, and (4) a concrete export path toward neuromorphic hardware.

This paper makes the following contributions:

1. It presents **Concept Cell Clusters (CCCs)** as the basic sparse computational unit for online recruitment, abstention, and refinement.
2. It integrates CCCs into a **visual ventral-stream hierarchy**, a **predictive coding stack**, a **Global Neuronal Workspace**, and an **ensemble OOD mechanism**.
3. It demonstrates a **multi-modal learning path** in which shared CCCs bind visual and textual signals through alternating local updates.
4. It provides a **functional Loihi 2 export path** and an analytic energy study suggesting large projected efficiency gains over dense transformer baselines.

[Figure 1: End-to-end Bio-ARN pipeline, from sensory encoding through predictive coding, CCC hierarchies, GNW broadcast, ensemble/OOD heads, and Loihi 2 export.]

## 2. Architecture

### 2.1 Cortical Column Circuits (CCCs)

The fundamental unit of Bio-ARN is the Concept Cell Cluster (CCC). A CCC contains: (i) an F1 sparse feature projection, (ii) an F2 concept-space activation, (iii) a margin gate that determines whether the current input matches the stored concept strongly enough to fire, and (iv) a top-down feedback pathway that supports resonance and refinement. Conceptually, the CCC is a compact cortical-column-like circuit with explicit abstention behavior.

This abstention behavior is important. Standard classifiers must map every input to one of a fixed set of classes. A CCC can instead emit "none of the above" when the margin gate is not crossed. If no committed CCC explains the input, an unused CCC can be recruited online through fast local learning. Once recruited, repeated resonance gradually refines the concept via slow local weight updates.

The result is a unit that supports one-shot commitment, sparse competition, and continual specialization without a global backward pass. In Bio-ARN, recognition is therefore open-set by construction rather than purely post hoc.

[Figure 2: Internal CCC computation: F1 sparse encoding, F2 concept activation, margin-gate abstention/firing, feedback prediction, and Hebbian refinement.]

### 2.2 Visual hierarchy: V1 -> V2 -> V4 -> IT

For vision, CCCs are organized into a four-stage hierarchy inspired by the ventral stream: V1, V2, V4, and inferotemporal-like (IT) representations [Riesenhuber and Poggio, 1999]. Inputs are first decomposed into receptive-field patches, then processed layer-by-layer with progressively larger effective receptive fields and more abstract concept vectors.

Each layer is backed by a CCC pool with its own dimensionality, threshold, and winner limit. Lower layers emphasize local feature capture; higher layers summarize increasingly abstract combinations. Intermediate representations are aggregated using confidence-weighted concept summaries, and optional adaptive capacity mechanisms allow layer pools to grow or prune when abstention pressure becomes sustained.

This hierarchy is not a conventional deep network trained end-to-end. Each level learns with local rules and its own competitive dynamics. The hierarchy therefore trades gradient-optimized accuracy for online specialization and sparse routing.

### 2.3 Spatial attention and lateral inhibition

Bio-ARN includes an explicit spatial attention stage in the visual hierarchy. Patch gains are modulated by local contrast, edge content, chromatic variation, and center bias, creating a lightweight saliency prior before concept competition. This resembles an early sensory gating mechanism rather than full self-attention.

Within and across layers, lateral inhibition reduces redundant co-activations. Candidate concepts are sorted by confidence and greedily filtered when their cosine similarity exceeds a threshold. In practice, this yields winner-take-most behavior: multiple compatible concepts can survive, but highly overlapping ones are suppressed. The system remains sparse not only because few units fire, but because similar candidates are actively prevented from co-dominating.

### 2.4 Predictive coding integration

A predictive hierarchy is attached to the feedforward visual path. Instead of propagating all activity upward, Bio-ARN iteratively settles hidden states to minimize local prediction error and free energy [Rao and Ballard, 1999; Friston, 2010]. Predictable structure is suppressed, while residual error is emphasized.

This matters for both efficiency and selectivity. If lower-level patterns are already well explained, less activity needs to reach the concept system. Predictive coding thus acts as a sparsity engine, a novelty signal, and a source of top-down context. The hierarchy also supports generation by cascading higher-level states back into lower sensory predictions.

Unlike backpropagation, these updates are local to adjacent levels and rely on state/error interactions rather than end-to-end gradient transport. In Bio-ARN, predictive coding is therefore both a representational hypothesis and an optimization constraint.

### 2.5 Global Neuronal Workspace

Bio-ARN uses a Global Neuronal Workspace (GNW) as a small-capacity bottleneck for competition, broadcast, and serial thought routing [Baars, 1988; Dehaene et al., 1998]. Active CCCs compete for entry into workspace slots. Occupants decay over time, accumulate fatigue, and may be displaced by stronger newcomers. The dominant contents are then broadcast to downstream consumers and to the system's own contextual buffer.

An enhanced workspace variant augments the short-term slot mechanism with a longer context buffer and spike-based attention over recent concepts. This lets the system bias current recognition toward recently relevant concepts while still maintaining a strict broadcast bottleneck.

The GNW contributes two useful behaviors. First, it offers a compact mechanism for selective access and internal routing that is missing from purely feedforward sparse classifiers. Second, it provides additional signals for uncertainty and OOD scoring: inputs that fail to stabilize in the workspace or fail to align with workspace context often receive weaker global support.

### 2.6 Ensemble pool and OOD detection

Bio-ARN's ensemble pool trains multiple CCC-based experts with diverse preprocessing and per-expert perturbations. Predictions are combined with weighted voting, optional Hebbian-style boosting, abstention-aware agreement scoring, and prototype-based label readout.

This ensemble design is central to Bio-ARN's robustness story. OOD detection is not treated as a separate calibration model bolted onto a closed-set classifier; instead, it emerges from abstention fractions, confidence margins, expert disagreement, and workspace support. The current ensemble OOD AUROC of 0.861 exceeds a baseline 0.778, while GNW workspace scoring reaches 0.868. These gains suggest that sparse concept competition plus abstention provides a useful substrate for uncertainty estimation.

### 2.7 Multi-modal binding

Bio-ARN supports multimodal learning by alternating paired visual and textual observations through a shared CCC space. When possible, image and text examples are attached to the same CCC identity; otherwise a new shared concept is recruited and then associated across modalities. A lightweight feature-binding mechanism strengthens co-active cross-layer and cross-modal links through synchrony-like Hebbian updates.

This design differs from contrastive representation learning in transformers. Bio-ARN does not rely on large-batch negatives or a globally optimized embedding geometry. Instead, it seeks to form stable shared concept anchors that can be revisited, extended, and reused online.

### 2.8 Learning without backpropagation

All core learning in Bio-ARN is local. Fast concept recruitment initializes new CCCs. Slow Hebbian refinement updates committed concepts under resonance. Predictive layers update from local error terms. Associative memory strengthens temporal and co-activation links through STDP-like rules [Bi and Poo, 1998]. Reward-style novelty signals modulate learning rates but do not introduce a global backward pass.

This is the architecture's strongest differentiator and also one of its main risks. The upside is online learning, hardware friendliness, and biological plausibility. The downside is weaker optimization efficiency on tasks where backprop-trained deep models dominate.

## 3. Neuromorphic Deployment

Bio-ARN includes a functional export path from trained CCC pools and visual hierarchies to a portable Loihi 2 graph. In this mapping, each CCC is decomposed into LIF populations corresponding to F1 feature neurons, F2 concept neurons, and a margin-gate readout. Feedback connections, local lateral inhibition, and winner-take-all competition are represented as explicit neuromorphic projections. Hierarchical exports further add feedforward layer-to-layer populations and projections.

The export format tracks population allocation, projection matrices, thresholds, delays, and metadata needed for downstream Lava/Loihi execution. This does not yet constitute a full claim of measured large-scale application performance on silicon. Rather, it establishes that the architecture was designed with deployable spike-compatible structure in mind, instead of treating neuromorphic mapping as a post hoc approximation.

From a systems perspective, this matters because Bio-ARN's central mechanisms—sparse event-driven activity, local plasticity, winner-take-all competition, and recurrent context—map more naturally to neuromorphic substrates than dense attention blocks do [Davies et al., 2021].

[Figure 3: CCC-to-Loihi 2 mapping, showing F1/F2/gate populations, inhibitory competition, feedback projections, and hierarchical composition.]

## 4. Experiments and Results

### 4.1 Experimental framing

The current experimental suite should be read as architectural validation rather than a final scaling study. The codebase includes MNIST, real CIFAR-10, text generation, multimodal demos, continual-learning benchmarks, OOD analyses, and analytic energy reports. The most mature strengths are uncertainty estimation, local learning, and hardware-oriented efficiency. The most obvious weakness is raw visual accuracy on harder datasets. The numbers below combine repository-documented benchmarks with the latest internal project metrics available for the June 2026 draft.

### 4.2 CIFAR-10 progression

Real CIFAR-10 remains a challenging benchmark for purely local Hebbian learning. Nevertheless, architectural additions have improved performance over the flat baseline.

| Configuration | Accuracy | Notes |
|---|---:|---|
| Flat online pool | lower than hierarchy variants | used mainly as a control for OOD and architecture ablations |
| Visual hierarchy | 26.4% | best confirmed hierarchy-only result in repository documentation |
| Hierarchy + ensemble | **30.0%** | current best reported real CIFAR-10 configuration |

The absolute number is modest compared with modern convolutional and transformer models. We emphasize this directly because it is important for honest positioning. Bio-ARN should currently be understood as a proof of architectural direction, not a state-of-the-art CIFAR classifier.

### 4.3 OOD detection

OOD behavior is more encouraging than closed-set accuracy alone would suggest.

| Method | OOD metric |
|---|---:|
| Baseline confidence scoring | 0.778 AUROC |
| Ensemble OOD | **0.861 AUROC** |
| GNW workspace OOD | **0.868 AUROC** |

These results support the view that abstention, expert disagreement, and workspace stability are useful uncertainty signals. In other words, Bio-ARN's sparse concept architecture appears to buy robustness earlier than it buys top-1 accuracy.

[Figure 4: Accuracy/OOD trade-off across baseline, hierarchy, ensemble, and hierarchy+ensemble configurations on CIFAR-10.]

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

Continual learning is a core motivation for the architecture, but the current results expose an important limitation. The benchmark suite shows a capacity bottleneck and substantial negative backward transfer, with current reported backward transfer at **-29.2 points**. This indicates that although Bio-ARN learns online, its present memory allocation and interference controls are not yet sufficient for strong sequential task retention at scale.

We consider this an encouraging failure mode: the system is revealing exactly where a no-backprop continual learner must improve. Future work should target better capacity management, stronger replay-free consolidation, and more explicit separation between stable and plastic concept subsets.

### 4.7 MNIST and export maturity

On MNIST, Bio-ARN reaches **85.2% accuracy**, which is useful as a moderate-complexity validation target for online sparse learning. The repository also reports a **functional Loihi 2 export path**, showing that the hardware mapping is executable rather than purely conceptual.

### 4.8 Scaling analysis

Scaling remains an open question.

- The current architecture benefits from sparsity as pool size grows.
- Larger CCC pools can, in principle, maintain low activation fractions even as capacity expands.
- A fuller scaling section should incorporate the forthcoming large-pool and systems-level analysis from Tank's experiments.

[Figure 5: Placeholder scaling plot showing accuracy, active CCC count, activation fraction, and projected sparse savings as pool capacity increases.]

## 5. Related Work

Bio-ARN intersects several research traditions but does not fit neatly into any single one.

**Transformers and dense sequence models.** Modern language and vision systems are dominated by transformer architectures [Vaswani et al., 2017; Brown et al., 2020; Dosovitskiy et al., 2021]. Relative to these models, Bio-ARN is currently weaker in raw accuracy and generative fluency. Its intended advantages are different: local online learning, sparse activation, OOD-aware abstention, and compatibility with spike-native hardware.

**Neuromorphic computing.** Platforms such as SpiNNaker and BrainScaleS have long argued for event-driven, massively parallel neural computation [Furber et al., 2014; Schmitt et al., 2017]. Intel's Loihi and Loihi 2 further demonstrate programmable neuromorphic substrates with on-chip learning support [Davies et al., 2018; Davies et al., 2021]. Bio-ARN contributes at the architecture level by offering a higher-level cognitive stack expressly shaped for such hardware.

**Hebbian and local-learning systems.** Hebbian learning and STDP have deep biological and computational roots [Hebb, 1949; Bi and Poo, 1998]. More recent work on local-learning alternatives to backpropagation has explored predictive coding, equilibrium methods, and contrastive local objectives. Bio-ARN belongs to this family but is distinctive in combining local concept recruitment, associative memory, workspace dynamics, and deployment-oriented sparsity in one framework.

**Predictive coding.** Predictive coding theories propose that cortical systems minimize prediction error through hierarchical feedback and residual signaling [Rao and Ballard, 1999; Friston, 2010]. Bio-ARN adopts this logic explicitly, using local prediction error to suppress predictable structure and highlight novelty.

**Global workspace theories.** Global workspace theory and the Global Neuronal Workspace provide a computational story for limited-capacity access, conscious broadcast, and serial integration [Baars, 1988; Dehaene and Changeux, 2011]. Bio-ARN does not claim consciousness. It uses the GNW idea instrumentally, as a bottlenecked global routing mechanism for sparse concepts.

## 6. Discussion

Bio-ARN's strengths are architectural coherence, strong inductive structure, and explicit attention to efficiency. The system combines online learning, abstention, predictive suppression, multimodal concept sharing, workspace broadcasting, and neuromorphic export in a single executable codebase. That combination is unusual and scientifically useful even where individual task performance is still limited.

Its limitations are equally clear:

1. **CIFAR-10 accuracy is still low.** A 30.0% result is meaningful as progress within this paradigm, but not competitive with mainstream deep learning.
2. **Continual learning is not solved.** Negative backward transfer remains substantial.
3. **Energy gains are projected, not end-to-end measured on deployed Loihi 2 workloads.**
4. **The current CPU prototype is not optimized.** Hardware-oriented claims should not be confused with present dense-software runtime performance.

Despite these limitations, Bio-ARN suggests a valuable research direction. If the field's next frontier includes efficient on-device intelligence, open-set robustness, and low-power continual adaptation, then architectures like Bio-ARN deserve study even before they match dense backprop-trained models on conventional benchmarks.

Promising future directions include:

- stronger capacity allocation and consolidation for continual learning;
- larger multimodal datasets and richer grounding tasks;
- improved generative decoding beyond short-horizon lexical validity;
- measured silicon experiments on Loihi 2 hardware;
- hybrid training schemes that preserve locality while improving optimization.

## 7. Conclusion

Bio-ARN 2.0 is a research architecture built around a simple but ambitious hypothesis: useful intelligence for edge and neuromorphic settings may require a different stack than the one optimized for large datacenter transformers. By combining CCC-based sparse concepts, predictive coding, workspace broadcast, multimodal binding, ensemble uncertainty, and local Hebbian learning, Bio-ARN offers a concrete alternative design.

The current evidence is mixed in the right way. Bio-ARN is not yet a high-accuracy vision model. But it already shows promising OOD behavior, online learning capability, functional neuromorphic export, and compelling projected energy efficiency. We therefore view it as an architecture worth scaling and stress-testing, especially for domains where efficiency, robustness, and continual adaptation matter as much as raw benchmark score.

[Figure 6: Summary comparison positioning Bio-ARN against transformer-class systems across accuracy, energy, OOD robustness, continual learning, and biological plausibility.]
