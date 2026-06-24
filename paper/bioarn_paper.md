# Bio-ARN 2.0: A Brain-Inspired Architecture for Honest, Efficient, and Continual Intelligence

## Abstract

Current AI systems are powerful but brittle. They typically rely on global backpropagation, must be trained offline in large batches, tend to catastrophically forget earlier knowledge when adapted online, provide poorly calibrated confidence, and consume substantial energy during both training and inference. Bio-ARN 2.0 explores an alternative design point inspired by cortical computation rather than by scaling dense attention. The architecture combines spiking neurons, margin-gated concept cell clusters (CCCs), sparse distributed memory, predictive coding, a small global neuronal workspace, and reward-modulated local learning. Its central design principle is that uncertainty should be structural rather than cosmetic: when a concept cell does not match strongly enough, it abstains instead of emitting a forced guess.

Across the bundled benchmark suite, Bio-ARN reaches 82.0% MNIST accuracy, 76.7% out-of-distribution abstention, 0.933 AUROC, and 3.2% class-incremental forgetting while using only about 295k active MACs per inference versus 4.43M for a matched tiny transformer. The broader project brief summarizes similar Phase-0 behavior at 81.2% accuracy, 81.7% abstention, 3.4% forgetting, and 9.82% activation density. On text tasks, Bio-ARN achieves 63.3% one-shot completion versus 14.0% for a bigram baseline and 0% sequential forgetting, while cross-modal alignment reaches 80.0% retrieval accuracy and 0.950 mean reciprocal rank. The energy report projects 179.65 uJ per inference on Loihi 2, corresponding to a 278x improvement over the matched transformer on an A100 and roughly 8050x lower projected online-training energy.

These results do not show that Bio-ARN surpasses dense deep learning on raw accuracy or fluency. Instead, they suggest a different and potentially valuable trade-off: honest abstention, continual local learning, sparse activity, and a plausible deployment path on neuromorphic hardware. Bio-ARN therefore serves as a concrete research program for studying non-backprop, brain-inspired intelligence at system scale.

## 1. Introduction

Modern AI has been propelled by a simple recipe: scale differentiable function approximators, optimize them end to end with backpropagation, and feed them vast amounts of data and compute. That recipe has produced remarkable capabilities, but it also imposes a narrow view of intelligence. In most deployed systems, learning is separated from inference, uncertainty is appended after the fact, and adaptation is expensive. Even when the resulting models appear flexible, their internal mechanics remain dominated by dense matrix multiplication, global error transport, and objective functions that reward a prediction on every input whether or not the model should genuinely commit to one.

This creates four persistent problems. First, catastrophic forgetting remains a serious obstacle for online or continual learning. Fine-tuning a dense model on new data often perturbs old solutions because the same parameters are repurposed for many tasks [Kirkpatrick et al., 2017; Rusu et al., 2016; Mallya and Lazebnik, 2018]. Second, most mainstream classifiers and generators are not structurally honest about uncertainty. A softmax layer always chooses something; confidence calibration is a downstream statistical correction rather than an intrinsic architectural property. Third, the energy cost of dense AI is substantial. Attention-heavy models perform large amounts of computation even when the input is predictable or irrelevant. Fourth, many present systems are disembodied in the strongest possible sense: they manipulate symbol streams without an internal account of sensorimotor grounding, predictive interaction, or self-monitoring.

Biology suggests a different organizing logic. The brain appears to operate through sparse activity, local plasticity, predictive processing, recurrent competition, and globally limited broadcasting of salient content [Hebb, 1949; Rao and Ballard, 1999; Friston, 2010; Dehaene and Changeux, 2011]. Individual neurons and small circuits do not backpropagate gradients through a monolithic graph. They adapt through local co-activity and neuromodulation. Concepts appear to be represented by selective neural populations, including highly sparse "concept cells" in medial temporal structures [Quiroga et al., 2005]. Cortical processing is strongly predictive: descending expectations suppress predictable inputs while ascending signals emphasize residual errors. Conscious access, insofar as it can be modeled computationally, seems bottlenecked rather than fully global, closer to a workspace than to an unrestricted activation flood [Baars, 1988; Dehaene and Changeux, 2011].

Bio-ARN 2.0 takes those ideas seriously at the level of a complete computational stack. Its repeating representational unit is the concept cell cluster, or CCC, inspired by cortical columns and adaptive resonance principles. Each CCC transforms sparse sensory features into a concept-space activation and then passes that activation through a margin gate. If the match between input and learned direction exceeds a threshold, the CCC fires; otherwise it abstains. This seemingly small design choice matters because it converts uncertainty from a post hoc score into a first-class computational event. Around the CCCs sits an associative fabric implemented with Kanerva-style sparse distributed memory [Kanerva, 1988], a predictive hierarchy that propagates errors instead of full activations [Rao and Ballard, 1999; Salvatori et al., 2023], and a compact global neuronal workspace that can broadcast a few active concepts at a time [Baars, 1988; Dehaene and Changeux, 2011].

The project is intentionally ambitious. It spans image classification, abstention, few-shot recall, continual learning, text generation, multimodal alignment, scaling, and energy modeling. The resulting picture is mixed but scientifically useful. Bio-ARN does not beat the tiny MLP baseline on MNIST accuracy, and it does not beat a trigram model on raw next-character prediction in the small text benchmark. However, it does demonstrate three properties that are rarely achieved together in one architecture: explicit abstention, local continuous learning, and sparse-neuromorphic efficiency. In the benchmark suite, Bio-ARN matches the tiny transformer on MNIST accuracy while strongly outperforming it on OOD abstention, forgetting, and projected energy. In the text suite, it underperforms n-gram baselines on some next-character metrics yet strongly outperforms them in one-shot adaptation and completely avoids sequential forgetting.

This paper makes six main contributions:

1. **Architecture:** It presents a full brain-inspired system design that integrates spiking computation, concept-cell matching, predictive coding, sparse associative memory, a global workspace, and reward-modulated learning.
2. **Honest uncertainty:** It operationalizes abstention through an explicit margin gate inside each concept cell cluster rather than by a downstream calibration heuristic.
3. **Local learning at system scale:** It demonstrates a non-backprop PyTorch implementation in which learning is carried by Hebbian, resonance-gated, and memory-local updates.
4. **Empirical trade-off map:** It reports comparative results versus MLP, transformer, n-gram, and recurrent baselines across classification, few-shot learning, OOD detection, continual learning, text generation, multimodal retrieval, and CIFAR-10 preprocessing.
5. **Scaling analysis:** It shows that vectorized CCC pools and sharded retrieval can extend the architecture to 10K concept cells with roughly linear memory growth and significant latency reduction when locality is exploited.
6. **Energy analysis:** It translates measured sparse activity into a hardware-oriented argument, projecting a 278x inference-energy advantage for Loihi 2 over an A100 transformer baseline at the matched MNIST accuracy tier.

The rest of the paper is organized as follows. Section 2 situates Bio-ARN relative to prior work in spiking models, continual learning, neuromorphic computing, active inference, and brain-inspired systems. Section 3 details the architecture and its governing equations. Section 4 describes the PyTorch implementation and scaling abstractions. Section 5 presents experiments. Section 6 analyzes energy and deployment implications. Sections 7 and 8 discuss limitations and future work, and Section 9 concludes.

## 2. Related Work

### Spiking neural networks and spiking language models

Spiking neural networks (SNNs) promise event-driven efficiency and closer alignment with biological dynamics [Maass, 1997; Davies et al., 2018]. Recent work has extended SNNs into high-level machine learning, including image models such as SEW-ResNet [Fang et al., 2021] and language-oriented systems such as SpikeGPT [Zhu et al., 2023] and spike-driven large language models [Xu et al., 2026]. These models demonstrate that spikes can support modern tasks, but many still inherit the training logic of conventional deep learning: dense objectives, offline datasets, and backpropagation through time or surrogate gradients. Bio-ARN differs in two ways. First, its spikes are embedded inside a broader cognitive architecture that includes workspace dynamics, associative memory, and sensorimotor loops rather than a task-specific spiking backbone. Second, the project explicitly avoids backpropagation in its main learning path, favoring fast recruitment, Hebbian tuning, and predictive-error settling.

### Continual learning

Continual-learning research has produced important mitigation strategies for forgetting, including elastic weight consolidation (EWC) [Kirkpatrick et al., 2017], progressive networks [Rusu et al., 2016], and parameter-pruning approaches such as PackNet [Mallya and Lazebnik, 2018]. These methods usually treat forgetting as a constraint imposed on a pre-existing dense model. They preserve knowledge by regularizing weight movement, allocating new subnetworks, or freezing sparsified parameters. Bio-ARN addresses the same problem through a different inductive bias: concepts are represented by recruited units and sparse associations, so new learning can often be added rather than overwritten. This does not make forgetting impossible in principle, but it changes the failure mode from "all shared weights drift" to "new concepts must find space without corrupting older attractors." The empirical result is modest but meaningful: low forgetting on MNIST and zero forgetting in the small text sequential benchmark.

### Neuromorphic computing

Neuromorphic hardware such as TrueNorth, Loihi, and BrainScaleS has long argued that efficient intelligence requires event-driven computation and memory locality [Merolla et al., 2014; Davies et al., 2018; Schuman et al., 2022]. What is often missing, however, is a systems-level algorithmic target that natively benefits from such hardware. Bio-ARN is designed with that target in mind. Its sparse firing, binary-like events, local plasticity, bounded workspace, and content-addressable retrieval all map more naturally to neuromorphic substrates than dense attention matrices. Unlike many software-only neuromorphic demonstrations, the project includes a hardware abstraction and deployment pipeline aimed at Loihi-style packaging. The energy claims in this paper remain projected rather than measured on real silicon, but they are grounded in component-wise sparse activity profiles rather than in vague aspiration.

### Brain-inspired cognitive architectures

Several research programs have moved beyond narrow neural networks toward broader cognitive architectures. Hawkins' Thousand Brains theory emphasizes cortical columns, sensorimotor object models, and distributed voting [Hawkins et al., 2024; Leadholm et al., 2026]. Active-inference toolkits such as pymdp operationalize perception and action as free-energy minimization [Heins et al., 2022]. Other embodied frameworks, such as Neural Brain and related world-model proposals, highlight multimodal grounding and loop closure [Liu et al., 2025]. Bio-ARN inherits from each of these lines but differs in its specific synthesis. It treats concept-cell clusters as the repeating unit, uses explicit margin-gated abstention rather than pure posterior precision, binds concepts through Kanerva-style memory instead of only through symbolic state updates, and makes generation emerge from prediction rather than from autoregressive next-token sampling.

### Predictive coding and active inference

Predictive coding has been proposed both as a theory of cortical processing and as an algorithmic alternative to backpropagation [Rao and Ballard, 1999; Friston, 2010; Salvatori et al., 2023]. Recent variants such as Predictive Coding Light emphasize suppressing predictable sensory spikes so that only residual errors are propagated. Bio-ARN is closely aligned with this perspective. Its predictive hierarchy tries to minimize error level by level, and its reward system treats successful error reduction as intrinsic value. Yet Bio-ARN is not a pure predictive-coding stack. It augments predictive processing with explicit concept recruitment, sparse address-based retrieval, and a workspace-style bottleneck. In other words, predictive coding provides the architecture's metabolic logic, while CCCs, SDM, and GNW provide its representational and control logic.

### Adaptive resonance and concept cells

Bio-ARN also resonates with adaptive resonance theory (ART), which couples vigilance thresholds with category recruitment [Carpenter and Grossberg, 1987; Grossberg, 2013]. The margin gate plays a role analogous to vigilance: insufficient match leads not to forced assignment but to abstention and potential recruitment. Likewise, neuroscience evidence for sparse, high-selectivity concept cells motivates the use of committed concept prototypes [Quiroga et al., 2005]. Bio-ARN differs from classical ART by embedding category matching inside a spiking, predictive, multimodal, and associative architecture rather than a standalone classifier.

Taken together, the literature suggests that the ingredients of Bio-ARN are not individually implausible. The novelty lies in combining them into a single stack and evaluating the resulting trade-offs. The project asks whether a system can be honest, sparse, local-learning, and at least moderately capable all at once. The answer emerging from the current experiments is not yet definitive, but it is more encouraging than any single component analysis would suggest.

## 3. Architecture

### 3.1 Overview

Bio-ARN 2.0 is organized as a layered but recurrent architecture whose major components are: (i) embodied sensorimotor streams, (ii) concept cell clusters, (iii) sparse distributed associative memory, (iv) a predictive hierarchy, (v) a global neuronal workspace, and (vi) reward/novelty modulation. Figure 1 should depict the system as a loop rather than a feed-forward tower. Raw inputs enter through spiking sensory encoders; prediction errors, not full sensory states, ascend into the predictive hierarchy; CCCs compete to explain the resulting sparse feature patterns; a small subset of successful concepts enters the workspace; and the workspace broadcasts selected concepts back through the associative fabric and predictive hierarchy to influence future perception and generation.

This architecture is meant to capture three biological intuitions. The first is **locality**: most computation and most learning should occur in small modules, not through global gradient transport. The second is **sparsity**: only a small number of concepts, memory locations, and predictions should be active at a time. The third is **closed-loop cognition**: perception, memory, and action are not isolated stages but parts of a continuously coupled dynamical process.

Two numerical observations from the repository motivate the design. In the benchmark suite, Bio-ARN uses only about 295k active MACs per inference despite a dense-equivalent footprint of 2.71M MACs. In the energy report, only 3.6 CCCs are active on average out of 7.0 committed concepts during the profiled inference loop, and predictive suppression removes 47.3% of hierarchy activity before higher-level processing. These are not proofs of biological realism, but they are exactly the sort of sparse behavior the architecture was built to induce.

### 3.2 Spiking neurons

The primitive computational element is the leaky integrate-and-fire (LIF) neuron, implemented in discrete time. For membrane voltage $v_t$, input current $I_t$, threshold $\\theta$, reset value $v_{reset}$, and decay $\\beta$, the update is

$$
v_{t+1} = \\beta v_t + I_t - s_t(v_t - v_{reset}), \qquad s_t = \mathbf{1}[v_t \ge \\theta].
$$

This formulation captures integration, thresholded emission, and reset without requiring dense continuous activations. In practice, Bio-ARN uses spikes in a hybrid way. Some modules are event-like and binary, while others maintain dense tensors that are sparsified through top-$k$ selection, thresholding, or margin gating. This hybrid design is a pragmatic compromise: it preserves the computational semantics of sparse spikes while staying implementable and inspectable in PyTorch.

Bio-ARN supports both rate-like and temporal interpretations. In the vision and sensory streams, event counts and local timing encode salience. In associative and concept layers, sparse activation vectors function more like instantaneous population codes. The aim is not strict neuron-by-neuron simulation, but a system-level reuse of spike-native principles: thresholding, silence as a default state, and event-driven update paths.

### 3.3 Concept Cell Clusters

The CCC is the architecture's central representational unit. Each cluster contains an F1 feature stage, an F2 concept stage, a margin gate, and a feedback predictor. Given sensory input $x \in \mathbb{R}^d$, the F1 stage computes a sparse feature embedding

$$
f^{(1)} = \\text{TopK}(\\text{ReLU}(W_1 x + b_1), k),
$$

where $k$ is a small feature budget. The F2 stage then maps these features into concept space:

$$
h_i = W_{2,i} f^{(1)}.
$$

Each CCC stores a learned concept direction $d_i$ in the same space. Intuitively, $d_i$ is a prototype or attractor for a reusable concept. Some CCCs may represent digits, others character chunks, visual structures, or multimodal correspondences depending on the task.

Unlike a standard classifier head, the CCC does not directly output a normalized label distribution. It asks a more basic question: *does this input sufficiently match my learned concept?* That question is answered by the margin gate.

### 3.4 Margin gate and honest abstention

For CCC $i$, the confidence score is the cosine similarity between its current activation and stored concept direction:

$$
c_i = \cos(h_i, d_i).
$$

The margin gate then applies a threshold $\\theta_i^{margin}$:

$$
y_i =
\\begin{cases}
h_i & \\text{if } c_i > \\theta_i^{margin} \\\\
\\text{ABSTAIN} & \\text{otherwise.}
\end{cases}
$$

This operation is the architectural source of honest uncertainty. Dense classifiers ordinarily emit a prediction for every input and later reinterpret low logit margins as uncertainty. Bio-ARN instead allows a concept to decline commitment. When no CCC fires strongly enough, the system can recruit a new concept, query associative support, or leave the input unresolved. In OOD evaluation, this mechanism yields the clearest empirical win: 76.7% OOD abstention and 0.933 AUROC in the archived benchmark suite, versus 21.3% and 0.707 for the MLP and 16.6% and 0.787 for the transformer.

The margin threshold can itself be locally adapted. If a CCC frequently fires on poor matches, its threshold can be nudged upward; if reward signals repeatedly validate a CCC's activations, the threshold can relax slightly. This makes abstention not just a static rule but a trainable part of the recognition policy.

### 3.5 Feedback prediction and resonance

When a CCC fires, it projects back into feature space through a feedback matrix $W_i^{fb}$:

$$
\hat f^{(1)}_i = W_i^{fb} h_i.
$$

The predicted feature pattern is compared to the actual feature pattern using a resonance score

$$
r_i = \cos(\hat f^{(1)}_i, f^{(1)}).
$$

and learning is permitted only if $r_i$ exceeds a resonance threshold $\\theta^{res}$. This yields a simple but useful distinction. Firing says, "I think I recognize this." Resonance says, "My top-down expectation genuinely matches the bottom-up evidence." Only in the latter case does the system update the concept through slow Hebbian tuning:

$$
d_i \leftarrow \\text{normalize}(d_i + \\alpha h_i),
$$
$$
W_i^{fb} \leftarrow W_i^{fb} + \\beta (f^{(1)} - \hat f^{(1)}_i) h_i^\\top.
$$

If no committed concept both fires and resonates, the pool can recruit a new CCC through fast learning. This combines one-shot plasticity with conservative refinement: novel inputs create new prototypes, while familiar inputs only adjust existing ones when prediction and observation agree closely enough.

### 3.6 Sparse distributed memory and associative fabric

CCCs are linked by an associative fabric based on sparse distributed memory (SDM) [Kanerva, 1988]. Each concept obtains a high-dimensional sparse address, and co-active concepts strengthen shared storage locations. Given a cue address $a(c)$ and memory locations $m$ within Hamming radius $r$, retrieval is

$$
\\text{Retrieve}(c) = \sum_{m: H(a(c), m) < r} D[m],
$$

where $D[m]$ stores accumulated associative traces. Two features matter here. First, recall is **content-addressable** rather than strictly positional. Second, memory updates are local and incremental. When CCC $i$ and $j$ co-activate, the relevant memory traces strengthen without requiring global retraining.

The associative fabric also supports lateral inhibition and temporal binding. Similar concepts compete so that one strong explanation can suppress redundant neighbors. Sequential co-activation strengthens directional links, allowing the system to model transitions and temporal expectations without a full attention matrix. This is especially important for text tasks: Bio-ARN does not memorize long contexts via self-attention, but it can bind local concept sequences and retrieve likely continuations from memory.

### 3.7 Predictive coding hierarchy

The predictive engine is a multi-level hierarchy in which each level predicts the state below it and receives only residual errors upward. For hierarchy level $\ell$,

$$
arepsilon_\ell = x_\ell - f(W_\ell x_{\ell+1}),
$$
$$
x_{\ell+1} \leftarrow x_{\ell+1} + \gamma W_\ell^\\top arepsilon_\ell,
$$
$$
W_\ell \leftarrow W_\ell + \eta \, arepsilon_\ell x_{\ell+1}^\\top.
$$

This is not backpropagation through an arbitrary computational graph. Errors are local to adjacent levels, states settle iteratively, and weights update through layer-local correlations. The practical consequence is twofold. Predictable information is suppressed, reducing unnecessary computation. Novel mismatches become salient and can trigger concept recruitment or exploratory action.

The energy report quantifies this intuition. In the profiled inference loop, the predictive hierarchy exhibits 47.3% suppression, contributing materially to the sparse compute story. This is a more specific claim than "spikes are efficient": predictable spikes are *explicitly removed* before downstream processing.

### 3.8 Global neuronal workspace

The global neuronal workspace (GNW) is a bounded buffer that selects a small number of active CCCs and broadcasts them across the system. Bio-ARN implements this through winner-take-most competition, limited capacity, and temporal fatigue. If $A_t$ denotes candidate concept activations at time $t$, then workspace contents are roughly

$$
\\text{GNW}_t = \\text{TopK}(A_t - F_t, K),
$$

where $F_t$ is a fatigue term and $K$ is the workspace capacity (configured to 7 in the current system, echoing the classical 7+-2 working-memory heuristic [Miller, 1956]). Broadcast amplifies winning activations and sends them back into the associative fabric, enabling short chains of concept-to-concept thought, planning, or generation.

This module is important because it prevents the architecture from degenerating into a fully parallel soup of weak associations. Many concepts may partially match, but only a few become globally consequential at any one time. That bottleneck is central both to efficiency and to interpretability: the broadcast set is the system's current working context.

### 3.9 Sensorimotor loop

Bio-ARN is not conceived as a purely textual model. Its embodied sensorimotor cortex includes vision, audio, touch, proprioceptive, and motor streams in the architectural specification, even though the present repository evaluates mostly vision and text. The loop can be summarized as:

1. Sensory streams encode raw input into sparse spike-like features.
2. Predictable components are suppressed by the predictive hierarchy.
3. CCCs compete to explain the residual input.
4. The workspace selects salient concepts and broadcasts them.
5. Broadcast concepts cue memory, prediction, and motor plans.
6. Motor output changes either the external world or the system's own generated sequence.
7. New sensory consequences re-enter the loop.

For language generation, this means the system does not simply sample the next token from a softmax. Instead, it forms a conceptual state, predicts what a consistent output sequence should look like, emits text through the motor stream, and then monitors that output as a new input. The current implementation remains much weaker than transformer generation, but the architectural point is clear: generation is treated as controlled prediction rather than unconditional token emission.

### 3.10 Reward and novelty modulation

A reward/novelty subsystem modulates learning rate and concept recruitment. Intrinsic reward tracks successful reduction of prediction error; novelty spikes when error is large and unresolved. A simple abstract update is

$$
\\alpha_t = \\alpha_0 (1 + \lambda_n n_t + \lambda_r r_t),
$$

where $n_t$ is novelty and $r_t$ is reward. High novelty can lower hesitation for recruitment or temporarily boost plasticity. This is biologically inspired rather than biophysically exact, but it captures the important idea that learning should depend not only on correlation but also on salience.

In summary, Bio-ARN's architecture is novel less იმიტომ कि any one component is unprecedented, and more because the components are arranged to make uncertainty, sparsity, and continual learning cohere. The CCC decides whether to know, the predictive hierarchy decides what is worth sending upward, the SDM decides what related experience to retrieve, the GNW decides what becomes globally relevant, and the reward system decides what should change fastest.

## 4. Implementation

Bio-ARN is implemented in PyTorch, but the implementation deliberately resists the usual deep-learning pattern of defining a differentiable module and optimizing it with an external optimizer. The repository's core modules use `torch.no_grad()` in the main learning path, store critical concept state in buffers rather than trainable parameters, and update those buffers with explicit local rules. In `bioarn.core.ccc`, the F1 linear projection has gradients disabled, concept directions are stored as buffers, and fast/slow learning is carried out through direct vector updates. In `bioarn.system`, perception and thought proceed through explicit orchestration of the CCC pool, associative fabric, and workspace rather than through a single end-to-end differentiable forward pass.

This matters for two reasons. First, it ensures that the repository's headline claim—no backpropagation in the Bio-ARN path—is reflected in code rather than in prose alone. Second, it makes the learning rules auditable. A reviewer can inspect `learn_fast`, `learn_slow`, the margin-gate resonance checks, and the SDM updates without hunting for hidden gradient flow. The dense baselines in the benchmark suite still use conventional learning where appropriate, which makes the comparison intentionally asymmetric: Bio-ARN is not merely a different architecture, but a different training philosophy.

Scaling introduces another implementation challenge. A naive CCC pool implemented as a Python list of modules becomes increasingly expensive as the number of concepts grows. To address this, the repository provides a `BatchedCCCPool` that stores all CCC weights, feedback matrices, thresholds, and statistics as batched tensors. This reduces Python overhead and allows many concept evaluations to be vectorized. The scaling report shows that memory grows approximately linearly, from 1.20 MB at 100 CCCs to 120.17 MB at 10K CCCs, corresponding to roughly 0.012 MB per CCC in the profiled setup.

The repository also includes `PoolSharding`, which partitions large pools into smaller shards. Sharding does not change the conceptual algorithm; it changes the search space considered per query when locality is exploitable. In the 10K-CCC benchmark, a flat vectorized pool processes the main scaling workload in 35.85 ms/sample, while a targeted fast-infer microbenchmark measures 134.38 ms/sample flat versus 29.29 ms/sample sharded. These numbers should not be conflated because they come from different access patterns, but together they show that vectorization and locality-aware partitioning are practical tools for growing the concept pool.

A hardware abstraction layer rounds out the implementation. The repository contains energy modeling, quantization, Lava bridging, and a Loihi deployment pipeline. `LoihiDeploymentPipeline` loads a trained model, quantizes it, validates equivalence, converts it into a process graph, and estimates hardware requirements such as core count, memory per core, power, and latency. This does not prove successful real-hardware deployment yet, but it materially strengthens the paper's claim that Bio-ARN is being designed as hardware-software co-design rather than as a CPU-only curiosity.

Overall, the implementation is best viewed as a research scaffold: detailed enough to support nontrivial experiments, explicit enough to audit the learning rules, and modular enough to support scaling and deployment studies.

## 5. Experiments

This section summarizes the repository's main empirical results. Detailed numeric tables are provided in [Table 1](tables/table1_mnist_comparison.md) through [Table 5](tables/table5_cifar_preprocessing.md).

### 5.1 MNIST classification and abstention

The primary benchmark suite in `experiments/benchmarks/results.json` evaluates three models—Bio-ARN, a small MLP, and a small transformer—across three random seeds with 5000 training samples, 1000 test samples, and a 500-sample calibration split. Scenario A measures classification accuracy, Scenario D measures OOD behavior, and Scenario E measures sparsity and latency. Bio-ARN achieves 82.0% mean MNIST accuracy, matching the transformer and trailing the 88.9% MLP. That raw ranking is important: Bio-ARN is not the best pure classifier here.

The more interesting result is what happens when the model is uncertain. Bio-ARN reaches 0.933 AUROC and abstains on 76.7% of OOD inputs, compared with 0.707 and 21.3% for the MLP and 0.787 and 16.6% for the transformer. In practical terms, Bio-ARN is much more willing to say "I do not know" when confronted with random noise, inverted digits, rotated digits, or Fashion-MNIST distractors. Per-distribution AUROC is especially strong for random noise (0.998) and inverted digits (0.999), while Fashion-MNIST remains the hardest negative set (0.792). This is exactly the regime in which architectural abstention matters: near-distribution nonsense should not be treated like a valid but low-confidence class.

The repository context also includes a project-level Phase-0 summary reporting 81.2% MNIST accuracy, 81.7% OOD abstention, 3.4% forgetting, and 9.82% sparsity. Those values are consistent in spirit with the archived multi-seed benchmark, but the formal tables in this paper use the reproducible JSON means: 82.0%, 76.7%, 3.2%, and 10.9% active-MAC density. We call attention to this because honest reporting includes documenting minor discrepancies between evolving experimental summaries.

### 5.2 Few-shot learning

Few-shot behavior is where Bio-ARN starts to separate itself from the dense baselines. In Scenario B of the benchmark suite, Bio-ARN achieves 41.8% accuracy with one example per class, versus 36.8% for the MLP and 23.6% for the transformer. At five examples, the gap widens to 63.8% for Bio-ARN versus 49.5% and 32.5%. At ten examples, Bio-ARN remains ahead at 65.9%.

The text-generation benchmark makes the same point even more strongly. In `text_gen_results.json`, Bio-ARN completes few-shot sequence tasks at 63.3% with a single example and 66.2% with five examples. The bigram baseline scores only 14.0% and 36.0%; the trigram baseline scores 26.9% and 58.3%. Put differently, the architecture's concept recruitment and associative binding seem well suited to "learn this new pattern now" tasks, even when they are not ideal for maximizing average next-character likelihood over a static corpus.

This is an important trade-off. Transformer-style learning excels when there is ample data and stable optimization. Bio-ARN's design hypothesis is different: if concepts can be recruited quickly and memory can bind sparse co-occurrence traces locally, then one- and few-shot acquisition should be a natural strength. The current experiments provide supportive, though still small-scale, evidence for that hypothesis.

### 5.3 Continual learning

Scenario C in the benchmark suite evaluates class-incremental MNIST learning. Bio-ARN starts at 77.0% accuracy on the original classes and retains 73.8% after training on new classes, corresponding to just 3.2% forgetting. The MLP drops from 62.8% to 4.1% (58.7% forgetting), and the transformer drops from 55.9% to 0.0% (55.9% forgetting). This is the clearest direct evidence that locally committed concept units can protect prior knowledge better than small dense baselines in an online setting.

The text benchmark provides an even stronger sanity check. Bio-ARN shows 0% forgetting on the simple sequential corpus probe, while a tiny recurrent baseline forgets 66.7%. The task itself is small, so the result should not be overgeneralized. Still, it aligns well with the architecture's intended inductive bias: adding a new concept or temporal association should not require globally repurposing the entire representational substrate.

### 5.4 Out-of-distribution detection

Because the margin gate is central to the architecture, OOD detection deserves separate emphasis. Bio-ARN's AUROC of 0.933 is accompanied by a strong F1 score of 0.936 and the previously noted 76.7% OOD abstention. The performance is not uniform across all shifts: Fashion-MNIST remains more confusable than random noise or inversion. Yet even there, Bio-ARN maintains better rejection behavior than the dense baselines.

This pattern supports a broader claim: honest abstention is not merely a calibration statistic but a behavioral regime. If the match between an input and a known concept falls below threshold, Bio-ARN declines to commit. In safety-critical or human-in-the-loop settings, that may matter more than squeezing out the last few percentage points of in-distribution accuracy.

### 5.5 Text generation

The text results are intentionally modest and are presented with caution. On raw next-character prediction, Bio-ARN is not the strongest model. In the small benchmark, its context-8 accuracy is 0.188 and context-32 accuracy is 0.177, both below the trigram baseline (0.604 and 0.615) and below the bigram baseline as well. This is unsurprising: n-gram models are excellent at short-context local statistics on small corpora.

However, the same benchmark reports an approximate perplexity proxy of 1.064 for Bio-ARN, compared with 2.135 for the bigram baseline and 1.461 for the trigram baseline, indicating that the architecture's confidence and surprisal proxy are doing something nontrivial even when token-choice accuracy is weaker. Bio-ARN also scores 0.800 on the pattern-learning composite, above trigram's 0.778, though still below bigram's 0.870. Its main failure mode is repetition: the repetition rate is 0.444, far worse than the n-gram baselines.

The project includes a separate text-improvement demo that starts from a baseline perplexity of 3.5 and is summarized in the project brief as improving to 2.75 after larger-corpus training and better decoding. We treat that number as an internal project result rather than a fully archived benchmark, and therefore the formal table in this paper focuses on the reproducible `text_gen_results.json` metrics. Even under that stricter standard, the central story remains: Bio-ARN is not yet a fluent text generator, but it is unusually strong at low-shot adaptation and does not catastrophically forget sequential text patterns.

### 5.6 Cross-modal binding

The multimodal demo shows that the architecture's associative mechanisms are not limited to single-modality classification. After supervised alignment of ten visual patterns and their textual labels, Bio-ARN achieves 80.0% supervised cross-modal retrieval accuracy and 0.950 mean reciprocal rank. The text-to-image reconstruction check for `vertical_mid` reaches cosine similarity 0.992, suggesting that aligned concepts are meaningfully shared across modalities rather than merely co-indexed.

The per-category breakdown is instructive. Most categories retrieve correctly, while the `cross` and `x_shape` patterns confuse with structurally related shapes, exactly the kind of error expected from similarity-based associative memory. This is encouraging because it means the system's failures are semantically legible. Instead of arbitrary class confusion, it tends to substitute neighboring structural concepts.

### 5.7 Scaling analysis

The scaling experiments evaluate pool sizes from 100 to 10K CCCs. Memory footprint grows almost linearly from 1.20 MB at 100 CCCs to 120.17 MB at 10K CCCs. In the main vectorized workload, inference rises from 0.87 ms/sample at 100 CCCs to 35.85 ms/sample at 10K. Learning time remains in a similar range, reaching 16.06 ms/sample at 10K on the profiled run. This is encouraging because the architecture's sparse logic does not immediately collapse under larger concept inventories.

Sharding provides an additional lever. In a focused 10K query benchmark, flat fast inference takes 134.38 ms/sample while the sharded version takes 29.29 ms/sample, a large reduction when locality is strong. The project brief cites a more optimistic 9.8 ms sharded inference figure; the current archived CPU reproduction does not reach that number, so we report both and interpret the discrepancy conservatively as environment sensitivity plus benchmark-shape differences.

### 5.8 CIFAR-10 with preprocessing

Bio-ARN struggles on raw CIFAR-10 streams, scoring only 9.7% in the best raw configuration. This is effectively CCC collapse: the concept matcher sees a high-dimensional, noisy input space without enough structure to recruit useful stable prototypes. Preprocessing changes the story substantially. Random projection improves the best accuracy to 12.7%, patch hashing to 14.3%, and contrast-normalized PCA to 11.3%, but PCA to 128 dimensions is the clear winner at 26.7% accuracy and 26.9% covered accuracy with only 1.0% abstention.

This result is scientifically important even though 26.7% is far from competitive CIFAR performance. It shows that Bio-ARN benefits from a preprocessing stage that compresses raw sensory variability into more stable, concept-friendly structure. Put bluntly, the current CCC design is not yet a strong raw-pixel learner. That limitation does not invalidate the architecture, but it sharply defines the next engineering problem.

## 6. Energy Analysis

The energy analysis in `experiments/energy_report_data.json` and `experiments/energy_report_results.md` is one of the project's strongest contributions. On the measured CPU prototype, Bio-ARN is slower than the tiny dense baselines: 184.42 ms wall-clock per inference versus roughly 0.012 ms for the MLP and 0.071 ms for the transformer benchmark. This is not hidden in the report; it is stated explicitly. The current software stack does not yet realize the full advantage of sparse computation because SDM address arithmetic and Python orchestration dominate runtime.

The projected hardware story is different. Component-wise activity accounting yields 179.65 uJ per inference on Loihi 2, 50.55 mJ on an A100 GPU, 7.98 mJ on a laptop CPU, and 74.36 uJ on an idealized ASIC estimate. Relative to the matched transformer reference at 50.01 mJ, the Loihi 2 projection corresponds to a 278x energy reduction. Training is even more asymmetric: online local learning over 5000 samples is estimated at 0.932 J for Bio-ARN versus about 7501.76 J for the transformer, a ratio of roughly 8050x.

The component breakdown explains where the savings originate. The CCC pool falls from 1.64M dense FLOPs to 148.7k sparse FLOPs, the predictive hierarchy suppresses 47.3% of activity, and the overall modeled-unit silence rate reaches 82.5%. Not every component is equally efficient; SDM retrieval remains a significant compute hotspot in the current implementation. This again supports a hardware-software co-design interpretation: the architecture is attractive precisely because its costly operations are sparse, structured, and local enough to benefit from custom substrates.

For edge deployment, the power numbers are plausible rather than merely symbolic. At 100 Hz, the Loihi 2 projection is about 17.97 mW, orders of magnitude below a GPU and comfortably in embedded territory. Whether real hardware realizes these projections remains an open experimental question, but the report provides a quantitatively grounded reason to pursue that validation.

## 7. Discussion

### What Bio-ARN does well

Bio-ARN's strongest properties are not hidden in aggregate accuracy; they are visible in its behavior under uncertainty, novelty, and sequential adaptation. The margin gate is the clearest example. By allowing a concept to abstain when the match is weak, the architecture avoids one of the most persistent pathologies of dense classifiers: confidently misclassifying unfamiliar inputs. The strong OOD results are not an incidental calibration artifact; they are a direct consequence of the computation each CCC performs.

The second strength is continual learning. Because concepts can be recruited and tuned locally, new learning does not automatically overwrite old solutions. The architecture therefore turns catastrophic forgetting from a default outcome into a contingent one. The MNIST and text results are small-scale, but they suggest that local concept commitment is a promising route for lifelong learning.

The third strength is efficiency in the *architectural* sense. Even where the current CPU implementation is slow, the computation is sparse, structured, and interpretable. Only a few concept cells fire, only nearby memory locations matter, only residual prediction errors propagate, and only a handful of concepts enter the workspace. This is exactly the kind of profile that conventional hardware underutilizes and neuromorphic hardware may exploit.

### Limitations

The limitations are equally important. First, Bio-ARN is not currently competitive with standard deep learning on raw supervised accuracy. The MLP still wins the main MNIST benchmark, and the CIFAR results are low in absolute terms. Second, the text generator is not yet fluent. Although the few-shot and forgetting behavior is impressive, the generated sequences show repetition and weak long-range coherence. Third, the strongest energy claim is projected rather than measured on actual Loihi hardware. Fourth, some results are environment sensitive; the 10K scaling latency in the current run differs from the project brief. Fifth, while the architecture is biologically inspired, it is not a faithful brain model. Many elements remain algorithmic abstractions rather than neuroscientific reconstructions.

### Safety and abstention

One reason these limitations may be acceptable is that Bio-ARN optimizes a different objective profile. In many real systems, being honestly uncertain can be more valuable than being slightly more accurate on average. Medical triage, embedded robotics, anomaly detection, and human-assistive tooling all benefit from agents that can withhold commitment. If an architecture can deliver moderate competence with high abstention integrity and low forgetting, that may justify its use in settings where dense end-to-end models remain difficult to trust.

### Biological plausibility

Bio-ARN should not be described as "the brain in code." Still, its design aligns with several broad biological themes: local learning [Hebb, 1949], concept selectivity [Quiroga et al., 2005], predictive suppression [Rao and Ballard, 1999; Friston, 2010], globally limited broadcasting [Baars, 1988; Dehaene and Changeux, 2011], and neuromodulated novelty. The architectural plausibility matters because it suggests that the project is not just engineering eccentricity. It is a serious attempt to operationalize long-standing computational neuroscience ideas in a modern software system.

### Deployment readiness

Finally, Bio-ARN is further along on neuromorphic readiness than many papers making similar claims. The repository includes quantization, equivalence validation, process-graph conversion, and hardware requirement estimation. What it lacks is real-chip validation. That is a major gap, but it is a concrete gap, not a conceptual void.

## 8. Future Work

The immediate next step is real neuromorphic validation. The Loihi 2 projection is compelling, but projected energy is not measured energy. Deploying a stable subset of the architecture to real hardware would test whether sparse CCC competition, predictive suppression, and associative retrieval survive conversion losses and timing constraints.

A second priority is larger-scale learning. The present scaling experiments show that 10K CCCs are tractable, but the architecture's claims about open-ended concept growth will only be convincing when tested at 100K or more concepts, with correspondingly richer associative memory and longer-lived online learning.

Third, the modality set should expand. The architectural specification includes audio, touch, and proprioception, but the current repository emphasizes vision and text. Speech and audio prediction would be a natural next domain because they stress temporal binding, prediction, and motor generation simultaneously.

Fourth, Bio-ARN should be embedded inside autonomous agents. Its workspace, memory, and abstention mechanisms are particularly relevant for systems that must decide when to act, ask, defer, or explore. Integrating the architecture into a closed-loop agent would better test active inference claims than static benchmark suites can.

Fifth, the learning rules themselves can become richer. Reward-modulated STDP, better novelty scheduling, and more expressive temporal credit assignment may improve both text quality and multimodal binding without abandoning the local-learning principle.

## 9. Conclusion

Bio-ARN 2.0 is not a drop-in replacement for transformers, nor does it currently outperform dense deep learning on mainstream accuracy metrics. That is not the right reading of the results. The architecture's contribution is to demonstrate that a single system can combine explicit abstention, continual local learning, sparse activity, cross-modal association, and a credible neuromorphic deployment pathway.

Empirically, Bio-ARN matches a tiny transformer on MNIST accuracy while strongly outperforming it on OOD abstention, forgetting, and projected energy; it dramatically outperforms simple text baselines on one-shot completion and sequential retention; and it scales to larger concept pools with manageable memory growth and useful sharding gains. Just as importantly, it reveals where the approach still falls short: raw-pixel learning, fluent generation, and real-hardware validation.

The central lesson is therefore not that brain-inspired AI has already won, but that it is now concrete enough to test. Bio-ARN turns several often-separate ideas—spikes, predictive coding, sparse memory, workspace dynamics, and honest uncertainty—into a working research platform. That alone makes it a valuable step toward alternatives to purely backprop-driven intelligence.

## References

1. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
2. Bellec, G., Scherr, F., Subramoney, A., et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*.
3. Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*.
4. Davies, M., Srinivasa, N., Lin, T.-H., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*.
5. Davies, M., Wild, A., Orchard, G., et al. (2021). Advancing neuromorphic computing with Loihi 2. *Computer*.
6. Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*.
7. Fang, W., Yu, Z., Zhou, Y., et al. (2021). Deep residual learning in spiking neural networks. *NeurIPS*.
8. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
9. Grossberg, S. (2013). Adaptive resonance theory: how a brain learns to consciously attend, learn, and recognize a changing world. *Neural Networks*.
10. Hasani, R., Lechner, M., Amini, A., et al. (2021). Liquid time-constant networks. *AAAI*.
11. Hawkins, J., Clay, V., & Leadholm, N. (2024). The Thousand Brains Project: A new paradigm for sensorimotor intelligence. *arXiv preprint arXiv:2412.18354*.
12. Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
13. Heins, C., Mirza, M. B., Parr, T., et al. (2022). pymdp: A Python library for active inference in discrete state spaces. *Journal of Open Source Software*.
14. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
15. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
16. Krotov, D., & Hopfield, J. J. (2021). Large associative memory problem in neurobiology and machine learning. *NeurIPS*.
17. Leadholm, N., Clay, V., et al. (2026). Thousand-Brains systems: sensorimotor intelligence for rapid, robust learning and inference. *Neural Computation*.
18. Liu, J., et al. (2025). Neural Brain: A neuroscience-inspired framework for embodied agents. *arXiv preprint arXiv:2505.07634*.
19. Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks*.
20. Mallya, A., & Lazebnik, S. (2018). PackNet: adding multiple tasks to a single network by iterative pruning. *CVPR*.
21. Merolla, P. A., Arthur, J. V., Alvarez-Icaza, R., et al. (2014). A million spiking-neuron integrated circuit with a scalable communication network and interface. *Science*.
22. Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*.
23. PCL Authors. (2025). Predictive Coding Light. *Nature Communications*.
24. Quiroga, R. Q., Reddy, L., Kreiman, G., et al. (2005). Invariant visual representation by single neurons in the human brain. *Nature*.
25. Ramsauer, H., Schäfl, B., Lehner, J., et al. (2021). Hopfield networks is all you need. *ICLR*.
26. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
27. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., et al. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.
28. Salvatori, T., Mali, A., Buckley, C. L., et al. (2023). Brain-inspired computational intelligence via predictive coding. *arXiv preprint arXiv:2308.07870*.
29. Schuman, C. D., Potok, T. E., Patton, R. M., et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*.
30. Xu, H., et al. (2026). Spike-driven large language model. *arXiv preprint arXiv:2604.16475*.
31. Zhu, R.-J., Zhao, Q., & Eshraghian, J. K. (2023). SpikeGPT: generative pre-trained language model with spiking neural networks. *arXiv preprint arXiv:2302.13939*.
