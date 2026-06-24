# Bio-ARN 2.0: The Embodied Mind Architecture

## A Brain-Inspired, Low-Power, Multi-Modal Generative Architecture

**Version:** 2.0 — Deep Research Synthesis  
**Date:** June 2026  
**Authors:** Bradley & Moss

---

## 0. Manifesto

Current LLMs are **giant statistical parrots** — they predict the next token by brute-forcing billions of parameters through dense matrix multiplications on power-hungry GPUs. A human brain runs on **20 watts**. GPT-4-class models require **megawatts** of training compute and hundreds of kilowatts for inference. Something is fundamentally broken.

The brain doesn't work by backpropagating error through a monolithic network. It works by:
- **Predicting** its own sensory input (predictive coding / free energy principle)
- **Learning locally** with Hebbian-style rules, not global backprop
- **Firing sparsely** — only the neurons that *need* to fire, fire
- **Knowing when it doesn't know** — metacognitive monitoring and honest abstention
- **Embodying cognition** — concepts are grounded in sensorimotor experience, not just text
- **Operating continuously** — no "training phase" then "inference phase"; it's always both

**Bio-ARN 2.0** is a blueprint for an architecture that embodies ALL of these principles. It is not a transformer with spikes bolted on. It is not an RNN with attention added. It is a fundamentally different computational paradigm.

---

## 1. The Grand Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BIO-ARN 2.0 SYSTEM                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 GLOBAL NEURONAL WORKSPACE (GNW)              │  │
│  │            "The Conscious Broadcast Channel"                 │  │
│  │   • Temporary amplification of salient CCC activations      │  │
│  │   • Sequential thought / inner speech / planning            │  │
│  │   • Attentional spotlight (winner-take-all + fatigue)       │  │
│  └──────┬──────────────┬──────────────┬────────────────────────┘  │
│         │              │              │                            │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐                   │
│  │  CONCEPT    │ │  CONCEPT   │ │  CONCEPT    │  ... × N         │
│  │  CELL       │ │  CELL      │ │  CELL       │                  │
│  │  CLUSTER    │ │  CLUSTER   │ │  CLUSTER    │                  │
│  │  (CCC)      │ │  (CCC)     │ │  (CCC)      │                  │
│  │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐  │                  │
│  │ │ F1     │ │ │ │ F1     │ │ │ │ F1     │  │                  │
│  │ │ Input  │ │ │ │ Input  │ │ │ │ Input  │  │                  │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤  │                  │
│  │ │ F2     │ │ │ │ F2     │ │ │ │ F2     │  │                  │
│  │ │Concept│ │ │ │Concept│ │ │ │Concept│   │                  │
│  │ │ Neuron│ │ │ │ Neuron│ │ │ │ Neuron│   │                  │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤  │                  │
│  │ │ Margin │ │ │ │ Margin │ │ │ │ Margin │  │                  │
│  │ │ Gate   │ │ │ │ Gate   │ │ │ │ Gate   │   │                  │
│  │ │(Abstain│ │ │ │(Abstain│ │ │ │(Abstain│   │                  │
│  │ │ or     │ │ │ │ or     │ │ │ │ or     │   │                  │
│  │ │ Fire)  │ │ │ │ Fire)  │ │ │ │ Fire)  │   │                  │
│  │ └────────┘ │ │ └────────┘ │ │ └────────┘  │                  │
│  └──────┬──────┘ └─────┬──────┘ └────┬───────┘                   │
│         │              │              │                            │
│  ┌──────▼──────────────▼──────────────▼─────────────────────────┐ │
│  │           ASSOCIATIVE FABRIC (Hebbian + SDM)                │ │
│  │   • Sparse distributed connections between CCCs              │ │
│  │   • Kanerva-style address-based retrieval                    │ │
│  │   • STDP-governed plasticity                                 │ │
│  │   • Lateral inhibition (winner-take-most)                    │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
│                             │                                      │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │              PREDICTIVE ENGINE (PE)                            │ │
│  │   • Hierarchical predictive coding                           │ │
│  │   • Top-down predictions → Bottom-up errors                   │ │
│  │   • Free energy minimization (Friston)                       │ │
│  │   • Active inference: actions to reduce prediction error      │ │
│  │   • Resonance detection: prediction ↔ input match → learn   │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
│                             │                                      │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │         EMBODIED SENSORIMOTOR CORTEX (eSMC)                  │ │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │ │
│  │   │ Vision  │ │ Audio   │ │ Touch   │ │ Proprio │          │ │
│  │   │ SNN     │ │ SNN     │ │ SNN     │ │ SNN     │          │ │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │ │
│  │        └────────────┴────────────┴────────────┘               │ │
│  │                    │                                          │ │
│  │   ┌────────────────▼────────────────────┐                    │ │
│  │   │    SENSORY CORTEX (Predictive SNN)  │                    │ │
│  │   │    Predict own input; send errors   │                    │ │
│  │   │    up to PE                         │                    │ │
│  │   └────────────────┬────────────────────┘                    │ │
│  │                    │                                          │ │
│  │   ┌────────────────▼────────────────────┐                    │ │
│  │   │    MOTOR CORTEX (Action SNN)        │                    │ │
│  │   │    Translate PE commands into        │                    │ │
│  │   │    motor sequences / text / speech   │                    │ │
│  │   └─────────────────────────────────────┘                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         REWARD & NOVELTY SYSTEM (Dopaminergic)               │ │
│  │   • Intrinsic: prediction error reduction → dopamine signal  │ │
│  │   • Extrinsic: external reward signals                       │ │
│  │   • Novelty detector: orienting response to unexpected input  │ │
│  │   • Modulates STDP strength and CCC margin thresholds        │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Deep Dives

### 2.1 Concept Cell Clusters (CCCs) — The "Cortical Columns"

**Inspiration:** Jeff Hawkins' Thousand Brains Theory (cortical columns as the repeating computational unit), combined with the user's concept-cell design (margin-based firing, honest abstention, unit-direction weights).

**What it is:** Each CCC is a self-contained micro-circuit that represents a *concept* — not a single feature, but a complete model of something in the world (an object, a word, an action, an abstract idea). Crucially, CCCs operate like cortical columns: each one can model an *entire object*, not just a feature. Multiple CCCs vote on what they're experiencing, producing a distributed, robust consensus.

**Internal Structure:**

```
         ┌─────────────────────────────────┐
         │          CCC (Concept Cell       │
         │             Cluster)             │
         │                                  │
         │  ┌────────────────────────────┐ │
         │  │  F1: Feature Input Layer    │ │
         │  │  - Receives sparse sensory  │ │
         │  │    patterns from eSMC       │ │
         │  │  - Competitive inhibition   │ │
         │  │  - Only strongest features  │ │
         │  │    survive                  │ │
         │  └──────────┬─────────────────┘ │
         │             │                   │
         │  ┌──────────▼─────────────────┐ │
         │  │  F2: Concept Neuron Layer  │ │
         │  │  - Unit-direction weights  │ │
         │  │  - Each neuron = a "concept │ │
         │  │    pointer" in high-dim     │ │
         │  │  - Sparse activation        │ │
         │  └──────────┬─────────────────┘ │
         │             │                   │
         │  ┌──────────▼─────────────────┐ │
         │  │  MARGIN GATE               │ │
         │  │  ┌──────────────────────┐  │ │
         │  │  │ confidence = cos(θ)  │  │ │
         │  │  │ between input pattern│  │ │
         │  │  │ and concept direction│  │ │
         │  │  │                      │  │ │
         │  │  │ if confidence > θ_margin│ │
         │  │  │   → FIRE (activate)  │  │ │
         │  │  │ else                  │  │ │
         │  │  │   → ABSTAIN (signal  │  │ │
         │  │  │     "I don't know")   │  │ │
         │  │  └──────────────────────┘  │ │
         │  └──────────┬─────────────────┘ │
         │             │                   │
         │  ┌──────────▼─────────────────┐ │
         │  │  FEEDBACK PREDICTION       │ │
         │  │  - When F2 fires, it sends  │ │
         │  │    expected features back   │ │
         │  │    to F1 (top-down pred)    │ │
         │  │  - Match → resonance →     │ │
         │  │    learning (weights tune)  │ │
         │  │  - Mismatch → orienting     │ │
         │  │    response → search for    │ │
         │  │    better CCC match         │ │
         │  └────────────────────────────┘ │
         └─────────────────────────────────┘
```

**Key Innovation — The Margin Gate:** This is the core of honest abstention. Unlike softmax which always picks *something* (even if wrong), the margin gate has an explicit threshold. If the angular similarity between the input pattern and the concept's learned direction vector is below θ_margin, the CCC outputs an **abstention signal** rather than a low-confidence guess. This is truthfulness built into the architecture.

**Learning in CCCs:**
- **Fast learning (one-shot):** When a novel pattern arrives and no existing CCC fires (all abstain), a new CCC is recruited. It immediately encodes the pattern direction.
- **Slow tuning (Hebbian):** When a CCC fires and resonates with input, its weights gently shift toward the input pattern (STDP-like, but in vector space: move the direction vector slightly toward the input).
- **No backpropagation required.** Learning is entirely local within each CCC.

**Connection to Thousand Brains:** Multiple CCCs process the same input in parallel. Each may form a different hypothesis. They vote through the Associative Fabric. Consensus emerges from distributed agreement, not from a single "decision layer."

---

### 2.2 Associative Fabric — The "Connectome"

**Inspiration:** Kanerva's Sparse Distributed Memory (SDM), Hebbian plasticity, and the brain's white-matter connectome.

**What it is:** A sparse, high-dimensional content-addressable memory that links CCCs together. It's not a dense attention matrix (O(n²)) — it's an O(k) sparse connection graph where k << n.

**How it works:**

1. **Address Space:** Each CCC has a high-dimensional (e.g., 10,000-dim) binary "address" derived from its concept direction vector. These addresses are sparse (mostly zeros), following Kanerva's principle.

2. **Content-Addressable Storage:** When CCC_A and CCC_B co-activate (fire simultaneously or in close temporal proximity), a Hebbian association is formed. The "data" stored at the intersection of their addresses is strengthened.

3. **Retrieval by Partial Cue:** Given a partial or noisy cue, the SDM retrieves the closest matching pattern by collecting data from all addresses within a Hamming radius of the cue. This gives us **robust associative recall from partial information** — just like the brain.

4. **Lateral Inhibition:** Nearby (similar) CCCs compete. When CCC_A fires strongly, it inhibits CCCs with overlapping addresses. This prevents runaway activation and ensures sparse, clean recall.

5. **Temporal Binding:** Associations include a temporal dimension. CCC_A → CCC_B connections strengthen when A fires before B (causal/predictive). This is how the system learns sequences — not by memorizing token order, but by learning causal-temporal associations.

**Why this beats attention:** Transformer attention is O(n²) in sequence length and must recompute from scratch every time. The Associative Fabric is:
- O(k) sparse (only k nearby addresses activated)
- Content-addressable (retrieve by meaning, not position)
- Incrementally updatable (new memories don't require retraining)
- Robust to noise (Kanerva's key insight)

---

### 2.3 Predictive Engine (PE) — The "World Model"

**Inspiration:** Karl Friston's Free Energy Principle, Predictive Coding (Rao & Ballard 1999, Salvatori et al. 2023), and Predictive Coding Light (PCL, Nature Communications 2025).

**What it is:** A hierarchical network where each layer predicts the activity of the layer below, and prediction errors propagate upward. This is the **generative engine** of the system — it doesn't just process input; it constantly generates predictions about what it *expects* to experience.

**How it works:**

```
        ┌───────────────────────────────────────────┐
        │           HIERARCHICAL PE                  │
        │                                           │
        │  Level 3: Abstract Concepts               │
        │  "This is a social situation,              │
        │   someone is asking a question"           │
        │        │                                   │
        │        │ Predictions ↓     Errors ↑        │
        │        ▼                                   │
        │  Level 2: Object/Event Models             │
        │  "A person is standing near me,            │
        │   their mouth is moving"                   │
        │        │                                   │
        │        │ Predictions ↓     Errors ↑        │
        │        ▼                                   │
        │  Level 1: Sensory Feature Predictions     │
        │  "I expect to hear speech sounds,          │
        │   see a face, feel vibrations"             │
        │        │                                   │
        │        │ Predictions ↓     Errors ↑        │
        │        ▼                                   │
        │  Level 0: eSMC (Raw Sensory Input)         │
        │  Actual auditory waveform, visual pixels   │
        └───────────────────────────────────────────┘
```

**Key Insight from PCL:** In Predictive Coding Light (published Nature Comms 2025), the key innovation is that **only the prediction errors are transmitted upward** — the predictable spikes are *suppressed*. This means:
- The system transmits minimal information (energy efficiency)
- Learning focuses on what's *surprising* or *novel*
- Well-predicted inputs require almost no processing (the brain is "bored" by the predictable)

**Generative Process:** When the PE is running in "generation mode" (imagination, planning, or producing output):
1. A high-level CCC activates (e.g., "I want to describe a sunset")
2. The PE generates top-down predictions at each level
3. These predictions cascade down through the eSMC
4. The Motor Cortex translates predicted features into output (text, speech, action)

This is **generation by prediction**. The system doesn't "sample tokens." It *predicts* what it would experience, and that prediction IS the output.

**Active Inference Extension:** The PE doesn't just passively predict — it can *act* to reduce prediction error. If the PE predicts "there should be a cat here" but sensory input doesn't confirm it, the system can:
- **Look** (move eyes/sensors to gather confirming input)
- **Ask** (generate language to request information)
- **Explore** (move the body to change sensory perspective)

This is the **embodied** part — cognition drives action, and action changes what is sensed, closing the loop.

---

### 2.4 Embodied Sensorimotor Cortex (eSMC) — The "Body"

**Inspiration:** The Neural Brain framework (Liu et al. 2025), SpikeGPT (Zhu et al. 2023), SDLLM (Xu et al. 2026), and neuroscience of the mammalian cortex.

**What it is:** The physical and functional interface with the world. Not a passive input/output layer — an active, predicting, acting system.

**Sub-components:**

#### 2.4.1 Sensory Sub-Cortex (Perception)
- **Visual Stream:** Event-driven SNN (like a DVS camera). Only changes are transmitted. A still scene costs near-zero energy.
- **Auditory Stream:** Cochlear-model SNN. Spectrotemporal features extracted via spike timing.
- **Somatosensory Stream:** Tactile SNN. Pressure, temperature, texture encoded as spike trains.
- **Proprioceptive Stream:** Body-position SNN. Where are my limbs? Am I balanced?

Each sensory stream runs **Predictive Coding Light**: it predicts its own next input. Predictable input → spikes suppressed → energy saved. Novel input → spikes transmitted → learning triggered.

#### 2.4.2 Motor Sub-Cortex (Action)
- **Language Motor Stream:** Converts CCC concept sequences into phoneme/character sequences via predictive SNN. This is how the system "speaks" or "writes."
- **Physical Motor Stream:** Converts high-level action CCCs into limb movement sequences. For a physical robot, this would drive actuators. For a virtual agent, it drives navigation and manipulation.

**The Sensorimotor Loop:**
```
Sensory Input → eSMC Sensory → PE Predictions → CCC Activation →
GNW Attention → Motor Planning → eSMC Motor → Action →
World Response → Sensory Input (loop continues)
```

This closed loop is the **embodied** part. The brain is never disconnected from the body. Every thought has a motor component (even if just inner speech or eye movements).

---

### 2.5 Global Neuronal Workspace (GNW) — "Consciousness Lite"

**Inspiration:** Dehaene's Global Neuronal Workspace Theory, Baars' Consciousness Theory, and working memory research.

**What it is:** A broadcast mechanism that amplifies a small subset of active CCCs, making their content globally available for:
- Sequential reasoning (chaining thoughts)
- Language generation (translating concepts to words)
- Planning (simulating future actions)
- Metacognitive monitoring (reflecting on one's own thoughts)

**How it works:**
1. Multiple CCCs activate in parallel (unconscious processing)
2. The GNW selects a subset via **competitive attention** — winner-take-most with temporal fatigue
3. Selected CCCs are "broadcast" to all other CCCs via the Associative Fabric
4. This broadcast triggers associations, predictions, and further processing
5. After a brief period, fatigue sets in and the current winners fade, allowing new CCCs to enter the workspace
6. This creates the **stream of consciousness** — a serial sequence of globally broadcast concepts

**Connection to Language Generation:**
When producing language, the GNW holds the current "thought" (a set of co-active CCCs), which activates associated CCCs via the Associative Fabric, which trigger motor predictions in the eSMC, which produce the next word. The produced word then feeds back through the sensory stream, confirming or correcting the prediction, and the cycle continues.

This is **fundamentally different from autoregressive token generation**. The system doesn't just predict the next token — it:
1. Holds a conceptual state in GNW
2. Activates associated concepts
3. Generates predictions across ALL modalities (not just text)
4. Only the language motor stream is what we see as "output"

---

### 2.6 Reward & Novelty System — The "Motivation"

**Inspiration:** Dopaminergic reward circuitry, intrinsic motivation, curiosity-driven learning.

**What it is:** The system's motivational engine. It provides the "why" — why learn anything at all?

**Sub-components:**

1. **Prediction Error Reward (Intrinsic):** When the PE successfully reduces prediction error, a reward signal is generated. This is the brain's "aha!" feeling. It drives the system to build better world models.

2. **Novelty Detector (Orienting Response):** When input is *very* different from predictions (high prediction error that can't be reduced quickly), an orienting response is triggered:
   - Current GNW content is disrupted
   - Attention shifts to the novel stimulus
   - Learning rates are temporarily boosted
   - New CCCs may be recruited

3. **External Reward (Optional):** Task-specific reward signals can be provided. These modulate the strength of STDP and CCC margin adjustments, allowing the system to be steered toward useful goals.

4. **Curiosity Drive:** The system actively seeks out situations where it can learn the most (maximum prediction error reduction per unit effort). This drives exploration and play-like behavior.

**Why this matters:** Without motivation, a system has no reason to learn, explore, or improve. The reward system provides the intrinsic drive that makes the system an *active learner* rather than a passive data processor.

---

## 3. How It All Flows Together: The Full Loop

### 3.1 Perception (Understanding)

```
1. Raw sensory input arrives at eSMC
2. Sensory SNNs encode input as sparse spike trains
3. PCL: Predictable spikes are suppressed; only novel/error spikes go up
4. Prediction errors propagate up through PE hierarchy
5. CCCs compete to explain the errors (margin gates fire or abstain)
6. Winning CCCs enter the GNW (conscious perception)
7. GNW broadcasts activate associated CCCs via Associative Fabric
8. Top-down predictions cascade back down (confirming expectations)
9. Resonance: when predictions match input → learning (weight tuning)
10. Mismatch: when predictions fail → novelty response, search for better CCCs
```

### 3.2 Generation (Language Output)

```
1. High-level goal or context activates CCCs in GNW (e.g., "Answer the question")
2. GNW holds conceptual state (what to say, not how to say it)
3. PE generates top-down predictions: "I expect to produce words about X"
4. Motor predictions cascade to language motor stream in eSMC
5. Language motor SNN converts concept sequence to phoneme/character sequence
6. Output is produced (text, speech, etc.)
7. Internal sensory feedback: the system "hears" or "sees" its own output
8. Prediction errors from self-monitoring adjust future generation
9. GNW shifts to next conceptual state (serial thought chain)
10. Repeat until concept sequence is complete
```

### 3.3 Learning (Continuous, No Separate Training)

```
1. During perception: CCC weights tune via Hebbian/STDP when resonance occurs
2. During generation: prediction errors from self-monitoring tune motor mappings
3. Novel input: new CCCs recruited (one-shot learning of new concepts)
4. Associative Fabric: co-activation strengthens temporal links
5. Reward signals: modulate learning rates (dopamine = "this matters, learn more")
6. Sleep/consolidation (optional): offline replay to strengthen recent memories
```

---

## 4. Comparison with Existing Architectures

| Feature | Transformer LLM | SpikeGPT | Thousand Brains (Monty) | Active Inference (pymdp) | **Bio-ARN 2.0** |
|---------|----------------|----------|--------------------------|--------------------------|-----------------|
| Core paradigm | Next-token prediction | Spiking next-token | Sensorimotor inference | Free energy minimization | Predictive embodied resonance |
| Learning | Backprop (batch) | Backprop (batch) | Hebbian (online) | Variational (online) | Hebbian + STDP (online, continuous) |
| Energy | O(n²) attention | O(T) linear, spikes | Moderate | Moderate | O(k) sparse + spikes |
| Multimodal | Grafted on | Text only | Vision + touch | Any | Native (all senses unified) |
| Abstention | No (softmax always picks) | No | Implicit | Yes (via precision) | **Explicit margin gate** |
| Continual learning | Catastrophic forgetting | Catastrophic forgetting | Yes (inherent) | Yes (inherent) | Yes (inherent, resonance-gated) |
| Embodiment | None | None | Yes (sensorimotor) | Yes (active inference) | **Full (sensorimotor loop)** |
| Generative | Autoregressive sampling | Autoregressive spikes | Action generation | Policy generation | **Prediction-as-generation** |
| Memory | Context window (finite) | Context window | Associative (SDM-like) | State-space | **Kanerva SDM + associative** |
| Consciousness analogue | None | None | None | None | **GNW broadcast** |

---

## 5. Research Foundations & Proven Precursors

Bio-ARN 2.0 doesn't invent from nothing. Each component is grounded in published, proven research:

| Component | Foundation | Key Reference |
|-----------|-----------|---------------|
| Concept Cell Clusters | Thousand Brains cortical columns | Hawkins et al. (2024) — arXiv:2412.18354 |
| Margin Gate / Abstention | Concept-cell margin firing (user's design) | Liebenberg (2026) — MarginLM concept |
| Predictive Coding | PC networks, Friston's Free Energy | Salvatori et al. (2023) — arXiv:2308.07870 |
| Predictive Coding Light | Spike suppression of predictable input | PCL (2025) — Nature Communications |
| Spiking Neural Networks | LIF neurons, event-driven processing | SpikeGPT (Zhu et al. 2023) |
| Spike-Driven LLM | Binary spikes, sparse addition | SDLLM (Xu et al. 2026) — arXiv:2604.16475 |
| Associative Memory | Kanerva's SDM, Modern Hopfield Networks | Krotov & Hopfield (2021); Ramsauer et al. (2020) |
| Active Inference | Free Energy Principle, pymdp | Friston; Heins et al. (2022) — pymdp |
| Global Neuronal Workspace | Dehaene's GNW theory | Dehaene & Changeux (2011) |
| Liquid Neural Networks | Adaptive time constants, causality | Hasani et al. (2021); Liquid AI |
| Embodied Cognition | Neural Brain framework | Liu et al. (2025) — arXiv:2505.07634 |
| Neuromorphic Hardware | Intel Loihi 2, IBM NorthPole | Intel Labs; IBM Research |

---

## 6. The Mathematical Core

### 6.1 Concept Cell Cluster Dynamics

Each CCC operates as follows:

**Input encoding:**
```
a_input = W_direction · x + b
```
where `W_direction` is the unit-direction weight matrix and `x` is the sparse input pattern.

**Margin gate (honest abstention):**
```
confidence = cos_similarity(a_input, concept_direction)
if confidence > θ_margin:
    output = a_input  (FIRE)
else:
    output = ABSTAIN_SIGNAL  (don't know)
```

**Resonance check (adaptive resonance):**
```
prediction = W_feedback · a_fired
match_score = cos_similarity(prediction, actual_input)
if match_score > θ_resonance:
    LEARN: W_direction ← normalize(W_direction + α · x)
           W_feedback ← W_feedback + β · (x - prediction)
else:
    SEARCH: trigger orienting response, try next CCC
```

### 6.2 Predictive Coding (Hierarchical)

At each level l of the PE hierarchy:
```
ε_l = x_l - f(W_l · x_{l+1})     # Prediction error (bottom-up)
x_{l+1} ← x_{l+1} + γ · W_l^T · ε_l  # State update (top-down)
W_l ← W_l + η · ε_l · x_{l+1}^T    # Weight update (local Hebbian)
```

Key: **No backpropagation.** All learning is local to each level. Errors propagate one level at a time, just like in the brain.

### 6.3 Associative Fabric (Kanerva SDM)

Given a high-dimensional address space of N dimensions (N >> number of CCCs):
```
Address(CCC_i) = threshold(CCC_i.concept_direction, 0.5)  # Binary address

Store: For co-active CCCs i, j:
    Data[Address(i) ⊙ Address(j)] += 1  (Hebbian increment)

Retrieve: Given partial cue address c:
    result = Σ(Data[addr] for addr where Hamming(addr, c) < r)
    # Where r is the Hamming radius (activation threshold)
```

### 6.4 Global Neuronal Workspace (GNW) Selection

```
GNW_contents = top_k(active_CCCs, by=activation_strength, k=7±2)
# The "7±2" reflects Miller's Law for working memory capacity

For each CCC in GNW_contents:
    CCC.activation *= broadcast_amplification
    CCC.associates = Fabric.retrieve(CCC.address)  # Trigger associations
    CCC.fatigue_timer = T_fatigue  # Will fade after T_fatigue steps
```

---

## 7. Energy Efficiency Analysis

### 7.1 Where the Savings Come From

| Mechanism | Transformer | Bio-ARN 2.0 | Savings |
|-----------|------------|-------------|---------|
| Attention | O(n²) dense softmax | O(k) sparse SDM retrieval | 100-1000× |
| Activation | All neurons compute | Only margin-gated neurons fire | 10-100× |
| Spikes | Continuous values | Binary spikes (event-driven) | 10-32× |
| Prediction | Full forward pass | Only errors transmitted (PCL) | 5-50× |
| Learning | Full backprop | Local Hebbian (no backward pass) | 100× |
| Memory | Full context window | Kanerva SDM (content-addressed) | 10× |

**Conservative estimate:** Bio-ARN 2.0 should achieve comparable cognitive performance at **100-1000× less compute** than a transformer LLM of equivalent capability, with **10-100× less energy** for inference.

### 7.2 Hardware Targets

- **Near-term:** Simulation on GPU/TPU with sparse operations (research prototype)
- **Mid-term:** Intel Loihi 2 / IBM NorthPole neuromorphic chips (spiking operations natively)
- **Long-term:** Custom neuromorphic ASIC with native SDM, predictive coding, and margin gates

---

## 8. Implementation Roadmap

### Phase 0: Proof of Concept (Weeks 1-4)
- Implement a single CCC with margin gate
- Test on simple pattern recognition (MNIST or similar)
- Verify one-shot learning and honest abstention
- Implement basic predictive coding loop (single level)

### Phase 1: Core System (Weeks 5-12)
- Implement multi-CCC system with Associative Fabric
- Implement GNW selection mechanism
- Test on multi-modal input (vision + text)
- Verify continual learning without catastrophic forgetting

### Phase 2: Embodied Loop (Weeks 13-20)
- Implement eSMC sensory streams (at least vision + language)
- Implement motor output stream (language generation)
- Close the sensorimotor loop
- Test on embodied tasks (virtual environment)

### Phase 3: Scale & Benchmark (Weeks 21-30)
- Scale to thousands of CCCs
- Benchmark against transformer baselines
- Optimize for energy efficiency
- Test on language generation tasks

### Phase 4: Hardware (Weeks 31+)
- Port to neuromorphic hardware (Loihi 2)
- Optimize spike encoding for real silicon
- Measure actual energy consumption
- Compare watts-per-inference with GPU-based LLMs

---

## 9. The Big Bet

The fundamental hypothesis of Bio-ARN 2.0 is:

> **Intelligence is not about processing more data faster. It's about building better predictive models with less data and less energy, by knowing when you're confident, when you're uncertain, and when to act to reduce uncertainty.**

If this hypothesis is correct, then:
1. A Bio-ARN system should learn faster than a transformer (one-shot vs. millions of examples)
2. It should run on far less energy (sparse + spikes + local learning)
3. It should be more trustworthy (honest abstention when uncertain)
4. It should be more adaptable (continual learning, no retraining)
5. It should be more general (embodied cognition, multi-modal by design)

The brain proves this is possible. We just need to understand its principles deeply enough to build them into silicon.

---

## 10. Open Questions & Risks

1. **Scaling associative retrieval:** Kanerva's SDM works beautifully at small scale. Can it scale to millions of concepts without degradation? Modern Hopfield Networks suggest yes, but this needs validation.

2. **Sequence generation quality:** Autoregressive transformers are extremely good at generating fluent text. Can a predictive-coding-based system achieve similar fluency? The PCL paper and predictive coding literature suggest generative quality can be high, but this is the biggest open question.

3. **Training data:** The brain uses multi-modal, embodied experience. What data does Bio-ARN need? Raw video + audio + text? Simulated environments? Real robot interaction?

4. **Hardware gap:** Current neuromorphic chips (Loihi 2) are limited in neuron count and connectivity. Simulation on GPUs will be needed for prototyping, but won't achieve the full energy benefits.

5. **The "cognitive catch-22":** To build a system that learns like a brain, you might need a brain-like training environment. A disembodied text-only system might not develop genuine conceptual understanding. How much embodiment is needed?

---

*"The brain is not a computer. It is an adaptive, predictive, embodied organ that builds models of the world through active engagement. Bio-ARN 2.0 aims to be the same."*

---

**References:**

1. Hawkins, J., Clay, V., Leadholm, N. (2024). "The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence." arXiv:2412.18354.
2. Leadholm, N., Clay, V., et al. (2026). "Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference." Neural Computation, 38(6), 845-896.
3. Salvatori, T., Mali, A., Buckley, C., et al. (2023). "Brain-inspired Computational Intelligence via Predictive Coding." arXiv:2308.07870.
4. PCL Authors (2025). "Predictive Coding Light." Nature Communications.
5. Zhu, R.-J., et al. (2023). "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks." arXiv:2302.13939.
6. Xu, H., et al. (2026). "Spike-driven Large Language Model." arXiv:2604.16475.
7. Friston, K. (2010). "The Free-Energy Principle: A Unified Brain Theory?" Nature Reviews Neuroscience.
8. Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
9. Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." ICLR 2021.
10. Krotov, D., & Hopfield, J. (2021). "Modern Hopfield Networks and Attention Mechanisms." arXiv:2008.02288.
11. Heins, C., et al. (2022). "pymdp: A Python library for active inference in discrete state spaces." JOSS.
12. Liu, J., et al. (2025). "Neural Brain: A Neuroscience-inspired Framework for Embodied Agents." arXiv:2505.07634.
13. Hasani, R., et al. (2021). "Liquid Time-Constant Networks." AAAI 2021.
14. Dehaene, S., & Changeux, J.-P. (2011). "Experimental and Theoretical Approaches to Conscious Processing." Neuron.
15. Intel Labs. "Loihi 2: A Research Chip for Neuromorphic Computing."
16. SDM implementations: github.com/SparseDistributedMemory/SparseDistributedMemory
17. Monty implementation: github.com/thousandbrainsproject/tbp.monty
18. pymdp: github.com/infer-actively/pymdp