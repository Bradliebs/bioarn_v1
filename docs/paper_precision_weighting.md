# Precision-Weighted Hebbian Learning: Hippocampal Ripple-Inspired Uncertainty Gating for Neuromorphic Systems

## Abstract

Predictive coding and Hebbian learning are often treated as mutually frustrating design commitments. Predictive coding wants iterative settling that suppresses predictable activity and refines latent states through recurrent error minimization. Hebbian concept learning, by contrast, depends on preserving the discriminative structure of the feedforward activity pattern that recruited the concept in the first place. In Bio-ARN 2.0, direct coupling of iterative predictive settling to sparse concept-cell learning produced exactly this conflict: CIFAR-10 accuracy collapsed from 30.0% to 11.8% when settling smoothed away the feature contrasts needed by local Hebbian updates. This paper argues that the conflict is not between prediction and Hebb per se, but between **state-settling predictive inference** and **feature-selective local plasticity**. We present a precision-weighted alternative inspired by Frank et al. (2026), who showed that human hippocampal ripples increase before uncertain stimuli and tune cortical responses based on predicted uncertainty. Bio-ARN instantiates this idea with a pool-level entropy estimator over recent Concept Cell Cluster (CCC) winners, a sigmoid precision transform, and single-pass prediction-error gating that modulates learning without rewriting the representation. The result is a predictive mechanism that works with Hebbian learning instead of against it: closed-set CIFAR-10 accuracy is preserved at 30.0%, class-split forgetting improves from 34.7% to 20.7%, OOD AUROC rises from 0.778 to 1.000, and the full Sprint I stack reaches 33.0% accuracy with 13.7% forgetting while preserving the projected 278x Loihi-2 energy advantage over a matched transformer baseline.

## 1. Introduction

Brain-inspired learning systems repeatedly run into a familiar tension. On one side sits the Hebbian principle: local co-activity should strengthen useful representations, preserving feature conjunctions that reliably predict future success. On the other side sits predictive coding: the system should suppress what is already predicted and propagate only residual error upward through a hierarchy [2, 4]. Both principles are attractive in isolation. Hebbian learning supports online plasticity, locality, and neuromorphic plausibility. Predictive coding promises efficient representation, novelty sensitivity, and a principled account of top-down expectation. Yet when these ideas are combined naively, they can become architecturally incompatible.

The source of the incompatibility is not philosophical. It is computational. In many predictive-coding formulations, latent states are iteratively updated until the system settles into a self-consistent representation. That dynamic is often harmless or even useful in gradient-based systems whose parameters are optimized end to end. In a sparse concept architecture such as Bio-ARN, however, the feedforward activation pattern is itself the substrate of concept commitment. A Concept Cell Cluster is recruited because a particular feature pattern stands out strongly enough to justify a new concept. If recurrent settling smooths or homogenizes that pattern before learning, the system no longer learns the feature contrast that made the input informative. It learns a softened reconstruction of its own prior expectations.

This failure mode was not hypothetical in Bio-ARN. The companion Bio-ARN paper documents that enabling iterative predictive settling in the visual hierarchy drove real CIFAR-10 accuracy from 30.0% to 11.8% [6]. Replacing settling with a single-pass error gate recovered 30.0%, showing that prediction can still help if it modulates learning rather than rewriting state. That result establishes the central claim of this paper: the real problem is not predictive processing itself, but the **form** in which predictive processing is introduced into a local-learning system.

The present work focuses on a more selective solution: **precision-weighted prediction-error gating**. The mechanism is motivated by recent human neuroscience from Frank et al. (2026), who report that hippocampal ripples rise in frequency and duration before uncertain stimuli and tune cortical responses based on predicted uncertainty [1]. Their finding matters because it reframes prediction as a two-part process. The brain predicts not only *what* is likely to occur, but also *how much weight* to assign to the next error signal. In other words, uncertainty is itself predictive content.

Bio-ARN operationalizes that insight with a lightweight control circuit. Each CCC pool tracks the recent distribution of winners with a ring buffer. A normalized entropy statistic estimates whether current concept usage is concentrated and familiar or diffuse and uncertain. A sigmoid maps that entropy to a precision weight. Prediction errors are then multiplied by this precision weight and applied only once, as a learning-rate gate on local Hebbian refinement. High-entropy contexts remain plastic. Low-entropy contexts are protected. The feedforward representation is preserved, but the system still benefits from prediction.

This paper is therefore narrower than the main Bio-ARN architecture paper [6]. It does not attempt to reintroduce every subsystem. Instead, it isolates one technical question: **how can predictive coding be made compatible with sparse Hebbian concept learning?** Our answer is that predictive coding should be used as a meta-learning signal rather than a settling routine. Precision weighting turns prediction into a regulator of plasticity, not a destroyer of discriminative features.

The paper makes four contributions. First, it clarifies the settling problem as a specific incompatibility between iterative state relaxation and Hebbian concept recruitment. Second, it introduces a hippocampal-ripple-inspired precision mechanism based on pool entropy. Third, it extends this mechanism with lateral predictive coding and elastic concept protection so uncertainty can guide routing, replay, and memory preservation. Fourth, it reports a focused experimental picture: precision weighting preserves CIFAR-10 accuracy while improving forgetting and OOD behavior, and it does so without undermining Bio-ARN's neuromorphic efficiency claim.

## 2. Background

### 2.1 The Settling Problem

Classical predictive coding, following Rao and Ballard, describes cortical processing as an interaction between top-down predictions and bottom-up residuals [2]. A higher layer sends a prediction downward; a lower layer compares that prediction with its current state; the mismatch becomes a prediction error; and recurrent updates continue until the hierarchy reaches a locally consistent configuration. In theoretical terms this is elegant. Each layer carries a hypothesis, and only the unexplained portion of sensory evidence continues to matter.

In a sparse Hebbian system, however, there is a hidden assumption inside that elegance: the representation being settled is allowed to move substantially before learning occurs. Bio-ARN's Concept Cell Clusters violate that assumption. A CCC is not a generic hidden unit that can be arbitrarily reparameterized later by global backpropagation. It is a local concept detector with three coupled commitments: a sparse F1 encoding, a concept-space direction vector, and a feedback pathway that predicts the features associated with that concept. The crucial information is not only in the residual error but in the *specific pattern of winning features* that drove the detector to fire.

When iterative settling is added on top of this design, three problems appear.

First, settling reduces contrast among candidate winners. Lower-level features are repeatedly nudged toward higher-level predictions. For dense latent-variable models, this may improve consistency. For CCCs, it blurs the very discriminative margins that allow a pool to decide whether a concept should fire, abstain, or recruit a new slot. A concept cell trained on settled states becomes a detector of consensus-smoothed activity rather than of task-relevant novelty.

Second, settling entangles inference and plasticity too tightly. In Bio-ARN, local learning occurs through `learn_fast()` for recruitment and `learn_slow()` for Hebbian refinement. Both depend on sparse feature patterns that still reflect the input. If the input has already been iteratively rewritten by top-down predictions, Hebbian learning is no longer a data-driven update; it becomes an echo of the model's own prior. This is precisely how a supposedly predictive mechanism can become anti-discriminative.

Third, settling introduces instability into continual learning. The system must preserve old concepts while adapting to new ones. When predictive recurrence repeatedly drags lower-level activity toward previously committed attractors, new task evidence can be under-represented early and then over-corrected later, producing either failure to recruit or drift of existing prototypes. In effect, settling spends plasticity budget on making the representation more self-consistent before the architecture has decided whether the stimulus is actually familiar.

The empirical consequence in Bio-ARN was severe. The companion publication reports a four-layer visual hierarchy that reaches 30.0% accuracy on real CIFAR-10. Activating predictive settling reduced that performance to 11.8% [6]. That number matters because it is too large a drop to dismiss as tuning noise. It indicates that the architecture's learned feature ecology was being damaged by the predictive update itself.

A useful control was then introduced: **error gating without settling**. Instead of iteratively modifying the hidden state, the system computes a prediction once, measures the residual once, and uses that residual to scale local Hebbian learning. This simple change restored accuracy to 30.0%. Predictive coding still contributed information, but only as a scalar learning signal attached to the existing representation. That restored performance reveals the correct decomposition of the problem:

1. Predictive information is useful.
2. Repeated state settling is harmful in this architecture.
3. Therefore, prediction must influence *plasticity* more than it influences *representation*.

This framing also helps situate Bio-ARN relative to Adaptive Resonance Theory. Grossberg's early stability-plasticity program emphasized that learning systems need mechanisms that preserve stable categories while remaining open to novelty [3]. CCC margin gating, abstention, and recruitment fit that lineage. Iterative settling, as implemented in the failed Bio-ARN variant, violated it by allowing old predictions to erase the feature boundaries that separate stable recognition from new-category recruitment. The goal of precision weighting is to recover predictive processing without abandoning stability-plasticity discipline.

### 2.2 Hippocampal Ripples as Precision Signals

Frank et al. (2026) provide a biologically suggestive answer to the question of how prediction should interact with uncertainty [1]. Using direct recordings from the human hippocampus and visual cortex, they show that hippocampal ripple activity increases in frequency and duration **before** stimulus presentation in unpredictable visual contexts. This is the crucial observation. Ripples do not merely mark surprise after the fact. They rise in anticipation of uncertainty.

The paper's abstract makes three points especially relevant for machine architecture. First, prestimulus ripples reflect context- and experience-dependent prediction of predictability. The system is estimating how informative the next input is likely to be before it arrives. Second, those ripples suppress changes in occipital gamma activity associated with uncertainty while modulating post-stimulus gamma responses in fusiform cortex to genuinely surprising events. Third, the authors explicitly connect these observations to predictive-coding accounts of message passing and precision-weighted prediction errors [1].

This shifts the interpretation of predictive coding. The standard simplified story is that higher areas predict lower ones and residuals are propagated upward. Frank et al. suggest a richer story: the brain also predicts the *gain* that future residuals deserve. In Bayesian language, uncertainty determines the effective influence of the next error signal. In systems language, the architecture needs a control channel that says, "learn aggressively here" or "do not let this residual rewrite a stable memory."

The biological plausibility of such a channel is strengthened by the physiology of ripples themselves. The underlying sharp-wave-ripple events depend on tightly coordinated excitatory-inhibitory interactions in hippocampal circuitry, and the observed cortical effects are consistent with transient rebalancing of cortical gain before sensory input arrives [1]. For a neuromorphic designer, that is an important clue: precision need not be an expensive global optimizer. It can be a sparse, anticipatory, modulatory signal that changes how much local plasticity is permitted.

Bio-ARN does not attempt to reproduce ripple oscillations literally. It abstracts their computational role. The analogue is a **pool-level precision controller** attached to each CCC pool. Rather than estimating full posterior uncertainty, the controller tracks a simpler statistic: how concentrated or diffuse recent winner usage has been. If the same small subset of CCCs repeatedly wins, the pool is operating in a familiar regime. If winner usage spreads across many concepts, abstentions rise, or contextual predictions fail, the pool is uncertain about how to parse the current stream. That uncertainty should not force the architecture to settle. It should raise the learning gate.

This anticipatory interpretation also explains why surprise and uncertainty should not be conflated. Surprise is an outcome: the input differed from what was predicted. Uncertainty is a context: the architecture already knew it was entering a regime where many outcomes were plausible. Frank et al. argue that ripples are tied more closely to the latter. Bio-ARN mirrors that distinction by letting precision depend primarily on recent winner entropy, then optionally blend in lateral and hierarchy prediction error. The system estimates whether it is in a stable or unstable conceptual context before any single local update is amplified.

That is exactly what Hebbian learning needs. Local co-activity rules are powerful but blunt. Without modulation, they will reinforce whatever fires, including spurious or already-saturated concepts. Precision provides the missing meta-signal. It tells the local rule when coincidence is informative enough to matter.

## 3. Architecture

### 3.1 Pool Entropy Estimator

The precision mechanism begins with a pool-level uncertainty estimate computed from recent CCC winners. Each pool maintains a ring buffer of the last `L` winner sets, where `L = 100` in the current implementation. Let `c_i` be the number of times CCC `i` appeared in that window and `p_i = c_i / \sum_j c_j` be the normalized winner frequency. Bio-ARN defines the normalized pool entropy as:

$$
H_t = - \frac{1}{\log M_t} \sum_{i=1}^{M_t} p_i \log p_i,
$$

where `M_t` is the number of distinct active concepts in the current history support. By construction, `H_t \in [0,1]`. When a single CCC or a small stable set dominates the window, entropy approaches zero. When winner usage is diffuse across many concepts, entropy approaches one.

This estimator is intentionally simple. It is not trying to reconstruct a latent posterior over classes. It measures something more local and more actionable: *how predictable the current concept ecology is*. In a sparse concept architecture, that is the right level of abstraction. The question is not whether the pixel stream has high Shannon entropy in the abstract. The question is whether the architecture already knows which concepts explain the stream.

A few implementation choices matter. If the history is too short to be meaningful, Bio-ARN defaults to maximal uncertainty, encouraging plasticity during the warm-up phase. The pool size is updated dynamically as CCC capacity grows, so entropy remains normalized even when the concept bank expands. Preview mode allows the system to estimate entropy for the next presentation without mutating history, which is important for anticipatory gating.

The computational cost is negligible relative to the rest of the architecture. The controller only needs winner indices, not dense activations. On neuromorphic hardware, this makes it attractive: the precision path can be implemented as a small counter-based side process rather than as a second heavyweight network.

### 3.2 Precision Signal

Entropy alone is not yet a learning gate. Bio-ARN converts entropy into a bounded precision weight with a sigmoid transform:

$$
\pi_t = \pi_{\min} + (\pi_{\max} - \pi_{\min})\,\sigma\big(\alpha (H_t - \theta)\big),
$$

where `\sigma(z) = 1 / (1 + e^{-z})`. The current implementation uses `\alpha = 5.0`, threshold `\theta = 0.5`, and precision range `[0.1, 1.0]`. Low-entropy states therefore produce a small but nonzero learning multiplier, while high-entropy states approach full learning rate.

This mapping deserves a semantic note. In some predictive-coding literature, precision is discussed as inverse variance, so higher precision corresponds to higher certainty. Bio-ARN uses the term in a more operational engineering sense inherited from the role Frank et al. assign to predicted uncertainty. Here, the precision signal is the *gain assigned to upcoming prediction errors in informative contexts*. High uncertainty means the next residual may teach the system something new, so the error gets more leverage over learning. The functional role matches precision-weighting even if the control semantics are phrased as a plasticity gain.

The implementation also supports an external uncertainty blend:

$$
U_t = H_t + E_t - H_t E_t,
$$

where `E_t` aggregates lateral and hierarchy prediction errors. This inclusion-exclusion form keeps the combined uncertainty bounded in `[0,1]` while allowing any strong external mismatch to raise the gate. In effect, pool entropy supplies the contextual prior and local prediction errors supply event-specific evidence.

The resulting precision state is persistent. It is previewed before a sample is fully processed and then updated after actual winners and errors are observed. This gives the system a modest form of anticipatory control that fits the Frank et al. story: uncertainty is estimated before the decisive Hebbian update, not only after the input has already been absorbed.

### 3.3 Precision-Weighted Error Gating

The central architectural move is to replace iterative settling with **single-pass precision-weighted error gating**. A CCC first computes its ordinary feedforward representation: sparse F1 features, concept-space activation, and margin-gated firing. If it fires, it generates a top-down prediction of the feature pattern associated with that concept. The prediction error is simply the residual between the observed sparse features and the predicted ones.

The key difference from settling is that this residual is **not** used to repeatedly rewrite the state. Instead, it modulates learning. In simplified form, the local Hebbian update becomes:

$$
\Delta W \propto \eta_0 \pi_t \, e_t \, a_t,
$$

where `\eta_0` is the base local learning rate, `\pi_t` is the current precision weight, `e_t` is the single-pass residual or resonance-derived learning signal, and `a_t` is the local activity responsible for the update. In code, both concept-direction updates and feedback-weight updates are scaled by `learning_rate_multiplier = base_lr \times \pi_t`.

This achieves three things simultaneously.

First, it preserves the discriminative feedforward pattern. Category competition, abstention, and recruitment are still decided by the raw or lightly processed sparse features, not by a recurrently settled attractor.

Second, it retains predictive selectivity. If the system already predicts the input well, the residual is small and learning remains weak. If the residual is large and the pool context is uncertain, precision amplifies the update.

Third, it decouples representation from plasticity. The same error signal can be used to guide learning without forcing the current hidden state to collapse toward prior expectations. This is precisely why error gating preserved CIFAR-10 performance while settling did not.

In practice, the current Bio-ARN comparison is stark. Settling produces 11.8% CIFAR-10 accuracy. Plain error gating restores 30.0%. Precision weighting keeps the same 30.0% top-1 accuracy while improving continual learning and OOD behavior. The implication is not that predictive coding is weak. It is that predictive coding is strongest here when treated as a **meta-learning policy**.

### 3.4 Lateral Predictive Coding (Sprint I addition)

Sprint I extends the precision idea beyond top-down prediction by adding **lateral predictive coding** within CCC pools. Instead of asking only what a higher layer predicts about a lower one, Bio-ARN asks what active concepts predict about their neighbors.

Each committed CCC maintains a sparse set of lateral neighbors chosen by cosine similarity in concept space. When CCC `i` fires, the lateral network predicts which neighboring CCCs ought to co-fire and with what strength. Prediction error is then computed as the absolute mismatch between predicted and observed co-firing. Co-active neighbors strengthen their lateral weights through a Hebbian rule; false expectations are weakened anti-Hebbianly. The result is a lightweight local model of contextual regularity among concepts.

Precision weighting makes this lateral mechanism useful rather than noisy. Lateral errors are summarized into a pool-level mismatch score and converted into an attention boost:

$$
A_t = 1 + g \, \pi_t \, \varepsilon_t^{lat},
$$

where `g` is a surprise gain and `\varepsilon_t^{lat}` is the mean lateral prediction error. In effect, co-firing patterns that violate local expectations become more salient when the pool is already uncertain. This is not top-down settling in disguise. It is sparse, within-pool error-driven attention.

The practical advantage is that concepts can warn each other about context without flattening the feedforward representation. A stable dog-face concept can predict likely ear, fur, or snout concepts; when those neighbors fail to appear, attention increases locally instead of globally relaxing the whole representation. That fits Hebbian learning much better, because the system is still free to recruit or refine sparse concepts based on the original evidence.

### 3.5 Elastic Concept Protection (Sprint I addition)

Precision weighting alone is a soft solution. It decides when learning should be stronger or weaker, but it does not by itself guarantee that mature concepts will survive long task sequences. Sprint I therefore pairs precision weighting with **elastic concept protection**.

Each CCC tracks an importance score derived from usage, confidence, and recency. When elastic protection is enabled, a second state variable—`protection`—is aligned gradually to that importance. Learning updates are then multiplied by `1 - protection`, so heavily protected concepts become less plastic even before they are permanently locked. This creates a continuum between fully plastic new concepts and stable old ones.

Two design features make this especially relevant to the precision story.

First, routing becomes precision-aware. When precision is high, indicating novelty or uncertainty, the pool biases routing toward less-protected or still-uncommitted concepts. Novel evidence is therefore more likely to recruit fresh representational resources. When precision is low, indicating familiarity, routing favors already protected concepts. Familiar evidence is therefore more likely to reinforce stable memory traces than to create redundant new ones.

Second, concept replay closes the loop. The pool stores one exemplar per committed CCC and periodically replays these exemplars through a `replay_boost()` update. Replay nudges drifted concept directions and feedback weights back toward their historical anchor. This matters because continual forgetting in Bio-ARN is driven less by catastrophic global weight sharing than by gradual rotation of committed concept directions during `learn_slow()`. Replay is a cheap, local repair mechanism.

Elastic protection is deliberately paired with **hard locking**. Once importance crosses a threshold, a CCC becomes read-only for concept and feedback updates. The combined effect is a two-regime stability policy:

- **soft protection** from precision weighting and elastic routing while a concept is still maturing,
- **hard protection** from locking once the concept has proven durable.

That combination is what allows the Sprint I stack to reduce class-split forgetting further than precision weighting alone.

## 4. Experiments

The results in this section summarize the focused predictive-processing comparison within the broader Bio-ARN program. Unless otherwise noted, the numbers come from the companion paper and sprint benchmarks already incorporated into the repository narrative [6]. The goal here is not to claim state-of-the-art image classification. It is to measure whether the precision mechanism resolves the Hebb-versus-prediction conflict without breaking Bio-ARN's broader efficiency and robustness profile.

### 4.1 Settling vs. Error Gating vs. Precision Weighting

Table 1 captures the main comparison.

| Metric | Settling | Error Gating | Precision-Weighted | Combined (Sprint I) |
|---|---:|---:|---:|---:|
| CIFAR-10 Accuracy | 11.8% | 30.0% | 30.0% | 33.0% |
| OOD AUROC | — | 0.778 | 1.000 | 1.000 |
| Forgetting (class-split) | — | 34.7% | 20.7% | 13.7% |
| Energy (vs A100) | — | — | 278x | 278x |

Two conclusions are immediate. First, iterative settling is decisively harmful in this architecture. It does not merely fail to help; it actively destroys discriminative performance. Second, precision weighting does **not** buy its value by sacrificing short-run accuracy. The precision-weighted model matches the 30.0% error-gating baseline on CIFAR-10 while strongly improving robustness and retention metrics.

This pattern matters more than a raw top-1 increase would. If precision weighting had improved forgetting only by lowering overall plasticity and hurting accuracy, it would be a trivial stability trade. That is not what happened. Accuracy remains flat relative to error gating, indicating that the gate is selectively attenuating harmful updates rather than globally freezing the learner.

The combined Sprint I stack then adds a modest accuracy lift to 33.0%. Importantly, this is not evidence that precision alone solved Bio-ARN's visual-learning challenge. It shows that once predictive processing is made non-destructive, it becomes composable with other mechanisms—elastic protection, concept replay, and lateral context modeling—that together improve the architecture's overall operating point.

### 4.2 Continual Learning

Continual learning is where precision weighting shows its clearest value. The class-split CIFAR-10 benchmark asks the system to learn five sequential two-class tasks without revisiting all previous data. This benchmark is particularly punishing for local concept systems because new classes compete for the same concept slots and can rotate already-committed prototypes.

Plain error gating reduces the most severe failure caused by settling, but it still leaves mean forgetting at 34.7%. In other words, removing destructive settling is necessary but not sufficient. The local learner is still too eager to rewrite previously useful CCCs when later tasks arrive.

Precision weighting cuts forgetting to 20.7%. The interpretation is straightforward: when recent winner usage is concentrated and familiar, the pool lowers the effective learning rate and protects previously consolidated concept directions; when winner usage becomes diffuse, the system raises plasticity and allows more aggressive adaptation. This is precisely the kind of context-sensitive stability-plasticity control that Hebbian learning lacks on its own.

The combined Sprint I stack reduces forgetting further to 13.7% when elastic protection and replay are layered on top of the precision gate. Although the prompt benchmark summary does not provide a single canonical backward-transfer number for every variant, the forgetting reduction implies a materially less negative BWT profile: older tasks lose less ground after later tasks are learned. Qualitatively, that is the right signature for a protection mechanism. The system is not necessarily mastering each new task faster; it is retaining prior concept ownership more faithfully.

This result also clarifies the division of labor among Bio-ARN's retention mechanisms.

- **Error gating** prevents prediction from flattening the feature representation.
- **Precision weighting** decides when residuals deserve strong Hebbian influence.
- **Elastic protection** reduces drift in already valuable concepts.
- **Hard locking** freezes mature detectors once their importance crosses threshold.
- **Replay** periodically restores historical anchors.

In dense continual-learning literature, these functions are often collapsed into a single regularizer. Bio-ARN separates them into sparse architectural operations. That separation is useful for neuromorphic deployment because it keeps each control loop local and event-driven.

### 4.3 OOD Detection

Precision weighting also improves open-set behavior. The baseline error-gating path reaches 0.778 OOD AUROC, while the precision-weighted and combined configurations reach 1.000. This gain is not an external calibration trick. It emerges from the architecture's own internal uncertainty signals.

OOD inputs create a distinctive signature in Bio-ARN. They tend to produce one or more of the following:

1. diffuse or unstable winner usage across recent history,
2. higher lateral mismatch because co-firing patterns are not contextually coherent,
3. weak resonance between fired CCCs and their feedback predictions,
4. greater abstention pressure because no committed concept cleanly matches.

All four effects push the precision controller toward a regime in which the system treats the sample as informative but untrusted. That makes it easier for the architecture to abstain, recruit cautiously, or isolate the event as novel rather than forcing it into an overconfident familiar category.

This is an important conceptual bridge between predictive coding and honest uncertainty. In many dense systems, OOD detection is a downstream score attached to a classifier that already committed. In Bio-ARN, uncertainty is structural. The same sparse winner dynamics used for learning also generate the precision signal used for uncertainty-aware control. That unification is one reason the architecture can improve OOD behavior without a separate confidence model.

### 4.4 Energy Efficiency

Bio-ARN's energy story is not a side note to the precision mechanism; it is part of why this solution is attractive. The energy report projects 179.65 µJ per inference on Loihi 2 versus 50.01 mJ for a matched transformer baseline on an A100, a 278x advantage [5, 6]. Precision weighting preserves that claim because its computational overhead is tiny.

The reason is architectural scale. Precision is computed once per pool, not once per synapse. The additional operations are: update a bounded winner-history buffer, count recent winners, compute a normalized entropy, and apply a sigmoid. Compared with CCC activation, sparse distributed memory, and hierarchy processing, this is effectively free. The mechanism therefore offers a rare combination of benefits: it improves retention and OOD behavior while fitting the sparse-event computational style required by neuromorphic hardware.

The energy report provides a second relevant detail: predictive processing already zeroes 47.3% of hierarchy activity through predictive suppression in the profiled path [6]. Precision weighting complements that suppression rather than replacing it. Predictable structure still consumes less computation, but now the residual learning signal is scaled intelligently before it can destabilize memory. In that sense the precision mechanism makes predictive coding not only more accurate for Hebbian learners, but also more *worth deploying* on low-power substrates.

## 5. Discussion

The central lesson of this paper is simple: **do not ask predictive coding to do the wrong job**. In a sparse Hebbian concept architecture, iterative settling tries to use prediction as a representational corrector. That is the source of conflict. The architecture needs prediction primarily as a **plasticity governor**.

Precision weighting provides that governor. By estimating uncertainty from pool-level concept usage and mapping it to a bounded learning gain, Bio-ARN lets predictive processing regulate when local Hebbian coincidence should matter. Predictable, familiar contexts are protected. Novel, diffuse contexts remain plastic. The discriminative feedforward pattern survives intact. Prediction becomes a guide instead of a bulldozer.

This also reframes the relationship between predictive coding and Hebbian learning more broadly. The two are often presented as rivals because predictive coding seems to privilege top-down generative consistency while Hebbian learning privileges bottom-up co-activity. The Bio-ARN results suggest a better synthesis: Hebbian learning should own the formation of sparse concepts, while predictive processing should own the allocation of plasticity around those concepts. In other words, Hebb learns *what* co-occurs; precision decides *when* that co-occurrence deserves memory.

The connection to Frank et al. strengthens the biological story. Their data imply that the hippocampus broadcasts an anticipatory uncertainty signal before sensory evidence arrives [1]. Bio-ARN does not simulate ripples as oscillatory events, but it preserves their functional role. Pool entropy stands in for expected uncertainty, lateral and hierarchy mismatch supply event-specific confirmation, and the resulting precision state tunes local learning. This is a plausible neuromorphic abstraction of a hippocampal-cortical control loop.

Several limitations remain.

First, the precision mechanism fixes a real architectural problem without solving Bio-ARN's entire accuracy ceiling. The best combined CIFAR-10 result is still only 33.0%. This is a meaningful internal improvement, not a competitive vision benchmark.

Second, pool entropy is a coarse proxy for uncertainty. It captures concept dispersion well, but it does not replace full Bayesian uncertainty estimation or task-structured epistemic modeling. The current advantage of this proxy is practicality: it is cheap, local, and already useful.

Third, continual-learning improvements still depend on capacity management. Precision weighting can protect stable concepts, but if the pool saturates too early, even a well-gated learner may run out of representational room. Elastic routing, growth, replay, and eviction are therefore not optional accessories; they are part of the same stability-plasticity solution.

Fourth, the energy claims remain projected for neuromorphic deployment. The precision controller is cheap by inspection and by software profile, but a full silicon validation remains future work.

These limitations suggest a clear research agenda. The most promising next step is **multi-modal precision fusion**. Bio-ARN already supports more than vision. A natural extension is to combine uncertainty signals from visual, auditory, and possibly language pools into a shared precision prior so that one modality can raise or lower plasticity in another. For example, an ambiguous visual scene paired with a reliable auditory cue should not look as uncertain as vision alone. The same control logic could also be routed through the Global Neuronal Workspace so that broadcast-level consensus influences whether new concept recruitment is even allowed.

A second direction is to learn richer precision sources while preserving locality. The current implementation blends entropy with lateral and hierarchy error. Future versions could incorporate task context, replay disagreement, or novelty rewards into the same bounded gate. The important constraint is architectural: these additions should still modulate local learning rather than reintroducing destructive global settling.

## 6. Conclusion

Precision weighting resolves the apparent conflict between predictive coding and Hebbian learning by changing what prediction is allowed to do. In Bio-ARN, predictive settling fought the feature selectivity that sparse concept cells need and collapsed CIFAR-10 accuracy from 30.0% to 11.8%. Single-pass error gating fixed the representation problem, and hippocampal-ripple-inspired precision weighting turned that fix into a principled learning policy. Pool entropy estimates uncertainty, a sigmoid maps uncertainty to a precision weight, and local Hebbian updates are amplified only when the context is genuinely informative.

The result is not a miracle accuracy jump. It is more important than that. Predictive processing stops being anti-Hebbian. It becomes the mechanism that tells Hebbian learning when to remain stable and when to adapt. Precision weighting preserves 30.0% CIFAR-10 accuracy, improves class-split forgetting from 34.7% to 20.7%, lifts OOD AUROC from 0.778 to 1.000, composes with elastic protection to reach 33.0% accuracy and 13.7% forgetting, and does so while keeping Bio-ARN on its 278x projected neuromorphic efficiency path.

The design lesson is concise: **do not fight Hebbian features—guide them with precision.**

## References

1. Frank, D., Moratti, S., Hellerstedt, R., Sarnthein, J., Li, N., Horn, A., Imbach, L., Stieglitz, L., Gil-Nagel, A., Toledano, R., Friston, K. J., & Strange, B. A. (2026). *Human hippocampal ripples tune cortical responses based on predicted uncertainty*. Nature Neuroscience. https://doi.org/10.1038/s41593-026-02345-6
2. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience, 2*(1), 79-87.
3. Grossberg, S. (1976). Adaptive pattern classification and universal recoding: I. Parallel development and coding of neural feature detectors. *Biological Cybernetics, 23*, 121-134.
4. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B: Biological Sciences, 360*(1456), 815-836.
5. Davies, M., Srinivasa, N., Lin, T.-H., Chinya, G., Cao, Y., Choday, S. H., Dimou, G., Joshi, P., Imam, N., Jain, S., Liao, Y., Lin, C.-K., Lines, A., Liu, R., Mathaikutty, D., McCoy, S., Paul, A., Tse, J., Venkataramanan, G., ... Weng, Y.-H. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro, 38*(1), 82-99.
6. Liebs, B., & the Bio-ARN team. (2026). *Bio-ARN: A Brain-Inspired Architecture for Energy-Efficient Multi-Modal Learning on Neuromorphic Hardware* (companion manuscript; see `docs/paper_draft.md`).
