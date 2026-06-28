# Squad Decisions

## Active Decisions

### 2026-06-27: Formal benchmark suite now covers MNIST, Fashion-MNIST, and CIFAR-10
**By:** Neo
**What:** Added `experiments/formal_benchmarks.py`, a reproducible multi-dataset benchmark suite reporting accuracy, OOD AUROC, energy, committed concepts, locked concepts, fire rate, and wall-clock training time across MNIST, Fashion-MNIST, and CIFAR-10, plus direct Conv CCC CIFAR runs.
**Why:** Establishes paper-quality benchmarking infrastructure even though this first run still used synthetic MNIST-family data and leaves CIFAR-10 at chance level pending tuning.
**References:** `experiments/formal_benchmarks.py`, `tests/test_cifar_training.py`, `tests/test_conv_ccc.py`, `tests/test_mnist_accuracy.py`

### 2026-06-27: Paper draft now reflects Sprint E retention gains and predictive framing
**By:** Morpheus
**What:** Updated `docs/paper_draft.md` to add concept locking, convolutional CCCs, precision-weighted predictive processing, and the Sprint E result that conv plus locking reduced mean forgetting from 34.7 percent to 20.7 percent.
**Why:** Keeps the paper narrative aligned with the latest continual-learning evidence without overstating current accuracy.
**References:** `docs/paper_draft.md`, `tests/test_ccc.py`, `tests/test_conv_ccc.py`

### 2026-06-27: Paper architecture figures are now reproducible from source
**By:** Trinity
**What:** Added `docs/figures/architecture_diagrams.py` and a terminal-readable companion so all five paper figures can be regenerated at publication quality.
**Why:** Ensures the paper's architecture visuals are versioned, reproducible, and consistent across sections.
**References:** `docs/figures/architecture_diagrams.py`, `docs/figures/architecture_diagrams.txt`, `docs/figures/figure1_full_bioarn_pipeline.png`, `docs/figures/figure2_ccc_internal_architecture.png`, `docs/figures/figure3_precision_weighted_predictive_processing.png`, `docs/figures/figure4_concept_locking_lifecycle.png`, `docs/figures/figure5_energy_comparison.png`

### 2026-06-27: Bio-ARN is packaged for editable pip install as version 2.0.0
**By:** Trinity
**What:** Updated packaging metadata, added `bioarn/__version__.py`, expanded top-level exports, and used lazy imports so `pip install -e .` exposes the Bio-ARN 2.0 API cleanly.
**Why:** Makes the repository installable and import-stable for publication, demo, and downstream use.
**References:** `pyproject.toml`, `bioarn/__init__.py`, `bioarn/__version__.py`, `bioarn/predictive/__init__.py`, `README.md`

### 2026-06-27: Production Gradio demo now covers classification, online learning, and continual learning
**By:** Tank
**What:** Rebuilt `demo/app.py` as a four-tab Gradio experience with cached MNIST training, CCC and OOD diagnostics, session-scoped online learning, and a two-task continual-learning walkthrough.
**Why:** Gives the project a deployable conference demo surface that stays usable even when dataset downloads fail.
**References:** `demo/app.py`, `demo/README.md`, `demo/requirements.txt`, `tests/test_demo.py`

### 2026-06-27: Sprint E combined benchmark shows conv locking lowers forgetting but not balanced CIFAR retention
**By:** Switch
**What:** On the modest CIFAR-10 and Split-CIFAR-10 benchmark, standard CCC variants tied at 13.0 percent accuracy, conv variants trailed slightly, and `conv_locked` and `conv_all` reduced mean forgetting to 20.7 percent while concentrating retention in the final task.
**Why:** Records that concept locking and convolution currently help continual-retention metrics more than short-run top-1 accuracy, and that precision tracked pool entropy instead of decaying with familiarity.
**References:** `experiments/sprint_e_benchmark.py`, `docs/paper_draft.md`

### 2026-06-27: Phase-gated maturation scheduling stages optional modules during training
**By:** Neo
**What:** Added a maturation schedule so hierarchy and curiosity dominate early training, workspace support turns on in phase 2, and predictive plus feedback modules activate in phase 3 with lower learning rates.
**Why:** Keeps optional modules architecturally present without forcing fragile all-at-once activation from sample 1.
**References:** `bioarn/training/maturation.py`, `bioarn/training/vision_training.py`, `experiments/combined_config_sweep.py`, `experiments/real_cifar_comparison.py`

### 2026-06-27: Sprint D combined benchmark winner is best-d1-multi
**By:** Tank
**What:** `experiments/sprint_d_benchmark.py` identified `best-d1-multi` as the strongest Sprint D tier-2 configuration at 28.0 percent accuracy and 1.000 OOD AUROC on 500 real CIFAR-10 training samples, scaling to 30.5 percent at 1k and 2k samples.
**Why:** Establishes the best measured Sprint D recipe even though it still misses the 35 percent and 40 percent targets.
**References:** `experiments/sprint_d_benchmark.py`, `experiments/real_cifar_comparison.py`

### 2026-06-27: Sprint D continual-learning retest shows only marginal CIFAR gains and worse MNIST forgetting
**By:** Switch
**What:** Retesting Sprint D settings barely improved Split-CIFAR-10 forgetting metrics and regressed both Split-MNIST and Permuted-MNIST retention.
**Why:** Suggests prediction-error gating helps natural-image splits only slightly, while curiosity plus curriculum plus GNW still over-recruits CCCs and fails to protect downstream concept ownership.
**References:** `experiments/continual_learning.py`, `experiments/continual_learning_mnist.py`, `docs/paper_draft.md`

### 2026-06-27: GNW consensus classification now gates both prediction and learning
**By:** Trinity
**What:** Multi-CCC classification now routes through GNW consensus, exposes normalized broadcast strength, and uses that signal as a trainer-side learning-rate gate.
**Why:** Lets strong consensus protect stable knowledge while uncertain consensus increases plasticity, and keeps the consensus gate composable with replay and curiosity logic.
**References:** `bioarn/workspace`, `bioarn/training/vision_training.py`, `Brad Liebs request 2026-06-27`

### 2026-06-27: Concept locking freezes stable CCCs to protect prior classes
**By:** Neo
**What:** Added `lock_threshold` to `CCCConfig`, a persistent `locked` buffer on CCCs, pool auto-locking, and locked-cell stats so high-importance concept cells become read-only detectors.
**Why:** Prevents committed concept directions from drifting when new classes arrive, reducing catastrophic forgetting without removing old detectors from inference.
**References:** `bioarn/core/ccc.py`, `bioarn/config.py`, `tests/test_ccc.py`, `tests/test_integration_improvements.py`

### 2026-06-27: Visual predictive processing now defaults to prediction-error-gated Hebbian learning
**By:** Tank
**What:** The default visual predictive path now modulates Hebbian learning rates with single-pass prediction error instead of iterative settling, while settling remains available as an opt-in mode.
**Why:** Preserves discriminative bottom-up features while still learning local predictive weights and exposing novelty signals downstream.
**References:** `bioarn/predictive/error_gating.py`, `bioarn/hierarchy/visual_hierarchy.py`, `bioarn/system.py`, `experiments/real_cifar_comparison.py`

### 2026-06-27: Task A now combines novelty-scaled learning, curriculum ordering, and boundary-aware replay
**By:** Trinity
**What:** Added trainer-side preview scoring, a curriculum scheduler, per-sample CCC learning-rate multipliers, and extra replay pressure for boundary samples.
**Why:** Composes multiple curiosity signals without rewriting unrelated online-learning flows and gives the trainer a controllable way to push plasticity where uncertainty is highest.
**References:** `bioarn/training/vision_training.py`, `TASK A Multi-Signal Curiosity with Difficulty Curriculum`

### 2026-06-27: Shared F1 encoders stay frozen while per-task adapters absorb drift
**By:** Neo
**What:** Chose a continual-learning design that freezes shared F1 encoders after a threshold and routes task-specific residual adapters to committed CCCs by adapter id.
**Why:** Targets feature-drift forgetting while preserving local learning and allowing older CCCs to keep seeing the feature geometry they were recruited on.
**References:** `bioarn/core/ccc.py`, `bioarn/core/consolidation.py`, `bioarn/scaling.py`, `bioarn/training/vision_training.py`, `experiments/continual_learning.py`

### 2026-06-27: Combined CIFAR sweep best config is baseline-2k at 33.0 percent
**By:** Tank
**What:** Real CIFAR-10 sweep results show the strongest measured combined configuration is baseline-2k with hierarchy on and workspace, curiosity, predictive, STDP, and feedback all off, reaching 33.0 percent accuracy and 1.000 OOD AUROC on 200 test and 200 OOD samples.
**Why:** Records that several added mechanisms were neutral or negative in this sweep, so the project should treat the simpler hierarchy path as the current best measured combined recipe.
**References:** `experiments/combined_config_sweep.py`, `experiments/real_cifar_comparison.py`

### 2026-06-27: Precision-weighted predictive processing scales CCC plasticity by uncertainty
**By:** Tank
**What:** Added entropy-based pool precision estimation, a precision gate for CCC and batched CCC pools, `PrecisionConfig`, and trainer wiring that multiplies the existing learning-rate modifier by a pool uncertainty signal.
**Why:** Makes prediction-error learning selective so novel, high-entropy states learn fast while familiar, low-entropy states protect stable representations.
**References:** `bioarn/predictive/precision_weighting.py`, `bioarn/config.py`, `bioarn/core/ccc.py`, `bioarn/scaling.py`, `bioarn/training/vision_training.py`

### 2026-06-27: Convolutional CCCs preserve spatial structure before concept matching
**By:** Morpheus
**What:** Added `ConvCCCConfig`, `bioarn/core/conv_ccc.py`, a shared convolutional F1 path, convolutional concept clusters, a convolutional pool, and `VisionTrainer` support behind `use_conv_ccc`.
**Why:** Replaces the flat 3072-dimensional CCC path for images with a spatially structured local-learning path that better fits vision inputs.
**References:** `bioarn/core/conv_ccc.py`, `bioarn/config.py`, `bioarn/training/vision_training.py`, `tests/test_conv_ccc.py`

### 2026-06-26: Hierarchy feedback uses previous-sample top-down multiplicative gating
**By:** Neo
**What:** Added opt-in hierarchy feedback projections from higher to lower visual layers and applies the previous sample's top-down signal as multiplicative gating on the next sample.
**Why:** Introduces local top-down modulation without breaking default feedforward behavior.
**References:** `bioarn/hierarchy/config.py`, `bioarn/hierarchy/visual_hierarchy.py`, `experiments/real_cifar_comparison.py`

### 2026-06-26: VisualHierarchy gained opt-in predictive settling without changing default classification
**By:** Neo
**What:** Added predictive refinement that settles layer summaries in concept space and exposes predictive traces, states, and errors while keeping the raw hierarchy classifier path intact.
**Why:** Safely wires predictive coding into the training path without regressing the baseline label path when predictive refinement is enabled.
**References:** `bioarn/hierarchy/config.py`, `bioarn/hierarchy/visual_hierarchy.py`, `bioarn/predictive/hierarchy.py`, `bioarn/config.py`, `tests/test_hierarchy.py`

### 2026-06-26: Predictive tuning emphasized deeper settling, hierarchical precision, and predictive label signatures
**By:** Neo
**What:** Tuned the predictive hierarchy to weight lower-level sensory precision more strongly and feed predictive states and errors into hierarchy label signatures when predictive refinement is active.
**Why:** Shifts evaluation toward the settled predictive representation instead of only the raw feedforward trace.
**References:** `bioarn/hierarchy/visual_hierarchy.py`, `experiments/real_cifar_comparison.py`

### 2026-06-26: Continual learning now favors additive CCC growth plus Hebbian consolidation
**By:** Neo
**What:** When a pool is full and committed CCCs do not match a novel sample, the system now allocates bounded fresh CCC slots and scales slow and feedback learning by CCC importance.
**Why:** Preserves old concepts additively and protects frequently reused cells without introducing backpropagation.
**References:** `bioarn/core/ccc.py`, `bioarn/core/consolidation.py`, `bioarn/scaling.py`, `bioarn/training/vision_training.py`

### 2026-06-26: CCC Hebbian updates can be modulated by opt-in trace-based STDP
**By:** Neo
**What:** Added an opt-in STDP rule with decaying pre and post traces and uses its signal to modulate CCC feedback and concept refinement updates.
**Why:** Brings a more biologically grounded local learning rule into CCC plasticity while remaining optional and backward compatible.
**References:** `bioarn/core/ccc.py`, `bioarn/core/stdp.py`, `bioarn/scaling.py`, `bioarn/config.py`

### 2026-06-26: Continual-learning benchmark exposed severe forgetting and early IT saturation
**By:** Switch
**What:** Split-CIFAR-10 and class-incremental CIFAR-10 runs showed 28.0 percent final average accuracy, negative backward transfer, meaningful forgetting, and IT-layer capacity saturating by task 2.
**Why:** Reframes continual learning as a capacity and recruitment bottleneck rather than evidence of graceful slot reuse.
**References:** `experiments/continual_learning.py`, `experiments/real_cifar_comparison.py`

### 2026-06-26: Paper positioning emphasizes efficiency, robustness, and biological plausibility over raw top-1 accuracy
**By:** Switch
**What:** The Bio-ARN 2.0 paper draft now foregrounds local Hebbian learning, OOD robustness, Loihi 2 export, and projected energy advantages while being explicit about modest real-CIFAR accuracy and current continual-learning limits.
**Why:** Aligns the narrative with the project's strongest evidence instead of overstating benchmark dominance.
**References:** `docs/paper_draft.md`, `README.md`, `docs/architecture.md`, `experiments/energy_report_results.md`

### 2026-06-26: Loihi 2 export round-trip preserved weight structure in portable fallback validation
**By:** Tank
**What:** Added a portable validation experiment showing exported CCC weights round-trip exactly and produce closely matched fallback LIF firing dynamics and near-preserved toy-task accuracy.
**Why:** Provides empirical support that the Loihi 2 export path preserves useful structure even without a hard Lava dependency.
**References:** `experiments/lava_validation.py`, `logs/lava_validation_summary.json`, `bioarn/export/loihi2.py`

### 2026-06-26: Scaling sweeps show early saturation in pool size, expert count, and data efficiency
**By:** Tank
**What:** CIFAR scaling-law runs found flat or weak gains beyond small pool sizes, diminishing returns past three experts, and only modest accuracy improvement as sample counts rise.
**Why:** Indicates the current architecture hits capacity and efficiency ceilings early, helping prioritize future tuning work.
**References:** `experiments/scaling_laws.py`, `logs/scaling_laws_summary.json`

### 2026-06-26: Curiosity-driven vision training uses novelty-triggered bounded replay
**By:** Trinity
**What:** `VisionTrainConfig.curiosity_weight` now enables trainer-side novelty estimation, curiosity traces, and one bounded replay for samples that remain surprising after an online step.
**Why:** Makes curiosity measurable and opt-in at the training loop level without changing default system behavior.
**References:** `bioarn/training/vision_training.py`, `bioarn/reward/novelty.py`, `tests/test_cifar_training.py`

### 2026-06-26: Opt-in GNW workspace participates in recognition, training, and evaluation
**By:** Trinity
**What:** Added explicit workspace config fields and a lightweight GNW-aware path so active CCC hypotheses can be filtered through broadcast and context before final voting in both inference and CIFAR evaluation.
**Why:** Preserves the legacy path when workspace is absent while improving OOD separation when it is enabled.
**References:** `bioarn/config.py`, `bioarn/system.py`, `bioarn/training/vision_training.py`, `experiments/real_cifar_comparison.py`, `tests/test_integration_improvements.py`

### 2026-06-26: Split-MNIST exposes much worse forgetting than Permuted-MNIST
**By:** Trinity
**What:** MNIST continual-learning benchmarks showed severe semantic-task interference on Split-MNIST but comparatively mild degradation on Permuted-MNIST.
**Why:** Suggests the main failure mode is class interference across tasks rather than generic inability to retain arbitrary pixel permutations.
**References:** `experiments/continual_learning.py`, `experiments/continual_learning_mnist.py`

### 2026-06-26: VisualHierarchy spatial attention and adaptive competition raised real CIFAR accuracy to 30.0 percent
**By:** Neo
**Date:** 2026-06-26
**What:** VisualHierarchy now uses spatial attention, stronger competitive inhibition, and higher adaptive IT capacity, lifting real CIFAR-10 hierarchy accuracy from 26.4 percent to 30.0 percent while preserving 1.000 OOD AUROC.
**Why:** The hierarchy needed better locality, stronger winner selection, and more adaptive capacity to break through the earlier ceiling.
**References:** `bioarn/hierarchy/visual_hierarchy.py`, `bioarn/hierarchy/config.py`, `bioarn/hierarchy/attention.py`, `bioarn/hierarchy/competition.py`, `experiments/real_cifar_comparison.py`, `commit 627e3e3`

### 2026-06-26: Complementary hierarchy and ensemble routing now beats hierarchy alone on real CIFAR-10
**By:** Trinity
**Date:** 2026-06-26
**What:** The shared-CCC multimodal demo was validated end to end and the old abstention-only combination logic was replaced with calibrated complementary routing, bringing the combined configuration to 30.0 percent accuracy and 1.000 OOD AUROC.
**Why:** The previous both path blocked useful ensemble contributions on hard samples; calibrated delegation keeps hierarchy as the default expert while still allowing strong ensemble overrides.
**References:** `experiments/real_cifar_comparison.py`, `experiments/multimodal_demo.py`, `tests/test_multimodal_training.py`, `tests/test_multimodal_fusion.py`, `tests/test_ensemble.py`, `tests/test_integration_improvements.py`, `commit faaa60f`

### 2026-06-26: Portable Loihi 2 export path added for CCC pools and visual hierarchies
**By:** Morpheus
**Date:** 2026-06-26
**What:** Added a portable `bioarn.export` package that serializes trained CCC pools and `VisualHierarchy` instances into Loihi 2 graph JSON plus an NIR-compatible sidecar without requiring Lava as a hard dependency.
**Why:** Gives the project a neuromorphic deployment path using explicit LIF populations, synaptic projections, and round-trippable intermediate export documents.
**References:** `bioarn/export`, `tests/test_export.py`, `commit 0b4a32a`

### 2026-06-26: Energy benchmark refresh still projects a 278x inference-efficiency advantage
**By:** Tank
**Date:** 2026-06-26
**What:** `experiments/energy_report.py` still projects 179.65 microjoules on Loihi 2 versus 50.01 millijoules on A100 for the matched transformer baseline, with the online-training gap still around 8050x.
**Why:** Confirms the headline energy claim remains current after recent architecture and demo changes.
**References:** `experiments/energy_report.py`, `experiments/energy_report_results.md`, `experiments/energy_report_data.json`

### 2026-06-25: Real CIFAR-10 ensemble OOD tuning beats the baseline
**By:** Morpheus
**Date:** 2026-06-25
**What:** `bioarn/ensemble/voting.py` now makes positive confidence more conservative and normalizes winning vote mass against expert support capacity, improving real CIFAR-10 ensemble OOD AUROC from 0.743 to 0.861 while nudging accuracy to 25.8 percent.
**Why:** Sparse low-evidence agreement was previously over-rewarded and needed calibration.
**References:** `bioarn/ensemble/voting.py`, `experiments/real_cifar_comparison.py`, `commit afec6d1`

### 2026-06-25: Text generation v3 improved real-word rate with stronger word-level control
**By:** Tank
**Date:** 2026-06-25
**What:** Word-level generation gained larger context tracking, frequency-aware ranking, stronger confidence blending, direct emission, and safer truncation, improving dual-model real-word rate from 53.6 percent to 85.7 percent while keeping repetition low.
**Why:** Stronger word-level control materially improved output quality while preserving diversity.
**References:** `bioarn/language/dual_processor.py`, `bioarn/language/word_level.py`, `bioarn/training/text_training.py`, `experiments/text_gen_v3.py`, `commit c00e8f5`

### 2026-06-25: Sprint work was recovered and committed after the second crash
**By:** Squad Coordinator
**Date:** 2026-06-25
**What:** The Let do them all sprint survived in the working tree and was preserved in six logical commits after validation showed the key suites still passed.
**Why:** Keeps completed sprint work from being lost to machine instability.

### 2026-06-25: Post-fix comparison runs confirmed ensemble AUROC recovery on synthetic data
**By:** Trinity
**Date:** 2026-06-25
**What:** Synthetic CIFAR-10 again reports ensemble OOD AUROC 1.000 after the inversion fix, while real-CIFAR runs showed hierarchy and both at 26.4 percent accuracy and ensemble at 23.4 percent with 0.743 AUROC before later tuning.
**Why:** Verified that the AUROC inversion bug was fixed while documenting remaining real-data calibration work.
**References:** `experiments/improvement_comparison.py`, `experiments/real_cifar_comparison.py`

### 2026-06-25: Text generation v3 improved prompt coherence while lowering repetition
**By:** Morpheus
**Date:** 2026-06-25
**What:** `experiments/text_gen_v3.py` produced prompt-aware phrases instead of collapsed char-only output and reduced repetition from 0.93 to 0.31.
**Why:** Confirms the dual char-plus-word approach is directionally stronger while leaving coherence and metric trade-offs for follow-up.
**References:** `experiments/text_gen_v3.py`, `tests/test_word_level.py`, `tests/test_sequence_memory.py`, `bioarn/language/dual_processor.py`, `bioarn/language/word_level.py`, `bioarn/training/text_training.py`

### 2026-06-25: Simplified GitHub Actions CI workflow added for push and pull request validation
**By:** Tank
**Date:** 2026-06-25
**What:** Added `.github/workflows/ci.yml` with ubuntu-latest, a Python 3.11 and 3.13 matrix, editable install of the dev and demo extras, and pytest configured to skip slow and performance tests.
**Why:** Establishes automated validation while isolating long-running suites and surfacing demo-path defects separately.
**References:** `.github/workflows/ci.yml`, `tests/test_demo.py`, `bioarn/training/text_training.py`, `demo/models.py`, `commit 10ef1bc`

### 2026-06-25: EnsembleTrainer and VisionTrainer gained multi-pass and interleave support
**By:** Morpheus
**Date:** 2026-06-25
**What:** `EnsembleTrainer` now trains experts on augmented views and `VisionTrainer` accepts `num_passes` and `interleave_classes`, improving convergence on synthetic and sorted-stream experiments.
**Why:** Interleaving prevents early class monopoly of pool slots in Hebbian learners.

### 2026-06-25: Hierarchy and ensemble are now wired into BioARNCore via lazy imports
**By:** Neo
**Date:** 2026-06-25
**What:** `VisualHierarchy` and `EnsemblePool` are optionally accessible through the standard BioARNCore API with backward-compatible config lookups and lazy imports to avoid circular dependencies.
**Why:** These modules were functional but orphaned from the default training path.

### 2026-06-25: Integration test coverage now spans hierarchy, ensemble, and combined flows
**By:** Switch
**Date:** 2026-06-25
**What:** Added 31 tests covering BioARNCore plus hierarchy, BioARNCore plus ensemble, combined operation, EnsembleTrainer behavior, and regressions.
**Why:** Validates the new integration wiring without regressions.

### 2026-06-25: BioARNConfig now carries optional hierarchy and ensemble fields
**By:** Trinity
**Date:** 2026-06-25
**What:** `BioARNConfig` gained optional hierarchy and ensemble configuration fields with `TYPE_CHECKING` guards to avoid circular imports.
**Why:** Makes the new modules activatable through configuration instead of ad hoc code changes.

### 2026-06-25: Shared-CCC multimodal training now has a canonical trainer
**By:** Trinity
**Date:** 2026-06-25
**What:** Added `MultimodalTrainer` that alternates vision and text on a shared CCC pool and binds representations through `MultimodalFusion`.
**Why:** Exercises the multimodal stack through a standard training API instead of one-off scripts.

### 2026-06-25: Ensemble voting confidence inversion that broke OOD AUROC was fixed
**By:** Neo
**Date:** 2026-06-25
**What:** Ensemble confidence scoring now uses a positive-only mapping from 0.5 to 1.0 and reliability weighting so OOD samples no longer receive higher confidence than in-distribution samples.
**Why:** Corrected a critical metric bug that made the ensemble appear worse than random on OOD detection.

### 2026-06-25: Infrastructure updates added Gradio support and slow-test markers
**By:** Tank
**Date:** 2026-06-25
**What:** Added `gradio` to the demo extra and marked long text-generation tests as slow so CI can exclude them.
**Why:** Unblocks demo test collection and keeps CPU-heavy suites out of normal validation.

## Archived Decisions

### 2026-06-26: CIFAR-10 best known config remained the 2k hierarchy control
**By:** Trinity
**Status:** SUPERSEDED by the later 30.0 percent hierarchy and both results
**What:** Before the spatial-attention and complementary-routing upgrades, `hierarchy-control` at 2000 training samples was the best-known real-CIFAR configuration at 26.4 percent accuracy.
**References:** `experiments/cifar_tuning.py`, `experiments/cifar_scaling.py`, `commit eb2ea53`

### 2026-06-26: Demo UI now uses capability-focused tabs with graceful optional Gradio handling
**By:** Trinity
**Status:** SUPERSEDED by the 2026-06-27 production Gradio demo decision
**What:** The demo kept the existing digit, text, cross-modal, live-learning, and energy flows while adding dedicated OOD, multimodal binding, CCC recruitment, and architecture views.
**References:** `demo/app.py`, `demo/models.py`, `demo/visualizations.py`, `tests/test_demo.py`

### 2026-06-24: Recovery commit preserved pre-crash work
**By:** Squad Coordinator
**What:** All in-progress work at the time of the first crash was committed as recovery commit `d99976c`.

### 2026-06-24: Team hired under the Matrix-style roster
**By:** Squad Coordinator
**What:** Neo, Tank, Morpheus, Trinity, Switch, Scribe, Ralph, and Rai were established as the operating team.

### 2026-06-25: Architecture assessment found hierarchy and ensemble orphaned from training
**By:** Neo
**Status:** RESOLVED
**What:** The system architecture was healthy overall, but hierarchy, ensemble, and multimodal modules were not wired into the main training path at the time.

### 2026-06-25: Test suite health was 411 passing with dependency gaps
**By:** Switch
**Status:** RESOLVED
**What:** The baseline suite was green, but PyYAML and Gradio environment issues still blocked portions of the full workflow.

### 2026-06-25: Integration complete and 13 experiments remained intact
**By:** Trinity
**Status:** RESOLVED
**What:** The existing experiments and integration wiring survived the recovery sprint intact.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
