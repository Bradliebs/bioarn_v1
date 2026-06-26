# Squad Decisions

## Active Decisions

### 2026-06-26: VisualHierarchy spatial attention and adaptive competition raise real CIFAR-10 hierarchy accuracy to 30.0%
**By:** Neo
**Date:** 2026-06-26
**What:** VisualHierarchy now uses spatial attention, competitive inhibition, and higher adaptive IT capacity, lifting real CIFAR-10 hierarchy accuracy from 26.4% to 30.0% while preserving 1.000 OOD AUROC.
**Why:** Implemented biologically plausible hierarchy upgrades — spatial V1 attention gating, stronger lateral inhibition during winner selection, adaptive pool growth/pruning with a much larger IT capacity bias, and deterministic initialization seeding. Validation passed with 67 targeted tests, and `experiments/real_cifar_comparison.py` now reports hierarchy and both at 30.0% accuracy with 1.000 OOD AUROC on real CIFAR-10.
**References:** `bioarn/hierarchy/visual_hierarchy.py`, `bioarn/hierarchy/config.py`, `bioarn/hierarchy/attention.py`, `bioarn/hierarchy/competition.py`, `experiments/real_cifar_comparison.py`, `commit 627e3e3`

### 2026-06-26: Complementary hierarchy+ensemble routing now beats hierarchy alone on real CIFAR-10
**By:** Trinity
**Date:** 2026-06-26
**What:** Validated the shared-CCC multimodal demo end-to-end and replaced abstention-only combination logic with calibrated complementary routing so the combined configuration now beats hierarchy alone on observed real CIFAR-10 runs.
**Why:** The old `both` path let hierarchy confidence block ensemble participation on hard samples. Calibrated routing keeps hierarchy as the default expert, delegates low-confidence cases to the ensemble, and permits strong high-agreement ensemble overrides on disagreement cases. Validation preserved 1.000 multimodal retrieval accuracy with 3/3 shared CCCs and improved real CIFAR comparison results to baseline 9.8% / 0.778 AUROC, hierarchy 29.0% / 1.000, ensemble 26.0% / 0.972, and both 30.0% / 1.000.
**References:** `experiments/real_cifar_comparison.py`, `experiments/multimodal_demo.py`, `tests/test_multimodal_training.py`, `tests/test_multimodal_fusion.py`, `tests/test_ensemble.py`, `tests/test_integration_improvements.py`, `commit faaa60f`

### 2026-06-26: Portable Loihi 2 export path added for CCC pools and visual hierarchies
**By:** Morpheus
**Date:** 2026-06-26
**What:** Added a portable `bioarn.export` package that serializes trained CCC pools and `VisualHierarchy` instances into Loihi 2-oriented graph JSON plus an NIR-compatible sidecar without requiring Lava as a hard dependency.
**Why:** Gives the project a neuromorphic deployment/export path by representing CCCs and hierarchy layers as LIF populations, explicit synaptic projections, and round-trippable intermediate graph/NIR documents. Validation passed targeted export coverage and preserved the existing Loihi/Lava bridge behavior.
**References:** `bioarn/export`, `tests/test_export.py`, `commit 0b4a32a`

### 2026-06-26: Energy benchmark refresh confirms Bio-ARN still projects a 278× inference-efficiency advantage
**By:** Tank
**Date:** 2026-06-26
**What:** Re-ran `experiments/energy_report.py` and confirmed Bio-ARN still projects 278× lower inference energy on Loihi 2 versus the matched 2-layer transformer baseline on A100 (179.65 µJ vs 50.01 mJ), with the online-training gap unchanged at 8,050×.
**Why:** Confirms the headline energy claim remains current after recent architecture and demo changes; no benchmark-code fixes were required beyond prior demo compatibility work.
**References:** `experiments/energy_report.py`, `experiments/energy_report_results.md`, `experiments/energy_report_data.json`

### 2026-06-25: Real CIFAR-10 ensemble OOD tuning beats baseline
**By:** Morpheus
**Date:** 2026-06-25
**What:** Updated `bioarn/ensemble/voting.py` to make positive confidence more conservative and normalize winning vote mass against expert support capacity with a mild consensus factor. On real CIFAR-10, ensemble OOD AUROC improved from 0.743 to 0.861, beating the 0.778 baseline, while accuracy rose from 23.4% to 25.8%.
**Why:** Real-data OOD scoring was over-rewarding agreement from sparse, low-evidence votes; the new calibration better separates in-distribution images from OOD noise without regressing targeted tests.
**References:** `bioarn/ensemble/voting.py`, `experiments/real_cifar_comparison.py`, `commit afec6d1`

### 2026-06-25: Text gen v3 real-word rate improved with stronger word-level control
**By:** Tank
**Date:** 2026-06-25
**What:** Strengthened word-level generation in `bioarn/language/dual_processor.py`, `bioarn/language/word_level.py`, and `bioarn/training/text_training.py` with larger context tracking, frequency-aware ranking, stronger confidence blending/direct emission, and safer truncation. `experiments/text_gen_v3.py` now improves dual-model real-word rate from 53.6% to 85.7% while keeping repetition low at 0.34.
**Why:** The previous decoder still drifted into partial or awkward word fragments; stronger word control materially improves output quality while preserving diversity.
**References:** `bioarn/language/dual_processor.py`, `bioarn/language/word_level.py`, `bioarn/training/text_training.py`, `experiments/text_gen_v3.py`, `commit c00e8f5`

### 2026-06-25: Sprint work recovered and committed — all improvements preserved
**By:** Squad (Coordinator), requested by Brad Liebs
**Date:** 2026-06-25
**What:** Second machine crash recovery completed. The "Let do them all" sprint's work (431 insertions, 15 files) survived in working tree and has been verified (86/86 key tests pass) and committed in 6 logical commits: ensemble OOD AUROC fix, language improvements, multimodal trainer, infra fixes, CIFAR tuning experiment, gitignore logs.
**Why:** Sprint agents completed their work but machine crashed before commits. All work is now preserved and verified.

### 2026-06-25: Post-fix comparison runs confirm ensemble AUROC recovery on synthetic, mixed performance on real CIFAR-10
**By:** Trinity (Integration Dev)
**Date:** 2026-06-25
**What:** `improvement_comparison.py` and `real_cifar_comparison.py` were rerun after adding repo-root bootstrap to the synthetic comparison script. Synthetic CIFAR-10 now reports ensemble OOD AUROC 1.000 (matching hierarchy and both), confirming the inversion is gone. On real CIFAR-10, hierarchy and both reach 26.4% accuracy / 1.000 AUROC, ensemble reaches 23.4% / 0.743, and baseline reaches 9.8% / 0.778.
**Why:** Confirms the AUROC bug fix worked, while showing that real-data ensemble calibration still needs follow-up.
**References:** `experiments/improvement_comparison.py`, `experiments/real_cifar_comparison.py`

### 2026-06-25: Text generation v3 validation shows better prompt coherence with lower repetition
**By:** Morpheus (Core Learning)
**Date:** 2026-06-25
**What:** `experiments/text_gen_v3.py` produces prompt-aware phrases instead of the char-only baseline's collapsed `the the` output, reducing repetition from 0.93 to 0.31. Validation also showed the repo-root script invocation needs module-path bootstrap to avoid `ModuleNotFoundError`, and the current metric mix still trades off coherence against dictionary-word score.
**Why:** Confirms the dual char+word approach is directionally stronger, while documenting the remaining script usability and evaluation trade-offs.
**References:** `experiments/text_gen_v3.py`, `tests/test_word_level.py`, `tests/test_sequence_memory.py`, `bioarn/language/dual_processor.py`, `bioarn/language/word_level.py`, `bioarn/training/text_training.py`

### 2026-06-25: Simplified GitHub Actions CI workflow added for push/PR validation
**By:** Tank (Core Neural)
**Date:** 2026-06-25
**What:** Added `.github/workflows/ci.yml` (commit `10ef1bc`) with ubuntu-latest, a Python 3.11/3.13 matrix, editable install of `.[dev,demo]`, and pytest configured to skip slow and performance tests. Local validation exposed two pre-existing demo-path failures, so the workflow definition is correct but branch health is not fully green yet.
**Why:** Establishes automated test coverage while isolating long-running suites and surfacing current demo-path defects.
**References:** `.github/workflows/ci.yml`, `tests/test_demo.py`, `bioarn/training/text_training.py`, `demo/models.py`, `commit 10ef1bc`

### 2026-06-25: EnsembleTrainer + VisionTrainer multi-pass/interleave (+24-63% convergence)
**By:** Morpheus (Core Learning)
**Date:** 2026-06-25
**What:** EnsembleTrainer wraps EnsemblePool with per-expert augmented views and Hebbian boosting during training. VisionTrainer gains `num_passes` (+24%) and `interleave_classes` (+63% on sorted streams) kwargs. Both backward-compatible defaults.
**Why:** Hebbian learners can't undo poor early CCC recruitments — interleaved presentation prevents class monopoly of early pool slots.

### 2026-06-25: Hierarchy+Ensemble wired into BioARNCore via lazy imports
**By:** Neo (Lead / Architect)
**Date:** 2026-06-25
**What:** VisualHierarchy and EnsemblePool optionally integrated into BioARNCore. Lazy imports resolve circular dependency chain. `getattr(config, 'hierarchy/ensemble', None)` pattern for backward compat. All 442 tests pass.
**Why:** These modules were orphaned from the training path — now accessible via standard BioARNCore API.

### 2026-06-25: Integration tests — 31 tests covering hierarchy+ensemble+combined
**By:** Switch (Tester)
**Date:** 2026-06-25
**What:** test_integration_improvements.py with 31 tests: BioARNCore+Hierarchy (7), BioARNCore+Ensemble (8), Combined (5), EnsembleTrainer (4), Regression (7). All pass.
**Why:** Validates the integration wiring without regressions.

### 2026-06-25: BioARNConfig extended with hierarchy/ensemble fields
**By:** Trinity (Integration Dev)
**Date:** 2026-06-25
**What:** Optional `hierarchy: HierarchyConfig | None` and `ensemble: EnsembleConfig | None` fields added to BioARNConfig. TYPE_CHECKING guard avoids circular imports. improvement_comparison.py shows hierarchy dominates synthetic (1.0 accuracy, 1.0 AUROC).
**Why:** Config fields activate the modules in BioARNCore without code changes.

### 2026-06-25: Shared-CCC multimodal trainer
**By:** Trinity (Integration Dev)
**Date:** 2026-06-25
**What:** MultimodalTrainer alternates vision/text on shared CCC pool with cross-modal binding via MultimodalFusion. Lazy exports avoid circular imports. Demo + tests included.
**Why:** Exercises the multimodal stack through a canonical training API rather than ad-hoc scripts.

### 2026-06-25: OOD AUROC inversion fixed in ensemble voting
**By:** Neo (Lead / Architect)
**Date:** 2026-06-25
**What:** Confidence scoring in EnsemblePool now uses positive-only confidence mapping [0.5,1.0]→[0,1] and reliability weighting. Fixes inverted AUROC where OOD samples scored higher confidence than in-distribution.
**Why:** Critical metric bug — ensemble was reporting worse OOD detection than random.

### 2026-06-25: Infrastructure — gradio dep + slow test markers
**By:** Tank (Core Neural)
**Date:** 2026-06-25
**What:** gradio>=4.0.0 added as `[demo]` optional dep. Text generation tests marked @pytest.mark.slow. CI can use `-m "not slow"`.
**Why:** Unblocks test_demo.py collection; prevents 20-60min CPU tests from blocking CI.

## Archived Decisions

### 2026-06-26: CIFAR-10 best known config remained the 2k hierarchy control
**By:** Trinity — **Status: SUPERSEDED** (replaced by the 2026-06-26 30.0% hierarchy/both results)
**What:** Real CIFAR-10 tuning kept `hierarchy-control` as the best-known configuration at 26.4% accuracy using 2000 training samples and a 0.33 warmup ratio. New 5000-sample hierarchy runs reached 26.2% and 26.0%, the best vision-only configuration reached 21.4%, and the new `experiments/cifar_scaling.py` run with interleaving plus 3 passes reached 24.4% accuracy / 1.000 OOD AUROC.
**Why:** This was the reference path before Neo's spatial attention / adaptive competition changes and Trinity's complementary routing lifted the current ceiling to 30.0%.
**References:** `experiments/cifar_tuning.py`, `experiments/cifar_scaling.py`, `commit eb2ea53`

### 2026-06-24: Recovery commit — pre-crash work preserved
**By:** Squad (Coordinator)
**What:** All in-progress work (44 files, 6797 insertions) committed as recovery commit (d99976c).

### 2026-06-24: Team hired — The Matrix cast
**By:** Squad (Coordinator)
**What:** Neo (Lead), Tank (Neural), Morpheus (Learning), Trinity (Integration), Switch (Tester) + Scribe, Ralph, Rai.

### 2026-06-25: Architecture assessment — hierarchy/ensemble orphaned from training
**By:** Neo — **Status: RESOLVED** (now wired into BioARNCore)

### 2026-06-25: Test suite health — 411/411 pass, dependency gaps
**By:** Switch — **Status: RESOLVED** (gradio added, slow tests marked, 467+ tests now)

### 2026-06-25: Integration complete — 13 experiments intact
**By:** Trinity — **Status: RESOLVED** (all committed and verified)

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
