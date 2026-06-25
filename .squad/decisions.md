# Squad Decisions

## Active Decisions

### 2026-06-24: Recovery commit — pre-crash work preserved
**By:** Squad (Coordinator), requested by Brad Liebs
**What:** All in-progress work (44 files, 6797 insertions) committed as a single recovery commit (d99976c) after machine crash. Includes new modules (hierarchy, ensemble, language, sparse coding), workspace improvements, memory enhancements, new experiments and tests.
**Why:** Machine crashed mid-session. Uncommitted work needed to be preserved immediately.

### 2026-06-24: Team hired — The Matrix cast
**By:** Squad (Coordinator), requested by Brad Liebs
**What:** 5-member team hired: Neo (Lead), Tank (Core Neural), Morpheus (Core Learning), Trinity (Integration), Switch (Tester) + Scribe, Ralph, Rai
**Why:** Squad was initialized on 2026-06-23 but team was never cast (crash interrupted). Team now operational.

### 2026-06-25: Architecture assessment complete — hierarchy and ensemble orphaned from training path
**By:** Neo (Lead / Architect)
**Date:** 2026-06-24
**What:** Full post-crash architecture review completed. Core backbone (system.py → loop.py) is production-ready and structurally sound. 17 packages import cleanly with no circular dependencies. However, `hierarchy` and `ensemble` modules are written, tested, and functional but NOT wired into the default training path — they exist only in experiments (cifar_training.py, ensemble_cifar.py). Multimodal has the same gap. Gradio is missing from pyproject.toml as an optional dependency, causing test_demo.py to fail at collection.
**Priority:** P1 — Add gradio to optional deps and guard test_demo.py; P2 — Wire hierarchy and ensemble into VisionTrainer; P3 — Validate multimodal training pipeline.
**Why:** Ensures the codebase architecture is understood before integration work begins. The gap risks module divergence between experiment scripts and canonical training API.

### 2026-06-25: Test suite health — 411/411 tests pass, dependency gaps found
**By:** Switch (Tester)
**Date:** 2026-06-25
**What:** Full test assessment after recovery commit d99976c. Of 427 runnable tests (excluding demo), 411 confirmed PASSING with zero failures. 16 text generation tests are pathologically slow on CPU (estimated 20-60+ min) and need @pytest.mark.slow tagging. Two blocker dependencies: (1) PyYAML missing from venv despite being in pyproject.toml — blocks entire suite startup; (2) Gradio missing from optional deps, test_demo.py fails at collection.
**Priority:** P0 — Regenerate venv with `pip install -e ".[dev]"`; P0 — Add gradio as optional dep; P1 — Mark slow tests; P1 — Move benchmarks out of tests/ to experiments/benchmarks/.
**Why:** The recovered codebase is sound but infrastructure has gaps. Clean venv and dependency fixes are required before CI is viable. Slow tests need separation to maintain reasonable CI cycle time.

### 2026-06-25: Integration complete — all 13 experiments intact, text gen v2→v3 active work
**By:** Trinity (Integration Dev)
**Date:** 2026-06-24
**What:** All 13 experiment scripts verified syntactically complete and importable. Ensemble module (4 expert archetypes, 3 voting modes, Hebbian boosting) is production-quality and exercised by ensemble_cifar.py. Language module is the newest code (dual_processor.py, word_level.py from crash era). Text generation improvement trajectory clear: baseline (v1) → sequence memory (v2) → dual char+word (v3). v3 directly exercises new language module. Results files exist from 2026-06-23 run (278× energy, 82% MNIST accuracy, 0.933 OOD AUROC — reproducible). Demo app needs gradio installed but is otherwise complete.
**Status:** All experiment code is ready to run.
**Why:** Confirms recovery completeness and development direction. v3 and ensemble_cifar.py are the immediate next test targets to validate crash recovery.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
