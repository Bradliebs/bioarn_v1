# Project Context

- **Owner:** Brad Liebs
- **Project:** Bio-ARN 2.0 — Brain-inspired, low-power, multi-modal generative architecture
- **Stack:** Python 3.11+, PyTorch, spiking neural networks, neuromorphic computing
- **Created:** 2026-06-24

## Architecture Overview

Learning & memory modules I own:
- `bioarn/training/` — Text training, online learning loops
- `bioarn/memory/` — Associative memory, sequence memory, memory config
- `bioarn/predictive/` — Predictive coding (top-down/bottom-up)
- `bioarn/reward/` — Reinforcement/reward signals
- `bioarn/persistence/` — Model saving/loading

Key metrics: 3.2% catastrophic forgetting, 0.932 J training energy (8050× lower than transformer), online learning without backprop.

## Current State (Recovery)

Recovery commit (d99976c) included sequence memory module, memory config, extended text training (v2/v3 experiments), and text generation improvements. The team was testing and seeing improvements in model output.

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->


## Session 2026-06-25T08:42:45Z

**Mission:** Enhancing training for ensemble + tuning learning

**Context:** Team-wide orchestration sprint to optimize training pipelines for new modules.

**Assigned Tasks:**
- Optimize training loop for ensemble module (voting, boosting, expert balance)
- Tune learning rates and schedules for hierarchy-aware training
- Implement adaptive capacity allocation for dynamic experts
- Coordinate with Neo (module integration), Trinity (experiments)

**Session Dependencies:**
- Waiting on Neo for wired system.py
- Switch must clear test suite blocker (venv, PyYAML, gradio)
- Results will be measured in Trinity's comparison experiments


## Session 2026-06-25T18:56:46Z

**Mission:** Recording text generation v3 validation

- Validated `experiments/text_gen_v3.py` against the char-only baseline
- v3 produced prompt-aware phrases while the baseline collapsed into repetitive output
- Metric snapshot: `real_word%` 53.6%, repetition 0.31 vs baseline 100.0% / 0.93
- Targeted regression suites passed: 27/27
- Follow-up: improve repo-root script usability and continue reducing awkward phrase splices while preserving diversity


## Session 2026-06-25T23:43:17Z

**Mission:** Recording real CIFAR ensemble tuning result

- Tuned `bioarn/ensemble/voting.py` for conservative positive confidence and normalized vote-mass scoring
- Real CIFAR-10 ensemble OOD AUROC improved from 0.743 to 0.861, beating the 0.778 baseline
- Accuracy improved from 23.4% to 25.8%
- Targeted suites passed: 44/44; committed `afec6d1`
- Team context: Trinity's `hierarchy-control` run remains the current real-data accuracy benchmark at 26.4%


## Session 2026-06-26T10:30:40Z

**Mission:** Recording portable Loihi 2 export delivery

- Added `bioarn.export` for CCC / hierarchy -> Loihi 2 graph export with an NIR sidecar
- Export path avoids a hard Lava dependency while preserving round-trippable graph structure
- Targeted export validation passed
- Committed `0b4a32a`
- Team context: the export path now complements the latest CIFAR and multimodal gains with a deployment-facing artifact story


## Session 2026-06-27T11:00:00Z

**Mission:** Recording Sprint E convolutional CCC delivery

- Logged the convolutional CCC path, including the shared convolutional F1 stage and `VisionTrainer` opt-in wiring
- Preserved the rationale that vision inputs should keep spatial structure before concept matching
- Team context: Neo's concept locking and Tank's precision weighting are the other merged Sprint E feature deliveries


## Session 2026-06-27T19:30:00Z

**Mission:** Recording publication-draft update delivery

- Updated `docs/paper_draft.md` with Sprint E retention results plus new sections on concept locking, precision weighting, and convolutional CCCs
- Added the Frank et al. (2026) citation and reported the conv-plus-locking forgetting reduction from 34.7% to 20.7%
- Targeted validation passed: 24 tests
- Team context: Trinity delivered the figures while Neo and Tank completed the benchmark and demo support around the paper
