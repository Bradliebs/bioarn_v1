# Project Context

- **Owner:** Brad Liebs
- **Project:** Bio-ARN 2.0 — Brain-inspired, low-power, multi-modal generative architecture
- **Stack:** Python 3.11+, PyTorch, spiking neural networks, neuromorphic computing
- **Created:** 2026-06-24

## Architecture Overview

Core neural modules I own:
- `bioarn/core/` — Spiking neurons, concept cells, lateral inhibition
- `bioarn/workspace/` — GNW, context buffer, selective attention, recurrent context
- `bioarn/preprocessing/` — Sparse coding, competitive learning, dictionary learning, pipeline
- `bioarn/hierarchy/` — Visual hierarchy, receptive fields, feature binding
- `bioarn/multimodal/` — Cross-modal integration

Key architecture: CCCs with F1 (input) → F2 (concept neuron) → F3 (prediction), margin-gated with Hebbian learning.

## Current State (Recovery)

Recovery commit (d99976c) included workspace improvements (context buffer, selective attention, recurrent context), new preprocessing (sparse coding, competitive/dictionary learning), and hierarchy module. The team was testing and seeing improvements.

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->


## Session 2026-06-25T18:56:46Z

**Mission:** Recording CI workflow delivery

- Added `.github/workflows/ci.yml` with ubuntu-latest and a Python 3.11/3.13 matrix
- CI command skips slow tests and `tests/test_performance.py`
- Workflow committed as `10ef1bc`
- Local validation surfaced two known demo-path failures, so CI is wired but the branch is not yet fully green


## Session 2026-06-25T23:43:17Z

**Mission:** Recording text generation v3 improvement delivery

- Strengthened word-level generation with frequency-aware ranking, high-confidence overrides, and tighter truncation
- `experiments/text_gen_v3.py` improved real-word rate from 53.6% to 85.7%
- Repetition stayed low at 0.34
- Targeted suites passed: 27/27; committed `c00e8f5`
- Team context: text-generation follow-up can now focus on coherence and usability instead of raw word-hit rate


## Session 2026-06-26T10:30:40Z

**Mission:** Recording demo repair and energy revalidation

- Fixed the demo app regression and revalidated the demo path
- Demo-focused validation passed: 12/12 tests
- Energy benchmarking still shows a 278× projected inference advantage and an 8,050× online-training advantage
- Committed `6954e43`
- Team context: Switch refreshed the README so the repaired demo and benchmark guidance are easier to consume


## Session 2026-06-27T11:00:00Z

**Mission:** Recording Sprint E predictive-processing and benchmark decisions

- Logged precision-weighted predictive processing as the new uncertainty-scaled CCC plasticity path
- Logged the prediction-error-gated default predictive route plus the latest combined and Sprint D benchmark outcomes
- Team context: Neo's concept locking and Morpheus's convolutional CCCs are now recorded as the companion Sprint E features
