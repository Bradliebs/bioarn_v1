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
