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
