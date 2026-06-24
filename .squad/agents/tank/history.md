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
