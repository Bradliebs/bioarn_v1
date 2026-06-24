# Project Context

- **Owner:** Brad Liebs
- **Project:** Bio-ARN 2.0 — Brain-inspired, low-power, multi-modal generative architecture
- **Stack:** Python 3.11+, PyTorch, spiking neural networks, neuromorphic computing
- **Created:** 2026-06-24

## Architecture Overview

Bio-ARN 2.0 is an embodied cognitive stack built from:
- **Spiking neurons** with margin-gated concept cells
- **Global Neuronal Workspace (GNW)** — conscious broadcast channel with attentional spotlight
- **Concept Cell Clusters (CCCs)** — F1 input → F2 concept neuron → F3 prediction layers
- **Sparse associative memory** with Hebbian learning
- **Predictive coding** for top-down/bottom-up inference
- **Continual online learning without backprop**
- **Neuromorphic energy efficiency** targeting Loihi 2

Key results: 82% MNIST accuracy, 0.933 OOD AUROC, 3.2% catastrophic forgetting, 278× lower energy than transformer.

## Current State (Recovery)

Machine crash interrupted active development. Recovery commit (d99976c) saved:
- New modules: hierarchy, ensemble, language, sparse coding
- Workspace improvements: context buffer, selective attention, recurrent context
- Memory: sequence memory
- New experiments and tests
- The team was testing model output and seeing improvements

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
