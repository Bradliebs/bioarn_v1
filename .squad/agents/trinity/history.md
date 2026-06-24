# Project Context

- **Owner:** Brad Liebs
- **Project:** Bio-ARN 2.0 — Brain-inspired, low-power, multi-modal generative architecture
- **Stack:** Python 3.11+, PyTorch, spiking neural networks, neuromorphic computing
- **Created:** 2026-06-24

## Architecture Overview

Integration & experiment areas I own:
- `experiments/` — CIFAR training, ensemble, text gen (v2/v3), feature comparison, context demo
- `bioarn/ensemble/` — Boosting, voting, diversity
- `bioarn/language/` — Word-level processing, dual processor, word trie
- `bioarn/config.py` — System configuration
- `configs/` — Experiment configurations
- `demo/` — Demo scripts

Key results: 82% MNIST accuracy, 0.933 OOD AUROC, 179.65 µJ/inference on projected Loihi 2.

## Current State (Recovery)

Recovery commit (d99976c) included new experiments (CIFAR, ensemble_cifar, text_gen_v2/v3, feature_comparison, context_demo), ensemble module, language module, and config updates. The team was testing and seeing improvements.

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
