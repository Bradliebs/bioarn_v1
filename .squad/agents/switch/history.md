# Project Context

- **Owner:** Brad Liebs
- **Project:** Bio-ARN 2.0 — Brain-inspired, low-power, multi-modal generative architecture
- **Stack:** Python 3.11+, PyTorch, pytest, spiking neural networks, neuromorphic computing
- **Created:** 2026-06-24

## Architecture Overview

Test areas I own:
- `tests/` — All test files
- Current tests: test_hierarchy, test_ensemble, test_sparse_coding, test_sequence_memory, test_word_level, test_context_attention
- Test framework: pytest
- README claims 202 tests passed

Key areas needing test coverage: spiking neurons, concept cells, GNW workspace, memory systems, predictive coding, training loops, OOD detection, energy metrics.

## Current State (Recovery)

Recovery commit (d99976c) included new tests: test_hierarchy, test_ensemble, test_sparse_coding, test_sequence_memory, test_word_level, test_context_attention. The team was actively testing and seeing improvements.

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
