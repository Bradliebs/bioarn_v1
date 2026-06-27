# Bio-ARN 2.0

[![Bio-ARN CI](https://github.com/Bradliebs/bioarn_v1/actions/workflows/ci.yml/badge.svg)](https://github.com/Bradliebs/bioarn_v1/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Brain-inspired, low-power, multi-modal generative architecture for spiking, online, neuromorphic AI.

Bio-ARN 2.0 is a research codebase for building **spiking neural systems with Hebbian learning instead of backprop**, sparse **Concept Cell Clusters (CCCs)**, a **visual ventral-stream hierarchy**, **multimodal binding**, **ensemble OOD detection**, and a **Loihi 2 export path**. The goal is a practical architecture that learns online, stays sparse, and maps cleanly onto neuromorphic hardware.

**Docs:** [Architecture Guide](docs/architecture.md) · [API Reference](docs/api_reference.md) · [Getting Started](docs/getting_started.md) · [Research Notes](docs/research_notes.md) · [Contributing](CONTRIBUTING.md)

## Installation

```bash
pip install bioarn
```

For development:

```bash
git clone https://github.com/Bradliebs/bioarn_v1.git
cd bioarn_v1
pip install -e ".[dev]"
```

## Key results

| Area | Current result | Notes |
|---|---|---|
| MNIST | **85.2% accuracy** | Online spiking classification baseline |
| Real CIFAR-10 hierarchy | **26.4% accuracy**, **1.000 OOD AUROC** | Best confirmed hierarchy config |
| Ensemble OOD | **0.861 AUROC** vs **0.778** baseline | Diverse experts + Hebbian boosting |
| Text generation v3 | **85.7% real-word rate**, **0.34 repetition** | Dual char+word generation |
| Energy efficiency | **179.65 µJ** vs **50.01 mJ** transformer baseline | **278×** lower projected inference energy |
| Online training gap | **8,050×** more efficient than transformer | Local updates instead of dense backprop |
| Test suite / CI | **483 collected tests**; CI matrix on **Python 3.11 + 3.13** | Workflow in `.github/workflows/ci.yml` |

## Architecture at a glance

```text
Images ──> preprocessing ──> visual hierarchy (V1 → V2 → V4 → IT) ──┐
                                                                     │
Text ──> tokenization ──> dual char+word language stack ─────────────┼──> CCC pools / associative memory
                                                                     │           │
Paired image+text data ──> MultimodalTrainer / shared CCC binding ───┘           │
                                                                                 ▼
                                                                  Global Neuronal Workspace
                                                                 (saliency, broadcast, binding)
                                                                                 │
                                  ┌──────────────────────────────┬───────────────┴──────────────┐
                                  ▼                              ▼                              ▼
                        generation / retrieval          ensemble voting + boosting         monitoring / energy

Model checkpoints ──> persistence ──> export.nir_format / export.loihi2 ──> Lava / Loihi 2 deployment
```

### Core ideas

- **Spiking neural networks + Hebbian learning:** local, online learning with no backprop in the core architecture.
- **Concept Cell Clusters (CCCs):** sparse distributed representations that recruit, refine, and abstain via margin gates.
- **Visual ventral stream:** hierarchical feature learning inspired by **V1 → V2 → V4 → IT**.
- **Global Neuronal Workspace (GNW):** consciousness-inspired broadcast mechanism for binding and selective attention.
- **Ensemble learning:** diverse experts combined with voting and Hebbian boosting for better robustness and OOD behavior.
- **Multimodal training:** shared CCCs connect vision and text through `MultimodalTrainer`.
- **Text generation:** dual character-level and word-level generation for more constrained outputs.
- **Neuromorphic deployment:** portable **Loihi 2** export path and hardware abstraction layers.

## Quick start

### Install for development

```powershell
pip install -e ".[dev]"
```

### Run the test suite

CI-equivalent local run:

```powershell
python -m pytest --ignore=tests/test_performance.py -m "not slow" -q --tb=short
```

Full local collection or regression run:

```powershell
python -m pytest
```

### Run representative experiments

```powershell
python experiments\mnist_poc.py
python experiments\real_cifar_comparison.py
python experiments\text_gen_v3.py
python experiments\multimodal_demo.py
python experiments\energy_report.py
```

### Use the CLI

```powershell
python -m bioarn train --preset mnist --data mnist --output models\mnist --max-steps 128
python -m bioarn evaluate --checkpoint models\mnist\latest.pt --data mnist_test --num-samples 128
python -m bioarn deploy --model <model-name> --store models --output deployments
```

## Module overview

| Package | Purpose |
|---|---|
| `bioarn.core` | Spiking neurons, margin gates, CCC recruitment, and low-level math utilities |
| `bioarn.data` | Vision, language, curriculum, augmentation, and multimodal dataset helpers |
| `bioarn.ensemble` | Expert diversity, voting, and Hebbian-style boosting |
| `bioarn.export` | Portable export formats, including the new **Loihi 2** graph export path |
| `bioarn.generation` | Decoding utilities, generation metrics, and n-gram caching |
| `bioarn.hardware` | Backend abstractions, profiling, energy models, Lava bridges, and Loihi deployment helpers |
| `bioarn.hierarchy` | Receptive fields, feature binding, and the visual **V1 → V2 → V4 → IT** hierarchy |
| `bioarn.language` | Dual-level language processing, word transitions, and constrained generation |
| `bioarn.memory` | Associative fabric, sparse distributed memory, and sequence memory |
| `bioarn.multimodal` | Cross-modal alignment, fusion, captioning, and shared representation utilities |
| `bioarn.persistence` | Checkpoint storage, migrations, quantization, and model packaging |
| `bioarn.predictive` | Predictive-coding layers and hierarchical inference |
| `bioarn.preprocessing` | Sparse coding, patch extraction, contrast, PCA, and projection pipelines |
| `bioarn.reward` | Novelty and curiosity-style modulation signals |
| `bioarn.sensorimotor` | Vision, language, and motor-facing interfaces for end-to-end loops |
| `bioarn.tokenization` | Character, BPE, and spike-aware tokenization plus vocabulary tools |
| `bioarn.training` | Online, vision, text, ensemble, and multimodal training flows |
| `bioarn.utils` | Logging, checkpoint management, reproducibility, and config helpers |
| `bioarn.workspace` | GNW broadcast, selective attention, context buffers, and recurrent context |

### Key top-level modules

- `bioarn.config`: dataclass configuration surface.
- `bioarn.system`: core Bio-ARN cognition stack.
- `bioarn.loop`: sensorimotor loop that wires perception, cognition, and action.
- `bioarn.scaling`: batched CCC pools and larger-scale execution paths.
- `bioarn.cli`: train, evaluate, profile, inspect, and deploy entry points.

## Experiments

### Top-level experiment scripts

| Script | What it does |
|---|---|
| `experiments\mnist_poc.py` | Phase-0 MNIST validation, one-shot learning, and continual-learning checks |
| `experiments\mnist_improved.py` | Tests whether interleaved class presentation improves MNIST accuracy |
| `experiments\real_cifar_comparison.py` | Compares baseline, hierarchy, ensemble, and combined configs on real CIFAR-10 |
| `experiments\cifar_training.py` | Runs CIFAR-10 training with multiple preprocessing pipelines |
| `experiments\cifar_tuning.py` | Tunes CIFAR-10 configurations beyond the current 26% plateau |
| `experiments\cifar_scaling.py` | Scales the real CIFAR-10 hierarchy experiment with more data and replay |
| `experiments\ensemble_cifar.py` | Trains a diverse Bio-ARN ensemble and measures ensemble OOD behavior |
| `experiments\improvement_comparison.py` | Compares hierarchy and ensemble improvements on a synthetic CIFAR-like setup |
| `experiments\feature_comparison.py` | Compares classic and learned visual feature pipelines |
| `experiments\interleave_scaling.py` | Measures how interleaving and multi-pass training scale with dataset size |
| `experiments\large_pool_scaling.py` | Pushes CCC pools into the 5K-10K range and measures scaling behavior |
| `experiments\scaling_report.py` | Produces a production-oriented scaling report for large CCC pools |
| `experiments\text_gen_v2.py` | Enhanced sequence-memory text generation baseline |
| `experiments\text_gen_v3.py` | Dual char+word text generation benchmark and comparison |
| `experiments\train_text_gen.py` | Trains on a larger built-in corpus and compares decoding strategies |
| `experiments\context_demo.py` | Demonstrates longer-context generation with buffer-based attention |
| `experiments\multimodal_demo.py` | Runs shared-CCC multimodal training and cross-modal retrieval |
| `experiments\energy_report.py` | Generates the neuromorphic energy analysis and report |

### Benchmark suites

| Script | What it does |
|---|---|
| `experiments\benchmarks\benchmark_suite.py` | Bio-ARN vs MLP vs Transformer benchmark suite on MNIST-style tasks |
| `experiments\benchmarks\text_gen_benchmarks.py` | Text-generation benchmarks against simple probabilistic baselines |

## Neuromorphic export

Bio-ARN now includes a dedicated **Loihi 2 export path**:

- `bioarn.export.loihi2` builds a portable neuromorphic graph from CCC pools and visual hierarchies.
- `bioarn.export.nir_format` provides an intermediate representation for downstream deployment tooling.
- `bioarn.hardware` contains Loihi/Lava bridge code, deployment helpers, and energy modeling.
- `python -m bioarn deploy ...` packages checkpoints for hardware-oriented deployment workflows.

This keeps the research stack aligned with the project's low-power target instead of treating neuromorphic execution as an afterthought.

## Demo

Install demo dependencies and launch the Gradio app:

```powershell
pip install -e ".[demo]"
python demo\app.py
```

The demo includes digit recognition, text generation, cross-modal retrieval, live learning, and energy visualizations.

## Contributing

1. Install with `pip install -e ".[dev]"`.
2. Run `python -m pytest` before opening a PR.
3. Keep core changes compatible with local learning and sparse execution.
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).
