# Bio-ARN 2.0: The Embodied Mind Architecture

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-202%20passed-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Docs](https://img.shields.io/badge/docs-research%20ready-purple)

> Brain-inspired, low-power, multi-modal generative architecture.

Bio-ARN 2.0 is an embodied cognitive stack built from spiking neurons, margin-gated concept cells, sparse associative memory, predictive coding, and a global workspace. It is designed for **honest abstention**, **continual online learning without backprop**, and **neuromorphic energy efficiency**.

**Docs:** [Architecture Guide](docs/architecture.md) · [API Reference](docs/api_reference.md) · [Getting Started](docs/getting_started.md) · [Research Notes](docs/research_notes.md) · [Contributing](CONTRIBUTING.md)

## Key results

| Result | Bio-ARN 2.0 | Why it matters |
|---|---:|---|
| Accuracy (MNIST subset benchmark) | 82.0% | Matches the transformer benchmark at the same accuracy tier |
| OOD AUROC / abstention | 0.933 / 76.7% | Rejects unfamiliar inputs instead of forcing a guess |
| Continual-learning forgetting | 3.2% | Retains old concepts while learning new ones |
| Projected Loihi 2 energy | 179.65 µJ / inference | 278× lower than the matched transformer on A100 |
| Online-training energy (5k samples) | 0.932 J | 8050× lower than dense transformer training |

## Architecture at a glance

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        BIO-ARN 2.0 SYSTEM                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 GLOBAL NEURONAL WORKSPACE (GNW)              │  │
│  │            "The Conscious Broadcast Channel"                 │  │
│  │   • Temporary amplification of salient CCC activations      │  │
│  │   • Sequential thought / inner speech / planning            │  │
│  │   • Attentional spotlight (winner-take-all + fatigue)       │  │
│  └──────┬──────────────┬──────────────┬────────────────────────┘  │
│         │              │              │                            │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐                   │
│  │  CONCEPT    │ │  CONCEPT   │ │  CONCEPT    │  ... × N         │
│  │  CELL       │ │  CELL      │ │  CELL       │                  │
│  │  CLUSTER    │ │  CLUSTER   │ │  CLUSTER    │                  │
│  │  (CCC)      │ │  (CCC)     │ │  (CCC)      │                  │
│  │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐  │                  │
│  │ │ F1     │ │ │ │ F1     │ │ │ │ F1     │  │                  │
│  │ │ Input  │ │ │ │ Input  │ │ │ │ Input  │  │                  │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤  │                  │
│  │ │ F2     │ │ │ │ F2     │ │ │ │ F2     │  │                  │
│  │ │Concept│ │ │ │Concept│ │ │ │Concept│   │                  │
│  │ │ Neuron│ │ │ │ Neuron│ │ │ │ Neuron│   │                  │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤  │                  │
│  │ │ Margin │ │ │ │ Margin │ │ │ │ Margin │  │                  │
│  │ │ Gate   │ │ │ │ Gate   │ │ │ │ Gate   │   │                  │
│  │ │(Abstain│ │ │ │(Abstain│ │ │ │(Abstain│   │                  │
│  │ │ or     │ │ │ │ or     │ │ │ │ or     │   │                  │
│  │ │ Fire)  │ │ │ │ Fire)  │ │ │ │ Fire)  │   │                  │
│  │ └────────┘ │ │ └────────┘ │ │ └────────┘  │                  │
│  └──────┬──────┘ └─────┬──────┘ └────┬───────┘                   │
│         │              │              │                            │
│  ┌──────▼──────────────▼──────────────▼─────────────────────────┐ │
│  │           ASSOCIATIVE FABRIC (Hebbian + SDM)                │ │
│  │   • Sparse distributed connections between CCCs              │ │
│  │   • Kanerva-style address-based retrieval                    │ │
│  │   • STDP-governed plasticity                                 │ │
│  │   • Lateral inhibition (winner-take-most)                    │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
│                             │                                      │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │              PREDICTIVE ENGINE (PE)                            │ │
│  │   • Hierarchical predictive coding                           │ │
│  │   • Top-down predictions → Bottom-up errors                   │ │
│  │   • Free energy minimization (Friston)                       │ │
│  │   • Active inference: actions to reduce prediction error      │ │
│  │   • Resonance detection: prediction ↔ input match → learn   │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
│                             │                                      │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │         EMBODIED SENSORIMOTOR CORTEX (eSMC)                  │ │
│  │   ┌─────────┐                ┌──────────────────────────┐     │ │
│  │   │ Vision  │                │ Language / Motor Stream │     │ │
│  │   │  SNN    │                │ self-monitoring output   │     │ │
│  │   └────┬────┘                └────────────┬─────────────┘     │ │
│  │        └─────────────────────┬────────────┘                    │ │
│  │   ┌──────────────────────────▼──────────────────────────────┐  │ │
│  │   │ Sensory cortex encodes sparse errors; motor cortex acts │  │ │
│  │   └─────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         REWARD & NOVELTY SYSTEM (Dopaminergic)               │ │
│  │   • Intrinsic reward from prediction error reduction         │ │
│  │   • Novelty boosts learning and lowers hesitation            │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick start

### 1) Install

```powershell
pip install torch
pip install -e .
```

### 2) Smoke-test the Python API

```python
from bioarn.system import BioARNCore
from bioarn.config import BioARNConfig

core = BioARNCore(BioARNConfig())
print(core.get_system_stats())
```

### 3) Train on MNIST

```powershell
python -m bioarn train --preset mnist --data mnist --output models\mnist --max-steps 128 --checkpoint-interval 32
```

### 4) Evaluate and inspect abstention

```powershell
python -m bioarn evaluate --checkpoint models\mnist\latest.pt --data mnist_test --num-samples 128
```

### 5) Run generation

```powershell
python -m bioarn generate --checkpoint models\mnist\latest.pt --prompt "bio arn" --max-tokens 64
```

## Installation

Bio-ARN depends on PyTorch plus a small scientific Python stack.

```powershell
git clone <your-fork-or-clone-url>
cd bioarn
pip install torch
pip install -e .
```

For contributor workflows, install development extras as well:

```powershell
pip install -e .[dev]
```

## CLI usage

Bio-ARN ships a single entry point via `python -m bioarn`.

| Command | Purpose | Example |
|---|---|---|
| `train` | Online training with checkpointing | `python -m bioarn train --preset mnist --data mnist --output models\mnist --max-steps 128` |
| `evaluate` | Accuracy, abstention, sparsity, latency, free energy | `python -m bioarn evaluate --checkpoint models\mnist\latest.pt --data mnist_test --num-samples 128` |
| `generate` | Seed concept-driven text generation | `python -m bioarn generate --checkpoint models\mnist\latest.pt --prompt "hello" --max-tokens 64` |
| `profile` | Quick sparsity, latency, and energy summary | `python -m bioarn profile --preset mnist --data mnist --num-samples 32` |
| `info` | Inspect checkpoint metadata | `python -m bioarn info --checkpoint models\mnist\latest.pt` |

## Project structure

```text
bioarn/
├── bioarn/
│   ├── core/           # Spiking neurons, margin gates, CCCs
│   ├── memory/         # Sparse distributed memory and associative fabric
│   ├── predictive/     # Predictive-coding layers and hierarchy
│   ├── reward/         # Novelty, curiosity, dopamine-style modulation
│   ├── sensorimotor/   # Visual, language, and motor streams
│   ├── hardware/       # PyTorch, Loihi, and mapping abstractions
│   ├── workspace/      # Global neuronal workspace and thought stream
│   ├── training/       # Online trainer and evaluation helpers
│   ├── config.py       # Dataclass configuration surface
│   ├── loop.py         # End-to-end embodied loop
│   ├── scaling.py      # Batched CCCs and scaled system variants
│   └── system.py       # Core Bio-ARN cognition stack
├── configs/            # Ready-to-use YAML presets (mnist, cifar, language)
├── docs/               # Architecture, API, getting-started, research notes
├── experiments/
│   ├── benchmarks/     # Benchmark suite and raw benchmark results
│   ├── energy_report.py
│   └── mnist_poc.py    # Streaming MNIST proof of concept
├── tests/              # 202 regression and research tests
├── CHANGELOG.md
├── CONTRIBUTING.md
├── BioARN_Architecture.md
├── README.md
└── pyproject.toml
```

## How it works

- **Spiking core (`bioarn.core`)** — Leaky integrate-and-fire neurons encode sparse events, while margin gates decide whether a concept cell should fire or honestly abstain. A `ConceptCellCluster` binds fast one-shot recruitment, slower Hebbian refinement, and top-down feedback into one cortical-column-like unit.
- **Associative memory (`bioarn.memory`)** — The sparse distributed memory stores concept-addressed traces and temporal links. The `AssociativeFabric` turns co-activation into associative recall, lateral inhibition, and sequence retrieval without attention matrices.
- **Predictive hierarchy (`bioarn.predictive`)** — Predictive-coding layers generate top-down expectations and propagate only residual error upward. This keeps computation sparse, exposes free energy as a diagnostic, and enables active inference.
- **Global workspace (`bioarn.workspace`)** — The GNW is the bottleneck that selects a few salient CCC activations, broadcasts them, and feeds a short thought stream. It acts as the architecture's conscious working set.
- **Embodied I/O (`bioarn.sensorimotor`)** — Vision and language encoders turn raw inputs into sparse feature codes, while the language motor stream turns concepts back into token sequences with self-monitoring.
- **Reward and novelty (`bioarn.reward`)** — Surprise, curiosity, and dopamine-style scheduling modulate learning rates and concept recruitment, so novel data is learned quickly while familiar data settles.
- **System orchestration (`bioarn.system`, `bioarn.loop`, `bioarn.scaling`, `bioarn.hardware`)** — `BioARNCore` wires cognition, `SensorimotorLoop` closes the perception-action loop, `ScaledBioARN` provides vectorized scaling paths, and hardware backends map the same ideas to PyTorch or neuromorphic targets.

## Benchmarks

Average results over the bundled benchmark seeds (`experiments/benchmarks/results.json`):

| Model | Accuracy | Few-shot k=1 | Forgetting ↓ | OOD AUROC | OOD Abstention | Active MACs | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bio-ARN | 82.0% | 41.8% | 3.2% | 0.933 | 76.7% | 295k | 2.008 ms |
| MLP | 88.9% | 36.8% | 58.7% | 0.707 | 21.2% | 235k | 0.012 ms |
| Transformer | 82.0% | 23.6% | 55.9% | 0.787 | 16.6% | 4.43M | 0.071 ms |

**Interpretation:** the dense MLP is still the strongest tiny digital baseline on raw accuracy and latency, but Bio-ARN wins on honest abstention, continual learning, sparse activity, and projected neuromorphic energy. On the measured inference profile, only 3.6 CCCs fire on average out of 7.0 committed concepts, total modeled-unit sparsity is 82.5%, and predictive suppression removes 47.3% of hierarchy activity before higher-level processing.

## Contributing

1. Create a branch and install `.[dev]` dependencies.
2. Add or update tests under `tests/` for every behavior change.
3. Keep changes local-learning friendly: no backprop-dependent core changes unless explicitly scoped as an experiment.
4. Run `pytest` before opening a PR.
5. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## References

- Hawkins et al. — Thousand Brains / cortical-column-inspired learning
- Friston; Heins et al. — Active inference and the Free Energy Principle
- Kanerva — Sparse Distributed Memory
- Dehaene & Changeux — Global Neuronal Workspace
- Zhu et al.; Xu et al. — Spiking language models (SpikeGPT, SDLLM)
- Hasani et al. — Liquid neural networks and adaptive time constants

## License

MIT.
