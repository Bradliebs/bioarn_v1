# Bio-ARN 2.0 🧠

> Brain-inspired, low-power, multi-modal generative architecture for neuromorphic hardware.

Bio-ARN 2.0 explores a different stack for efficient AI: sparse **Concept Cell Clusters (CCCs)**, **Hebbian-only local learning**, **precision-weighted predictive processing**, **associative memory**, **Global Neuronal Workspace (GNW)** broadcast, and a practical **Loihi 2 export path**.

**Read next:** [Docs Index](docs/README.md) · [Architecture Guide](docs/architecture.md) · [API Reference](docs/api_reference.md) · [System Paper](docs/paper_draft.md) · [Precision Paper](docs/paper_precision_weighting.md) · [Demo Guide](demo/README.md)

> **Honest status:** Bio-ARN is strongest today on open-set/OOD behavior, continual-learning retention, online learning, and projected neuromorphic efficiency. It is **not** yet competitive with modern conv/transformer baselines on hard supervised vision accuracy.

## ✨ Highlights

- 🧠 **Hebbian learning only** in the core architecture — no backpropagation for CCC learning, predictive plasticity, associative memory, or the convolutional front-end.
- ⚡ **278× lower projected inference energy** than a matched transformer baseline (**179.65 µJ** on projected Loihi 2 vs **50.01 mJ** on A100).
- 🎯 **Perfect OOD detection in the current best combined CIFAR configuration** (**AUROC 1.000**).
- 🔊 **Multi-modal** stack spanning vision, audio, temporal/video, language, and multimodal binding.
- 🎮 **RL-capable** with a Bio-ARN world model and local curiosity-driven control (`bioarn/rl/`, `experiments/rl_demo.py`).
- 🖥️ **Neuromorphic-first deployment path** with Loihi 2 export, Lava validation, and round-trip weight fidelity checks.
- 📦 **Importable package interface** with `bioarn.__version__`, CLI entry points, and a memory API.

## 🚀 Quick Start

```bash
pip install -e ".[dev,demo]"
python -c "import bioarn; print(bioarn.__version__)"
python demo/app.py
```

The Gradio demo launches at `http://127.0.0.1:7860` and showcases digit recognition, OOD scoring, precision signals, online learning, and continual-learning behavior.

## 🏗️ Architecture in One Minute

Bio-ARN's key idea is to keep learning **local, sparse, and hardware-friendly**.

1. **CCC pools** recruit, refine, or abstain on concepts online.
2. **Precision weighting** uses uncertainty to decide when Hebbian updates should be strong or conservative.
3. **GNW broadcast + associative memory** bind active concepts into a small shared workspace for routing, generation, retrieval, and control.

```text
sensory input (vision / audio / temporal / language)
        ↓
preprocessing + sparse encoders
        ↓
predictive hierarchy + CCC pools
        ↓
associative memory + GNW workspace
        ↓
generation / retrieval / RL control / neuromorphic export
```

### Core architectural insight

- **CCCs** give Bio-ARN explicit abstention, one-shot recruitment, and sparse concept competition.
- **Precision-weighted predictive processing** turns prediction into a **plasticity governor** instead of destructive iterative settling.
- **GNW** keeps a small, competitive broadcast workspace so the system can bind context without dense global attention.

## 🧩 Modules

| Area | Key paths | What lives there |
|---|---|---|
| Vision | `bioarn/hierarchy/`, `bioarn/core/conv_ccc.py` | V1→V2→V4→IT-style hierarchy and Hebbian convolutional CCC front-end |
| Audio | `bioarn/hierarchy/audio_hierarchy.py`, `bioarn/preprocessing/audio.py` | Audio feature extraction, auditory hierarchy, and online audio training |
| Temporal / Video | `bioarn/temporal/`, `bioarn/data/video.py` | Sequence context, STDP-style temporal learning, and causal prediction |
| RL / World Models | `bioarn/rl/`, `bioarn/training/rl_training.py` | Local world model, action selection, curiosity, and lightweight control environments |
| Associative Memory | `bioarn/memory/associative_engine.py`, `bioarn/memory/associative_fabric.py` | Store/query/reconstruct memory, sparse associations, replay, and sequence recall |
| Memory API | `bioarn/api/` | HTTP server, client, and LangChain-style memory integration |
| Multimodal | `bioarn/multimodal/`, `bioarn/training/multimodal_training.py` | Shared concept spaces, alignment, fusion, and cross-modal retrieval |
| Neuromorphic Export | `bioarn/export/loihi2.py`, `bioarn/export/nir_format.py`, `bioarn/hardware/` | Loihi 2 graph export, NIR packaging, Lava bridge, and energy modeling |

## 🔬 Key Features

### Precision-Weighted Predictive Processing
Inspired by the hippocampal uncertainty-signaling story in **Frank et al. (2026)**, Bio-ARN estimates pool-level uncertainty from recent CCC winner entropy and uses that signal to scale Hebbian plasticity.

- Destructive predictive settling dropped CIFAR-10 accuracy to **11.8%**.
- Single-pass error gating restored **30.0%**.
- Precision weighting kept **30.0%** while improving forgetting to **20.7%** and OOD AUROC to **1.000**.

### Concept Locking & Elastic Protection
Bio-ARN protects mature concepts in two stages:

- **Soft protection** reduces plasticity as concept importance rises.
- **Hard locking** freezes mature concept directions and feedback weights once importance crosses threshold.

This Sprint E path reduced Split-CIFAR-10 mean forgetting from **34.7%** to **20.7%**.

### Lateral Predictive Coding
Sprint I adds **within-pool lateral prediction** so active concepts can predict likely co-firing neighbors.

- Prediction remains **read-only during inference**.
- Lateral mismatch becomes a local surprise/attention signal.
- The combined Sprint I stack improves the operating point without flattening feedforward features.

### Hebbian Conv Learning
`ConvCCCPool` adds a spatial front-end without abandoning Bio-ARN's local-learning constraint.

- Local 2D filters replace flat raw-pixel matching.
- Updates are correlation/Hebbian style, not backprop.
- The current strongest measured benefit is better retention support when combined with locking.

## 📊 Current Snapshot

| Metric | Best current headline |
|---|---:|
| MNIST accuracy | **85.2%** |
| Real CIFAR-10 baseline | **30.0%** |
| Sprint D best online CIFAR | **33.8%** |
| Best combined CIFAR config | **33.0%** |
| OOD AUROC (best combined) | **1.000** |
| Loihi-vs-GPU inference energy | **278× lower projected** |
| Loihi/Lava fidelity delta | **≤ 0.047** absolute accuracy |

## 🎬 Demos & Experiments

| Entry point | Purpose |
|---|---|
| `demo/app.py` | Interactive Gradio showcase for recognition, OOD, online learning, and continual learning |
| `experiments/loihi_e2e_demo.py` | End-to-end Bio-ARN → Loihi 2 export and simulation walkthrough |
| `experiments/audio_demo.py` | Synthetic audio classification with the auditory hierarchy |
| `experiments/temporal_demo.py` | Temporal STDP / sequence prediction demo |
| `experiments/rl_demo.py` | CartPole world-model RL with local curiosity |
| `experiments/associative_memory_demo.py` | Store / query / associate / reconstruct memory workflow |
| `experiments/memory_api_demo.py` | REST memory API server + client demo |
| `experiments/multimodal_demo.py` | Shared-CCC multimodal retrieval and alignment |

## 📚 Papers

1. **Bio-ARN: A Brain-Inspired Architecture for Energy-Efficient Multi-Modal Learning on Neuromorphic Hardware** — [`docs/paper_draft.md`](docs/paper_draft.md)
2. **Precision-Weighted Hebbian Learning: Hippocampal Ripple-Inspired Uncertainty Gating for Neuromorphic Systems** — [`docs/paper_precision_weighting.md`](docs/paper_precision_weighting.md)

## 🖥️ Neuromorphic Deployment

Bio-ARN is designed to stay close to spike-native deployment constraints.

- `bioarn.export.loihi2` exports CCC pools and hierarchies to a Loihi 2-oriented graph.
- `bioarn.export.nir_format` packages a portable intermediate representation.
- `experiments/lava_validation.py` validates round-trip export fidelity.
- `experiments/loihi_e2e_demo.py` shows the full training → export → simulated deployment path.

Current repository benchmarks report:

- **179.65 µJ** projected inference energy on Loihi 2
- **50.01 mJ** for the matched transformer baseline on A100
- **8,050×** lower projected online-training energy
- **Exact weight preservation** in export plus **≤ 0.047** accuracy delta in Lava-style validation

## 🧪 API Reference at a Glance

| Module / class | Role |
|---|---|
| `bioarn.BioARNCore` | Core cognition stack around CCCs, memory, and workspace |
| `bioarn.SensorimotorLoop` | End-to-end perception → cognition → action loop |
| `bioarn.ConvCCCPool` | Convolutional CCC front-end with local learning |
| `bioarn.AssociativeMemoryEngine` | Store/query/reconstruct associative memory engine |
| `bioarn.api.MemoryAPI` | In-process REST-style memory API surface |
| `bioarn.api.BioARNMemoryClient` | Python client for the memory server |
| `bioarn.BioARNWorldModel` / `bioarn.BioARNAgent` | RL world model and controller |
| `bioarn.training.AudioTrainer`, `TemporalTrainer`, `RLTrainer`, `VisionTrainer` | Modality-specific online trainers |

For the full surface, see [`docs/api_reference.md`](docs/api_reference.md).

## 🛠️ Development

```bash
python -m pytest tests/ -q --tb=short -m "not slow"
```

Useful companion commands:

```bash
python -m bioarn --help
python experiments/memory_api_demo.py
python experiments/loihi_e2e_demo.py
```

## 📖 Citation

<details>
<summary>BibTeX</summary>

```bibtex
@misc{liebs2026bioarn,
  title        = {Bio-ARN: A Brain-Inspired Architecture for Energy-Efficient Multi-Modal Learning on Neuromorphic Hardware},
  author       = {Liebs, Brad and the Bio-ARN Team},
  year         = {2026},
  howpublished = {Repository manuscript},
  note         = {See docs/paper_draft.md}
}

@misc{liebs2026precision,
  title        = {Precision-Weighted Hebbian Learning: Hippocampal Ripple-Inspired Uncertainty Gating for Neuromorphic Systems},
  author       = {Liebs, Brad and the Bio-ARN Team},
  year         = {2026},
  howpublished = {Repository manuscript},
  note         = {See docs/paper_precision_weighting.md}
}
```

</details>

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE).
