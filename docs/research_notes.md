# Research Notes

This document captures the motivation behind Bio-ARN 2.0, how it sits relative to adjacent research programs, and where promising extension points remain.

## Hypothesis

Bio-ARN is built around a simple hypothesis: **an embodied, predictive, sparse, margin-gated architecture can achieve useful generative and recognition behavior with much lower online learning cost than backprop-trained dense models**. The critical ingredients are:

1. **Concept-cell competition with abstention** — do not force a label when no concept fits.
2. **Associative memory instead of dense context mixing** — recall by content and temporal proximity, not attention over all tokens.
3. **Predictive coding instead of all-upstream activation** — transmit residual error, not everything.
4. **Workspace bottleneck** — a small global broadcast encourages serial thought and control.
5. **Local learning rules** — recruit, strengthen, or weaken based on resonance, co-activation, and novelty.

The current repository’s benchmark artifacts support the direction of that hypothesis: 82.0% classification accuracy at the transformer accuracy tier, 76.7% OOD abstention, 3.2% forgetting, and projected Loihi-2 inference energy of 179.65 µJ.

## Comparison with related work

| System / idea | What it contributes | Where Bio-ARN differs |
|---|---|---|
| Thousand Brains / cortical columns | Concept learning through column-like local circuits | Bio-ARN adds explicit abstention, predictive coding, and generation |
| `pymdp` / active inference | Free-energy minimization and action as inference | Bio-ARN grounds the idea in sparse concept memory and spiking-style I/O |
| SpikeGPT / SDLLM | Event-driven or spiking language modeling | Bio-ARN is multimodal, associative, and not just autoregressive next-token prediction |
| Global Neuronal Workspace | Broadcast bottleneck for conscious access | Bio-ARN operationalizes GNW as a planning and thought-stream substrate |
| Kanerva SDM / modern Hopfield memory | Content-addressable sparse memory | Bio-ARN integrates SDM with temporal association, inhibition, and concept voting |
| Neuromorphic hardware (Loihi 2, NorthPole) | Spike-native execution substrate | Bio-ARN is designed so the software abstractions can map to these devices directly |

## Open questions

1. **Scaling laws** — how do CCC capacity, SDM address space, and predictive depth trade off at larger vocabularies and richer visual data?
2. **Generation quality** — can concept-seeded generation remain coherent over longer spans without collapsing into a transformer-like decoder?
3. **Reward shaping** — what novelty / curiosity schedules best stabilize continual learning across tasks?
4. **Motor grounding** — how much does explicit action grounding improve representation quality over passive perception?
5. **Hardware realism** — how closely do the analytic energy gains survive on real sparse kernels or neuromorphic hardware?

## Future directions

### Add new sensory modalities

To add a modality such as audio, touch, or proprioception:

1. Create an encoder in `bioarn\sensorimotor\` that mirrors `VisualEncoder` or `LanguageEncoder`.
2. Emit the same kind of sparse feature tensor expected by `SensorimotorLoop.predict()`.
3. Extend `SensorimotorLoop.sense()` to route the new modality.
4. Add targeted tests for sparsity, predictive suppression, and end-to-end loop integration.

### Add new learning rules

To experiment with alternatives to the current Hebbian / STDP mix:

1. Keep the **no global backprop in core cognition** rule intact unless the experiment explicitly studies that change.
2. Add the rule where local plasticity already exists: CCC update, SDM association, predictive weight update, or reward modulation.
3. Surface the rule through config if it changes behavior enough to need reproducibility.
4. Add regression tests plus a benchmark script that demonstrates why the new rule matters.

### Extend scaling and hardware paths

- Use `BatchedCCCPool` and `ScaledBioARN` to eliminate Python loops in larger studies.
- Add new `NeuromorphicBackend` implementations for sparse accelerators.
- Expand `LoihiMapping` if a concrete deployment toolchain becomes available.

## Publication-ready experiment scripts

These scripts already serve as the backbone for reproducible reports:

| Script | Purpose | Output |
|---|---|---|
| `experiments\mnist_poc.py` | Streaming MNIST proof-of-concept and abstention calibration | Baseline concept-bank behavior |
| `experiments/benchmarks/benchmark_suite.py` | Main Bio-ARN vs MLP vs Transformer comparison | `experiments/benchmarks/results.json` |
| `experiments\energy_report.py` | Sparse compute and hardware-energy analysis | `experiments\energy_report_data.json`, `experiments\energy_report_results.md` |

A simple paper appendix structure for this repo is:

1. **Methods** — cite [Architecture Guide](architecture.md) and [API Reference](api_reference.md).
2. **Training protocol** — report preset/config, seeds, dataset sizes, and checkpoint cadence.
3. **Benchmark section** — use the five benchmark scenarios from `benchmark_suite.py`.
4. **Energy section** — use `energy_report.py` and explicitly separate analytic projections from measured CPU time.
5. **Ablations** — vary `theta_margin`, `max_pool_size`, `concept_dim`, and predictive depth.

## Caveats to state clearly in publications

- The current PyTorch implementation is a research prototype, not a production sparse kernel stack.
- The strongest energy claims are hardware-software co-design claims, not CPU wall-clock claims.
- Dense baselines still win on raw latency in this repository’s current implementation.
- Concept-seeded generation is present, but this repo is not claiming frontier LLM quality.
