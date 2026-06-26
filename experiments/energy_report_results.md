# Bio-ARN 2.0 Energy Efficiency Report

## Executive Summary
Bio-ARN achieves 278x less inference energy than the benchmark 2-layer transformer when projected onto Loihi 2 versus an A100 at the same ~82% MNIST accuracy tier. The measured prototype stays sparse—82.5% of modeled units are silent per inference—with 47.3% predictive sensory suppression and 8,050x lower projected online-training energy than batch backprop.

Key takeaways:
- Sparse activation: only 3.6 CCCs fire on average out of 7.0 committed concepts.
- PCL suppression: 47.3% of predictive-hierarchy activity is zeroed by precision-weighted error suppression.
- Local learning: projected Loihi online learning is 8,050x cheaper than transformer batch training on A100.
- Caveat: current PyTorch CPU inference is slower than the dense MLP/transformer baselines (253.40 ms vs 0.012 / 0.071 ms) because SDM address math and Python orchestration dominate.

## Measured Computation Profile (PyTorch CPU)

| Component | FLOPs (Dense) | FLOPs (Sparse) | Sparsity | Time (ms) |
|---|---:|---:|---:|---:|
| CCC Pool | 1,642,368 | 148,659 | 74.5% | 13.19 |
| SDM Retrieval | 98,476,800 | 42,560,000 | 100.0% | 177.02 |
| PE Hierarchy | 2,940,928 | 1,199,012 | 47.3% | 24.53 |
| GNW | 745 | 285 | 48.6% | 10.74 |
| Motor Stream | 622,592 | 304,947 | 77.9% | 20.62 |
| **TOTAL** | **103,683,433** | **44,212,903** | **82.5%** | **246.09** |

Measured wall-clock per inference: 253.40 ms total (6.31 ms in the visual front-end, excluded from the table above).

## Projected Energy Per Inference

| Hardware | Energy / Inf | Power @100 Hz | Cost / 1M inf | vs Brain |
|---|---:|---:|---:|---:|
| Loihi 2 | 179.65 µJ | 17.97 mW | $0.000005 | 58,298x |
| GPU (A100) | 50.55 mJ | 5.06 W | $0.001404 | 16,404,038x |
| CPU (laptop) | 7.98 mJ | 797.68 mW | $0.000222 | 2,588,491x |
| Ideal ASIC | 74.36 µJ | 7.44 mW | $0.000002 | 24,131x |
| Brain (scaled) | 3.08 nJ | 308.16 µW | $0.000000 | 1x |

## Comparison vs Baselines

Benchmark reference accuracy: Bio-ARN 0.820, Transformer 0.820, MLP 0.889.

| Metric | Bio-ARN (Loihi 2) | Transformer (A100) | Ratio |
|---|---:|---:|---:|
| Energy per inference | 179.65 µJ | 50.01 mJ | 278x |
| Power @ 1k inf/sec | 179.65 mW | 50.01 W | 278x |
| Annual energy cost @1k inf/sec | $0.1574 | $43.81 | 278x |
| Training energy (5k samples) | 931.94 mJ | 7501.76 J | 8,050x |

Reference dense MLP energy on A100: 50.00 mJ per inference. Because the MLP is tiny and dense, it remains a strong digital baseline; Bio-ARN’s energy edge appears primarily against attention-heavy transformers and during online learning on spike-native hardware.

## Sparsity Analysis

| Component | Mechanism | Measured Sparsity |
|---|---|---:|
| CCC pool | Margin-gate winner sparsity | 48.6% |
| SDM | Hamming-radius active-location sparsity | 100.0% |
| PE hierarchy | Predictive suppression / zeroed errors | 47.3% |
| GNW | Limited broadcast capacity | 48.6% |

| Pool Size | Active CCCs | Activation % | Sparse Savings |
|---:|---:|---:|---:|
| 100 | 0.6 | 0.6% | 1,399x |
| 500 | 1.0 | 0.2% | 4,198x |
| 1000 | 1.6 | 0.2% | 5,248x |
| 5000 | 2.0 | 0.0% | 20,990x |

## Learning Profile

Fresh-loop recruitment steps average 44,130,055 sparse FLOPs and 112.65 ms. They are cheaper than warmed-loop inference on this CPU prototype because one-shot learning touches fewer already-committed CCCs and less accumulated SDM state (82,848 fewer sparse FLOPs, 140.75 ms faster). Recruitment occurred on 100.0% of profiled novel samples.

## Limitations

- FLOPs and memory accesses are analytic counts derived from actual tensor shapes and non-zero activity, not hardware-counter reads.
- CPU wall-clock reflects the dense PyTorch prototype; it does not automatically realize the sparse-event savings projected for Loihi 2 / ASIC hardware.
- Benchmark accuracy references come from experiments\benchmarks\results.json.
- Transformer/MLP training energy assumes ~6× inference energy per sample for forward+backward+optimizer work over 5 epochs on 5,000 samples.

## Conclusion
Bio-ARN 2.0 supports the sparse-computation thesis against transformer-class baselines: projected Loihi 2 inference is 278x cheaper than the matched transformer while online learning is 8,050x cheaper than dense backprop training. The current CPU prototype is not yet biologically efficient—still 58,298x above a neuron-count-scaled brain baseline—and it is slower than compact dense models on PyTorch. The result is therefore strongest as a hardware-software co-design claim: sparse, predictive, event-driven Bio-ARN is most compelling on neuromorphic or custom ASIC targets, not in an unoptimized dense CPU implementation.