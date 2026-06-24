# Table 1. MNIST benchmark comparison across five scenarios

| Model | Accuracy | Few-shot k=1 | Few-shot k=5 | Forgetting ↓ | OOD AUROC | OOD abstention | Active MACs | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bio-ARN | 82.0% | 41.8% | 63.8% | 3.2% | 0.933 | 76.7% | 295,427 | 2.008 |
| MLP | 88.9% | 36.8% | 49.5% | 58.7% | 0.707 | 21.3% | 234,752 | 0.012 |
| Transformer | 82.0% | 23.6% | 32.5% | 55.9% | 0.787 | 16.6% | 4,427,008 | 0.071 |

Notes:
- Means are taken across seeds 42, 123, and 777 from `experiments/benchmarks/results.json`.
- Bio-ARN mean fired CCCs across seeds: 2.72.
- Archived Scenario E `sparsity` in the JSON corresponds to active-MAC density (Bio-ARN mean 0.1089), not silence rate.
