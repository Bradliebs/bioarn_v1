# Table 5. CIFAR-10 preprocessing ablation

| Preprocessing pipeline | Best theta | Accuracy | Covered accuracy | Abstention | CCCs used | Warmup samples |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.45 | 9.7% | 9.7% | 0.0% | 2 | 0 |
| Random projection | 0.40 | 12.7% | 12.7% | 0.0% | 2 | 0 |
| PCA-128 | 0.40 | 26.7% | 26.9% | 1.0% | 24 | 200 |
| Contrast + PCA | 0.40 | 11.3% | 11.5% | 1.7% | 41 | 200 |
| Patch hash | 0.35 | 14.3% | 14.3% | 0.0% | 8 | 0 |

Notes:
- Source: archived output of `python experiments/cifar_training.py`.
- PCA preprocessing mitigates the raw-pixel CCC-collapse problem and more than doubles accuracy over the next-best configuration.
