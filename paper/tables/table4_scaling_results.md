# Table 4. Scaling results for large CCC pools

| CCC pool size | Init memory (MB) | Vectorized inference (ms/sample) | Learning (ms/sample) | Activation fraction |
|---:|---:|---:|---:|---:|
| 100 | 1.20 | 0.868 | 0.496 | 0.004% |
| 500 | 6.01 | 1.103 | 3.441 | 0.001% |
| 1,000 | 12.02 | 10.306 | 3.020 | 0.001% |
| 2,000 | 24.03 | 10.089 | 18.200 | 0.000% |
| 5,000 | 60.09 | 19.135 | 14.480 | 0.000% |
| 10,000 | 120.17 | 35.847 | 16.060 | 0.000% |

## 10K targeted fast-infer benchmark

| 10K query mode | Latency (ms/sample) |
|---|---:|
| Flat pool | 134.381 |
| Sharded pool | 29.293 |

Notes:
- Source: archived output of `python experiments/scaling_report.py` in the current repository environment.
- The vectorized inference column and the 10K fast-infer microbenchmark use different workloads and are not directly interchangeable.
- Observed memory slope: approximately 0.012 MB per CCC.
