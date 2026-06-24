# Table 3. Energy comparison and deployment projections

| Hardware / Metric | Energy / inference | Power @100 Hz | Cost / 1M inf | Notes |
|---|---:|---:|---:|---|
| Loihi 2 | 179.65 uJ | 17.97 mW | $0.000005 | 278x lower than matched transformer on A100 |
| GPU (A100) | 50.55 mJ | 5.06 W | $0.001404 | Dense reference platform |
| CPU (laptop) | 7.98 mJ | 797.68 mW | $0.000222 | Current software prototype class |
| Ideal ASIC | 74.36 uJ | 7.44 mW | $0.000002 | Long-term projection |
| Brain (scaled baseline) | 3.08 nJ | 308.16 uW | ~0 | Reference only |

## Training-energy comparison

| Model / platform | Training energy for 5K samples |
|---|---:|
| Bio-ARN (projected online local learning) | 0.932 J |
| Transformer (A100 estimate) | 7501.762 J |
| MLP (A100 estimate) | 7500.425 J |

Notes:
- Source: `experiments/energy_report_data.json` and `experiments/energy_report_results.md`.
- Reported Loihi 2 to transformer ratio: 278x for inference, ~8050x for online-training energy.
