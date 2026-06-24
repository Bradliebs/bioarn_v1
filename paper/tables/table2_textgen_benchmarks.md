# Table 2. Text-generation benchmark summary

| Model | Next-char acc (ctx=8) | Next-char acc (ctx=32) | Approx. perplexity ↓ | Few-shot 1 | Few-shot 5 | Pattern learning | Repetition rate ↓ | Continual forgetting ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.027 | 0.027 | 3.611 | n/a | n/a | 0.367 | 0.021 | n/a |
| Frequency | 0.271 | 0.271 | 2.830 | n/a | n/a | 0.413 | 0.500 | n/a |
| Bigram | 0.406 | 0.385 | 2.135 | 0.140 | 0.360 | 0.870 | 0.042 | n/a |
| Trigram | 0.604 | 0.615 | 1.461 | 0.269 | 0.583 | 0.778 | 0.042 | n/a |
| Bio-ARN | 0.188 | 0.177 | 1.064 | 0.633 | 0.662 | 0.800 | 0.444 | 0.000 |

Notes:
- Source: `experiments/benchmarks/text_gen_results.json`.
- Bio-ARN's strongest text results are one-shot recall and zero sequential forgetting rather than next-character accuracy.
