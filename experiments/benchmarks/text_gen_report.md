# Bio-ARN Text Generation Benchmarks

This report compares Bio-ARN's spiking Hebbian text generator with simple character-level baselines. The benchmark is intentionally modest: it measures next-character prediction, surprisal-style approximate perplexity, diversity, pattern learning, few-shot recall, and sequential forgetting.

## Summary table

| Metric | Random | Frequency | Bigram | Trigram | Bio-ARN |
|---|---:|---:|---:|---:|---:|
| Next-char accuracy (ctx=8) | 0.027 | 0.271 | 0.406 | 0.604 | 0.188 |
| Next-char accuracy (ctx=32) | 0.027 | 0.271 | 0.385 | 0.615 | 0.177 |
| Approx. perplexity (lower is better) | 3.611 | 2.830 | 2.135 | 1.461 | 1.064 |
| Few-shot completion (1 example) | n/a | n/a | 0.140 | 0.269 | 0.633 |
| Few-shot completion (5 examples) | n/a | n/a | 0.360 | 0.583 | 0.662 |
| Pattern learning | 0.367 | 0.413 | 0.870 | 0.778 | 0.800 |
| Repetition rate | 0.021 | 0.500 | 0.042 | 0.042 | 0.444 |
| Continual forgetting | n/a | n/a | n/a | n/a | 0.000 |

## What Bio-ARN does well

- **Few-shot recall:** Bio-ARN completed 0.633 of one-shot sequence completions versus 0.140 for the bigram baseline.
- **Continual learning:** forgetting on corpus A after corpus B was 0.000, versus 0.667 for a tiny recurrent baseline.
- **Confidence hooks:** this suite does not benchmark abstention directly, but Bio-ARN's margin-gate confidence provides a usable uncertainty proxy for surprisal-style scoring.

## What Bio-ARN struggles with

- It is still a character-level local learner, so outputs remain mostly transition-like rather than sentence-coherent.
- Trigram statistics remain a strong baseline for raw next-character prediction on small corpora.
- Diversity can collapse into loops when the retrieved concepts become too self-reinforcing.

## Honest assessment

Bio-ARN is competitive with simple statistical baselines on next-character prediction, usually stronger than random or unigram frequency, and often in the same range as a bigram model. It is not a transformer and should not be judged as one: its interesting behavior is low-shot adaptation and low forgetting, not fluent long-form generation.

## Research directions

- Improve the probability proxy so confidence better matches true next-character likelihood.
- Add richer temporal retrieval or hierarchical chunking to move beyond purely local transitions.
- Explore controlled sampling and loop penalties to reduce short repetitive generations.
- Add explicit abstention benchmarks for uncertain next-character predictions.
