# Getting Started

This tutorial walks through the shortest path from installation to a trained checkpoint, evaluation metrics, generation, persistence, and benchmark scripts.

## 1. Installation

```powershell
git clone <your-fork-or-clone-url>
cd bioarn
pip install torch
pip install -e .[dev]
```

Sanity check the install:

```powershell
python -c "from bioarn.system import BioARNCore; from bioarn.config import BioARNConfig; print(type(BioARNCore(BioARNConfig())).__name__)"
```

## 2. Train on MNIST

The quickest supported path is the CLI. This uses the built-in online trainer and writes checkpoints plus a resolved config file.

```powershell
python -m bioarn train --preset mnist --data mnist --output models\mnist --max-steps 128 --log-every 25 --checkpoint-interval 32
```

What you get:

- `models\mnist\latest.pt` — latest checkpoint
- `models/mnist/resolved-config.yaml` — fully expanded config used for the run
- `models\mnist\logs\*.jsonl` — structured training logs

## 3. Evaluate accuracy and abstention

```powershell
python -m bioarn evaluate --checkpoint models\mnist\latest.pt --data mnist_test --num-samples 128
```

The evaluation command prints:

- `accuracy` — fraction of correctly recognized samples
- `abstention_rate` — how often the model refused to guess
- `sparsity` — overall sparsity estimate from the core
- `latency_ms` — average step time in milliseconds
- `mean_free_energy` — predictive-coding residual summary

## 4. Generate text

Generation uses a checkpoint plus a prompt-derived concept seed.

```powershell
python -m bioarn generate --checkpoint models\mnist\latest.pt --prompt "bio arn" --max-tokens 64
```

For direct Python control:

```python
import torch
from bioarn.config import BioARNConfig
from bioarn.loop import SensorimotorLoop

loop = SensorimotorLoop(BioARNConfig())
seed = torch.nn.functional.one_hot(torch.tensor(2), num_classes=loop.concept_dim).float()
print(loop.generate_text(seed, max_tokens=16))
```

## 5. Save and load models

Checkpointing is a first-class part of the repository and is used throughout the tests.

```python
from pathlib import Path
import torch

from bioarn.config import BioARNConfig
from bioarn.loop import SensorimotorLoop
from bioarn.utils import CheckpointManager

loop = SensorimotorLoop(BioARNConfig())
loop.step(language_input=torch.tensor([1, 2, 3], dtype=torch.long))

manager = CheckpointManager()
checkpoint = Path('models') / 'tutorial.pt'
manager.save(loop, checkpoint)
loaded = manager.load(checkpoint)
print(type(loaded).__name__, loaded.timestep)
```

## 6. Run the benchmark suite

The publication-style benchmark script compares Bio-ARN with an MLP and a small transformer across five scenarios.

```powershell
python experiments/benchmarks/benchmark_suite.py
```

It writes `experiments/benchmarks/results.json` and prints tables for:

1. Standard classification
2. Few-shot accuracy
3. Continual-learning forgetting
4. OOD detection / abstention
5. Energy-efficiency proxies

## 7. Profile energy

There are two supported profiling paths:

### Quick CLI profile

```powershell
python -m bioarn profile --preset mnist --data mnist --num-samples 32
```

### Full research energy report

```powershell
python experiments\energy_report.py
```

This writes:

- `experiments\energy_report_data.json`
- `experiments\energy_report_results.md`

## Suggested next steps

- Read [Architecture Guide](architecture.md) to understand the perception and generation loops.
- Use [API Reference](api_reference.md) if you want to script experiments directly.
- Read [Research Notes](research_notes.md) before extending the architecture with new modalities or learning rules.
