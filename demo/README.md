# Bio-ARN 2.0 Gradio Demo

## Run locally

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

The app starts on `http://127.0.0.1:7860`.

## What it shows

- **Image Classification** — digit/image prediction, CCC firing view, OOD score, precision signal, and locked vs active CCC counts.
- **Online Learning** — one-shot recruitment of a new CCC from a single labelled example.
- **Continual Learning Demo** — Task 1 → Task 2 → Evaluate All with live BWT comparison for concept locking.
- **Architecture Info** — figure 1 plus headline project metrics and links.

## Model cache

On first launch the demo trains a small CPU-only MNIST model on ~2,000 samples and caches it in:

```text
demo/model_cache/
```

If MNIST cannot be downloaded, the app falls back to synthetic 28×28 data so the demo still launches.

## Hugging Face Spaces

1. Create a **Gradio** Space.
2. Push this repository.
3. Set the app file to:

```text
demo/app.py
```

4. Install dependencies from:

```text
demo/requirements.txt
```

After the first successful startup, the cached demo model allows offline reuse on the same Space storage.
