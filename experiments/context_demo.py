"""Demonstrate longer-context generation with buffer-based attention."""

from __future__ import annotations

import re

import torch

from bioarn.generation.metrics import GenerationQualityMetrics
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer


TOPICAL_CORPUS = (
    "The cat sat on the mat while the cat watched the warm fire by the chair. "
    "The kitten purred near the rug and the cat returned to the mat again. "
    "The river moved slowly under the bridge while the water carried silver light. "
    "The river touched the shore, the bridge crossed the water, and the current stayed calm. "
) * 32

PROMPTS = {
    "cat": ("The cat sat on the ", {"cat", "mat", "chair", "rug", "kitten", "fire"}),
    "river": ("The river moved slowly ", {"river", "bridge", "water", "shore", "current", "light"}),
}


def topic_consistency(samples: list[str], keywords: set[str]) -> float:
    words = [word.lower() for sample in samples for word in re.findall(r"[A-Za-z]+", sample)]
    if not words:
        return 0.0
    return float(sum(word in keywords for word in words) / len(words))


def run_condition(
    trainer: TextGenerationTrainer,
    *,
    prompt: str,
    enable_context: bool,
    num_samples: int = 4,
) -> dict[str, object]:
    trainer.enable_generation_context = enable_context
    samples: list[str] = []
    utilizations: list[float] = []
    repetitions: list[float] = []
    drifts: list[float] = []

    for seed in range(num_samples):
        torch.manual_seed(seed)
        samples.append(trainer.generate(prompt, max_tokens=18, temperature=0.85, method="beam"))
        utilizations.append(float(trainer._last_context_utilization))
        repetitions.append(float(trainer._last_context_repetition))
        drifts.append(float(trainer._last_topic_drift))

    quality = GenerationQualityMetrics().evaluate(samples, TOPICAL_CORPUS)
    return {
        "samples": samples,
        "repetition_rate": quality.repetition_score,
        "context_utilization": sum(utilizations) / len(utilizations),
        "repetition_signal": sum(repetitions) / len(repetitions),
        "topic_drift": sum(drifts) / len(drifts),
    }


def main() -> None:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=24,
        spike_dim=64,
        num_timesteps=4,
        max_pool_size=96,
        temperature=0.85,
        learning_rate_hebbian=0.02,
        sdm_addresses=512,
        generate_max_tokens=32,
        beam_width=5,
    )
    trainer = TextGenerationTrainer(config)
    trainer.train_on_text(TOPICAL_CORPUS, context_length=config.context_length)

    print("=== Bio-ARN Context Attention Demo ===")
    print(f"Corpus characters: {len(TOPICAL_CORPUS)}")
    print()

    for label, (prompt, keywords) in PROMPTS.items():
        baseline = run_condition(trainer, prompt=prompt, enable_context=False)
        contextual = run_condition(trainer, prompt=prompt, enable_context=True)

        baseline_consistency = topic_consistency(baseline["samples"], keywords)
        contextual_consistency = topic_consistency(contextual["samples"], keywords)

        print(f"[topic: {label}]")
        print(f"prompt: {prompt!r}")
        print(
            "topic consistency     "
            f"baseline={baseline_consistency:.3f} "
            f"context={contextual_consistency:.3f}"
        )
        print(
            "repetition rate       "
            f"baseline={baseline['repetition_rate']:.3f} "
            f"context={contextual['repetition_rate']:.3f}"
        )
        print(
            "context utilization   "
            f"baseline={baseline['context_utilization']:.3f} "
            f"context={contextual['context_utilization']:.3f}"
        )
        print(
            "repetition signal     "
            f"baseline={baseline['repetition_signal']:.3f} "
            f"context={contextual['repetition_signal']:.3f}"
        )
        print(
            "topic drift           "
            f"baseline={baseline['topic_drift']:.3f} "
            f"context={contextual['topic_drift']:.3f}"
        )
        print("baseline sample:", baseline["samples"][0])
        print("context sample: ", contextual["samples"][0])
        print()


if __name__ == "__main__":
    main()

