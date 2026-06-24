"""Train Bio-ARN with dual char+word processing and compare against char-only generation."""

from __future__ import annotations

import statistics

import torch

from bioarn.generation.metrics import GenerationQualityMetrics
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer

PASSAGES = [
    "The cat sat on the mat. The dog ran in the park. The bird sang in the tree. The fish swam in the sea. ",
    "Once upon a time there was a baker who made bread. The baker opened the shop every morning at dawn. ",
    "The warm bread filled the air with a wonderful smell. People came from far away to buy the fresh bread. ",
    "The sun rose over the hills. The moon shone at night. Stars twinkled in the dark sky above the sleeping town. ",
    "The river moved under the bridge. The lamps along the road made long gold lines on the water. ",
    "A child asked for a story. The old baker smiled and said the town always woke with the smell of bread. ",
    "The market grew busy at noon. The baker wrapped warm loaves in paper and placed them on the wooden shelf. ",
    "Rain tapped on the window. The little bell rang at the door. The fire burned low in the quiet room. ",
]


def build_corpus(target_chars: int = 15_000) -> str:
    corpus = "".join(PASSAGES)
    while len(corpus) < target_chars:
        corpus += "".join(PASSAGES)
    return corpus[:target_chars]


CORPUS = build_corpus()


def make_config(*, use_word_level: bool) -> TextGenConfig:
    return TextGenConfig(
        tokenizer_type="char",
        vocab_size=64,
        context_length=8,
        spike_dim=16,
        num_timesteps=2,
        max_pool_size=24,
        temperature=0.8,
        learning_rate_hebbian=0.02,
        sdm_addresses=64,
        generate_max_tokens=16,
        num_passes=1 if not use_word_level else 2,
        beam_width=1 if not use_word_level else 3,
        frequency_boost=0.2 if not use_word_level else 0.45,
        repetition_penalty=1.05 if not use_word_level else 1.18,
        repetition_window=16,
        use_contextual_patterns=use_word_level,
        enable_ngram_cache=use_word_level,
        use_word_level=use_word_level,
    )


def extract_words(text: str) -> list[str]:
    import re

    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z']+", text)]


def avg_sentence_length(samples: list[str]) -> float:
    lengths = [len(extract_words(sample)) for sample in samples if sample.strip()]
    return float(statistics.fmean(lengths) if lengths else 0.0)


def avg_word_length(samples: list[str]) -> float:
    words = [word for sample in samples for word in extract_words(sample)]
    return float(statistics.fmean(len(word) for word in words) if words else 0.0)


def print_metrics(label: str, samples: list[str], reference_corpus: str) -> None:
    report = GenerationQualityMetrics().evaluate(samples, reference_corpus)
    print(
        f"{label:<10} | real_word%={report.word_likeness * 100:5.1f} | "
        f"avg_word_len={avg_word_length(samples):4.2f} | "
        f"avg_sentence_words={avg_sentence_length(samples):4.2f} | "
        f"repetition={report.repetition_score:4.2f}",
    )


def main() -> None:
    torch.manual_seed(0)
    train_text = CORPUS[:420]
    eval_text = CORPUS[420:960]

    char_only = TextGenerationTrainer(make_config(use_word_level=False))
    dual = TextGenerationTrainer(make_config(use_word_level=True))

    char_only.train_on_text(train_text, context_length=24)
    dual.train_on_text(train_text, context_length=24)

    prompts = [
        "The ",
        "Once ",
        "People ",
        "The baker ",
        "The warm ",
        "Stars ",
        "Rain ",
        "A child ",
        "The market ",
        "The river ",
    ]
    char_samples = [char_only.generate(prompt, max_tokens=16, method="beam") for prompt in prompts]
    dual_samples = [dual.generate(prompt, max_tokens=16, method="beam") for prompt in prompts]

    print("Bio-ARN text generation v3")
    print(f"Corpus size: {len(CORPUS)} chars")
    print(f"Training slice: {len(train_text)} chars | Eval slice: {len(eval_text)} chars")
    print("")
    print("Top 50 learned words:")
    for word, count in dual.word_processor.word_counts.most_common(50):
        print(f"  {word:<12} {count}")
    print("")
    print("Word transitions from 'the':")
    for word, score in dual.word_processor.suggest_next_word(["the"], top_k=10):
        print(f"  the -> {word:<12} {score:.3f}")
    print("")
    print("Generated sentences (char-only vs dual-level):")
    for prompt, baseline, improved in zip(prompts, char_samples, dual_samples, strict=False):
        print(f"  prompt={prompt!r}")
        print(f"    char-only : {baseline}")
        print(f"    dual-level: {improved}")
    print("")
    print("Metrics:")
    print("model      | real_word% | avg_word_len | avg_sentence_words | repetition")
    print_metrics("char-only", char_samples, eval_text)
    print_metrics("dual", dual_samples, eval_text)


if __name__ == "__main__":
    main()
