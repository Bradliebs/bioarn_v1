"""Train Bio-ARN with enhanced sequence memory and evaluate text generation."""

from __future__ import annotations

import json

import torch

from bioarn.memory import SequenceMemoryConfig
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer


def build_corpus(min_chars: int = 20_000) -> str:
    passages = [
        "The baker opened the door and the warm bread filled the room with a quiet smell. ",
        "Mira said, \"The lantern is by the gate,\" and Jon said, \"Then we will find the letter.\" ",
        "Rain tapped on the window, and the little cat slept by the chair while the clock kept time. ",
        "Once upon a time the fox walked down the road, turned at the bridge, and found a bright key. ",
        "Tea on the table, light in the hall, kind words at the door, and a promise in the letter. ",
        "The river moved under the moon and the small boat rocked in a silver line of light. ",
        "A child asked for a story, and the old baker smiled and said, \"Sit here and listen.\" ",
        "Morning came clearly, the town woke slowly, and the bells began their patient song again. ",
        "\"Hello there,\" said Mira. \"Hello back,\" said Jon, and both of them laughed by the fire. ",
        "The garden was green and the wind bent the leaves while birds sang in the tree above the wall. ",
        "Step by step, the fox learned the path, the door, the key, and the room beyond the gate. ",
        "The moon was bright, the stars were small, and the road was still beside the water. ",
    ]
    corpus = "".join(passages)
    while len(corpus) < min_chars:
        corpus += "".join(passages)
    return corpus[:min_chars]


def baseline_sequence_config() -> SequenceMemoryConfig:
    return SequenceMemoryConfig(
        sdm_addresses=128,
        sdm_content_dim=16,
        max_concepts=512,
        transition_decay=1.0,
        replay_buffer_size=64,
        replay_ratio=0,
        replay_interval=10_000,
        prioritize_surprising=False,
        min_chunk_frequency=999,
        max_chunk_length=4,
        chunk_vocab_size=8,
        sdm_weight=1.0,
        transition_weight=0.0,
        ngram_weight=0.0,
        chunk_weight=0.0,
    )


def improved_sequence_config() -> SequenceMemoryConfig:
    return SequenceMemoryConfig(
        sdm_addresses=128,
        sdm_content_dim=16,
        max_concepts=512,
        transition_decay=0.999,
        replay_buffer_size=128,
        replay_ratio=3,
        replay_interval=40,
        prioritize_surprising=True,
        min_chunk_frequency=4,
        max_chunk_length=5,
        chunk_vocab_size=64,
        sdm_weight=0.3,
        transition_weight=0.4,
        ngram_weight=0.2,
        chunk_weight=0.1,
    )


def make_config(sequence_memory: SequenceMemoryConfig, **overrides) -> TextGenConfig:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=6,
        spike_dim=16,
        num_timesteps=3,
        max_pool_size=32,
        temperature=0.8,
        learning_rate_hebbian=0.02,
        sdm_addresses=128,
        generate_max_tokens=18,
        num_passes=1,
        beam_width=4,
        frequency_boost=0.4,
        repetition_penalty=1.15,
        repetition_window=14,
        use_contextual_patterns=True,
        enable_ngram_cache=True,
        sequence_memory=sequence_memory,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def transition_summary(trainer: TextGenerationTrainer, token: str, top_k: int = 5) -> str:
    token_id = trainer.tokenizer.vocab.get_id(token)
    predictions = trainer.sequence_memory.transition_matrix.predict_next(token_id, top_k=top_k)
    parts = []
    for next_id, probability in predictions:
        parts.append(f"{trainer.tokenizer.decode([next_id])!r}:{probability:.2f}")
    return f"{token!r} -> {', '.join(parts) if parts else '(none)'}"


def main() -> None:
    torch.manual_seed(0)
    corpus = build_corpus()
    train_text = corpus[:4_000]
    eval_text = corpus[4_000:5_200]
    baseline_train_samples = 120
    improved_train_samples = 180

    baseline = TextGenerationTrainer(
        make_config(
            baseline_sequence_config(),
            enable_ngram_cache=False,
            use_contextual_patterns=False,
            beam_width=1,
            num_passes=1,
        )
    )
    improved = TextGenerationTrainer(
        make_config(
            improved_sequence_config(),
            num_passes=2,
            beam_width=4,
        )
    )

    baseline.train_on_corpus(train_text, num_samples=baseline_train_samples)
    improved.train_on_corpus(train_text, num_samples=improved_train_samples)

    baseline_metrics = baseline.evaluate_generation(eval_text, num_samples=32)
    improved_metrics = improved.evaluate_generation(eval_text, num_samples=32)

    prompts = ["The ", "Once ", "\"Hello "]
    baseline_outputs = [baseline.generate(prompt, max_tokens=24, method="beam") for prompt in prompts]
    improved_outputs = [improved.generate(prompt, max_tokens=24, method="beam") for prompt in prompts]
    learned_chunks = improved.sequence_memory.chunk_library.learned_chunks(12)
    transitions = [
        transition_summary(improved, token)
        for token in ["t", "h", "e", " "]
    ]

    summary = {
        "corpus_chars": len(corpus),
        "train_samples": {"baseline": baseline_train_samples, "improved": improved_train_samples},
        "baseline": {
            "prediction_accuracy": round(baseline_metrics.prediction_accuracy, 4),
            "perplexity": round(baseline_metrics.perplexity, 4),
            "word_likeness": round(
                baseline_metrics.quality_report.word_likeness if baseline_metrics.quality_report else 0.0,
                4,
            ),
            "repetition": round(
                baseline_metrics.quality_report.repetition_score if baseline_metrics.quality_report else 0.0,
                4,
            ),
            "samples": baseline_outputs,
        },
        "improved": {
            "prediction_accuracy": round(improved_metrics.prediction_accuracy, 4),
            "perplexity": round(improved_metrics.perplexity, 4),
            "word_likeness": round(
                improved_metrics.quality_report.word_likeness if improved_metrics.quality_report else 0.0,
                4,
            ),
            "repetition": round(
                improved_metrics.quality_report.repetition_score if improved_metrics.quality_report else 0.0,
                4,
            ),
            "samples": improved_outputs,
        },
        "learned_chunks": learned_chunks,
        "transition_examples": transitions,
    }

    print("Bio-ARN text generation v2 experiment")
    print(f"Corpus size: {len(corpus)} chars")
    print(
        "Baseline  | "
        f"accuracy={summary['baseline']['prediction_accuracy']:.3f} "
        f"perplexity={summary['baseline']['perplexity']:.3f} "
        f"word_likeness={summary['baseline']['word_likeness']:.3f} "
        f"repetition={summary['baseline']['repetition']:.3f}"
    )
    print(
        "Improved  | "
        f"accuracy={summary['improved']['prediction_accuracy']:.3f} "
        f"perplexity={summary['improved']['perplexity']:.3f} "
        f"word_likeness={summary['improved']['word_likeness']:.3f} "
        f"repetition={summary['improved']['repetition']:.3f}"
    )
    print("")
    print("Baseline samples:")
    for prompt, sample in zip(prompts, baseline_outputs, strict=False):
        print(f"  {prompt!r} -> {sample}")
    print("Improved samples:")
    for prompt, sample in zip(prompts, improved_outputs, strict=False):
        print(f"  {prompt!r} -> {sample}")
    print("")
    print("Learned chunks:")
    for chunk, frequency in learned_chunks:
        print(f"  {chunk!r}: {frequency}")
    print("")
    print("Transition examples:")
    for line in transitions:
        print(f"  {line}")
    print("")
    print("JSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
