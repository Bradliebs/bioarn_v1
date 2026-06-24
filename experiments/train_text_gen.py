"""Train Bio-ARN on a larger built-in corpus and compare decoding strategies."""

from __future__ import annotations

from bioarn.generation import GenerationQualityMetrics
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer, build_builtin_corpus


PROMPTS = ["The ", "Once "]
BEFORE_METRICS = {
    "prediction_accuracy": 0.26,
    "perplexity": 3.5,
    "concepts_learned": 2,
    "memory_utilization": 0.89,
    "word_likeness": 0.05,
    "spacing_quality": 0.10,
    "bigram_naturalness": 0.20,
    "repetition_score": 0.75,
}


def main() -> None:
    corpus = build_builtin_corpus(12000)
    eval_text = corpus[900:1500]
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=8,
        spike_dim=24,
        num_timesteps=3,
        max_pool_size=2000,
        temperature=0.85,
        learning_rate_hebbian=0.025,
        sdm_addresses=20000,
        generate_max_tokens=48,
        num_passes=2,
        beam_width=5,
        frequency_boost=0.55,
        repetition_penalty=1.2,
        repetition_window=24,
        use_contextual_patterns=True,
        enable_ngram_cache=True,
    )
    trainer = TextGenerationTrainer(config)
    metrics = trainer.train_on_corpus(corpus, num_samples=360)
    evaluation = trainer.evaluate_generation(eval_text, num_samples=12)
    quality = GenerationQualityMetrics().evaluate(evaluation.generated_examples, eval_text)

    print("=== Bio-ARN Text Generation Improvement Demo ===")
    print(f"Built-in corpus chars: {len(corpus)}")
    print(f"Observed training chars: {metrics.tokens_processed}")
    print()
    print("Metric                     Before      After")
    print("------------------------------------------------")
    print(f"Prediction accuracy        {BEFORE_METRICS['prediction_accuracy']:>7.3f}   {evaluation.prediction_accuracy:>7.3f}")
    print(f"Approx. perplexity         {BEFORE_METRICS['perplexity']:>7.3f}   {evaluation.perplexity:>7.3f}")
    print(f"Concepts learned           {BEFORE_METRICS['concepts_learned']:>7.0f}   {metrics.concepts_learned:>7d}")
    print(f"SDM utilization            {BEFORE_METRICS['memory_utilization']:>7.3f}   {metrics.memory_utilization:>7.3f}")
    print(f"Word likeness             {BEFORE_METRICS['word_likeness']:>7.3f}   {quality.word_likeness:>7.3f}")
    print(f"Spacing quality           {BEFORE_METRICS['spacing_quality']:>7.3f}   {quality.spacing_quality:>7.3f}")
    print(f"Bigram naturalness        {BEFORE_METRICS['bigram_naturalness']:>7.3f}   {quality.bigram_naturalness:>7.3f}")
    print(f"Repetition score          {BEFORE_METRICS['repetition_score']:>7.3f}   {quality.repetition_score:>7.3f}")
    print()

    print("=== Decoding Samples ===")
    for method in ("greedy", "beam", "top-k", "top-p"):
        print(f"\n[{method}]")
        for prompt in PROMPTS:
            generated = trainer.generate(prompt, max_tokens=12, temperature=config.temperature, method=method, beam_width=5, top_k=10, top_p=0.9)
            print(f"{prompt!r} -> {prompt}{generated}")


if __name__ == "__main__":
    main()
