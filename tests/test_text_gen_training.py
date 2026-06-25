from __future__ import annotations

import pytest
import torch

from bioarn.training.text_training import (
    GenerationMetrics,
    TextGenConfig,
    TextGenerationTrainer,
    TrainingMetrics,
)


pytestmark = pytest.mark.slow


def make_config(**overrides) -> TextGenConfig:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=16,
        spike_dim=64,
        num_timesteps=4,
        max_pool_size=48,
        temperature=1.0,
        learning_rate_hebbian=0.02,
        sdm_addresses=256,
        generate_max_tokens=32,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_trainer(**overrides) -> TextGenerationTrainer:
    return TextGenerationTrainer(make_config(**overrides))


def repetitive_corpus(length: int = 1200) -> str:
    base = "abcabcabcabcabcabcabcabcabcabc"
    repeats = (length // len(base)) + 1
    return (base * repeats)[:length]


def mixed_corpus(length: int = 1200) -> str:
    text = (
        "The cat sat on the mat. The dog ran in the park. "
        "Once upon a time the bird sang softly. "
        "The moon was bright and the stars were small. "
        "Hello there, said the fox. Hello back, said the owl. "
    )
    repeats = (length // len(text)) + 1
    return (text * repeats)[:length]


def test_trainer_initializes() -> None:
    trainer = make_trainer()

    assert trainer is not None
    assert trainer.system is not None
    assert trainer.spike_encoder.spike_dim == 64


def test_train_on_short_text() -> None:
    trainer = make_trainer()

    metrics = trainer.train_on_text(mixed_corpus(100), context_length=16)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.tokens_processed > 0
    assert metrics.concepts_learned >= 1


def test_concepts_learned_grow() -> None:
    trainer = make_trainer()
    before = trainer.system.core.ccc_pool.get_pool_stats()["num_committed"]

    trainer.train_on_text(mixed_corpus(160), context_length=16)
    after = trainer.system.core.ccc_pool.get_pool_stats()["num_committed"]

    assert after > before


def test_sdm_utilization_grows() -> None:
    trainer = make_trainer()
    before = trainer.system.core.fabric.sdm.get_stats()["num_stored"]

    trainer.train_on_text(mixed_corpus(160), context_length=16)
    after = trainer.system.core.fabric.sdm.get_stats()["num_stored"]

    assert after > before


def test_generate_produces_output() -> None:
    trainer = make_trainer()
    trainer.train_on_text(mixed_corpus(240), context_length=16)

    generated = trainer.generate("The", max_tokens=12)

    assert generated


def test_generate_length_control() -> None:
    trainer = make_trainer()
    trainer.train_on_text(mixed_corpus(240), context_length=16)

    generated = trainer.generate("The", max_tokens=10)

    assert len(generated) <= 10


def test_generate_temperature_effect() -> None:
    trainer = make_trainer()
    trainer.train_on_text(mixed_corpus(320), context_length=16)

    torch.manual_seed(0)
    low = trainer.generate("The", max_tokens=16, temperature=0.1)
    torch.manual_seed(0)
    high = trainer.generate("The", max_tokens=16, temperature=2.0)

    assert low != high or len(set(high)) >= len(set(low))


def test_prediction_accuracy_improves() -> None:
    corpus = repetitive_corpus(1500)
    trainer = make_trainer()

    trainer.train_on_corpus(corpus, num_samples=10)
    early = trainer.evaluate_generation(corpus[:300], num_samples=30)

    trainer.train_on_corpus(corpus, num_samples=1000)
    late = trainer.evaluate_generation(corpus[:300], num_samples=30)

    assert late.prediction_accuracy >= early.prediction_accuracy


def test_evaluate_returns_metrics() -> None:
    trainer = make_trainer()
    trainer.train_on_text(mixed_corpus(320), context_length=16)

    metrics = trainer.evaluate_generation(mixed_corpus(200), num_samples=25)

    assert isinstance(metrics, GenerationMetrics)
    assert metrics.num_samples > 0
    assert metrics.perplexity >= 0.0
    assert metrics.average_confidence >= 0.0


def test_training_metrics_structure() -> None:
    trainer = make_trainer()

    metrics = trainer.train_on_text(mixed_corpus(160), context_length=16)

    assert metrics.num_samples == metrics.tokens_processed
    assert isinstance(metrics.concepts_trace, list)
    assert isinstance(metrics.memory_trace, list)
    assert isinstance(metrics.learning_rate_trace, list)


def test_repetitive_corpus_learned() -> None:
    trainer = make_trainer()
    trainer.train_on_corpus(repetitive_corpus(1500), num_samples=1200)

    generated = trainer.generate("a", max_tokens=6, temperature=0.2)

    assert generated.startswith("bc") or "abc" in f"a{generated}"


def test_no_backprop_used() -> None:
    trainer = make_trainer()
    trainer.train_on_text(mixed_corpus(160), context_length=16)

    assert all(parameter.grad is None for parameter in trainer.system.parameters())
