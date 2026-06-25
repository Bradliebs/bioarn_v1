from __future__ import annotations

import pytest
import torch

from bioarn.generation import BeamSearchDecoder, GenerationQualityMetrics, NGramCache, RepetitionPenalty
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer, build_builtin_corpus


pytestmark = pytest.mark.slow


def make_config(**overrides) -> TextGenConfig:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=6,
        spike_dim=24,
        num_timesteps=3,
        max_pool_size=64,
        temperature=0.85,
        learning_rate_hebbian=0.025,
        sdm_addresses=256,
        generate_max_tokens=24,
        num_passes=1,
        beam_width=4,
        frequency_boost=0.45,
        repetition_penalty=1.2,
        repetition_window=12,
        use_contextual_patterns=True,
        enable_ngram_cache=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def train_trainer(corpus: str, **overrides) -> TextGenerationTrainer:
    train_samples = int(overrides.pop("num_samples", len(corpus)))
    trainer = TextGenerationTrainer(make_config(**overrides))
    trainer.train_on_corpus(corpus, num_samples=min(len(corpus), train_samples))
    return trainer


CORPUS = build_builtin_corpus(1600)
TRAIN_SLICE = CORPUS[:900]
EVAL_SLICE = CORPUS[900:1300]


def quick_prediction_score(trainer: TextGenerationTrainer, text: str, sample_count: int = 8) -> tuple[float, float]:
    token_ids = trainer.tokenizer.encode(text)
    positions = trainer._sample_positions(token_ids, sample_count)
    correct = 0
    confidences: list[float] = []
    for position in positions:
        torch.manual_seed(1_000 + position)
        context = token_ids[max(0, position - trainer.config.context_length) : position]
        prediction = trainer._predict_from_tokens(context, temperature=0.8, repetition_penalty=None)
        correct += int(prediction.token_id == token_ids[position])
        confidences.append(float(prediction.confidence))
    accuracy = float(correct / max(1, len(positions)))
    confidence = float(sum(confidences) / max(1, len(confidences)))
    return accuracy, confidence


def quick_recognition_confidence(trainer: TextGenerationTrainer, text: str, sample_count: int = 8) -> float:
    token_ids = trainer.tokenizer.encode(text)
    positions = trainer._sample_positions(token_ids, sample_count)
    confidences: list[float] = []
    for position in positions:
        torch.manual_seed(2_000 + position)
        context = token_ids[max(0, position - trainer.config.context_length) : position]
        observation = trainer._recognize_without_learning(token_ids[position], context_ids=context[-(trainer.config.context_length - 1) :])
        confidences.append(float(observation.confidence))
    return float(sum(confidences) / max(1, len(confidences)))


@pytest.fixture(scope="module")
def trained_trainer() -> TextGenerationTrainer:
    torch.manual_seed(0)
    trainer = TextGenerationTrainer(make_config(num_passes=2))
    trainer.train_on_corpus(TRAIN_SLICE, num_samples=320)
    return trainer


def test_beam_search_produces_output(trained_trainer: TextGenerationTrainer) -> None:
    results = BeamSearchDecoder(beam_width=4).decode(trained_trainer, "The ", max_tokens=10)

    assert results
    assert results[0].text


def test_beam_wider_more_diverse(trained_trainer: TextGenerationTrainer) -> None:
    narrow = BeamSearchDecoder(beam_width=1).decode(trained_trainer, "The ", max_tokens=8)
    wide = BeamSearchDecoder(beam_width=6).decode(trained_trainer, "The ", max_tokens=8)

    assert len({result.text for result in wide if result.text}) >= len({result.text for result in narrow if result.text})
    assert len({result.text for result in wide if result.text}) >= 2


def test_repetition_penalty_works() -> None:
    scores = torch.tensor([0.6, 0.25, 0.15], dtype=torch.float32)
    penalty = RepetitionPenalty(penalty=1.3, window=5)

    penalized = penalty.apply(scores, {"history": [0, 0, 1, 0], "candidate_ids": [0, 1, 2]})

    assert penalized[0] < scores[0]
    assert penalized[1] <= scores[1]


def test_ngram_cache_learns() -> None:
    cache = NGramCache(max_n=4)
    cache.learn("the the then there ")

    assert cache.counts[2]["t"]["h"] > 0
    assert cache.counts[4]["the"][" "] > 0


def test_ngram_predict() -> None:
    cache = NGramCache(max_n=4)
    cache.learn("the the then there ")

    predictions = cache.predict_next("the", top_k=3)

    assert predictions
    assert predictions[0][0] == " "


def test_top_k_sampling(trained_trainer: TextGenerationTrainer) -> None:
    generated = trained_trainer.generate("The ", max_tokens=10, method="top-k", top_k=5)

    assert generated
    assert set(generated).issubset(set(CORPUS))


def test_top_p_sampling(trained_trainer: TextGenerationTrainer) -> None:
    generated = trained_trainer.generate("Once ", max_tokens=10, method="top-p", top_p=0.9)

    assert generated
    assert set(generated).issubset(set(CORPUS))


def test_larger_corpus_more_concepts() -> None:
    torch.manual_seed(0)
    small_trainer = TextGenerationTrainer(make_config(max_pool_size=96, sdm_addresses=512))
    large_trainer = TextGenerationTrainer(make_config(max_pool_size=96, sdm_addresses=512))

    small_metrics = small_trainer.train_on_corpus(CORPUS[:120], num_samples=120)
    large_metrics = large_trainer.train_on_corpus(CORPUS[:360], num_samples=360)

    assert large_metrics.concepts_learned > small_metrics.concepts_learned


def test_multi_pass_strengthens() -> None:
    torch.manual_seed(0)
    one_pass = TextGenerationTrainer(make_config(num_passes=1))
    two_pass = TextGenerationTrainer(make_config(num_passes=2))

    one_pass.train_on_corpus(TRAIN_SLICE, num_samples=220)
    two_pass.train_on_corpus(TRAIN_SLICE, num_samples=220)

    one_confidence = quick_recognition_confidence(one_pass, EVAL_SLICE, sample_count=6)
    two_confidence = quick_recognition_confidence(two_pass, EVAL_SLICE, sample_count=6)

    assert two_confidence >= (one_confidence - 1e-3)


def test_quality_metrics_compute() -> None:
    metrics = GenerationQualityMetrics()

    report = metrics.evaluate(["the cat sat on the mat", "once upon a time"], reference_corpus=CORPUS)

    assert 0.0 <= report.word_likeness <= 1.0
    assert 0.0 <= report.spacing_quality <= 1.0
    assert report.character_entropy >= 0.0
    assert 0.0 <= report.bigram_naturalness <= 1.0
    assert report.longest_real_word >= 0
    assert 0.0 <= report.repetition_score <= 1.0


def test_generation_has_spaces(trained_trainer: TextGenerationTrainer) -> None:
    generated = trained_trainer.generate("The ", max_tokens=12, method="beam", beam_width=4)

    assert " " in generated


def test_improved_vs_baseline() -> None:
    torch.manual_seed(0)
    baseline = TextGenerationTrainer(
        make_config(
            context_length=4,
            max_pool_size=64,
            sdm_addresses=512,
            num_passes=1,
            beam_width=1,
            frequency_boost=0.0,
            repetition_penalty=1.0,
            use_contextual_patterns=False,
            enable_ngram_cache=False,
        )
    )
    improved = TextGenerationTrainer(
        make_config(
            context_length=6,
            max_pool_size=96,
            sdm_addresses=512,
            num_passes=2,
            beam_width=4,
            frequency_boost=0.55,
            repetition_penalty=1.2,
            use_contextual_patterns=True,
            enable_ngram_cache=True,
        )
    )

    baseline.train_on_corpus(TRAIN_SLICE, num_samples=220)
    improved.train_on_corpus(TRAIN_SLICE, num_samples=360)

    baseline_accuracy, _ = quick_prediction_score(baseline, EVAL_SLICE, sample_count=6)
    improved_accuracy, _ = quick_prediction_score(improved, EVAL_SLICE, sample_count=6)

    assert improved_accuracy >= baseline_accuracy
