from __future__ import annotations

import re

import torch

from bioarn.generation.metrics import GenerationQualityMetrics
from bioarn.language import DualLevelProcessor, WordLevelConfig, WordLevelProcessor, WordTrie
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer

CORPUS = (
    "The cat sat on the mat. "
    "The dog ran in the park. "
    "The bird sang in the tree. "
    "The fish swam in the sea. "
    "Once upon a time there was a baker who made bread. "
    "The baker opened the shop every morning at dawn. "
    "The warm bread filled the air with a wonderful smell. "
    "People came from far away to buy the fresh bread. "
    "The sun rose over the hills. "
    "The moon shone at night. "
) * 8

PREFIX_TEXT = "the that this those there then their theme thick thin"


def make_word_config(**overrides) -> WordLevelConfig:
    config = WordLevelConfig(
        max_vocabulary=128,
        word_ccc_pool_size=96,
        word_concept_dim=48,
        word_spike_dim=64,
        min_word_frequency=1,
        trie_max_completions=8,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_text_config(**overrides) -> TextGenConfig:
    config = TextGenConfig(
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
        num_passes=1,
        beam_width=1,
        repetition_penalty=1.15,
        repetition_window=8,
        use_contextual_patterns=False,
        enable_ngram_cache=False,
        use_word_level=True,
        word_level=make_word_config(),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_trainer(**overrides) -> TextGenerationTrainer:
    return TextGenerationTrainer(make_text_config(**overrides))


def real_word_ratio(texts: list[str]) -> float:
    report = GenerationQualityMetrics().evaluate(texts, CORPUS)
    return report.word_likeness


def repetition_score(texts: list[str]) -> float:
    report = GenerationQualityMetrics().evaluate(texts, CORPUS)
    return report.repetition_score


def extract_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z']+", text)]


def test_word_processor_init() -> None:
    processor = WordLevelProcessor(make_word_config())

    assert processor.config.max_vocabulary == 128
    assert processor.word_ccc_pool.get_pool_stats()["total_concepts"] == 96


def test_learn_vocabulary() -> None:
    processor = WordLevelProcessor(make_word_config())

    processor.learn_vocabulary(CORPUS)

    assert "the" in processor.vocabulary
    assert "baker" in processor.vocabulary
    assert processor.word_counts["bread"] >= 8


def test_word_trie_insert_search() -> None:
    trie = WordTrie()
    trie.insert("the")
    trie.insert("there")

    assert trie.is_complete_word("the")
    assert "there" in trie.search("the")


def test_trie_prefix_completion() -> None:
    processor = WordLevelProcessor(make_word_config())
    processor.learn_vocabulary(PREFIX_TEXT)

    completions = processor.trie.get_completions("th", top_k=6)

    assert "the" in completions
    assert any(word in completions for word in {"that", "this", "those", "there"})


def test_word_transitions_learned() -> None:
    processor = WordLevelProcessor(make_word_config())
    processor.learn_vocabulary(CORPUS)
    processor.learn_word_transitions(CORPUS)

    predictions = processor.suggest_next_word(["the"], top_k=6)
    predicted_words = {word for word, _ in predictions}

    assert predictions
    assert predicted_words & {"cat", "dog", "bird", "fish", "baker", "sun", "moon", "warm"}


def test_constrain_generation() -> None:
    processor = WordLevelProcessor(make_word_config())
    processor.learn_vocabulary(PREFIX_TEXT)

    constrained = processor.constrain_generation(
        [("z", 0.4), ("e", 0.2), ("a", 0.2), ("i", 0.2), ("o", 0.2)],
        "th",
    )
    chars = [char for char, _ in constrained[:4]]

    assert "e" in chars
    assert any(char in chars for char in {"a", "i", "o"})


def test_generate_word_valid() -> None:
    processor = WordLevelProcessor(make_word_config())
    processor.learn_vocabulary(CORPUS[:240])
    processor.learn_word_transitions(CORPUS[:240])
    dual = DualLevelProcessor(object(), processor)

    generated = dual.generate_word(["the"], temperature=0.6)

    assert generated
    assert dual.word_processor.trie.is_complete_word(generated)


def test_generate_sentence_coherent() -> None:
    trainer = make_trainer()
    trainer.train_on_text(CORPUS[:160], context_length=12)

    generated = trainer.dual_processor.generate_sentence("The", max_words=5, temperature=0.5)
    words = extract_words(generated)

    assert len(words) >= 3
    assert all(trainer.word_processor.trie.is_complete_word(word) for word in words)


def test_dual_processor_trains() -> None:
    class StubCharSystem:
        def __init__(self) -> None:
            self.seen_text: str | None = None

        def train_on_text(self, text: str, context_length: int = 64) -> None:
            del context_length
            self.seen_text = text

    stub = StubCharSystem()
    dual = DualLevelProcessor(stub, WordLevelProcessor(make_word_config()))

    dual.train(CORPUS[:240])

    assert stub.seen_text == CORPUS[:240]
    assert dual.word_processor.vocabulary


def test_word_boundary_detection() -> None:
    processor = WordLevelProcessor(make_word_config())

    assert processor.detect_word_boundary("hello ")
    assert processor.detect_word_boundary("hello.")
    assert not processor.detect_word_boundary("hello")


def test_word_level_beats_char_only() -> None:
    torch.manual_seed(0)
    char_only = make_trainer(
        use_word_level=False,
        use_contextual_patterns=False,
        enable_ngram_cache=False,
        beam_width=1,
    )
    dual = make_trainer(use_word_level=True, beam_width=1)

    train_text = CORPUS[:140]
    char_only.train_on_text(train_text, context_length=12)
    dual.train_on_text(train_text, context_length=12)

    prompts = ["The ", "Once ", "People "]
    char_samples = [char_only.generate(prompt, max_tokens=16, method="beam") for prompt in prompts]
    dual_samples = [dual.generate(prompt, max_tokens=16, method="beam") for prompt in prompts]

    assert real_word_ratio(dual_samples) >= 0.8
    assert repetition_score(dual_samples) < repetition_score(char_samples)


def test_no_backprop_word_level() -> None:
    trainer = make_trainer()
    trainer.train_on_text(CORPUS[:160], context_length=12)

    assert trainer.word_processor is not None
    assert all(parameter.grad is None for parameter in trainer.system.parameters())
    assert all(parameter.grad is None for parameter in trainer.word_processor.word_ccc_pool.parameters())
