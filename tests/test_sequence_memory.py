from __future__ import annotations

import torch

from bioarn.generation.metrics import GenerationQualityMetrics
from bioarn.memory import (
    ChunkLibrary,
    PredictiveRetrieval,
    ReplayBuffer,
    SequenceMemory,
    SequenceMemoryConfig,
    TransitionMatrix,
)
from bioarn.tokenization import CharTokenizer
from bioarn.training.text_training import TextGenConfig, TextGenerationTrainer, build_builtin_corpus


def _sequence_config(**overrides) -> SequenceMemoryConfig:
    config = SequenceMemoryConfig(
        sdm_addresses=256,
        sdm_content_dim=32,
        max_concepts=256,
        transition_decay=1.0,
        replay_buffer_size=32,
        replay_ratio=3,
        replay_interval=20,
        prioritize_surprising=True,
        min_chunk_frequency=3,
        max_chunk_length=4,
        chunk_vocab_size=32,
        sdm_weight=0.3,
        transition_weight=0.4,
        ngram_weight=0.2,
        chunk_weight=0.1,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _trainer_config(sequence_memory: SequenceMemoryConfig, **overrides) -> TextGenConfig:
    config = TextGenConfig(
        tokenizer_type="char",
        vocab_size=128,
        context_length=6,
        spike_dim=24,
        num_timesteps=3,
        max_pool_size=32,
        temperature=0.8,
        learning_rate_hebbian=0.02,
        sdm_addresses=128,
        generate_max_tokens=18,
        num_passes=1,
        beam_width=3,
        frequency_boost=0.35,
        repetition_penalty=1.12,
        repetition_window=10,
        use_contextual_patterns=True,
        enable_ngram_cache=True,
        sequence_memory=sequence_memory,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _quick_accuracy(trainer: TextGenerationTrainer, text: str, sample_count: int = 8) -> float:
    token_ids = trainer.tokenizer.encode(text)
    positions = trainer._sample_positions(token_ids, sample_count)
    correct = 0
    for position in positions:
        context = token_ids[max(0, position - trainer.config.context_length) : position]
        prediction = trainer._predict_from_tokens(context, temperature=0.7, repetition_penalty=None)
        correct += int(prediction.token_id == token_ids[position])
    return float(correct / max(1, len(positions)))


def test_transition_matrix_records() -> None:
    matrix = TransitionMatrix(max_concepts=16)

    matrix.record_transition(1, 2)
    matrix.record_transition(1, 2)
    matrix.record_transition(1, 3)

    assert matrix.counts[1][2] == 2
    assert matrix.counts[1][3] == 1


def test_transition_matrix_predicts() -> None:
    matrix = TransitionMatrix(max_concepts=16)
    for _ in range(3):
        matrix.record_transition(1, 4)
    matrix.record_transition(1, 2)

    predictions = matrix.predict_next(1, top_k=2)

    assert predictions[0][0] == 4
    assert predictions[0][1] > predictions[1][1]


def test_transition_chain() -> None:
    matrix = TransitionMatrix(max_concepts=16)
    matrix.record_transition(1, 2)
    matrix.record_transition(2, 3)
    matrix.record_transition(3, 4)

    assert matrix.get_chain(1, length=4) == [1, 2, 3, 4]


def test_replay_buffer_stores() -> None:
    replay = ReplayBuffer(buffer_size=4, replay_ratio=2)
    replay.store([1, 2, 3])

    assert len(replay.buffer) == 1
    assert replay.buffer[0].sequence == [1, 2, 3]


def test_replay_strengthens() -> None:
    memory = SequenceMemory(_sequence_config(replay_ratio=4))
    concept_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    concept_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    concept_c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    memory.record_token(1, concept_a)
    memory.record_token(2, concept_b)
    memory.record_token(3, concept_c)
    memory.record_transition(1, 2, concept_a, concept_b)
    memory.record_transition(1, 3, concept_a, concept_c)
    memory.store_sequence([1, 2], prediction_error=0.9)
    before = dict(memory.transition_matrix.predict_next(1, top_k=2))

    replayed = memory.maybe_replay(step_count=20, prediction_errors=[0.9])
    after = dict(memory.transition_matrix.predict_next(1, top_k=2))

    assert replayed > 0
    assert after[2] > before[2]


def test_prioritized_replay() -> None:
    replay = ReplayBuffer(buffer_size=4, replay_ratio=2)
    replay.store([1, 2])
    replay.store([1, 3])
    replay.store([1, 4])

    prioritized = replay.prioritized_replay([0.1, 0.9, 0.2])

    assert prioritized[0] == [1, 3]


def test_chunk_discovery() -> None:
    library = ChunkLibrary(min_frequency=3, max_chunk_length=4)
    tokenizer = CharTokenizer()
    library.bind_tokenizer(tokenizer)
    library.learn_chunks("the cat and the dog and the fox and the tree ")

    learned = dict(library.learned_chunks(10))
    assert "the" in learned


def test_chunk_encode_decode() -> None:
    tokenizer = CharTokenizer()
    library = ChunkLibrary(min_frequency=2, max_chunk_length=4)
    library.bind_tokenizer(tokenizer)
    library.learn_chunks("the and the and ")
    sequence = tokenizer.encode("the and the")

    encoded = library.encode_with_chunks(sequence)
    decoded = library.decode_chunks(encoded)

    assert decoded == sequence
    assert any(isinstance(item, tuple) for item in encoded)


def test_predictive_retrieval_combines() -> None:
    retrieval = PredictiveRetrieval()
    token_id, confidence = retrieval.retrieve_next(
        [1],
        {
            "sdm": [(4, 0.7)],
            "transition": [(5, 0.9)],
            "ngram": [(5, 0.8)],
            "chunk": [(4, 0.4)],
        },
    )

    assert token_id == 5
    assert confidence > 0.45


def test_prediction_accuracy_improves() -> None:
    torch.manual_seed(0)
    corpus = build_builtin_corpus(1000)
    train_text = corpus[:500]
    eval_text = corpus[500:760]
    baseline = TextGenerationTrainer(
        _trainer_config(
            _sequence_config(
                sdm_weight=1.0,
                transition_weight=0.0,
                ngram_weight=0.0,
                chunk_weight=0.0,
                replay_ratio=0,
                replay_interval=10_000,
                min_chunk_frequency=999,
            ),
            enable_ngram_cache=False,
            use_contextual_patterns=False,
        )
    )
    improved = TextGenerationTrainer(
        _trainer_config(
            _sequence_config(replay_interval=20, replay_ratio=3),
            num_passes=2,
            beam_width=4,
        )
    )

    baseline.train_on_corpus(train_text, num_samples=120)
    improved.train_on_corpus(train_text, num_samples=160)

    assert _quick_accuracy(improved, eval_text, sample_count=6) >= _quick_accuracy(baseline, eval_text, sample_count=6)


def test_generation_more_coherent() -> None:
    torch.manual_seed(0)
    corpus = build_builtin_corpus(1000)
    train_text = corpus[:500]
    baseline = TextGenerationTrainer(
        _trainer_config(
            _sequence_config(
                sdm_weight=1.0,
                transition_weight=0.0,
                ngram_weight=0.0,
                chunk_weight=0.0,
                replay_ratio=0,
                replay_interval=10_000,
                min_chunk_frequency=999,
            ),
            enable_ngram_cache=False,
            use_contextual_patterns=False,
        )
    )
    improved = TextGenerationTrainer(_trainer_config(_sequence_config(replay_interval=20), num_passes=2))
    baseline.train_on_corpus(train_text, num_samples=120)
    improved.train_on_corpus(train_text, num_samples=160)

    baseline_sample = baseline.generate("The ", max_tokens=12, method="beam")
    improved_sample = improved.generate("The ", max_tokens=12, method="beam")
    metrics = GenerationQualityMetrics()
    baseline_report = metrics.evaluate([baseline_sample], corpus)
    improved_report = metrics.evaluate([improved_sample], corpus)

    assert improved_report.word_likeness >= baseline_report.word_likeness
    assert improved_report.longest_real_word >= baseline_report.longest_real_word


def test_replay_no_backprop() -> None:
    memory = SequenceMemory(_sequence_config(replay_interval=2, replay_ratio=2))
    memory.record_token(1, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    memory.record_token(2, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    memory.record_transition(1, 2)
    memory.store_sequence([1, 2], prediction_error=0.8)
    memory.maybe_replay(step_count=2, prediction_errors=[0.8])

    assert all(parameter.grad is None for parameter in memory.sdm.parameters())
