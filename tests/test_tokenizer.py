"""Tests for Bio-ARN tokenization components."""

from __future__ import annotations

import torch

from bioarn.tokenization import BPETokenizer, CharTokenizer, SpikeTokenEncoder, Vocabulary


def test_char_tokenizer_encode_decode() -> None:
    tokenizer = CharTokenizer()
    text = "Bio-ARN 2.0"

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text


def test_char_tokenizer_special_tokens() -> None:
    tokenizer = CharTokenizer()

    assert tokenizer.char_to_id["<PAD>"] == 0
    assert tokenizer.char_to_id["<UNK>"] == 1
    assert tokenizer.char_to_id["<BOS>"] == 2
    assert tokenizer.char_to_id["<EOS>"] == 3


def test_char_tokenizer_unknown_char() -> None:
    tokenizer = CharTokenizer(vocab="abc")

    token_ids = tokenizer.encode("a🙂c")

    assert token_ids == [tokenizer.char_to_id["a"], tokenizer.char_to_id["<UNK>"], tokenizer.char_to_id["c"]]


def test_bpe_train() -> None:
    tokenizer = BPETokenizer(vocab_size=24)
    tokenizer.train("banana bandana banana")

    assert tokenizer.merges


def test_bpe_encode_decode() -> None:
    tokenizer = BPETokenizer(vocab_size=32)
    tokenizer.train("hello world hello world")
    text = "hello world"

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text


def test_bpe_vocab_size() -> None:
    tokenizer = BPETokenizer(vocab_size=18)
    tokenizer.train("mississippi river mississippi")

    assert tokenizer.vocab_size <= 18


def test_bpe_subword_splitting() -> None:
    tokenizer = BPETokenizer(vocab_size=32)
    tokenizer.train("neuron neuron neuron spikes")

    token_strings = [tokenizer.vocabulary.get_token(token_id) for token_id in tokenizer.encode("neuron")]

    assert token_strings == ["neuron"]


def test_spike_encoder_deterministic() -> None:
    encoder = SpikeTokenEncoder(vocab_size=32, spike_dim=128, num_timesteps=6)

    first = encoder.encode_token(7)
    second = encoder.encode_token(7)

    assert torch.equal(first, second)


def test_spike_encoder_sparse() -> None:
    encoder = SpikeTokenEncoder(vocab_size=16, spike_dim=128)

    pattern = encoder.encode_token(3)
    active_fraction = float(pattern.mean().item())

    assert 0.1 <= active_fraction <= 0.2


def test_spike_encoder_distinct() -> None:
    encoder = SpikeTokenEncoder(vocab_size=32, spike_dim=256)

    first = encoder.encode_token(1)
    second = encoder.encode_token(2)
    overlap = float((first * second).sum().item())

    assert overlap < 0.5 * float(first.sum().item())


def test_spike_sequence_shape() -> None:
    encoder = SpikeTokenEncoder(vocab_size=32, spike_dim=64, num_timesteps=5)

    sequence = encoder.encode_sequence([1, 2, 3, 4])

    assert sequence.shape == (20, 64)


def test_spike_decode_roundtrip() -> None:
    encoder = SpikeTokenEncoder(vocab_size=32, spike_dim=128)

    token_id = 11
    decoded = encoder.decode_spikes(encoder.encode_token(token_id))

    assert decoded == token_id
    assert decoded.confidence == 1.0


def test_vocabulary_save_load(tmp_path) -> None:
    vocabulary = Vocabulary(["a", "b", "c"])
    vocabulary.add_token("bio", count=3)
    path = tmp_path / "vocab.json"

    vocabulary.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.token_to_id == vocabulary.token_to_id
    assert loaded.id_to_token == vocabulary.id_to_token
    assert loaded.frequencies == vocabulary.frequencies


def test_tokenizer_save_load(tmp_path) -> None:
    tokenizer = BPETokenizer(vocab_size=32)
    tokenizer.train("brain inspired brain inspired")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "brain inspired"

    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
