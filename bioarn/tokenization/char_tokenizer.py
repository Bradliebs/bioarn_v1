"""Character-level tokenizer for Bio-ARN."""

from __future__ import annotations

import json
import string
from collections import Counter
from pathlib import Path

from bioarn.tokenization.base import Tokenizer
from bioarn.tokenization.vocab import Vocabulary


def _unique_characters(characters: str) -> list[str]:
    return list(dict.fromkeys(characters))


class CharTokenizer(Tokenizer):
    """Character-level tokenizer. Simple, fine-grained, good for Bio-ARN's spike patterns."""

    def __init__(self, vocab: str | None = None) -> None:
        default_vocab = vocab if vocab is not None else string.printable
        self.vocabulary = Vocabulary(_unique_characters(default_vocab))

    @property
    def char_to_id(self) -> dict[str, int]:
        return self.vocabulary.token_to_id

    @property
    def vocab(self) -> Vocabulary:
        return self.vocabulary

    @property
    def id_to_char(self) -> dict[int, str]:
        return self.vocabulary.id_to_token

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [self.vocabulary.get_id(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""

        characters: list[str] = []
        for token_id in token_ids:
            token = self.vocabulary.get_token(token_id)
            if token == "<PAD>":
                continue
            characters.append(token)
        return "".join(characters)

    def train(self, text_corpus: str) -> None:
        """Learn the vocabulary directly from a text corpus."""

        ordered_characters = _unique_characters(text_corpus)
        counts = Counter(text_corpus)

        vocabulary = Vocabulary()
        for character in ordered_characters:
            vocabulary.add_token(character, count=counts[character])
        self.vocabulary = vocabulary

    def save(self, path: str | Path) -> None:
        """Persist tokenizer state as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "char",
            "vocabulary": self.vocabulary.to_state(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a saved tokenizer."""

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(vocab="")
        tokenizer.vocabulary = Vocabulary.from_state(state["vocabulary"])
        return tokenizer
