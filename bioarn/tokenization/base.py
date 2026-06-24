"""Base tokenizer interfaces for Bio-ARN text preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back to text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of text strings."""

        return [self.encode(text) for text in texts]

    def decode_batch(self, token_ids_list: list[list[int]]) -> list[str]:
        """Decode a batch of token id sequences."""

        return [self.decode(token_ids) for token_ids in token_ids_list]

