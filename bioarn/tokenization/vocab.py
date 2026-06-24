"""Vocabulary management for Bio-ARN tokenizers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class Vocabulary:
    """Manages token↔id mappings with special tokens."""

    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        self.token_to_id: dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: dict[int, str] = {idx: token for token, idx in self.SPECIAL_TOKENS.items()}
        self.frequencies: Counter[str] = Counter({token: 0 for token in self.SPECIAL_TOKENS})
        if tokens is not None:
            for token in tokens:
                self.add_token(token, count=0)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def tokens(self) -> list[str]:
        """Return tokens ordered by id."""

        return [self.id_to_token[idx] for idx in range(len(self.id_to_token))]

    def add_token(self, token: str, count: int = 1) -> int:
        """Add a token or update its frequency."""

        if not token:
            raise ValueError("token must be a non-empty string.")

        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        self.frequencies[token] += int(count)
        return self.token_to_id[token]

    def remove_token(self, token: str) -> None:
        """Remove a non-special token and compact ids."""

        if token in self.SPECIAL_TOKENS:
            raise ValueError("Special tokens cannot be removed.")
        if token not in self.token_to_id:
            return

        retained = [item for item in self.tokens() if item != token]
        frequencies = {item: int(self.frequencies.get(item, 0)) for item in retained}
        self._rebuild(retained, frequencies)

    def prune(self, min_frequency: int = 1, max_size: int | None = None) -> None:
        """Remove rare tokens while preserving special tokens."""

        if min_frequency < 0:
            raise ValueError("min_frequency must be non-negative.")

        retained = list(self.SPECIAL_TOKENS)
        candidates = [
            token
            for token in self.tokens()
            if token not in self.SPECIAL_TOKENS and self.frequencies.get(token, 0) >= min_frequency
        ]
        candidates.sort(key=lambda token: (-self.frequencies.get(token, 0), self.token_to_id[token]))

        if max_size is not None:
            if max_size < len(self.SPECIAL_TOKENS):
                raise ValueError("max_size must be at least the number of special tokens.")
            candidates = candidates[: max_size - len(self.SPECIAL_TOKENS)]

        retained.extend(candidates)
        frequencies = {token: int(self.frequencies.get(token, 0)) for token in retained}
        self._rebuild(retained, frequencies)

    def get_id(self, token: str) -> int:
        """Lookup token id, defaulting to the unknown token."""

        return self.token_to_id.get(token, self.SPECIAL_TOKENS["<UNK>"])

    def lookup_id(self, token: str) -> int:
        """Alias for token-to-id lookup."""

        return self.get_id(token)

    def get_token(self, token_id: int) -> str:
        """Lookup token string, defaulting to the unknown token."""

        return self.id_to_token.get(int(token_id), "<UNK>")

    def lookup_token(self, token_id: int) -> str:
        """Alias for id-to-token lookup."""

        return self.get_token(token_id)

    def to_state(self) -> dict:
        """Serialize vocabulary state."""

        return {
            "tokens": self.tokens(),
            "frequencies": {token: int(count) for token, count in self.frequencies.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "Vocabulary":
        """Construct a vocabulary from serialized state."""

        vocab = cls()
        tokens = state.get("tokens", [])
        frequencies = state.get("frequencies", {})
        vocab._rebuild(tokens, frequencies)
        return vocab

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from JSON."""

        path = Path(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_state(state)

    def _rebuild(self, ordered_tokens: Iterable[str], frequencies: dict[str, int] | Counter[str]) -> None:
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {idx: token for token, idx in self.SPECIAL_TOKENS.items()}
        self.frequencies = Counter({token: int(frequencies.get(token, 0)) for token in self.SPECIAL_TOKENS})

        seen = set(self.SPECIAL_TOKENS)
        next_id = len(self.SPECIAL_TOKENS)
        for token in ordered_tokens:
            if token in seen:
                continue
            seen.add(token)
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            self.frequencies[token] = int(frequencies.get(token, 0))
            next_id += 1
