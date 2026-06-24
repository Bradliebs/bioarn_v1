"""Byte Pair Encoding tokenizer for Bio-ARN."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bioarn.tokenization.base import Tokenizer
from bioarn.tokenization.vocab import Vocabulary


class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer for efficient sub-word tokenization."""

    _UNIT_PATTERN = re.compile(r"\S+|\s+", flags=re.UNICODE)

    def __init__(self, vocab_size: int = 1000) -> None:
        if vocab_size <= len(Vocabulary.SPECIAL_TOKENS):
            raise ValueError("vocab_size must exceed the number of special tokens.")

        self.target_vocab_size = int(vocab_size)
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.vocabulary = Vocabulary()

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def vocab(self) -> Vocabulary:
        return self.vocabulary

    def train(self, corpus: str, vocab_size: int | None = None) -> None:
        """Learn BPE merge rules from a text corpus."""

        target_size = int(vocab_size or self.target_vocab_size)
        if target_size <= len(Vocabulary.SPECIAL_TOKENS):
            raise ValueError("vocab_size must exceed the number of special tokens.")

        units = [list(unit) for unit in self._split_units(corpus)]
        base_symbols = self._ordered_unique_symbol_list(units)
        representation = [unit.copy() for unit in units]
        merges: list[tuple[str, str]] = []

        while len(Vocabulary.SPECIAL_TOKENS) + len(base_symbols) + len(merges) < target_size:
            pair_counts = self._count_pairs(representation)
            if not pair_counts:
                break

            best_pair, best_frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
            if best_frequency < 2:
                break

            representation = [self._merge_unit(unit, best_pair) for unit in representation]
            merges.append(best_pair)

        self.target_vocab_size = target_size
        self.merges = merges
        self.merge_ranks = {pair: index for index, pair in enumerate(self.merges)}

        final_counts = Counter(symbol for unit in representation for symbol in unit)
        ordered_tokens = base_symbols + [left + right for left, right in self.merges]

        vocabulary = Vocabulary()
        for token in ordered_tokens:
            vocabulary.add_token(token, count=final_counts.get(token, 0))
        self.vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        token_ids: list[int] = []
        for unit in self._split_units(text):
            symbols = [character if character in self.vocabulary else "<UNK>" for character in unit]
            merged_symbols = self._apply_merges(symbols)
            token_ids.extend(self.vocabulary.get_id(symbol) for symbol in merged_symbols)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""

        parts: list[str] = []
        for token_id in token_ids:
            token = self.vocabulary.get_token(token_id)
            if token == "<PAD>":
                continue
            parts.append(token)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        """Save the tokenizer state as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "merges": [[left, right] for left, right in self.merges],
            "vocabulary": self.vocabulary.to_state(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from disk."""

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(vocab_size=state["target_vocab_size"])
        tokenizer.merges = [tuple(pair) for pair in state.get("merges", [])]
        tokenizer.merge_ranks = {pair: index for index, pair in enumerate(tokenizer.merges)}
        tokenizer.vocabulary = Vocabulary.from_state(state["vocabulary"])
        return tokenizer

    def _split_units(self, text: str) -> list[str]:
        if not text:
            return []
        return self._UNIT_PATTERN.findall(text)

    def _ordered_unique_symbol_list(self, units: list[list[str]]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for unit in units:
            for symbol in unit:
                if symbol not in seen:
                    seen.add(symbol)
                    ordered.append(symbol)
        return ordered

    def _count_pairs(self, units: list[list[str]]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for unit in units:
            for index in range(len(unit) - 1):
                pair_counts[(unit[index], unit[index + 1])] += 1
        return pair_counts

    def _merge_unit(self, unit: list[str], pair: tuple[str, str]) -> list[str]:
        if len(unit) < 2:
            return unit

        merged: list[str] = []
        index = 0
        while index < len(unit):
            if index < len(unit) - 1 and (unit[index], unit[index + 1]) == pair:
                merged.append(unit[index] + unit[index + 1])
                index += 2
            else:
                merged.append(unit[index])
                index += 1
        return merged

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        if len(symbols) < 2 or not self.merges:
            return symbols

        merged_symbols = symbols.copy()
        while len(merged_symbols) > 1:
            ranked_pairs = [
                (self.merge_ranks[(merged_symbols[index], merged_symbols[index + 1])], index)
                for index in range(len(merged_symbols) - 1)
                if (merged_symbols[index], merged_symbols[index + 1]) in self.merge_ranks
            ]
            if not ranked_pairs:
                break

            _, best_index = min(ranked_pairs)
            merged_symbols = (
                merged_symbols[:best_index]
                + [merged_symbols[best_index] + merged_symbols[best_index + 1]]
                + merged_symbols[best_index + 2 :]
            )
        return merged_symbols
