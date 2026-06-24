"""Vocabulary trie for prefix-constrained word generation."""

from __future__ import annotations


class WordTrie:
    """Prefix tree for fast word completion and validation."""

    _END = "__end__"
    _COUNT = "__count__"

    def __init__(self) -> None:
        self.root: dict[str, dict] = {}

    def insert(self, word: str, weight: int = 1) -> None:
        token = str(word).strip().lower()
        if not token:
            return
        node = self.root
        for char in token:
            node = node.setdefault(char, {})
        node[self._END] = True
        node[self._COUNT] = int(node.get(self._COUNT, 0)) + max(1, int(weight))

    def search(self, prefix: str) -> list[str]:
        return self.get_completions(prefix, top_k=10_000)

    def is_valid_prefix(self, prefix: str) -> bool:
        node = self._find_node(prefix)
        return node is not None

    def is_complete_word(self, word: str) -> bool:
        node = self._find_node(word)
        return bool(node and node.get(self._END, False))

    def get_completions(self, prefix: str, top_k: int = 10) -> list[str]:
        token = str(prefix).lower()
        node = self._find_node(token)
        if node is None:
            return []
        results: list[tuple[str, int]] = []
        self._collect(node, token, results)
        results.sort(key=lambda item: (-item[1], item[0]))
        limit = max(1, int(top_k))
        return [word for word, _ in results[:limit]]

    def next_characters(self, prefix: str) -> set[str]:
        token = str(prefix).lower()
        node = self._find_node(token)
        if node is None:
            return set()
        return {
            char
            for char in node
            if char not in {self._END, self._COUNT}
        }

    def _find_node(self, prefix: str) -> dict | None:
        token = str(prefix).lower()
        node = self.root
        for char in token:
            next_node = node.get(char)
            if not isinstance(next_node, dict):
                return None
            node = next_node
        return node

    def _collect(self, node: dict, prefix: str, results: list[tuple[str, int]]) -> None:
        if node.get(self._END, False):
            results.append((prefix, int(node.get(self._COUNT, 1))))
        for char, child in node.items():
            if char in {self._END, self._COUNT} or not isinstance(child, dict):
                continue
            self._collect(child, prefix + char, results)


__all__ = ["WordTrie"]
