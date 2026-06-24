"""N-gram statistics to supplement spike-based text generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import torch


class NGramCache:
    """Cache learned n-gram statistics to boost generation."""

    def __init__(self, max_n: int = 4) -> None:
        if max_n < 2:
            raise ValueError("max_n must be at least 2.")
        self.max_n = int(max_n)
        self.counts: dict[int, defaultdict[str, Counter[str]]] = {
            n: defaultdict(Counter) for n in range(2, self.max_n + 1)
        }
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def bind_tokenizer(self, tokenizer) -> None:
        """Attach a tokenizer so score boosting can target token ids."""

        vocab = getattr(tokenizer, "vocab", None)
        if vocab is None:
            return
        self.token_to_id = dict(vocab.token_to_id)
        self.id_to_token = dict(vocab.id_to_token)

    def learn(self, text: str) -> None:
        """Count n-gram frequencies from training text."""

        if not text:
            return
        for n in range(2, self.max_n + 1):
            if len(text) < n:
                continue
            for index in range(len(text) - n + 1):
                gram = text[index : index + n]
                context = gram[:-1]
                next_char = gram[-1]
                self.counts[n][context][next_char] += 1

    def predict_next(self, context: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict next char given context using n-gram probabilities."""

        if not context:
            return []
        limit = max(1, int(top_k))
        for n in range(min(self.max_n, len(context) + 1), 1, -1):
            key = context[-(n - 1) :]
            choices = self.counts[n].get(key)
            if not choices:
                continue
            total = float(sum(choices.values()))
            ranked = sorted(
                ((char, float(count) / max(total, 1.0)) for char, count in choices.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            return ranked[:limit]
        return []

    def boost_scores(self, sdm_scores: torch.Tensor, context: str) -> torch.Tensor:
        """Combine SDM retrieval with n-gram statistics."""

        boosted = sdm_scores.clone()
        if boosted.numel() == 0 or not context or not self.token_to_id:
            return boosted

        predictions = self.predict_next(context, top_k=min(8, len(self.token_to_id)))
        if not predictions:
            return boosted

        for token, probability in predictions:
            token_id = self.token_to_id.get(token)
            if token_id is None or token_id >= boosted.numel():
                continue
            boosted[token_id] = boosted[token_id] + (0.9 * float(probability))
        return boosted


__all__ = ["NGramCache"]
