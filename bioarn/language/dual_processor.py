"""Dual character+word processing for constrained text generation."""

from __future__ import annotations

import math
import re

import torch

from .word_level import WordLevelProcessor

_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_SEPARATORS = {" ", ".", ",", "!", "?", ";", ":", "\n", "\t"}


class DualLevelProcessor:
    """Coordinate character-level and word-level processing."""

    def __init__(self, char_system, word_processor: WordLevelProcessor):
        self.char_system = char_system
        self.word_processor = word_processor

    def generate_word(self, context_words: list[str], temperature: float = 1.0) -> str:
        """Generate one vocabulary-constrained word."""

        if not self.word_processor.vocabulary:
            return ""

        suggestions = self.word_processor.suggest_next_word(context_words, top_k=5)
        target_word = self._select_target_word(context_words, suggestions, temperature=temperature)
        if not target_word:
            return ""

        partial = ""
        for _ in range(max(2, len(target_word) + 2)):
            prompt_text = self._compose_prompt_text(context_words, partial)
            candidates = self._char_candidates(context_words, partial)
            reranked = self.rerank_char_candidates(prompt_text, candidates, temperature=temperature)
            if target_word and len(partial) < len(target_word):
                reranked = self._boost_candidate(
                    reranked,
                    target_word[len(partial)],
                    0.9 + (0.4 / max(0.2, float(temperature))),
                )
            if not reranked:
                break
            next_char = reranked[0][0] if isinstance(reranked[0], tuple) else reranked[0]
            if next_char in _SEPARATORS and self.word_processor.trie.is_complete_word(partial):
                break
            if not next_char or (not next_char.isalpha() and next_char != "'"):
                break
            partial += next_char.lower()
            if partial == target_word:
                break
            if self.word_processor.trie.is_complete_word(partial):
                completions = self.word_processor.trie.get_completions(partial, top_k=1)
                if completions and completions[0] == partial:
                    break

        if self.word_processor.trie.is_complete_word(partial):
            return partial
        if self.word_processor.trie.is_complete_word(target_word):
            return target_word
        completions = self.word_processor.trie.get_completions(partial, top_k=1)
        if completions:
            return completions[0]
        return target_word

    def generate_sentence(self, prompt: str, max_words: int = 20, temperature: float = 1.0) -> str:
        """Generate a word-constrained continuation for a prompt."""

        context_words = [match.group(0).lower() for match in _WORD_PATTERN.finditer(prompt)]
        generated_words: list[str] = []
        max_words = max(1, int(max_words))

        for _ in range(max_words):
            word = self.generate_word(context_words, temperature=temperature)
            if not word:
                break
            generated_words.append(word)
            context_words.append(word)
            if len(generated_words) >= 4 and self.word_processor.sentence_end_probability(word) >= 0.3:
                break

        continuation = " ".join(generated_words).strip()
        if continuation and continuation[-1] not in ".!?":
            continuation = f"{continuation}."
        return continuation

    def train(self, text: str) -> None:
        """Train both character-level and word-level systems on the same text."""

        if hasattr(self.char_system, "train_on_text"):
            context_length = int(getattr(getattr(self.char_system, "config", None), "context_length", 64))
            self.char_system.train_on_text(text, context_length=context_length)
        self.word_processor.learn_vocabulary(text)
        self.word_processor.learn_word_transitions(text)

    def _char_candidates(self, context_words: list[str], partial_word: str) -> list[tuple[str, float]]:
        candidates: list[tuple[str, float]] = []
        tokenizer = getattr(self.char_system, "tokenizer", None)
        predictor = getattr(self.char_system, "_predict_from_tokens", None)
        if tokenizer is None or not callable(predictor):
            return candidates

        prompt = " ".join(context_words[-6:]).strip()
        prompt_text = f"{prompt} {partial_word}".strip() if prompt else partial_word
        token_ids = tokenizer.encode(prompt_text)
        if not token_ids:
            return candidates
        prediction = predictor(token_ids, temperature=max(0.2, float(getattr(self.char_system.config, "temperature", 1.0))), repetition_penalty=None)
        contextualize = getattr(self.char_system, "_apply_generation_context", None)
        if callable(contextualize):
            prediction = contextualize(token_ids, prediction)
        if not getattr(prediction, "candidate_ids", None):
            return candidates
        top_k = min(8, len(prediction.candidate_ids))
        values, indices = torch.topk(prediction.probabilities, k=top_k)
        for value, index in zip(values.tolist(), indices.tolist(), strict=False):
            token_id = int(prediction.candidate_ids[index])
            char = tokenizer.decode([token_id])
            if len(char) == 1:
                candidates.append((char.lower(), float(value)))
        return candidates

    def rerank_char_candidates(
        self,
        prompt_text: str,
        char_candidates: list[tuple[str, float]],
        *,
        temperature: float,
    ) -> list[tuple[str, float]]:
        """Blend character evidence with trie and next-word constraints."""

        if not char_candidates:
            return []

        context_words, partial_word = self._split_prompt_context(prompt_text)
        constrained = self.word_processor.constrain_generation(char_candidates, partial_word)
        scored = {
            str(char)[:1].lower(): float(score)
            for char, score in constrained
            if isinstance(char, str) and str(char)[:1]
        }

        if not scored:
            return []

        for char, bonus in self._suggested_next_chars(context_words, partial_word).items():
            scored[char] = scored.get(char, 0.0) + (
                float(bonus) * (0.45 + (0.5 / max(0.25, float(temperature))))
            )

        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        return [(char, float(score)) for char, score in ranked]

    def _select_target_word(
        self,
        context_words: list[str],
        suggestions: list[tuple[str, float]],
        *,
        temperature: float,
    ) -> str:
        if not suggestions:
            return self.word_processor.vocabulary[0] if self.word_processor.vocabulary else ""

        last_word = context_words[-1].lower() if context_words else None
        filtered = [
            (word, probability)
            for word, probability in suggestions
            if word != last_word or len(suggestions) == 1
        ]
        if not filtered:
            filtered = suggestions

        if len(filtered) == 1 or temperature <= 0.35:
            return filtered[0][0]

        weights = torch.tensor([max(1e-6, float(prob)) for _, prob in filtered], dtype=torch.float32)
        logits = torch.log(weights) / max(0.15, float(temperature))
        probabilities = torch.softmax(logits, dim=0)
        chosen = int(torch.multinomial(probabilities, num_samples=1).item())
        return filtered[chosen][0]

    @staticmethod
    def _compose_prompt_text(context_words: list[str], partial_word: str) -> str:
        prompt = " ".join(context_words[-6:]).strip()
        return f"{prompt} {partial_word}".strip() if prompt else partial_word

    @staticmethod
    def _boost_candidate(
        candidates: list[tuple[str, float]],
        token: str,
        bonus: float,
    ) -> list[tuple[str, float]]:
        char = str(token)[:1].lower()
        if not char:
            return candidates
        scored = {candidate: float(score) for candidate, score in candidates}
        scored[char] = scored.get(char, 0.0) + float(bonus)
        return sorted(scored.items(), key=lambda item: (-item[1], item[0]))

    def _split_prompt_context(self, prompt_text: str) -> tuple[list[str], str]:
        prompt = str(prompt_text or "")
        words = [match.group(0).lower() for match in _WORD_PATTERN.finditer(prompt)]
        if not words:
            return [], ""
        if prompt and prompt[-1] not in _SEPARATORS:
            return words[:-1], words[-1]
        return words, ""

    def _suggested_next_chars(self, context_words: list[str], partial_word: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        suggestions = self.word_processor.suggest_next_word(
            context_words,
            top_k=max(4, int(self.word_processor.config.trie_max_completions)),
        )

        for word, probability in suggestions:
            if partial_word:
                if not word.startswith(partial_word):
                    continue
                if len(word) > len(partial_word):
                    char = word[len(partial_word)]
                    scores[char] = scores.get(char, 0.0) + float(probability)
                elif word == partial_word:
                    sentence_end = self.word_processor.sentence_end_probability(word)
                    scores[" "] = scores.get(" ", 0.0) + max(0.2, 1.0 - sentence_end)
                    scores["."] = scores.get(".", 0.0) + max(0.05, sentence_end)
                continue

            if word:
                first = word[0]
                scores[first] = scores.get(first, 0.0) + float(probability)

        if partial_word:
            completions = self.word_processor.trie.get_completions(
                partial_word,
                top_k=max(4, int(self.word_processor.config.trie_max_completions)),
            )
            for rank, word in enumerate(completions, start=1):
                if len(word) > len(partial_word):
                    char = word[len(partial_word)]
                    scores[char] = scores.get(char, 0.0) + (1.0 / float(rank + 1))

        return scores


__all__ = ["DualLevelProcessor"]
