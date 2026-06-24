"""Dual character+word processing for constrained text generation."""

from __future__ import annotations

import math
import re

import torch

from .word_level import WordLevelProcessor

_WORD_PATTERN = re.compile(r"[A-Za-z']+")


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
            candidates = self._char_candidates(context_words, partial)
            next_index = min(len(partial), len(target_word) - 1)
            expected = target_word[next_index] if target_word else ""
            if expected:
                candidates.append((expected, 1.0 + (0.5 / max(0.2, float(temperature)))))
            constrained = self.word_processor.constrain_generation(candidates, partial)
            if not constrained:
                break
            next_char = constrained[0][0] if isinstance(constrained[0], tuple) else constrained[0]
            if not next_char or not next_char.isalpha():
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


__all__ = ["DualLevelProcessor"]
