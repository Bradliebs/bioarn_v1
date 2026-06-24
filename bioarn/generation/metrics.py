"""Quality metrics for lightweight text generation experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re


_COMMON_WORDS = {
    "a",
    "and",
    "are",
    "as",
    "at",
    "back",
    "bird",
    "bright",
    "came",
    "cat",
    "day",
    "dog",
    "door",
    "down",
    "ever",
    "fire",
    "fox",
    "garden",
    "gold",
    "good",
    "green",
    "hall",
    "hello",
    "in",
    "key",
    "kind",
    "kitchen",
    "lantern",
    "letter",
    "light",
    "long",
    "mat",
    "moon",
    "morning",
    "night",
    "of",
    "on",
    "once",
    "open",
    "park",
    "promise",
    "quiet",
    "rain",
    "ran",
    "river",
    "road",
    "rose",
    "said",
    "sat",
    "small",
    "softly",
    "song",
    "stars",
    "table",
    "tea",
    "the",
    "there",
    "time",
    "town",
    "tree",
    "under",
    "upon",
    "warm",
    "was",
    "water",
    "while",
    "window",
    "wind",
    "words",
}


@dataclass
class QualityReport:
    """Summary of generation quality indicators."""

    word_likeness: float
    spacing_quality: float
    character_entropy: float
    bigram_naturalness: float
    longest_real_word: int
    repetition_score: float


class GenerationQualityMetrics:
    """Comprehensive text generation quality assessment."""

    _word_pattern = re.compile(r"[A-Za-z]+")

    def _reference_dictionary(self, reference_corpus: str) -> set[str]:
        words = {word.lower() for word in self._word_pattern.findall(reference_corpus) if len(word) >= 2}
        return words | _COMMON_WORDS

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = float(len(text))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(max(probability, 1e-12))
        return float(entropy)

    @staticmethod
    def _repetition_score(text: str, n: int = 3) -> float:
        if len(text) < n:
            return 0.0
        grams = [text[index : index + n] for index in range(len(text) - n + 1)]
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return float(repeated / max(1, len(grams)))

    @staticmethod
    def _spacing_quality(generated_texts: list[str], reference_corpus: str) -> float:
        if not generated_texts:
            return 0.0
        reference_words = [word for word in reference_corpus.split(" ") if word]
        generated_words = [word for text in generated_texts for word in text.split(" ") if word]
        if not generated_words:
            return 0.0
        ref_mean = sum(len(word) for word in reference_words) / max(1, len(reference_words))
        gen_mean = sum(len(word) for word in generated_words) / max(1, len(generated_words))
        gap = abs(gen_mean - ref_mean) / max(ref_mean, 1.0)
        space_density = sum(text.count(" ") for text in generated_texts) / max(1, sum(len(text) for text in generated_texts))
        return float(max(0.0, min(1.0, (1.0 - gap) * min(1.0, space_density * 8.0))))

    @staticmethod
    def _bigram_naturalness(generated_texts: list[str], reference_corpus: str) -> float:
        reference_bigrams = {
            reference_corpus[index : index + 2]
            for index in range(max(0, len(reference_corpus) - 1))
        }
        generated_bigrams = [
            text[index : index + 2]
            for text in generated_texts
            for index in range(max(0, len(text) - 1))
        ]
        if not generated_bigrams:
            return 0.0
        hits = sum(1 for bigram in generated_bigrams if bigram in reference_bigrams)
        return float(hits / len(generated_bigrams))

    def evaluate(self, generated_texts: list[str], reference_corpus: str) -> QualityReport:
        """Evaluate generated samples against a reference corpus."""

        combined = " ".join(generated_texts)
        dictionary = self._reference_dictionary(reference_corpus)
        words = [word.lower() for word in self._word_pattern.findall(combined)]
        real_words = [word for word in words if word in dictionary]
        longest_real = max((len(word) for word in real_words), default=0)

        return QualityReport(
            word_likeness=float(len(real_words) / max(1, len(words))),
            spacing_quality=self._spacing_quality(generated_texts, reference_corpus),
            character_entropy=self._entropy(combined),
            bigram_naturalness=self._bigram_naturalness(generated_texts, reference_corpus),
            longest_real_word=int(longest_real),
            repetition_score=self._repetition_score(combined),
        )


__all__ = ["GenerationQualityMetrics", "QualityReport"]
