"""Decoding strategies for Bio-ARN text generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class GenerationResult:
    """One generated candidate sequence."""

    text: str
    token_ids: list[int]
    score: float
    average_confidence: float
    method: str


class RepetitionPenalty:
    """Penalize repeated n-grams in generation."""

    def __init__(self, penalty: float = 1.2, window: int = 20) -> None:
        if penalty < 1.0:
            raise ValueError("penalty must be at least 1.0.")
        if window <= 0:
            raise ValueError("window must be positive.")
        self.penalty = float(penalty)
        self.window = int(window)

    def apply(self, token_scores: torch.Tensor, generated_so_far) -> torch.Tensor:
        """Reduce scores for recently generated tokens."""

        adjusted = token_scores.clone()
        if adjusted.numel() == 0:
            return adjusted

        if isinstance(generated_so_far, dict):
            history = [int(token) for token in generated_so_far.get("history", [])]
            candidate_ids = [int(token) for token in generated_so_far.get("candidate_ids", range(adjusted.numel()))]
        else:
            history = [int(token) for token in generated_so_far]
            candidate_ids = list(range(adjusted.numel()))

        recent = history[-self.window :]
        if not recent:
            return adjusted

        recent_counts: dict[int, int] = {}
        for token_id in recent:
            recent_counts[token_id] = recent_counts.get(token_id, 0) + 1

        for index, token_id in enumerate(candidate_ids):
            repeats = recent_counts.get(int(token_id), 0)
            if repeats:
                adjusted[index] = adjusted[index] / (self.penalty ** repeats)
            if len(history) >= 2 and len(set(history[-2:] + [int(token_id)])) == 1:
                adjusted[index] = adjusted[index] / (self.penalty * 1.1)
            if len(history) >= 3 and history[-2:] == [int(token_id), int(token_id)]:
                adjusted[index] = adjusted[index] / (self.penalty * 1.2)
        return adjusted


class BeamSearchDecoder:
    """Beam search for spike-based text generation."""

    def __init__(self, beam_width: int = 5, length_penalty: float = 0.6) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        self.beam_width = int(beam_width)
        self.length_penalty = float(length_penalty)

    @staticmethod
    def _length_norm(length: int, penalty: float) -> float:
        return ((5.0 + max(1, length)) / 6.0) ** penalty

    def _finalize(self, system, prompt_ids: list[int], token_ids: list[int], score: float, confidences: list[float], method: str) -> GenerationResult:
        generated = token_ids[len(prompt_ids) :]
        text = system._decode_token(generated)
        avg_confidence = float(sum(confidences) / max(1, len(confidences)))
        return GenerationResult(
            text=text,
            token_ids=generated,
            score=float(score),
            average_confidence=avg_confidence,
            method=method,
        )

    def _normalize_prompt(self, system, prompt_spikes: Any) -> list[int]:
        return system._normalize_generation_input(prompt_spikes)

    def decode(self, system, prompt_spikes, max_tokens: int = 100) -> list[GenerationResult]:
        """Generate multiple candidate sequences and return best."""

        prompt_ids = self._normalize_prompt(system, prompt_spikes)
        repetition_penalty = getattr(system, "repetition_penalty", None)
        beams: list[tuple[list[int], float, list[float], bool]] = [(prompt_ids, 0.0, [], False)]
        finished: list[GenerationResult] = []

        for _ in range(max(0, max_tokens)):
            expanded: list[tuple[list[int], float, list[float], bool]] = []
            for token_ids, score, confidences, done in beams:
                if done:
                    expanded.append((token_ids, score, confidences, done))
                    continue

                prediction = system._predict_from_tokens(
                    token_ids,
                    temperature=system.config.temperature,
                    repetition_penalty=repetition_penalty,
                )
                if not prediction.candidate_ids:
                    expanded.append((token_ids, score, confidences, True))
                    continue

                top_k = min(self.beam_width, len(prediction.candidate_ids))
                top_values, top_indices = torch.topk(prediction.probabilities, k=top_k)
                for probability, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
                    next_token = int(prediction.candidate_ids[index])
                    local_confidence = max(
                        1e-6,
                        float(probability)
                        * max(0.05, float(prediction.sdm_confidence))
                        * max(0.05, float(prediction.margin_confidence)),
                    )
                    next_ids = token_ids + [next_token]
                    norm = self._length_norm(len(next_ids) - len(prompt_ids), self.length_penalty)
                    next_score = score + (math.log(local_confidence) / norm)
                    next_confidences = confidences + [float(probability)]
                    should_stop = (
                        next_token in system.generation_stop_token_ids
                        or system._loop_detected(next_ids[len(prompt_ids) :])
                    )
                    expanded.append((next_ids, next_score, next_confidences, should_stop))

            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[: self.beam_width]
            if all(done for _, _, _, done in beams):
                break

        for token_ids, score, confidences, _ in beams:
            finished.append(self._finalize(system, prompt_ids, token_ids, score, confidences, "beam"))
        finished.sort(key=lambda item: item.score, reverse=True)
        return finished[: self.beam_width]

    def greedy_decode(self, system, prompt_spikes, max_tokens: int = 100) -> GenerationResult:
        """Greedy argmax decoding."""

        prompt_ids = self._normalize_prompt(system, prompt_spikes)
        token_ids = prompt_ids.copy()
        confidences: list[float] = []

        for _ in range(max(0, max_tokens)):
            prediction = system._predict_from_tokens(
                token_ids,
                temperature=max(0.2, float(system.config.temperature) * 0.75),
                repetition_penalty=getattr(system, "repetition_penalty", None),
            )
            if not prediction.candidate_ids:
                break
            best_index = int(torch.argmax(prediction.probabilities).item())
            next_token = int(prediction.candidate_ids[best_index])
            if next_token in system.generation_stop_token_ids:
                break
            token_ids.append(next_token)
            confidences.append(float(prediction.probabilities[best_index].item()))
            if system._loop_detected(token_ids[len(prompt_ids) :]):
                break

        score = float(sum(math.log(max(value, 1e-6)) for value in confidences))
        return self._finalize(system, prompt_ids, token_ids, score, confidences, "greedy")

    def sample_decode(self, system, prompt_spikes, max_tokens: int = 100, temperature: float = 1.0) -> GenerationResult:
        """Random sampling with temperature."""

        prompt_ids = self._normalize_prompt(system, prompt_spikes)
        token_ids = prompt_ids.copy()
        confidences: list[float] = []

        for _ in range(max(0, max_tokens)):
            prediction = system._predict_from_tokens(
                token_ids,
                temperature=temperature,
                repetition_penalty=getattr(system, "repetition_penalty", None),
            )
            if not prediction.candidate_ids:
                break
            next_index = int(torch.multinomial(prediction.probabilities, num_samples=1).item())
            next_token = int(prediction.candidate_ids[next_index])
            if next_token in system.generation_stop_token_ids:
                break
            token_ids.append(next_token)
            confidences.append(float(prediction.probabilities[next_index].item()))
            if system._loop_detected(token_ids[len(prompt_ids) :]):
                break

        score = float(sum(math.log(max(value, 1e-6)) for value in confidences))
        return self._finalize(system, prompt_ids, token_ids, score, confidences, "sample")

    def top_k_decode(self, system, prompt_spikes, max_tokens: int = 100, k: int = 10) -> GenerationResult:
        """Sample from the top-k tokens only."""

        prompt_ids = self._normalize_prompt(system, prompt_spikes)
        token_ids = prompt_ids.copy()
        confidences: list[float] = []

        for _ in range(max(0, max_tokens)):
            prediction = system._predict_from_tokens(
                token_ids,
                temperature=system.config.temperature,
                repetition_penalty=getattr(system, "repetition_penalty", None),
            )
            if not prediction.candidate_ids:
                break
            top_k = min(max(1, int(k)), len(prediction.candidate_ids))
            values, indices = torch.topk(prediction.probabilities, k=top_k)
            next_local = int(torch.multinomial(values / values.sum(), num_samples=1).item())
            next_index = int(indices[next_local].item())
            next_token = int(prediction.candidate_ids[next_index])
            if next_token in system.generation_stop_token_ids:
                break
            token_ids.append(next_token)
            confidences.append(float(prediction.probabilities[next_index].item()))
            if system._loop_detected(token_ids[len(prompt_ids) :]):
                break

        score = float(sum(math.log(max(value, 1e-6)) for value in confidences))
        return self._finalize(system, prompt_ids, token_ids, score, confidences, "top-k")

    def top_p_decode(self, system, prompt_spikes, max_tokens: int = 100, p: float = 0.9) -> GenerationResult:
        """Nucleus sampling."""

        prompt_ids = self._normalize_prompt(system, prompt_spikes)
        token_ids = prompt_ids.copy()
        confidences: list[float] = []
        nucleus_threshold = min(max(float(p), 0.05), 1.0)

        for _ in range(max(0, max_tokens)):
            prediction = system._predict_from_tokens(
                token_ids,
                temperature=system.config.temperature,
                repetition_penalty=getattr(system, "repetition_penalty", None),
            )
            if not prediction.candidate_ids:
                break
            values, indices = torch.sort(prediction.probabilities, descending=True)
            cumulative = torch.cumsum(values, dim=0)
            cutoff = int(torch.nonzero(cumulative >= nucleus_threshold, as_tuple=False)[0].item()) + 1
            nucleus_values = values[:cutoff]
            nucleus_indices = indices[:cutoff]
            next_local = int(torch.multinomial(nucleus_values / nucleus_values.sum(), num_samples=1).item())
            next_index = int(nucleus_indices[next_local].item())
            next_token = int(prediction.candidate_ids[next_index])
            if next_token in system.generation_stop_token_ids:
                break
            token_ids.append(next_token)
            confidences.append(float(prediction.probabilities[next_index].item()))
            if system._loop_detected(token_ids[len(prompt_ids) :]):
                break

        score = float(sum(math.log(max(value, 1e-6)) for value in confidences))
        return self._finalize(system, prompt_ids, token_ids, score, confidences, "top-p")


__all__ = ["BeamSearchDecoder", "GenerationResult", "RepetitionPenalty"]
