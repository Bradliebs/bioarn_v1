"""Voting-based ensembles over diverse Bio-ARN CCC experts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from bioarn.core.ccc import CCCPool, CCCPoolOutput
from bioarn.core.math_utils import normalize
from bioarn.ensemble.boosting import HebbianBoosting
from bioarn.ensemble.config import EnsembleConfig
from bioarn.scaling import BatchedCCCPool, PoolInferenceSummary


@dataclass
class ExpertPrediction:
    name: str
    predicted_class: int
    confidence: float
    abstained: bool
    vote_weight: float = 0.0


@dataclass
class EnsembleResult:
    predicted_class: int
    confidence: float
    abstained: bool
    agreement: float
    vote_totals: dict[int, float]
    expert_results: list[ExpertPrediction]
    abstention_fraction: float


@dataclass
class _PrototypeBank:
    prototypes: dict[int, torch.Tensor] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)

    def update(self, label: int, concept_direction: torch.Tensor) -> None:
        if float(concept_direction.norm().item()) <= 1e-8:
            return
        normalized = normalize(concept_direction.reshape(1, -1)).squeeze(0)
        count = self.counts.get(int(label), 0) + 1
        if int(label) not in self.prototypes:
            self.prototypes[int(label)] = normalized.detach().clone()
        else:
            updated = ((self.prototypes[int(label)] * self.counts[int(label)]) + normalized) / count
            self.prototypes[int(label)] = normalize(updated.reshape(1, -1)).squeeze(0)
        self.counts[int(label)] = count

    def predict(self, concept_direction: torch.Tensor) -> int | None:
        if not self.prototypes or float(concept_direction.norm().item()) <= 1e-8:
            return None
        labels = list(self.prototypes.keys())
        stacked = torch.stack([self.prototypes[label].to(concept_direction) for label in labels], dim=0)
        query = normalize(concept_direction.reshape(1, -1)).expand_as(stacked)
        similarities = torch.sum(stacked * query, dim=-1)
        return int(labels[int(torch.argmax(similarities).item())])


@dataclass
class _ExpertState:
    name: str
    pool: Any
    preprocessor: Any | None = None
    label_bank: _PrototypeBank = field(default_factory=_PrototypeBank)
    ccc_label_counts: defaultdict[int, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    total_predictions: int = 0
    correct_predictions: int = 0
    timestep: int = 0


class EnsemblePool:
    """Multiple CCC pools with different specializations that vote on classification."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.experts: list[_ExpertState] = []
        self.boosting = HebbianBoosting(num_experts=1, num_classes=1) if config.use_boosting else None
        self.last_result: EnsembleResult | None = None
        for expert_config in config.expert_configs or []:
            self.add_expert(expert_config.name, expert_config.pool, expert_config.preprocessor)

    def add_expert(self, name: str, pool: Any, preprocessor: Any | None = None) -> None:
        self.experts.append(_ExpertState(name=str(name), pool=pool, preprocessor=preprocessor))
        if self.boosting is not None:
            self.boosting.ensure_capacity(len(self.experts), self.boosting.num_classes)

    @staticmethod
    def _ensure_vector(image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32).reshape(-1)

    @staticmethod
    def _prepare_preprocessor_input(image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32).reshape(-1)

    def _fit_preprocessor(self, state: _ExpertState, image: torch.Tensor) -> None:
        if state.preprocessor is None:
            return
        sample = self._prepare_preprocessor_input(image)
        if hasattr(state.preprocessor, "partial_fit"):
            state.preprocessor.partial_fit(sample)
        elif hasattr(state.preprocessor, "learn"):
            state.preprocessor.learn(sample)
        elif hasattr(state.preprocessor, "fit") and not bool(getattr(state.preprocessor, "is_fitted", True)):
            state.preprocessor.fit(sample.unsqueeze(0))

    def _transform(self, state: _ExpertState, image: torch.Tensor) -> torch.Tensor:
        if state.preprocessor is None:
            return self._ensure_vector(image)
        transformed = state.preprocessor.transform(self._prepare_preprocessor_input(image))
        return transformed.to(torch.float32).reshape(-1)

    @staticmethod
    def _coerce_prediction(name: str, prediction: Any) -> ExpertPrediction:
        if isinstance(prediction, ExpertPrediction):
            return prediction
        if isinstance(prediction, dict):
            predicted_class = int(prediction.get("predicted_class", prediction.get("label", -1)))
            confidence = float(prediction.get("confidence", 0.0))
            abstained = bool(prediction.get("abstained", predicted_class < 0))
            return ExpertPrediction(name=name, predicted_class=predicted_class, confidence=confidence, abstained=abstained)
        if isinstance(prediction, tuple):
            if len(prediction) == 3:
                predicted_class, confidence, abstained = prediction
            elif len(prediction) == 2:
                predicted_class, confidence = prediction
                abstained = int(predicted_class) < 0
            else:
                raise ValueError("Expert classify tuples must have length 2 or 3.")
            return ExpertPrediction(
                name=name,
                predicted_class=int(predicted_class),
                confidence=float(confidence),
                abstained=bool(abstained),
            )
        if prediction is None:
            return ExpertPrediction(name=name, predicted_class=-1, confidence=0.0, abstained=True)
        return ExpertPrediction(name=name, predicted_class=int(prediction), confidence=1.0, abstained=int(prediction) < 0)

    @staticmethod
    def _predict_label_from_cccs(
        state: _ExpertState,
        concept_direction: torch.Tensor,
        fired_indices: list[int],
    ) -> int | None:
        ccc_votes: defaultdict[int, float] = defaultdict(float)
        for index in fired_indices:
            counts = state.ccc_label_counts.get(int(index))
            if not counts:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / max(sum(counts.values()), 1)
            ccc_votes[int(dominant_label)] += purity
        if ccc_votes:
            return int(max(ccc_votes.items(), key=lambda item: item[1])[0])
        return state.label_bank.predict(concept_direction)

    @staticmethod
    def _pool_concept_from_directions(
        directions: torch.Tensor,
        fired_indices: list[int],
        winner_confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not fired_indices:
            width = int(directions.shape[-1]) if directions.ndim == 2 else 0
            return torch.zeros(width, dtype=torch.float32), 0.0
        selected = directions.index_select(0, torch.tensor(fired_indices, dtype=torch.long, device=directions.device))
        weights = winner_confidences.to(selected).reshape(-1, 1)
        concept = normalize((selected * weights).sum(dim=0, keepdim=True)).squeeze(0)
        return concept, float(weights.max().item())

    @staticmethod
    def _pool_concept_from_ccc_outputs(
        pool: CCCPool,
        pool_output: CCCPoolOutput,
    ) -> tuple[torch.Tensor, float]:
        if not pool_output.fired_indices:
            return torch.zeros(pool.config.concept_dim, dtype=torch.float32), 0.0
        directions = torch.stack(
            [pool.cccs[index].concept_direction.detach().clone() for index in pool_output.fired_indices],
            dim=0,
        )
        weights = pool_output.winner_confidences.to(directions).reshape(-1, 1)
        concept = normalize((directions * weights).sum(dim=0, keepdim=True)).squeeze(0)
        return concept, float(weights.max().item())

    @staticmethod
    def _infer_ccc_pool(pool: CCCPool, tensor: torch.Tensor) -> tuple[list[int], torch.Tensor, float]:
        fired_indices: list[int] = []
        confidences: list[float] = []
        for index, ccc in enumerate(pool.cccs):
            if not bool(ccc.is_committed.item()):
                continue
            f1_output = ccc.f1_encode(tensor)
            f2_activation = ccc.f2_activate(f1_output)
            gate_output = ccc.margin_gate(f2_activation, ccc.concept_direction)
            if bool(gate_output.fired.any().item()):
                fired_indices.append(index)
                confidences.append(float(gate_output.confidence.reshape(-1).mean().item()))
        if not fired_indices:
            return [], torch.empty(0, dtype=torch.float32), 0.0
        winner_confidences = torch.tensor(confidences, dtype=torch.float32)
        concept, confidence = EnsemblePool._pool_concept_from_directions(
            torch.stack([ccc.concept_direction for ccc in pool.cccs], dim=0),
            fired_indices,
            winner_confidences,
        )
        return fired_indices, concept, confidence

    def _classify_ccc_like(self, state: _ExpertState, tensor: torch.Tensor) -> ExpertPrediction:
        pool = state.pool
        if isinstance(pool, BatchedCCCPool):
            summary = self._infer_batched_pool(pool, tensor)
            if not summary.fired_indices:
                return ExpertPrediction(name=state.name, predicted_class=-1, confidence=0.0, abstained=True)
            concept, confidence = self._pool_concept_from_directions(
                pool.concept_directions,
                list(summary.fired_indices),
                summary.winner_confidences,
            )
            predicted = self._predict_label_from_cccs(state, concept, list(summary.fired_indices))
            if predicted is None:
                return ExpertPrediction(name=state.name, predicted_class=-1, confidence=confidence, abstained=True)
            return ExpertPrediction(name=state.name, predicted_class=int(predicted), confidence=confidence, abstained=False)

        if isinstance(pool, CCCPool):
            fired_indices, concept, confidence = self._infer_ccc_pool(pool, tensor)
            if not fired_indices:
                return ExpertPrediction(name=state.name, predicted_class=-1, confidence=0.0, abstained=True)
            predicted = self._predict_label_from_cccs(state, concept, fired_indices)
            if predicted is None:
                return ExpertPrediction(name=state.name, predicted_class=-1, confidence=confidence, abstained=True)
            return ExpertPrediction(name=state.name, predicted_class=int(predicted), confidence=confidence, abstained=False)

        raise TypeError("Unsupported CCC-style pool.")

    @staticmethod
    def _infer_batched_pool(pool: BatchedCCCPool, tensor: torch.Tensor) -> PoolInferenceSummary:
        raw_batch, _ = pool._ensure_batch(tensor)  # noqa: SLF001
        if int(pool.committed_mask.sum().item()) == 0:
            return PoolInferenceSummary(
                fired_indices=[],
                winner_confidences=torch.empty(0, dtype=torch.float32, device=raw_batch.device),
                recruited=False,
                recruited_index=None,
                num_fired=0,
                num_abstained=int(pool.config.max_pool_size),
                sparsity=0.0,
                mean_confidence=0.0,
            )

        committed_indices = pool.committed_mask.nonzero(as_tuple=False).reshape(-1)
        shared_projection = raw_batch.to(pool.f1_weights.dtype) @ pool.f1_weights[0].transpose(0, 1)
        shared_projection = shared_projection + pool.f1_bias[0]
        shared_activated = torch.relu(shared_projection)
        top_k = min(pool.config.f1_top_k, shared_activated.shape[-1])
        top_values, top_indices = torch.topk(shared_activated, k=top_k, dim=-1)
        shared_f1 = torch.zeros_like(shared_activated).scatter(-1, top_indices, top_values)

        committed_weights = pool.f2_weights.index_select(0, committed_indices)
        committed_f2 = torch.matmul(committed_weights, shared_f1.transpose(0, 1)).transpose(1, 2)
        directions = normalize(pool.concept_directions.index_select(0, committed_indices)).unsqueeze(1)
        confidence = (normalize(committed_f2).to(directions.dtype) * directions).sum(dim=-1)
        fired = confidence > pool.theta_margin.index_select(0, committed_indices).unsqueeze(-1)
        fired_mask = fired.any(dim=-1)
        fired_indices = committed_indices[fired_mask].tolist()
        winner_confidences = (
            confidence[fired_mask].mean(dim=-1)
            if fired_indices
            else torch.empty(0, dtype=torch.float32, device=raw_batch.device)
        )
        return PoolInferenceSummary(
            fired_indices=[int(index) for index in fired_indices],
            winner_confidences=winner_confidences.detach().clone(),
            recruited=False,
            recruited_index=None,
            num_fired=len(fired_indices),
            num_abstained=int(pool.config.max_pool_size) - len(fired_indices),
            sparsity=float(len(fired_indices) / max(int(pool.config.max_pool_size), 1)),
            mean_confidence=float(winner_confidences.mean().item()) if fired_indices else 0.0,
        )

    def _learn_ccc_like(self, state: _ExpertState, tensor: torch.Tensor, label: int | None) -> None:
        pool = state.pool
        if isinstance(pool, BatchedCCCPool):
            summary = pool.fast_infer(tensor, timestep=state.timestep, allow_recruit=True)
            state.timestep += 1
            if not summary.fired_indices:
                return
            concept, confidence = self._pool_concept_from_directions(
                pool.concept_directions,
                list(summary.fired_indices),
                summary.winner_confidences,
            )
            if label is not None:
                state.label_bank.update(int(label), concept)
                for index in summary.fired_indices:
                    state.ccc_label_counts[int(index)][int(label)] += 1
            del confidence
            return

        if isinstance(pool, CCCPool):
            pool_output = pool(tensor, timestep=state.timestep)
            state.timestep += 1
            if not pool_output.fired_indices:
                return
            concept, _ = self._pool_concept_from_ccc_outputs(pool, pool_output)
            if label is not None:
                state.label_bank.update(int(label), concept)
                for index in pool_output.fired_indices:
                    state.ccc_label_counts[int(index)][int(label)] += 1
            return

        raise TypeError("Unsupported CCC-style pool.")

    def _classify_expert(self, state: _ExpertState, image: torch.Tensor) -> ExpertPrediction:
        if isinstance(state.pool, (BatchedCCCPool, CCCPool)):
            return self._classify_ccc_like(state, self._transform(state, image))

        prepared = self._transform(state, image) if state.preprocessor is not None else image.to(torch.float32)
        if hasattr(state.pool, "classify"):
            return self._coerce_prediction(state.name, state.pool.classify(prepared))
        raise TypeError(f"Expert '{state.name}' does not expose a classify method.")

    def _learn_expert(self, state: _ExpertState, image: torch.Tensor, label: int | None) -> None:
        self._fit_preprocessor(state, image)
        if isinstance(state.pool, (BatchedCCCPool, CCCPool)):
            self._learn_ccc_like(state, self._transform(state, image), label)
            return

        prepared = self._transform(state, image) if state.preprocessor is not None else image.to(torch.float32)
        if hasattr(state.pool, "learn"):
            try:
                state.pool.learn(prepared, label=label)
            except TypeError:
                if label is None:
                    state.pool.learn(prepared)
                else:
                    state.pool.learn(prepared, int(label))
            return
        if hasattr(state.pool, "fit"):
            state.pool.fit(prepared.unsqueeze(0))

    def _expert_reliability(self, state: _ExpertState) -> float:
        if state.total_predictions == 0:
            return 1.0
        accuracy = state.correct_predictions / max(state.total_predictions, 1)
        return max(0.25, accuracy / max(1.0 - accuracy, 0.05))

    @staticmethod
    def _positive_confidence(confidence: float) -> float:
        scaled = min(1.0, max((float(confidence) - 0.5) * 2.0, 0.0))
        return scaled * scaled

    def _vote_capacity(self, expert_index: int, state: _ExpertState, predicted_class: int) -> float:
        if predicted_class < 0:
            return 0.0
        if self.config.voting_method == "majority":
            base_weight = 1.0
        elif self.config.voting_method == "confidence":
            base_weight = 1.0
        else:
            base_weight = self._expert_reliability(state)
        return base_weight * self._class_weight(expert_index, predicted_class)

    def _class_weight(self, expert_index: int, predicted_class: int) -> float:
        if self.boosting is None or predicted_class < 0:
            return 1.0
        self.boosting.ensure_capacity(len(self.experts), predicted_class + 1)
        return float(self.boosting.weights[expert_index, predicted_class].item())

    def _vote_weight(self, expert_index: int, state: _ExpertState, prediction: ExpertPrediction) -> float:
        if prediction.abstained or prediction.predicted_class < 0:
            return 0.0
        if self.config.voting_method == "majority":
            base_weight = 1.0
        elif self.config.voting_method == "confidence":
            base_weight = self._positive_confidence(prediction.confidence)
        else:
            base_weight = self._positive_confidence(prediction.confidence) * self._expert_reliability(state)
        return base_weight * self._class_weight(expert_index, prediction.predicted_class)

    @torch.no_grad()
    def classify(self, image: torch.Tensor) -> EnsembleResult:
        expert_results: list[ExpertPrediction] = []
        for expert_index, state in enumerate(self.experts):
            prediction = self._classify_expert(state, image)
            prediction.vote_weight = self._vote_weight(expert_index, state, prediction)
            expert_results.append(prediction)

        abstained_count = sum(int(result.abstained) for result in expert_results)
        abstention_fraction = abstained_count / max(len(expert_results), 1)
        if not expert_results or abstained_count > (len(expert_results) / 2):
            result = EnsembleResult(
                predicted_class=-1,
                confidence=0.0,
                abstained=True,
                agreement=0.0,
                vote_totals={},
                expert_results=expert_results,
                abstention_fraction=abstention_fraction,
            )
            self.last_result = result
            return result

        vote_totals: defaultdict[int, float] = defaultdict(float)
        vote_counts: Counter[int] = Counter()
        confidence_sums: defaultdict[int, float] = defaultdict(float)
        for prediction in expert_results:
            if prediction.abstained or prediction.predicted_class < 0:
                continue
            vote_totals[prediction.predicted_class] += float(prediction.vote_weight)
            vote_counts[prediction.predicted_class] += 1
            confidence_sums[prediction.predicted_class] += float(prediction.confidence)

        if not vote_totals:
            result = EnsembleResult(
                predicted_class=-1,
                confidence=0.0,
                abstained=True,
                agreement=0.0,
                vote_totals={},
                expert_results=expert_results,
                abstention_fraction=abstention_fraction,
            )
            self.last_result = result
            return result

        predicted_class = max(
            vote_totals,
            key=lambda label: (
                vote_totals[label],
                vote_counts[label],
                confidence_sums[label],
                -int(label),
            ),
        )
        active_count = sum(int(not result.abstained and result.predicted_class >= 0) for result in expert_results)
        agreement = vote_counts[predicted_class] / max(active_count, 1)
        support_fraction = vote_counts[predicted_class] / max(len(expert_results), 1)
        total_vote_weight = sum(
            float(prediction.vote_weight)
            for prediction in expert_results
            if not prediction.abstained and prediction.predicted_class >= 0
        )
        support_mass = float(vote_totals[predicted_class])
        support_capacity = sum(
            self._vote_capacity(expert_index, state, int(predicted_class))
            for expert_index, state in enumerate(self.experts)
        )
        normalized_support = min(1.0, support_mass / max(support_capacity, 1e-6))
        consensus = support_mass / max(total_vote_weight, 1e-6) if total_vote_weight > 0.0 else 0.0
        confidence = normalized_support * (0.75 + 0.25 * consensus)
        if support_fraction < self.config.abstention_threshold:
            confidence *= max(support_fraction / max(self.config.abstention_threshold, 1e-6), 0.1)

        result = EnsembleResult(
            predicted_class=int(predicted_class),
            confidence=float(confidence),
            abstained=False,
            agreement=float(agreement),
            vote_totals={int(label): float(weight) for label, weight in vote_totals.items()},
            expert_results=expert_results,
            abstention_fraction=abstention_fraction,
        )
        self.last_result = result
        return result

    def update_with_feedback(self, result: EnsembleResult, true_label: int) -> None:
        expert_predictions: list[int | None] = []
        for state, prediction in zip(self.experts, result.expert_results, strict=False):
            state.total_predictions += 1
            if not prediction.abstained and prediction.predicted_class == int(true_label):
                state.correct_predictions += 1
            expert_predictions.append(None if prediction.abstained else int(prediction.predicted_class))

        if self.boosting is not None:
            self.boosting.ensure_capacity(len(self.experts), int(true_label) + 1)
            self.boosting.update_weights(expert_predictions, int(true_label))

    @torch.no_grad()
    def learn(self, image: torch.Tensor, label: int | None = None) -> None:
        if label is not None:
            pre_update = self.classify(image)
            self.update_with_feedback(pre_update, int(label))
        for state in self.experts:
            self._learn_expert(state, image, label)

    def get_agreement(self) -> float:
        return 0.0 if self.last_result is None else float(self.last_result.agreement)

    def get_expert_accuracies(self) -> dict[str, float]:
        return {
            state.name: (
                state.correct_predictions / state.total_predictions if state.total_predictions else 0.0
            )
            for state in self.experts
        }


__all__ = ["EnsemblePool", "EnsembleResult", "ExpertPrediction"]
