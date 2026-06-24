"""Integrated multi-CCC perception, voting, and workspace orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bioarn.config import BioARNConfig
from bioarn.core.ccc import CCCOutput, CCCPool, CCCPoolOutput
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.associative_fabric import AssociationResult, AssociativeFabric, VoteResult
from bioarn.workspace.gnw import (
    BroadcastOutput,
    GlobalNeuronalWorkspace,
    StreamOfConsciousness,
    ThoughtOutput,
)


@dataclass
class PerceptionOutput:
    """Full output of one external perception step."""

    pool_output: CCCPoolOutput
    vote_result: VoteResult
    broadcast: BroadcastOutput
    associations: AssociationResult
    num_fired: int
    num_abstained: int
    is_novel: bool
    timestep: int


@dataclass
class RecognitionOutput:
    """High-level recognition summary."""

    concept_direction: torch.Tensor
    confidence: float
    abstained: bool
    num_hypotheses: int
    agreement: float


@dataclass
class BioARNCoreOutput:
    """Unified forward-pass output."""

    perception: PerceptionOutput
    thought: ThoughtOutput
    learned: bool
    system_stats: dict


@dataclass
class ContinualLearningResult:
    """Sequential-learning evaluation summary."""

    stage_accuracies: list[dict]
    forgetting_scores: list[float]
    mean_forgetting: float
    passed: bool


class BioARNCore(nn.Module):
    """Central multi-CCC perception and attention loop."""

    def __init__(self, config: BioARNConfig):
        super().__init__()
        torch.manual_seed(int(config.seed))

        self.config = config
        self.ccc_pool = CCCPool(config.ccc, config.margin_gate)
        self.fabric = AssociativeFabric(config.sdm, config.ccc)
        self.gnw = GlobalNeuronalWorkspace(config.gnw)
        self.stream = StreamOfConsciousness(self.gnw, config.gnw)
        self.timestep = 0
        self.last_perception: PerceptionOutput | None = None
        self.last_thought: ThoughtOutput = self._empty_thought()

    @staticmethod
    def _empty_associations() -> AssociationResult:
        return AssociationResult(directions=[], strengths=[], indices=[], temporal_order=[])

    @staticmethod
    def _empty_broadcast() -> BroadcastOutput:
        return BroadcastOutput(
            directions=[],
            activations=[],
            indices=[],
            num_occupied=0,
            total_broadcast_energy=0.0,
        )

    def _empty_thought(self) -> ThoughtOutput:
        return ThoughtOutput(
            broadcast=self._empty_broadcast(),
            new_entries=[],
            evicted=[],
            thought_chain_length=0,
            is_ruminating=False,
        )

    @staticmethod
    def _mean_confidence(output: CCCOutput) -> float:
        return float(output.confidence.reshape(-1).mean().item())

    def _run_pool(self, raw_input: torch.Tensor, *, allow_recruit: bool) -> CCCPoolOutput:
        if allow_recruit:
            return self.ccc_pool(raw_input, timestep=self.timestep)

        outputs: list[CCCOutput] = []
        for ccc in self.ccc_pool.cccs:
            if bool(ccc.is_committed.item()):
                outputs.append(ccc(raw_input, timestep=self.timestep))
            else:
                outputs.append(ccc.empty_output(raw_input))

        fired_indices = [index for index, output in enumerate(outputs) if output.fired]
        abstained_indices = [index for index, output in enumerate(outputs) if output.abstained]
        winner_confidences = (
            torch.stack(
                [
                    self.ccc_pool._confidence_score(outputs[index].confidence)
                    for index in fired_indices
                ]
            )
            if fired_indices
            else torch.empty(0, dtype=torch.float32)
        )

        return CCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=False,
            recruited_index=None,
            winner_confidences=winner_confidences,
        )

    def _active_cccs(
        self, pool_output: CCCPoolOutput
    ) -> list[tuple[int, torch.Tensor, float]]:
        active_cccs: list[tuple[int, torch.Tensor, float]] = []
        for index in pool_output.fired_indices:
            direction = self.ccc_pool.cccs[index].concept_direction.detach().clone()
            active_cccs.append((index, direction, self._mean_confidence(pool_output.outputs[index])))
        return active_cccs

    @staticmethod
    def _surviving_cccs(
        active_cccs: list[tuple[int, torch.Tensor, float]],
        inhibited_winners: list[tuple[int, float]],
    ) -> list[tuple[int, torch.Tensor, float]]:
        if not active_cccs:
            return []
        if not inhibited_winners:
            return active_cccs
        winning_indices = {index for index, _ in inhibited_winners}
        return [activation for activation in active_cccs if activation[0] in winning_indices]

    def _learning_occurred(self, perception: PerceptionOutput) -> bool:
        if perception.pool_output.recruited:
            return True
        return any(
            output.resonance is not None
            and bool(output.resonance.resonated.reshape(-1).any().item())
            for output in perception.pool_output.outputs
            if output.fired
        )

    @torch.no_grad()
    def _perceive_impl(
        self, raw_input: torch.Tensor, *, allow_recruit: bool
    ) -> PerceptionOutput:
        pool_output = self._run_pool(raw_input, allow_recruit=allow_recruit)
        active_cccs = self._active_cccs(pool_output)

        for index, direction, confidence in active_cccs:
            self.fabric.register_activation(index, direction, confidence, self.timestep)

        self.fabric.form_associations(self.timestep)

        inhibited_winners = (
            self.fabric.lateral_inhibition(active_cccs, k=max(1, len(active_cccs)))
            if active_cccs
            else []
        )
        surviving_cccs = self._surviving_cccs(active_cccs, inhibited_winners)
        vote_result = self.fabric.vote(surviving_cccs)
        self.last_thought = self.stream.think_step(surviving_cccs, timestep=self.timestep)

        associations = (
            self.fabric.retrieve_associates(vote_result.winning_direction, k=5)
            if surviving_cccs
            else self._empty_associations()
        )

        perception = PerceptionOutput(
            pool_output=pool_output,
            vote_result=vote_result,
            broadcast=self.last_thought.broadcast,
            associations=associations,
            num_fired=len(pool_output.fired_indices),
            num_abstained=len(pool_output.abstained_indices),
            is_novel=bool(pool_output.recruited),
            timestep=self.timestep,
        )
        self.last_perception = perception
        self.timestep += 1
        return perception

    @torch.no_grad()
    def perceive(self, raw_input: torch.Tensor) -> PerceptionOutput:
        """Run the full external perception loop, including CCC recruitment."""

        return self._perceive_impl(raw_input, allow_recruit=True)

    @torch.no_grad()
    def think(self, num_steps: int = 5) -> list[ThoughtOutput]:
        """Run internal association-driven reasoning without new external input."""

        thoughts: list[ThoughtOutput] = []
        for _ in range(max(0, num_steps)):
            primed_candidates: dict[int, tuple[torch.Tensor, float]] = {}

            for slot in list(self.gnw.slots):
                associates = self.fabric.retrieve_associates(
                    slot.direction,
                    k=max(1, self.config.gnw.capacity),
                )
                for index, direction, strength in zip(
                    associates.indices,
                    associates.directions,
                    associates.strengths,
                    strict=False,
                ):
                    if index == slot.ccc_index:
                        continue
                    confidence = float(max(0.0, min(1.0, strength)))
                    current = primed_candidates.get(index)
                    if current is None or confidence > current[1]:
                        primed_candidates[index] = (direction.detach().clone(), confidence)

            candidates = [
                (index, direction, confidence)
                for index, (direction, confidence) in primed_candidates.items()
            ]

            for index, direction, confidence in candidates:
                self.fabric.register_activation(index, direction, confidence, self.timestep)
            if candidates:
                self.fabric.form_associations(self.timestep)

            self.last_thought = self.stream.think_step(candidates, timestep=self.timestep)
            thoughts.append(self.last_thought)
            self.timestep += 1

        return thoughts

    @torch.no_grad()
    def recognize(self, raw_input: torch.Tensor) -> RecognitionOutput:
        """Recognize an input without recruiting a new CCC for novel patterns."""

        perception = self._perceive_impl(raw_input, allow_recruit=False)
        abstained = perception.num_fired == 0 or perception.vote_result.voter_count == 0
        concept_direction = perception.vote_result.winning_direction.detach().clone()
        if abstained:
            concept_direction = torch.zeros_like(concept_direction)

        return RecognitionOutput(
            concept_direction=concept_direction,
            confidence=float(perception.vote_result.confidence),
            abstained=abstained,
            num_hypotheses=perception.num_fired,
            agreement=float(perception.vote_result.agreement_score),
        )

    @torch.no_grad()
    def learn_from_perception(self, perception: PerceptionOutput, raw_input: torch.Tensor) -> None:
        """Apply any deferred learning that was not handled during perception."""

        if perception.pool_output.recruited or perception.pool_output.fired_indices:
            return

        recruit_index = self.ccc_pool._first_uncommitted_index()
        if recruit_index is None:
            return

        recruited_ccc = self.ccc_pool.cccs[recruit_index]
        f1_output = recruited_ccc.f1_encode(raw_input)
        recruited_ccc.learn_fast(raw_input, f1_output)
        direction = recruited_ccc.concept_direction.detach().clone()
        self.fabric.register_activation(recruit_index, direction, confidence=1.0, timestep=self.timestep)
        self.fabric.form_associations(self.timestep)
        self.gnw.inject(recruit_index, direction, priority=1.0)

    @torch.no_grad()
    def forward(self, raw_input: torch.Tensor, learn: bool = True) -> BioARNCoreOutput:
        """Run one full system step with optional continual learning."""

        perception = (
            self.perceive(raw_input)
            if learn
            else self._perceive_impl(raw_input, allow_recruit=False)
        )

        learned = False
        if learn:
            self.learn_from_perception(perception, raw_input)
            learned = self._learning_occurred(perception)

        return BioARNCoreOutput(
            perception=perception,
            thought=self.last_thought,
            learned=learned,
            system_stats=self.get_system_stats(),
        )

    def get_system_stats(self) -> dict:
        """Summarize pool, fabric, workspace, and system-level state."""

        pool_stats = self.ccc_pool.get_pool_stats()
        fabric_stats = self.fabric.get_stats()
        gnw_stats = self.gnw.get_stats()

        committed = int(pool_stats["num_committed"])
        fired = self.last_perception.num_fired if self.last_perception is not None else 0
        active_fraction = (fired / committed) if committed else 0.0
        sparsity = 1.0 - active_fraction if committed else 1.0

        return {
            "pool": pool_stats,
            "fabric": fabric_stats,
            "gnw": gnw_stats,
            "timesteps": int(self.timestep),
            "concepts_learned": committed,
            "sparsity": float(max(0.0, sparsity)),
        }


class ContinualLearningEvaluator:
    """Utility for sequential class learning and forgetting evaluation."""

    def __init__(self, core: BioARNCore):
        self.core = core
        self.class_prototypes: dict[int, torch.Tensor] = {}
        self.class_datasets: dict[int, torch.Tensor] = {}

    @staticmethod
    def _iter_samples(inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.dim() == 1:
            return [inputs.detach().clone()]
        return [sample.detach().clone() for sample in inputs]

    def _predict_label(self, concept_direction: torch.Tensor) -> int | None:
        if not self.class_prototypes:
            return None

        labels = list(self.class_prototypes.keys())
        prototypes = torch.stack([self.class_prototypes[label] for label in labels], dim=0)
        query = normalize(concept_direction.unsqueeze(0)).expand_as(prototypes)
        similarities = cosine_similarity(prototypes, query)
        return labels[int(torch.argmax(similarities).item())]

    def train_on_class(self, inputs: torch.Tensor, class_label: int):
        """Stream one class of inputs through the online learner."""

        samples = self._iter_samples(inputs)
        if not samples:
            return

        directions: list[torch.Tensor] = []
        for sample in samples:
            output = self.core.forward(sample, learn=True)
            if output.perception.vote_result.voter_count > 0:
                directions.append(output.perception.vote_result.winning_direction.detach().clone())

        if directions:
            prototype = normalize(torch.stack(directions, dim=0).mean(dim=0, keepdim=True)).squeeze(0)
            self.class_prototypes[class_label] = prototype
        self.class_datasets[class_label] = torch.stack(samples, dim=0)

    def evaluate_class(self, inputs: torch.Tensor, class_label: int) -> float:
        """Evaluate how often the learned concept maps back to the requested class."""

        if class_label not in self.class_prototypes:
            return 0.0

        samples = self._iter_samples(inputs)
        if not samples:
            return 0.0

        correct = 0
        for sample in samples:
            recognition = self.core.recognize(sample)
            predicted_label = (
                None
                if recognition.abstained
                else self._predict_label(recognition.concept_direction)
            )
            if predicted_label == class_label:
                correct += 1

        return correct / len(samples)

    def run_sequential_test(self, class_groups: list[list[torch.Tensor]]) -> ContinualLearningResult:
        """Train classes sequentially and measure forgetting after each stage."""

        stage_accuracies: list[dict[int, float]] = []
        reference_accuracies: dict[int, float] = {}
        seen_labels: list[int] = []
        next_label = 0

        for group in class_groups:
            for class_inputs in group:
                class_label = next_label
                next_label += 1
                seen_labels.append(class_label)
                self.train_on_class(class_inputs, class_label)

            stage_result: dict[int, float] = {}
            for class_label in seen_labels:
                accuracy = self.evaluate_class(self.class_datasets[class_label], class_label)
                stage_result[class_label] = accuracy
                reference_accuracies[class_label] = max(
                    reference_accuracies.get(class_label, 0.0),
                    accuracy,
                )
            stage_accuracies.append(stage_result)

        forgetting_scores = [
            max(0.0, reference_accuracies[label] - stage_accuracies[-1].get(label, 0.0))
            for label in seen_labels
        ] if stage_accuracies else []
        mean_forgetting = (
            sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0.0
        )

        return ContinualLearningResult(
            stage_accuracies=stage_accuracies,
            forgetting_scores=forgetting_scores,
            mean_forgetting=float(mean_forgetting),
            passed=bool(mean_forgetting < 0.05),
        )


__all__ = [
    "BioARNCore",
    "BioARNCoreOutput",
    "ContinualLearningEvaluator",
    "ContinualLearningResult",
    "PerceptionOutput",
    "RecognitionOutput",
]
