"""Associative fabric linking concept cell clusters through sparse memory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from bioarn.config import CCCConfig, SDMConfig
from bioarn.core.ccc import CCCPool, CCCPoolOutput
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.sdm import SparseDistributedMemory, TemporalAssociator


@dataclass
class AssociationResult:
    """Top associated concepts retrieved from the fabric."""

    directions: list[torch.Tensor]
    strengths: list[float]
    indices: list[int]
    temporal_order: list[bool]


@dataclass
class VoteResult:
    """Consensus concept formed by distributed CCC voting."""

    winning_direction: torch.Tensor
    confidence: float
    agreement_score: float
    voter_count: int
    voter_indices: list[int]


@dataclass
class FabricPoolOutput:
    """Unified output for a pool connected to the associative fabric."""

    pool_output: CCCPoolOutput
    active_cccs: list[tuple[int, torch.Tensor, float]]
    inhibited_winners: list[tuple[int, float]]
    consensus: VoteResult
    associates: AssociationResult


@dataclass
class _ActivationRecord:
    ccc_index: int
    direction: torch.Tensor
    confidence: float
    timestep: int
    address: torch.Tensor


class AssociativeFabric(nn.Module):
    """Sparse associative memory that links CCCs by content and time."""

    def __init__(self, sdm_config: SDMConfig, ccc_config: CCCConfig):
        super().__init__()
        self.sdm = SparseDistributedMemory(sdm_config)
        self.temporal_associator = TemporalAssociator(self.sdm, sdm_config)
        self.sdm_config = sdm_config
        self.ccc_config = ccc_config

        self.temporal_window = max(1, int(getattr(sdm_config, "stdp_window", 5)))
        self.association_strength: dict[tuple[int, int], float] = {}
        self.association_temporal: dict[tuple[int, int], bool] = {}
        self.activation_history: list[_ActivationRecord] = []
        self.concept_directions: dict[int, torch.Tensor] = {}
        self._last_decay_timestep: int | None = None
        self._strength_epsilon = 1e-6

    @staticmethod
    def _empty_association_result() -> AssociationResult:
        return AssociationResult(directions=[], strengths=[], indices=[], temporal_order=[])

    def _empty_vote_result(self, device: torch.device) -> VoteResult:
        return VoteResult(
            winning_direction=torch.zeros(
                self.ccc_config.concept_dim,
                device=device,
                dtype=torch.float32,
            ),
            confidence=0.0,
            agreement_score=0.0,
            voter_count=0,
            voter_indices=[],
        )

    @torch.no_grad()
    def _decay_associations(self, timestep: int) -> None:
        if self._last_decay_timestep is None:
            self._last_decay_timestep = int(timestep)
            return

        delta_t = max(0, int(timestep) - self._last_decay_timestep)
        if delta_t == 0 or not self.association_strength:
            self._last_decay_timestep = int(timestep)
            return

        decay = float(self.sdm_config.decay_rate) ** delta_t
        stale_keys: list[tuple[int, int]] = []
        for key, strength in self.association_strength.items():
            decayed = strength * decay
            if decayed <= self._strength_epsilon:
                stale_keys.append(key)
            else:
                self.association_strength[key] = decayed

        for key in stale_keys:
            self.association_strength.pop(key, None)
            self.association_temporal.pop(key, None)

        self._last_decay_timestep = int(timestep)

    def _prune_activation_history(self, timestep: int) -> None:
        cutoff = int(timestep) - self.temporal_window
        self.activation_history = [
            activation
            for activation in self.activation_history
            if activation.timestep >= cutoff
        ]

    @staticmethod
    def _normalize_direction(direction: torch.Tensor) -> torch.Tensor:
        if direction.dim() != 1:
            direction = direction.reshape(-1)
        return normalize(direction.unsqueeze(0).to(torch.float32)).squeeze(0).detach().clone()

    def _resolve_ccc_index(self, cue_direction: torch.Tensor) -> int | None:
        if not self.concept_directions:
            return None

        cue = self._normalize_direction(cue_direction)
        indices = list(self.concept_directions.keys())
        directions = torch.stack(
            [self.concept_directions[index].to(device=cue.device, dtype=cue.dtype) for index in indices]
        )
        cue_batch = cue.unsqueeze(0).expand_as(directions)
        similarities = cosine_similarity(directions, cue_batch)
        best = int(torch.argmax(similarities).item())
        if float(similarities[best].item()) <= 0.0:
            return None
        return indices[best]

    def _add_association(
        self,
        src_index: int,
        dst_index: int,
        strength: float,
        *,
        temporal: bool,
    ) -> None:
        if src_index == dst_index or strength <= 0.0:
            return
        key = (src_index, dst_index)
        self.association_strength[key] = self.association_strength.get(key, 0.0) + float(strength)
        self.association_temporal[key] = self.association_temporal.get(key, False) or temporal

    @torch.no_grad()
    def register_activation(
        self,
        ccc_index: int,
        concept_direction: torch.Tensor,
        confidence: float,
        timestep: int,
    ) -> None:
        """Record a CCC activation and write it into sparse distributed memory."""

        self._decay_associations(timestep)
        direction = self._normalize_direction(concept_direction)
        address = self.sdm.compute_address(direction)
        data = direction * float(confidence)

        self.concept_directions[int(ccc_index)] = direction
        self.activation_history.append(
            _ActivationRecord(
                ccc_index=int(ccc_index),
                direction=direction,
                confidence=float(confidence),
                timestep=int(timestep),
                address=address.detach().clone(),
            )
        )
        self._prune_activation_history(timestep)

        self.sdm.write(address, data)
        self.temporal_associator.record_activation(address, data, float(timestep))

    @torch.no_grad()
    def form_associations(self, timestep: int) -> None:
        """Strengthen recent co-activations and temporal causal links."""

        self._decay_associations(timestep)
        self._prune_activation_history(timestep)
        if len(self.activation_history) < 2:
            return

        ordered = sorted(self.activation_history, key=lambda activation: activation.timestep)
        for idx, activation_a in enumerate(ordered[:-1]):
            for activation_b in ordered[idx + 1 :]:
                delta_t = activation_b.timestep - activation_a.timestep
                if delta_t < 0 or delta_t > self.temporal_window:
                    continue

                mean_confidence = 0.5 * (
                    activation_a.confidence + activation_b.confidence
                )
                if delta_t == 0:
                    self._add_association(
                        activation_a.ccc_index,
                        activation_b.ccc_index,
                        mean_confidence,
                        temporal=False,
                    )
                    self._add_association(
                        activation_b.ccc_index,
                        activation_a.ccc_index,
                        mean_confidence,
                        temporal=False,
                    )
                    self.sdm.associate(
                        activation_a.address,
                        activation_b.address,
                        activation_a.direction * mean_confidence,
                        activation_b.direction * mean_confidence,
                        temporal_order=False,
                    )
                    continue

                causal_strength = mean_confidence * math.exp(
                    -delta_t / float(self.temporal_window)
                )
                self._add_association(
                    activation_a.ccc_index,
                    activation_b.ccc_index,
                    causal_strength,
                    temporal=True,
                )
                self._add_association(
                    activation_b.ccc_index,
                    activation_a.ccc_index,
                    causal_strength * 0.5,
                    temporal=False,
                )

        self.temporal_associator.form_associations()

    def retrieve_associates(
        self,
        cue_direction: torch.Tensor,
        k: int = 5,
    ) -> AssociationResult:
        """Retrieve the top-k associates for a concept cue."""

        if k <= 0 or not self.concept_directions:
            return self._empty_association_result()

        cue = self._normalize_direction(cue_direction)
        retrieved = self.sdm.read(cue)
        cue_index = self._resolve_ccc_index(cue)

        ranked: list[tuple[int, float, bool]] = []
        if cue_index is not None:
            for (src_index, dst_index), strength in self.association_strength.items():
                if src_index != cue_index or dst_index not in self.concept_directions:
                    continue
                ranked.append(
                    (
                        dst_index,
                        float(strength),
                        self.association_temporal.get((src_index, dst_index), False),
                    )
                )
            ranked.sort(key=lambda item: item[1], reverse=True)

        if not ranked:
            retrieved_norm = (
                normalize(retrieved.unsqueeze(0)).squeeze(0)
                if float(retrieved.norm().item()) > 0.0
                else retrieved
            )
            for index, direction in self.concept_directions.items():
                if cue_index is not None and index == cue_index:
                    continue
                similarity = float(
                    cosine_similarity(
                        direction.unsqueeze(0),
                        retrieved_norm.unsqueeze(0).to(direction.device),
                    ).item()
                )
                if similarity > 0.0:
                    ranked.append((index, similarity, False))
            ranked.sort(key=lambda item: item[1], reverse=True)

        top_ranked = ranked[:k]
        return AssociationResult(
            directions=[self.concept_directions[index].detach().clone() for index, _, _ in top_ranked],
            strengths=[strength for _, strength, _ in top_ranked],
            indices=[index for index, _, _ in top_ranked],
            temporal_order=[temporal for _, _, temporal in top_ranked],
        )

    def retrieve_sequence(self, start_direction: torch.Tensor, steps: int = 5) -> list[torch.Tensor]:
        """Follow the strongest learned temporal chain from a starting concept."""

        if steps <= 0:
            return []

        current_index = self._resolve_ccc_index(start_direction)
        if current_index is None:
            return []

        visited = {current_index}
        sequence: list[torch.Tensor] = []
        for _ in range(steps):
            temporal_candidates = [
                (dst_index, strength)
                for (src_index, dst_index), strength in self.association_strength.items()
                if src_index == current_index
                and self.association_temporal.get((src_index, dst_index), False)
                and dst_index not in visited
            ]
            if not temporal_candidates:
                break

            next_index, _ = max(temporal_candidates, key=lambda item: item[1])
            next_direction = self.concept_directions[next_index].detach().clone()
            sequence.append(next_direction)
            visited.add(next_index)
            current_index = next_index

        return sequence

    def lateral_inhibition(
        self,
        active_cccs: list[tuple[int, torch.Tensor, float]],
        k: int,
    ) -> list[tuple[int, float]]:
        """Keep the strongest non-overlapping active CCCs."""

        if not active_cccs or k <= 0:
            return []

        indices = [index for index, _, _ in active_cccs]
        directions = torch.stack([self._normalize_direction(direction) for _, direction, _ in active_cccs])
        confidences = torch.tensor(
            [confidence for _, _, confidence in active_cccs],
            device=directions.device,
            dtype=directions.dtype,
        )
        addresses = self.sdm.compute_address(directions)
        distances = (addresses.unsqueeze(1) != addresses.unsqueeze(0)).sum(dim=-1)

        selected: list[int] = []
        order = torch.argsort(confidences, descending=True).tolist()
        for candidate in order:
            if len(selected) >= k or float(confidences[candidate].item()) <= 0.0:
                break
            if any(
                bool((distances[candidate, winner] <= self.sdm.hamming_radius).item())
                for winner in selected
            ):
                continue
            selected.append(candidate)

        return [(indices[position], float(confidences[position].item())) for position in selected]

    def vote(self, active_cccs: list[tuple[int, torch.Tensor, float]]) -> VoteResult:
        """Form a distributed consensus across active CCCs."""

        if not active_cccs:
            return self._empty_vote_result(self.sdm.data_matrix.device)

        indices = [index for index, _, _ in active_cccs]
        directions = torch.stack([self._normalize_direction(direction) for _, direction, _ in active_cccs])
        confidences = torch.tensor(
            [confidence for _, _, confidence in active_cccs],
            device=directions.device,
            dtype=directions.dtype,
        ).clamp_min(0.0)

        pairwise_similarity = torch.matmul(directions, directions.T).clamp(min=0.0, max=1.0)
        support = pairwise_similarity @ confidences
        winner_position = int(torch.argmax(support * confidences).item())

        alignment = pairwise_similarity[winner_position]
        consensus_weights = confidences * alignment
        if float(consensus_weights.sum().item()) <= 0.0:
            consensus_weights[winner_position] = confidences[winner_position].clamp_min(1e-6)

        weighted_direction = (consensus_weights.unsqueeze(-1) * directions).sum(dim=0, keepdim=True)
        winning_direction = normalize(weighted_direction).squeeze(0)

        agreement = cosine_similarity(
            directions,
            winning_direction.unsqueeze(0).expand_as(directions),
        ).clamp(min=0.0, max=1.0)
        agreement_score = float(
            (agreement * confidences).sum().item() / confidences.sum().clamp_min(1e-6).item()
        )

        participation_threshold = float(consensus_weights.max().item()) * 0.1
        voter_mask = consensus_weights > participation_threshold
        if not bool(voter_mask.any().item()):
            voter_mask[winner_position] = True
        voter_indices = [index for index, included in zip(indices, voter_mask.tolist()) if included]

        base_confidence = float(confidences[voter_mask].mean().item()) if voter_indices else 0.0
        population_factor = min(1.0, len(active_cccs) / 3.0)
        confidence = base_confidence * agreement_score * population_factor

        return VoteResult(
            winning_direction=winning_direction.detach().clone(),
            confidence=float(confidence),
            agreement_score=float(agreement_score),
            voter_count=len(voter_indices),
            voter_indices=voter_indices,
        )

    def get_stats(self) -> dict[str, float | int | tuple[int, int, float] | None]:
        """Return high-level sparse-association statistics."""

        if not self.association_strength:
            strongest_pair: tuple[int, int, float] | None = None
            mean_strength = 0.0
        else:
            strongest_key, strongest_value = max(
                self.association_strength.items(),
                key=lambda item: item[1],
            )
            strongest_pair = (
                int(strongest_key[0]),
                int(strongest_key[1]),
                float(strongest_value),
            )
            mean_strength = sum(self.association_strength.values()) / len(self.association_strength)

        temporal_chains_count = sum(
            1 for is_temporal in self.association_temporal.values() if is_temporal
        )
        return {
            "num_associations": len(self.association_strength),
            "mean_strength": float(mean_strength),
            "strongest_pair": strongest_pair,
            "temporal_chains_count": temporal_chains_count,
        }


class FabricConnectedPool(nn.Module):
    """Wrap a CCC pool with the associative fabric."""

    def __init__(self, pool: CCCPool, fabric: AssociativeFabric):
        super().__init__()
        self.pool = pool
        self.fabric = fabric

    @torch.no_grad()
    def forward(self, raw_input: torch.Tensor, timestep: int) -> FabricPoolOutput:
        """Run pool activation, learning, inhibition, voting, and recall."""

        pool_output = self.pool(raw_input, timestep=timestep)
        active_cccs: list[tuple[int, torch.Tensor, float]] = []
        for index in pool_output.fired_indices:
            direction = self.pool.cccs[index].concept_direction.detach().clone()
            confidence = float(pool_output.outputs[index].confidence.reshape(-1).mean().item())
            active_cccs.append((index, direction, confidence))
            self.fabric.register_activation(index, direction, confidence, timestep)

        self.fabric.form_associations(timestep)
        inhibited_winners = self.fabric.lateral_inhibition(
            active_cccs,
            k=max(1, len(active_cccs)),
        )
        winning_indices = {index for index, _ in inhibited_winners}
        surviving_cccs = (
            [activation for activation in active_cccs if activation[0] in winning_indices]
            if inhibited_winners
            else active_cccs
        )

        consensus = self.fabric.vote(surviving_cccs)
        associates = (
            self.fabric.retrieve_associates(consensus.winning_direction, k=5)
            if surviving_cccs
            else self.fabric._empty_association_result()
        )

        return FabricPoolOutput(
            pool_output=pool_output,
            active_cccs=active_cccs,
            inhibited_winners=inhibited_winners,
            consensus=consensus,
            associates=associates,
        )

    def predict_next(self, current_direction: torch.Tensor) -> torch.Tensor:
        """Predict the next concept by following the strongest temporal link."""

        sequence = self.fabric.retrieve_sequence(current_direction, steps=1)
        if sequence:
            return sequence[0]

        fallback = self.fabric.sdm.read(current_direction)
        if float(fallback.norm().item()) > 0.0:
            return normalize(fallback.unsqueeze(0)).squeeze(0)
        return self.fabric._normalize_direction(current_direction)


__all__ = [
    "AssociationResult",
    "AssociativeFabric",
    "FabricConnectedPool",
    "FabricPoolOutput",
    "VoteResult",
]
