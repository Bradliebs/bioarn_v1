"""Global Neuronal Workspace and stream-of-consciousness controllers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import GNWConfig
from bioarn.core.math_utils import cosine_similarity, normalize


@dataclass
class GNWSlot:
    """A single occupied workspace slot."""

    ccc_index: int
    direction: torch.Tensor
    activation: float
    confidence: float
    age: int
    fatigue: float


@dataclass
class BroadcastOutput:
    """Current broadcast content emitted by the workspace."""

    directions: list[torch.Tensor]
    activations: list[float]
    indices: list[int]
    num_occupied: int
    total_broadcast_energy: float


@dataclass
class AttentionResult:
    """Query-conditioned relevance over occupied workspace slots."""

    best_match_index: int
    best_match_direction: torch.Tensor
    attention_weights: torch.Tensor
    relevance_score: float


@dataclass
class ThoughtOutput:
    """High-level output for one conscious thought step."""

    broadcast: BroadcastOutput
    new_entries: list[int]
    evicted: list[int]
    thought_chain_length: int
    is_ruminating: bool


class GlobalNeuronalWorkspace(nn.Module):
    """Winner-take-most workspace with fatigue, inertia, and broadcast."""

    def __init__(self, config: GNWConfig):
        super().__init__()
        if config.capacity <= 0:
            raise ValueError("GNW capacity must be positive.")

        self.config = config
        self.slots: list[GNWSlot] = []
        self.history_capacity = max(32, config.capacity * 16)
        self.broadcast_history: list[GNWSlot | None] = [None] * self.history_capacity
        self._history_cursor = 0
        self._history_count = 0
        self.last_new_entries: list[int] = []
        self.last_evicted: list[int] = []

        self.register_buffer("_device_anchor", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("timestep_tensor", torch.zeros((), dtype=torch.long))
        self.register_buffer("turnover_events", torch.zeros((), dtype=torch.long))

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self.slots = [replace(slot, direction=fn(slot.direction)) for slot in self.slots]
        self.broadcast_history = [
            replace(slot, direction=fn(slot.direction)) if slot is not None else None
            for slot in self.broadcast_history
        ]
        return self

    def _normalize_direction(self, direction: torch.Tensor) -> torch.Tensor:
        tensor = direction.detach().reshape(-1).to(device=self._device_anchor.device)
        return normalize(tensor.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _clone_slot(slot: GNWSlot) -> GNWSlot:
        return GNWSlot(
            ccc_index=slot.ccc_index,
            direction=slot.direction.detach().clone(),
            activation=float(slot.activation),
            confidence=float(slot.confidence),
            age=int(slot.age),
            fatigue=float(slot.fatigue),
        )

    def _record_history(self, slot: GNWSlot) -> None:
        self.broadcast_history[self._history_cursor] = self._clone_slot(slot)
        self._history_cursor = (self._history_cursor + 1) % self.history_capacity
        self._history_count = min(self._history_count + 1, self.history_capacity)

    def _history_items(self) -> list[GNWSlot]:
        if self._history_count == 0:
            return []
        start = (self._history_cursor - self._history_count) % self.history_capacity
        ordered: list[GNWSlot] = []
        for offset in range(self._history_count):
            slot = self.broadcast_history[(start + offset) % self.history_capacity]
            if slot is not None:
                ordered.append(self._clone_slot(slot))
        return ordered

    def _find_slot_position(self, ccc_index: int) -> int | None:
        for position, slot in enumerate(self.slots):
            if slot.ccc_index == ccc_index:
                return position
        return None

    def _replace_slot(self, position: int, new_slot: GNWSlot) -> int:
        evicted = self.slots[position].ccc_index
        self.slots[position] = new_slot
        self.turnover_events.add_(torch.tensor(2, device=self.turnover_events.device))
        return evicted

    def _candidate_map(
        self, candidates: list[tuple[int, torch.Tensor, float]]
    ) -> dict[int, tuple[torch.Tensor, float]]:
        deduped: dict[int, tuple[torch.Tensor, float]] = {}
        for ccc_index, direction, confidence in candidates:
            confidence_value = float(confidence)
            if ccc_index not in deduped or confidence_value > deduped[ccc_index][1]:
                deduped[ccc_index] = (self._normalize_direction(direction), confidence_value)
        return deduped

    def compete(self, candidates: list[tuple[int, torch.Tensor, float]]) -> list[int]:
        """Run softmax competition and return winning candidate CCC indices."""

        candidate_map = self._candidate_map(candidates)
        fresh_candidates = [
            (ccc_index, direction, confidence)
            for ccc_index, (direction, confidence) in candidate_map.items()
            if self._find_slot_position(ccc_index) is None
        ]
        if not fresh_candidates:
            return []

        temp = max(self.config.competition_temp, 1e-6)
        scores = torch.tensor(
            [candidate[2] for candidate in fresh_candidates],
            device=self._device_anchor.device,
            dtype=self._device_anchor.dtype,
        )
        weights = F.softmax(scores / temp, dim=0)
        ranking = sorted(
            range(len(fresh_candidates)),
            key=lambda index: (float(weights[index].item()), fresh_candidates[index][2]),
            reverse=True,
        )

        inertia_margin = max(0.05, self.config.fatigue_threshold)
        remaining_empty = max(0, self.config.capacity - len(self.slots))
        available_replacements = sorted(self.slots, key=lambda slot: slot.activation)
        winners: list[int] = []

        for ranked_index in ranking:
            ccc_index, _, confidence = fresh_candidates[ranked_index]
            if remaining_empty > 0:
                winners.append(ccc_index)
                remaining_empty -= 1
                continue

            if not available_replacements:
                break

            weakest = available_replacements[0]
            if weakest.activation <= self.config.fatigue_threshold or (
                confidence > weakest.activation + inertia_margin
            ):
                winners.append(ccc_index)
                available_replacements.pop(0)

            if len(winners) >= self.config.capacity:
                break

        return winners[: self.config.capacity]

    @torch.no_grad()
    def update(
        self, fired_cccs: list[tuple[int, torch.Tensor, float]], timestep: int
    ) -> tuple[list[int], list[int]]:
        """Advance workspace dynamics and add new competition winners."""

        self.timestep_tensor.fill_(int(timestep))
        self.last_new_entries = []
        self.last_evicted = []

        candidate_map = self._candidate_map(fired_cccs)
        refreshed_indices = set(candidate_map)
        retained_slots: list[GNWSlot] = []

        for slot in self.slots:
            age = slot.age + 1
            fatigue = min(0.999, slot.fatigue + self.config.fatigue_rate)
            activation = slot.activation * max(0.0, 1.0 - fatigue)

            if slot.ccc_index in refreshed_indices:
                direction, confidence = candidate_map[slot.ccc_index]
                activation = max(activation, confidence * self.config.broadcast_gain)
                retained_slots.append(
                    GNWSlot(
                        ccc_index=slot.ccc_index,
                        direction=direction,
                        activation=float(activation),
                        confidence=float(confidence),
                        age=age,
                        fatigue=fatigue,
                    )
                )
                continue

            if activation < self.config.fatigue_threshold:
                self.last_evicted.append(slot.ccc_index)
                self.turnover_events.add_(torch.tensor(1, device=self.turnover_events.device))
                continue

            retained_slots.append(
                GNWSlot(
                    ccc_index=slot.ccc_index,
                    direction=slot.direction,
                    activation=float(activation),
                    confidence=float(slot.confidence),
                    age=age,
                    fatigue=fatigue,
                )
            )

        self.slots = retained_slots

        winners = self.compete(fired_cccs)
        inertia_margin = max(0.05, self.config.fatigue_threshold)

        for winner_index in winners:
            direction, confidence = candidate_map[winner_index]
            new_slot = GNWSlot(
                ccc_index=winner_index,
                direction=direction,
                activation=float(confidence * self.config.broadcast_gain),
                confidence=float(confidence),
                age=0,
                fatigue=0.0,
            )

            if len(self.slots) < self.config.capacity:
                self.slots.append(new_slot)
                self.last_new_entries.append(winner_index)
                self.turnover_events.add_(torch.tensor(1, device=self.turnover_events.device))
                continue

            weakest_position = min(range(len(self.slots)), key=lambda pos: self.slots[pos].activation)
            weakest = self.slots[weakest_position]
            if confidence <= weakest.activation + inertia_margin and (
                weakest.activation > self.config.fatigue_threshold
            ):
                continue

            displaced = self._replace_slot(weakest_position, new_slot)
            self.last_evicted.append(displaced)
            self.last_new_entries.append(winner_index)

        self.slots.sort(key=lambda slot: slot.activation, reverse=True)
        return list(self.last_new_entries), list(self.last_evicted)

    @torch.no_grad()
    def broadcast(self) -> BroadcastOutput:
        """Emit the current broadcast signal and record the dominant thought."""

        if not self.slots:
            return BroadcastOutput(
                directions=[],
                activations=[],
                indices=[],
                num_occupied=0,
                total_broadcast_energy=0.0,
            )

        dominant = max(self.slots, key=lambda slot: slot.activation)
        self._record_history(dominant)

        directions = [slot.direction.detach().clone() for slot in self.slots]
        activations = [float(slot.activation) for slot in self.slots]
        indices = [slot.ccc_index for slot in self.slots]
        return BroadcastOutput(
            directions=directions,
            activations=activations,
            indices=indices,
            num_occupied=len(self.slots),
            total_broadcast_energy=float(sum(activations)),
        )

    def get_stream(self, last_n: int = 10) -> list[GNWSlot]:
        """Return the recent dominant broadcast history in temporal order."""

        history = self._history_items()
        if last_n <= 0:
            return []
        return history[-last_n:]

    def attend(self, query_direction: torch.Tensor) -> AttentionResult:
        """Attend over current slot occupants using cosine-similarity relevance."""

        query = self._normalize_direction(query_direction)
        if not self.slots:
            return AttentionResult(
                best_match_index=-1,
                best_match_direction=query,
                attention_weights=torch.empty(0, device=query.device),
                relevance_score=0.0,
            )

        slot_directions = torch.stack([slot.direction for slot in self.slots], dim=0)
        similarities = cosine_similarity(slot_directions, query.unsqueeze(0).expand_as(slot_directions))
        attention_weights = F.softmax(similarities / max(self.config.competition_temp, 1e-6), dim=0)
        best_index = int(torch.argmax(attention_weights).item())
        return AttentionResult(
            best_match_index=best_index,
            best_match_direction=self.slots[best_index].direction.detach().clone(),
            attention_weights=attention_weights,
            relevance_score=float(similarities[best_index].item()),
        )

    @torch.no_grad()
    def inject(self, ccc_index: int, direction: torch.Tensor, priority: float = 1.0) -> None:
        """Force a concept into the workspace for top-down control."""

        normalized_direction = self._normalize_direction(direction)
        activation = float(priority * self.config.broadcast_gain)
        position = self._find_slot_position(ccc_index)
        if position is not None:
            slot = self.slots[position]
            self.slots[position] = GNWSlot(
                ccc_index=ccc_index,
                direction=normalized_direction,
                activation=max(slot.activation, activation),
                confidence=float(priority),
                age=0,
                fatigue=0.0,
            )
            self.last_new_entries = [ccc_index]
            self.last_evicted = []
            return

        new_slot = GNWSlot(
            ccc_index=ccc_index,
            direction=normalized_direction,
            activation=activation,
            confidence=float(priority),
            age=0,
            fatigue=0.0,
        )

        if len(self.slots) < self.config.capacity:
            self.slots.append(new_slot)
            self.slots.sort(key=lambda slot: slot.activation, reverse=True)
            self.last_new_entries = [ccc_index]
            self.last_evicted = []
            self.turnover_events.add_(torch.tensor(1, device=self.turnover_events.device))
            return

        weakest_position = min(range(len(self.slots)), key=lambda pos: self.slots[pos].activation)
        weakest = self.slots[weakest_position]
        if activation >= weakest.activation:
            displaced = self._replace_slot(weakest_position, new_slot)
            self.slots.sort(key=lambda slot: slot.activation, reverse=True)
            self.last_new_entries = [ccc_index]
            self.last_evicted = [displaced]
            return

        self.last_new_entries = []
        self.last_evicted = []

    def is_full(self) -> bool:
        """Return whether all workspace slots are occupied."""

        return len(self.slots) >= self.config.capacity

    @torch.no_grad()
    def clear(self) -> None:
        """Empty the current workspace contents while preserving history."""

        self.slots = []
        self.last_new_entries = []
        self.last_evicted = []

    def get_stats(self) -> dict[str, float]:
        """Summarize occupancy and turnover statistics."""

        if not self.slots:
            return {
                "occupancy": 0.0,
                "mean_age": 0.0,
                "mean_fatigue": 0.0,
                "mean_activation": 0.0,
                "turnover_rate": float(self.turnover_events.item())
                / max(1.0, float(self.timestep_tensor.item()) or 1.0),
                "longest_occupant": 0.0,
            }

        mean_age = sum(slot.age for slot in self.slots) / len(self.slots)
        mean_fatigue = sum(slot.fatigue for slot in self.slots) / len(self.slots)
        mean_activation = sum(slot.activation for slot in self.slots) / len(self.slots)
        return {
            "occupancy": len(self.slots) / self.config.capacity,
            "mean_age": float(mean_age),
            "mean_fatigue": float(mean_fatigue),
            "mean_activation": float(mean_activation),
            "turnover_rate": float(self.turnover_events.item())
            / max(1.0, float(self.timestep_tensor.item()) or 1.0),
            "longest_occupant": float(max(slot.age for slot in self.slots)),
        }


class StreamOfConsciousness(nn.Module):
    """Sequential reasoning controller built on top of the GNW."""

    def __init__(self, gnw: GlobalNeuronalWorkspace, config: GNWConfig):
        super().__init__()
        self.gnw = gnw
        self.config = config

    def think_step(
        self, fired_cccs: list[tuple[int, torch.Tensor, float]], timestep: int
    ) -> ThoughtOutput:
        """Advance one conscious reasoning step."""

        new_entries, evicted = self.gnw.update(fired_cccs, timestep=timestep)
        broadcast = self.gnw.broadcast()
        thought_chain = self.get_thought_chain(n=self.gnw.history_capacity)
        return ThoughtOutput(
            broadcast=broadcast,
            new_entries=new_entries,
            evicted=evicted,
            thought_chain_length=len(thought_chain),
            is_ruminating=self.detect_rumination(),
        )

    def get_thought_chain(self, n: int = 5) -> list[torch.Tensor]:
        """Return the last N distinct dominant thoughts."""

        distinct: list[torch.Tensor] = []
        for slot in self.gnw.get_stream(last_n=self.gnw.history_capacity):
            if not distinct:
                distinct.append(slot.direction.detach().clone())
                continue

            similarity = cosine_similarity(distinct[-1], slot.direction).item()
            if similarity < 0.995:
                distinct.append(slot.direction.detach().clone())

        if n <= 0:
            return []
        return distinct[-n:]

    def detect_rumination(self) -> bool:
        """Detect repeated fixation on the same dominant concept."""

        stream = self.gnw.get_stream(last_n=self.gnw.history_capacity)
        if not stream:
            return False

        trailing_index = stream[-1].ccc_index
        run_length = 0
        for slot in reversed(stream):
            if slot.ccc_index != trailing_index:
                break
            run_length += 1

        occupancy_count = max(1, len(self.gnw.slots))
        return run_length > (3 * occupancy_count)

