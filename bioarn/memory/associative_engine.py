"""Neuromorphic associative memory engine built from existing Bio-ARN modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from bioarn.config import (
    AssociativeMemoryConfig,
    CCCConfig,
    GNWConfig,
    LateralPredictionConfig,
    MarginGateConfig,
    PrecisionConfig,
    SDMConfig,
)
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.workspace.gnw import BroadcastOutput, EnhancedGNW


@dataclass
class MemoryResult:
    """One ranked associative-memory retrieval result."""

    memory_id: str
    confidence: float
    content: torch.Tensor
    metadata: dict[str, Any]
    age: int
    importance: float


@dataclass
class _MemoryRecord:
    """Bookkeeping for an active memory slot."""

    memory_id: str
    content: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    created_step: int = 0
    last_access_step: int = 0
    query_hits: int = 0


class AssociativeMemoryEngine:
    """Neuromorphic associative memory — Store, Query, Reconstruct."""

    def __init__(self, config: AssociativeMemoryConfig):
        self.config = config
        self._clock = 0
        self._stores = 0
        self._queries = 0
        self._reconstructions = 0
        self._auto_consolidations = 0
        self._last_broadcast: BroadcastOutput | None = None

        precision = (
            PrecisionConfig(
                enabled=True,
                pool_size=config.capacity,
                entropy_window=max(16, config.capacity),
                precision_threshold=0.35,
                min_precision=0.2,
                max_precision=1.0,
            )
            if config.use_precision
            else None
        )
        lateral = LateralPredictionConfig(
            enabled=True,
            max_neighbors=min(8, max(1, config.capacity - 1)),
            hebbian_lr=0.15,
            anti_hebbian_lr=0.0,
            min_weight=0.5,
            max_weight=3.0,
            refresh_interval=1,
            prediction_threshold=0.05,
            surprise_gain=1.5,
        )
        ccc_config = CCCConfig(
            input_dim=config.input_dim,
            concept_dim=config.concept_dim,
            num_f1_features=config.concept_dim,
            f1_top_k=max(1, min(config.concept_dim, max(4, config.concept_dim // 4))),
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.1,
            max_pool_size=config.capacity,
            max_growth_factor=1.0,
            lock_threshold=config.importance_threshold if config.lock_important else 1.1,
            enable_replay=True,
            enable_eviction=True,
            precision=precision,
            lateral_prediction=lateral,
        )
        margin_config = MarginGateConfig(
            theta_margin=0.3,
            theta_margin_lr=0.001,
            theta_resonance=0.7,
        )
        self.ccc_pool = CCCPool(ccc_config, margin_config)

        sdm_config = SDMConfig(
            address_dim=config.concept_dim,
            hamming_radius=max(1, config.concept_dim // 32),
            num_hard_locations=max(config.capacity * 4, config.concept_dim),
            data_dim=config.input_dim,
            decay_rate=1.0,
            stdp_window=max(4, config.top_k_retrieval),
        )
        self.sdm = SparseDistributedMemory(sdm_config)
        self._base_hard_locations = self.sdm.hard_locations.detach().clone()

        self.workspace = (
            EnhancedGNW(
                GNWConfig(
                    capacity=max(1, min(config.top_k_retrieval, 7)),
                    concept_dim=config.concept_dim,
                    context_size=max(32, config.capacity),
                    context_top_k=max(1, min(config.top_k_retrieval, 5)),
                )
            )
            if config.use_workspace
            else None
        )

        self._records: dict[int, _MemoryRecord] = {}
        self._associations: dict[tuple[int, int], float] = {}

    def store(
        self,
        content: torch.Tensor,
        *,
        metadata: dict | None = None,
        importance: float = 1.0,
    ) -> str:
        """Store a memory and return its slot-backed memory id."""

        self._clock += 1
        self._stores += 1
        vector = self._prepare_input(content)
        slot = self._reserve_slot()
        importance_value = float(min(max(importance, 0.0), 1.0))
        ccc = self.ccc_pool.cccs[slot]
        f1_output = ccc.f1_encode(vector)
        ccc.learn_fast(vector, f1_output)
        if self.ccc_pool.replay_buffer is not None:
            self.ccc_pool.replay_buffer.store(slot, f1_output)
        self.ccc_pool.update_importance([slot], confidences=[importance_value])
        ccc.importance.fill_(importance_value)
        self.ccc_pool.consolidation.importance[slot].fill_(importance_value)
        if self.config.lock_important and importance_value >= self.config.importance_threshold:
            ccc.lock()

        record = _MemoryRecord(
            memory_id=self._memory_id(slot),
            content=vector.detach().clone().to(torch.float32),
            metadata=deepcopy(metadata or {}),
            importance=importance_value,
            created_step=self._clock,
            last_access_step=self._clock,
        )
        self._records[slot] = record
        self._rebuild_sdm()

        if (
            self.config.auto_consolidate_interval > 0
            and self._stores % self.config.auto_consolidate_interval == 0
        ):
            self._auto_consolidations += 1
            self.consolidate()

        return record.memory_id

    def query(
        self,
        probe: torch.Tensor,
        *,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[MemoryResult]:
        """Query by similarity and return ranked reconstructed matches."""

        self._clock += 1
        self._queries += 1
        if top_k <= 0 or not self._records:
            return []

        vector = self._prepare_input(probe)
        preview = self.ccc_pool.preview(vector)
        candidate_scores = {
            int(index): float(confidence.item())
            for index, confidence in zip(
                preview.fired_indices,
                preview.winner_confidences,
                strict=False,
            )
            if index in self._records
        }
        concept_probe = self._concept_probe(vector)

        for index in self._records:
            similarity = self._concept_similarity(index, concept_probe)
            candidate_scores[index] = max(candidate_scores.get(index, 0.0), similarity)

        if candidate_scores:
            best_index = max(candidate_scores, key=candidate_scores.get)
            self._apply_association_boosts(best_index, candidate_scores)

        precision = self.ccc_pool.get_precision()
        ranked: list[tuple[int, float]] = []
        for index, base_confidence in candidate_scores.items():
            if index not in self._records:
                continue
            record = self._records[index]
            score = min(1.0, base_confidence + (0.15 * precision * record.importance))
            if score >= float(threshold):
                ranked.append((index, score))
        ranked.sort(
            key=lambda item: (
                item[1],
                self._records[item[0]].importance,
                -self._age_for(item[0]),
            ),
            reverse=True,
        )

        results: list[MemoryResult] = []
        for index, score in ranked[: max(1, top_k)]:
            record = self._records[index]
            record.last_access_step = self._clock
            record.query_hits += 1
            results.append(
                MemoryResult(
                    memory_id=record.memory_id,
                    confidence=float(score),
                    content=self._reconstruct_slot(index),
                    metadata=deepcopy(record.metadata),
                    age=self._age_for(index),
                    importance=record.importance,
                )
            )

        if results and self.workspace is not None:
            winner_index = self._index_from_memory_id(results[0].memory_id)
            direction = self.ccc_pool.cccs[winner_index].concept_direction.detach().clone()
            self._last_broadcast = self.workspace.broadcast_with_context(
                (winner_index, direction, results[0].confidence),
                context_query=concept_probe,
            )

        return results

    def reconstruct(
        self,
        memory_id: str,
        *,
        partial_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct stored content from slot id or a partial cue."""

        self._clock += 1
        self._reconstructions += 1
        index = self._index_from_memory_id(memory_id)
        if partial_cue is not None:
            cue = self._prepare_input(partial_cue)
            reconstructed = self.sdm.retrieve_associates(self._concept_probe(cue)).reshape(-1)
            if float(reconstructed.norm().item()) > 1e-6:
                self._records[index].last_access_step = self._clock
                return reconstructed.to(torch.float32)
        self._records[index].last_access_step = self._clock
        return self._reconstruct_slot(index)

    def associate(
        self,
        memory_id_a: str,
        memory_id_b: str,
        *,
        strength: float = 1.0,
    ) -> None:
        """Create a bidirectional lateral association between two memories."""

        self._clock += 1
        index_a = self._index_from_memory_id(memory_id_a)
        index_b = self._index_from_memory_id(memory_id_b)
        if index_a == index_b:
            return
        strength_value = float(max(0.0, strength))
        if strength_value <= 0.0:
            return
        self._associations[(index_a, index_b)] = self._associations.get((index_a, index_b), 0.0) + strength_value
        self._associations[(index_b, index_a)] = self._associations.get((index_b, index_a), 0.0) + strength_value
        self._strengthen_lateral_link(index_a, index_b, strength_value)
        self._rebuild_sdm()

    def forget(self, memory_id: str) -> bool:
        """Explicitly forget a memory and free its CCC slot."""

        self._clock += 1
        try:
            index = self._index_from_memory_id(memory_id)
        except KeyError:
            return False
        self._clear_slot(index)
        self._rebuild_sdm()
        if self.workspace is not None and any(slot.ccc_index == index for slot in self.workspace.slots):
            self.workspace.clear()
            self._last_broadcast = None
        return True

    def consolidate(self) -> int:
        """Lock important memories, replay exemplars, and evict weak saturated slots."""

        self._clock += 1
        locked = self.ccc_pool.auto_lock() if self.config.lock_important else []
        replayed = self.ccc_pool.replay_exemplars()
        evicted: list[int] = []
        if len(self._records) >= self.config.capacity:
            weak_indices = [
                index
                for index, record in self._records.items()
                if (
                    not bool(self.ccc_pool.cccs[index].locked.item())
                    and record.importance < max(0.1, self.config.importance_threshold * 0.5)
                )
            ]
            if weak_indices:
                evicted = self.ccc_pool.evict_weakest(num_slots=max(1, len(weak_indices) // 2))
                for index in evicted:
                    self._remove_record(index)
        self._rebuild_sdm()
        return len(locked) + int(replayed) + len(evicted)

    @property
    def stats(self) -> dict:
        """Return current memory, workspace, and retrieval statistics."""

        workspace_stats = self.workspace.get_stats() if self.workspace is not None else {}
        active_ids = [self._records[index].memory_id for index in sorted(self._records)]
        mean_importance = (
            sum(record.importance for record in self._records.values()) / len(self._records)
            if self._records
            else 0.0
        )
        return {
            "capacity": self.config.capacity,
            "active_memories": len(self._records),
            "free_slots": self.config.capacity - len(self._records),
            "locked_memories": sum(bool(self.ccc_pool.cccs[index].locked.item()) for index in self._records),
            "active_memory_ids": active_ids,
            "stores": self._stores,
            "queries": self._queries,
            "reconstructions": self._reconstructions,
            "associations": len(self._associations) // 2,
            "auto_consolidations": self._auto_consolidations,
            "precision": self.ccc_pool.get_precision(),
            "mean_importance": float(mean_importance),
            "workspace_occupancy": float(workspace_stats.get("occupancy", 0.0)),
            "workspace": workspace_stats,
            "pool": self.ccc_pool.get_pool_stats(),
            "sdm": self.sdm.get_stats(),
        }

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        vector = tensor.detach().clone().to(torch.float32).reshape(-1)
        if vector.numel() != self.config.input_dim:
            raise ValueError(
                f"AssociativeMemoryEngine expected input_dim={self.config.input_dim}, "
                f"received {vector.numel()} values."
            )
        return vector

    def _memory_id(self, index: int) -> str:
        return f"mem_{int(index):04d}"

    def _index_from_memory_id(self, memory_id: str) -> int:
        if not memory_id.startswith("mem_"):
            raise KeyError(f"Unknown memory id: {memory_id}")
        try:
            index = int(memory_id.split("_", maxsplit=1)[1])
        except ValueError as exc:
            raise KeyError(f"Unknown memory id: {memory_id}") from exc
        if index not in self._records:
            raise KeyError(f"Unknown memory id: {memory_id}")
        return index

    def _age_for(self, index: int) -> int:
        return max(0, self._clock - self._records[index].created_step)

    def _reserve_slot(self) -> int:
        for index, ccc in enumerate(self.ccc_pool.cccs):
            if not bool(ccc.is_committed.item()):
                return index
        evicted = self.ccc_pool.evict_weakest(num_slots=1)
        if not evicted:
            raise RuntimeError("Associative memory is full and no eviction candidate was found.")
        slot = int(evicted[0])
        self._remove_record(slot)
        return slot

    def _remove_record(self, index: int) -> None:
        self._records.pop(index, None)
        for key in [key for key in self._associations if index in key]:
            self._associations.pop(key, None)

    def _clear_slot(self, index: int) -> None:
        self.ccc_pool.cccs[index].reset_state()
        self.ccc_pool._reset_consolidation_slot(index)
        if self.ccc_pool.replay_buffer is not None:
            self.ccc_pool.replay_buffer.drop(index)
        self._remove_record(index)

    def _seed_hard_locations(self) -> None:
        if not self._records:
            self.sdm.hard_locations.copy_(self._base_hard_locations)
            return
        addresses = [
            self.sdm.compute_address(self.ccc_pool.cccs[index].concept_direction).reshape(-1)
            for index in sorted(self._records)
        ]
        unique_addresses = torch.unique(torch.stack(addresses, dim=0), dim=0)
        hard_locations = self._base_hard_locations.clone()
        limit = min(unique_addresses.shape[0], hard_locations.shape[0])
        hard_locations[:limit] = unique_addresses[:limit].to(hard_locations)
        self.sdm.hard_locations.copy_(hard_locations)

    def _rebuild_sdm(self) -> None:
        self.sdm.data_matrix.zero_()
        self.sdm.activation_counts.zero_()
        self._seed_hard_locations()
        for index in sorted(self._records):
            record = self._records[index]
            self.sdm._write_impl(
                self.ccc_pool.cccs[index].concept_direction,
                record.content,
                apply_decay=False,
            )
        for (source, target), strength in sorted(self._associations.items()):
            if source not in self._records or target not in self._records:
                continue
            weight = float(max(0.0, strength))
            if weight <= 0.0:
                continue
            self.sdm._write_impl(
                self.ccc_pool.cccs[source].concept_direction,
                self._records[target].content * weight,
                apply_decay=False,
            )
        if self.ccc_pool.lateral_network is not None:
            self.ccc_pool.lateral_network.refresh_neighbors(
                self.ccc_pool.concept_directions,
                force=True,
            )

    def _concept_probe(self, vector: torch.Tensor) -> torch.Tensor:
        encoder = self.ccc_pool.cccs[0]
        f1_output = encoder.f1_encode(vector)
        concept = encoder.f2_activate(f1_output).reshape(-1).to(torch.float32)
        if float(concept.norm().item()) <= 1e-6:
            return concept
        return normalize(concept.unsqueeze(0)).squeeze(0)

    def _concept_similarity(self, index: int, concept_probe: torch.Tensor) -> float:
        if float(concept_probe.norm().item()) <= 1e-6:
            return 0.0
        direction = self.ccc_pool.cccs[index].concept_direction
        similarity = float(cosine_similarity(direction, concept_probe).item())
        return max(0.0, min(1.0, similarity))

    def _apply_association_boosts(
        self,
        winner_index: int,
        candidate_scores: dict[int, float],
    ) -> None:
        winner_score = float(candidate_scores.get(winner_index, 0.0))
        for (source, target), strength in self._associations.items():
            if source != winner_index or target not in self._records:
                continue
            boost = min(0.95, winner_score * min(max(strength, 0.0), 2.0) * 0.7)
            candidate_scores[target] = max(candidate_scores.get(target, 0.0), boost)

        lateral_network = self.ccc_pool.lateral_network
        if lateral_network is None:
            return
        predictions = lateral_network.predict_lateral([winner_index], self.ccc_pool.concept_directions)
        for target, prediction in predictions.items():
            target_index = int(target)
            if target_index in self._records:
                candidate_scores[target_index] = max(
                    candidate_scores.get(target_index, 0.0),
                    min(0.9, 0.5 * float(prediction.item())),
                )

    def _decode_f1_to_input(self, f1_pattern: torch.Tensor) -> torch.Tensor:
        encoder = self.ccc_pool.cccs[0].f1_layer
        weight = encoder.weight.detach().to(torch.float32)
        bias = (
            encoder.bias.detach().to(torch.float32)
            if encoder.bias is not None
            else torch.zeros(weight.shape[0], dtype=torch.float32)
        )
        pseudoinverse = torch.linalg.pinv(weight)
        return pseudoinverse @ (f1_pattern.reshape(-1).to(torch.float32) - bias)

    def _reconstruct_slot(self, index: int) -> torch.Tensor:
        record = self._records[index]
        ccc = self.ccc_pool.cccs[index]
        predicted_f1 = ccc.generate_prediction(ccc.concept_direction).reshape(-1)
        decoded = self._decode_f1_to_input(predicted_f1)
        if float(decoded.norm().item()) <= 1e-6:
            return record.content.detach().clone()
        similarity = float(cosine_similarity(decoded, record.content).item())
        if similarity < 0.25:
            return ((0.5 * decoded) + (0.5 * record.content)).to(torch.float32)
        return decoded.to(torch.float32)

    def _strengthen_lateral_link(self, index_a: int, index_b: int, strength: float) -> None:
        network = self.ccc_pool.lateral_network
        if network is None:
            return
        network.refresh_neighbors(
            self.ccc_pool.concept_directions,
            source_indices=[index_a, index_b],
            force=True,
        )
        max_weight = float(network.config.max_weight)
        min_weight = float(network.config.min_weight)
        fill_value = max(min_weight, min(max_weight, 1.0 + strength))
        for source, target in ((index_a, index_b), (index_b, index_a)):
            slot = None
            for position, candidate in enumerate(network.neighbor_indices[source].tolist()):
                if bool(network.neighbor_mask[source, position].item()) and int(candidate) == target:
                    slot = position
                    break
            if slot is None:
                if network.neighbor_indices.shape[1] == 0:
                    continue
                inactive = torch.nonzero(~network.neighbor_mask[source], as_tuple=False).reshape(-1)
                slot = int(inactive[0].item()) if inactive.numel() else network.neighbor_indices.shape[1] - 1
                network.neighbor_indices[source, slot] = int(target)
                network.neighbor_mask[source, slot] = True
            network.lateral_weights[source, slot].fill_(fill_value)
        self.ccc_pool.hebbian_update_lateral([index_a, index_b])


__all__ = ["AssociativeMemoryEngine", "MemoryResult"]
