"""Online temporal learning over synthetic video streams."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import torch

from bioarn.config import CCCConfig, GNWConfig, MarginGateConfig, STDPConfig, TemporalTrainConfig
from bioarn.core.ccc import CCCPool, CCCPoolOutput
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.data.video import SyntheticVideoStream, VideoSequence
from bioarn.temporal import TemporalContextBuffer, TemporalSequenceLayer
from bioarn.workspace.gnw import EnhancedGNW


@dataclass
class FrameTemporalResult:
    """Per-frame temporal learning summary."""

    frame_index: int
    fired_cccs: list[int]
    fired_features: list[int]
    predicted_features: list[int]
    prediction_score: float
    surprise: float
    workspace_indices: list[int]


@dataclass
class SequenceResult:
    """Aggregated result for one processed video sequence."""

    label: int
    temporal_label: str
    frame_results: list[FrameTemporalResult] = field(default_factory=list)
    prediction_accuracy: float = 0.0
    mean_surprise: float = 0.0
    max_surprise: float = 0.0
    workspace_occupancy: float = 0.0


class TemporalTrainer:
    """Online temporal pattern learning with STDP and GNW context."""

    def __init__(self, config: TemporalTrainConfig):
        self.config = config
        torch.manual_seed(int(config.seed))
        workspace_config = copy.deepcopy(config.workspace) if config.workspace is not None else GNWConfig()
        workspace_config.concept_dim = int(config.concept_dim)
        self.gnw = EnhancedGNW(workspace_config)
        self.temporal_layer = TemporalSequenceLayer(copy.deepcopy(config.temporal))
        self.context_buffer = TemporalContextBuffer(
            window_size=int(config.temporal.context_window),
            concept_dim=int(config.concept_dim),
        )

        input_dim = int(config.frame_shape[0] * config.frame_shape[1])
        ccc_stdp = copy.deepcopy(config.stdp) if config.stdp is not None else STDPConfig()
        self.pool = CCCPool(
            CCCConfig(
                input_dim=input_dim,
                concept_dim=int(config.concept_dim),
                num_f1_features=int(config.concept_dim),
                f1_top_k=max(4, min(int(config.concept_dim), int(config.prediction_top_k * 2))),
                fast_lr=1.0,
                slow_lr=0.1,
                feedback_lr=0.15,
                max_pool_size=int(config.max_pool_size),
                stdp=ccc_stdp,
            ),
            MarginGateConfig(
                theta_margin=float(config.margin_threshold),
                theta_margin_lr=0.01,
                theta_resonance=0.85,
            ),
        )
        self.stream = SyntheticVideoStream(
            num_sequences=int(config.num_sequences),
            frames_per_sequence=int(config.frames_per_sequence),
            frame_shape=config.frame_shape,
            seed=int(config.seed),
            violation_rate=float(config.causal_violation_rate),
        )
        self.timestep = 0

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if float(flattened.norm().item()) <= 1e-8:
            return torch.zeros_like(flattened)
        return normalize(flattened.unsqueeze(0)).squeeze(0)

    def _flatten_frame(self, frame: torch.Tensor) -> torch.Tensor:
        flattened = frame.detach().reshape(-1).to(torch.float32)
        expected = int(self.config.frame_shape[0] * self.config.frame_shape[1])
        if flattened.numel() != expected:
            raise ValueError(f"Expected frame size {expected}, received {flattened.numel()}.")
        return flattened

    def _feature_indices(self, concept_activation: torch.Tensor) -> list[int]:
        vector = concept_activation.detach().reshape(-1)
        if vector.numel() == 0:
            return []
        top_k = min(int(self.config.prediction_top_k), int(vector.numel()))
        values, indices = torch.topk(vector, k=top_k)
        active = [int(index) for value, index in zip(values.tolist(), indices.tolist(), strict=False) if value > 0.0]
        if active:
            return sorted(set(active))
        return [int(index) for index in indices[:1].tolist()] if indices.numel() > 0 else []

    def _pool_confidence(self, pool_output: CCCPoolOutput, ccc_index: int) -> float:
        return float(pool_output.outputs[ccc_index].confidence.reshape(-1).mean().item())

    def _align_to_concept_dim(self, vector: torch.Tensor) -> torch.Tensor:
        flattened = vector.detach().reshape(-1).to(torch.float32)
        if flattened.numel() > self.config.concept_dim:
            flattened = flattened[: self.config.concept_dim]
        elif flattened.numel() < self.config.concept_dim:
            flattened = torch.nn.functional.pad(
                flattened,
                (0, self.config.concept_dim - flattened.numel()),
            )
        return self._normalize(flattened)

    def _concept_activation(self, pool_output: CCCPoolOutput, fallback: torch.Tensor) -> torch.Tensor:
        sensory_projection = self._align_to_concept_dim(fallback)
        if not pool_output.fired_indices:
            return sensory_projection

        directions: list[torch.Tensor] = []
        weights: list[float] = []
        for ccc_index in pool_output.fired_indices:
            directions.append(self.pool.cccs[ccc_index].concept_direction.detach().clone())
            weights.append(max(1e-6, self._pool_confidence(pool_output, ccc_index)))

        stacked = torch.stack(directions, dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        pooled = (stacked * weight_tensor.unsqueeze(-1)).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6)
        return self._normalize((0.7 * pooled) + (0.3 * sensory_projection.to(pooled)))

    def _context_similarity(self, left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = self._normalize(left)
        right_norm = self._normalize(right)
        if float(left_norm.norm().item()) <= 1e-8 or float(right_norm.norm().item()) <= 1e-8:
            return 0.0
        similarity = float(
            cosine_similarity(
                left_norm.unsqueeze(0).to(right_norm),
                right_norm.unsqueeze(0),
            ).item()
        )
        return max(0.0, similarity)

    def _workspace_candidates(
        self,
        pool_output: CCCPoolOutput,
        temporal_context: torch.Tensor,
    ) -> list[tuple[int, torch.Tensor, float]]:
        candidates: list[tuple[int, torch.Tensor, float]] = []
        for ccc_index in pool_output.fired_indices:
            direction = self.pool.cccs[ccc_index].concept_direction.detach().clone()
            confidence = self._pool_confidence(pool_output, ccc_index)
            if self.config.use_workspace_context and float(temporal_context.norm().item()) > 0.0:
                confidence = min(
                    1.0,
                    confidence + (
                        float(self.config.context_gain)
                        * self._context_similarity(direction, temporal_context)
                    ),
                )
            candidates.append((int(ccc_index), direction, confidence))
        return candidates

    @staticmethod
    def _prediction_score(predicted: list[int], actual: list[int]) -> float:
        predicted_set = set(int(index) for index in predicted)
        actual_set = set(int(index) for index in actual)
        if not predicted_set and not actual_set:
            return 1.0
        if not actual_set:
            return 0.0
        return len(predicted_set & actual_set) / float(len(actual_set))

    def _reset_runtime_state(self) -> None:
        self.temporal_layer.reset_state(clear_weights=False)
        self.context_buffer.clear()
        self.gnw.clear()

    def train_sequence(
        self,
        frames: list[torch.Tensor],
        label: int,
        temporal_label: str = "unknown",
    ) -> SequenceResult:
        """Process one video sequence: per-frame CCC + STDP between frames."""

        self._reset_runtime_state()
        frame_results: list[FrameTemporalResult] = []
        prediction_scores: list[float] = []
        surprises: list[float] = []

        for frame_index, frame in enumerate(frames):
            flattened = self._flatten_frame(frame)
            temporal_context = self.context_buffer.get_context()
            pool_output = self.pool(flattened, timestep=self.timestep)
            concept_activation = self._concept_activation(pool_output, fallback=flattened)
            fired_features = self._feature_indices(concept_activation)
            candidates = self._workspace_candidates(pool_output, temporal_context)
            if candidates:
                self.gnw.update(candidates, timestep=self.timestep)
            temporal_output = self.temporal_layer.observe_frame(concept_activation, fired_features)
            self.context_buffer.push(concept_activation, fired_features)
            if self.config.use_workspace_context:
                refreshed_context = self.context_buffer.get_context()
                if float(refreshed_context.norm().item()) > 0.0:
                    self.gnw.context.update(
                        refreshed_context,
                        max(0.1, 1.0 - float(temporal_output.surprise)),
                    )

            prediction_score = self._prediction_score(
                temporal_output.prior_predicted_indices,
                fired_features,
            )
            if frame_index > 0:
                prediction_scores.append(prediction_score)
                surprises.append(float(temporal_output.surprise))

            frame_results.append(
                FrameTemporalResult(
                    frame_index=frame_index,
                    fired_cccs=[int(index) for index in pool_output.fired_indices],
                    fired_features=fired_features,
                    predicted_features=temporal_output.prior_predicted_indices,
                    prediction_score=prediction_score,
                    surprise=float(temporal_output.surprise),
                    workspace_indices=[slot.ccc_index for slot in self.gnw.slots],
                )
            )
            self.timestep += 1

        stats = self.gnw.get_stats()
        return SequenceResult(
            label=int(label),
            temporal_label=temporal_label,
            frame_results=frame_results,
            prediction_accuracy=float(sum(prediction_scores) / max(1, len(prediction_scores))),
            mean_surprise=float(sum(surprises) / max(1, len(surprises))),
            max_surprise=float(max(surprises) if surprises else 0.0),
            workspace_occupancy=float(stats.get("occupancy", 0.0)),
        )

    def train_online(self, stream: Iterable[VideoSequence] | None = None) -> dict:
        """Train on streaming video sequences and summarize temporal performance."""

        source = self.stream if stream is None else stream
        results: list[SequenceResult] = []
        for sequence in source:
            results.append(
                self.train_sequence(
                    sequence.frames,
                    sequence.label,
                    temporal_label=sequence.temporal_label,
                )
            )

        def _mean(values: list[float]) -> float:
            return float(sum(values) / max(1, len(values)))

        causal = [result for result in results if result.temporal_label == "causal"]
        violations = [result for result in results if result.temporal_label == "causal_violation"]
        return {
            "num_sequences": len(results),
            "mean_prediction_accuracy": _mean([result.prediction_accuracy for result in results]),
            "causal_prediction_accuracy": _mean([result.prediction_accuracy for result in causal]),
            "mean_surprise": _mean([result.mean_surprise for result in results]),
            "violation_surprise": _mean([result.max_surprise for result in violations]),
            "workspace_occupancy": _mean([result.workspace_occupancy for result in results]),
            "committed_cccs": int(sum(bool(ccc.is_committed.item()) for ccc in self.pool.cccs)),
        }
