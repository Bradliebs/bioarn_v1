"""Concept Cell Cluster (CCC) primitives for Bio-ARN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.margin_gate import MarginGate, MarginGateOutput, ResonanceOutput
from bioarn.core.math_utils import normalize, sparse_top_k


@dataclass
class CCCOutput:
    """Output of a single concept cell cluster."""

    fired: bool
    abstained: bool
    confidence: torch.Tensor
    f1_output: torch.Tensor
    f2_activation: torch.Tensor
    gate_output: MarginGateOutput
    prediction: torch.Tensor | None
    resonance: ResonanceOutput | None


@dataclass
class CCCPoolOutput:
    """Aggregated output for a pool of concept cell clusters."""

    outputs: list[CCCOutput]
    fired_indices: list[int]
    abstained_indices: list[int]
    recruited: bool
    recruited_index: int | None
    winner_confidences: torch.Tensor


class ConceptCellCluster(nn.Module):
    """A self-contained cortical-column-like concept circuit."""

    def __init__(self, config: CCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config

        self.f1_layer = nn.Linear(config.input_dim, config.num_f1_features)
        for parameter in self.f1_layer.parameters():
            parameter.requires_grad_(False)

        f2_weights = normalize(
            torch.randn(config.concept_dim, config.num_f1_features, dtype=torch.float32)
        )
        self.register_buffer("f2_weights", f2_weights)
        self.margin_gate = MarginGate(margin_config)
        self.register_buffer(
            "feedback_weights",
            torch.zeros(config.num_f1_features, config.concept_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "concept_direction", torch.zeros(config.concept_dim, dtype=torch.float32)
        )
        self.register_buffer("is_committed", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("age", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_fired", torch.tensor(-1, dtype=torch.long))

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() != 2:
            raise ValueError("CCC inputs must have shape (input_dim,) or (batch, input_dim).")
        return x, False

    @staticmethod
    def _maybe_squeeze(x: torch.Tensor, squeeze: bool) -> torch.Tensor:
        return x.squeeze(0) if squeeze else x

    @staticmethod
    def _to_bool_tensor(value: bool, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), value, device=device, dtype=torch.bool)

    @staticmethod
    def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
        return normalize(x.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _normalize_feedback_rows(x: torch.Tensor) -> torch.Tensor:
        return normalize(x)

    @staticmethod
    def _prepare_batch_vector(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        vector = x.reshape(-1)
        if vector.numel() == 1 and batch_size > 1:
            return vector.expand(batch_size)
        if vector.numel() != batch_size:
            raise ValueError(
                f"Expected batch-aligned vector of length {batch_size}, got {vector.numel()}."
            )
        return vector

    def _format_gate_output(
        self, gate_output: MarginGateOutput, squeeze: bool
    ) -> MarginGateOutput:
        if not squeeze:
            return gate_output
        return MarginGateOutput(
            output=gate_output.output.squeeze(0),
            confidence=gate_output.confidence.squeeze(0),
            fired=gate_output.fired.squeeze(0),
            abstained=gate_output.abstained.squeeze(0),
        )

    def _format_resonance_output(
        self, resonance: ResonanceOutput, squeeze: bool
    ) -> ResonanceOutput:
        if not squeeze:
            return resonance
        return ResonanceOutput(
            match_score=resonance.match_score.squeeze(0),
            resonated=resonance.resonated.squeeze(0),
            learn_signal=resonance.learn_signal.squeeze(0),
        )

    def _make_abstention_output(
        self,
        f1_output: torch.Tensor,
        f2_activation: torch.Tensor,
        *,
        squeeze: bool,
    ) -> CCCOutput:
        batch_size = f2_activation.shape[0]
        device = f2_activation.device
        confidence = torch.zeros(batch_size, device=device, dtype=f2_activation.dtype)
        gate_output = MarginGateOutput(
            output=torch.zeros_like(f2_activation),
            confidence=confidence,
            fired=self._to_bool_tensor(False, batch_size, device),
            abstained=self._to_bool_tensor(True, batch_size, device),
        )
        formatted_gate = self._format_gate_output(gate_output, squeeze=squeeze)
        return CCCOutput(
            fired=False,
            abstained=True,
            confidence=self._maybe_squeeze(confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_output, squeeze),
            f2_activation=self._maybe_squeeze(f2_activation, squeeze),
            gate_output=formatted_gate,
            prediction=None,
            resonance=None,
        )

    @torch.no_grad()
    def empty_output(self, raw_input: torch.Tensor) -> CCCOutput:
        """Return an abstention placeholder without processing input."""

        raw_batch, squeeze = self._ensure_batch(raw_input)
        device = raw_batch.device
        dtype = self.f1_layer.weight.dtype
        f1_output = torch.zeros(
            raw_batch.shape[0], self.config.num_f1_features, device=device, dtype=dtype
        )
        f2_activation = torch.zeros(
            raw_batch.shape[0], self.config.concept_dim, device=device, dtype=dtype
        )
        return self._make_abstention_output(f1_output, f2_activation, squeeze=squeeze)

    @torch.no_grad()
    def f1_encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Project raw input into sparse F1 feature space."""

        raw_batch, squeeze = self._ensure_batch(raw_input)
        projected = self.f1_layer(raw_batch.to(self.f1_layer.weight.dtype))
        activated = F.relu(projected)
        sparse = sparse_top_k(activated, self.config.f1_top_k)
        return self._maybe_squeeze(sparse, squeeze)

    @torch.no_grad()
    def f2_activate(self, f1_output: torch.Tensor) -> torch.Tensor:
        """Project F1 activity into concept space."""

        f1_batch, squeeze = self._ensure_batch(f1_output)
        activation = F.linear(f1_batch.to(self.f2_weights.dtype), self.f2_weights)
        return self._maybe_squeeze(activation, squeeze)

    @torch.no_grad()
    def generate_prediction(self, f2_activation: torch.Tensor) -> torch.Tensor:
        """Generate a top-down prediction in F1 space."""

        f2_batch, squeeze = self._ensure_batch(f2_activation)
        prediction = F.linear(f2_batch.to(self.feedback_weights.dtype), self.feedback_weights)
        return self._maybe_squeeze(prediction, squeeze)

    @torch.no_grad()
    def forward(self, raw_input: torch.Tensor, timestep: int = 0) -> CCCOutput:
        """Run the full CCC processing pipeline."""

        raw_batch, squeeze = self._ensure_batch(raw_input)
        f1_batch = self._ensure_batch(self.f1_encode(raw_batch))[0]
        f2_batch = self._ensure_batch(self.f2_activate(f1_batch))[0]

        self.age.add_(raw_batch.shape[0])

        if not bool(self.is_committed.item()):
            return self._make_abstention_output(f1_batch, f2_batch, squeeze=squeeze)

        gate_output = self.margin_gate(f2_batch, self.concept_direction)
        fired = bool(gate_output.fired.any().item())
        abstained = bool(gate_output.abstained.all().item())

        prediction: torch.Tensor | None = None
        resonance: ResonanceOutput | None = None

        if fired:
            prediction = self._ensure_batch(self.generate_prediction(gate_output.output))[0]
            resonance = self.margin_gate.check_resonance(prediction, f1_batch)
            if bool(resonance.resonated.any().item()):
                self.learn_slow(raw_batch, f1_batch, resonance)
            self.last_fired.fill_(int(timestep))

        formatted_gate = self._format_gate_output(gate_output, squeeze=squeeze)
        formatted_resonance = (
            self._format_resonance_output(resonance, squeeze=squeeze)
            if resonance is not None
            else None
        )

        return CCCOutput(
            fired=fired,
            abstained=abstained,
            confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_batch, squeeze),
            f2_activation=self._maybe_squeeze(f2_batch, squeeze),
            gate_output=formatted_gate,
            prediction=self._maybe_squeeze(prediction, squeeze) if prediction is not None else None,
            resonance=formatted_resonance,
        )

    @torch.no_grad()
    def learn_fast(self, raw_input: torch.Tensor, f1_output: torch.Tensor) -> None:
        """Commit an unassigned CCC to a new concept in one shot."""

        del raw_input
        f1_batch = self._ensure_batch(f1_output)[0]
        prototype_f2 = self._ensure_batch(self.f2_activate(f1_batch))[0].mean(dim=0)
        updated_direction = self.concept_direction + (self.config.fast_lr * prototype_f2)
        self.concept_direction.copy_(self._normalize_vector(updated_direction))

        prototype_f1 = f1_batch.mean(dim=0)
        feedback_init = prototype_f1.unsqueeze(-1) * self.concept_direction.unsqueeze(0)
        self.feedback_weights.copy_(feedback_init)
        self.is_committed.fill_(True)

    @torch.no_grad()
    def learn_slow(
        self,
        raw_input: torch.Tensor,
        f1_output: torch.Tensor,
        resonance: ResonanceOutput,
    ) -> None:
        """Apply local Hebbian refinement after resonance."""

        del raw_input
        if not bool(self.is_committed.item()):
            return

        f1_batch = self._ensure_batch(f1_output)[0]
        f2_batch = self._ensure_batch(self.f2_activate(f1_batch))[0]
        learn_signal = self._prepare_batch_vector(
            resonance.learn_signal, f2_batch.shape[0]
        ).to(f2_batch.dtype)

        concept_delta = (f2_batch * learn_signal.unsqueeze(-1)).mean(dim=0)
        updated_direction = self.concept_direction + (self.config.slow_lr * concept_delta)
        self.concept_direction.copy_(self._normalize_vector(updated_direction))

        prediction = self._ensure_batch(self.generate_prediction(f2_batch))[0]
        residual = (f1_batch - prediction) * learn_signal.unsqueeze(-1)
        hebbian_update = residual.transpose(0, 1) @ f2_batch
        hebbian_update /= max(f1_batch.shape[0], 1)
        self.feedback_weights.add_(self.config.feedback_lr * hebbian_update)
        self.feedback_weights.copy_(self._normalize_feedback_rows(self.feedback_weights))

    @torch.no_grad()
    def get_info(self) -> dict[str, bool | float | int | dict[str, float | int]]:
        """Return CCC status and gate statistics."""

        return {
            "is_committed": bool(self.is_committed.item()),
            "age": int(self.age.item()),
            "last_fired": int(self.last_fired.item()),
            "concept_direction_norm": float(self.concept_direction.norm().item()),
            "margin_gate": self.margin_gate.get_stats(),
        }


class CCCPool(nn.Module):
    """Pool of concept cell clusters with fast recruitment."""

    def __init__(self, config: CCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config
        self.cccs = nn.ModuleList(
            [ConceptCellCluster(config, margin_config) for _ in range(config.max_pool_size)]
        )
        if self.cccs:
            shared_f1 = self.cccs[0].f1_layer
            for ccc in self.cccs[1:]:
                ccc.f1_layer = shared_f1

    @staticmethod
    def _confidence_score(confidence: torch.Tensor) -> torch.Tensor:
        return confidence.reshape(-1).mean()

    def _first_uncommitted_index(self) -> int | None:
        for index, ccc in enumerate(self.cccs):
            if not bool(ccc.is_committed.item()):
                return index
        return None

    @torch.no_grad()
    def forward(self, raw_input: torch.Tensor, timestep: int = 0) -> CCCPoolOutput:
        """Run all committed CCCs and recruit a new one if all abstain."""

        outputs: list[CCCOutput] = []
        for ccc in self.cccs:
            if bool(ccc.is_committed.item()):
                outputs.append(ccc(raw_input, timestep=timestep))
            else:
                outputs.append(ccc.empty_output(raw_input))

        recruited = False
        recruited_index: int | None = None
        if not any(output.fired for output in outputs):
            recruited_index = self._first_uncommitted_index()
            if recruited_index is not None:
                recruited = True
                recruited_ccc = self.cccs[recruited_index]
                f1_output = recruited_ccc.f1_encode(raw_input)
                recruited_ccc.learn_fast(raw_input, f1_output)
                outputs[recruited_index] = recruited_ccc(raw_input, timestep=timestep)

        fired_indices = [index for index, output in enumerate(outputs) if output.fired]
        abstained_indices = [index for index, output in enumerate(outputs) if output.abstained]
        winner_confidences = (
            torch.stack(
                [self._confidence_score(outputs[index].confidence) for index in fired_indices]
            )
            if fired_indices
            else torch.empty(0, dtype=torch.float32)
        )

        return CCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=recruited,
            recruited_index=recruited_index,
            winner_confidences=winner_confidences,
        )

    @torch.no_grad()
    def get_winners(self, pool_output: CCCPoolOutput, k: int = 5) -> list[int]:
        """Return the top-k firing CCC indices by confidence."""

        if not pool_output.fired_indices:
            return []

        top_k = min(k, len(pool_output.fired_indices))
        _, top_indices = torch.topk(pool_output.winner_confidences, k=top_k)
        return [pool_output.fired_indices[index] for index in top_indices.tolist()]

    @torch.no_grad()
    def get_pool_stats(self) -> dict[str, float | int]:
        """Return high-level pool statistics."""

        num_committed = sum(bool(ccc.is_committed.item()) for ccc in self.cccs)
        total_presentations = sum(
            int(ccc.margin_gate.total_presentations.item()) for ccc in self.cccs
        )
        total_fires = sum(int(ccc.margin_gate.total_fires.item()) for ccc in self.cccs)
        total_confidence = sum(
            float(ccc.margin_gate.avg_confidence_when_fired.item())
            * int(ccc.margin_gate.total_fires.item())
            for ccc in self.cccs
        )
        mean_confidence = total_confidence / total_fires if total_fires else 0.0
        fire_rate = total_fires / total_presentations if total_presentations else 0.0

        return {
            "num_committed": num_committed,
            "num_uncommitted": len(self.cccs) - num_committed,
            "mean_confidence": float(mean_confidence),
            "fire_rate": float(fire_rate),
            "total_concepts": len(self.cccs),
        }


__all__ = [
    "CCCOutput",
    "CCCPool",
    "CCCPoolOutput",
    "ConceptCellCluster",
]
