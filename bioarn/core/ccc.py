"""Concept Cell Cluster (CCC) primitives for Bio-ARN."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.consolidation import SynapticConsolidation
from bioarn.core.margin_gate import MarginGate, MarginGateOutput, ResonanceOutput
from bioarn.core.math_utils import normalize, sparse_top_k
from bioarn.core.stdp import STDPRule


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


class F1Adapter(nn.Module):
    """Task-specific residual adapter applied after the shared F1 encoder."""

    def __init__(self, f1_dim: int, adapter_dim: int = 16):
        super().__init__()
        self.f1_dim = int(f1_dim)
        self.adapter_dim = int(max(1, adapter_dim))
        self.down = nn.Linear(self.f1_dim, self.adapter_dim, bias=False)
        self.up = nn.Linear(self.adapter_dim, self.f1_dim, bias=False)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        bootstrap = normalize(torch.randn(self.adapter_dim, self.f1_dim, dtype=torch.float32))
        self.register_buffer("bootstrap_projection", bootstrap)
        self.register_buffer("is_frozen", torch.tensor(False, dtype=torch.bool))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def freeze(self) -> None:
        self.is_frozen.fill_(True)

    def unfreeze(self) -> None:
        self.is_frozen.fill_(False)

    def forward(self, f1_output: torch.Tensor) -> torch.Tensor:
        return f1_output + self.up(F.relu(self.down(f1_output)))

    @staticmethod
    def _clamp_row_norms(weight: torch.Tensor, *, max_norm: float = 1.0) -> None:
        norms = weight.norm(dim=1, keepdim=True)
        scale = torch.clamp(norms / max(max_norm, 1e-6), min=1.0)
        weight.div_(scale)

    @torch.no_grad()
    def hebbian_update(
        self,
        f1_output: torch.Tensor,
        *,
        learning_signal: torch.Tensor | None = None,
        lr: float = 0.01,
    ) -> None:
        if bool(self.is_frozen.item()):
            return

        if f1_output.dim() == 1:
            f1_batch = f1_output.unsqueeze(0)
        elif f1_output.dim() == 2:
            f1_batch = f1_output
        else:
            raise ValueError("Adapter input must have shape (f1_dim,) or (batch, f1_dim).")
        if f1_batch.numel() == 0:
            return

        if learning_signal is None:
            signal = torch.ones(f1_batch.shape[0], device=f1_batch.device, dtype=f1_batch.dtype)
        else:
            signal = learning_signal.reshape(-1).to(device=f1_batch.device, dtype=f1_batch.dtype)
            if signal.numel() == 1 and f1_batch.shape[0] > 1:
                signal = signal.expand(f1_batch.shape[0])
            if signal.numel() != f1_batch.shape[0]:
                raise ValueError("learning_signal must align with the adapter batch size.")
        if not bool((signal > 0).any().item()):
            return

        bootstrap = F.relu(
            F.linear(
                f1_batch.to(self.bootstrap_projection.dtype),
                self.bootstrap_projection.to(device=f1_batch.device),
            )
        ).to(f1_batch.dtype)
        weighted_hidden = bootstrap * signal.unsqueeze(-1)
        weighted_input = f1_batch * signal.unsqueeze(-1)
        batch_norm = max(f1_batch.shape[0], 1)
        down_delta = weighted_hidden.transpose(0, 1) @ f1_batch
        up_delta = weighted_input.transpose(0, 1) @ bootstrap
        self.down.weight.add_(float(lr) * (down_delta / batch_norm).to(self.down.weight.dtype))
        self.up.weight.add_(float(lr) * (up_delta / batch_norm).to(self.up.weight.dtype))
        self._clamp_row_norms(self.down.weight.data)
        self._clamp_row_norms(self.up.weight.data)


class ConceptCellCluster(nn.Module):
    """A self-contained cortical-column-like concept circuit."""

    def __init__(self, config: CCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config
        self.stdp_rule = (
            STDPRule(config.stdp, num_pre=config.num_f1_features, num_post=config.concept_dim)
            if config.stdp is not None
            else None
        )

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
        self.register_buffer("importance", torch.tensor(0.0, dtype=torch.float32))
        self.adapter_id = -1
        object.__setattr__(self, "adapter", None)

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

    def set_adapter(self, adapter: F1Adapter | None, adapter_id: int = -1) -> None:
        object.__setattr__(self, "adapter", adapter)
        self.adapter_id = int(adapter_id)

    def _effective_lr(self, base_lr: float) -> float:
        scale = max(
            0.0,
            1.0 - (float(self.config.consolidation_strength) * float(self.importance.item())),
        )
        return float(base_lr) * scale

    def _adapter_lr(self, base_lr: float) -> float:
        return min(0.05, max(1e-3, float(base_lr)))

    def _stdp_enabled(self) -> bool:
        return self.stdp_rule is not None and bool(getattr(self.config, "stdp_enabled", True))

    @staticmethod
    def _pre_spikes_from_f1(f1_batch: torch.Tensor) -> torch.Tensor:
        return (f1_batch > 0).to(torch.float32).amax(dim=0)

    @staticmethod
    def _post_activity_from_f2(f2_batch: torch.Tensor) -> torch.Tensor:
        return f2_batch.to(torch.float32).clamp_min(0.0).mean(dim=0)

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
    def _base_f1_encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        raw_batch, squeeze = self._ensure_batch(raw_input)
        projected = self.f1_layer(raw_batch.to(self.f1_layer.weight.dtype))
        activated = F.relu(projected)
        sparse = sparse_top_k(activated, self.config.f1_top_k)
        return self._maybe_squeeze(sparse, squeeze)

    @torch.no_grad()
    def _apply_adapter(self, f1_output: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            return f1_output
        f1_batch, squeeze = self._ensure_batch(f1_output)
        adapted = self.adapter(f1_batch)
        adapted = sparse_top_k(F.relu(adapted), self.config.f1_top_k)
        return self._maybe_squeeze(adapted, squeeze)

    @torch.no_grad()
    def f1_encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Project raw input into sparse F1 feature space."""

        base_f1 = self._base_f1_encode(raw_input)
        return self._apply_adapter(base_f1)

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
    def preview(self, raw_input: torch.Tensor) -> CCCOutput:
        """Run a read-only CCC pass without updating learning state."""

        raw_batch, squeeze = self._ensure_batch(raw_input)
        f1_batch = self._ensure_batch(self.f1_encode(raw_batch))[0]
        f2_batch = self._ensure_batch(self.f2_activate(f1_batch))[0]

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
    @torch.no_grad()
    def forward(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> CCCOutput:
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
        pre_spikes = self._pre_spikes_from_f1(f1_batch) if self._stdp_enabled() else None

        prediction: torch.Tensor | None = None
        resonance: ResonanceOutput | None = None

        if fired:
            prediction = self._ensure_batch(self.generate_prediction(gate_output.output))[0]
            resonance = self.margin_gate.check_resonance(prediction, f1_batch)
            if bool(resonance.resonated.any().item()):
                self.learn_slow(
                    raw_batch,
                    f1_batch,
                    resonance,
                    timestep=timestep,
                    learning_rate_multiplier=learning_rate_multiplier,
                )
            elif pre_spikes is not None:
                self.stdp_rule.observe_pre_spikes(pre_spikes, timestep=timestep)
                self.stdp_rule.observe_post_spike(
                    self._post_activity_from_f2(gate_output.output),
                    timestep=timestep,
                )
            self.last_fired.fill_(int(timestep))
        elif pre_spikes is not None:
            self.stdp_rule.observe_pre_spikes(pre_spikes, timestep=timestep)

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
    def learn_fast(
        self,
        raw_input: torch.Tensor,
        f1_output: torch.Tensor,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        """Commit an unassigned CCC to a new concept in one shot."""

        lr_multiplier = float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item())
        base_f1_batch = self._ensure_batch(self._base_f1_encode(raw_input))[0]
        if self.adapter is not None:
            self.adapter.hebbian_update(
                base_f1_batch,
                learning_signal=torch.ones(
                    base_f1_batch.shape[0],
                    device=base_f1_batch.device,
                    dtype=base_f1_batch.dtype,
                ),
                lr=self._adapter_lr(self.config.fast_lr) * max(lr_multiplier, 0.0),
            )
            f1_batch = self._ensure_batch(self._apply_adapter(base_f1_batch))[0]
        else:
            f1_batch = self._ensure_batch(f1_output)[0]
        prototype_f2 = self._ensure_batch(self.f2_activate(f1_batch))[0].mean(dim=0)
        updated_direction = self.concept_direction + (
            max(0.0, float(self.config.fast_lr) * lr_multiplier) * prototype_f2
        )
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
        *,
        timestep: int | None = None,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        """Apply local Hebbian refinement after resonance."""

        if not bool(self.is_committed.item()):
            return

        f1_batch = self._ensure_batch(f1_output)[0]
        f2_batch = self._ensure_batch(self.f2_activate(f1_batch))[0]
        learn_signal = self._prepare_batch_vector(
            resonance.learn_signal, f2_batch.shape[0]
        ).to(f2_batch.dtype)
        stdp_signal = torch.zeros(self.config.concept_dim, dtype=f2_batch.dtype, device=f2_batch.device)
        stdp_update = torch.zeros_like(self.feedback_weights)
        if self._stdp_enabled() and timestep is not None:
            stdp_update = self.stdp_rule.step(
                self._pre_spikes_from_f1(f1_batch),
                post_spike=True,
                post_activity=self._post_activity_from_f2(f2_batch),
                timestep=timestep,
            ).to(self.feedback_weights.dtype)
            stdp_signal = stdp_update.mean(dim=0).to(f2_batch.dtype)

        concept_delta = (f2_batch * learn_signal.unsqueeze(-1)).mean(dim=0)
        concept_delta = concept_delta + stdp_signal
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        concept_lr = self._effective_lr(self.config.slow_lr) * lr_multiplier
        updated_direction = self.concept_direction + (concept_lr * concept_delta)
        self.concept_direction.copy_(self._normalize_vector(updated_direction))

        prediction = self._ensure_batch(self.generate_prediction(f2_batch))[0]
        residual = (f1_batch - prediction) * learn_signal.unsqueeze(-1)
        hebbian_update = residual.transpose(0, 1) @ f2_batch
        hebbian_update /= max(f1_batch.shape[0], 1)
        if self._stdp_enabled() and timestep is not None:
            hebbian_update = hebbian_update + stdp_update
        feedback_lr = self._effective_lr(self.config.feedback_lr) * lr_multiplier
        self.feedback_weights.add_(feedback_lr * hebbian_update)
        self.feedback_weights.copy_(self._normalize_feedback_rows(self.feedback_weights))
        if self.adapter is not None:
            base_f1_batch = self._ensure_batch(self._base_f1_encode(raw_input))[0]
            self.adapter.hebbian_update(
                base_f1_batch,
                learning_signal=learn_signal,
                lr=self._adapter_lr(self.config.slow_lr) * lr_multiplier,
            )

    @torch.no_grad()
    def get_info(self) -> dict[str, bool | float | int | dict[str, float | int]]:
        """Return CCC status and gate statistics."""

        return {
            "is_committed": bool(self.is_committed.item()),
            "age": int(self.age.item()),
            "last_fired": int(self.last_fired.item()),
            "adapter_id": int(self.adapter_id),
            "concept_direction_norm": float(self.concept_direction.norm().item()),
            "margin_gate": self.margin_gate.get_stats(),
        }


class CCCPool(nn.Module):
    """Pool of concept cell clusters with fast recruitment."""

    def __init__(self, config: CCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config
        self.margin_config = margin_config
        self.initial_capacity = int(config.max_pool_size)
        self.max_capacity = max(
            self.initial_capacity,
            int(math.ceil(self.initial_capacity * float(config.max_growth_factor))),
        )
        self.cccs = nn.ModuleList(
            [ConceptCellCluster(config, margin_config) for _ in range(self.initial_capacity)]
        )
        if self.cccs:
            shared_f1 = self.cccs[0].f1_layer
            for ccc in self.cccs[1:]:
                ccc.f1_layer = shared_f1
        self.task_adapters = nn.ModuleList()
        self.active_adapter_id = -1
        self.register_buffer("f1_samples_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("f1_frozen", torch.tensor(False, dtype=torch.bool))
        self.consolidation = SynapticConsolidation(
            len(self.cccs),
            strength=self.config.consolidation_strength,
        )
        self._sync_importance_buffers()

    @property
    def current_adapter(self) -> F1Adapter | None:
        if 0 <= self.active_adapter_id < len(self.task_adapters):
            return self.task_adapters[self.active_adapter_id]
        return None

    def freeze_f1(self) -> None:
        self.f1_frozen.fill_(True)
        if self.cccs:
            for parameter in self.cccs[0].f1_layer.parameters():
                parameter.requires_grad_(False)

    def observe_samples(self, sample_count: int) -> None:
        count = int(max(0, sample_count))
        if count <= 0:
            return
        self.f1_samples_seen.add_(count)
        if (
            not bool(self.f1_frozen.item())
            and int(self.config.freeze_f1_after) > 0
            and int(self.f1_samples_seen.item()) >= int(self.config.freeze_f1_after)
        ):
            self.freeze_f1()

    def create_task_adapter(self) -> F1Adapter | None:
        if int(self.config.freeze_f1_after) <= 0:
            return None
        self.freeze_f1()
        current = self.current_adapter
        if current is not None:
            current.freeze()
        adapter = F1Adapter(
            self.config.num_f1_features,
            adapter_dim=self.config.f1_adapter_dim,
        )
        adapter.unfreeze()
        self.task_adapters.append(adapter)
        self.active_adapter_id = len(self.task_adapters) - 1
        return adapter

    def _assign_current_adapter(self, ccc: ConceptCellCluster) -> None:
        ccc.set_adapter(self.current_adapter, self.active_adapter_id)

    def _sync_importance_buffers(self) -> None:
        self.consolidation.ensure_capacity(len(self.cccs))
        for index, ccc in enumerate(self.cccs):
            ccc.importance.copy_(self.consolidation.importance[index].to(ccc.importance))

    def grow(self, min_extra_slots: int = 1) -> int:
        current_size = len(self.cccs)
        if current_size >= self.max_capacity:
            return current_size

        growth_target = max(
            current_size + int(max(1, min_extra_slots)),
            int(math.ceil(current_size * 1.5)),
        )
        target_size = min(self.max_capacity, growth_target)
        if target_size <= current_size:
            return current_size

        shared_f1 = self.cccs[0].f1_layer if self.cccs else None
        for _ in range(target_size - current_size):
            ccc = ConceptCellCluster(self.config, self.margin_config)
            if shared_f1 is None:
                shared_f1 = ccc.f1_layer
            else:
                ccc.f1_layer = shared_f1
            self.cccs.append(ccc)
        self.consolidation.ensure_capacity(target_size)
        self.config.max_pool_size = target_size
        self._sync_importance_buffers()
        return target_size

    def update_importance(
        self,
        fired_indices: list[int] | torch.Tensor,
        *,
        confidences: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        scores = self.consolidation.update_importance(
            fired_indices,
            current_size=len(self.cccs),
            confidences=confidences,
        )
        self._sync_importance_buffers()
        return scores

    @staticmethod
    def _confidence_score(confidence: torch.Tensor) -> torch.Tensor:
        return confidence.reshape(-1).mean()

    def _first_uncommitted_index(self) -> int | None:
        for index, ccc in enumerate(self.cccs):
            if not bool(ccc.is_committed.item()):
                return index
        return None

    @torch.no_grad()
    def preview(self, raw_input: torch.Tensor) -> CCCPoolOutput:
        """Run the pool read-only without recruitment or learning."""

        outputs: list[CCCOutput] = []
        for ccc in self.cccs:
            if bool(ccc.is_committed.item()):
                outputs.append(ccc.preview(raw_input))
            else:
                outputs.append(ccc.empty_output(raw_input))

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
            recruited=False,
            recruited_index=None,
            winner_confidences=winner_confidences,
        )

    def forward(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> CCCPoolOutput:
        """Run all committed CCCs and recruit a new one if all abstain."""

        batch_size = raw_input.shape[0] if raw_input.dim() == 2 else 1
        self.observe_samples(batch_size)
        outputs: list[CCCOutput] = []
        for ccc in self.cccs:
            if bool(ccc.is_committed.item()):
                outputs.append(
                    ccc(
                        raw_input,
                        timestep=timestep,
                        learning_rate_multiplier=learning_rate_multiplier,
                    )
                )
            else:
                outputs.append(ccc.empty_output(raw_input))

        recruited = False
        recruited_index: int | None = None
        if not any(output.fired for output in outputs):
            recruited_index = self._first_uncommitted_index()
            if recruited_index is None:
                previous_size = len(self.cccs)
                new_size = self.grow()
                if new_size > previous_size:
                    outputs.extend(
                        self.cccs[index].empty_output(raw_input)
                        for index in range(previous_size, new_size)
                    )
                    recruited_index = self._first_uncommitted_index()
            if recruited_index is not None:
                recruited = True
                recruited_ccc = self.cccs[recruited_index]
                self._assign_current_adapter(recruited_ccc)
                f1_output = recruited_ccc.f1_encode(raw_input)
                recruited_ccc.learn_fast(
                    raw_input,
                    f1_output,
                    learning_rate_multiplier=learning_rate_multiplier,
                )
                outputs[recruited_index] = recruited_ccc(
                    raw_input,
                    timestep=timestep,
                    learning_rate_multiplier=learning_rate_multiplier,
                )

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
            "initial_capacity": self.initial_capacity,
            "max_capacity": self.max_capacity,
            "mean_importance": float(self.consolidation.importance[: len(self.cccs)].mean().item()),
            "f1_frozen": bool(self.f1_frozen.item()),
            "task_adapters": len(self.task_adapters),
        }


__all__ = [
    "CCCOutput",
    "F1Adapter",
    "CCCPool",
    "CCCPoolOutput",
    "ConceptCellCluster",
]
