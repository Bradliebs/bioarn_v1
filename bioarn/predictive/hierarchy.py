"""Hierarchical predictive coding with perception, generation, and resonance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import PredictiveConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import cosine_similarity
from bioarn.predictive.pc_layer import PCStack, free_energy


def _as_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0), True
    if tensor.dim() != 2:
        raise ValueError("Expected a 1D or 2D tensor.")
    return tensor, False


def _restore_shape(tensor: torch.Tensor, squeeze: bool) -> torch.Tensor:
    return tensor.squeeze(0) if squeeze and tensor.shape[0] == 1 else tensor


def _match_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1)
    raise ValueError("Batch size mismatch in predictive hierarchy.")


def _clone_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().clone() for tensor in tensors]


@dataclass
class HierarchyPerceptionOutput:
    """Settled states and errors after iterative perception."""

    states: list[torch.Tensor]
    errors: list[torch.Tensor]
    free_energy_trace: list[float]
    converged: bool
    iterations_used: int
    surprise: float


@dataclass
class HierarchyGenerationOutput:
    """Top-down predictive generation outputs."""

    generated_sensory: torch.Tensor
    level_predictions: list[torch.Tensor]
    generation_confidence: float


@dataclass
class PredictionQualityOutput:
    """Comparison between expected and observed sensory input."""

    prediction: torch.Tensor
    actual: torch.Tensor
    error: torch.Tensor
    surprise_score: float
    novel: bool


@dataclass
class ActionSignal:
    """Action proposal for active inference."""

    direction: torch.Tensor
    urgency: float
    expected_reduction: float


@dataclass
class ResonanceLoopOutput:
    """Outcome of bottom-up/top-down resonance settling."""

    resonated: bool
    iterations: int
    final_error: float
    concept_state: torch.Tensor
    free_energy_trace: list[float]


class PredictiveHierarchy(PCStack):
    """Four-level predictive coding hierarchy with perception and generation modes."""

    def __init__(self, layer_dims: list[int], config: PredictiveConfig):
        super().__init__(layer_dims=layer_dims, config=config)
        self.layer_dims = [int(dim) for dim in layer_dims]
        self.config = config
        self.convergence_epsilon = max(1e-4, float(config.error_threshold) * 0.1)
        self.novelty_threshold = max(0.05, float(config.error_threshold) * 5.0)
        self._last_states = [torch.zeros(dim, dtype=torch.float32) for dim in self.layer_dims]
        self._last_errors = [torch.zeros(dim, dtype=torch.float32) for dim in self.layer_dims]

    def _infer_generation_level(self, state_dim: int, explicit_level: int | None = None) -> int:
        if explicit_level is not None:
            candidate = int(explicit_level)
            if 0 <= candidate < len(self.layer_dims) and self.layer_dims[candidate] == state_dim:
                return candidate

        matches = [idx for idx, dim in enumerate(self.layer_dims) if dim == state_dim]
        hidden_matches = [idx for idx in matches if idx > 0]
        if hidden_matches:
            return hidden_matches[-1]
        if matches:
            return matches[-1]
        raise ValueError(f"No hierarchy level matches state dimension {state_dim}.")

    def _initial_states(self, sensory_batch: torch.Tensor) -> list[torch.Tensor]:
        batch_size = sensory_batch.shape[0]
        states = [sensory_batch]
        for layer in self.layers:
            state = layer.state.unsqueeze(0).to(device=sensory_batch.device, dtype=sensory_batch.dtype)
            states.append(state.expand(batch_size, -1).clone())
        return states

    def _evaluate_states(
        self, states: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
        predictions: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            prediction = layer.predict(states[idx + 1])
            prediction_batch, _ = _as_batch(prediction)
            predictions.append(prediction_batch)

            error = layer.compute_error(states[idx], prediction_batch)
            error_batch, _ = _as_batch(error)
            errors.append(error_batch)

        errors.append(torch.zeros_like(states[-1]))
        energy = float(
            free_energy(errors[:-1], [layer.precision.detach() for layer in self.layers]).item()
        )
        return predictions, errors, energy

    def _proposed_hidden_states(
        self, states: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        proposed: list[torch.Tensor] = []
        for level in range(1, len(states)):
            lower_drive = errors[level - 1] @ self.layers[level - 1].W.t()
            if level < len(states) - 1:
                delta = lower_drive - errors[level]
            else:
                delta = lower_drive
            proposed.append(torch.relu(states[level] + (self.config.gamma * delta)))
        return proposed

    def _commit_states(self, states: list[torch.Tensor], errors: list[torch.Tensor]) -> None:
        for idx, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.state.copy_(states[idx + 1].detach().mean(dim=0))
            layer.update_precision(errors[idx])

    def _store_last_state(self, states: list[torch.Tensor], errors: list[torch.Tensor], squeeze: bool) -> None:
        self._last_states = [_restore_shape(state.detach().clone(), squeeze) for state in states]
        self._last_errors = [_restore_shape(error.detach().clone(), squeeze) for error in errors]

    def _cascade_down(
        self, state_batch: torch.Tensor, start_level: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        generated = state_batch
        predictions: list[torch.Tensor] = []
        for layer_index in range(start_level - 1, -1, -1):
            generated = self.layers[layer_index].predict(generated)
            generated, _ = _as_batch(generated)
            predictions.append(generated.detach().clone())
        return generated, predictions

    def _confidence_from_precision(self, start_level: int) -> float:
        if not self.layers:
            return 1.0
        relevant_precisions = [
            float(self.layers[layer_index].precision.mean().item())
            for layer_index in range(max(0, start_level))
        ]
        if not relevant_precisions:
            return 1.0
        mean_precision = sum(relevant_precisions) / len(relevant_precisions)
        return float(mean_precision / (1.0 + mean_precision))

    def forward(
        self, sensory_input: torch.Tensor, num_iterations: int = 10
    ) -> HierarchyPerceptionOutput:
        return self.perceive(sensory_input=sensory_input, num_iterations=num_iterations)

    def perceive(
        self, sensory_input: torch.Tensor, num_iterations: int = 10
    ) -> HierarchyPerceptionOutput:
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1.")

        sensory_batch, squeeze = _as_batch(sensory_input.to(torch.float32))
        states = self._initial_states(sensory_batch)
        free_energy_trace: list[float] = []
        converged = False
        final_errors: list[torch.Tensor] = [torch.zeros_like(state) for state in states]
        previous_energy: float | None = None

        for _ in range(num_iterations):
            _, errors, current_energy = self._evaluate_states(states)
            proposed_hidden = self._proposed_hidden_states(states, errors)

            step_scale = 1.0
            accepted_states = states
            accepted_errors = errors
            accepted_energy = current_energy

            while step_scale >= 1e-3:
                candidate_states = [sensory_batch]
                for level in range(1, len(states)):
                    candidate_states.append(torch.lerp(states[level], proposed_hidden[level - 1], step_scale))

                _, candidate_errors, candidate_energy = self._evaluate_states(candidate_states)
                if candidate_energy <= current_energy + 1e-6:
                    accepted_states = candidate_states
                    accepted_errors = candidate_errors
                    accepted_energy = candidate_energy
                    break
                step_scale *= 0.5

            states = accepted_states
            final_errors = accepted_errors
            free_energy_trace.append(float(accepted_energy))

            if accepted_energy <= self.convergence_epsilon:
                converged = True
                break
            if previous_energy is not None and abs(previous_energy - accepted_energy) <= self.convergence_epsilon:
                converged = True
                break
            previous_energy = accepted_energy

        self._commit_states(states, final_errors)
        self._store_last_state(states, final_errors, squeeze=squeeze)

        surprise = (
            float(sum(error.abs().mean().item() for error in final_errors[:-1]))
            if final_errors
            else 0.0
        )
        return HierarchyPerceptionOutput(
            states=_clone_tensors(self._last_states),
            errors=_clone_tensors(self._last_errors),
            free_energy_trace=free_energy_trace,
            converged=converged,
            iterations_used=len(free_energy_trace),
            surprise=surprise,
        )

    def generate(
        self, top_state: torch.Tensor, num_levels: int | None = None
    ) -> HierarchyGenerationOutput:
        top_batch, squeeze = _as_batch(top_state.to(torch.float32))
        start_level = self._infer_generation_level(top_batch.shape[-1], explicit_level=num_levels)
        generated_sensory, level_predictions = self._cascade_down(top_batch, start_level)
        return HierarchyGenerationOutput(
            generated_sensory=_restore_shape(generated_sensory, squeeze),
            level_predictions=[_restore_shape(prediction, squeeze) for prediction in level_predictions],
            generation_confidence=self._confidence_from_precision(start_level),
        )

    def predict_and_compare(self, sensory_input: torch.Tensor) -> PredictionQualityOutput:
        actual_batch, squeeze = _as_batch(sensory_input.to(torch.float32))

        if self._last_states:
            predicted = _as_batch(self._last_states[0].to(torch.float32))[0]
        else:
            predicted = self.layers[0].predict()
        prediction_batch, _ = _as_batch(predicted)
        prediction_batch = _match_batch(prediction_batch, actual_batch.shape[0])

        error = actual_batch - prediction_batch
        surprise_score = float(error.abs().mean().item())
        return PredictionQualityOutput(
            prediction=_restore_shape(prediction_batch.detach().clone(), squeeze),
            actual=_restore_shape(actual_batch.detach().clone(), squeeze),
            error=_restore_shape(error.detach().clone(), squeeze),
            surprise_score=surprise_score,
            novel=bool(surprise_score > self.novelty_threshold),
        )

    def active_inference_step(
        self, current_state: torch.Tensor, goal_state: torch.Tensor
    ) -> ActionSignal:
        current_batch, squeeze = _as_batch(current_state.to(torch.float32))
        goal_batch, _ = _as_batch(goal_state.to(torch.float32))
        goal_batch = _match_batch(goal_batch, current_batch.shape[0])

        direction = goal_batch - current_batch
        if direction.shape[-1] == self.layer_dims[0]:
            precision = self.layers[0].precision.to(device=direction.device, dtype=direction.dtype)
            direction = direction * precision.unsqueeze(0)

        urgency = float(direction.abs().mean().item())
        precision_gain = float(self.layers[0].precision.mean().item()) if self.layers else 1.0
        expected_reduction = float(urgency * min(1.0, self.config.gamma * precision_gain))
        return ActionSignal(
            direction=_restore_shape(direction.detach().clone(), squeeze),
            urgency=urgency,
            expected_reduction=expected_reduction,
        )

    def get_level_states(self) -> list[torch.Tensor]:
        return _clone_tensors(self._last_states)

    def get_level_errors(self) -> list[torch.Tensor]:
        return _clone_tensors(self._last_errors)

    def get_precision_map(self) -> list[torch.Tensor]:
        if not self.layers:
            return []

        precision_map = [layer.precision.detach().clone() for layer in self.layers]
        top_precision = torch.full(
            (self.layer_dims[-1],),
            float(precision_map[-1].mean().item()),
            device=precision_map[-1].device,
            dtype=precision_map[-1].dtype,
        )
        precision_map.append(top_precision)
        return precision_map

    def reset(self) -> None:
        super().reset()
        device = self.layers[0].state.device
        dtype = self.layers[0].state.dtype
        self._last_states = [
            torch.zeros(dim, device=device, dtype=dtype) for dim in self.layer_dims
        ]
        self._last_errors = [
            torch.zeros(dim, device=device, dtype=dtype) for dim in self.layer_dims
        ]


class HierarchyConnector(nn.Module):
    """Bridge between the predictive hierarchy and the CCC concept pool."""

    def __init__(self, hierarchy: PredictiveHierarchy, ccc_pool: CCCPool, config: PredictiveConfig):
        super().__init__()
        self.hierarchy = hierarchy
        self.ccc_pool = ccc_pool
        self.config = config
        self.level2_index = min(2, len(self.hierarchy.layer_dims) - 1)

    @staticmethod
    def _align_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        batch, squeeze = _as_batch(tensor.to(torch.float32))
        current_dim = batch.shape[-1]
        if current_dim == target_dim:
            return _restore_shape(batch, squeeze)
        if current_dim > target_dim:
            return _restore_shape(batch[..., :target_dim], squeeze)
        return _restore_shape(F.pad(batch, (0, target_dim - current_dim)), squeeze)

    def _resonance_threshold(self) -> float:
        if not self.ccc_pool.cccs:
            return 0.9
        return float(self.ccc_pool.cccs[0].margin_gate.theta_resonance.item())

    def bottom_up(self, sensory: torch.Tensor) -> torch.Tensor:
        perception = self.hierarchy.perceive(sensory)
        level2_state = perception.states[self.level2_index]
        return self._align_dim(level2_state, self.ccc_pool.config.input_dim)

    def top_down(self, concept_direction: torch.Tensor) -> torch.Tensor:
        injected = self._align_dim(concept_direction, self.hierarchy.layer_dims[self.level2_index])
        generated = self.hierarchy.generate(injected, num_levels=self.level2_index)
        return generated.generated_sensory

    def resonance_loop(
        self, sensory: torch.Tensor, concept: torch.Tensor, max_iters: int = 20
    ) -> ResonanceLoopOutput:
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1.")

        sensory_batch, squeeze = _as_batch(sensory.to(torch.float32))
        concept_state = _as_batch(
            self._align_dim(concept, self.hierarchy.layer_dims[self.level2_index]).to(torch.float32)
        )[0]

        resonated = False
        final_error = float("inf")
        free_energy_trace: list[float] = []
        threshold = self._resonance_threshold()

        for _ in range(max_iters):
            bottom_up_state = _as_batch(
                self._align_dim(self.bottom_up(sensory_batch), self.hierarchy.layer_dims[self.level2_index])
            )[0]
            concept_state = torch.relu(0.5 * concept_state + 0.5 * bottom_up_state)

            predicted_sensory = _as_batch(self.top_down(concept_state))[0]
            error = sensory_batch - _match_batch(predicted_sensory, sensory_batch.shape[0])
            final_error = float(error.abs().mean().item())
            match_score = float(
                cosine_similarity(
                    predicted_sensory,
                    _match_batch(sensory_batch, predicted_sensory.shape[0]),
                )
                .mean()
                .item()
            )
            energy = 0.5 * final_error + 0.5 * (1.0 - match_score)
            free_energy_trace.append(float(energy))

            if match_score >= threshold and final_error <= max(0.1, self.hierarchy.novelty_threshold):
                resonated = True
                if bottom_up_state.shape[-1] == self.ccc_pool.config.input_dim:
                    self.ccc_pool(_restore_shape(bottom_up_state.detach().clone(), squeeze))
                break

            correction = error
            for layer_index in range(self.level2_index):
                correction = correction @ self.hierarchy.layers[layer_index].W.t()
            concept_state = torch.relu(concept_state + (self.config.gamma * correction))

        return ResonanceLoopOutput(
            resonated=resonated,
            iterations=len(free_energy_trace),
            final_error=final_error,
            concept_state=_restore_shape(concept_state.detach().clone(), squeeze),
            free_energy_trace=free_energy_trace,
        )


__all__ = [
    "ActionSignal",
    "HierarchyConnector",
    "HierarchyGenerationOutput",
    "HierarchyPerceptionOutput",
    "PredictionQualityOutput",
    "PredictiveHierarchy",
    "ResonanceLoopOutput",
]
