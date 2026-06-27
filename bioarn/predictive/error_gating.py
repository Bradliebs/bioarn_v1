"""Prediction-error-gated Hebbian learning for the visual hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bioarn.config import PredictiveConfig
from bioarn.predictive.pc_layer import PCStack, _as_batch, _restore_shape, free_energy


def _match_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1)
    raise ValueError("Batch size mismatch in prediction error gating.")


@dataclass
class ErrorGatingOutput:
    """Single-pass prediction errors and Hebbian learning gates."""

    states: list[torch.Tensor]
    errors: list[torch.Tensor]
    gates: list[torch.Tensor]
    free_energy_trace: list[float]
    surprise: float


class PredictionErrorGate(PCStack):
    """Use top-down prediction errors to gate local Hebbian learning."""

    def __init__(self, layer_dims: list[int], config: PredictiveConfig):
        super().__init__(layer_dims=layer_dims, config=config)
        self.error_scale = float(config.gamma)
        self.error_baseline = max(float(config.error_threshold), 1e-3)
        self.novelty_threshold = max(0.05, float(config.error_threshold) * 5.0)

    def predict(self, higher_state: torch.Tensor, *, target_level: int) -> torch.Tensor:
        """Predict one hierarchy level from the level above it."""

        return self.layers[target_level].predict(higher_state)

    def compute_error_gates(
        self,
        level_activations: list[torch.Tensor],
        *,
        higher_activations: list[torch.Tensor] | None = None,
        learn: bool = False,
    ) -> ErrorGatingOutput:
        """Compute per-level prediction-error gates in a single forward pass."""

        if len(level_activations) != len(self.layer_dims):
            raise ValueError("level_activations must match the predictive hierarchy depth.")
        if higher_activations is not None and len(higher_activations) != len(self.layers):
            raise ValueError("higher_activations must provide one predictor state per predictive layer.")

        states: list[torch.Tensor] = []
        squeeze_flags: list[bool] = []
        batch_size: int | None = None

        for activation, expected_dim in zip(level_activations, self.layer_dims, strict=False):
            state_batch, squeeze = _as_batch(activation.to(torch.float32))
            if state_batch.shape[-1] != expected_dim:
                raise ValueError(
                    f"Prediction error gate received dim {state_batch.shape[-1]}, expected {expected_dim}."
                )
            if higher_activations is None:
                if batch_size is None:
                    batch_size = state_batch.shape[0]
                state_batch = _match_batch(state_batch, batch_size)
            states.append(state_batch)
            squeeze_flags.append(squeeze)

        if batch_size is None and higher_activations is None:
            raise ValueError("Prediction error gating requires at least one hierarchy level.")

        predictor_states: list[torch.Tensor] = []
        if higher_activations is None:
            predictor_states = states[1:]
        else:
            for activation, expected_dim in zip(
                higher_activations,
                self.layer_dims[1:],
                strict=False,
            ):
                predictor_batch, _ = _as_batch(activation.to(torch.float32))
                if predictor_batch.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Prediction error gate received predictor dim {predictor_batch.shape[-1]}, expected {expected_dim}."
                    )
                predictor_states.append(predictor_batch)

        errors: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        precisions = [layer.precision.detach() for layer in self.layers]

        for level, layer in enumerate(self.layers):
            actual = states[level]
            higher_state = _match_batch(predictor_states[level], actual.shape[0])
            predicted = _as_batch(self.predict(higher_state, target_level=level))[0]
            predicted = _match_batch(predicted, actual.shape[0])
            error = actual - predicted
            errors.append(error.detach().clone())

            error_magnitude = error.abs().mean(dim=-1, keepdim=True)
            normalized_error = error_magnitude / self.error_baseline
            gate = torch.sigmoid(normalized_error * self.error_scale).clamp(0.0, 1.0)
            actual_magnitude = actual.abs().mean(dim=-1, keepdim=True)
            higher_magnitude = higher_state.abs().mean(dim=-1, keepdim=True)
            inactive_context = (actual_magnitude <= self.error_baseline) | (
                higher_magnitude <= self.error_baseline
            )
            gate = torch.where(inactive_context, torch.ones_like(gate), gate)
            gates.append(gate.detach().clone())

            if learn:
                with torch.no_grad():
                    layer.state.copy_(higher_state.detach().mean(dim=0))
                layer.update_weights(error, higher_state)
                layer.update_precision(error)

        errors.append(torch.zeros_like(states[-1]))
        top_batch_size = states[-1].shape[0]
        gates.append(
            torch.ones((top_batch_size, 1), device=states[-1].device, dtype=states[-1].dtype)
        )

        energy = float(free_energy(errors[:-1], precisions).item()) if errors[:-1] else 0.0
        surprise = (
            float(sum(error.abs().mean().item() for error in errors[:-1]) / max(len(errors) - 1, 1))
            if errors
            else 0.0
        )

        return ErrorGatingOutput(
            states=[
                _restore_shape(state.detach().clone(), squeeze)
                for state, squeeze in zip(states, squeeze_flags, strict=False)
            ],
            errors=[
                _restore_shape(error.detach().clone(), squeeze)
                for error, squeeze in zip(errors[:-1], squeeze_flags[:-1], strict=False)
            ]
            + [_restore_shape(errors[-1].detach().clone(), squeeze_flags[-1])],
            gates=[
                _restore_shape(gate.detach().clone(), squeeze)
                for gate, squeeze in zip(gates, squeeze_flags, strict=False)
            ],
            free_energy_trace=[energy],
            surprise=surprise,
        )


__all__ = ["ErrorGatingOutput", "PredictionErrorGate"]
