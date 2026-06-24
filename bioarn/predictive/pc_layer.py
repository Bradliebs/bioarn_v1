"""Predictive coding layers for hierarchical error minimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bioarn.config import PredictiveConfig
from bioarn.core.math_utils import normalize


def _as_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Convert a tensor to batch-first form."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0), True
    return tensor, False


def _restore_shape(tensor: torch.Tensor, squeeze: bool) -> torch.Tensor:
    """Restore a tensor to its original rank."""
    if squeeze and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


@dataclass
class PCLayerOutput:
    """Outputs from a predictive coding layer update."""

    prediction: torch.Tensor
    error: torch.Tensor
    state: torch.Tensor
    precision: torch.Tensor
    error_magnitude: float
    suppressed_fraction: float


@dataclass
class PCStackOutput:
    """Outputs from a predictive coding hierarchy."""

    layer_outputs: list[PCLayerOutput]
    errors: list[torch.Tensor]
    states: list[torch.Tensor]
    precisions: list[torch.Tensor]
    free_energy: float
    free_energy_trace: list[float]


class PCLayer(nn.Module):
    """Single predictive coding layer with local Hebbian learning."""

    def __init__(self, input_dim: int, output_dim: int, config: PredictiveConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.activation_fn = torch.relu
        self.precision_alpha = 0.9
        self.eps = 1e-6

        weights = torch.randn(output_dim, input_dim) * 0.05
        weights = normalize(weights)
        self.W = nn.Parameter(weights, requires_grad=False)

        self.register_buffer(
            "precision",
            torch.full((input_dim,), float(config.precision_init)),
        )
        self.register_buffer("state", torch.zeros(output_dim))

        self._working_state: torch.Tensor | None = None
        self._last_suppression_mask: torch.Tensor | None = None

    def _resolve_state(
        self,
        batch_size: int,
        higher_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if higher_state is None:
            return self.state.unsqueeze(0).expand(batch_size, -1)

        state_batch, _ = _as_batch(higher_state)
        if state_batch.shape[0] == 1 and batch_size > 1:
            state_batch = state_batch.expand(batch_size, -1)
        return state_batch

    def _predict_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(state_batch @ self.W)

    def predict(self, higher_state: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a prediction for the level below."""
        if higher_state is None:
            return self._predict_from_state(self.state.unsqueeze(0)).squeeze(0)

        state_batch, squeezed = _as_batch(higher_state)
        prediction = self._predict_from_state(state_batch)
        return _restore_shape(prediction, squeezed)

    def compute_error(
        self,
        actual_input: torch.Tensor,
        prediction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute precision-weighted and PCL-suppressed prediction errors."""
        actual_batch, squeezed = _as_batch(actual_input)
        if prediction is None:
            prediction = self.predict()
        prediction_batch, _ = _as_batch(prediction)
        if prediction_batch.shape[0] == 1 and actual_batch.shape[0] > 1:
            prediction_batch = prediction_batch.expand(actual_batch.shape[0], -1)

        raw_error = actual_batch - prediction_batch
        weighted_error = raw_error * self.precision.unsqueeze(0)
        suppression_mask = weighted_error.abs() < self.config.error_threshold
        suppressed_error = weighted_error.masked_fill(suppression_mask, 0.0)

        self._last_suppression_mask = suppression_mask.detach()
        return _restore_shape(suppressed_error, squeezed)

    def update_state(self, error: torch.Tensor) -> torch.Tensor:
        """Update the hidden state from bottom-up prediction errors."""
        error_batch, squeezed = _as_batch(error)
        current_state = self._working_state
        if current_state is None:
            current_state = self.state.unsqueeze(0).expand(error_batch.shape[0], -1)
        elif current_state.shape[0] == 1 and error_batch.shape[0] > 1:
            current_state = current_state.expand(error_batch.shape[0], -1)

        delta = self.config.gamma * (error_batch @ self.W.t())
        updated_state = self.activation_fn(current_state + delta)

        with torch.no_grad():
            self.state.copy_(updated_state.detach().mean(dim=0))

        self._working_state = updated_state.detach()
        return _restore_shape(updated_state, squeezed)

    def update_weights(self, error: torch.Tensor, state: torch.Tensor | None = None) -> None:
        """Apply local Hebbian learning and renormalize weight rows."""
        error_batch, _ = _as_batch(error)
        if state is None:
            if self._working_state is None:
                state_batch = self.state.unsqueeze(0).expand(error_batch.shape[0], -1)
            else:
                state_batch = self._working_state
        else:
            state_batch, _ = _as_batch(state)

        if state_batch.shape[0] == 1 and error_batch.shape[0] > 1:
            state_batch = state_batch.expand(error_batch.shape[0], -1)

        delta = self.config.eta * (state_batch.t() @ error_batch) / error_batch.shape[0]

        with torch.no_grad():
            self.W.add_(delta)
            self.W.copy_(normalize(self.W))

    def update_precision(self, error: torch.Tensor) -> None:
        """Adapt precision inversely to prediction error magnitude."""
        error_batch, _ = _as_batch(error)
        error_magnitude = error_batch.abs().mean(dim=0).clamp_min(self.eps)
        target_precision = 1.0 / error_magnitude
        updated_precision = (
            self.precision_alpha * self.precision
            + (1.0 - self.precision_alpha) * target_precision
        )

        with torch.no_grad():
            self.precision.copy_(updated_precision.clamp(0.1, 10.0))

    def forward(
        self,
        actual_input: torch.Tensor,
        higher_state: torch.Tensor | None = None,
        learn: bool = True,
    ) -> PCLayerOutput:
        """Run predict → error → state update → local learning."""
        actual_batch, squeezed = _as_batch(actual_input)
        state_batch = self._resolve_state(actual_batch.shape[0], higher_state)
        self._working_state = state_batch.detach()

        prediction_batch = self._predict_from_state(state_batch)
        error_batch = self.compute_error(actual_batch, prediction_batch)
        error_batch, _ = _as_batch(error_batch)
        updated_state_batch = self.update_state(error_batch)
        updated_state_batch, _ = _as_batch(updated_state_batch)

        if learn:
            self.update_weights(error_batch, updated_state_batch)
            self.update_precision(error_batch)

        suppressed_fraction = 0.0
        if self._last_suppression_mask is not None:
            suppressed_fraction = float(self._last_suppression_mask.float().mean().item())

        output = PCLayerOutput(
            prediction=_restore_shape(prediction_batch, squeezed),
            error=_restore_shape(error_batch, squeezed),
            state=_restore_shape(updated_state_batch, squeezed),
            precision=self.precision.detach().clone(),
            error_magnitude=float(error_batch.abs().mean().item()),
            suppressed_fraction=suppressed_fraction,
        )

        self._working_state = None
        return output

    def reset_state(self) -> None:
        """Reset the persistent state for a new episode."""
        with torch.no_grad():
            self.state.zero_()
        self._working_state = None


class PCStack(nn.Module):
    """Hierarchical stack of predictive coding layers."""

    def __init__(self, layer_dims: list[int], config: PredictiveConfig):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least two levels.")

        self.layer_dims = layer_dims
        self.config = config
        self.layers = nn.ModuleList(
            [
                PCLayer(input_dim=layer_dims[idx], output_dim=layer_dims[idx + 1], config=config)
                for idx in range(len(layer_dims) - 1)
            ]
        )

    def forward(
        self,
        sensory_input: torch.Tensor,
        num_iterations: int = 5,
        learn: bool = True,
    ) -> PCStackOutput:
        """Iteratively settle the hierarchy toward lower free energy."""
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1.")

        current_input, _ = _as_batch(sensory_input)
        final_outputs: list[PCLayerOutput] = []
        free_energy_trace: list[float] = []

        for _ in range(num_iterations):
            iteration_outputs: list[PCLayerOutput] = []
            layer_input = current_input

            for layer in self.layers:
                output = layer(layer_input, learn=learn)
                iteration_outputs.append(output)
                layer_input, _ = _as_batch(output.state)

            energy = free_energy(
                [output.error for output in iteration_outputs],
                [output.precision for output in iteration_outputs],
            )
            free_energy_trace.append(float(energy.item()))
            final_outputs = iteration_outputs

        return PCStackOutput(
            layer_outputs=final_outputs,
            errors=[output.error for output in final_outputs],
            states=[output.state for output in final_outputs],
            precisions=[output.precision for output in final_outputs],
            free_energy=free_energy_trace[-1] if free_energy_trace else 0.0,
            free_energy_trace=free_energy_trace,
        )

    def generate(self, top_state: torch.Tensor, num_levels: int | None = None) -> torch.Tensor:
        """Cascade top-down predictions from the top state."""
        if num_levels is None:
            num_levels = len(self.layers)
        num_levels = max(1, min(num_levels, len(self.layers)))

        generated = top_state
        for layer in reversed(list(self.layers)[-num_levels:]):
            generated = layer.predict(generated)
        return generated

    def reset(self) -> None:
        """Reset all persistent layer states."""
        for layer in self.layers:
            layer.reset_state()


def free_energy(
    errors: list[torch.Tensor],
    precisions: list[torch.Tensor],
) -> torch.Tensor:
    """Compute the precision-weighted variational free energy."""
    if not errors:
        return torch.tensor(0.0)

    total = torch.zeros((), device=errors[0].device, dtype=errors[0].dtype)
    for error, precision in zip(errors, precisions, strict=False):
        error_batch, _ = _as_batch(error)
        precision_batch = precision
        while precision_batch.dim() < error_batch.dim():
            precision_batch = precision_batch.unsqueeze(0)
        total = total + (precision_batch * error_batch.pow(2)).sum(dim=-1).mean()

    return total
