"""Spiking neuron primitives and encoders for Bio-ARN."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from bioarn.config import SpikingConfig


class SurrogateSpike(torch.autograd.Function):
    """Binary spike with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (input_tensor,) = ctx.saved_tensors
        slope = 25.0
        grad = 1.0 / torch.pow(1.0 + slope * input_tensor.abs(), 2)
        return grad_output * grad


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with refractory dynamics."""

    def __init__(
        self,
        num_neurons: Optional[int] = None,
        config: Optional[SpikingConfig] = None,
    ) -> None:
        super().__init__()
        config = config or SpikingConfig()
        self.num_neurons = num_neurons
        self.beta = float(config.beta)
        self.threshold = float(config.threshold)
        self.reset = float(config.reset)
        self.dt = float(config.dt)
        self.refractory_steps = int(config.refractory_steps)

        self.register_buffer("V", torch.empty(0))
        self.register_buffer("refractory_counter", torch.empty(0, dtype=torch.long))

    @torch.no_grad()
    def _ensure_state(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        expected_shape = (batch_size, num_neurons)
        if self.V.shape != expected_shape or self.V.device != device or self.V.dtype != dtype:
            self.V = torch.full(expected_shape, self.reset, device=device, dtype=dtype)
        if self.refractory_counter.shape != expected_shape or self.refractory_counter.device != device:
            self.refractory_counter = torch.zeros(expected_shape, device=device, dtype=torch.long)

    @torch.no_grad()
    def reset_state(
        self,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if batch_size is None or self.num_neurons is None:
            empty_kwargs = {}
            if device is not None:
                empty_kwargs["device"] = device
            self.V = torch.empty(0, dtype=dtype or torch.float32, **empty_kwargs)
            self.refractory_counter = torch.empty(0, dtype=torch.long, **empty_kwargs)
            return

        if self.V.numel():
            target_device = device or self.V.device
            target_dtype = dtype or self.V.dtype
        else:
            target_device = device or torch.device("cpu")
            target_dtype = dtype or torch.float32
        self.V = torch.full((batch_size, self.num_neurons), self.reset, device=target_device, dtype=target_dtype)
        self.refractory_counter = torch.zeros(
            (batch_size, self.num_neurons),
            device=target_device,
            dtype=torch.long,
        )

    def _forward_step(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_current.dim() != 2:
            raise ValueError("Single-step input must have shape (batch, neurons).")

        batch_size, num_neurons = input_current.shape
        if self.num_neurons is not None and num_neurons != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} neurons, received {num_neurons}."
            )

        self._ensure_state(batch_size, num_neurons, input_current.device, input_current.dtype)

        refractory_mask = self.refractory_counter > 0
        integrated = self.beta * self.V + (self.dt * input_current)
        reset_value = torch.full_like(integrated, self.reset)
        integrated = torch.where(refractory_mask, reset_value, integrated)

        raw_spike = SurrogateSpike.apply(integrated - self.threshold)
        spike = raw_spike * (~refractory_mask).to(input_current.dtype)
        next_voltage = torch.where(spike > 0, reset_value, integrated)

        with torch.no_grad():
            next_refractory = torch.clamp(self.refractory_counter - 1, min=0)
            if self.refractory_steps > 0:
                next_refractory = torch.where(
                    spike > 0,
                    torch.full_like(next_refractory, self.refractory_steps),
                    next_refractory,
                )
            self.V.copy_(next_voltage.detach())
            self.refractory_counter.copy_(next_refractory)

        return spike, next_voltage

    def forward(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process single-step or multi-step current input."""

        if input_current.dim() == 2:
            return self._forward_step(input_current)

        if input_current.dim() != 3:
            raise ValueError(
                "Input must have shape (batch, neurons) or (time, batch, neurons)."
            )

        spikes = []
        voltages = []
        for current_t in input_current.unbind(dim=0):
            spike_t, voltage_t = self._forward_step(current_t)
            spikes.append(spike_t)
            voltages.append(voltage_t)

        return torch.stack(spikes, dim=0), torch.stack(voltages, dim=0)


class LIFLayer(nn.Module):
    """Linear projection followed by LIF spiking dynamics."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        config: Optional[SpikingConfig] = None,
        spike_history_steps: int = 16,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = LIFNeuron(num_neurons=out_features, config=config)
        self.spike_history_steps = int(spike_history_steps)
        self.register_buffer("spike_history", torch.empty(0))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.neuron.reset_state()
        self.spike_history = torch.empty(0, device=self.linear.weight.device, dtype=self.linear.weight.dtype)

    @torch.no_grad()
    def _update_spike_history(self, spikes: torch.Tensor) -> None:
        if self.spike_history_steps <= 0:
            self.spike_history = torch.empty(0, device=spikes.device, dtype=spikes.dtype)
            return

        history = spikes.detach()
        if history.dim() == 2:
            history = history.unsqueeze(0)

        history = history[-self.spike_history_steps :].clone()
        if self.spike_history.numel() == 0 or self.spike_history.shape[1:] != history.shape[1:]:
            self.spike_history = history
            return

        self.spike_history = torch.cat((self.spike_history, history), dim=0)[-self.spike_history_steps :]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.linear(inputs)
        spikes, membrane = self.neuron(current)
        self._update_spike_history(spikes)
        return spikes, membrane


def rate_encode(data: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Encode values in [0, 1] as Bernoulli spike trains."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    probabilities = data.clamp(0.0, 1.0)
    random_values = torch.rand(
        (num_steps, *probabilities.shape),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return (random_values < probabilities.unsqueeze(0)).to(probabilities.dtype)


def latency_encode(data: torch.Tensor, num_steps: int, tau: float) -> torch.Tensor:
    """Encode higher values as earlier single spikes."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")

    values = data.clamp(0.0, 1.0)
    flat_values = values.reshape(-1)
    spikes = torch.zeros(
        (num_steps, flat_values.numel()),
        device=values.device,
        dtype=values.dtype,
    )

    valid = flat_values > 0
    if valid.any():
        scaled = flat_values.pow(1.0 / tau)
        spike_times = torch.clamp(
            ((1.0 - scaled) * (num_steps - 1)).round().long(),
            min=0,
            max=num_steps - 1,
        )
        columns = torch.arange(flat_values.numel(), device=values.device)[valid]
        spikes[spike_times[valid], columns] = 1.0

    return spikes.view(num_steps, *values.shape)


def delta_encode(data: torch.Tensor) -> torch.Tensor:
    """Emit spikes only where consecutive frames differ."""

    if data.dim() < 1:
        raise ValueError("delta_encode expects at least one temporal dimension.")
    if data.shape[0] == 0:
        return torch.zeros_like(data)

    dtype = data.dtype if torch.is_floating_point(data) else torch.float32
    spikes = torch.zeros(data.shape, device=data.device, dtype=dtype)
    if data.shape[0] == 1:
        return spikes

    changes = torch.diff(data, dim=0).ne(0)
    spikes[1:] = changes.to(dtype)
    return spikes


def spike_count(spikes: torch.Tensor) -> torch.Tensor:
    """Count total binary spike events in a tensor."""

    return (spikes > 0).sum()


def firing_rate(spikes: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute mean firing rate along a dimension."""

    return (spikes > 0).to(torch.float32).mean(dim=dim)


def interspike_interval(spikes: torch.Tensor) -> torch.Tensor:
    """Compute interspike intervals along the leading time dimension."""

    if spikes.dim() == 0:
        raise ValueError("interspike_interval expects at least one temporal dimension.")

    binary_spikes = (spikes > 0).to(torch.float32)
    if binary_spikes.dim() == 1:
        binary_spikes = binary_spikes.unsqueeze(-1)
        squeeze_output = True
    else:
        squeeze_output = False

    time_steps = binary_spikes.shape[0]
    flattened = binary_spikes.reshape(time_steps, -1)
    intervals = []
    max_intervals = 0

    for unit_index in range(flattened.shape[1]):
        spike_times = torch.nonzero(flattened[:, unit_index], as_tuple=False).flatten()
        if spike_times.numel() < 2:
            intervals.append(torch.empty(0, device=spikes.device, dtype=torch.float32))
            continue
        isi = torch.diff(spike_times.to(torch.float32))
        intervals.append(isi)
        max_intervals = max(max_intervals, int(isi.numel()))

    output = torch.full(
        (max_intervals, flattened.shape[1]),
        float("nan"),
        device=spikes.device,
        dtype=torch.float32,
    )
    for unit_index, isi in enumerate(intervals):
        if isi.numel() > 0:
            output[: isi.numel(), unit_index] = isi

    if squeeze_output:
        return output.squeeze(-1)

    return output.view(max_intervals, *spikes.shape[1:])


__all__ = [
    "LIFLayer",
    "LIFNeuron",
    "SurrogateSpike",
    "delta_encode",
    "firing_rate",
    "interspike_interval",
    "latency_encode",
    "rate_encode",
    "spike_count",
]
