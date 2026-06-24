"""Event-driven visual sensory stream for the embodied sensorimotor cortex."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import PredictiveConfig, SpikingConfig
from bioarn.core.math_utils import normalize, sparse_top_k
from bioarn.core.spiking import LIFLayer


@dataclass
class VisionOutput:
    """Outputs from the visual sensory stream."""

    features: torch.Tensor
    event_count: int
    suppressed_fraction: float
    raw_events: torch.Tensor


class VisualEncoder(nn.Module):
    """Sparse, event-driven visual encoder with predictive suppression."""

    def __init__(self, input_shape: tuple, output_dim: int, config: SpikingConfig):
        super().__init__()
        if not input_shape:
            raise ValueError("input_shape must be non-empty.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.output_dim = int(output_dim)
        self.config = config
        self.predictive_config = PredictiveConfig()

        self.channels, self.height, self.width = self._resolve_spatial_shape(self.input_shape)
        self.frame_dim = self.channels * self.height * self.width
        self.event_dim = self.frame_dim * 2
        self.hidden_dim = max(32, min(128, self.output_dim * 2))
        self.hidden_top_k = max(1, self.hidden_dim // 10)
        self.output_top_k = max(1, self.output_dim // 20)

        self.layer1 = LIFLayer(
            self.event_dim,
            self.hidden_dim,
            bias=False,
            config=config,
            spike_history_steps=8,
        )
        self.layer2 = LIFLayer(
            self.hidden_dim,
            self.output_dim,
            bias=False,
            config=config,
            spike_history_steps=8,
        )

        self.register_buffer("last_frame", torch.empty(0))
        self.register_buffer("predicted_frame", torch.empty(0))
        self.register_buffer("total_frames", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_raw_events", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_transmitted_events", torch.tensor(0, dtype=torch.long))
        self.register_buffer("suppression_sum", torch.tensor(0.0, dtype=torch.float32))

        self._initialize_filters()

    @staticmethod
    def _resolve_spatial_shape(input_shape: tuple[int, ...]) -> tuple[int, int, int]:
        if len(input_shape) == 1:
            side = int(round(math.sqrt(input_shape[0])))
            if side * side != input_shape[0]:
                raise ValueError("Flat visual inputs must correspond to a square image.")
            return 1, side, side
        if len(input_shape) == 2:
            return 1, input_shape[0], input_shape[1]
        if len(input_shape) == 3:
            return input_shape[0], input_shape[1], input_shape[2]
        raise ValueError("input_shape must be (H, W), (C, H, W), or (H*W,).")

    @torch.no_grad()
    def _initialize_filters(self) -> None:
        templates = self._build_edge_templates(device=self.layer1.linear.weight.device)
        layer1_weights = torch.empty(
            self.hidden_dim,
            self.event_dim,
            device=self.layer1.linear.weight.device,
            dtype=self.layer1.linear.weight.dtype,
        )

        gain1 = max(3.0, min(8.0, math.sqrt(self.frame_dim) / 4.0))
        for idx in range(self.hidden_dim):
            template = templates[idx % templates.shape[0]]
            polarity = -1.0 if idx % 2 else 1.0
            row = torch.cat((template * polarity, -template * polarity), dim=0)
            layer1_weights[idx] = normalize(row.unsqueeze(0)).squeeze(0) * gain1

        self.layer1.linear.weight.copy_(layer1_weights)

        layer2_weights = torch.zeros_like(self.layer2.linear.weight)
        gain2 = 1.75
        for idx in range(self.output_dim):
            row = torch.cos(
                torch.linspace(
                    0.0,
                    math.pi * (idx + 1),
                    self.hidden_dim,
                    device=layer2_weights.device,
                    dtype=layer2_weights.dtype,
                )
            )
            row = row * torch.sign(torch.roll(row, shifts=idx % max(self.hidden_dim, 1)))
            layer2_weights[idx] = normalize(row.unsqueeze(0)).squeeze(0) * gain2

        self.layer2.linear.weight.copy_(layer2_weights)

    def _build_edge_templates(self, device: torch.device) -> torch.Tensor:
        y_coords = torch.linspace(-1.0, 1.0, self.height, device=device)
        x_coords = torch.linspace(-1.0, 1.0, self.width, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        envelope = torch.exp(-2.5 * (grid_x.pow(2) + grid_y.pow(2)))

        base_templates = [
            grid_x * envelope,
            grid_y * envelope,
            (grid_x + grid_y) * envelope,
            (grid_x - grid_y) * envelope,
            torch.sin(math.pi * grid_x) * envelope,
            torch.sin(math.pi * grid_y) * envelope,
            (grid_x * grid_y) * envelope,
            (grid_x.pow(2) - grid_y.pow(2)) * envelope,
        ]
        stacked = torch.stack(base_templates, dim=0).unsqueeze(1).repeat(1, self.channels, 1, 1)
        return normalize(stacked.reshape(stacked.shape[0], -1))

    def reset_state(self) -> None:
        """Reset recurrent state and predictive buffers."""
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.last_frame = torch.empty(0, device=self.layer1.linear.weight.device, dtype=self.layer1.linear.weight.dtype)
        self.predicted_frame = torch.empty(0, device=self.layer1.linear.weight.device, dtype=self.layer1.linear.weight.dtype)

    def _flatten_frame_batch(self, frame: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if frame.dim() == 1:
            if frame.shape[0] != self.frame_dim:
                raise ValueError(f"Expected flattened frame_dim={self.frame_dim}, received {frame.shape[0]}.")
            return frame.unsqueeze(0).to(torch.float32), True
        if frame.dim() == 2:
            if frame.shape[-1] != self.frame_dim:
                raise ValueError(f"Expected flattened frame_dim={self.frame_dim}, received {frame.shape[-1]}.")
            return frame.to(torch.float32), False
        if frame.dim() == 3 and self.channels == 1:
            if frame.shape[-2:] != (self.height, self.width):
                raise ValueError("Image frame shape does not match configured input_shape.")
            return frame.unsqueeze(1).reshape(frame.shape[0], -1).to(torch.float32), False
        if frame.dim() == 4 and frame.shape[1:] == (self.channels, self.height, self.width):
            return frame.reshape(frame.shape[0], -1).to(torch.float32), False
        raise ValueError("frame must be shaped (batch, flat), (batch, H, W), or (batch, C, H, W).")

    def _restore_event_shape(self, events: torch.Tensor, squeeze: bool) -> torch.Tensor:
        if self.input_shape == (self.frame_dim,):
            restored = events
        elif len(self.input_shape) == 2:
            restored = events.view(events.shape[0], self.height, self.width)
        else:
            restored = events.view(events.shape[0], self.channels, self.height, self.width)
        return restored.squeeze(0) if squeeze and restored.shape[0] == 1 else restored

    def _resolve_reference(self, current_frame: torch.Tensor, prev_frame: torch.Tensor | None) -> torch.Tensor:
        if prev_frame is not None:
            prev_flat, _ = self._flatten_frame_batch(prev_frame)
            if prev_flat.shape != current_frame.shape:
                raise ValueError("prev_frame must match frame batch shape.")
            return prev_flat.to(device=current_frame.device, dtype=current_frame.dtype)
        if self.last_frame.shape == current_frame.shape:
            return self.last_frame.to(device=current_frame.device, dtype=current_frame.dtype)
        return torch.zeros_like(current_frame)

    def _resolve_prediction(self, current_frame: torch.Tensor) -> torch.Tensor:
        if self.predicted_frame.shape == current_frame.shape:
            return self.predicted_frame.to(device=current_frame.device, dtype=current_frame.dtype)
        return torch.zeros_like(current_frame)

    def delta_encode_frame(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signed ON/OFF events between two consecutive frames."""
        current_flat, _ = self._flatten_frame_batch(current_frame)
        previous_flat, _ = self._flatten_frame_batch(previous_frame)
        if current_flat.shape != previous_flat.shape:
            raise ValueError("current_frame and previous_frame must have identical shapes.")
        diff = current_flat - previous_flat
        return torch.where(diff.abs() > 0, diff, torch.zeros_like(diff))

    def _apply_predictive_suppression(
        self,
        raw_events: torch.Tensor,
        current_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        prediction = self._resolve_prediction(current_frame)
        surprise_mask = (current_frame - prediction).abs() > self.predictive_config.error_threshold
        transmitted = raw_events * surprise_mask.to(raw_events.dtype)

        active_events = raw_events.abs() > 0
        if active_events.any():
            kept_fraction = transmitted[active_events].abs().gt(0).float().mean().item()
            suppressed_fraction = 1.0 - kept_fraction
        else:
            suppressed_fraction = 1.0
        return transmitted, float(suppressed_fraction)

    def _event_vector(self, events: torch.Tensor) -> torch.Tensor:
        on_events = F.relu(events)
        off_events = F.relu(-events)
        return torch.cat((on_events, off_events), dim=-1)

    def _sparse_stage_output(self, layer: LIFLayer, inputs: torch.Tensor, top_k: int) -> torch.Tensor:
        spikes, _ = layer(inputs)
        currents = layer.linear(inputs).clamp_min(0.0)
        activity = currents * (spikes > 0).to(currents.dtype)
        return sparse_top_k(activity, k=top_k)

    def forward(self, frame: torch.Tensor, prev_frame: torch.Tensor | None = None) -> VisionOutput:
        current_flat, squeeze = self._flatten_frame_batch(frame)
        reference = self._resolve_reference(current_flat, prev_frame)
        raw_events = self.delta_encode_frame(current_flat, reference)
        transmitted_events, suppressed_fraction = self._apply_predictive_suppression(raw_events, current_flat)

        hidden = self._sparse_stage_output(self.layer1, self._event_vector(transmitted_events), self.hidden_top_k)
        features = self._sparse_stage_output(self.layer2, hidden, self.output_top_k)

        raw_event_count = int(raw_events.abs().gt(0).sum().item())
        transmitted_count = int(transmitted_events.abs().gt(0).sum().item())

        with torch.no_grad():
            self.total_frames.add_(1)
            self.total_raw_events.add_(raw_event_count)
            self.total_transmitted_events.add_(transmitted_count)
            self.suppression_sum.add_(suppressed_fraction)
            self.last_frame = current_flat.detach().clone()
            self.predicted_frame = current_flat.detach().clone()

        return VisionOutput(
            features=features.squeeze(0) if squeeze and features.shape[0] == 1 else features,
            event_count=raw_event_count,
            suppressed_fraction=suppressed_fraction,
            raw_events=self._restore_event_shape(raw_events, squeeze),
        )

    def encode_sequence(self, frames: torch.Tensor) -> list[VisionOutput]:
        """Encode a time-first frame sequence of shape (time, batch, ...)."""
        if frames.dim() < 2:
            raise ValueError("frames must include a time dimension.")

        self.reset_state()
        outputs: list[VisionOutput] = []
        prev_frame = None
        for frame_t in frames.unbind(dim=0):
            output = self.forward(frame_t, prev_frame=prev_frame)
            outputs.append(output)
            prev_frame = frame_t
        return outputs

    @torch.no_grad()
    def get_suppression_stats(self) -> dict[str, float | int]:
        """Return cumulative predictive suppression statistics."""
        total_frames = int(self.total_frames.item())
        total_raw = int(self.total_raw_events.item())
        total_transmitted = int(self.total_transmitted_events.item())
        suppression_rate = 1.0
        if total_raw > 0:
            suppression_rate = 1.0 - (total_transmitted / total_raw)
        mean_suppressed_fraction = (
            float(self.suppression_sum.item()) / total_frames if total_frames else 0.0
        )
        return {
            "total_frames": total_frames,
            "total_raw_events": total_raw,
            "total_transmitted_events": total_transmitted,
            "suppression_rate": float(suppression_rate),
            "mean_suppressed_fraction": float(mean_suppressed_fraction),
        }


__all__ = ["VisionOutput", "VisualEncoder"]
