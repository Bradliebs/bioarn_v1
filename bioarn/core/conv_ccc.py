"""Convolutional concept cell clusters for spatial vision inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.config import ConvCCCConfig, MarginGateConfig
from bioarn.core.consolidation import SynapticConsolidation
from bioarn.core.margin_gate import MarginGate, MarginGateOutput, ResonanceOutput
from bioarn.core.math_utils import normalize, sparse_top_k


@dataclass
class ConvCCCOutput:
    """Output of a convolutional concept cell cluster."""

    fired: bool
    abstained: bool
    confidence: torch.Tensor
    f1_output: torch.Tensor
    f2_activation: torch.Tensor
    gate_output: MarginGateOutput
    prediction: torch.Tensor | None
    resonance: ResonanceOutput | None


@dataclass
class ConvCCCPoolOutput:
    """Aggregated output for a pool of convolutional concept cell clusters."""

    outputs: list[ConvCCCOutput]
    fired_indices: list[int]
    abstained_indices: list[int]
    recruited: bool
    recruited_index: int | None
    winner_confidences: torch.Tensor


@dataclass
class _ConvHebbianTrace:
    pre_activations: tuple[torch.Tensor, ...]
    post_activations: tuple[torch.Tensor, ...]


class ConvF1Layer(nn.Module):
    """Convolutional F1 encoder that preserves spatial structure."""

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        spatial_size: int = 32,
        top_k: int = 32,
        spatial_grid: int = 4,
        *,
        num_layers: int = 3,
        hidden_channels: tuple[int, int] = (32, 64),
        spatial_top_k: int = 4,
        competitive_k: int = 8,
        hebbian_lr: float = 0.0025,
        weight_norm_target: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_features = int(num_features)
        self.spatial_size = int(spatial_size)
        self.spatial_grid = int(spatial_grid)
        self.top_k = int(max(1, top_k))
        self.num_layers = int(min(max(1, num_layers), 3))
        hidden1 = int(max(1, hidden_channels[0]))
        hidden2 = int(max(1, hidden_channels[1]))
        self.hidden_channels = (hidden1, hidden2)
        self.spatial_top_k = int(max(1, spatial_top_k))
        self.competitive_k = int(max(1, competitive_k))
        self.hebbian_lr = float(max(0.0, hebbian_lr))
        self.weight_norm_target = float(max(1e-6, weight_norm_target))

        if self.num_layers == 1:
            self.conv1 = nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, padding=1)
            self.conv2 = None
            self.conv3 = None
        elif self.num_layers == 2:
            self.conv1 = nn.Conv2d(self.in_channels, hidden1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(hidden1, self.num_features, kernel_size=3, padding=1)
            self.conv3 = None
        else:
            self.conv1 = nn.Conv2d(self.in_channels, hidden1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(hidden2, self.num_features, kernel_size=3, padding=1)
        self.register_buffer("is_frozen", torch.tensor(False, dtype=torch.bool))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return sum(self.feature_channels) * self.spatial_grid * self.spatial_grid

    @property
    def feature_channels(self) -> tuple[int, ...]:
        if self.num_layers <= 1:
            return (self.num_features,)
        if self.num_layers == 2:
            return (self.hidden_channels[0], self.num_features)
        return (self.hidden_channels[0], self.hidden_channels[1], self.num_features)

    def freeze(self) -> None:
        self.is_frozen.fill_(True)

    def _ensure_spatial_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        expected = self.in_channels * self.spatial_size * self.spatial_size
        if x.dim() == 1:
            if x.numel() != expected:
                raise ValueError(
                    f"Expected flattened input with {expected} values, got {x.numel()}."
                )
            return x.reshape(1, self.in_channels, self.spatial_size, self.spatial_size), True
        if x.dim() == 2:
            if x.shape[-1] != expected:
                raise ValueError(
                    f"Expected flattened batch with last dimension {expected}, got {x.shape[-1]}."
                )
            return x.reshape(-1, self.in_channels, self.spatial_size, self.spatial_size), False
        if x.dim() == 3:
            if tuple(x.shape) != (self.in_channels, self.spatial_size, self.spatial_size):
                raise ValueError(
                    "Expected image input with shape "
                    f"({self.in_channels}, {self.spatial_size}, {self.spatial_size})."
                )
            return x.unsqueeze(0), True
        if x.dim() == 4:
            if tuple(x.shape[1:]) != (self.in_channels, self.spatial_size, self.spatial_size):
                raise ValueError(
                    "Expected batched image input with shape "
                    f"(batch, {self.in_channels}, {self.spatial_size}, {self.spatial_size})."
                )
            return x, False
        raise ValueError("ConvF1 inputs must be flat vectors or 3D/4D image tensors.")

    @staticmethod
    def _maybe_squeeze(x: torch.Tensor, squeeze: bool) -> torch.Tensor:
        return x.squeeze(0) if squeeze else x

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        if min(x.shape[-2:]) <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    def _competitive_spatial_pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        flat = feature_map.flatten(2)
        top_k = min(self.spatial_top_k, flat.shape[-1])
        top_values, top_indices = torch.topk(flat, k=top_k, dim=-1)
        sparse_flat = torch.zeros_like(flat)
        sparse_flat.scatter_(2, top_indices, top_values)
        sparse_map = sparse_flat.view_as(feature_map)
        pooled = F.adaptive_max_pool2d(sparse_map, (self.spatial_grid, self.spatial_grid))
        return normalize(pooled.flatten(1))

    def _forward_dense(self, x: torch.Tensor) -> tuple[torch.Tensor, _ConvHebbianTrace]:
        batch = x.to(self.conv1.weight.dtype)
        pre_activations: list[torch.Tensor] = []
        post_activations: list[torch.Tensor] = []
        feature_maps: list[torch.Tensor] = []

        pre_activations.append(batch)
        current = F.relu(self.conv1(batch))
        post_activations.append(current)
        feature_maps.append(current)

        if self.conv2 is not None:
            pre_conv2 = self._downsample(current)
            pre_activations.append(pre_conv2)
            current = F.relu(self.conv2(pre_conv2))
            post_activations.append(current)
            feature_maps.append(current)

        if self.conv3 is not None:
            pre_conv3 = self._downsample(current)
            pre_activations.append(pre_conv3)
            current = F.relu(self.conv3(pre_conv3))
            post_activations.append(current)
            feature_maps.append(current)

        pooled_scales = [self._competitive_spatial_pool(feature_map) for feature_map in feature_maps]
        dense = torch.cat(pooled_scales, dim=1)
        return dense, _ConvHebbianTrace(tuple(pre_activations), tuple(post_activations))

    def _scale_top_k_budgets(self) -> tuple[int, ...]:
        channels = self.feature_channels
        total_channels = max(sum(channels), 1)
        budgets = [
            max(1, int(round(self.top_k * (channel_count / total_channels))))
            for channel_count in channels
        ]
        difference = self.top_k - sum(budgets)
        index = 0
        while difference != 0 and budgets:
            position = index % len(budgets)
            if difference > 0:
                budgets[position] += 1
                difference -= 1
            elif budgets[position] > 1:
                budgets[position] -= 1
                difference += 1
            index += 1
            if index > len(budgets) * max(self.top_k, 1) * 2:
                break
        return tuple(max(1, budget) for budget in budgets)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_batch, squeeze = self._ensure_spatial_batch(x)
        dense, _ = self._forward_dense(raw_batch)
        scale_sizes = [channel_count * self.spatial_grid * self.spatial_grid for channel_count in self.feature_channels]
        budgets = self._scale_top_k_budgets()
        sparse_scales: list[torch.Tensor] = []
        start = 0
        for size, budget in zip(scale_sizes, budgets, strict=True):
            scale = dense[:, start : start + size]
            sparse_scales.append(sparse_top_k(scale, min(budget, size)))
            start += size
        sparse = normalize(torch.cat(sparse_scales, dim=1))
        return self._maybe_squeeze(sparse, squeeze)

    @staticmethod
    def _align_signal(
        signal: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if signal is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        aligned = signal.reshape(-1).to(device=device, dtype=dtype)
        if aligned.numel() == 1 and batch_size > 1:
            return aligned.expand(batch_size)
        if aligned.numel() != batch_size:
            raise ValueError(
                f"Expected learning signal with {batch_size} values, got {aligned.numel()}."
            )
        return aligned

    @staticmethod
    def _clamp_filter_norms(weight: torch.Tensor, *, max_norm: float = 1.0) -> None:
        flat = weight.view(weight.shape[0], -1)
        flat.sub_(flat.mean(dim=1, keepdim=True))
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        flat.mul_(float(max_norm) / norms)

    @staticmethod
    def _competitive_channel_mask(post: torch.Tensor, competitive_k: int) -> torch.Tensor:
        if competitive_k >= post.shape[1]:
            return torch.ones(post.shape[0], post.shape[1], 1, device=post.device, dtype=post.dtype)
        channel_energy = post.mean(dim=(2, 3))
        _, top_indices = torch.topk(channel_energy, k=min(competitive_k, channel_energy.shape[1]), dim=1)
        mask = torch.zeros_like(channel_energy)
        mask.scatter_(1, top_indices, 1.0)
        return mask.unsqueeze(-1)

    @staticmethod
    def _hebbian_conv_update(
        layer: nn.Conv2d,
        pre: torch.Tensor,
        post: torch.Tensor,
        signal: torch.Tensor,
        *,
        lr: float,
        competitive_k: int,
        weight_norm_target: float,
    ) -> None:
        if lr <= 0.0 or not bool((signal > 0).any().item()):
            return
        patches = F.unfold(
            pre.to(layer.weight.dtype),
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )
        post_flat = post.to(layer.weight.dtype).reshape(post.shape[0], post.shape[1], -1)
        channel_mask = ConvF1Layer._competitive_channel_mask(post, competitive_k).to(post_flat.dtype)
        weighted_post = post_flat * channel_mask * signal.to(post_flat.dtype).view(-1, 1, 1)
        denom = max(pre.shape[0] * post_flat.shape[-1], 1)
        delta = torch.einsum("bol,bil->oi", weighted_post, patches) / denom
        layer.weight.add_(float(lr) * delta.reshape_as(layer.weight))
        if layer.bias is not None:
            layer.bias.mul_(0.99).add_(float(lr) * weighted_post.mean(dim=(0, 2)))
        ConvF1Layer._clamp_filter_norms(layer.weight.data, max_norm=weight_norm_target)

    @torch.no_grad()
    def hebbian_update(
        self,
        x: torch.Tensor,
        *,
        learning_signal: torch.Tensor | None = None,
        lr: float | None = None,
    ) -> None:
        effective_lr = self.hebbian_lr if lr is None else float(lr)
        if bool(self.is_frozen.item()) or effective_lr <= 0.0:
            return
        raw_batch, _ = self._ensure_spatial_batch(x)
        _, traces = self._forward_dense(raw_batch)
        signal = self._align_signal(
            learning_signal,
            raw_batch.shape[0],
            raw_batch.device,
            self.conv1.weight.dtype,
        ).clamp_min(0.0)
        layers = [self.conv1]
        if self.conv2 is not None:
            layers.append(self.conv2)
        if self.conv3 is not None:
            layers.append(self.conv3)
        for layer, pre, post in zip(layers, traces.pre_activations, traces.post_activations, strict=True):
            self._hebbian_conv_update(
                layer,
                pre,
                post,
                signal,
                lr=effective_lr,
                competitive_k=self.competitive_k,
                weight_norm_target=self.weight_norm_target,
            )


class ConvConceptCellCluster(nn.Module):
    """CCC variant using convolutional features for spatial inputs."""

    def __init__(
        self,
        config: ConvCCCConfig,
        margin_config: MarginGateConfig,
        *,
        f1_layer: ConvF1Layer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.f1_layer = f1_layer or ConvF1Layer(
            in_channels=config.in_channels,
            num_features=config.num_conv_features,
            spatial_size=config.spatial_size,
            top_k=config.f1_top_k,
            spatial_grid=config.spatial_grid,
            num_layers=config.num_conv_layers,
            hidden_channels=config.conv_hidden_channels,
            spatial_top_k=config.spatial_top_k,
            competitive_k=config.conv_competitive_k,
            hebbian_lr=config.conv_hebbian_lr,
            weight_norm_target=config.conv_weight_norm,
        )
        self.margin_gate = MarginGate(margin_config)
        self.register_buffer("feedback_template", torch.zeros(config.concept_dim, dtype=torch.float32))
        self.register_buffer("concept_direction", torch.zeros(config.concept_dim, dtype=torch.float32))
        self.register_buffer("is_committed", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("is_locked", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("age", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_fired", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("importance", torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _ensure_feature_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() != 2:
            raise ValueError("Expected feature tensor with shape (dim,) or (batch, dim).")
        return x, False

    @staticmethod
    def _maybe_squeeze(x: torch.Tensor | None, squeeze: bool) -> torch.Tensor | None:
        if x is None:
            return None
        return x.squeeze(0) if squeeze else x

    @staticmethod
    def _to_bool_tensor(value: bool, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), value, device=device, dtype=torch.bool)

    @staticmethod
    def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
        return normalize(x.unsqueeze(0)).squeeze(0)

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

    def maybe_lock(self) -> None:
        if bool(self.is_locked.item()) or not bool(self.is_committed.item()):
            return
        if float(self.importance.item()) >= float(self.config.lock_threshold):
            self.is_locked.fill_(True)

    @torch.no_grad()
    def empty_output(self, raw_input: torch.Tensor) -> ConvCCCOutput:
        raw_batch, squeeze = self.f1_layer._ensure_spatial_batch(raw_input)
        batch_size = raw_batch.shape[0]
        device = raw_batch.device
        dtype = self.feedback_template.dtype
        f1_output = torch.zeros(batch_size, self.config.concept_dim, device=device, dtype=dtype)
        f2_activation = torch.zeros_like(f1_output)
        confidence = torch.zeros(batch_size, device=device, dtype=dtype)
        gate_output = MarginGateOutput(
            output=torch.zeros_like(f2_activation),
            confidence=confidence,
            fired=self._to_bool_tensor(False, batch_size, device),
            abstained=self._to_bool_tensor(True, batch_size, device),
        )
        return ConvCCCOutput(
            fired=False,
            abstained=True,
            confidence=self._maybe_squeeze(confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_output, squeeze),
            f2_activation=self._maybe_squeeze(f2_activation, squeeze),
            gate_output=MarginGateOutput(
                output=self._maybe_squeeze(gate_output.output, squeeze),
                confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
                fired=self._maybe_squeeze(gate_output.fired, squeeze),
                abstained=self._maybe_squeeze(gate_output.abstained, squeeze),
            ),
            prediction=None,
            resonance=None,
        )

    @torch.no_grad()
    def f1_encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        return self.f1_layer(raw_input)

    @torch.no_grad()
    def f2_activate(self, f1_output: torch.Tensor) -> torch.Tensor:
        return f1_output

    @torch.no_grad()
    def generate_prediction(self, f2_activation: torch.Tensor) -> torch.Tensor:
        f2_batch, squeeze = self._ensure_feature_batch(f2_activation)
        if float(self.feedback_template.norm().item()) <= 0.0:
            prediction = torch.zeros_like(f2_batch)
            return self._maybe_squeeze(prediction, squeeze)
        cue = self.concept_direction.unsqueeze(0).expand_as(f2_batch)
        strength = F.relu(F.cosine_similarity(f2_batch, cue, dim=-1)).unsqueeze(-1)
        prediction = strength * self.feedback_template.unsqueeze(0)
        return self._maybe_squeeze(prediction, squeeze)

    @torch.no_grad()
    def preview(self, raw_input: torch.Tensor) -> ConvCCCOutput:
        raw_batch, squeeze = self.f1_layer._ensure_spatial_batch(raw_input)
        f1_batch = self._ensure_feature_batch(self.f1_encode(raw_batch))[0]
        f2_batch = self._ensure_feature_batch(self.f2_activate(f1_batch))[0]
        if not bool(self.is_committed.item()):
            return self.empty_output(raw_batch if not squeeze else raw_batch.squeeze(0))

        gate_output = self.margin_gate(f2_batch, self.concept_direction)
        fired = bool(gate_output.fired.any().item())
        abstained = bool(gate_output.abstained.all().item())
        prediction = None
        resonance = None
        if fired:
            prediction = self._ensure_feature_batch(self.generate_prediction(gate_output.output))[0]
            resonance = self.margin_gate.check_resonance(prediction, f1_batch)

        return ConvCCCOutput(
            fired=fired,
            abstained=abstained,
            confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_batch, squeeze),
            f2_activation=self._maybe_squeeze(f2_batch, squeeze),
            gate_output=MarginGateOutput(
                output=self._maybe_squeeze(gate_output.output, squeeze),
                confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
                fired=self._maybe_squeeze(gate_output.fired, squeeze),
                abstained=self._maybe_squeeze(gate_output.abstained, squeeze),
            ),
            prediction=self._maybe_squeeze(prediction, squeeze) if prediction is not None else None,
            resonance=(
                ResonanceOutput(
                    match_score=self._maybe_squeeze(resonance.match_score, squeeze),
                    resonated=self._maybe_squeeze(resonance.resonated, squeeze),
                    learn_signal=self._maybe_squeeze(resonance.learn_signal, squeeze),
                )
                if resonance is not None
                else None
            ),
        )

    @torch.no_grad()
    def learn_fast(
        self,
        raw_input: torch.Tensor,
        f1_output: torch.Tensor,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        if bool(self.is_locked.item()):
            return
        raw_batch, _ = self.f1_layer._ensure_spatial_batch(raw_input)
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        if not bool(self.f1_layer.is_frozen.item()):
            self.f1_layer.hebbian_update(
                raw_batch,
                learning_signal=torch.ones(
                    raw_batch.shape[0],
                    device=raw_batch.device,
                    dtype=torch.float32,
                ),
                lr=float(self.config.conv_hebbian_lr) * 1.5 * lr_multiplier,
            )
            f1_batch = self._ensure_feature_batch(self.f1_encode(raw_batch))[0]
        else:
            f1_batch = self._ensure_feature_batch(f1_output)[0]
        prototype = f1_batch.mean(dim=0)
        self.concept_direction.copy_(self._normalize_vector(prototype))
        self.feedback_template.copy_(prototype)
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
        del timestep
        if not bool(self.is_committed.item()) or bool(self.is_locked.item()):
            return
        raw_batch, _ = self.f1_layer._ensure_spatial_batch(raw_input)
        learn_signal = self._prepare_batch_vector(
            resonance.learn_signal,
            raw_batch.shape[0],
        ).to(torch.float32).clamp_min(0.0)
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        if not bool((learn_signal > 0).any().item()) or lr_multiplier <= 0.0:
            return

        if not bool(self.f1_layer.is_frozen.item()):
            self.f1_layer.hebbian_update(
                raw_batch,
                learning_signal=learn_signal,
                lr=float(self.config.conv_hebbian_lr) * lr_multiplier,
            )
            f1_batch = self._ensure_feature_batch(self.f1_encode(raw_batch))[0]
        else:
            f1_batch = self._ensure_feature_batch(f1_output)[0]
        learn_signal = learn_signal.to(f1_batch.dtype)

        concept_delta = (f1_batch * learn_signal.unsqueeze(-1)).mean(dim=0)
        updated_direction = self.concept_direction + (
            float(self.config.slow_lr) * lr_multiplier * concept_delta
        )
        self.concept_direction.copy_(self._normalize_vector(updated_direction))

        residual = (f1_batch - self.feedback_template.unsqueeze(0)) * learn_signal.unsqueeze(-1)
        feedback_delta = residual.mean(dim=0)
        self.feedback_template.add_(
            float(self.config.feedback_lr) * lr_multiplier * feedback_delta
        )

    @torch.no_grad()
    def forward(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> ConvCCCOutput:
        raw_batch, squeeze = self.f1_layer._ensure_spatial_batch(raw_input)
        f1_batch = self._ensure_feature_batch(self.f1_encode(raw_batch))[0]
        f2_batch = self._ensure_feature_batch(self.f2_activate(f1_batch))[0]
        self.age.add_(raw_batch.shape[0])

        if not bool(self.is_committed.item()):
            return self.empty_output(raw_batch if not squeeze else raw_batch.squeeze(0))

        gate_output = self.margin_gate(f2_batch, self.concept_direction)
        fired = bool(gate_output.fired.any().item())
        abstained = bool(gate_output.abstained.all().item())
        prediction = None
        resonance = None

        if fired:
            prediction = self._ensure_feature_batch(self.generate_prediction(gate_output.output))[0]
            resonance = self.margin_gate.check_resonance(prediction, f1_batch)
            if bool(resonance.resonated.any().item()) and not bool(self.is_locked.item()):
                self.learn_slow(
                    raw_batch,
                    f1_batch,
                    resonance,
                    timestep=timestep,
                    learning_rate_multiplier=learning_rate_multiplier,
                )
            self.last_fired.fill_(int(timestep))

        return ConvCCCOutput(
            fired=fired,
            abstained=abstained,
            confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
            f1_output=self._maybe_squeeze(f1_batch, squeeze),
            f2_activation=self._maybe_squeeze(f2_batch, squeeze),
            gate_output=MarginGateOutput(
                output=self._maybe_squeeze(gate_output.output, squeeze),
                confidence=self._maybe_squeeze(gate_output.confidence, squeeze),
                fired=self._maybe_squeeze(gate_output.fired, squeeze),
                abstained=self._maybe_squeeze(gate_output.abstained, squeeze),
            ),
            prediction=self._maybe_squeeze(prediction, squeeze) if prediction is not None else None,
            resonance=(
                ResonanceOutput(
                    match_score=self._maybe_squeeze(resonance.match_score, squeeze),
                    resonated=self._maybe_squeeze(resonance.resonated, squeeze),
                    learn_signal=self._maybe_squeeze(resonance.learn_signal, squeeze),
                )
                if resonance is not None
                else None
            ),
        )

    @torch.no_grad()
    def get_info(self) -> dict[str, bool | float | int | dict[str, float | int]]:
        return {
            "is_committed": bool(self.is_committed.item()),
            "is_locked": bool(self.is_locked.item()),
            "age": int(self.age.item()),
            "last_fired": int(self.last_fired.item()),
            "concept_direction_norm": float(self.concept_direction.norm().item()),
            "margin_gate": self.margin_gate.get_stats(),
        }


class ConvCCCPool(nn.Module):
    """Pool of convolutional concept cell clusters."""

    def __init__(self, config: ConvCCCConfig, margin_config: MarginGateConfig):
        super().__init__()
        self.config = config
        self.margin_config = margin_config
        self.initial_capacity = int(config.max_pool_size)
        self.max_capacity = max(
            self.initial_capacity,
            int(math.ceil(self.initial_capacity * float(config.max_growth_factor))),
        )
        self.shared_f1 = ConvF1Layer(
            in_channels=config.in_channels,
            num_features=config.num_conv_features,
            spatial_size=config.spatial_size,
            top_k=config.f1_top_k,
            spatial_grid=config.spatial_grid,
            num_layers=config.num_conv_layers,
            hidden_channels=config.conv_hidden_channels,
            spatial_top_k=config.spatial_top_k,
            competitive_k=config.conv_competitive_k,
            hebbian_lr=config.conv_hebbian_lr,
            weight_norm_target=config.conv_weight_norm,
        )
        self.cccs = nn.ModuleList(
            [
                ConvConceptCellCluster(config, margin_config, f1_layer=self.shared_f1)
                for _ in range(self.initial_capacity)
            ]
        )
        self.register_buffer("f1_samples_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("f1_frozen", torch.tensor(False, dtype=torch.bool))
        self.consolidation = SynapticConsolidation(
            len(self.cccs),
            strength=self.config.consolidation_strength,
        )
        self._sync_importance_buffers()

    @property
    def concept_directions(self) -> torch.Tensor:
        if not self.cccs:
            return torch.empty(0, self.config.concept_dim, dtype=torch.float32)
        return torch.stack([ccc.concept_direction for ccc in self.cccs], dim=0)

    def freeze_f1(self) -> None:
        self.shared_f1.freeze()
        self.f1_frozen.fill_(True)

    def observe_samples(self, sample_count: int) -> None:
        count = int(max(0, sample_count))
        if count > 0:
            self.f1_samples_seen.add_(count)

    def create_task_adapter(self) -> None:
        return None

    def _sync_importance_buffers(self) -> None:
        self.consolidation.ensure_capacity(len(self.cccs))
        for index, ccc in enumerate(self.cccs):
            ccc.importance.copy_(self.consolidation.importance[index].to(ccc.importance))
            ccc.maybe_lock()

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

        for _ in range(target_size - current_size):
            self.cccs.append(
                ConvConceptCellCluster(self.config, self.margin_config, f1_layer=self.shared_f1)
            )
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
    def hebbian_update(
        self,
        raw_input: torch.Tensor,
        *,
        fired_indices: list[int] | None = None,
        winner_confidences: torch.Tensor | None = None,
        recruited: bool = False,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> None:
        if bool(self.shared_f1.is_frozen.item()) or float(self.config.conv_hebbian_lr) <= 0.0:
            return
        if recruited or (fired_indices and len(fired_indices) > 0):
            return

        raw_batch, _ = self.shared_f1._ensure_spatial_batch(raw_input)
        lr_multiplier = max(
            0.0,
            float(torch.as_tensor(learning_rate_multiplier, dtype=torch.float32).mean().item()),
        )
        if lr_multiplier <= 0.0:
            return

        confidence_signal = 0.25
        if winner_confidences is not None and winner_confidences.numel() > 0:
            confidence_signal = max(confidence_signal, float(winner_confidences.mean().item()))
        learning_signal = torch.full(
            (raw_batch.shape[0],),
            confidence_signal,
            device=raw_batch.device,
            dtype=torch.float32,
        )
        self.shared_f1.hebbian_update(
            raw_batch,
            learning_signal=learning_signal,
            lr=float(self.config.conv_hebbian_lr) * 0.5 * lr_multiplier,
        )

    @torch.no_grad()
    def preview(self, raw_input: torch.Tensor) -> ConvCCCPoolOutput:
        outputs: list[ConvCCCOutput] = []
        for ccc in self.cccs:
            if bool(ccc.is_committed.item()):
                outputs.append(ccc.preview(raw_input))
            else:
                outputs.append(ccc.empty_output(raw_input))

        fired_indices = [index for index, output in enumerate(outputs) if output.fired]
        abstained_indices = [index for index, output in enumerate(outputs) if output.abstained]
        winner_confidences = (
            torch.stack([self._confidence_score(outputs[index].confidence) for index in fired_indices])
            if fired_indices
            else torch.empty(0, dtype=torch.float32)
        )
        return ConvCCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=False,
            recruited_index=None,
            winner_confidences=winner_confidences,
        )

    @torch.no_grad()
    def forward(
        self,
        raw_input: torch.Tensor,
        timestep: int = 0,
        allow_recruit: bool = True,
        *,
        learning_rate_multiplier: float | torch.Tensor = 1.0,
    ) -> ConvCCCPoolOutput:
        raw_batch, _ = self.shared_f1._ensure_spatial_batch(raw_input)
        self.observe_samples(raw_batch.shape[0])

        outputs: list[ConvCCCOutput] = []
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
        recruited_index = None
        if allow_recruit and not any(output.fired for output in outputs):
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
            torch.stack([self._confidence_score(outputs[index].confidence) for index in fired_indices])
            if fired_indices
            else torch.empty(0, dtype=torch.float32)
        )
        return ConvCCCPoolOutput(
            outputs=outputs,
            fired_indices=fired_indices,
            abstained_indices=abstained_indices,
            recruited=recruited,
            recruited_index=recruited_index,
            winner_confidences=winner_confidences,
        )

    @torch.no_grad()
    def get_winners(self, pool_output: ConvCCCPoolOutput, k: int = 5) -> list[int]:
        if not pool_output.fired_indices:
            return []
        top_k = min(k, len(pool_output.fired_indices))
        _, top_indices = torch.topk(pool_output.winner_confidences, k=top_k)
        return [pool_output.fired_indices[index] for index in top_indices.tolist()]

    @torch.no_grad()
    def get_pool_stats(self) -> dict[str, float | int | bool]:
        num_committed = sum(bool(ccc.is_committed.item()) for ccc in self.cccs)
        num_locked = sum(bool(ccc.is_locked.item()) for ccc in self.cccs)
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
        mean_importance = (
            float(self.consolidation.importance[: len(self.cccs)].mean().item())
            if self.cccs
            else 0.0
        )
        return {
            "num_committed": num_committed,
            "num_uncommitted": len(self.cccs) - num_committed,
            "num_locked": num_locked,
            "mean_confidence": float(mean_confidence),
            "fire_rate": float(fire_rate),
            "total_concepts": len(self.cccs),
            "initial_capacity": self.initial_capacity,
            "max_capacity": self.max_capacity,
            "mean_importance": mean_importance,
            "f1_frozen": bool(self.f1_frozen.item()),
            "task_adapters": 0,
        }


__all__ = [
    "ConvCCCOutput",
    "ConvCCCPool",
    "ConvCCCPoolOutput",
    "ConvConceptCellCluster",
    "ConvF1Layer",
]
