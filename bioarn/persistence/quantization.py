"""Quantization helpers for Bio-ARN persistence and deployment."""

from __future__ import annotations

import copy
import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bioarn.core.ccc import ConceptCellCluster
from bioarn.loop import SensorimotorLoop
from bioarn.persistence._common import atomic_write_bytes, clone_state_dict, state_size_bytes
from bioarn.system import BioARNCore


@dataclass
class QuantizedModel:
    """Quantized model state with scale factors for reconstruction."""

    quantized_state: dict[str, torch.Tensor]
    scale_factors: dict[str, float]
    bits: int
    activation_bits: int
    activation_ranges: dict[str, dict[str, float]]
    quantized_keys: list[str]
    original_bytes: int
    quantized_bytes: int

    def dequantized_state_dict(self) -> dict[str, torch.Tensor]:
        restored: dict[str, torch.Tensor] = {}
        for key, value in self.quantized_state.items():
            if key in self.scale_factors:
                restored[key] = value.to(torch.float32) * float(self.scale_factors[key])
            else:
                restored[key] = value.detach().cpu().clone()
        return restored


@dataclass
class QuantizationReport:
    original_accuracy: float
    quantized_accuracy: float
    accuracy_drop: float
    max_weight_error: float
    mean_weight_error: float
    bits: int
    compression_ratio: float
    component_errors: dict[str, float] = field(default_factory=dict)
    max_output_deviation: float = 0.0


class ModelQuantizer:
    """Quantize Bio-ARN model for neuromorphic deployment."""

    _FLOAT_WEIGHT_SUFFIXES = ("weight", "weights", "f2_weights", "feedback_weights", "concept_direction")
    _EXCLUDED_KEYS = ("theta_margin", "theta_resonance", "fire_rate", "avg_confidence", "total_")

    def _should_quantize_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if not tensor.is_floating_point():
            return False
        if any(token in name for token in self._EXCLUDED_KEYS):
            return False
        return tensor.dim() > 1 or name.split(".")[-1] in self._FLOAT_WEIGHT_SUFFIXES

    @staticmethod
    def _int_dtype(bits: int) -> torch.dtype:
        if bits <= 8:
            return torch.int8
        if bits <= 16:
            return torch.int16
        return torch.int32

    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
        int_max = float((2 ** (bits - 1)) - 1)
        max_abs = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
        if max_abs < 1e-12:
            return torch.zeros_like(tensor, dtype=self._int_dtype(bits), device="cpu"), 1.0
        scale = max_abs / int_max
        quantized = torch.clamp(torch.round(tensor.detach().cpu() / scale), -int_max, int_max)
        return quantized.to(self._int_dtype(bits)), scale

    def quantize_weights(self, model: nn.Module, bits: int = 8) -> QuantizedModel:
        """Quantize the model state using symmetric per-tensor scaling."""

        state_dict = clone_state_dict(model.state_dict())
        quantized_state: dict[str, torch.Tensor] = {}
        scale_factors: dict[str, float] = {}
        quantized_keys: list[str] = []

        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and self._should_quantize_weight(name, value):
                quantized, scale = self._quantize_tensor(value, bits)
                quantized_state[name] = quantized
                scale_factors[name] = scale
                quantized_keys.append(name)
            elif isinstance(value, torch.Tensor):
                quantized_state[name] = value.detach().cpu().clone()

        activation_ranges = self.quantize_activations(model, bits=16)
        original_bytes = state_size_bytes(state_dict)
        quantized_bytes = state_size_bytes(quantized_state)
        return QuantizedModel(
            quantized_state=quantized_state,
            scale_factors=scale_factors,
            bits=int(bits),
            activation_bits=16,
            activation_ranges=activation_ranges,
            quantized_keys=sorted(quantized_keys),
            original_bytes=original_bytes,
            quantized_bytes=quantized_bytes,
        )

    def quantize_activations(self, model: nn.Module, bits: int = 16) -> dict[str, dict[str, float]]:
        """Estimate fixed-point activation ranges per module from stored state."""

        int_max = float((2 ** (bits - 1)) - 1)
        ranges: dict[str, dict[str, float]] = {}
        for name, module in model.named_modules():
            tensors = [
                parameter.detach().cpu().to(torch.float32)
                for parameter in module.parameters(recurse=False)
                if parameter.is_floating_point() and parameter.numel() > 0
            ]
            tensors.extend(
                buffer.detach().cpu().to(torch.float32)
                for buffer in module.buffers(recurse=False)
                if buffer.is_floating_point() and buffer.numel() > 0
            )
            if not tensors:
                continue
            max_abs = max(float(tensor.abs().max().item()) for tensor in tensors)
            scale = max_abs / int_max if max_abs >= 1e-12 else 1.0
            ranges[name or "<root>"] = {
                "bits": float(bits),
                "min": -max_abs,
                "max": max_abs,
                "scale": scale,
            }
        return ranges

    @staticmethod
    def _restore_quantized_model(original: nn.Module, quantized: QuantizedModel) -> nn.Module:
        restored = copy.deepcopy(original)
        restored.load_state_dict(quantized.dequantized_state_dict(), strict=False)
        return restored

    @staticmethod
    def _model_decision(model: nn.Module, sample: torch.Tensor) -> tuple[float, torch.Tensor]:
        if isinstance(model, ConceptCellCluster):
            output = model(sample)
            return float(output.fired), output.confidence.reshape(-1).to(torch.float32)
        if isinstance(model, BioARNCore):
            output = model.recognize(sample)
            return (0.0 if output.abstained else 1.0), output.concept_direction.reshape(-1).to(torch.float32)
        if isinstance(model, SensorimotorLoop):
            if sample.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.uint8}:
                output = model.step(language_input=sample)
            else:
                output = model.step(visual_input=sample)
            return (0.0 if output.recognition.abstained else 1.0), output.recognition.concept_direction.reshape(-1).to(torch.float32)

        result = model(sample)
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Unsupported model output type for quantization validation: {type(result)!r}.")
        tensor = result.detach().reshape(-1).to(torch.float32)
        if tensor.numel() == 1:
            return float(tensor.item() > 0), tensor
        return float(torch.argmax(tensor).item()), tensor

    def validate_quantization(
        self,
        original: nn.Module,
        quantized: QuantizedModel,
        test_data: list[tuple[torch.Tensor, Any]],
    ) -> QuantizationReport:
        """Compare original and dequantized models on the same test data."""

        quantized_model = self._restore_quantized_model(original, quantized)
        original_hits = 0
        quantized_hits = 0
        max_output_deviation = 0.0

        for sample, expected in test_data:
            original_decision, original_tensor = self._model_decision(original, sample)
            quantized_decision, quantized_tensor = self._model_decision(quantized_model, sample)

            if isinstance(expected, bool):
                expected_value = 1.0 if expected else 0.0
            elif isinstance(expected, (int, float)):
                expected_value = float(expected)
            else:
                expected_value = original_decision

            original_hits += int(original_decision == expected_value)
            quantized_hits += int(quantized_decision == expected_value)
            max_output_deviation = max(
                max_output_deviation,
                float((original_tensor - quantized_tensor).abs().max().item()) if original_tensor.numel() else 0.0,
            )

        component_errors = {
            name: float(
                (
                    original.state_dict()[name].detach().cpu().to(torch.float32)
                    - quantized.dequantized_state_dict()[name].to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
            for name in quantized.quantized_keys
        }
        all_errors = []
        for name in quantized.quantized_keys:
            error = (
                original.state_dict()[name].detach().cpu().to(torch.float32)
                - quantized.dequantized_state_dict()[name].to(torch.float32)
            ).abs()
            all_errors.append(error.reshape(-1))

        error_tensor = torch.cat(all_errors) if all_errors else torch.zeros(1, dtype=torch.float32)
        total = max(len(test_data), 1)
        original_accuracy = original_hits / total
        quantized_accuracy = quantized_hits / total
        return QuantizationReport(
            original_accuracy=float(original_accuracy),
            quantized_accuracy=float(quantized_accuracy),
            accuracy_drop=float(max(0.0, original_accuracy - quantized_accuracy)),
            max_weight_error=float(error_tensor.max().item()),
            mean_weight_error=float(error_tensor.mean().item()),
            bits=int(quantized.bits),
            compression_ratio=float(
                quantized.original_bytes / max(quantized.quantized_bytes, 1)
            ),
            component_errors=component_errors,
            max_output_deviation=max_output_deviation,
        )

    def export_quantized(self, quantized_model: QuantizedModel, path: str | Path) -> Path:
        """Persist a quantized model payload for deployment."""

        target = Path(path)
        payload = {
            "bits": quantized_model.bits,
            "activation_bits": quantized_model.activation_bits,
            "scale_factors": quantized_model.scale_factors,
            "quantized_keys": quantized_model.quantized_keys,
            "activation_ranges": quantized_model.activation_ranges,
            "state": quantized_model.quantized_state,
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        data = stream.getvalue()
        atomic_write_bytes(target, gzip.compress(data) if target.suffix == ".gz" else data)
        return target
