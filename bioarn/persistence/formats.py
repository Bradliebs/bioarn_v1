"""Export Bio-ARN models to deployment and inspection formats."""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bioarn.hardware import LoihiMapping
from bioarn.persistence._common import atomic_write_bytes, atomic_write_json, ensure_dir
from bioarn.persistence.quantization import ModelQuantizer


class ModelExporter:
    """Export Bio-ARN models to various formats."""

    def __init__(self) -> None:
        self.quantizer = ModelQuantizer()

    def to_onnx(self, model: nn.Module, sample_input: Any, path: str | os.PathLike[str]) -> Path:
        """Export the model graph to ONNX when supported by the runtime graph."""

        target = Path(path)
        ensure_dir(target.parent)
        model.eval()
        try:
            torch.onnx.export(
                model,
                sample_input,
                target,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        except Exception as exc:
            raise RuntimeError("ONNX export failed for this Bio-ARN model.") from exc
        return target

    def to_json(self, model: nn.Module, path: str | os.PathLike[str]) -> Path:
        """Export all weights as parseable JSON for debugging and visualization."""

        payload = {
            "state_dict": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "values": tensor.detach().cpu().tolist(),
                }
                for name, tensor in model.state_dict().items()
            }
        }
        target = Path(path)
        atomic_write_json(target, payload)
        return target

    def to_loihi_lava(self, model: nn.Module, path: str | os.PathLike[str]) -> Path:
        """Export Loihi/Lava deployment artifacts."""

        target = Path(path)
        ensure_dir(target)
        quantized = self.quantizer.quantize_weights(model, bits=8)
        mapping = LoihiMapping(model.config).map_full_system()  # type: ignore[attr-defined]
        config_payload = {
            "system": asdict(mapping),
            "activation_ranges": quantized.activation_ranges,
            "scale_factors": quantized.scale_factors,
            "fixed_point": {"weights": 8, "activations": 16},
        }
        atomic_write_json(target / "lava_config.json", config_payload)
        self.quantizer.export_quantized(quantized, target / "quantized_weights.pt")
        return target

    def to_numpy(self, model: nn.Module, path: str | os.PathLike[str]) -> Path:
        """Export all model tensors to a compressed NPZ archive."""

        target = Path(path)
        ensure_dir(target.parent)
        temp = target.with_name(f"{target.stem}.tmp{target.suffix}")
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        }
        try:
            np.savez_compressed(temp, **arrays)
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)
        return target

