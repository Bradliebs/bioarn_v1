"""Filesystem-backed model registry for Bio-ARN systems."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from bioarn import __version__ as BIOARN_VERSION
from bioarn.hardware import LoihiMapping
from bioarn.hardware.lava_bridge import LavaBridge
from bioarn.persistence._common import (
    CHECKPOINT_NAMES,
    atomic_write_json,
    atomic_write_yaml,
    count_parameters,
    diff_dicts,
    dir_size_bytes,
    ensure_dir,
    normalize_version,
    semver_key,
    write_optional_gzip_copy,
)
from bioarn.persistence.formats import ModelExporter
from bioarn.persistence.quantization import ModelQuantizer
from bioarn.utils import CheckpointManager, ConfigManager


@dataclass
class ModelInfo:
    name: str
    version: str
    created_at: str
    config: dict
    metrics: dict
    param_count: int
    checkpoint_path: str
    size_bytes: int


@dataclass
class ComparisonResult:
    model_a: ModelInfo
    model_b: ModelInfo
    param_count_delta: int
    metric_diff: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_diff: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class LoihiExport:
    output_dir: str
    num_components: int
    total_neurons: int
    total_synapses: int
    quantization_bits: int
    files_created: list[str]


class ModelStore:
    """Local model registry for managing trained Bio-ARN models."""

    def __init__(self, store_dir: str = "models/"):
        self.store_dir = Path(store_dir)
        ensure_dir(self.store_dir)
        self.checkpoint_manager = CheckpointManager()
        self.quantizer = ModelQuantizer()
        self.exporter = ModelExporter()

    def _model_dir(self, name: str) -> Path:
        return self.store_dir / name

    def _version_dir(self, name: str, version: str) -> Path:
        return self._model_dir(name) / f"v{normalize_version(version)}"

    def _staging_dir(self, name: str, version: str) -> Path:
        model_dir = self._model_dir(name)
        ensure_dir(model_dir)
        return model_dir / f"v{normalize_version(version)}.staging-{uuid.uuid4().hex}"

    def _manifest_path(self, version_dir: Path) -> Path:
        return version_dir / "manifest.json"

    def _load_info(self, version_dir: Path) -> ModelInfo:
        payload = json.loads(self._manifest_path(version_dir).read_text(encoding="utf-8"))
        return ModelInfo(
            name=payload["name"],
            version=payload["version"],
            created_at=payload["created_at"],
            config=payload["config"],
            metrics=payload["metrics"],
            param_count=int(payload["param_count"]),
            checkpoint_path=payload["checkpoint_path"],
            size_bytes=int(payload["size_bytes"]),
        )

    def _resolve_version_dir(self, name: str, version: str) -> Path:
        model_dir = self._model_dir(name)
        if version != "latest":
            version_dir = self._version_dir(name, version)
            if not version_dir.exists():
                raise FileNotFoundError(f"Model {name!r} version {version!r} was not found.")
            return version_dir

        versions = [
            candidate
            for candidate in model_dir.iterdir()
            if candidate.is_dir() and candidate.name.startswith("v")
        ] if model_dir.exists() else []
        if not versions:
            raise FileNotFoundError(f"No saved versions found for model {name!r}.")
        return max(versions, key=lambda item: semver_key(item.name[1:]))

    def _swap_version_directory(self, staging_dir: Path, final_dir: Path) -> None:
        backup_dir: Path | None = None
        ensure_dir(final_dir.parent)
        try:
            if final_dir.exists():
                backup_dir = final_dir.with_name(f"{final_dir.name}.backup-{uuid.uuid4().hex}")
                final_dir.rename(backup_dir)
            staging_dir.rename(final_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
                backup_dir.rename(final_dir)
            raise
        else:
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def save_model(
        self,
        system: nn.Module,
        name: str,
        version: str,
        metrics: dict,
        config,
    ) -> ModelInfo:
        """Save a trained model version and its registry metadata."""

        version = normalize_version(version)
        config_obj = ConfigManager.from_dict(config)
        if hasattr(system, "config"):
            self.checkpoint_manager._validate_compatibility(  # type: ignore[attr-defined]
                ConfigManager.from_dict(system.config),
                config_obj,
            )

        staging_dir = self._staging_dir(name, version)
        ensure_dir(staging_dir)
        created_at = datetime.now(timezone.utc).isoformat()
        checkpoint_path = staging_dir / "model.pt"
        self.checkpoint_manager.save(
            system,
            checkpoint_path,
            metadata={
                "model_name": name,
                "model_version": version,
                "metrics": metrics,
                "bioarn_version": BIOARN_VERSION,
            },
        )
        write_optional_gzip_copy(checkpoint_path)

        config_dict = ConfigManager.to_dict(config_obj)
        atomic_write_yaml(staging_dir / "config.yaml", config_dict)
        atomic_write_json(staging_dir / "metrics.json", metrics)
        final_checkpoint_path = self._version_dir(name, version) / "model.pt"

        info = ModelInfo(
            name=name,
            version=version,
            created_at=created_at,
            config=config_dict,
            metrics=dict(metrics),
            param_count=count_parameters(system),
            checkpoint_path=str(final_checkpoint_path.resolve()),
            size_bytes=0,
        )
        manifest_payload = {
            **asdict(info),
            "bioarn_version": BIOARN_VERSION,
            "system_type": type(system).__name__,
        }
        atomic_write_json(staging_dir / "manifest.json", manifest_payload)

        info.size_bytes = dir_size_bytes(staging_dir)
        manifest_payload["size_bytes"] = info.size_bytes
        atomic_write_json(staging_dir / "manifest.json", manifest_payload)

        final_dir = self._version_dir(name, version)
        self._swap_version_directory(staging_dir, final_dir)
        return self._load_info(final_dir)

    def load_model(self, name: str, version: str = "latest") -> tuple[nn.Module, ModelInfo]:
        """Load a saved model version and its metadata."""

        version_dir = self._resolve_version_dir(name, version)
        info = self._load_info(version_dir)
        config_path = version_dir / "config.yaml"
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        checkpoint_path = Path(info.checkpoint_path)
        if not checkpoint_path.exists():
            for candidate_name in CHECKPOINT_NAMES:
                candidate = version_dir / candidate_name
                if candidate.exists():
                    checkpoint_path = candidate
                    break

        checkpoint = self.checkpoint_manager.read_checkpoint(checkpoint_path, resolve=False)
        self.checkpoint_manager._validate_compatibility(  # type: ignore[attr-defined]
            ConfigManager.from_dict(checkpoint["config"]),
            ConfigManager.from_dict(config_dict),
        )
        system = self.checkpoint_manager.load(checkpoint_path, config=config_dict)
        return system, info

    def list_models(self) -> list[ModelInfo]:
        """List all stored model versions in semantic-version order."""

        manifests = list(self.store_dir.glob("*/*/manifest.json"))
        infos = [self._load_info(manifest.parent) for manifest in manifests]
        return sorted(infos, key=lambda item: (item.name, semver_key(item.version)))

    def compare_models(
        self,
        name_a: str,
        version_a: str,
        name_b: str,
        version_b: str,
    ) -> ComparisonResult:
        """Compare two saved model versions."""

        _, info_a = self.load_model(name_a, version_a)
        _, info_b = self.load_model(name_b, version_b)
        metric_diff = {
            key: {
                "left": info_a.metrics.get(key),
                "right": info_b.metrics.get(key),
                "delta": (
                    float(info_b.metrics[key]) - float(info_a.metrics[key])
                    if isinstance(info_a.metrics.get(key), (int, float))
                    and isinstance(info_b.metrics.get(key), (int, float))
                    else None
                ),
            }
            for key in sorted(set(info_a.metrics) | set(info_b.metrics))
        }
        return ComparisonResult(
            model_a=info_a,
            model_b=info_b,
            param_count_delta=int(info_b.param_count - info_a.param_count),
            metric_diff=metric_diff,
            config_diff=diff_dicts(info_a.config, info_b.config),
        )

    def delete_model(self, name: str, version: str) -> None:
        """Remove a stored model version."""

        version_dir = self._version_dir(name, version)
        if version_dir.exists():
            shutil.rmtree(version_dir)
        model_dir = self._model_dir(name)
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()

    def export_for_loihi(self, name: str, version: str, output_dir: str) -> LoihiExport:
        """Export a stored model into Loihi/Lava-aligned artifacts."""

        model, info = self.load_model(name, version)
        core = getattr(model, "core", model)
        mapping = LoihiMapping(core.config).map_full_system()  # type: ignore[attr-defined]
        bridge = LavaBridge()
        process_graph = bridge.convert_full_model(model)
        staging_dir = Path(f"{output_dir}.staging-{uuid.uuid4().hex}")
        ensure_dir(staging_dir)

        quantized = self.quantizer.quantize_weights(core, bits=8)
        self.quantizer.export_quantized(quantized, staging_dir / "quantized_weights.pt")
        atomic_write_json(staging_dir / "activation_ranges.json", quantized.activation_ranges)
        atomic_write_json(
            staging_dir / "lava_config.json",
            {
                "lif": LoihiMapping(core.config).map_lif_neuron().__dict__,  # type: ignore[attr-defined]
                "ccc": LoihiMapping(core.config).map_ccc_to_cores().__dict__,  # type: ignore[attr-defined]
                "sdm": LoihiMapping(core.config).map_sdm_to_memory().__dict__,  # type: ignore[attr-defined]
                "predictive": LoihiMapping(core.config).map_pe_to_pipeline().__dict__,  # type: ignore[attr-defined]
                "gnw": LoihiMapping(core.config).map_gnw_to_circuit().__dict__,  # type: ignore[attr-defined]
                "system": mapping.__dict__,
                "process_graph": bridge.serialize_process_graph(process_graph),
                "model_info": asdict(info),
            },
        )
        atomic_write_json(staging_dir / "deployment_config.json", bridge.serialize_process_graph(process_graph))
        self.exporter.to_numpy(core, staging_dir / "weights_float.npz")
        atomic_write_json(
            staging_dir / "component_manifest.json",
            {
                "components": ["ccc", "sdm", "predictive", "gnw"],
                "quantization_bits": 8,
                "activation_bits": 16,
            },
        )

        final_dir = Path(output_dir)
        self._swap_version_directory(staging_dir, final_dir)
        files_created = sorted(str(path.resolve()) for path in final_dir.rglob("*") if path.is_file())
        return LoihiExport(
            output_dir=str(final_dir.resolve()),
            num_components=4,
            total_neurons=int(mapping.total_neurons),
            total_synapses=int(mapping.total_synapses),
            quantization_bits=8,
            files_created=files_created,
        )
