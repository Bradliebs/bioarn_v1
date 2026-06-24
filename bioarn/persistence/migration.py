"""Checkpoint compatibility and migration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioarn import __version__ as BIOARN_VERSION
from bioarn.persistence._common import (
    flatten_dict,
    load_torch_payload,
    normalize_version,
    resize_tensor,
    save_torch_payload,
)
from bioarn.utils import CheckpointManager, ConfigManager


@dataclass
class CompatibilityResult:
    compatible: bool
    detected_version: str
    current_version: str
    missing_fields: list[str] = field(default_factory=list)
    unexpected_fields: list[str] = field(default_factory=list)
    shape_mismatches: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    config_issues: list[str] = field(default_factory=list)


@dataclass
class MigratedCheckpoint:
    path: str
    original_version: str
    target_version: str
    migrated_fields: list[str]
    compatibility: CompatibilityResult


class ModelMigrator:
    """Handle model format changes across Bio-ARN versions."""

    _REQUIRED_FIELDS = (
        "format_version",
        "checkpoint_type",
        "system_type",
        "timestamp",
        "config",
        "state_dict",
        "extra_state",
        "metadata",
    )

    def __init__(self, current_version: str = BIOARN_VERSION) -> None:
        self.current_version = normalize_version(current_version)
        self.checkpoint_manager = CheckpointManager()

    def _resolve_payload(self, checkpoint_path: str | Path) -> dict[str, Any]:
        target = Path(checkpoint_path)
        return self.checkpoint_manager.read_checkpoint(target, resolve=True) if target.exists() else load_torch_payload(target)

    def check_compatibility(self, checkpoint_path: str | Path) -> CompatibilityResult:
        """Check whether a checkpoint can be loaded by the current code."""

        payload = load_torch_payload(checkpoint_path)
        resolved = self.checkpoint_manager.read_checkpoint(checkpoint_path, resolve=True)
        detected_version = str(payload.get("metadata", {}).get("bioarn_version", "0.0.0")).lstrip("v") or "0.0.0"

        missing_fields = [field for field in self._REQUIRED_FIELDS if field not in payload]
        config_issues: list[str] = []
        if int(payload.get("format_version", -1)) != int(CheckpointManager.FORMAT_VERSION):
            config_issues.append(
                f"format_version_mismatch:{payload.get('format_version')}!={CheckpointManager.FORMAT_VERSION}"
            )
        try:
            config = ConfigManager.from_dict(payload.get("config", {}))
        except Exception as exc:
            config = ConfigManager.defaults()
            config_issues.append(f"config_parse_error: {exc}")

        current_flat = flatten_dict(ConfigManager.defaults_dict())
        saved_flat = flatten_dict(payload.get("config", {}))
        config_issues.extend(
            f"missing_config:{key}" for key in sorted(set(current_flat) - set(saved_flat))
        )

        state_dict = resolved.get("state_dict", {})
        missing_state: list[str] = []
        unexpected_fields: list[str] = []
        shape_mismatches: dict[str, dict[str, list[int]]] = {}

        try:
            system = self.checkpoint_manager._instantiate_system(  # type: ignore[attr-defined]
                resolved.get("system_type", "BioARNCore"),
                config,
            )
            expected_state = system.state_dict()
            missing_state = [key for key in expected_state if key not in state_dict]
            unexpected_fields = [key for key in state_dict if key not in expected_state]
            for key in sorted(set(expected_state) & set(state_dict)):
                if expected_state[key].shape != state_dict[key].shape:
                    shape_mismatches[key] = {
                        "expected": list(expected_state[key].shape),
                        "found": list(state_dict[key].shape),
                    }
        except Exception as exc:
            config_issues.append(f"instantiation_error: {exc}")

        missing_fields.extend(missing_state)
        compatible = not any([missing_fields, unexpected_fields, shape_mismatches, config_issues])
        return CompatibilityResult(
            compatible=compatible,
            detected_version=normalize_version(detected_version),
            current_version=self.current_version,
            missing_fields=missing_fields,
            unexpected_fields=unexpected_fields,
            shape_mismatches=shape_mismatches,
            config_issues=config_issues,
        )

    def migrate(self, checkpoint_path: str | Path, target_version: str) -> MigratedCheckpoint:
        """Migrate an older checkpoint into the current runtime format."""

        source_path = Path(checkpoint_path)
        target_semver = normalize_version(target_version)
        raw_payload = load_torch_payload(source_path)
        resolved = self.checkpoint_manager.read_checkpoint(source_path, resolve=True)
        config = ConfigManager.from_dict(raw_payload.get("config", {}))
        normalized_config = ConfigManager.to_dict(config)
        system_type = resolved.get("system_type", raw_payload.get("system_type", "BioARNCore"))
        system = self.checkpoint_manager._instantiate_system(system_type, config)  # type: ignore[attr-defined]
        expected_state = system.state_dict()
        source_state = resolved.get("state_dict", {})

        migrated_state: dict[str, Any] = {}
        migrated_fields: list[str] = []
        for key, reference in expected_state.items():
            current = source_state.get(key)
            if current is None:
                migrated_state[key] = reference.detach().cpu().clone()
                migrated_fields.append(f"added:{key}")
            elif current.shape != reference.shape:
                migrated_state[key] = resize_tensor(current, reference).detach().cpu()
                migrated_fields.append(f"resized:{key}")
            else:
                migrated_state[key] = current.detach().cpu().to(dtype=reference.dtype)

        for key in sorted(set(source_state) - set(expected_state)):
            migrated_fields.append(f"dropped:{key}")

        migrated_payload = {
            "format_version": CheckpointManager.FORMAT_VERSION,
            "checkpoint_type": raw_payload.get("checkpoint_type", "full"),
            "system_type": system_type,
            "timestamp": raw_payload.get("timestamp"),
            "config": normalized_config,
            "state_dict": migrated_state,
            "extra_state": resolved.get("extra_state", raw_payload.get("extra_state", {})),
            "metadata": {
                **raw_payload.get("metadata", {}),
                "bioarn_version": target_semver,
                "migrated_from": str(raw_payload.get("metadata", {}).get("bioarn_version", "0.0.0")),
            },
        }

        suffix = ".pt.gz" if source_path.name.endswith(".pt.gz") else source_path.suffix
        stem = source_path.name[:-6] if source_path.name.endswith(".pt.gz") else source_path.stem
        output_path = source_path.with_name(f"{stem}.migrated.v{target_semver}{suffix}")
        save_torch_payload(output_path, migrated_payload, compress=output_path.suffix == ".gz")
        compatibility = self.check_compatibility(output_path)
        return MigratedCheckpoint(
            path=str(output_path.resolve()),
            original_version=normalize_version(
                str(raw_payload.get("metadata", {}).get("bioarn_version", "0.0.0"))
            ),
            target_version=target_semver,
            migrated_fields=migrated_fields,
            compatibility=compatibility,
        )
