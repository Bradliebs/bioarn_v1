"""Checkpoint persistence for Bio-ARN systems."""

from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bioarn.loop import SensorimotorLoop
from bioarn.memory.associative_fabric import _ActivationRecord
from bioarn.system import BioARNCore
from bioarn.utils.config_manager import ConfigManager
from bioarn.workspace.gnw import GNWSlot


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: _clone_tensor(value) for key, value in state_dict.items()}


@dataclass
class CheckpointInfo:
    """Summary metadata for an available checkpoint file."""

    path: Path
    created_at: str
    system_type: str
    training_step: int | None
    metrics: dict[str, Any]
    checkpoint_type: str


class _AutoCheckpoint:
    def __init__(
        self,
        manager: "CheckpointManager",
        system: BioARNCore | SensorimotorLoop,
        directory: str | os.PathLike[str],
        interval_steps: int,
        keep_last: int,
    ) -> None:
        self.manager = manager
        self.system = system
        self.directory = Path(directory)
        self.interval_steps = max(1, int(interval_steps))
        self.keep_last = max(1, int(keep_last))
        self.directory.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "_AutoCheckpoint":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is KeyboardInterrupt:
            self.save_now(
                step=getattr(self.system, "timestep", 0),
                metadata={"interrupted": True},
                filename="interrupt.pt",
            )
        return False

    def maybe_save(self, step: int, metadata: dict[str, Any] | None = None) -> Path | None:
        if step <= 0 or step % self.interval_steps != 0:
            return None
        return self.save_now(step=step, metadata=metadata)

    def save_now(
        self,
        *,
        step: int,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> Path:
        path = self.directory / (filename or f"checkpoint-step-{step:08d}.pt")
        saved_path = self.manager.save(self.system, path, metadata=metadata)
        self._prune()
        return saved_path

    def _prune(self) -> None:
        checkpoint_files = sorted(
            self.directory.glob("checkpoint-step-*.pt"),
            key=lambda item: item.stat().st_mtime,
        )
        while len(checkpoint_files) > self.keep_last:
            checkpoint_files[0].unlink(missing_ok=True)
            checkpoint_files.pop(0)


class CheckpointManager:
    """Save and load Bio-ARN system state."""

    FORMAT_VERSION = 1

    def __init__(self, *, compression: bool = False, keep_last: int = 5) -> None:
        self.compression = compression
        self.keep_last = keep_last
        self._last_snapshot: dict[str, Any] | None = None
        self._last_checkpoint_path: Path | None = None

    def save(
        self,
        system: BioARNCore | SensorimotorLoop,
        path: str | os.PathLike[str],
        metadata: dict[str, Any] | None = None,
        *,
        compress: bool | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self._build_snapshot(system, metadata=metadata, checkpoint_type="full")
        self._write_checkpoint(target, snapshot, compress=compress)
        self._last_snapshot = {
            "state_dict": _clone_state_dict(snapshot["state_dict"]),
            "extra_state": snapshot["extra_state"],
        }
        self._last_checkpoint_path = target
        return target

    def save_incremental(
        self,
        system: BioARNCore | SensorimotorLoop,
        path: str | os.PathLike[str],
        step: int,
    ) -> Path:
        if self._last_snapshot is None or self._last_checkpoint_path is None:
            return self.save(system, path, metadata={"training_step": int(step), "incremental": False})

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        current_state = _clone_state_dict(system.state_dict())
        previous_state = self._last_snapshot["state_dict"]
        delta = {
            key: value
            for key, value in current_state.items()
            if key not in previous_state or not torch.equal(value, previous_state[key])
        }
        payload = self._build_snapshot(
            system,
            metadata={"training_step": int(step), "incremental": True},
            checkpoint_type="incremental",
        )
        payload["state_delta"] = delta
        payload["base_checkpoint"] = self._last_checkpoint_path.name
        payload.pop("state_dict", None)
        self._write_checkpoint(target, payload, compress=None)
        self._last_snapshot = {
            "state_dict": current_state,
            "extra_state": payload["extra_state"],
        }
        self._last_checkpoint_path = target
        return target

    def load(
        self,
        path: str | os.PathLike[str],
        config: Any | None = None,
    ) -> BioARNCore | SensorimotorLoop:
        checkpoint = self.read_checkpoint(path, resolve=True)
        saved_config = ConfigManager.from_dict(checkpoint["config"])
        effective_config = saved_config if config is None else ConfigManager.from_dict(config)
        self._validate_compatibility(saved_config, effective_config)

        system = self._instantiate_system(checkpoint["system_type"], effective_config)
        self._load_state_dict_robust(system, checkpoint["state_dict"])
        self._restore_extra_state(system, checkpoint.get("extra_state", {}))
        try:
            system.to(torch.device(effective_config.device))
        except Exception:
            system.to(torch.device("cpu"))
        return system

    def read_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        resolve: bool = True,
    ) -> dict[str, Any]:
        target = Path(path)
        checkpoint = self._read_raw_checkpoint(target)
        if not resolve or checkpoint.get("checkpoint_type") != "incremental":
            return checkpoint

        base_name = checkpoint["base_checkpoint"]
        base_path = target.parent / base_name
        base_checkpoint = self.read_checkpoint(base_path, resolve=True)
        base_checkpoint["state_dict"].update(checkpoint.get("state_delta", {}))
        base_checkpoint["extra_state"] = checkpoint.get("extra_state", base_checkpoint.get("extra_state", {}))
        base_checkpoint["metadata"] = {
            **base_checkpoint.get("metadata", {}),
            **checkpoint.get("metadata", {}),
        }
        base_checkpoint["timestamp"] = checkpoint.get("timestamp", base_checkpoint.get("timestamp"))
        base_checkpoint["checkpoint_type"] = "resolved_incremental"
        return base_checkpoint

    def list_checkpoints(self, directory: str | os.PathLike[str]) -> list[CheckpointInfo]:
        root = Path(directory)
        patterns = ("*.pt", "*.ckpt", "*.pt.gz", "*.ckpt.gz")
        files = sorted(
            {
                file
                for pattern in patterns
                for file in root.glob(pattern)
            }
        )
        checkpoints: list[CheckpointInfo] = []
        for file in files:
            payload = self.read_checkpoint(file, resolve=False)
            metadata = payload.get("metadata", {})
            checkpoints.append(
                CheckpointInfo(
                    path=file,
                    created_at=payload.get("timestamp", ""),
                    system_type=payload.get("system_type", "unknown"),
                    training_step=metadata.get("training_step"),
                    metrics=metadata.get("metrics", {}),
                    checkpoint_type=payload.get("checkpoint_type", "full"),
                )
            )
        return sorted(checkpoints, key=lambda item: item.created_at)

    def auto_checkpoint(
        self,
        system: BioARNCore | SensorimotorLoop,
        directory: str | os.PathLike[str],
        *,
        interval_steps: int = 1000,
        keep_last: int | None = None,
    ) -> _AutoCheckpoint:
        return _AutoCheckpoint(
            self,
            system,
            directory,
            interval_steps=interval_steps,
            keep_last=keep_last or self.keep_last,
        )

    def _write_checkpoint(
        self,
        path: Path,
        checkpoint: dict[str, Any],
        *,
        compress: bool | None,
    ) -> None:
        use_compression = self.compression if compress is None else compress
        if path.suffix == ".gz":
            use_compression = True

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        if use_compression:
            with gzip.open(path, "wb") as handle:
                handle.write(buffer.getvalue())
        else:
            path.write_bytes(buffer.getvalue())

    def _read_raw_checkpoint(self, path: Path) -> dict[str, Any]:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                return torch.load(handle, map_location="cpu")
        return torch.load(path, map_location="cpu")

    def _build_snapshot(
        self,
        system: BioARNCore | SensorimotorLoop,
        *,
        metadata: dict[str, Any] | None,
        checkpoint_type: str,
    ) -> dict[str, Any]:
        config = ConfigManager.to_dict(system.config)
        payload_metadata = dict(metadata or {})
        payload_metadata.setdefault("training_step", int(getattr(system, "timestep", 0)))
        return {
            "format_version": self.FORMAT_VERSION,
            "checkpoint_type": checkpoint_type,
            "system_type": type(system).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "state_dict": _clone_state_dict(system.state_dict()),
            "extra_state": self._capture_extra_state(system),
            "metadata": payload_metadata,
        }

    def _capture_extra_state(self, system: BioARNCore | SensorimotorLoop) -> dict[str, Any]:
        if isinstance(system, SensorimotorLoop):
            return {
                "timestep": int(system.timestep),
                "feedback_features": _clone_tensor(system._feedback_features),
                "generated_token_history": list(system._generated_token_history),
                "hierarchy": {
                    "last_states": [_clone_tensor(state) for state in system.hierarchy.get_level_states()],
                    "last_errors": [_clone_tensor(error) for error in system.hierarchy.get_level_errors()],
                },
                "core": self._capture_core_state(system.core),
            }
        return self._capture_core_state(system)

    def _capture_core_state(self, core: BioARNCore) -> dict[str, Any]:
        fabric = core.fabric
        gnw = core.gnw
        return {
            "timestep": int(core.timestep),
            "fabric": {
                "association_strength": [
                    {
                        "src": int(src),
                        "dst": int(dst),
                        "strength": float(strength),
                        "temporal": bool(fabric.association_temporal.get((src, dst), False)),
                    }
                    for (src, dst), strength in fabric.association_strength.items()
                ],
                "activation_history": [
                    {
                        "ccc_index": int(record.ccc_index),
                        "direction": _clone_tensor(record.direction),
                        "confidence": float(record.confidence),
                        "timestep": int(record.timestep),
                        "address": _clone_tensor(record.address),
                    }
                    for record in fabric.activation_history
                ],
                "concept_directions": {
                    int(index): _clone_tensor(direction)
                    for index, direction in fabric.concept_directions.items()
                },
                "last_decay_timestep": fabric._last_decay_timestep,
                "temporal_recent_activations": [
                    {
                        "address": _clone_tensor(address),
                        "data": _clone_tensor(data),
                        "timestamp": float(timestamp),
                    }
                    for address, data, timestamp in fabric.temporal_associator.recent_activations
                ],
            },
            "gnw": {
                "slots": [self._serialize_slot(slot) for slot in gnw.slots],
                "broadcast_history": [
                    self._serialize_slot(slot) if slot is not None else None
                    for slot in gnw.broadcast_history
                ],
                "history_cursor": int(gnw._history_cursor),
                "history_count": int(gnw._history_count),
                "last_new_entries": list(gnw.last_new_entries),
                "last_evicted": list(gnw.last_evicted),
            },
        }

    @staticmethod
    def _serialize_slot(slot: GNWSlot) -> dict[str, Any]:
        return {
            "ccc_index": int(slot.ccc_index),
            "direction": _clone_tensor(slot.direction),
            "activation": float(slot.activation),
            "confidence": float(slot.confidence),
            "age": int(slot.age),
            "fatigue": float(slot.fatigue),
        }

    @staticmethod
    def _deserialize_slot(payload: dict[str, Any]) -> GNWSlot:
        return GNWSlot(
            ccc_index=int(payload["ccc_index"]),
            direction=payload["direction"].detach().clone(),
            activation=float(payload["activation"]),
            confidence=float(payload["confidence"]),
            age=int(payload["age"]),
            fatigue=float(payload["fatigue"]),
        )

    def _restore_extra_state(self, system: BioARNCore | SensorimotorLoop, state: dict[str, Any]) -> None:
        if isinstance(system, SensorimotorLoop):
            system.timestep = int(state.get("timestep", 0))
            system._feedback_features = state.get("feedback_features", torch.zeros_like(system._feedback_features)).detach().clone()
            system._generated_token_history = list(state.get("generated_token_history", []))
            hierarchy_state = state.get("hierarchy", {})
            if hierarchy_state:
                system.hierarchy._last_states = [
                    tensor.detach().clone()
                    for tensor in hierarchy_state.get("last_states", [])
                ]
                system.hierarchy._last_errors = [
                    tensor.detach().clone()
                    for tensor in hierarchy_state.get("last_errors", [])
                ]
            self._restore_core_state(system.core, state.get("core", {}))
            system._apply_modulation(system.reward.get_modulation())
            return
        self._restore_core_state(system, state)

    def _restore_core_state(self, core: BioARNCore, state: dict[str, Any]) -> None:
        core.timestep = int(state.get("timestep", 0))

        fabric_state = state.get("fabric", {})
        fabric = core.fabric
        fabric.association_strength = {
            (int(item["src"]), int(item["dst"])): float(item["strength"])
            for item in fabric_state.get("association_strength", [])
        }
        fabric.association_temporal = {
            (int(item["src"]), int(item["dst"])): bool(item["temporal"])
            for item in fabric_state.get("association_strength", [])
        }
        fabric.activation_history = [
            _ActivationRecord(
                ccc_index=int(item["ccc_index"]),
                direction=item["direction"].detach().clone(),
                confidence=float(item["confidence"]),
                timestep=int(item["timestep"]),
                address=item["address"].detach().clone(),
            )
            for item in fabric_state.get("activation_history", [])
        ]
        fabric.concept_directions = {
            int(index): tensor.detach().clone()
            for index, tensor in fabric_state.get("concept_directions", {}).items()
        }
        fabric._last_decay_timestep = fabric_state.get("last_decay_timestep")
        fabric.temporal_associator.recent_activations = [
            (
                item["address"].detach().clone(),
                item["data"].detach().clone(),
                float(item["timestamp"]),
            )
            for item in fabric_state.get("temporal_recent_activations", [])
        ]

        gnw_state = state.get("gnw", {})
        gnw = core.gnw
        gnw.slots = [self._deserialize_slot(slot) for slot in gnw_state.get("slots", [])]
        gnw.broadcast_history = [
            self._deserialize_slot(slot) if slot is not None else None
            for slot in gnw_state.get("broadcast_history", [])
        ]
        gnw._history_cursor = int(gnw_state.get("history_cursor", 0))
        gnw._history_count = int(gnw_state.get("history_count", 0))
        gnw.last_new_entries = list(gnw_state.get("last_new_entries", []))
        gnw.last_evicted = list(gnw_state.get("last_evicted", []))

    @staticmethod
    def _instantiate_system(system_type: str, config: Any) -> BioARNCore | SensorimotorLoop:
        normalized = system_type.lower()
        if normalized == "sensorimotorloop":
            return SensorimotorLoop(config)
        if normalized == "bioarncore":
            return BioARNCore(config)
        raise ValueError(f"Unsupported checkpoint system type: {system_type}")

    def _load_state_dict_robust(
        self,
        system: BioARNCore | SensorimotorLoop,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        current_state = system.state_dict()
        matched: dict[str, torch.Tensor] = {}
        deferred: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if isinstance(current_value, torch.Tensor) and isinstance(value, torch.Tensor):
                if current_value.shape == value.shape:
                    matched[key] = value
                else:
                    deferred[key] = value
            else:
                matched[key] = value

        incompatible = system.load_state_dict(matched, strict=False)
        unresolved_missing = [key for key in incompatible.missing_keys if key not in deferred]
        unresolved_unexpected = [key for key in incompatible.unexpected_keys if key not in deferred]
        if unresolved_missing or unresolved_unexpected:
            raise RuntimeError(
                f"Checkpoint state mismatch. missing={unresolved_missing}, unexpected={unresolved_unexpected}"
            )

        for key, value in deferred.items():
            self._assign_tensor(system, key, value)

    @staticmethod
    def _assign_tensor(system: BioARNCore | SensorimotorLoop, key: str, value: torch.Tensor) -> None:
        target: Any = system
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)

        name = parts[-1]
        current = getattr(target, name)
        tensor = value.detach().clone()
        if isinstance(current, nn.Parameter):
            setattr(target, name, nn.Parameter(tensor, requires_grad=current.requires_grad))
        else:
            setattr(target, name, tensor)

    @staticmethod
    def _validate_compatibility(saved_config: Any, requested_config: Any) -> None:
        saved = ConfigManager.to_dict(saved_config)
        requested = ConfigManager.to_dict(requested_config)
        critical_paths = (
            "ccc.input_dim",
            "ccc.concept_dim",
            "ccc.num_f1_features",
            "ccc.max_pool_size",
            "sdm.address_dim",
            "sdm.num_hard_locations",
            "sdm.data_dim",
            "predictive.num_levels",
            "gnw.capacity",
        )
        for path in critical_paths:
            saved_value = CheckpointManager._get_path(saved, path)
            requested_value = CheckpointManager._get_path(requested, path)
            if saved_value != requested_value:
                raise ValueError(
                    f"Checkpoint config mismatch for {path}: saved={saved_value}, requested={requested_value}."
                )

    @staticmethod
    def _get_path(payload: dict[str, Any], path: str) -> Any:
        current: Any = payload
        for part in path.split("."):
            current = current[part]
        return current


__all__ = ["CheckpointInfo", "CheckpointManager"]
