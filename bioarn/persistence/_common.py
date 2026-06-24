"""Shared helpers for Bio-ARN model persistence."""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

SEMVER_PATTERN = re.compile(r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
CHECKPOINT_NAMES = ("model.pt", "model.pt.gz")
MODEL_FILE_THRESHOLD_BYTES = 1_000_000


def normalize_version(version: str) -> str:
    match = SEMVER_PATTERN.fullmatch(str(version).strip())
    if match is None:
        raise ValueError(f"Expected semantic version 'MAJOR.MINOR.PATCH', got {version!r}.")
    return ".".join(match.groups())


def semver_key(version: str) -> tuple[int, int, int]:
    normalized = normalize_version(version)
    return tuple(int(part) for part in normalized.split("."))


def flatten_dict(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, path))
        else:
            flattened[path] = value
    return flattened


def diff_dicts(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    left_flat = flatten_dict(left)
    right_flat = flatten_dict(right)
    return {
        key: {"left": left_flat.get(key), "right": right_flat.get(key)}
        for key in sorted(set(left_flat) | set(right_flat))
        if left_flat.get(key) != right_flat.get(key)
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_size_bytes(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def dir_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp.write_bytes(data)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_yaml(path: str | os.PathLike[str], payload: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def save_torch_payload(
    path: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    compress: bool | None = None,
) -> None:
    target = Path(path)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    use_compression = target.suffix == ".gz" if compress is None else bool(compress)
    atomic_write_bytes(target, gzip.compress(raw) if use_compression else raw)


def load_torch_payload(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    if target.suffix == ".gz":
        with gzip.open(target, "rb") as handle:
            return torch.load(handle, map_location="cpu")
    return torch.load(target, map_location="cpu")


def resize_tensor(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    resized = torch.zeros_like(reference)
    if source.dim() == reference.dim():
        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, reference.shape))
        resized[slices] = source.detach().to(dtype=reference.dtype, device=reference.device)[slices]
        return resized

    source_flat = source.detach().reshape(-1).to(dtype=reference.dtype, device=reference.device)
    resized_flat = resized.reshape(-1)
    count = min(source_flat.numel(), resized_flat.numel())
    resized_flat[:count] = source_flat[:count]
    return resized


def clone_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = value
    return cloned


def write_optional_gzip_copy(path: Path) -> Path | None:
    if not path.exists() or path.stat().st_size < MODEL_FILE_THRESHOLD_BYTES:
        return None
    compressed_path = path.with_suffix(path.suffix + ".gz")
    atomic_write_bytes(compressed_path, gzip.compress(path.read_bytes()))
    return compressed_path

