"""Reproducibility helpers for Bio-ARN experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError("config must be a dataclass instance or dictionary.")


class ReproducibilityManager:
    """Ensure reproducible experiments."""

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        return {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count() or 0,
        }

    @classmethod
    def hash_config(cls, config: Any) -> str:
        payload = json.dumps(_config_dict(config), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def save_experiment_manifest(
        cls,
        path: str | os.PathLike[str],
        config: Any,
        git_hash: str,
        results: dict[str, Any],
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": git_hash,
            "config_hash": cls.hash_config(config),
            "config": _config_dict(config),
            "results": results,
            "system_info": cls.get_system_info(),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


__all__ = ["ReproducibilityManager"]
