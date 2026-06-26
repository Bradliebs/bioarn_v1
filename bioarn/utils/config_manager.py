"""Configuration loading and validation for Bio-ARN."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from bioarn.config import (
    BioARNConfig,
    CCCConfig,
    GNWConfig,
    MarginGateConfig,
    PredictiveConfig,
    RewardConfig,
    SDMConfig,
    SpikingConfig,
)

_NESTED_CONFIGS = {
    "spiking": SpikingConfig,
    "margin_gate": MarginGateConfig,
    "ccc": CCCConfig,
    "sdm": SDMConfig,
    "predictive": PredictiveConfig,
    "gnw": GNWConfig,
    "reward": RewardConfig,
}


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _assign_path(target: dict[str, Any], path: str, value: Any) -> None:
    current = target
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


class ConfigManager:
    """Production configuration management."""

    PRESETS: dict[str, dict[str, Any]] = {
        "mnist": {
            "device": "cpu",
            "seed": 42,
            "ccc": {
                "input_dim": 784,
                "concept_dim": 256,
                "num_f1_features": 128,
                "f1_top_k": 32,
                "max_pool_size": 1000,
            },
            "sdm": {
                "address_dim": 10000,
                "hamming_radius": 451,
                "num_hard_locations": 1000,
                "data_dim": 256,
            },
        },
        "cifar": {
            "device": "cpu",
            "seed": 42,
            "ccc": {
                "input_dim": 3072,
                "concept_dim": 512,
                "num_f1_features": 256,
                "f1_top_k": 48,
                "slow_lr": 0.02,
                "feedback_lr": 0.02,
                "max_pool_size": 2000,
            },
            "sdm": {
                "address_dim": 12000,
                "hamming_radius": 600,
                "num_hard_locations": 2048,
                "data_dim": 512,
            },
            "predictive": {
                "num_levels": 5,
            },
        },
        "language": {
            "device": "cpu",
            "seed": 42,
            "ccc": {
                "input_dim": 256,
                "concept_dim": 256,
                "num_f1_features": 192,
                "f1_top_k": 48,
                "max_pool_size": 2048,
            },
            "sdm": {
                "address_dim": 8192,
                "hamming_radius": 368,
                "num_hard_locations": 2048,
                "data_dim": 256,
            },
            "predictive": {
                "num_levels": 5,
            },
        },
        "language_small": {
            "device": "cpu",
            "seed": 42,
            "ccc": {
                "input_dim": 256,
                "concept_dim": 256,
                "num_f1_features": 192,
                "f1_top_k": 48,
                "max_pool_size": 2048,
            },
            "sdm": {
                "address_dim": 8192,
                "hamming_radius": 368,
                "num_hard_locations": 2048,
                "data_dim": 256,
            },
            "predictive": {
                "num_levels": 5,
            },
        },
        "language_large": {
            "device": "cpu",
            "seed": 42,
            "ccc": {
                "input_dim": 512,
                "concept_dim": 512,
                "num_f1_features": 384,
                "f1_top_k": 64,
                "slow_lr": 0.02,
                "feedback_lr": 0.02,
                "max_pool_size": 4096,
            },
            "sdm": {
                "address_dim": 16384,
                "hamming_radius": 768,
                "num_hard_locations": 4096,
                "data_dim": 512,
            },
            "predictive": {
                "num_levels": 6,
            },
            "gnw": {
                "capacity": 9,
            },
        },
        "production": {
            "device": "cpu",
            "dtype": "float32",
            "seed": 42,
            "ccc": {
                "max_pool_size": 4096,
                "slow_lr": 0.02,
                "feedback_lr": 0.02,
            },
            "sdm": {
                "num_hard_locations": 4096,
                "decay_rate": 0.9995,
            },
            "predictive": {
                "num_levels": 5,
            },
            "gnw": {
                "capacity": 9,
            },
        },
    }

    ENV_ALIASES = {
        "BIOARN_CCC_POOL_SIZE": "ccc.max_pool_size",
    }

    @staticmethod
    def to_dict(config: BioARNConfig | Mapping[str, Any]) -> dict[str, Any]:
        if is_dataclass(config):
            return asdict(config)
        return dict(config)

    @classmethod
    def defaults(cls) -> BioARNConfig:
        return BioARNConfig()

    @classmethod
    def defaults_dict(cls) -> dict[str, Any]:
        return cls.to_dict(cls.defaults())

    @classmethod
    def load_preset(cls, name: str) -> BioARNConfig:
        preset = cls.PRESETS.get(name.lower())
        if preset is None:
            raise KeyError(f"Unknown preset: {name}")
        return cls.from_dict(preset)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | BioARNConfig) -> BioARNConfig:
        if isinstance(data, BioARNConfig):
            cls.validate(data)
            return data

        merged = _deep_merge(cls.defaults_dict(), dict(data))
        nested_kwargs = {
            name: _NESTED_CONFIGS[name](**merged[name])
            for name in _NESTED_CONFIGS
        }
        workspace_config = merged.get("workspace")
        config = BioARNConfig(
            **nested_kwargs,
            workspace=GNWConfig(**workspace_config) if isinstance(workspace_config, Mapping) else None,
            device=str(merged["device"]),
            dtype=str(merged["dtype"]),
            seed=int(merged["seed"]),
        )
        cls.validate(config)
        return config

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> BioARNConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> BioARNConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_env(cls) -> BioARNConfig:
        return cls.from_dict(cls._env_overrides())

    @classmethod
    def load(
        cls,
        *,
        file_path: str | os.PathLike[str] | None = None,
        preset: str | None = None,
        cli_args: Mapping[str, Any] | Any | None = None,
        include_env: bool = True,
    ) -> BioARNConfig:
        merged = cls.defaults_dict()
        if preset:
            merged = _deep_merge(merged, cls.to_dict(cls.load_preset(preset)))
        if file_path:
            source_path = Path(file_path)
            if source_path.suffix.lower() in {".yaml", ".yml"}:
                merged = _deep_merge(merged, cls.to_dict(cls.from_yaml(source_path)))
            elif source_path.suffix.lower() == ".json":
                merged = _deep_merge(merged, cls.to_dict(cls.from_json(source_path)))
            else:
                raise ValueError(f"Unsupported config format: {source_path.suffix}")
        if include_env:
            merged = _deep_merge(merged, cls._env_overrides())
        if cli_args is not None:
            merged = _deep_merge(merged, cls._cli_overrides(cli_args))
        return cls.from_dict(merged)

    @classmethod
    def to_yaml(cls, config: BioARNConfig, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(cls.to_dict(config), sort_keys=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def validate(cls, config: BioARNConfig) -> None:
        if config.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if config.dtype not in {"float16", "float32", "float64", "bfloat16"}:
            raise ValueError("dtype must be one of float16, float32, float64, or bfloat16.")
        if config.seed < 0:
            raise ValueError("seed must be non-negative.")

        cls._require(0.0 <= config.spiking.beta <= 1.0, "spiking.beta must be between 0 and 1.")
        cls._require(config.spiking.threshold > 0.0, "spiking.threshold must be positive.")
        cls._require(config.spiking.dt > 0.0, "spiking.dt must be positive.")
        cls._require(config.spiking.refractory_steps >= 0, "spiking.refractory_steps must be >= 0.")

        cls._require(0.0 <= config.margin_gate.theta_margin <= 1.0, "margin_gate.theta_margin must be between 0 and 1.")
        cls._require(config.margin_gate.theta_margin_lr >= 0.0, "margin_gate.theta_margin_lr must be >= 0.")
        cls._require(0.0 <= config.margin_gate.theta_resonance <= 1.0, "margin_gate.theta_resonance must be between 0 and 1.")

        cls._require(config.ccc.input_dim > 0, "ccc.input_dim must be positive.")
        cls._require(config.ccc.concept_dim > 0, "ccc.concept_dim must be positive.")
        cls._require(config.ccc.num_f1_features > 0, "ccc.num_f1_features must be positive.")
        cls._require(0 < config.ccc.f1_top_k <= config.ccc.num_f1_features, "ccc.f1_top_k must be in (0, num_f1_features].")
        cls._require(config.ccc.fast_lr >= 0.0, "ccc.fast_lr must be >= 0.")
        cls._require(config.ccc.slow_lr >= 0.0, "ccc.slow_lr must be >= 0.")
        cls._require(config.ccc.feedback_lr >= 0.0, "ccc.feedback_lr must be >= 0.")
        cls._require(config.ccc.max_pool_size > 0, "ccc.max_pool_size must be positive.")

        cls._require(config.sdm.address_dim > 0, "sdm.address_dim must be positive.")
        cls._require(0 <= config.sdm.hamming_radius <= config.sdm.address_dim, "sdm.hamming_radius must be between 0 and address_dim.")
        cls._require(config.sdm.num_hard_locations > 0, "sdm.num_hard_locations must be positive.")
        cls._require(config.sdm.data_dim > 0, "sdm.data_dim must be positive.")
        cls._require(0.0 < config.sdm.decay_rate <= 1.0, "sdm.decay_rate must be in (0, 1].")
        cls._require(config.sdm.stdp_window > 0, "sdm.stdp_window must be positive.")

        cls._require(config.predictive.num_levels >= 2, "predictive.num_levels must be >= 2.")
        cls._require(config.predictive.gamma >= 0.0, "predictive.gamma must be >= 0.")
        cls._require(config.predictive.eta >= 0.0, "predictive.eta must be >= 0.")
        cls._require(config.predictive.precision_init > 0.0, "predictive.precision_init must be positive.")
        cls._require(config.predictive.error_threshold >= 0.0, "predictive.error_threshold must be >= 0.")

        cls._require(config.gnw.capacity > 0, "gnw.capacity must be positive.")
        cls._require(config.gnw.broadcast_gain > 0.0, "gnw.broadcast_gain must be positive.")
        cls._require(0.0 <= config.gnw.fatigue_rate < 1.0, "gnw.fatigue_rate must be in [0, 1).")
        cls._require(0.0 <= config.gnw.fatigue_threshold <= 1.0, "gnw.fatigue_threshold must be in [0, 1].")
        cls._require(config.gnw.competition_temp > 0.0, "gnw.competition_temp must be positive.")
        if config.workspace is not None:
            cls._require(config.workspace.capacity > 0, "workspace.capacity must be positive.")
            cls._require(config.workspace.broadcast_gain > 0.0, "workspace.broadcast_gain must be positive.")
            cls._require(0.0 <= config.workspace.fatigue_rate < 1.0, "workspace.fatigue_rate must be in [0, 1).")
            cls._require(0.0 <= config.workspace.fatigue_threshold <= 1.0, "workspace.fatigue_threshold must be in [0, 1].")
            cls._require(config.workspace.competition_temp > 0.0, "workspace.competition_temp must be positive.")

        cls._require(config.reward.intrinsic_scale >= 0.0, "reward.intrinsic_scale must be >= 0.")
        cls._require(config.reward.novelty_threshold > 0.0, "reward.novelty_threshold must be positive.")
        cls._require(config.reward.novelty_boost >= 1.0, "reward.novelty_boost must be >= 1.")
        cls._require(0.0 < config.reward.novelty_decay <= 1.0, "reward.novelty_decay must be in (0, 1].")
        cls._require(0.0 <= config.reward.curiosity_weight <= 1.0, "reward.curiosity_weight must be in [0, 1].")

        cls._require(
            config.sdm.data_dim == config.ccc.concept_dim,
            "sdm.data_dim must match ccc.concept_dim for associative fabric compatibility.",
        )

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @classmethod
    def _env_overrides(cls) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        defaults = cls.defaults_dict()

        for env_name, path in cls.ENV_ALIASES.items():
            if env_name in os.environ:
                _assign_path(overrides, path, cls._coerce_value(os.environ[env_name], cls._path_type(defaults, path)))

        for key, value in defaults.items():
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    env_name = f"BIOARN_{key.upper()}_{nested_key.upper()}"
                    if env_name in os.environ:
                        _assign_path(
                            overrides,
                            f"{key}.{nested_key}",
                            cls._coerce_value(os.environ[env_name], type(nested_value)),
                        )
            else:
                env_name = f"BIOARN_{key.upper()}"
                if env_name in os.environ:
                    overrides[key] = cls._coerce_value(os.environ[env_name], type(value))
        return overrides

    @classmethod
    def _cli_overrides(cls, cli_args: Mapping[str, Any] | Any) -> dict[str, Any]:
        values = dict(vars(cli_args)) if hasattr(cli_args, "__dict__") else dict(cli_args)
        overrides: dict[str, Any] = {}
        nested_keys = sorted(_NESTED_CONFIGS.keys(), key=len, reverse=True)
        alias_map = {"ccc_pool_size": "ccc.max_pool_size"}

        for key, value in values.items():
            if value is None or key in {"command", "config", "preset", "func"}:
                continue
            if key in alias_map:
                _assign_path(overrides, alias_map[key], value)
                continue
            matched_path: str | None = None
            for nested_key in nested_keys:
                prefix = f"{nested_key}_"
                if key.startswith(prefix):
                    matched_path = f"{nested_key}.{key[len(prefix):]}"
                    break
            if matched_path is None:
                matched_path = key
            _assign_path(overrides, matched_path, value)
        return overrides

    @staticmethod
    def _coerce_value(value: str, expected_type: type[Any]) -> Any:
        if expected_type is bool:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if expected_type is int:
            return int(value)
        if expected_type is float:
            return float(value)
        return value

    @staticmethod
    def _path_type(defaults: Mapping[str, Any], path: str) -> type[Any]:
        current: Any = defaults
        for part in path.split("."):
            current = current[part]
        return type(current)


__all__ = ["ConfigManager"]
