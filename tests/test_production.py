"""Tests for Bio-ARN production infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bioarn.cli import main
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
from bioarn.loop import SensorimotorLoop
from bioarn.training import OnlineTrainer
from bioarn.utils import BioARNLogger, CheckpointManager, ConfigManager, ReproducibilityManager


def make_small_config() -> BioARNConfig:
    return BioARNConfig(
        spiking=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        ccc=CCCConfig(
            input_dim=16,
            concept_dim=16,
            num_f1_features=16,
            f1_top_k=4,
            fast_lr=1.0,
            slow_lr=0.1,
            feedback_lr=0.1,
            max_pool_size=8,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.0,
            theta_margin_lr=0.01,
            theta_resonance=0.5,
        ),
        sdm=SDMConfig(
            address_dim=16,
            hamming_radius=2,
            num_hard_locations=32,
            data_dim=16,
            decay_rate=0.99,
            stdp_window=4,
        ),
        predictive=PredictiveConfig(
            num_levels=4,
            gamma=0.2,
            eta=0.05,
            precision_init=1.0,
            error_threshold=0.0,
        ),
        gnw=GNWConfig(
            capacity=3,
            broadcast_gain=2.0,
            fatigue_rate=0.05,
            fatigue_threshold=0.1,
            competition_temp=0.5,
        ),
        reward=RewardConfig(
            intrinsic_scale=1.0,
            novelty_threshold=1.5,
            novelty_boost=2.5,
            novelty_decay=0.8,
            curiosity_weight=0.5,
        ),
        seed=5,
    )


def make_loop() -> SensorimotorLoop:
    return SensorimotorLoop(make_small_config())


def make_data() -> list[tuple[torch.Tensor, int]]:
    return [
        (torch.zeros(16), 0),
        (torch.ones(16), 1),
        (torch.tensor([1.0, 0.0] * 8), 0),
        (torch.tensor([0.0, 1.0] * 8), 1),
    ]


def test_logger_creates_file(tmp_path: Path) -> None:
    logger = BioARNLogger(component="test", log_dir=tmp_path)
    logger.log_event("test", "startup", {"ok": True})
    logger.flush()

    files = list(tmp_path.glob("*.jsonl"))
    assert files
    assert "startup" in files[0].read_text(encoding="utf-8")


def test_logger_metrics(tmp_path: Path) -> None:
    logger = BioARNLogger(component="metrics", log_dir=tmp_path)
    logger.log_metric("trainer", "loss", 0.25, 7)
    logger.flush()

    payload = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["level"] == "METRIC"
    assert payload["data"]["metric_name"] == "loss"
    assert payload["data"]["step"] == 7


def test_checkpoint_save_load(tmp_path: Path) -> None:
    loop = make_loop()
    loop.step(language_input=torch.tensor([1, 2, 3], dtype=torch.long))

    manager = CheckpointManager()
    checkpoint_path = tmp_path / "loop.pt"
    manager.save(loop, checkpoint_path)
    loaded = manager.load(checkpoint_path)

    assert isinstance(loaded, SensorimotorLoop)
    assert loaded.timestep == loop.timestep
    assert loaded.core.get_system_stats()["concepts_learned"] == loop.core.get_system_stats()["concepts_learned"]
    assert torch.allclose(loaded._feedback_features, loop._feedback_features)


def test_checkpoint_metadata(tmp_path: Path) -> None:
    loop = make_loop()
    manager = CheckpointManager()
    checkpoint_path = tmp_path / "meta.pt"
    manager.save(loop, checkpoint_path, metadata={"training_step": 11, "metrics": {"accuracy": 0.5}})

    payload = manager.read_checkpoint(checkpoint_path)
    assert payload["metadata"]["training_step"] == 11
    assert payload["metadata"]["metrics"]["accuracy"] == 0.5


def test_config_from_dict() -> None:
    config = ConfigManager.from_dict(
        {
            "device": "cpu",
            "ccc": {"input_dim": 32, "concept_dim": 32, "max_pool_size": 16},
            "sdm": {"data_dim": 32},
        }
    )

    assert config.device == "cpu"
    assert config.ccc.input_dim == 32
    assert config.ccc.concept_dim == 32
    assert config.sdm.data_dim == 32


def test_config_presets() -> None:
    mnist = ConfigManager.load_preset("mnist")
    cifar = ConfigManager.load_preset("cifar")
    language_small = ConfigManager.load_preset("language_small")
    language_large = ConfigManager.load_preset("language_large")
    production = ConfigManager.load_preset("production")

    assert mnist.ccc.input_dim == 784
    assert cifar.ccc.input_dim == 3072
    assert language_small.ccc.input_dim == 256
    assert language_large.ccc.input_dim == 512
    assert production.ccc.max_pool_size >= 1000


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ConfigManager.from_dict({"ccc": {"input_dim": -1}})

    with pytest.raises(ValueError):
        ConfigManager.from_dict({"ccc": {"concept_dim": 32}, "sdm": {"data_dim": 16}})


def test_reproducibility_seed() -> None:
    ReproducibilityManager.set_seed(123)
    first = torch.rand(4)
    ReproducibilityManager.set_seed(123)
    second = torch.rand(4)
    assert torch.allclose(first, second)


def test_trainer_runs(tmp_path: Path) -> None:
    loop = make_loop()
    trainer = OnlineTrainer(log_every=1, checkpoint_every=10, output_dir=tmp_path)
    result = trainer.train(loop, make_data(), make_small_config())

    assert result.total_steps == len(make_data())
    assert result.metrics["concepts_learned"] >= 1


def test_trainer_checkpointing(tmp_path: Path) -> None:
    loop = make_loop()
    trainer = OnlineTrainer(log_every=1, checkpoint_every=2, output_dir=tmp_path, keep_last=2)
    trainer.train(loop, make_data() * 2, make_small_config())

    checkpoints = list(tmp_path.glob("checkpoint-step-*.pt"))
    assert checkpoints
    assert (tmp_path / "latest.pt").exists()


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_system_info() -> None:
    info = ReproducibilityManager.get_system_info()
    assert info["python_version"]
    assert info["torch_version"]
    assert info["os"]
