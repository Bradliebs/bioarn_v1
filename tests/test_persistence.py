"""Tests for Bio-ARN model persistence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

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
from bioarn.core.ccc import ConceptCellCluster
from bioarn.persistence import ModelExporter, ModelMigrator, ModelQuantizer, ModelStore
from bioarn.system import BioARNCore
from bioarn.utils import CheckpointManager


def make_small_config() -> BioARNConfig:
    return BioARNConfig(
        spiking=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        margin_gate=MarginGateConfig(theta_margin=0.1, theta_margin_lr=0.01, theta_resonance=0.4),
        ccc=CCCConfig(
            input_dim=8,
            concept_dim=8,
            num_f1_features=8,
            f1_top_k=3,
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=4,
        ),
        sdm=SDMConfig(
            address_dim=8,
            hamming_radius=2,
            num_hard_locations=16,
            data_dim=8,
            decay_rate=0.99,
            stdp_window=4,
        ),
        predictive=PredictiveConfig(num_levels=2, gamma=0.2, eta=0.05, precision_init=1.0, error_threshold=0.0),
        gnw=GNWConfig(capacity=2, broadcast_gain=2.0, fatigue_rate=0.05, fatigue_threshold=0.1, competition_temp=0.5),
        reward=RewardConfig(intrinsic_scale=1.0, novelty_threshold=1.5, novelty_boost=2.0, novelty_decay=0.8, curiosity_weight=0.5),
        seed=7,
    )


def make_patterns() -> list[torch.Tensor]:
    return [
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.1], dtype=torch.float32),
    ]


def make_trained_core() -> BioARNCore:
    config = make_small_config()
    core = BioARNCore(config)
    for pattern in make_patterns() * 2:
        core.forward(pattern, learn=True)
    return core


def make_committed_ccc() -> tuple[ConceptCellCluster, torch.Tensor]:
    config = make_small_config()
    ccc = ConceptCellCluster(config.ccc, config.margin_gate)
    prototype = make_patterns()[0]
    with torch.no_grad():
        f1_output = ccc.f1_encode(prototype)
        ccc.learn_fast(prototype, f1_output)
    return ccc, prototype


def test_model_store_save_load(tmp_path: Path) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()
    info = store.save_model(core, "demo-core", "1.0.0", {"accuracy": 1.0}, core.config)

    loaded, loaded_info = store.load_model("demo-core", "1.0.0")

    assert isinstance(loaded, BioARNCore)
    assert Path(info.checkpoint_path).exists()
    assert loaded_info.metrics["accuracy"] == pytest.approx(1.0)
    for key, value in core.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key])


def test_model_store_list(tmp_path: Path) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()

    store.save_model(core, "alpha", "1.0.0", {"accuracy": 0.8}, core.config)
    store.save_model(core, "beta", "1.0.0", {"accuracy": 0.9}, core.config)

    listed = store.list_models()

    assert [(item.name, item.version) for item in listed] == [("alpha", "1.0.0"), ("beta", "1.0.0")]


def test_model_store_versioning(tmp_path: Path) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()

    store.save_model(core, "demo-core", "1.0.0", {"accuracy": 0.75}, core.config)
    store.save_model(core, "demo-core", "1.1.0", {"accuracy": 0.85}, core.config)

    versions = [item.version for item in store.list_models() if item.name == "demo-core"]
    assert versions == ["1.0.0", "1.1.0"]


def test_model_store_latest(tmp_path: Path) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()

    store.save_model(core, "demo-core", "1.9.0", {"accuracy": 0.75}, core.config)
    store.save_model(core, "demo-core", "1.10.0", {"accuracy": 0.85}, core.config)

    _, info = store.load_model("demo-core", "latest")
    assert info.version == "1.10.0"


def test_quantize_weights_8bit() -> None:
    quantizer = ModelQuantizer()
    ccc, _ = make_committed_ccc()

    quantized = quantizer.quantize_weights(ccc, bits=8)

    assert quantized.bits == 8
    assert quantized.quantized_keys
    assert all(quantized.quantized_state[key].dtype == torch.int8 for key in quantized.quantized_keys)


def test_quantize_preserves_accuracy() -> None:
    quantizer = ModelQuantizer()
    ccc, prototype = make_committed_ccc()
    quantized = quantizer.quantize_weights(ccc, bits=8)
    samples = [
        prototype,
        prototype * 0.95,
        torch.tensor([-1.0, -1.0, 0.0, 0.0, -0.2, -0.1, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.1], dtype=torch.float32),
    ]
    test_data = [(sample, ccc(sample).fired) for sample in samples]

    report = quantizer.validate_quantization(ccc, quantized, test_data)

    assert report.accuracy_drop < 0.05
    assert report.quantized_accuracy >= 0.95


def test_quantize_margin_gate() -> None:
    quantizer = ModelQuantizer()
    ccc, prototype = make_committed_ccc()
    quantized = quantizer.quantize_weights(ccc, bits=8)
    restored = copy.deepcopy(ccc)
    restored.load_state_dict(quantized.dequantized_state_dict(), strict=False)
    samples = [
        prototype,
        prototype * 0.9,
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.1], dtype=torch.float32),
    ]

    for sample in samples:
        original_output = ccc(sample)
        restored_output = restored(sample)
        assert original_output.fired == restored_output.fired
        assert original_output.abstained == restored_output.abstained


def test_loihi_export_creates_files(tmp_path: Path) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()
    store.save_model(core, "demo-core", "1.0.0", {"accuracy": 0.9}, core.config)

    export = store.export_for_loihi("demo-core", "1.0.0", str(tmp_path / "loihi-export"))
    created_files = {Path(path).name for path in export.files_created}

    assert export.num_components == 4
    assert created_files >= {
        "quantized_weights.pt",
        "activation_ranges.json",
        "lava_config.json",
        "weights_float.npz",
        "component_manifest.json",
    }


def test_migration_compatibility_check(tmp_path: Path) -> None:
    core = make_trained_core()
    checkpoint_path = tmp_path / "legacy.pt"
    manager = CheckpointManager()
    manager.save(core, checkpoint_path, metadata={"bioarn_version": "0.0.1"})

    payload = manager.read_checkpoint(checkpoint_path, resolve=False)
    removed_key = next(iter(payload["state_dict"]))
    del payload["state_dict"][removed_key]
    payload["format_version"] = 0
    torch.save(payload, checkpoint_path)

    result = ModelMigrator().check_compatibility(checkpoint_path)

    assert result.compatible is False
    assert removed_key in result.missing_fields
    assert any(issue.startswith("format_version_mismatch") for issue in result.config_issues)


def test_export_numpy(tmp_path: Path) -> None:
    exporter = ModelExporter()
    core = make_trained_core()
    export_path = tmp_path / "weights.npz"

    exporter.to_numpy(core, export_path)
    arrays = np.load(export_path)

    for name, tensor in core.state_dict().items():
        assert np.allclose(arrays[name], tensor.detach().cpu().numpy())


def test_export_json(tmp_path: Path) -> None:
    exporter = ModelExporter()
    core = make_trained_core()
    export_path = tmp_path / "weights.json"

    exporter.to_json(core, export_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))

    assert "state_dict" in payload
    assert payload["state_dict"]


def test_atomic_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ModelStore(str(tmp_path / "models"))
    core = make_trained_core()
    store.save_model(core, "demo-core", "1.0.0", {"accuracy": 0.25}, core.config)

    def fail_swap(staging_dir: Path, final_dir: Path) -> None:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(store, "_swap_version_directory", fail_swap)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        store.save_model(core, "demo-core", "1.0.0", {"accuracy": 0.95}, core.config)

    _, info = store.load_model("demo-core", "1.0.0")
    assert info.metrics["accuracy"] == pytest.approx(0.25)
