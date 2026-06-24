"""Tests for the Bio-ARN Lava bridge and deployment pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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
from bioarn.core.ccc import CCCPool, ConceptCellCluster
from bioarn.core.spiking import LIFNeuron
from bioarn.hardware.deployment import LoihiDeploymentPipeline
from bioarn.hardware.lava_bridge import (
    DeploymentPackage,
    HardwareRequirements,
    LavaBridge,
)
from bioarn.hardware.lava_processes import DenseLavaProcess, MarginGateLavaProcess, MockLIFProcess
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.persistence import ModelStore
from bioarn.system import BioARNCore


def make_small_config() -> BioARNConfig:
    return BioARNConfig(
        spiking=SpikingConfig(beta=0.8, threshold=0.5, reset=0.0, dt=1.0, refractory_steps=0),
        margin_gate=MarginGateConfig(theta_margin=0.1, theta_margin_lr=0.01, theta_resonance=0.4),
        ccc=CCCConfig(
            input_dim=8,
            concept_dim=8,
            num_f1_features=8,
            f1_top_k=3,
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=3,
        ),
        sdm=SDMConfig(
            address_dim=8,
            hamming_radius=2,
            num_hard_locations=16,
            data_dim=8,
            decay_rate=0.99,
            stdp_window=4,
        ),
        predictive=PredictiveConfig(
            num_levels=2,
            gamma=0.2,
            eta=0.05,
            precision_init=1.0,
            error_threshold=0.0,
        ),
        gnw=GNWConfig(
            capacity=2,
            broadcast_gain=2.0,
            fatigue_rate=0.05,
            fatigue_threshold=0.1,
            competition_temp=0.5,
        ),
        reward=RewardConfig(
            intrinsic_scale=1.0,
            novelty_threshold=1.5,
            novelty_boost=2.0,
            novelty_decay=0.8,
            curiosity_weight=0.5,
        ),
        seed=7,
    )


def make_patterns() -> list[torch.Tensor]:
    return [
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.1], dtype=torch.float32),
    ]


def make_committed_ccc() -> tuple[ConceptCellCluster, torch.Tensor]:
    config = make_small_config()
    ccc = ConceptCellCluster(config.ccc, config.margin_gate)
    prototype = make_patterns()[0]
    with torch.no_grad():
        f1_output = ccc.f1_encode(prototype)
        ccc.learn_fast(prototype, f1_output)
    return ccc, prototype


def make_ccc_pool() -> tuple[CCCPool, torch.Tensor]:
    config = make_small_config()
    pool = CCCPool(config.ccc, config.margin_gate)
    prototype = make_patterns()[0]
    with torch.no_grad():
        f1_output = pool.cccs[0].f1_encode(prototype)
        pool.cccs[0].learn_fast(prototype, f1_output)
    return pool, prototype


def make_sdm() -> tuple[SparseDistributedMemory, torch.Tensor, torch.Tensor]:
    sdm = SparseDistributedMemory(make_small_config().sdm)
    address = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    data = torch.tensor([0.5, 0.25, 0.0, 0.0, 0.25, 0.5, 0.0, 0.0], dtype=torch.float32)
    sdm.write(address, data)
    return sdm, address, data


def make_trained_core() -> BioARNCore:
    core = BioARNCore(make_small_config())
    for pattern in make_patterns() * 2:
        core.forward(pattern, learn=True)
    return core


def save_core(store_dir: Path) -> ModelStore:
    store = ModelStore(str(store_dir))
    core = make_trained_core()
    store.save_model(core, "demo-core", "1.0.0", {"accuracy": 1.0}, core.config)
    return store


def test_lava_bridge_init() -> None:
    bridge = LavaBridge()
    assert isinstance(bridge.lava_available, bool)


def test_convert_ccc_to_lava() -> None:
    bridge = LavaBridge()
    pool, _ = make_ccc_pool()
    graph = bridge.convert_ccc_to_lava(pool)

    assert graph.component_type == "ccc_pool"
    assert len(graph.processes) == pool.config.max_pool_size
    assert graph.num_neurons > 0
    assert graph.connections


def test_convert_sdm_to_lava() -> None:
    bridge = LavaBridge()
    sdm, _, _ = make_sdm()
    graph = bridge.convert_sdm_to_lava(sdm)

    assert graph.component_type == "sdm"
    assert graph.num_synapses == (sdm.num_hard_locations * sdm.address_dim) + (sdm.num_hard_locations * sdm.data_dim)
    assert graph.connections


def test_convert_full_model() -> None:
    bridge = LavaBridge()
    core = make_trained_core()
    graph = bridge.convert_full_model(core)

    assert graph.component_type == "full_system"
    assert graph.num_cores_estimate >= 1
    assert any(connection["signal"] == "workspace_candidates" for connection in graph.connections)


def test_mock_lif_matches_pytorch() -> None:
    config = SpikingConfig(beta=0.8, threshold=0.5, reset=0.0, dt=1.0, refractory_steps=0)
    currents = torch.tensor(
        [
            [[0.2, 0.6]],
            [[0.2, 0.1]],
            [[0.3, 0.1]],
            [[0.1, 0.7]],
        ],
        dtype=torch.float32,
    )
    reference = LIFNeuron(num_neurons=2, config=config)
    reference.reset_state(batch_size=1, dtype=torch.float32)
    ref_spikes, ref_voltages = reference(currents)

    candidate = MockLIFProcess(num_neurons=2, config=config, name="lif")
    candidate.reset_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    lava_spikes, lava_voltages = candidate.forward(currents)

    assert torch.equal(ref_spikes, lava_spikes)
    assert torch.allclose(ref_voltages, lava_voltages)


def test_mock_dense_matches_pytorch() -> None:
    weights = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    inputs = torch.tensor([[0.2, 0.8]], dtype=torch.float32)

    dense = DenseLavaProcess(weights, bias, name="dense")
    expected = F.linear(inputs, weights, bias)

    assert torch.allclose(dense.step(inputs), expected)


def test_margin_gate_lava_process() -> None:
    gate = MarginGateLavaProcess(
        MarginGateConfig(theta_margin=0.6, theta_margin_lr=0.01, theta_resonance=0.8),
        concept_direction=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    fires = gate.step(torch.tensor([[0.9, 0.0]], dtype=torch.float32))
    abstains = gate.step(torch.tensor([[0.1, 0.9]], dtype=torch.float32))

    assert bool(fires.fired.item()) is True
    assert bool(abstains.abstained.item()) is True


def test_run_lava_inference() -> None:
    bridge = LavaBridge()
    ccc, prototype = make_committed_ccc()
    graph = bridge.convert_ccc_to_lava(ccc)
    result = bridge.run_lava_inference(graph, prototype, num_timesteps=4)

    assert "outputs" in result
    assert result["spike_counts"]
    assert result["output"].shape[-1] == ccc.config.concept_dim


def test_equivalence_validation() -> None:
    bridge = LavaBridge()
    ccc, prototype = make_committed_ccc()
    graph = bridge.convert_ccc_to_lava(ccc)
    report = bridge.validate_equivalence(
        ccc,
        graph,
        [prototype, prototype * 0.95],
        tolerance=0.05,
    )

    assert report.match_rate >= 0.5
    assert report.max_deviation <= 0.05
    assert report.passed is True


def test_deployment_pipeline(tmp_path: Path) -> None:
    store = save_core(tmp_path / "models")
    pipeline = LoihiDeploymentPipeline(store)
    package = pipeline.prepare_for_deployment("demo-core", "1.0.0")

    assert isinstance(package, DeploymentPackage)
    assert package.quantization_bits == 8
    assert package.equivalence.passed is True
    assert "process_graph" in package.config


def test_hardware_requirements(tmp_path: Path) -> None:
    store = save_core(tmp_path / "models")
    pipeline = LoihiDeploymentPipeline(store)
    package = pipeline.prepare_for_deployment("demo-core", "1.0.0")
    requirements = pipeline.estimate_hardware_requirements(package)

    assert isinstance(requirements, HardwareRequirements)
    assert requirements.num_cores >= 1
    assert requirements.total_neurons > 0
    assert requirements.total_synapses > 0
    assert requirements.estimated_power_mw > 0.0


def test_deployment_config_generated(tmp_path: Path) -> None:
    store = save_core(tmp_path / "models")
    pipeline = LoihiDeploymentPipeline(store)
    package = pipeline.prepare_for_deployment("demo-core", "1.0.0")
    config = pipeline.generate_deployment_config(package)

    json.loads(json.dumps(config))
    assert "model" in config
    assert "hardware" in config
    assert "process_graph" in config
