from pathlib import Path

import pytest
import torch

from bioarn.config import BioARNConfig, CCCConfig
from bioarn.hardware import ASICSpec, ComponentMapper, HardwareProfiler, LoihiBackend, PyTorchBackend
from bioarn.hardware.backend import ComparisonReport


def make_backend() -> PyTorchBackend:
    return PyTorchBackend()


def make_small_config() -> BioARNConfig:
    config = BioARNConfig()
    config.ccc = CCCConfig(
        input_dim=8,
        concept_dim=6,
        num_f1_features=4,
        f1_top_k=2,
        max_pool_size=3,
    )
    config.sdm.address_dim = 16
    config.sdm.hamming_radius = 6
    config.sdm.num_hard_locations = 12
    config.sdm.data_dim = 6
    config.predictive.num_levels = 2
    config.gnw.capacity = 4
    return config


def test_pytorch_backend_create_neurons() -> None:
    backend = make_backend()
    handle = backend.create_neuron_group(4, "lif", {"threshold": 0.5, "refractory_steps": 0})

    assert handle.num_neurons == 4
    assert handle.neuron_type == "lif"
    assert handle.backend == "pytorch"


def test_pytorch_backend_create_synapse() -> None:
    backend = make_backend()
    source = backend.create_neuron_group(3, "lif", {"refractory_steps": 0})
    target = backend.create_neuron_group(2, "lif", {"refractory_steps": 0})
    synapse = backend.create_synapse(source, target, torch.ones(2, 3), "hebbian")

    assert synapse.source == source.id
    assert synapse.target == target.id
    assert synapse.num_synapses == 6
    assert synapse.learning_rule == "hebbian"


def test_pytorch_backend_step() -> None:
    backend = make_backend()
    group = backend.create_neuron_group(2, "lif", {"beta": 1.0, "threshold": 0.5, "refractory_steps": 0})
    backend.inject_spikes(group, torch.tensor([[1.0, 0.25]]))
    result = backend.step()

    assert result.timestep == 1
    assert group.id in result.spikes
    assert result.spikes[group.id].shape == (1, 2)
    assert result.spikes[group.id][0, 0].item() == 1.0
    assert group.id in result.potentials


def test_pytorch_backend_inject_read() -> None:
    backend = make_backend()
    source = backend.create_neuron_group(2, "lif", {"beta": 1.0, "threshold": 0.5, "refractory_steps": 0})
    target = backend.create_neuron_group(2, "lif", {"beta": 1.0, "threshold": 0.5, "refractory_steps": 0})
    backend.create_synapse(source, target, torch.eye(2), "fixed")

    backend.inject_spikes(source, torch.tensor([[1.0, 1.0]]))
    backend.step()
    backend.step()
    spikes = backend.read_spikes(target)

    assert spikes.shape == (1, 2)
    assert torch.equal(spikes, torch.tensor([[1.0, 1.0]]))


def test_pytorch_backend_energy() -> None:
    backend = make_backend()
    group = backend.create_neuron_group(4, "lif", {"beta": 1.0, "threshold": 0.5, "refractory_steps": 0})
    backend.inject_spikes(group, torch.ones(1, 4))
    backend.step()
    estimate = backend.get_energy_estimate()

    assert estimate.total_joules > 0.0
    assert estimate.watts_at_freq > 0.0
    assert 0.1 <= estimate.comparison_gpu <= 100.0


def test_loihi_backend_stubs() -> None:
    backend = LoihiBackend()
    group = backend.get_hardware_info()

    with pytest.raises(NotImplementedError, match="LoihiBackend is a documented stub"):
        backend.create_neuron_group(4, "lif", {})
    with pytest.raises(NotImplementedError, match="compartments"):
        backend.step()
    with pytest.raises(NotImplementedError, match="synaptic memory"):
        backend.update_weights(
            synapse=type("StubSynapse", (), {"id": "s"})(),  # type: ignore[arg-type]
            new_weights=torch.ones(1, 1),
        )

    assert group.backend_name == "Intel Loihi 2"


def test_loihi_hardware_info() -> None:
    info = LoihiBackend().get_hardware_info()

    assert info.backend_name == "Intel Loihi 2"
    assert info.native_spike is True
    assert info.supports_stdp is True
    assert info.estimated_power_per_spike == pytest.approx(23e-12)


def test_component_mapper_ccc() -> None:
    mapper = ComponentMapper(make_backend())
    component = mapper.map_ccc(make_small_config().ccc)

    assert component.name == "ccc"
    assert component.neuron_count > 0
    assert component.synapse_count > component.neuron_count
    assert "hebbian" in component.learning_rules


def test_component_mapper_full_system() -> None:
    config = make_small_config()
    mapper = ComponentMapper(make_backend())
    system = mapper.map_full_system(config)

    assert system.total_neurons == sum(component.neuron_count for component in system.components)
    assert system.total_synapses == sum(component.synapse_count for component in system.components)
    assert system.total_memory_bytes == sum(component.memory_bytes for component in system.components)
    assert system.estimated_power_watts > 0.0
    assert system.estimated_die_area_mm2 > 0.0


def test_profiler_power_estimate() -> None:
    backend = make_backend()
    profiler = HardwareProfiler(backend)
    mapping = ComponentMapper(backend).map_full_system(make_small_config())
    estimate = profiler.estimate_power(mapping)

    assert estimate.inference_watts > 0.0
    assert estimate.training_watts >= estimate.inference_watts
    assert estimate.idle_watts > 0.0
    assert "Loihi 2" in estimate.comparison


def test_profiler_compare_backends() -> None:
    report = HardwareProfiler(make_backend()).compare_backends(make_small_config())

    assert isinstance(report, ComparisonReport)
    assert "PyTorch CPU" in report.power_watts
    assert "Loihi 2" in report.latency_ms
    assert "Ideal ASIC" in report.area_mm2
    assert report.summary


def test_asic_spec_generation() -> None:
    spec = ASICSpec(make_small_config())
    document = spec.generate_spec()

    assert "Bio-ARN 2.0 Neuromorphic ASIC Research Specification" in document
    assert "Total neurons" in document
    assert len(document) > 200

    output_path = Path(__file__).with_name("_asic_spec_test.txt")
    try:
        spec.save_spec(str(output_path))
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8").startswith("Bio-ARN 2.0")
    finally:
        if output_path.exists():
            output_path.unlink()
