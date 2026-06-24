import pytest
import torch

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, SDMConfig, SpikingConfig
from bioarn.hardware import EnergyModel, FunctionalEquivalenceValidator, LoihiMapping


def make_loihi_config() -> BioARNConfig:
    config = BioARNConfig()
    config.spiking = SpikingConfig(beta=0.875, threshold=1.0, reset=0.0, dt=1.0, refractory_steps=2)
    config.ccc = CCCConfig(
        input_dim=32,
        concept_dim=96,
        num_f1_features=128,
        f1_top_k=16,
        max_pool_size=6,
    )
    config.sdm = SDMConfig(
        address_dim=256,
        hamming_radius=48,
        num_hard_locations=64,
        data_dim=96,
        decay_rate=0.999,
        stdp_window=20,
    )
    config.predictive.num_levels = 3
    config.gnw = GNWConfig(capacity=4, broadcast_gain=2.0, fatigue_rate=0.1, fatigue_threshold=0.3)
    config.seed = 7
    return config


def test_lif_neuron_mapping() -> None:
    mapping = LoihiMapping(make_loihi_config()).map_lif_neuron()

    assert 0 <= mapping.decay_constant <= LoihiMapping.DECAY_REGISTER_MAX
    assert mapping.threshold > 0
    assert mapping.refractory_period == 2
    assert mapping.compartment_config["v_decay"] == mapping.decay_constant
    assert mapping.compartment_config["vth"] == mapping.threshold


def test_ccc_core_allocation() -> None:
    mapping = LoihiMapping(make_loihi_config()).map_ccc_to_cores()

    assert mapping.cores_required == 2
    assert max(mapping.neurons_per_core.values()) <= LoihiMapping.COMPARTMENTS_PER_CORE
    assert sum(mapping.neurons_per_core.values()) == 224
    assert mapping.routing_complexity == "inter-core"
    assert "theta_margin" in mapping.custom_microcode


def test_sdm_memory_mapping() -> None:
    loihi = LoihiMapping(make_loihi_config())
    mapping = loihi.map_sdm_to_memory()

    assert mapping.storage_synapses == 64 * 96
    assert mapping.retrieval_latency_steps >= 3
    assert "Hamming" in mapping.addressing_scheme
    assert (
        mapping.storage_synapses * loihi.BYTES_PER_SYNAPSE
        < mapping.cores_required * loihi.SYNAPSE_SRAM_PER_CORE_BYTES
    )


def test_pe_pipeline_mapping() -> None:
    mapping = LoihiMapping(make_loihi_config()).map_pe_to_pipeline()

    assert mapping.cores_required > 0
    assert mapping.pipeline_stages["prediction"] > 0
    assert mapping.pipeline_stages["error"] >= mapping.pipeline_stages["prediction"]
    assert mapping.latency_steps >= 2
    assert "xtrace" in str(mapping.learning_engine["rule"])


def test_gnw_circuit_mapping() -> None:
    mapping = LoihiMapping(make_loihi_config()).map_gnw_to_circuit()

    assert mapping.cores_required == 1
    assert mapping.competition_neurons >= make_loihi_config().gnw.capacity
    assert mapping.broadcast_fanout > 0
    assert mapping.routing_mode == "multicast"
    assert mapping.adaptation_registers == make_loihi_config().gnw.capacity


def test_full_system_mapping() -> None:
    mapping = LoihiMapping(make_loihi_config()).map_full_system()

    assert mapping.total_cores > 0
    assert mapping.total_chips == 1
    assert mapping.total_cores <= mapping.total_chips * LoihiMapping.CORES_PER_CHIP
    assert mapping.total_neurons > 0
    assert mapping.total_synapses > 0
    assert mapping.memory_bytes > 0
    assert mapping.estimated_power_watts > 0.0
    assert mapping.feasibility == "feasible"


def test_lif_functional_equivalence() -> None:
    validator = FunctionalEquivalenceValidator(make_loihi_config())
    result = validator.validate_lif_equivalence(num_steps=64)

    assert result.equivalent
    assert result.mean_deviation <= 0.05
    assert result.max_deviation <= 1.0


def test_ccc_quantization_tolerance() -> None:
    config = make_loihi_config()
    validator = FunctionalEquivalenceValidator(config)
    test_input = torch.linspace(0.05, 0.95, steps=config.ccc.input_dim, dtype=torch.float32)
    result = validator.validate_ccc_equivalence(test_input)

    assert result.equivalent
    assert result.max_deviation <= 0.15


def test_energy_model_loihi() -> None:
    model = EnergyModel()
    breakdown = model.estimate_inference_energy(make_loihi_config(), "loihi2", num_cccs_active=3)

    assert breakdown.total_joules > 0.0
    assert breakdown.watts_at_1khz < 1.0
    assert breakdown.component_breakdown["ccc"] > 0.0
    assert breakdown.component_breakdown["static"] > 0.0


def test_energy_model_comparison() -> None:
    report = EnergyModel().compare_all_backends(make_loihi_config())

    assert set(report.backends) == {"loihi2", "gpu_a100", "cpu_laptop", "ideal_asic"}
    assert report.best_backend in report.backends
    assert all(result.total_joules > 0.0 for result in report.backends.values())
    assert all(ratio >= 1.0 for ratio in report.efficiency_ratios.values())


def test_energy_vs_gpu() -> None:
    model = EnergyModel()
    config = make_loihi_config()
    loihi = model.estimate_inference_energy(config, "loihi2", num_cccs_active=3)
    gpu = model.estimate_inference_energy(config, "gpu_a100", num_cccs_active=3)

    assert loihi.total_joules < gpu.total_joules
    assert loihi.total_joules < (gpu.total_joules / 100.0)


def test_brain_comparison() -> None:
    report = EnergyModel().brain_comparison(make_loihi_config())

    assert report.total_neurons > 0
    assert report.scaled_brain_power_watts > 0.0
    assert report.power_ratio_vs_brain > 1.0
    assert report.energy_per_neuron_ratio > 1.0
    assert report.approaching_biology is False

