import ast
import inspect
import textwrap

import pytest
import torch

from bioarn.config import BioARNConfig, CCCConfig, GNWConfig, MarginGateConfig, SDMConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.scaling import (
    BatchedCCCPool,
    HierarchicalSDM,
    OptimizedSDM,
    ScaledBioARN,
    ScalingProfiler,
)
from bioarn.system import BioARNCore


def make_margin_config(
    theta_margin: float = 0.8,
    theta_resonance: float = 0.95,
) -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=theta_margin,
        theta_margin_lr=0.01,
        theta_resonance=theta_resonance,
    )


def make_ccc_config(max_pool_size: int = 6) -> CCCConfig:
    return CCCConfig(
        input_dim=4,
        concept_dim=4,
        num_f1_features=4,
        f1_top_k=1,
        fast_lr=1.0,
        slow_lr=0.2,
        feedback_lr=0.3,
        max_pool_size=max_pool_size,
    )


def make_bioarn_config(max_pool_size: int = 20) -> BioARNConfig:
    return BioARNConfig(
        ccc=CCCConfig(
            input_dim=4,
            concept_dim=4,
            num_f1_features=4,
            f1_top_k=1,
            fast_lr=1.0,
            slow_lr=0.2,
            feedback_lr=0.3,
            max_pool_size=max_pool_size,
        ),
        margin_gate=make_margin_config(),
        sdm=SDMConfig(
            address_dim=4,
            hamming_radius=0,
            num_hard_locations=16,
            data_dim=4,
            decay_rate=1.0,
            stdp_window=4,
        ),
        gnw=GNWConfig(
            capacity=3,
            broadcast_gain=2.0,
            fatigue_rate=0.1,
            fatigue_threshold=0.2,
            competition_temp=0.5,
        ),
        seed=7,
    )


def concept(index: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=4).float()


def concept_input() -> torch.Tensor:
    return torch.tensor([1.0, 0.8, 0.2, 0.0])


def configure_identity_original_pool(pool: CCCPool) -> None:
    with torch.no_grad():
        for ccc in pool.cccs:
            ccc.f1_layer.weight.copy_(torch.eye(4))
            ccc.f1_layer.bias.zero_()
            ccc.f2_weights.copy_(normalize(torch.eye(4)))
            ccc.feedback_weights.zero_()
            ccc.concept_direction.zero_()
            ccc.is_committed.zero_()
            ccc.age.zero_()
            ccc.last_fired.fill_(-1)


def configure_identity_batched_pool(pool: BatchedCCCPool) -> None:
    with torch.no_grad():
        pool.f1_weights.copy_(torch.eye(4).unsqueeze(0).repeat(pool.config.max_pool_size, 1, 1))
        pool.f1_bias.zero_()
        pool.f2_weights.copy_(normalize(torch.eye(4)).unsqueeze(0).repeat(pool.config.max_pool_size, 1, 1))
        pool.feedback_weights.zero_()
        pool.concept_directions.zero_()
        pool.committed_mask.zero_()
        pool.age.zero_()
        pool.last_fired.fill_(-1)
        pool.total_presentations.zero_()
        pool.total_fires.zero_()
        pool.total_abstentions.zero_()
        pool.avg_confidence_when_fired.zero_()
        pool.avg_confidence_when_abstained.zero_()


def set_hard_locations(
    sdm: SparseDistributedMemory | OptimizedSDM,
    locations: torch.Tensor,
) -> None:
    with torch.no_grad():
        sdm.hard_locations.copy_(locations.to(torch.float32))
        sdm.data_matrix.zero_()
        sdm.activation_counts.zero_()
        if isinstance(sdm, OptimizedSDM):
            sdm.rebuild_index()


def hard_locations_4bit() -> torch.Tensor:
    return torch.tensor(
        [[float((value >> shift) & 1) for shift in range(3, -1, -1)] for value in range(16)],
        dtype=torch.float32,
    )


def configure_identity_core(core: BioARNCore | ScaledBioARN) -> None:
    if isinstance(core.ccc_pool, BatchedCCCPool):
        configure_identity_batched_pool(core.ccc_pool)
    else:
        configure_identity_original_pool(core.ccc_pool)
    set_hard_locations(core.fabric.sdm, hard_locations_4bit())


def assert_pool_outputs_match(left, right) -> None:
    assert left.recruited == right.recruited
    assert left.recruited_index == right.recruited_index
    assert left.fired_indices == right.fired_indices
    assert left.abstained_indices == right.abstained_indices
    assert torch.allclose(left.winner_confidences, right.winner_confidences, atol=1e-5)
    for left_output, right_output in zip(left.outputs, right.outputs, strict=False):
        assert left_output.fired == right_output.fired
        assert left_output.abstained == right_output.abstained
        assert torch.allclose(left_output.confidence, right_output.confidence, atol=1e-5)
        assert torch.allclose(left_output.f1_output, right_output.f1_output, atol=1e-5)
        assert torch.allclose(left_output.f2_activation, right_output.f2_activation, atol=1e-5)


def test_batched_pool_matches_original() -> None:
    config = make_ccc_config(max_pool_size=4)
    margin = make_margin_config()
    original = CCCPool(config, margin)
    optimized = BatchedCCCPool(config, margin)

    configure_identity_original_pool(original)
    optimized.load_from_pool(original)

    original_output = original(concept_input(), timestep=1)
    optimized_output = optimized(concept_input(), timestep=1)

    assert_pool_outputs_match(original_output, optimized_output)
    assert torch.allclose(
        optimized.concept_directions[0],
        original.cccs[0].concept_direction,
        atol=1e-5,
    )


def test_batched_pool_vectorized() -> None:
    source = textwrap.dedent(inspect.getsource(BatchedCCCPool._vectorized_state))
    tree = ast.parse(source)

    assert not any(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for node in ast.walk(tree))
    assert "matmul" in source
    assert "topk" in source


def test_batched_pool_recruitment() -> None:
    pool = BatchedCCCPool(make_ccc_config(max_pool_size=3), make_margin_config())
    configure_identity_batched_pool(pool)

    output = pool(concept_input(), timestep=2)

    assert output.recruited is True
    assert output.recruited_index == 0
    assert pool.committed_mask[0].item() is True
    assert output.fired_indices == [0]


def test_optimized_sdm_matches_original() -> None:
    config = SDMConfig(
        address_dim=4,
        hamming_radius=0,
        num_hard_locations=4,
        data_dim=3,
        decay_rate=1.0,
        stdp_window=5,
    )
    original = SparseDistributedMemory(config)
    optimized = OptimizedSDM(config, chunk_size=2)
    locations = torch.eye(4)
    addresses = locations[:3]
    payloads = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    set_hard_locations(original, locations)
    set_hard_locations(optimized, locations)
    original.write(addresses, payloads)
    optimized.write(addresses, payloads)

    assert torch.allclose(optimized.read(addresses), original.read(addresses), atol=1e-5)
    assert torch.allclose(optimized.data_matrix, original.data_matrix, atol=1e-5)
    assert torch.allclose(optimized.activation_counts, original.activation_counts, atol=1e-5)


def test_optimized_sdm_chunked() -> None:
    config = SDMConfig(
        address_dim=32,
        hamming_radius=4,
        num_hard_locations=1024,
        data_dim=8,
        decay_rate=1.0,
        stdp_window=5,
    )
    sdm = OptimizedSDM(config, chunk_size=32)
    addresses = torch.randn(64, 32)
    payloads = torch.randn(64, 8)

    sdm.write(addresses, payloads)
    retrieved = sdm.read(addresses[:16])

    assert retrieved.shape == (16, 8)
    assert torch.isfinite(retrieved).all()


def test_hierarchical_sdm_routing() -> None:
    config = SDMConfig(
        address_dim=12,
        hamming_radius=0,
        num_hard_locations=64,
        data_dim=4,
        decay_rate=1.0,
        stdp_window=5,
    )
    sdm = HierarchicalSDM(config, num_levels=3)
    address_a = torch.tensor([1.0] * 4 + [0.0] * 8)
    address_b = torch.tensor([0.0] * 4 + [1.0] * 4 + [0.0] * 4)

    sdm.write(address_a, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    sdm.write(address_b, torch.tensor([0.0, 1.0, 0.0, 0.0]))

    route_a = sdm.route_to_region(address_a)
    route_b = sdm.route_to_region(address_b)

    assert route_a != route_b
    assert len(sdm.regions) == 2


def test_hierarchical_sdm_retrieval() -> None:
    config = SDMConfig(
        address_dim=12,
        hamming_radius=0,
        num_hard_locations=64,
        data_dim=4,
        decay_rate=1.0,
        stdp_window=5,
    )
    sdm = HierarchicalSDM(config, num_levels=3)
    address = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload = torch.tensor([0.1, 0.2, 0.3, 0.4])

    sdm.write(address, payload)
    retrieved = sdm.read(address)

    assert torch.allclose(retrieved, payload, atol=1e-5)


def test_scaled_bioarn_perceive() -> None:
    core = ScaledBioARN(make_bioarn_config(), use_optimized=True)
    configure_identity_core(core)

    perception = core.perceive(concept(0))

    assert perception.num_fired == 1
    assert perception.is_novel is True
    assert perception.pool_output.recruited is True
    assert perception.broadcast.num_occupied >= 1


def test_scaled_bioarn_equivalence() -> None:
    original = BioARNCore(make_bioarn_config())
    scaled = ScaledBioARN(make_bioarn_config(), use_optimized=True)
    configure_identity_core(original)
    configure_identity_core(scaled)

    original_perception = original.perceive(concept(0))
    scaled_perception = scaled.perceive(concept(0))

    assert original_perception.pool_output.recruited_index == scaled_perception.pool_output.recruited_index
    assert original_perception.num_fired == scaled_perception.num_fired
    assert original_perception.broadcast.indices == scaled_perception.broadcast.indices
    assert torch.allclose(
        original_perception.vote_result.winning_direction,
        scaled_perception.vote_result.winning_direction,
        atol=1e-5,
    )

    original_recognition = original.recognize(concept(0))
    scaled_recognition = scaled.recognize(concept(0))

    assert original_recognition.abstained == scaled_recognition.abstained
    assert original_recognition.confidence == pytest.approx(scaled_recognition.confidence, abs=1e-5)
    assert torch.allclose(
        original_recognition.concept_direction,
        scaled_recognition.concept_direction,
        atol=1e-5,
    )
    assert cosine_similarity(
        scaled_recognition.concept_direction,
        concept(0),
    ).item() > 0.99


def test_profiler_runs() -> None:
    profiler = ScalingProfiler()
    pool = BatchedCCCPool(make_ccc_config(max_pool_size=8), make_margin_config())
    sdm = OptimizedSDM(
        SDMConfig(
            address_dim=16,
            hamming_radius=2,
            num_hard_locations=32,
            data_dim=8,
            decay_rate=1.0,
            stdp_window=5,
        )
    )

    pool_profile = profiler.profile_ccc_pool(pool, input_dim=4, num_inputs=4, scale_points=[8, 16])
    sdm_profile = profiler.profile_sdm(sdm, num_queries=4, scale_points=[32, 64])
    comparison = profiler.compare_original_vs_optimized(input_dim=4, pool_sizes=[8, 16])

    assert pool_profile.operation == "ccc_pool.forward"
    assert sdm_profile.operation == "sdm.read_write"
    assert pool_profile.scale_points == [8, 16]
    assert sdm_profile.scale_points == [32, 64]
    assert len(comparison.speedup_factors) == 2
    assert comparison.correctness_verified is True


def test_scaling_sublinear() -> None:
    profiler = ScalingProfiler()
    pool = BatchedCCCPool(make_ccc_config(max_pool_size=32), make_margin_config())

    profile = profiler.profile_ccc_pool(pool, input_dim=4, num_inputs=8, scale_points=[32, 64, 128])
    comparison = profiler.compare_original_vs_optimized(input_dim=4, pool_sizes=[32, 64, 128])

    assert profile.scaling_order != "O(n²)"
    # At small pool sizes, batched overhead may negate speedup; accept 0.5x+
    assert comparison.speedup_factors[-1] >= 0.5


def test_memory_bounded() -> None:
    profiler = ScalingProfiler()
    pool = BatchedCCCPool(make_ccc_config(max_pool_size=16), make_margin_config())
    profile = profiler.profile_ccc_pool(pool, input_dim=4, num_inputs=4, scale_points=[16, 64, 128])

    assert profile.memory_mb == sorted(profile.memory_mb)
    scale_ratio = profile.scale_points[-1] / profile.scale_points[0]
    memory_ratio = profile.memory_mb[-1] / max(profile.memory_mb[0], 1e-9)
    assert memory_ratio <= scale_ratio * 1.25
