import time

import torch

from bioarn.config import CCCConfig, MarginGateConfig, SDMConfig
from bioarn.core.ccc import CCCPool
from bioarn.core.math_utils import normalize
from bioarn.memory.sdm import SparseDistributedMemory
from bioarn.scaling import (
    AdaptiveCapacity,
    BatchedCCCPool,
    MemoryEfficientSDM,
    PoolSharding,
    estimate_module_memory_mb,
)


def make_large_config(max_pool_size: int, dim: int = 32) -> CCCConfig:
    return CCCConfig(
        input_dim=dim,
        concept_dim=dim,
        num_f1_features=dim,
        f1_top_k=max(4, dim // 4),
        fast_lr=1.0,
        slow_lr=0.01,
        feedback_lr=0.01,
        max_pool_size=max_pool_size,
    )


def make_margin_config(theta_margin: float = 0.6, theta_resonance: float = 1.1) -> MarginGateConfig:
    return MarginGateConfig(
        theta_margin=theta_margin,
        theta_margin_lr=0.0,
        theta_resonance=theta_resonance,
    )


def prime_random_pool(pool: BatchedCCCPool, *, seed: int = 0, theta_margin: float = 0.6) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        pool.committed_mask.fill_(True)
        pool.concept_directions.copy_(
            normalize(torch.randn_like(pool.concept_directions, dtype=torch.float32))
        )
        pool.feedback_weights.copy_(torch.randn_like(pool.feedback_weights, dtype=torch.float32) * 0.05)
        pool.theta_margin.fill_(theta_margin)
        pool.theta_resonance.fill_(1.1)
        pool.total_presentations.zero_()
        pool.total_fires.zero_()
        pool.total_abstentions.zero_()
        pool.avg_confidence_when_fired.zero_()
        pool.avg_confidence_when_abstained.zero_()


def measure_fast_infer_ms(pool: BatchedCCCPool, inputs: torch.Tensor) -> float:
    pool.fast_infer(inputs[0], timestep=0, allow_recruit=False)
    start = time.perf_counter()
    for index, sample in enumerate(inputs):
        pool.fast_infer(sample, timestep=index + 1, allow_recruit=False)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0) / max(inputs.shape[0], 1)


def average_sparsity(pool: BatchedCCCPool, inputs: torch.Tensor) -> float:
    sparsities: list[float] = []
    for index, sample in enumerate(inputs):
        summary = pool.fast_infer(sample, timestep=index, allow_recruit=False)
        sparsities.append(summary.sparsity)
    return sum(sparsities) / max(len(sparsities), 1)


def configure_routed_pool(pool: BatchedCCCPool, shard_size: int) -> None:
    dim = pool.config.input_dim
    num_shards = (pool.config.max_pool_size + shard_size - 1) // shard_size
    assert dim >= num_shards
    identity = torch.eye(dim, dtype=torch.float32)
    with torch.no_grad():
        pool.f1_weights.copy_(identity.unsqueeze(0).repeat(pool.config.max_pool_size, 1, 1))
        pool.f1_bias.zero_()
        pool.f2_weights.copy_(normalize(identity).unsqueeze(0).repeat(pool.config.max_pool_size, 1, 1))
        pool.feedback_weights.copy_(identity.unsqueeze(0).repeat(pool.config.max_pool_size, 1, 1))
        pool.committed_mask.fill_(True)
        pool.theta_margin.fill_(0.95)
        pool.theta_resonance.fill_(1.1)
        for index in range(pool.config.max_pool_size):
            shard_index = min(index // shard_size, dim - 1)
            pool.concept_directions[index].zero_()
            pool.concept_directions[index, shard_index] = 1.0


def make_routed_query(dim: int, shard_index: int) -> torch.Tensor:
    query = torch.zeros(dim, dtype=torch.float32)
    query[shard_index] = 1.0
    return query


def test_5k_pool_initializes() -> None:
    start = time.perf_counter()
    pool = BatchedCCCPool(make_large_config(5000), make_margin_config())
    elapsed = time.perf_counter() - start
    assert pool.config.max_pool_size == 5000
    assert elapsed < 10.0


def test_5k_inference_latency() -> None:
    pool = BatchedCCCPool(make_large_config(5000), make_margin_config(theta_margin=0.65))
    prime_random_pool(pool, seed=5, theta_margin=0.65)
    inputs = torch.randn(16, pool.config.input_dim)
    latency_ms = measure_fast_infer_ms(pool, inputs)
    assert latency_ms < 100.0


def test_10k_pool_initializes() -> None:
    start = time.perf_counter()
    pool = BatchedCCCPool(make_large_config(10000), make_margin_config())
    elapsed = time.perf_counter() - start
    assert pool.config.max_pool_size == 10000
    assert elapsed < 30.0


def test_sparsity_at_scale() -> None:
    pool = BatchedCCCPool(make_large_config(5000), make_margin_config(theta_margin=0.7))
    prime_random_pool(pool, seed=7, theta_margin=0.7)
    inputs = torch.randn(32, pool.config.input_dim)
    sparsity = average_sparsity(pool, inputs)
    assert sparsity < 0.05


def test_memory_linear_scaling() -> None:
    small = BatchedCCCPool(make_large_config(2500), make_margin_config())
    large = BatchedCCCPool(make_large_config(5000), make_margin_config())
    ratio = estimate_module_memory_mb(large) / max(estimate_module_memory_mb(small), 1e-9)
    assert 1.6 <= ratio <= 2.4


def test_pool_sharding_correctness() -> None:
    total_size = 5000
    shard_size = 500
    config = make_large_config(total_size, dim=16)
    flat = BatchedCCCPool(config, make_margin_config(theta_margin=0.95))
    configure_routed_pool(flat, shard_size=shard_size)
    sharded = PoolSharding(total_size, shard_size=shard_size, config=config, margin_config=make_margin_config(theta_margin=0.95)).load_from_pool(flat)

    query = make_routed_query(config.input_dim, shard_index=3)
    flat_summary = flat.fast_infer(query, allow_recruit=False)
    sharded_summary = sharded.fast_infer(query, allow_recruit=False)

    assert sharded_summary.fired_indices == flat_summary.fired_indices
    assert torch.allclose(
        sharded_summary.winner_confidences,
        flat_summary.winner_confidences,
        atol=1e-5,
    )


def test_pool_sharding_faster() -> None:
    total_size = 10000
    shard_size = 1000
    config = make_large_config(total_size, dim=16)
    flat = BatchedCCCPool(config, make_margin_config(theta_margin=0.95))
    configure_routed_pool(flat, shard_size=shard_size)
    sharded = PoolSharding(total_size, shard_size=shard_size, config=config, margin_config=make_margin_config(theta_margin=0.95)).load_from_pool(flat)
    queries = torch.stack([make_routed_query(config.input_dim, shard_index=index % 10) for index in range(20)])

    flat.fast_infer(queries[0], allow_recruit=False)
    sharded.fast_infer(queries[0], allow_recruit=False)

    start = time.perf_counter()
    for query in queries:
        flat.fast_infer(query, allow_recruit=False)
    flat_ms = (time.perf_counter() - start) * 1000.0 / len(queries)

    start = time.perf_counter()
    for query in queries:
        sharded.fast_infer(query, allow_recruit=False)
    sharded_ms = (time.perf_counter() - start) * 1000.0 / len(queries)

    assert sharded_ms < flat_ms


def test_adaptive_capacity_grows() -> None:
    adaptive = AdaptiveCapacity(
        initial_size=64,
        max_size=256,
        config=make_large_config(64, dim=16),
        margin_config=make_margin_config(),
        abstention_window=8,
        abstention_threshold=0.5,
    )
    for _ in range(8):
        adaptive.observe_abstention(True)
    assert adaptive.current_size > 64


def test_adaptive_capacity_prunes() -> None:
    adaptive = AdaptiveCapacity(
        initial_size=64,
        max_size=128,
        config=make_large_config(64, dim=16),
        margin_config=make_margin_config(),
    )
    with torch.no_grad():
        adaptive.pool.committed_mask[:6] = True
        adaptive.pool.total_presentations[:3] = 64
        adaptive.pool.total_fires[:3] = 0
        adaptive.pool.total_presentations[3:6] = 64
        adaptive.pool.total_fires[3:6] = 4
    pruned = adaptive.prune_dead_cccs(min_presentations=32, max_fire_count=0)
    assert pruned == [0, 1, 2]
    assert not bool(adaptive.pool.committed_mask[:3].any().item())
    assert bool(adaptive.pool.committed_mask[3:6].all().item())


def test_batched_matches_sequential() -> None:
    config = make_large_config(5000, dim=8)
    margin = make_margin_config(theta_margin=0.5)
    original = CCCPool(config, margin)
    batched = BatchedCCCPool(config, margin).load_from_pool(original)

    torch.manual_seed(13)
    with torch.no_grad():
        for ccc in original.cccs[:24]:
            ccc.is_committed.fill_(True)
            direction = normalize(torch.randn(1, config.concept_dim)).squeeze(0)
            ccc.concept_direction.copy_(direction)
            ccc.feedback_weights.copy_(torch.randn_like(ccc.feedback_weights) * 0.05)
        batched.load_from_pool(original)

    sample = torch.randn(config.input_dim)
    left = original(sample, timestep=1)
    right = batched(sample, timestep=1)

    assert left.recruited == right.recruited
    assert left.recruited_index == right.recruited_index
    assert left.fired_indices == right.fired_indices
    assert left.abstained_indices == right.abstained_indices
    assert torch.allclose(left.winner_confidences, right.winner_confidences, atol=1e-5)


def test_memory_efficient_sdm() -> None:
    config = SDMConfig(
        address_dim=32,
        hamming_radius=4,
        num_hard_locations=4096,
        data_dim=16,
        decay_rate=1.0,
        stdp_window=4,
    )
    dense = SparseDistributedMemory(config)
    efficient = MemoryEfficientSDM(config, chunk_size=128, storage_dtype=torch.float16)
    addresses = torch.randn(24, 32)
    payloads = torch.randn(24, 16)

    dense.write(addresses, payloads)
    efficient.write(addresses, payloads)
    retrieved = efficient.read(addresses[:4])

    assert estimate_module_memory_mb(efficient) < estimate_module_memory_mb(dense)
    assert retrieved.shape == (4, 16)
    assert torch.isfinite(retrieved).all()


def test_scaling_linear() -> None:
    small = BatchedCCCPool(make_large_config(2000), make_margin_config(theta_margin=0.65))
    large = BatchedCCCPool(make_large_config(4000), make_margin_config(theta_margin=0.65))
    prime_random_pool(small, seed=17, theta_margin=0.65)
    prime_random_pool(large, seed=17, theta_margin=0.65)
    inputs = torch.randn(12, small.config.input_dim)

    small_ms = measure_fast_infer_ms(small, inputs)
    large_ms = measure_fast_infer_ms(large, inputs)
    assert large_ms / max(small_ms, 1e-9) < 3.0
