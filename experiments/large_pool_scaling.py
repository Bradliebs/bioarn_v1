"""Scale Bio-ARN to 5K-10K CCCs and validate performance."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from bioarn.config import CCCConfig, MarginGateConfig
from bioarn.core.math_utils import normalize
from bioarn.scaling import BatchedCCCPool, PoolSharding, estimate_module_memory_mb

SCALES = [100, 500, 1000, 2000, 5000, 10000]
NUM_PATTERNS = 1000
BATCH_SIZE = 64


@dataclass
class ScaleMeasurement:
    scale: int
    init_seconds: float
    init_memory_mb: float
    inference_ms: float
    learning_ms: float
    sparsity: float
    post_learning_memory_mb: float


def make_config(scale: int) -> CCCConfig:
    return CCCConfig(
        input_dim=32,
        concept_dim=32,
        num_f1_features=32,
        f1_top_k=8,
        fast_lr=1.0,
        slow_lr=0.01,
        feedback_lr=0.01,
        max_pool_size=scale,
    )


def make_margin(scale: int) -> MarginGateConfig:
    margin = 0.52 + (0.05 * math.log10(max(scale, 10)))
    return MarginGateConfig(
        theta_margin=min(0.72, margin),
        theta_margin_lr=0.0,
        theta_resonance=1.1,
    )


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prime_pool(pool: BatchedCCCPool, *, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        pool.committed_mask.fill_(True)
        pool.concept_directions.copy_(normalize(torch.randn_like(pool.concept_directions)))
        pool.feedback_weights.copy_(torch.randn_like(pool.feedback_weights) * 0.05)
        pool.total_presentations.zero_()
        pool.total_fires.zero_()
        pool.total_abstentions.zero_()
        pool.avg_confidence_when_fired.zero_()
        pool.avg_confidence_when_abstained.zero_()


def memory_mb(pool: BatchedCCCPool) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    return estimate_module_memory_mb(pool)


def average_batch_sparsity(pool: BatchedCCCPool, patterns: torch.Tensor) -> float:
    fired_total = 0.0
    sample_total = 0
    with torch.no_grad():
        for start in range(0, patterns.shape[0], BATCH_SIZE):
            batch = patterns[start : start + BATCH_SIZE]
            state = pool._vectorized_state(batch, timestep=start)
            fired_per_sample = state.fired.to(torch.float32).sum(dim=0)
            fired_total += float(fired_per_sample.sum().item())
            sample_total += int(batch.shape[0])
    return fired_total / max(sample_total * pool.config.max_pool_size, 1)


def timed_batches(pool: BatchedCCCPool, patterns: torch.Tensor) -> float:
    synchronize_if_needed()
    start = time.perf_counter()
    with torch.no_grad():
        for batch_index, offset in enumerate(range(0, patterns.shape[0], BATCH_SIZE)):
            batch = patterns[offset : offset + BATCH_SIZE]
            pool._vectorized_state(batch, timestep=batch_index)
    synchronize_if_needed()
    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0) / max(patterns.shape[0], 1)


def measure_scale(scale: int) -> ScaleMeasurement:
    patterns = torch.randn(NUM_PATTERNS, make_config(scale).input_dim)

    init_start = time.perf_counter()
    inference_pool = BatchedCCCPool(make_config(scale), make_margin(scale))
    init_seconds = time.perf_counter() - init_start
    prime_pool(inference_pool, seed=scale)
    init_memory = memory_mb(inference_pool)

    inference_ms = timed_batches(inference_pool, patterns)
    sparsity = average_batch_sparsity(inference_pool, patterns)

    learning_pool = BatchedCCCPool(make_config(scale), make_margin(scale))
    prime_pool(learning_pool, seed=scale)
    learning_ms = timed_batches(learning_pool, patterns)
    learned_memory = memory_mb(learning_pool)

    return ScaleMeasurement(
        scale=scale,
        init_seconds=init_seconds,
        init_memory_mb=init_memory,
        inference_ms=inference_ms,
        learning_ms=learning_ms,
        sparsity=sparsity,
        post_learning_memory_mb=learned_memory,
    )


def benchmark_sharding(scale: int = 10000) -> tuple[float, float]:
    config = CCCConfig(
        input_dim=16,
        concept_dim=16,
        num_f1_features=16,
        f1_top_k=4,
        fast_lr=1.0,
        slow_lr=0.01,
        feedback_lr=0.01,
        max_pool_size=scale,
    )
    margin = MarginGateConfig(theta_margin=0.95, theta_margin_lr=0.0, theta_resonance=1.1)
    flat = BatchedCCCPool(config, margin)
    with torch.no_grad():
        identity = torch.eye(config.input_dim)
        flat.f1_weights.copy_(identity.unsqueeze(0).repeat(scale, 1, 1))
        flat.f1_bias.zero_()
        flat.f2_weights.copy_(normalize(identity).unsqueeze(0).repeat(scale, 1, 1))
        flat.feedback_weights.copy_(identity.unsqueeze(0).repeat(scale, 1, 1))
        flat.committed_mask.fill_(True)
        flat.theta_margin.fill_(0.95)
        for index in range(scale):
            shard_index = min(index // 1000, config.input_dim - 1)
            flat.concept_directions[index].zero_()
            flat.concept_directions[index, shard_index] = 1.0

    sharded = PoolSharding(scale, shard_size=1000, config=config, margin_config=margin).load_from_pool(flat)
    queries = torch.stack(
        [torch.nn.functional.one_hot(torch.tensor(index % 10), num_classes=config.input_dim).float() for index in range(20)]
    )

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
    return flat_ms, sharded_ms


def render_table(results: list[ScaleMeasurement]) -> str:
    header = (
        "+-------+---------+-----------+-----------+-----------+-----------+-----------+\n"
        "| Scale | Init s  | Init MB   | Infer ms  | Learn ms  | Sparsity  | Post MB   |\n"
        "+-------+---------+-----------+-----------+-----------+-----------+-----------+"
    )
    rows = [
        f"| {result.scale:>5} | {result.init_seconds:>7.3f} | {result.init_memory_mb:>9.2f} | "
        f"{result.inference_ms:>9.3f} | {result.learning_ms:>9.3f} | {result.sparsity * 100:>8.3f}% | "
        f"{result.post_learning_memory_mb:>9.2f} |"
        for result in results
    ]
    return "\n".join([header, *rows, "+-------+---------+-----------+-----------+-----------+-----------+-----------+"])


def run_scaling_experiment() -> list[ScaleMeasurement]:
    torch.manual_seed(42)
    return [measure_scale(scale) for scale in SCALES]


def main() -> None:
    results = run_scaling_experiment()
    flat_10k_ms, sharded_10k_ms = benchmark_sharding(10000)
    print("Large CCC pool scaling")
    print(render_table(results))
    print(
        f"\n10K flat vs sharded latency: flat={flat_10k_ms:.3f} ms/sample, "
        f"sharded={sharded_10k_ms:.3f} ms/sample"
    )


if __name__ == "__main__":
    main()
