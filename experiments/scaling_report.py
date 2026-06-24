"""Generate a production-oriented scaling report for large CCC pools."""

from __future__ import annotations

from experiments.large_pool_scaling import benchmark_sharding, render_table, run_scaling_experiment


def recommend_pool_size(results) -> list[str]:
    recommendations: list[str] = []
    small = next(result for result in results if result.scale == 1000)
    medium = next(result for result in results if result.scale == 5000)
    large = next(result for result in results if result.scale == 10000)

    recommendations.append(
        f"- Up to 1K CCCs: use a flat BatchedCCCPool ({small.inference_ms:.2f} ms/sample, {small.init_memory_mb:.1f} MB)."
    )
    recommendations.append(
        f"- Around 5K CCCs: flat pooling remains practical ({medium.inference_ms:.2f} ms/sample, {medium.init_memory_mb:.1f} MB)."
    )
    recommendations.append(
        f"- At 10K CCCs: use sharding when CPU latency matters most ({large.inference_ms:.2f} ms/sample flat baseline)."
    )
    return recommendations


def memory_budget_guidance(results) -> list[str]:
    first = results[0]
    last = results[-1]
    mb_per_ccc = (last.init_memory_mb - first.init_memory_mb) / max(last.scale - first.scale, 1)
    return [
        f"- Observed pool memory slope: ~{mb_per_ccc:.4f} MB per CCC.",
        "- Reserve ~25% headroom above the measured pool footprint for activations and experiment buffers.",
        "- Prefer MemoryEfficientSDM when associative memory occupancy stays sparse or when using float16 storage.",
    ]


def sharding_guidance(flat_ms: float, sharded_ms: float) -> list[str]:
    winner = "sharding" if sharded_ms < flat_ms else "flat pooling"
    return [
        f"- 10K latency comparison: flat={flat_ms:.3f} ms/sample, sharded={sharded_ms:.3f} ms/sample.",
        f"- Use {winner} when query locality is strong and concept prototypes cluster cleanly.",
        "- Keep flat pooling for smaller pools or when exact global competition must be evaluated every step.",
    ]


def main() -> None:
    results = run_scaling_experiment()
    flat_ms, sharded_ms = benchmark_sharding(10000)

    print("Bio-ARN large-pool scaling report")
    print(render_table(results))
    print("\nRecommendations")
    for line in recommend_pool_size(results):
        print(line)
    print("\nMemory budget guidelines")
    for line in memory_budget_guidance(results):
        print(line)
    print("\nSharding guidance")
    for line in sharding_guidance(flat_ms, sharded_ms):
        print(line)


if __name__ == "__main__":
    main()
