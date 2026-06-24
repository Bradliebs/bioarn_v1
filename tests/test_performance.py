from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
import torch

from bioarn.config import MarginGateConfig
from bioarn.core.ccc import CCCPool
from bioarn.scaling import BatchedCCCPool, ScaledBioARN
from bioarn.system import BioARNCore
from bioarn.utils.checkpoint import CheckpointManager


pytestmark = pytest.mark.slow


def _baseline() -> dict[str, float]:
    path = Path(__file__).with_name("performance_baseline.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _measure_ms(callback, *, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        callback()
    return ((time.perf_counter() - start) * 1000.0) / max(repeats, 1)


def _module_tensor_memory_mb(module: torch.nn.Module) -> float:
    bytes_total = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(module.parameters()) + list(module.buffers())
    )
    return bytes_total / (1024.0 * 1024.0)


def _commit_sparse_patterns(core: BioARNCore, patterns: list[torch.Tensor]) -> None:
    for pattern in patterns:
        core.forward(pattern, learn=True)


class TestPerformanceBaseline:
    """Ensure performance doesn't regress across changes."""

    def test_inference_latency(self, performance_config, performance_sample) -> None:
        baseline = _baseline()
        core = BioARNCore(deepcopy(performance_config))
        for _ in range(5):
            core.forward(performance_sample, learn=True)

        latency_ms = _measure_ms(lambda: core.recognize(performance_sample), repeats=30)

        assert latency_ms <= baseline["inference_latency_ms"] * 1.10
        assert latency_ms < 50.0

    def test_batch_inference_latency(self, performance_config, performance_sample) -> None:
        baseline = _baseline()
        core = ScaledBioARN(deepcopy(performance_config), use_optimized=True)
        for _ in range(5):
            core.forward(performance_sample, learn=True)
        batch = performance_sample.unsqueeze(0).repeat(32, 1)

        latency_ms = _measure_ms(
            lambda: core.ccc_pool(batch, timestep=0, allow_recruit=False),
            repeats=20,
        )

        assert latency_ms <= baseline["batch_32_latency_ms"] * 1.10
        assert latency_ms < 200.0

    def test_memory_usage(self, performance_config) -> None:
        baseline = _baseline()
        config = deepcopy(performance_config)
        config.ccc.max_pool_size = 1_000
        model = ScaledBioARN(config, use_optimized=True)

        memory_mb = _module_tensor_memory_mb(model)

        assert memory_mb <= baseline["memory_1000_cccs_mb"] * 1.10
        assert memory_mb < 500.0

    def test_sparsity_maintained(self, performance_config, performance_patterns) -> None:
        baseline = _baseline()
        config = deepcopy(performance_config)
        config.ccc.max_pool_size = 20
        core = BioARNCore(config)
        _commit_sparse_patterns(core, performance_patterns)

        perception = core.perceive(performance_patterns[0])
        concepts = max(1, core.get_system_stats()["concepts_learned"])
        activation_density = perception.num_fired / concepts

        assert activation_density <= baseline["sparsity_threshold"]

    def test_learning_speed(self, performance_config, performance_sample) -> None:
        baseline = _baseline()
        core = BioARNCore(deepcopy(performance_config))
        stream = [performance_sample.roll(shifts=index % 8) for index in range(100)]

        start = time.perf_counter()
        for sample in stream:
            core.forward(sample, learn=True)
        duration = time.perf_counter() - start

        assert duration <= baseline["learning_100_samples_sec"]
        assert duration < 5.0

    def test_checkpoint_size(self, tmp_path, performance_config) -> None:
        config = deepcopy(performance_config)
        config.ccc.max_pool_size = 1_000
        model = ScaledBioARN(config, use_optimized=True)

        checkpoint_path = tmp_path / "scaled-bioarn.pt"
        CheckpointManager().save(model, checkpoint_path)
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024.0 * 1024.0)

        assert checkpoint_size_mb < 100.0

    def test_ccc_vectorized_speedup(self, performance_config, performance_sample) -> None:
        config = deepcopy(performance_config)
        config.ccc.max_pool_size = 500
        margin = MarginGateConfig(theta_margin=0.2, theta_margin_lr=0.0, theta_resonance=1.1)
        original = CCCPool(config.ccc, margin)
        optimized = BatchedCCCPool(config.ccc, margin).load_from_pool(original)

        with torch.no_grad():
            for index in range(config.ccc.max_pool_size):
                optimized.committed_mask[index] = True
                optimized.concept_directions[index] = torch.randn(config.ccc.concept_dim)
                optimized.feedback_weights[index] = torch.randn(
                    config.ccc.num_f1_features,
                    config.ccc.concept_dim,
                )

        sample_batch = performance_sample.unsqueeze(0)

        def loop_kernel() -> None:
            for index in range(config.ccc.max_pool_size):
                optimized._single_forward_index(index, sample_batch, False, 1)  # noqa: SLF001

        def vectorized_kernel() -> None:
            optimized._vectorized_state(sample_batch, 1)  # noqa: SLF001

        loop_ms = _measure_ms(loop_kernel, repeats=3)
        vectorized_ms = _measure_ms(vectorized_kernel, repeats=3)
        speedup = loop_ms / max(vectorized_ms, 1e-9)

        assert speedup >= 2.0
