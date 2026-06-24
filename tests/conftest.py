from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

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
from bioarn.system import BioARNCore
from bioarn.training import OnlineTrainer


@dataclass
class SyntheticMNISTData:
    """Small deterministic MNIST-like dataset for fast CI coverage."""

    train_stream: list[tuple[torch.Tensor, int]]
    eval_stream: list[tuple[torch.Tensor, int]]
    early_stream: list[tuple[torch.Tensor, int]]
    late_stream: list[tuple[torch.Tensor, int]]
    noise_stream: list[torch.Tensor]
    class_a_batch: torch.Tensor
    class_b_batch: torch.Tensor
    novel_pattern: torch.Tensor
    visual_seed: torch.Tensor
    reward_sequence: list[torch.Tensor]


@dataclass
class TrainedSystem:
    """Fixture payload containing a trained core and its trainer state."""

    system: BioARNCore
    trainer: OnlineTrainer
    train_result: object
    data: SyntheticMNISTData


def _draw_synthetic_digit(label: int) -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    if label == 0:
        image[6:22, 6] = 1.0
        image[6:22, 21] = 1.0
        image[6, 6:22] = 1.0
        image[21, 6:22] = 1.0
    elif label == 1:
        image[6:22, 14] = 1.0
        image[21, 10:18] = 1.0
        image[7, 12:16] = 1.0
    elif label == 2:
        image[6, 6:22] = 1.0
        image[13, 6:22] = 1.0
        image[21, 6:22] = 1.0
        image[7:13, 21] = 1.0
        image[14:21, 6] = 1.0
    elif label == 3:
        image[6, 6:22] = 1.0
        image[13, 8:22] = 1.0
        image[21, 6:22] = 1.0
        image[7:21, 21] = 1.0
    else:
        raise ValueError(f"Unsupported synthetic digit label: {label}")
    return image.reshape(-1)


def _noisy_variant(base: torch.Tensor, *, seed: int, noise: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    sample = base + (noise * torch.randn(base.shape, generator=generator, dtype=base.dtype))
    return sample.clamp_(0.0, 1.0)


def _make_stream(labels: list[int], *, per_label: int, noise: float, seed_offset: int) -> list[tuple[torch.Tensor, int]]:
    stream: list[tuple[torch.Tensor, int]] = []
    for label in labels:
        base = _draw_synthetic_digit(label)
        for index in range(per_label):
            stream.append(
                (
                    _noisy_variant(base, seed=(seed_offset + (label * 1000) + index), noise=noise),
                    label,
                )
            )
    return stream


def _make_class_batch(label: int, *, count: int, noise: float, seed_offset: int) -> torch.Tensor:
    base = _draw_synthetic_digit(label)
    samples = [
        _noisy_variant(base, seed=(seed_offset + index), noise=noise)
        for index in range(count)
    ]
    return torch.stack(samples, dim=0)


@pytest.fixture
def small_config() -> BioARNConfig:
    """Small config for fast tests."""

    return BioARNConfig(
        spiking=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        ccc=CCCConfig(
            input_dim=784,
            concept_dim=64,
            num_f1_features=128,
            f1_top_k=16,
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=50,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.5,
            theta_margin_lr=0.001,
            theta_resonance=0.7,
        ),
        sdm=SDMConfig(
            address_dim=64,
            hamming_radius=8,
            num_hard_locations=128,
            data_dim=64,
            decay_rate=0.999,
            stdp_window=8,
        ),
        predictive=PredictiveConfig(
            num_levels=3,
            gamma=0.15,
            eta=0.04,
            precision_init=1.0,
            error_threshold=0.0,
        ),
        gnw=GNWConfig(
            capacity=5,
            broadcast_gain=2.0,
            fatigue_rate=0.05,
            fatigue_threshold=0.1,
            competition_temp=0.7,
        ),
        reward=RewardConfig(
            intrinsic_scale=1.0,
            novelty_threshold=1.25,
            novelty_boost=2.0,
            novelty_decay=0.9,
            curiosity_weight=0.5,
        ),
        seed=11,
    )


@pytest.fixture
def performance_config() -> BioARNConfig:
    """Lean config for CPU-oriented performance regression tests."""

    return BioARNConfig(
        spiking=SpikingConfig(beta=0.0, threshold=0.5, reset=0.0, refractory_steps=0),
        ccc=CCCConfig(
            input_dim=128,
            concept_dim=32,
            num_f1_features=64,
            f1_top_k=8,
            fast_lr=1.0,
            slow_lr=0.05,
            feedback_lr=0.05,
            max_pool_size=20,
        ),
        margin_gate=MarginGateConfig(
            theta_margin=0.5,
            theta_margin_lr=0.001,
            theta_resonance=0.7,
        ),
        sdm=SDMConfig(
            address_dim=32,
            hamming_radius=4,
            num_hard_locations=64,
            data_dim=32,
            decay_rate=0.999,
            stdp_window=8,
        ),
        predictive=PredictiveConfig(
            num_levels=3,
            gamma=0.15,
            eta=0.04,
            precision_init=1.0,
            error_threshold=0.0,
        ),
        gnw=GNWConfig(
            capacity=5,
            broadcast_gain=2.0,
            fatigue_rate=0.05,
            fatigue_threshold=0.1,
            competition_temp=0.7,
        ),
        reward=RewardConfig(
            intrinsic_scale=1.0,
            novelty_threshold=1.25,
            novelty_boost=2.0,
            novelty_decay=0.9,
            curiosity_weight=0.5,
        ),
        seed=11,
    )


@pytest.fixture
def sample_mnist_data() -> SyntheticMNISTData:
    """Small batch of MNIST-like data for testing."""

    train_stream = _make_stream([0, 1], per_label=50, noise=0.01, seed_offset=0)
    eval_stream = _make_stream([0, 1], per_label=20, noise=0.02, seed_offset=10_000)
    noise_stream = [
        torch.rand(784, generator=torch.Generator().manual_seed(20_000 + index), dtype=torch.float32)
        for index in range(20)
    ]
    reward_base = _draw_synthetic_digit(0).reshape(28, 28)
    reward_sequence = [
        _noisy_variant(reward_base.reshape(-1), seed=30_000 + index, noise=noise)
        .reshape(1, 1, 28, 28)
        for index, noise in enumerate((0.05, 0.08, 0.06, 0.07))
    ]
    novel_pattern = torch.zeros(784, dtype=torch.float32)
    novel_pattern[::7] = 1.0
    visual_seed = train_stream[0][0].reshape(1, 1, 28, 28)
    return SyntheticMNISTData(
        train_stream=train_stream,
        eval_stream=eval_stream,
        early_stream=train_stream[:20],
        late_stream=train_stream[20:],
        noise_stream=noise_stream,
        class_a_batch=_make_class_batch(0, count=8, noise=0.01, seed_offset=50_000),
        class_b_batch=_make_class_batch(1, count=8, noise=0.01, seed_offset=51_000),
        novel_pattern=novel_pattern,
        visual_seed=visual_seed,
        reward_sequence=reward_sequence,
    )


@pytest.fixture
def performance_sample() -> torch.Tensor:
    """Stable performance benchmark sample."""

    sample = torch.zeros(128, dtype=torch.float32)
    sample[:16] = 1.0
    return sample


@pytest.fixture
def performance_patterns() -> list[torch.Tensor]:
    """Distinct sparse patterns that recruit multiple CCCs."""

    patterns: list[torch.Tensor] = []
    for index in range(10):
        pattern = torch.zeros(128, dtype=torch.float32)
        start = index * 8
        pattern[start : start + 8] = 1.0
        pattern[(start + 32) % 128 : ((start + 32) % 128) + 4] = 0.5
        patterns.append(pattern)
    return patterns


@pytest.fixture
def trained_system(small_config: BioARNConfig, sample_mnist_data: SyntheticMNISTData) -> TrainedSystem:
    """A system that's been trained on a few MNIST samples."""

    trainer = OnlineTrainer(log_every=1_000, checkpoint_every=1_000)
    system = BioARNCore(deepcopy(small_config))
    train_result = trainer.train(system, sample_mnist_data.train_stream, deepcopy(small_config))
    return TrainedSystem(
        system=system,
        trainer=trainer,
        train_result=train_result,
        data=sample_mnist_data,
    )
