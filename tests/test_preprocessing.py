from __future__ import annotations

from functools import lru_cache

import torch

from bioarn.preprocessing import (
    ContrastNormalizer,
    OnlinePCA,
    PatchEncoder,
    PreprocessingPipeline,
    SparseRandomProjection,
)
from bioarn.training import VisionTrainConfig, VisionTrainer, load_cifar10_or_synthetic, take_samples


def _structured_data(
    num_samples: int,
    input_dim: int,
    latent_dim: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(num_samples, latent_dim, generator=generator)
    basis = torch.randn(latent_dim, input_dim, generator=generator)
    noise = 0.02 * torch.randn(num_samples, input_dim, generator=generator)
    return (latent @ basis + noise).to(torch.float32)


def _subspace_projection(components: torch.Tensor) -> torch.Tensor:
    return components.transpose(0, 1) @ components


@lru_cache(maxsize=4)
def _cifar_subset(
    train_size: int = 480,
    test_size: int = 120,
    seed: int = 31,
) -> tuple[list[tuple[torch.Tensor, int | None]], list[tuple[torch.Tensor, int | None]], str]:
    train_stream, test_stream, source = load_cifar10_or_synthetic(
        data_dir="data",
        train_samples=train_size,
        test_samples=test_size,
        seed=seed,
        timeout_seconds=5.0,
    )
    return (
        take_samples(train_stream, train_size),
        take_samples(test_stream, test_size),
        source,
    )


def _trainer(
    *,
    threshold: float = 0.4,
    max_pool_size: int = 192,
    num_train_samples: int = 480,
    num_test_samples: int = 120,
    preprocessing: PreprocessingPipeline | None = None,
) -> VisionTrainer:
    return VisionTrainer(
        VisionTrainConfig(
            input_dim=3072,
            concept_dim=256,
            max_pool_size=max_pool_size,
            margin_threshold=threshold,
            use_batched=True,
            batch_size=32,
            learning_rate=0.01,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            preprocessing_warmup_samples=200,
        ),
        preprocessing=preprocessing,
    )


def test_online_pca_reduces_dim() -> None:
    data = _structured_data(32, 3072, 24, seed=1)
    pca = OnlinePCA(3072, 128, max_samples=64, seed=2)

    transformed = pca.fit_transform(data)

    assert transformed.shape == (32, 128)


def test_online_pca_incremental() -> None:
    data = _structured_data(64, 96, 12, seed=3)
    batch_pca = OnlinePCA(96, 8, max_samples=128, seed=4).fit(data)
    online_pca = OnlinePCA(96, 8, max_samples=128, seed=4)
    for sample in data:
        online_pca.partial_fit(sample)

    assert torch.allclose(batch_pca.mean, online_pca.mean, atol=1e-5)
    assert torch.allclose(
        _subspace_projection(batch_pca.components),
        _subspace_projection(online_pca.components),
        atol=1e-3,
    )


def test_patch_encoder_shape() -> None:
    encoder = PatchEncoder(image_size=(32, 32, 3), patch_size=8, output_dim=128, seed=5)
    batch = torch.rand(4, 3072)

    encoded = encoder.transform(batch)

    assert encoded.shape == (4, 128)


def test_contrast_norm_effect() -> None:
    generator = torch.Generator().manual_seed(6)
    batch = torch.rand(8, 3072, generator=generator)
    normalizer = ContrastNormalizer(kernel_size=3)

    transformed = normalizer.transform(batch).reshape(8, 3, 32, 32)

    assert abs(float(transformed.mean().item())) < 0.1
    assert 0.75 < float(transformed.std(unbiased=False).item()) < 1.25


def test_random_projection_shape() -> None:
    projector = SparseRandomProjection(3072, output_dim=256, density=0.1, seed=7)
    batch = torch.rand(5, 3072)

    projected = projector.transform(batch)

    assert projected.shape == (5, 256)


def test_random_projection_preserves_distance() -> None:
    data = _structured_data(24, 512, 32, seed=8)
    projector = SparseRandomProjection(512, output_dim=192, density=0.15, seed=9)

    original = torch.pdist(data)
    projected = torch.pdist(projector.transform(data))
    relative_error = ((projected - original).abs() / original.clamp_min(1e-6)).mean()

    assert float(relative_error.item()) < 0.30


def test_pipeline_chains() -> None:
    pipeline = PreprocessingPipeline(
        [
            ("contrast", ContrastNormalizer(kernel_size=3)),
            ("pca", OnlinePCA(3072, 128, max_samples=64, seed=10)),
        ]
    )
    batch = torch.rand(12, 3072)

    transformed = pipeline.fit_transform(batch)

    assert transformed.shape == (12, 128)


def test_pipeline_fit_transform() -> None:
    batch = torch.rand(20, 3072, generator=torch.Generator().manual_seed(11))
    pipeline_a = PreprocessingPipeline(
        [
            ("contrast", ContrastNormalizer(kernel_size=3)),
            ("pca", OnlinePCA(3072, 64, max_samples=64, seed=12)),
        ]
    )
    pipeline_b = PreprocessingPipeline(
        [
            ("contrast", ContrastNormalizer(kernel_size=3)),
            ("pca", OnlinePCA(3072, 64, max_samples=64, seed=12)),
        ]
    )

    fit_then_transform = pipeline_a.fit(batch).transform(batch)
    fit_transform = pipeline_b.fit_transform(batch)

    assert torch.allclose(fit_then_transform, fit_transform, atol=1e-5)


def test_cifar_no_collapse() -> None:
    train_samples, _, _ = _cifar_subset(train_size=360, test_size=80, seed=32)
    trainer = _trainer(
        threshold=0.4,
        max_pool_size=192,
        num_train_samples=360,
        num_test_samples=80,
        preprocessing=PreprocessingPipeline(
            [("pca", OnlinePCA(3072, 128, max_samples=200, seed=13))]
        ),
    )

    trainer.train_online(train_samples, num_samples=len(train_samples))
    analysis = trainer.get_ccc_analysis()

    assert analysis["committed_cccs"] > 5


def test_cifar_accuracy_above_chance() -> None:
    train_samples, test_samples, _ = _cifar_subset(train_size=480, test_size=120, seed=33)
    trainer = _trainer(
        threshold=0.4,
        max_pool_size=224,
        num_train_samples=480,
        num_test_samples=120,
        preprocessing=PreprocessingPipeline(
            [("pca", OnlinePCA(3072, 128, max_samples=200, seed=14))]
        ),
    )

    trainer.train_online(train_samples, num_samples=len(train_samples))
    metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))

    assert metrics["accuracy"] > 0.15


def test_different_preprocessors_compare() -> None:
    train_samples, test_samples, _ = _cifar_subset(train_size=360, test_size=100, seed=34)
    configs = [
        (
            "raw",
            None,
        ),
        (
            "pca",
            PreprocessingPipeline([("pca", OnlinePCA(3072, 128, max_samples=180, seed=15))]),
        ),
        (
            "patches",
            PreprocessingPipeline(
                [("patches", PatchEncoder(image_size=(32, 32, 3), patch_size=8, output_dim=128, seed=16))]
            ),
        ),
    ]
    results: list[tuple[float, int]] = []

    for _, pipeline in configs:
        trainer = _trainer(
            threshold=0.4,
            max_pool_size=192,
            num_train_samples=360,
            num_test_samples=100,
            preprocessing=pipeline,
        )
        trainer.train_online(train_samples, num_samples=len(train_samples))
        metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
        analysis = trainer.get_ccc_analysis()
        results.append((round(float(metrics["accuracy"]), 3), int(analysis["committed_cccs"])))

    assert len(set(results)) > 1


def test_margin_threshold_tuning() -> None:
    train_samples, test_samples, _ = _cifar_subset(train_size=360, test_size=100, seed=35)
    outcomes: list[tuple[int, float]] = []

    for threshold in (0.30, 0.40, 0.45):
        trainer = _trainer(
            threshold=threshold,
            max_pool_size=192,
            num_train_samples=360,
            num_test_samples=100,
            preprocessing=PreprocessingPipeline(
                [("pca", OnlinePCA(3072, 128, max_samples=180, seed=17))]
            ),
        )
        trainer.train_online(train_samples, num_samples=len(train_samples))
        metrics = trainer.evaluate(test_samples, num_samples=len(test_samples))
        analysis = trainer.get_ccc_analysis()
        outcomes.append((int(analysis["committed_cccs"]), round(float(metrics["abstention_rate"]), 3)))

    assert len(set(outcomes)) > 1
