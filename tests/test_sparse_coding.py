from __future__ import annotations

import inspect

import torch

from bioarn.preprocessing import (
    CompetitiveLearner,
    HebbianSparseCoder,
    OnlineDictionaryLearner,
    PreprocessingPipeline,
)


def _low_rank_data(
    num_samples: int = 256,
    input_dim: int = 24,
    latent_dim: int = 6,
    *,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.rand(num_samples, latent_dim, generator=generator)
    basis = torch.rand(latent_dim, input_dim, generator=generator)
    noise = 0.02 * torch.randn(num_samples, input_dim, generator=generator)
    return (latent @ basis + noise).to(torch.float32)


def _mean_abs_off_diagonal(weights: torch.Tensor) -> float:
    gram = weights @ weights.transpose(0, 1)
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return float(gram[mask].abs().mean().item())


def test_sparse_coder_init() -> None:
    coder = HebbianSparseCoder(24, num_features=48, sparsity=0.1, learning_rate=0.02, seed=1)

    assert coder.dictionary.shape == (48, 24)
    assert coder.mean_vector.shape == (24,)
    assert coder.get_output_dim(24) == 48


def test_encode_is_sparse() -> None:
    coder = HebbianSparseCoder(24, num_features=50, sparsity=0.1, learning_rate=0.02, seed=2)
    encoded = coder.encode(torch.rand(8, 24))

    non_zero = (encoded.abs() > 1e-6).sum(dim=1)
    assert int(non_zero.max().item()) <= 5


def test_encode_shape() -> None:
    coder = HebbianSparseCoder(16, num_features=40, sparsity=0.1, learning_rate=0.02, seed=3)
    encoded = coder.encode(torch.rand(7, 16))

    assert encoded.shape == (7, 40)


def test_hebbian_learning_updates_weights() -> None:
    coder = HebbianSparseCoder(20, num_features=32, sparsity=0.1, learning_rate=0.03, seed=4)
    initial = coder.get_dictionary()

    coder.learn(torch.rand(20))
    updated = coder.get_dictionary()

    assert not torch.allclose(initial, updated)


def test_no_weight_explosion() -> None:
    coder = HebbianSparseCoder(18, num_features=36, sparsity=0.1, learning_rate=0.03, seed=5)
    data = _low_rank_data(num_samples=1000, input_dim=18, latent_dim=4, seed=6)

    for sample in data:
        coder.partial_fit(sample)

    norms = coder.get_dictionary().norm(dim=1)
    assert torch.all(norms < 1.1)
    assert torch.all(norms > 0.9)


def test_features_decorrelate() -> None:
    data = _low_rank_data(num_samples=512, input_dim=12, latent_dim=3, seed=7)
    coder = HebbianSparseCoder(
        12,
        num_features=48,
        sparsity=0.1,
        learning_rate=0.02,
        anti_hebbian_rate=0.05,
        seed=8,
    )
    initial_corr = _mean_abs_off_diagonal(coder.get_dictionary())

    for _ in range(4):
        coder.fit(data)

    final_corr = _mean_abs_off_diagonal(coder.get_dictionary())
    assert final_corr < initial_corr


def test_reconstruction_quality() -> None:
    data = _low_rank_data(num_samples=256, input_dim=24, latent_dim=6, seed=9)
    coder = HebbianSparseCoder(24, num_features=48, sparsity=0.1, learning_rate=0.03, seed=10)
    initial_error = coder.reconstruction_error(data[:64])

    coder.fit(data)
    final_error = coder.reconstruction_error(data[:64])

    assert final_error < initial_error


def test_dictionary_learning_converges() -> None:
    data = _low_rank_data(num_samples=256, input_dim=24, latent_dim=5, seed=11)
    learner = OnlineDictionaryLearner(
        24,
        dict_size=32,
        sparsity_target=0.1,
        learning_rate=0.05,
        max_matching_iters=4,
        seed=12,
    )

    learner.fit(data)

    assert learner.is_fitted
    assert learner.update_ema < 0.015


def test_matching_pursuit_sparse() -> None:
    learner = OnlineDictionaryLearner(
        24,
        dict_size=32,
        sparsity_target=0.1,
        learning_rate=0.05,
        max_matching_iters=4,
        seed=13,
    )
    learner.fit(_low_rank_data(num_samples=128, input_dim=24, latent_dim=5, seed=14))

    code = learner.matching_pursuit(torch.rand(24))

    assert code.shape == (32,)
    assert int((code.abs() > 1e-6).sum().item()) <= learner.max_active


def test_competitive_learning_clusters() -> None:
    generator = torch.Generator().manual_seed(15)
    centers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    points = torch.cat(
        [torch.randn(80, 2, generator=generator) * 0.05 + center for center in centers],
        dim=0,
    )
    learner = CompetitiveLearner(2, num_neurons=6, learning_rate=0.15, seed=16)

    learner.fit(points)
    codebook = learner.get_codebook()
    similarities = (codebook @ torch.nn.functional.normalize(centers, dim=1).transpose(0, 1)).abs()

    assert torch.all(similarities.max(dim=0).values > 0.9)


def test_pipeline_integration() -> None:
    pipeline = PreprocessingPipeline(
        [("sparse", HebbianSparseCoder(3072, 64, sparsity=0.05, learning_rate=0.02, seed=17))]
    )
    batch = torch.rand(6, 3072)

    transformed = pipeline.fit_transform(batch)

    assert transformed.shape == (6, 64)


def test_no_backprop_anywhere() -> None:
    coder = HebbianSparseCoder(16, num_features=32, sparsity=0.1, learning_rate=0.02, seed=18)
    dictionary = OnlineDictionaryLearner(
        16,
        dict_size=24,
        sparsity_target=0.1,
        learning_rate=0.05,
        max_matching_iters=3,
        seed=19,
    )
    competitive = CompetitiveLearner(16, num_neurons=12, learning_rate=0.02, seed=20)

    sample = torch.rand(16)
    coder.partial_fit(sample)
    dictionary.partial_fit(sample)
    competitive.partial_fit(sample)

    for module in (coder, dictionary, competitive):
        for _, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                assert value.grad is None

    for cls in (HebbianSparseCoder, OnlineDictionaryLearner, CompetitiveLearner):
        source = inspect.getsource(cls)
        assert ".backward(" not in source

