from __future__ import annotations

import pickle
from pathlib import Path

import torch

from bioarn.data import WhitenedCIFARStream, ZCAWhitening


def _structured_correlated_batch(
    num_samples: int = 256,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
    *,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    y_coords = torch.linspace(-1.0, 1.0, height)
    x_coords = torch.linspace(-1.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    bases = [
        torch.exp(-((grid_x**2) + (grid_y**2)) * 4.0),
        torch.sin(grid_x * 3.14159265),
        torch.cos(grid_y * 2.5),
        grid_x,
        grid_y,
    ]
    channel_patterns = []
    for channel in range(channels):
        for basis_index, basis in enumerate(bases):
            scale = 1.0 + (0.15 * channel) + (0.05 * basis_index)
            pattern = torch.zeros(channels, height, width, dtype=torch.float32)
            pattern[channel] = basis * scale
            if channel + 1 < channels:
                pattern[channel + 1] = basis.flip(0) * (0.15 + 0.05 * basis_index)
            channel_patterns.append(pattern.reshape(-1))
    dictionary = torch.stack(channel_patterns, dim=0)
    coefficients = torch.randn(num_samples, dictionary.shape[0], generator=generator)
    coefficients[:, 1:] *= 0.35
    batch = coefficients @ dictionary
    batch += 0.02 * torch.randn(batch.shape, generator=generator)
    return batch.to(torch.float32)


def _neighbor_correlation(image: torch.Tensor) -> float:
    horizontal = torch.nn.functional.cosine_similarity(
        image[:, :, 1:].reshape(1, -1),
        image[:, :, :-1].reshape(1, -1),
    )
    vertical = torch.nn.functional.cosine_similarity(
        image[:, 1:, :].reshape(1, -1),
        image[:, :-1, :].reshape(1, -1),
    )
    return float(((horizontal + vertical) * 0.5).item())


def _write_fake_cifar(path: Path, labels: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    batch = []
    for label in labels:
        image = torch.zeros(3, 32, 32, dtype=torch.uint8)
        top = 2 + (label % 4) * 6
        left = 2 + ((label // 4) % 4) * 6
        image[label % 3, top : top + 10, left : left + 10] = 180
        image[(label + 1) % 3, top + 2 : top + 8, left + 2 : left + 8] = 255
        image += torch.arange(32, dtype=torch.uint8).view(1, 32, 1)
        batch.append(image.reshape(-1))
    payload = {
        b"data": torch.stack(batch, dim=0).numpy(),
        b"labels": labels,
    }
    with (path / "data_batch_1").open("wb") as handle:
        pickle.dump(payload, handle)
    empty_payload = {b"data": torch.empty((0, 3072), dtype=torch.uint8).numpy(), b"labels": []}
    for index in range(2, 6):
        with (path / f"data_batch_{index}").open("wb") as handle:
            pickle.dump(empty_payload, handle)
    with (path / "test_batch").open("wb") as handle:
        pickle.dump(payload, handle)


def test_zca_decorrelates_inputs() -> None:
    generator = torch.Generator().manual_seed(1)
    input_dim = 36
    mixing = torch.randn(input_dim, input_dim, generator=generator)
    source = torch.randn(384, input_dim, generator=generator)
    data = (source @ mixing + 0.2).to(torch.float32)
    whitened = ZCAWhitening(epsilon=1e-4).fit_transform(data)

    centered = whitened - whitened.mean(dim=0)
    covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
    diagonal = torch.diag(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)

    assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.12, rtol=0.12)
    assert float(off_diagonal.abs().mean().item()) < 0.05


def test_zca_preserves_structure() -> None:
    data = _structured_correlated_batch(num_samples=256, seed=2)
    whitened = ZCAWhitening(epsilon=1e-4).fit_transform(data).reshape(-1, 3, 8, 8)

    sample = whitened[0]
    shuffled = sample.reshape(-1)[torch.randperm(sample.numel(), generator=torch.Generator().manual_seed(3))].reshape_as(sample)

    assert _neighbor_correlation(sample) > _neighbor_correlation(shuffled) + 0.08


def test_zca_fit_transform(tmp_path: Path) -> None:
    cifar_root = tmp_path / "cifar-10-batches-py"
    _write_fake_cifar(cifar_root, [0, 1, 2, 3, 4, 5])

    train_stream = WhitenedCIFARStream(
        split="train",
        data_dir=tmp_path,
        shuffle=False,
        n_fit_samples=4,
    )
    train_sample = next(train_stream.stream())
    restored = ZCAWhitening().load_state_dict(train_stream.whitener.state_dict())
    test_stream = WhitenedCIFARStream(
        split="test",
        data_dir=tmp_path,
        shuffle=False,
        whitening=restored,
        flatten=False,
    )
    test_sample = next(test_stream.stream())
    raw_sample = next(train_stream.base_stream.stream()).data

    assert train_stream.whitener.is_fitted is True
    assert train_sample.data.shape == (3072,)
    assert train_sample.metadata["whitening"] == "zca"
    assert test_sample.data.shape == (3, 32, 32)
    assert torch.allclose(
        restored.transform(raw_sample),
        train_stream.whitener.transform(raw_sample),
        atol=1e-6,
    )


def test_zca_deterministic() -> None:
    data = _structured_correlated_batch(num_samples=128, height=6, width=6, seed=4)
    first = ZCAWhitening(epsilon=1e-4, n_components=32).fit(data)
    second = ZCAWhitening(epsilon=1e-4, n_components=32).fit(data.clone())

    assert torch.allclose(first.mean, second.mean, atol=1e-6)
    assert torch.allclose(first.whitening_matrix, second.whitening_matrix, atol=1e-6)
    assert torch.allclose(first.transform(data), second.transform(data), atol=1e-6)
