from __future__ import annotations

import pickle
from pathlib import Path

import torch

from bioarn.data.vision import AugmentedCIFARStream, HebbianAugmentation


def _write_cifar_batch(path: Path, labels: list[int]) -> None:
    images = []
    for label in labels:
        image = torch.full((3, 32, 32), fill_value=label * 8, dtype=torch.uint8)
        image[0, 0, 0] = label
        images.append(image.reshape(-1))
    payload = {
        b"data": (
            torch.stack(images, dim=0).numpy()
            if images
            else torch.empty((0, 3072), dtype=torch.uint8).numpy()
        ),
        b"labels": labels,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _build_cifar_dir(tmp_path: Path) -> Path:
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir(parents=True, exist_ok=True)
    _write_cifar_batch(root / "data_batch_1", [1, 2])
    for index in range(2, 6):
        _write_cifar_batch(root / f"data_batch_{index}", [])
    _write_cifar_batch(root / "test_batch", [1, 2])
    return tmp_path


def test_random_flip_changes_image(monkeypatch) -> None:
    augmentation = HebbianAugmentation(random_flip=True, random_crop=False, color_jitter=False, cutout=False)
    monkeypatch.setattr(augmentation, "_rand", lambda: 0.0)
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4) / 255.0

    augmented = augmentation(image)

    assert not torch.allclose(augmented, image)
    assert torch.allclose(augmented, torch.flip(image, dims=[2]))


def test_random_crop_preserves_size() -> None:
    augmentation = HebbianAugmentation(random_flip=False, random_crop=True, color_jitter=False, cutout=False)
    augmentation.generator.manual_seed(0)
    image = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(1))

    augmented = augmentation(image)

    assert augmented.shape == image.shape


def test_augmentation_factor(tmp_path: Path) -> None:
    cifar_dir = _build_cifar_dir(tmp_path)
    stream = AugmentedCIFARStream(
        num_samples=2,
        augmentation=HebbianAugmentation(random_flip=True, random_crop=False, color_jitter=False, cutout=False),
        augmentation_factor=3,
        seed=0,
        data_dir=cifar_dir,
    )

    samples = list(stream.stream())

    assert len(samples) == 6
    assert len(stream) == 6


def test_no_augmentation_passthrough() -> None:
    augmentation = HebbianAugmentation(random_flip=False, random_crop=False, color_jitter=False, cutout=False)
    image = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(2))

    augmented = augmentation(image)

    assert torch.allclose(augmented, image)
