"""Tests for streaming Bio-ARN data pipelines."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

import pytest
import torch

from bioarn.data import (
    CIFAR10Stream,
    CharacterStream,
    CurriculumScheduler,
    DataSample,
    MNISTStream,
    MultimodalStream,
    OnlineAugmenter,
    StreamingDataSource,
    WikiTextStream,
)


class StaticStream(StreamingDataSource):
    def __init__(self, samples: list[DataSample]) -> None:
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def stream(self) -> Iterator[DataSample]:
        for sample in self.samples:
            yield DataSample(
                data=sample.data.clone(),
                label=sample.label,
                modality=sample.modality,
                metadata=dict(sample.metadata),
            )


def _write_idx(path: Path, magic: int, shape: tuple[int, ...], payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(magic.to_bytes(4, "big"))
        for dimension in shape:
            handle.write(int(dimension).to_bytes(4, "big"))
        handle.write(payload)


@pytest.fixture()
def mnist_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "MNIST" / "raw"
    labels = torch.tensor([2, 0, 1, 1, 0], dtype=torch.uint8)
    images = []
    for label in labels.tolist():
        image = torch.full((28, 28), fill_value=label * 32, dtype=torch.uint8)
        image[0, 0] = label
        images.append(image)
    payload = torch.stack(images, dim=0)
    _write_idx(raw / "train-images-idx3-ubyte", 2051, payload.shape, payload.numpy().tobytes())
    _write_idx(raw / "train-labels-idx1-ubyte", 2049, labels.shape, labels.numpy().tobytes())
    return tmp_path


@pytest.fixture()
def cifar_dir(tmp_path: Path) -> Path:
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir(parents=True, exist_ok=True)
    batch = []
    labels = [3, 1, 9]
    for label in labels:
        image = torch.full((3, 32, 32), fill_value=label * 8, dtype=torch.uint8)
        image[0, 0, 0] = label
        batch.append(image.reshape(-1))
    payload = {
        b"data": torch.stack(batch, dim=0).numpy(),
        b"labels": labels,
        b"filenames": [f"sample_{index}.png".encode("utf-8") for index in range(len(labels))],
    }
    with (root / "data_batch_1").open("wb") as handle:
        pickle.dump(payload, handle)
    for index in range(2, 6):
        with (root / f"data_batch_{index}").open("wb") as handle:
            pickle.dump({b"data": torch.empty((0, 3072), dtype=torch.uint8).numpy(), b"labels": []}, handle)
    with (root / "test_batch").open("wb") as handle:
        pickle.dump(payload, handle)
    return tmp_path


@pytest.fixture()
def wikitext_dir(tmp_path: Path) -> Path:
    root = tmp_path / "WikiText" / "wikitext-2-raw-v1"
    root.mkdir(parents=True, exist_ok=True)
    (root / "wiki.train.raw").write_text("Bio-ARN streams text one character at a time.", encoding="utf-8")
    return tmp_path


def test_mnist_stream_shape(mnist_dir: Path) -> None:
    stream = MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False)
    sample = next(stream.stream())
    assert sample.data.shape == (784,)


def test_mnist_stream_labels(mnist_dir: Path) -> None:
    stream = MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False)
    labels = [sample.label for sample in stream.stream()]
    assert all(label is not None and 0 <= label <= 9 for label in labels)


def test_cifar10_stream_shape(cifar_dir: Path) -> None:
    stream = CIFAR10Stream(data_dir=cifar_dir, flatten=True, shuffle=False)
    sample = next(stream.stream())
    assert sample.data.shape == (3072,)


def test_character_stream() -> None:
    stream = CharacterStream("BioARN online", context_length=4, stride=2)
    sample = next(stream.stream())
    assert sample.modality == "language"
    assert sample.data.dtype == torch.long
    assert sample.data.shape == (4,)


def test_wikitext_stream_yields(wikitext_dir: Path) -> None:
    stream = WikiTextStream(version="2", split="train", context_length=8, data_dir=wikitext_dir, stride=4)
    sample = next(stream.stream())
    assert sample.data.shape == (8,)
    assert sample.data.dtype == torch.long


def test_batched_streaming(mnist_dir: Path) -> None:
    stream = MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False)
    batch = next(stream.stream_batched(2))
    assert batch.batch_size == 2
    assert batch.data.shape == (2, 784)
    assert batch.labels is not None
    assert batch.labels.shape == (2,)


def test_class_sequential(mnist_dir: Path) -> None:
    stream = MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False, class_sequential=True)
    labels = [sample.label for sample in stream.stream()]
    assert labels == sorted(labels)


def test_augmentation_changes_input() -> None:
    augmenter = OnlineAugmenter(
        flip_prob=1.0,
        rotation_prob=1.0,
        crop_prob=1.0,
        noise_prob=1.0,
        occlusion_prob=1.0,
        noise_std=0.2,
        seed=0,
    )
    sample = DataSample(data=torch.full((784,), 0.5), label=1, modality="vision", metadata={})
    augmented = augmenter.augment(sample)
    assert not torch.allclose(augmented.data, sample.data)


def test_curriculum_easy_first() -> None:
    stream = StaticStream(
        [
            DataSample(torch.tensor([0.1]), 0, "vision", {"difficulty": 0.8}),
            DataSample(torch.tensor([0.2]), 1, "vision", {"difficulty": 0.2}),
            DataSample(torch.tensor([0.3]), 2, "vision", {"difficulty": 0.5}),
        ]
    )
    ordered = CurriculumScheduler.easy_first(stream)
    difficulties = [sample.metadata["difficulty"] for sample in ordered.stream()]
    assert difficulties == [0.2, 0.5, 0.8]


def test_multimodal_alternates() -> None:
    vision = StaticStream(
        [
            DataSample(torch.tensor([1.0]), 0, "vision", {}),
            DataSample(torch.tensor([2.0]), 1, "vision", {}),
        ]
    )
    language = StaticStream(
        [
            DataSample(torch.tensor([3]), None, "language", {}),
            DataSample(torch.tensor([4]), None, "language", {}),
        ]
    )
    stream = MultimodalStream(vision, language, ratio=0.5)
    modalities = [sample.modality for sample in stream.stream()]
    assert modalities == ["vision", "language", "vision", "language"]


def test_stream_repeatable(mnist_dir: Path) -> None:
    stream = MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False)
    first = list(stream.stream())
    second = list(stream.stream())
    assert len(first) == len(second)
    assert torch.allclose(first[0].data, second[0].data)
    assert first[0].label == second[0].label


def test_data_sample_structure(mnist_dir: Path) -> None:
    sample = next(MNISTStream(data_dir=mnist_dir, flatten=True, shuffle=False).stream())
    assert isinstance(sample, DataSample)
    assert hasattr(sample, "data")
    assert hasattr(sample, "label")
    assert hasattr(sample, "modality")
    assert hasattr(sample, "metadata")
