"""Streaming vision datasets for Bio-ARN."""

from __future__ import annotations

import gzip
import os
import pickle
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from bioarn.data.base import DataSample, StreamingDataSource

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:  # pragma: no cover - optional dependency
    from torchvision.io import read_image
except Exception:  # pragma: no cover - optional dependency
    read_image = None

_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    part_path = destination.with_suffix(destination.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    try:
        with urllib.request.urlopen(url) as response, part_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            chunk_size = 1024 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100.0
                    print(f"Downloading {destination.name}: {percent:5.1f}% ({downloaded}/{total} bytes)", end="\r")
        if total:
            print(" " * 80, end="\r")
        part_path.replace(destination)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise


def _read_idx_metadata(path: Path) -> tuple[int, tuple[int, ...], int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        dims = magic % 256
        shape = tuple(int.from_bytes(handle.read(4), "big") for _ in range(dims))
    if not shape:
        raise ValueError(f"Invalid IDX file: {path}")
    return shape[0], shape[1:], 4 + 4 * dims


def _ensure_idx_file(raw_path: Path, gz_path: Path, url: str) -> Path:
    if raw_path.exists():
        return raw_path
    if not gz_path.exists():
        _download_with_progress(url, gz_path)
    with gzip.open(gz_path, "rb") as compressed, raw_path.open("wb") as raw:
        shutil.copyfileobj(compressed, raw)
    return raw_path


class _VisionStreamBase(StreamingDataSource):
    def __init__(
        self,
        *,
        split: str,
        flatten: bool,
        normalize: bool,
        shuffle: bool | None,
        class_sequential: bool,
        seed: int,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.split = split
        self.flatten = flatten
        self.normalize = normalize
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.class_sequential = class_sequential
        self.seed = seed

    def _ordered_indices_from_labels(self, labels: torch.Tensor) -> list[int]:
        indices = list(range(len(labels)))
        if self.class_sequential:
            indices.sort(key=lambda index: (int(labels[index].item()), index))
            return indices
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            return torch.randperm(len(labels), generator=generator).tolist()
        return indices

    def _reshape_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)
        if self.normalize:
            image = image / 255.0
        if self.flatten:
            image = image.reshape(-1)
        return self._move_tensor(image)


class MNISTStream(_VisionStreamBase):
    """MNIST streaming (downloads if needed)."""

    def __init__(
        self,
        split: str = "train",
        data_dir: str | os.PathLike[str] = "data/",
        flatten: bool = True,
        normalize: bool = True,
        shuffle: bool | None = None,
        class_sequential: bool = False,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        super().__init__(
            split=split,
            flatten=flatten,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=device,
        )
        root = Path(data_dir) / "MNIST" / "raw"
        prefix = "train" if split == "train" else "t10k"
        self.image_path = _ensure_idx_file(
            root / f"{prefix}-images-idx3-ubyte",
            root / f"{prefix}-images-idx3-ubyte.gz",
            f"{_MNIST_MIRROR}/{prefix}-images-idx3-ubyte.gz",
        )
        self.label_path = _ensure_idx_file(
            root / f"{prefix}-labels-idx1-ubyte",
            root / f"{prefix}-labels-idx1-ubyte.gz",
            f"{_MNIST_MIRROR}/{prefix}-labels-idx1-ubyte.gz",
        )
        self._length, self._image_shape, self._image_offset = _read_idx_metadata(self.image_path)
        label_length, _, self._label_offset = _read_idx_metadata(self.label_path)
        if label_length != self._length:
            raise ValueError("Image and label counts do not match for MNIST")
        self._labels = self._load_all_labels()
        self._ordered_indices = self._ordered_indices_from_labels(self._labels)

    def _load_all_labels(self) -> torch.Tensor:
        with self.label_path.open("rb") as handle:
            handle.seek(self._label_offset)
            return torch.frombuffer(bytearray(handle.read()), dtype=torch.uint8).clone().to(torch.long)

    def __len__(self) -> int:
        return self._length

    def stream(self) -> Iterator[DataSample]:
        image_size = int(np.prod(self._image_shape))
        with self.image_path.open("rb") as images, self.label_path.open("rb") as labels:
            for dataset_index in self._ordered_indices:
                images.seek(self._image_offset + dataset_index * image_size)
                labels.seek(self._label_offset + dataset_index)
                image = torch.frombuffer(bytearray(images.read(image_size)), dtype=torch.uint8).clone().reshape(self._image_shape)
                label = int.from_bytes(labels.read(1), "big")
                yield DataSample(
                    data=self._reshape_image(image),
                    label=label,
                    modality="vision",
                    metadata={"index": dataset_index, "split": self.split, "dataset": "mnist"},
                )


class _CIFARStreamBase(_VisionStreamBase):
    archive_url: str = ""
    extracted_dir: str = ""
    label_key: bytes = b"labels"
    data_key: bytes = b"data"
    train_files: tuple[str, ...] = ()
    test_files: tuple[str, ...] = ()
    dataset_name: str = ""

    def __init__(
        self,
        *,
        split: str,
        data_dir: str | os.PathLike[str],
        flatten: bool,
        normalize: bool,
        shuffle: bool | None,
        class_sequential: bool,
        seed: int,
        device: str | torch.device | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        super().__init__(
            split=split,
            flatten=flatten,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=device,
        )
        self.data_root = Path(data_dir)
        self.dataset_root = self._ensure_downloaded()
        self.batch_files = [self.dataset_root / name for name in (self.train_files if split == "train" else self.test_files)]
        self._batch_cache: dict[Path, dict[bytes, object]] = {}
        self._sample_refs = self._build_sample_refs()

    def _ensure_downloaded(self) -> Path:
        dataset_root = self.data_root / self.extracted_dir
        if dataset_root.exists():
            return dataset_root
        archive_path = self.data_root / Path(self.archive_url).name
        _download_with_progress(self.archive_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(self.data_root)
        return dataset_root

    def _load_batch(self, path: Path) -> dict[bytes, object]:
        if path not in self._batch_cache:
            with path.open("rb") as handle:
                self._batch_cache[path] = pickle.load(handle, encoding="bytes")
        return self._batch_cache[path]

    def _build_sample_refs(self) -> list[tuple[Path, int, int]]:
        refs: list[tuple[Path, int, int]] = []
        for batch_file in self.batch_files:
            batch = self._load_batch(batch_file)
            labels = batch[self.label_key]
            for local_index, label in enumerate(labels):
                refs.append((batch_file, local_index, int(label)))
        if self.class_sequential:
            refs.sort(key=lambda ref: (ref[2], str(ref[0]), ref[1]))
        elif self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(len(refs), generator=generator).tolist()
            refs = [refs[index] for index in order]
        return refs

    def __len__(self) -> int:
        return len(self._sample_refs)

    def stream(self) -> Iterator[DataSample]:
        for global_index, (batch_file, local_index, label) in enumerate(self._sample_refs):
            batch = self._load_batch(batch_file)
            raw = np.asarray(batch[self.data_key][local_index], dtype=np.uint8).reshape(3, 32, 32)
            image = torch.from_numpy(raw.copy())
            yield DataSample(
                data=self._reshape_image(image),
                label=label,
                modality="vision",
                metadata={
                    "index": global_index,
                    "local_index": local_index,
                    "split": self.split,
                    "dataset": self.dataset_name,
                    "batch_file": batch_file.name,
                },
            )


class CIFAR10Stream(_CIFARStreamBase):
    """CIFAR-10 streaming. Downloads, unpickles, normalizes."""

    archive_url = _CIFAR10_URL
    extracted_dir = "cifar-10-batches-py"
    label_key = b"labels"
    data_key = b"data"
    train_files = tuple(f"data_batch_{index}" for index in range(1, 6))
    test_files = ("test_batch",)
    dataset_name = "cifar10"

    def __init__(
        self,
        split: str = "train",
        data_dir: str | os.PathLike[str] = "data/",
        flatten: bool = True,
        normalize: bool = True,
        shuffle: bool | None = None,
        class_sequential: bool = False,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            split=split,
            data_dir=data_dir,
            flatten=flatten,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=device,
        )


class CIFAR100Stream(_CIFARStreamBase):
    """CIFAR-100 streaming."""

    archive_url = _CIFAR100_URL
    extracted_dir = "cifar-100-python"
    label_key = b"fine_labels"
    data_key = b"data"
    train_files = ("train",)
    test_files = ("test",)
    dataset_name = "cifar100"

    def __init__(
        self,
        split: str = "train",
        data_dir: str | os.PathLike[str] = "data/",
        flatten: bool = True,
        normalize: bool = True,
        shuffle: bool | None = None,
        class_sequential: bool = False,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            split=split,
            data_dir=data_dir,
            flatten=flatten,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=device,
        )


class ImageFolderStream(_VisionStreamBase):
    """Stream from a folder of images (for custom datasets)."""

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        split: str = "train",
        flatten: bool = True,
        normalize: bool = True,
        shuffle: bool | None = None,
        class_sequential: bool = False,
        seed: int = 0,
        device: str | torch.device | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        super().__init__(
            split=split,
            flatten=flatten,
            normalize=normalize,
            shuffle=shuffle,
            class_sequential=class_sequential,
            seed=seed,
            device=device,
        )
        self.root_dir = Path(root_dir)
        class_names = sorted(path.name for path in self.root_dir.iterdir() if path.is_dir())
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        grouped_files: dict[int, list[Path]] = defaultdict(list)
        for class_name, class_index in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    grouped_files[class_index].append(path)

        self._records: list[tuple[Path, int]] = []
        for class_index, files in grouped_files.items():
            for file_path in files:
                self._records.append((file_path, class_index))

        if self.class_sequential:
            self._records.sort(key=lambda record: (record[1], str(record[0])))
        elif self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(len(self._records), generator=generator).tolist()
            self._records = [self._records[index] for index in order]

    def __len__(self) -> int:
        return len(self._records)

    def _load_image(self, path: Path) -> torch.Tensor:
        if read_image is not None:  # pragma: no branch - optional fast path
            return read_image(str(path))
        if Image is None:  # pragma: no cover - optional dependency
            raise RuntimeError("ImageFolderStream requires torchvision or Pillow to read images")
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def stream(self) -> Iterator[DataSample]:
        for index, (path, label) in enumerate(self._records):
            image = self._load_image(path)
            yield DataSample(
                data=self._reshape_image(image),
                label=label,
                modality="vision",
                metadata={"index": index, "path": str(path), "split": self.split, "dataset": "image_folder"},
            )


__all__ = ["CIFAR10Stream", "CIFAR100Stream", "ImageFolderStream", "MNISTStream"]
