"""Streaming language datasets for Bio-ARN."""

from __future__ import annotations

import json
import os
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import torch

from bioarn.data.base import DataSample, StreamingDataSource

_WIKITEXT_URLS = {
    "2": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    "103": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
}


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


class CharacterStream(StreamingDataSource):
    """Character-level text streaming."""

    def __init__(
        self,
        text_source: str | Path,
        context_length: int = 64,
        vocab_size: int = 256,
        stride: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        super().__init__(device=device)
        self.text_source = Path(text_source) if isinstance(text_source, Path) or Path(str(text_source)).exists() else text_source
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.stride = context_length if stride is None else stride
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.char_to_idx: dict[str, int] = {"<unk>": 0}
        self.idx_to_char: dict[int, str] = {0: "<unk>"}
        self._token_count = 0
        self._length = 0
        self._build_vocab_and_stats()

    def _iter_text_chunks(self) -> Iterator[str]:
        if isinstance(self.text_source, Path):
            with self.text_source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield line
        else:
            yield self.text_source

    def _sanitize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        sanitized: list[str] = []
        for char in normalized:
            if char in {"\r"}:
                continue
            if ord(char) < 32 and char not in {"\n", "\t"}:
                sanitized.append(" ")
            else:
                sanitized.append(char)
        return "".join(sanitized)

    def _encode_char(self, char: str) -> int:
        if char in self.char_to_idx:
            return self.char_to_idx[char]
        if len(self.char_to_idx) < self.vocab_size:
            index = len(self.char_to_idx)
            self.char_to_idx[char] = index
            self.idx_to_char[index] = char
            return index
        return 0

    def _build_vocab_and_stats(self) -> None:
        total_tokens = 0
        for chunk in self._iter_text_chunks():
            sanitized = self._sanitize_text(chunk)
            for char in sanitized:
                self._encode_char(char)
                total_tokens += 1

        self._token_count = total_tokens
        if total_tokens == 0:
            self._length = 0
        elif total_tokens <= self.context_length:
            self._length = 1
        else:
            self._length = 1 + (total_tokens - self.context_length) // self.stride

    def __len__(self) -> int:
        return self._length

    def _iter_encoded_tokens(self) -> Iterator[int]:
        for chunk in self._iter_text_chunks():
            sanitized = self._sanitize_text(chunk)
            for char in sanitized:
                yield self.char_to_idx.get(char, 0)

    def stream(self) -> Iterator[DataSample]:
        if self._length == 0:
            return

        buffer: list[int] = []
        offset = 0
        yielded = 0

        for token in self._iter_encoded_tokens():
            buffer.append(token)
            while len(buffer) >= self.context_length:
                window = buffer[: self.context_length]
                metadata = {
                    "offset": offset,
                    "context_length": self.context_length,
                    "source": str(self.text_source),
                }
                yield DataSample(
                    data=self._move_tensor(torch.tensor(window, dtype=torch.long)),
                    label=None,
                    modality="language",
                    metadata=metadata,
                )
                yielded += 1
                offset += self.stride
                del buffer[: self.stride]

        if yielded == 0 and buffer:
            padded = buffer + [0] * (self.context_length - len(buffer))
            yield DataSample(
                data=self._move_tensor(torch.tensor(padded, dtype=torch.long)),
                label=None,
                modality="language",
                metadata={"offset": 0, "context_length": self.context_length, "source": str(self.text_source)},
            )


class WikiTextStream(CharacterStream):
    """WikiText-2 or WikiText-103 streaming."""

    def __init__(
        self,
        version: str = "2",
        split: str = "train",
        context_length: int = 128,
        data_dir: str | os.PathLike[str] = "data/",
        stride: int | None = None,
        vocab_size: int = 256,
        device: str | torch.device | None = None,
    ) -> None:
        if version not in _WIKITEXT_URLS:
            raise ValueError("version must be '2' or '103'")
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be 'train', 'valid', or 'test'")

        data_root = Path(data_dir) / "WikiText"
        candidate_roots = [
            data_root / f"wikitext-{version}-raw-v1",
            data_root / f"wikitext-{version}-raw",
        ]
        file_path = next((root / f"wiki.{split}.raw" for root in candidate_roots if (root / f"wiki.{split}.raw").exists()), None)
        if file_path is None:
            archive_path = data_root / Path(_WIKITEXT_URLS[version]).name
            _download_with_progress(_WIKITEXT_URLS[version], archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(data_root)
            file_path = next((root / f"wiki.{split}.raw" for root in candidate_roots if (root / f"wiki.{split}.raw").exists()), None)
        if file_path is None:
            raise FileNotFoundError(f"Could not locate wiki.{split}.raw under {data_root}")

        super().__init__(
            text_source=file_path,
            context_length=context_length,
            vocab_size=vocab_size,
            stride=stride,
            device=device,
        )


class TinyStoriesStream(CharacterStream):
    """TinyStories dataset streaming (for small LM training)."""

    def __init__(
        self,
        split: str = "train",
        context_length: int = 256,
        data_dir: str | os.PathLike[str] = "data/",
        stride: int | None = None,
        vocab_size: int = 256,
        device: str | torch.device | None = None,
    ) -> None:
        if split not in {"train", "validation", "valid", "test"}:
            raise ValueError("split must be 'train', 'validation', 'valid', or 'test'")

        data_root = Path(data_dir) / "TinyStories"
        normalized_split = "valid" if split == "validation" else split
        txt_path = data_root / f"{normalized_split}.txt"
        jsonl_path = data_root / f"{normalized_split}.jsonl"

        if txt_path.exists():
            text_source: str | Path = txt_path
        elif jsonl_path.exists():
            text_source = self._read_jsonl(jsonl_path)
        else:
            raise FileNotFoundError(
                "TinyStories is not auto-downloaded. Place a local split file at "
                f"{txt_path} or {jsonl_path} (for example from the HuggingFace TinyStories dataset)."
            )

        super().__init__(
            text_source=text_source,
            context_length=context_length,
            vocab_size=vocab_size,
            stride=stride,
            device=device,
        )

    @staticmethod
    def _read_jsonl(path: Path) -> str:
        parts: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                payload = json.loads(line)
                text = payload.get("text") or payload.get("story") or payload.get("content") or ""
                if text:
                    parts.append(text)
        return "\n".join(parts)


__all__ = ["CharacterStream", "TinyStoriesStream", "WikiTextStream"]
