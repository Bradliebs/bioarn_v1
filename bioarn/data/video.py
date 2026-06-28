"""Synthetic video streams with simple temporal regularities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class VideoSequence:
    """One synthetic video clip with a temporal pattern label."""

    frames: list[torch.Tensor]
    label: int
    temporal_label: str
    metadata: dict[str, int | float | bool | str] = field(default_factory=dict)


class SyntheticVideoStream:
    """Generate synthetic video sequences with temporal patterns."""

    def __init__(
        self,
        num_sequences: int = 100,
        frames_per_sequence: int = 8,
        *,
        frame_shape: tuple[int, int] = (16, 16),
        seed: int = 0,
        violation_rate: float = 0.2,
    ) -> None:
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive.")
        if frames_per_sequence <= 1:
            raise ValueError("frames_per_sequence must be at least 2.")
        self.num_sequences = int(num_sequences)
        self.frames_per_sequence = int(frames_per_sequence)
        self.frame_shape = (int(max(4, frame_shape[0])), int(max(4, frame_shape[1])))
        self.seed = int(seed)
        self.violation_rate = float(min(max(violation_rate, 0.0), 1.0))
        self._patterns = (
            "moving_right",
            "moving_down",
            "appear_disappear",
            "periodic",
            "causal",
        )

    def __len__(self) -> int:
        return self.num_sequences

    def _blank_frame(self) -> torch.Tensor:
        return torch.zeros(self.frame_shape, dtype=torch.float32)

    def _square_frame(self, top: int, left: int, *, size: int = 3, value: float = 1.0) -> torch.Tensor:
        frame = self._blank_frame()
        bottom = min(self.frame_shape[0], top + size)
        right = min(self.frame_shape[1], left + size)
        frame[top:bottom, left:right] = value
        return frame

    def _vertical_bar(self, column: int, *, value: float = 1.0) -> torch.Tensor:
        frame = self._blank_frame()
        frame[:, max(0, column): min(self.frame_shape[1], column + 2)] = value
        return frame

    def _horizontal_bar(self, row: int, *, value: float = 1.0) -> torch.Tensor:
        frame = self._blank_frame()
        frame[max(0, row): min(self.frame_shape[0], row + 2), :] = value
        return frame

    def _pattern_frame(self, code: str) -> torch.Tensor:
        height, width = self.frame_shape
        if code == "A":
            return self._vertical_bar(max(1, width // 4), value=1.0)
        if code == "B":
            return self._horizontal_bar(max(1, height // 2), value=1.0)
        if code == "C":
            return self._square_frame(max(1, height // 3), max(1, width // 3), size=4, value=1.0)
        if code == "blank":
            return self._blank_frame()
        raise ValueError(f"Unknown frame code: {code}")

    def _moving_sequence(self, *, axis: str, sequence_index: int) -> VideoSequence:
        height, width = self.frame_shape
        frames: list[torch.Tensor] = []
        for step in range(self.frames_per_sequence):
            if axis == "x":
                top = 2 + (sequence_index % max(1, height - 6))
                left = 1 + ((step * 2) % max(1, width - 4))
            else:
                top = 1 + ((step * 2) % max(1, height - 4))
                left = 2 + (sequence_index % max(1, width - 6))
            frames.append(self._square_frame(top, left, size=3))
        label = 0 if axis == "x" else 1
        temporal_label = "moving_right" if axis == "x" else "moving_down"
        return VideoSequence(frames=frames, label=label, temporal_label=temporal_label)

    def _appearance_sequence(self) -> VideoSequence:
        frames: list[torch.Tensor] = []
        transition = max(1, self.frames_per_sequence // 2)
        for step in range(self.frames_per_sequence):
            if step < transition - 1:
                frames.append(self._pattern_frame("blank"))
            elif step == transition - 1:
                frames.append(self._pattern_frame("A"))
            elif step == transition:
                frames.append(self._pattern_frame("C"))
            else:
                frames.append(self._pattern_frame("blank"))
        return VideoSequence(frames=frames, label=2, temporal_label="appear_disappear")

    def _periodic_sequence(self) -> VideoSequence:
        frames = [
            self._pattern_frame("A") if step % 2 == 0 else self._pattern_frame("B")
            for step in range(self.frames_per_sequence)
        ]
        return VideoSequence(frames=frames, label=3, temporal_label="periodic")

    def _causal_sequence(self, *, violated: bool = False) -> VideoSequence:
        frames: list[torch.Tensor] = []
        violation_step = max(1, self.frames_per_sequence // 2)
        for step in range(self.frames_per_sequence):
            phase = step % 3
            if phase == 0:
                frames.append(self._pattern_frame("A"))
                continue
            if phase == 1:
                if violated and step == violation_step:
                    frames.append(self._pattern_frame("C"))
                else:
                    frames.append(self._pattern_frame("B"))
                continue
            frames.append(self._pattern_frame("blank"))
        return VideoSequence(
            frames=frames,
            label=4,
            temporal_label="causal_violation" if violated else "causal",
            metadata={"violated": violated},
        )

    def build_sequence(self, pattern_name: str, sequence_index: int = 0) -> VideoSequence:
        """Construct a named sequence deterministically."""

        if pattern_name == "moving_right":
            return self._moving_sequence(axis="x", sequence_index=sequence_index)
        if pattern_name == "moving_down":
            return self._moving_sequence(axis="y", sequence_index=sequence_index)
        if pattern_name == "appear_disappear":
            return self._appearance_sequence()
        if pattern_name == "periodic":
            return self._periodic_sequence()
        if pattern_name == "causal":
            return self._causal_sequence(violated=False)
        if pattern_name == "causal_violation":
            return self._causal_sequence(violated=True)
        raise ValueError(f"Unknown pattern name: {pattern_name}")

    def __iter__(self) -> Iterator[VideoSequence]:
        generator = torch.Generator().manual_seed(self.seed)
        pattern_count = len(self._patterns)
        for sequence_index in range(self.num_sequences):
            pattern_name = self._patterns[sequence_index % pattern_count]
            if pattern_name == "causal":
                violated = bool(torch.rand(1, generator=generator).item() < self.violation_rate)
                yield self._causal_sequence(violated=violated)
                continue
            yield self.build_sequence(pattern_name, sequence_index=sequence_index)
