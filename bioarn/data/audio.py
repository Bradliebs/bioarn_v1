"""Synthetic audio streams for Bio-ARN."""

from __future__ import annotations

import math
from typing import Iterator

import torch

from bioarn.data.base import DataSample, StreamingDataSource


class SyntheticAudioStream(StreamingDataSource):
    """Generate distinct synthetic command-like waveforms for online audio tests."""

    class_names = (
        "low_sine",
        "mid_sine",
        "high_sine",
        "two_tone",
        "rising_chirp",
        "falling_chirp",
        "am_tone",
        "pulse_train",
        "fm_vibrato",
        "harmonic_stack",
    )

    def __init__(
        self,
        num_samples: int,
        *,
        sample_rate: int = 16000,
        duration_ms: int = 1000,
        shuffle: bool = True,
        seed: int = 0,
        class_labels: list[int] | None = None,
        noise_std: float = 0.015,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.num_samples = int(max(1, num_samples))
        self.sample_rate = int(max(1, sample_rate))
        self.duration_ms = int(max(50, duration_ms))
        self.num_samples_per_clip = max(1, int(round(self.sample_rate * (self.duration_ms / 1000.0))))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.noise_std = float(max(0.0, noise_std))
        labels = list(range(10)) if class_labels is None else [int(label) for label in class_labels]
        if not labels:
            raise ValueError("class_labels must be non-empty.")
        self.class_labels = labels
        base_labels = [self.class_labels[index % len(self.class_labels)] for index in range(self.num_samples)]
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=torch.Generator().manual_seed(self.seed)).tolist()
            self._labels = [base_labels[index] for index in order]
        else:
            self._labels = base_labels

    def __len__(self) -> int:
        return self.num_samples

    def _time_axis(self) -> torch.Tensor:
        return torch.linspace(
            0.0,
            self.duration_ms / 1000.0,
            self.num_samples_per_clip,
            dtype=torch.float32,
        )

    @staticmethod
    def _chirp(t: torch.Tensor, start_hz: float, end_hz: float) -> torch.Tensor:
        duration = max(float(t[-1].item()), 1e-6)
        slope = (end_hz - start_hz) / duration
        phase = (2.0 * math.pi) * ((start_hz * t) + (0.5 * slope * t.pow(2.0)))
        return torch.sin(phase)

    def _pattern(self, label: int, *, generator: torch.Generator) -> torch.Tensor:
        t = self._time_axis()
        fade = torch.sin(math.pi * torch.linspace(0.0, 1.0, self.num_samples_per_clip, dtype=torch.float32)).pow(2.0)
        phase_offset = float(torch.rand(1, generator=generator).item()) * (2.0 * math.pi)
        jitter = 1.0 + (0.03 * (torch.rand(1, generator=generator).item() - 0.5))

        if label == 0:
            waveform = torch.sin((2.0 * math.pi * 180.0 * jitter * t) + phase_offset)
        elif label == 1:
            waveform = torch.sin((2.0 * math.pi * 320.0 * jitter * t) + phase_offset)
        elif label == 2:
            waveform = torch.sin((2.0 * math.pi * 520.0 * jitter * t) + phase_offset)
        elif label == 3:
            waveform = (
                0.65 * torch.sin((2.0 * math.pi * 240.0 * jitter * t) + phase_offset)
                + 0.35 * torch.sin((2.0 * math.pi * 720.0 * jitter * t) + (0.5 * phase_offset))
            )
        elif label == 4:
            waveform = self._chirp(t, 160.0 * jitter, 720.0 * jitter)
        elif label == 5:
            waveform = self._chirp(t, 720.0 * jitter, 160.0 * jitter)
        elif label == 6:
            envelope = 0.55 + (0.45 * torch.sin((2.0 * math.pi * 3.0 * t) + phase_offset))
            waveform = envelope * torch.sin((2.0 * math.pi * 420.0 * jitter * t) + phase_offset)
        elif label == 7:
            gate = (torch.sin((2.0 * math.pi * 5.0 * t) + phase_offset) > 0.35).to(torch.float32)
            waveform = gate * torch.sin((2.0 * math.pi * 360.0 * jitter * t) + phase_offset)
        elif label == 8:
            instantaneous = 460.0 + (45.0 * torch.sin((2.0 * math.pi * 6.0 * t) + phase_offset))
            phase = torch.cumsum((2.0 * math.pi * instantaneous / self.sample_rate), dim=0)
            waveform = torch.sin(phase)
        elif label == 9:
            waveform = (
                0.55 * torch.sin((2.0 * math.pi * 200.0 * jitter * t) + phase_offset)
                + 0.3 * torch.sin((2.0 * math.pi * 400.0 * jitter * t) + (0.75 * phase_offset))
                + 0.15 * torch.sin((2.0 * math.pi * 600.0 * jitter * t) + (1.25 * phase_offset))
            )
        else:
            raise ValueError(f"Unsupported synthetic audio label: {label}")

        noise = self.noise_std * torch.randn(self.num_samples_per_clip, generator=generator)
        waveform = (waveform * fade) + noise
        return waveform.clamp_(-1.0, 1.0)

    def stream(self) -> Iterator[DataSample]:
        generator = torch.Generator().manual_seed(self.seed)
        for index, label in enumerate(self._labels):
            waveform = self._pattern(int(label), generator=generator)
            yield DataSample(
                data=self._move_tensor(waveform),
                label=int(label),
                modality="audio",
                metadata={
                    "index": index,
                    "dataset": "synthetic-audio",
                    "class_name": self.class_names[int(label)],
                    "sample_rate": self.sample_rate,
                    "duration_ms": self.duration_ms,
                },
            )


__all__ = ["SyntheticAudioStream"]
